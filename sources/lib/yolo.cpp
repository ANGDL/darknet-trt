#include <assert.h>
#include <iostream>
#include <algorithm>
#include "yolo.h"
#include "darknet_utils.h"
#include "nms_plugin.h"
#include "decode_plugin.h"
#include "yolo_layer_plugin.h"


darknet::Yolo::Yolo(NetConfig *config, uint batch_size) :
        config_(config),
        batch_size_(batch_size),
        input_index_(-1),
        engine_(nullptr),
        context_(nullptr),
        cuda_stream_(nullptr),
        bindings_(0),
        trt_output_buffers_(0),
        tiny_maxpool_padding_formula_(new YoloTinyMaxpoolPaddingFormula()),
        is_init_(false) {
    std::string network_type = config->get_network_type();
    assert(network_type == "yolov3" || network_type == "yolov3-tiny");

    std::string precision = config->PRECISION;
    std::string planfile = "./" + network_type + "-" + precision + "-batch-" + to_string(batch_size) + ".engine";
    if (!file_exits(planfile)) {
        std::cout << "Unable to find cached TensorRT engine for network : " << network_type
                  << " precision : " << precision << " and batch size :" << batch_size
                  << std::endl;
        std::cout << "Creating a new TensorRT Engine" << std::endl;

        if (precision == "kFLOAT") {
            is_init_ = build(nvinfer1::DataType::kFLOAT, planfile);
        } else if (precision == "kHALF") {
            is_init_ = build(nvinfer1::DataType::kHALF, planfile);
        } else if (precision == "KINT8") {
            Int8EntropyCalibrator calibrator(
                    batch_size_,
                    config->calib_images_list_file,
                    config->calib_images_path,
                    config->calib_table_file_path,
                    config->INPUT_SIZE,
                    config->INPUT_H,
                    config->INPUT_W,
                    config->INPUT_BLOB_NAME
                    );
            is_init_ = build(nvinfer1::DataType::kINT8, planfile, &calibrator);
        } else {
            std::cout << "Unrecognized precision type " << precision << std::endl;
        }
    }

    if (!is_init_ && (!file_exits(planfile))) {
        return;
    }
    engine_ = load_trt_engine(planfile, logger_);
    if (nullptr == engine_) {
        is_init_ = false;
        return;
    }
    context_ = engine_->createExecutionContext();
    if (nullptr == context_) {
        is_init_ = false;
        return;
    }

    input_index_ = engine_->getBindingIndex(config->INPUT_BLOB_NAME.c_str());
    if (input_index_ == -1) {
        is_init_ = false;
        return;
    }

    NV_CUDA_CHECK(cudaStreamCreate(&cuda_stream_));
    if (cuda_stream_ == nullptr) {
        is_init_ = false;
        return;
    }

    auto n_binding = engine_->getNbBindings();
    bindings_.resize(n_binding, nullptr);
    trt_output_buffers_.resize(bindings_.size() - 1, nullptr); // 减去一个输入
}


darknet::Yolo::~Yolo() {
    if (cuda_stream_ != nullptr) NV_CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
    for (auto buffer : trt_output_buffers_) NV_CUDA_CHECK(cudaFreeHost(buffer));
    for (auto binding : bindings_) NV_CUDA_CHECK(cudaFree(binding));
    if (context_ != nullptr) {
        context_->destroy();
        context_ = nullptr;
    }
    if (engine_ != nullptr) {
        engine_->destroy();
        engine_ = nullptr;
    }
}

bool darknet::Yolo::good() const {
    return is_init_;
}

bool darknet::Yolo::build(const nvinfer1::DataType data_type,
                          const std::string& planfile_path, Int8EntropyCalibrator* calibrator) {
    assert(file_exits(config_->WEIGHTS_FLIE));
    // 解析网络结构
    const darknet::Blocks &blocks = config_->blocks;
    // 读取训练的权重
    std::vector<float> weights = load_weights(config_->WEIGHTS_FLIE, config_->get_network_type());
    //
    std::vector<nvinfer1::Weights> trt_weights;
    int weight_ptr = 0;
    int channels = config_->INPUT_C;
    // 创建builder
    auto builder = unique_ptr_infer<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
    // 创建builder config
    auto builder_config = unique_ptr_infer<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    // 创建network
    auto network = unique_ptr_infer<nvinfer1::INetworkDefinition>(builder->createNetwork());

    if ((data_type == nvinfer1::DataType::kINT8 && !builder->platformHasFastInt8())
        || (data_type == nvinfer1::DataType::kHALF && !builder->platformHasFastFp16())) {
        std::cout << "Platform doesn't support this precision." << __func__ << ": " << __LINE__ << std::endl;
        return false;
    }

    // 添加nput层
    nvinfer1::ITensor *data = network->addInput(
            config_->INPUT_BLOB_NAME.c_str(),
            nvinfer1::DataType::kFLOAT,
            nvinfer1::DimsCHW{
                    channels,
                    static_cast<int>(config_->INPUT_H),
                    static_cast<int>(config_->INPUT_W)}
    );
    if (nullptr == data) {
        std::cout << "add input layer error " << __func__ << ": " << __LINE__ << std::endl;
        return false;
    }

    // 数据预处理
    // 归一化
    auto *div_wights = new float[config_->INPUT_SIZE];
    std::fill(div_wights, div_wights + config_->INPUT_SIZE, 255.0);
    nvinfer1::Dims div_dim{
            3,
            {static_cast<int>(config_->INPUT_C), static_cast<int>(config_->INPUT_H),
             static_cast<int>(config_->INPUT_W)},
            {nvinfer1::DimensionType::kCHANNEL, nvinfer1::DimensionType::kSPATIAL, nvinfer1::DimensionType::kSPATIAL}
    };
    nvinfer1::Weights div_weights_trt{nvinfer1::DataType::kFLOAT, div_wights,
                                      static_cast<int64_t>(config_->INPUT_SIZE)};
    trt_weights.push_back(div_weights_trt);

    nvinfer1::IConstantLayer *div_layer = network->addConstant(div_dim, div_weights_trt);
    if (nullptr == div_layer) {
        std::cout << "add constant layer error in  image normalization " << __func__ << ": " << __LINE__ << std::endl;
        return false;
    }

    nvinfer1::IElementWiseLayer *norm_layer = network->addElementWise(
            *data,
            *div_layer->getOutput(0),
            nvinfer1::ElementWiseOperation::kDIV);
    if (nullptr == norm_layer) {
        std::cout << "add norm layer error image normalization " << __func__ << ": " << __LINE__ << std::endl;
        return false;
    }


    nvinfer1::ITensor *previous = norm_layer->getOutput(0);
    std::vector<nvinfer1::ITensor *> output_tensors;
    std::vector<nvinfer1::ITensor *> yolo_tensors;

    //// Set the output dimensions formula for pooling layers
    network->setPoolingOutputDimensionsFormula(tiny_maxpool_padding_formula_.get());

    // 构建网络
    for (int i = 0; i < blocks.size(); ++i) {
        const Block &block = blocks[i];
        const std::string b_type = block.at("type");
        assert(get_num_channels(previous) == channels);

        if (b_type == "net") {
            // print
            print_layer_info("", "layer", "input_dims", "output_dims", to_string(weight_ptr));
        } else if (b_type == "convolutional") {
            nvinfer1::ILayer *conv;
            if (block.find("batch_normalize") == block.end()) {
                conv = add_conv_linear(i, block, weights, trt_weights, weight_ptr, channels, previous, network.get());
            } else {
                conv = add_conv_bn_leaky(i, block, weights, trt_weights, weight_ptr, channels, previous, network.get());
            }

            if (nullptr == conv) {
                std::cout << "add convolutional_" + to_string(i) << " layer error " << __func__ << ": " << __LINE__
                          << std::endl;
                return false;
            }

            //print
            print_layer_info(i, "conv_" + to_string(i), previous->getDimensions(), conv->getOutput(0)->getDimensions(),
                             weight_ptr);

            previous = conv->getOutput(0);
            output_tensors.push_back(previous);

            channels = get_num_channels(previous);
            previous->setName(conv->getName());
        } else if (b_type == "shortcut") {
            assert(block.find("from") != block.end());
            assert(block.at("activation") == "linear");

            int from = stoi(block.at("from"));

            assert((i - 2 >= 0) && (i - 2 < output_tensors.size()));
            assert((i + from - 1 >= 0) && (i + from - 1 < output_tensors.size()));
            assert(i + from - 1 < i - 2);

            nvinfer1::IElementWiseLayer *shortcut_layer = network->addElementWise(
                    *(output_tensors[i - 2]),
                    *(output_tensors[i + from - 1]),
                    nvinfer1::ElementWiseOperation::kSUM
            );
            std::string layer_name = "shortcut_" + to_string(i);

            if (nullptr == shortcut_layer) {
                std::cout << "add " << layer_name << " layer error " << __func__ << ": " << __LINE__ << std::endl;
                return false;
            }

            shortcut_layer->setName(layer_name.c_str());

            //print
            print_layer_info(i, shortcut_layer->getName(), previous->getDimensions(),
                             shortcut_layer->getOutput(0)->getDimensions(), weight_ptr);


            nvinfer1::ITensor *shortcut_out = shortcut_layer->getOutput(0);
            output_tensors.push_back(shortcut_out);
            channels = get_num_channels(shortcut_out);
            previous = shortcut_out;
            previous->setName(layer_name.c_str());
        } else if (b_type == "route") {
            size_t found = block.at("layers").find(",");
            if (found != std::string::npos) {
                int idx1 = std::stoi(trim(block.at("layers").substr(0, found)));
                int idx2 = std::stoi(trim(block.at("layers").substr(found + 1)));
                if (idx1 < 0) {
                    idx1 = output_tensors.size() + idx1;
                }
                if (idx2 < 0) {
                    idx2 = output_tensors.size() + idx2;
                }
                assert(idx1 < static_cast<int>(output_tensors.size()) && idx1 >= 0);
                assert(idx2 < static_cast<int>(output_tensors.size()) && idx2 >= 0);
                auto **concatInputs
                        = reinterpret_cast<nvinfer1::ITensor **>(malloc(sizeof(nvinfer1::ITensor *) * 2));
                concatInputs[0] = output_tensors[idx1];
                concatInputs[1] = output_tensors[idx2];
                nvinfer1::IConcatenationLayer *concat
                        = network->addConcatenation(concatInputs, 2);
                assert(concat != nullptr);
                std::string concatLayerName = "route_" + std::to_string(i - 1);
                concat->setName(concatLayerName.c_str());
                // concatenate along the channel dimension
                concat->setAxis(0);
                previous = concat->getOutput(0);
                assert(previous != nullptr);
                // set the output volume depth
                channels
                        = get_num_channels(output_tensors[idx1]) + get_num_channels(output_tensors[idx2]);
                output_tensors.push_back(concat->getOutput(0));

                print_layer_info(i, concat->getName(), previous->getDimensions(),
                                 concat->getOutput(0)->getDimensions(), weight_ptr);
            } else {
                int idx = std::stoi(trim(block.at("layers")));
                if (idx < 0) {
                    idx = output_tensors.size() + idx;
                }
                assert(idx < static_cast<int>(output_tensors.size()) && idx >= 0);
                previous = output_tensors[idx];
                assert(previous != nullptr);

                // set the output volume depth
                channels = get_num_channels(output_tensors[idx]);
                output_tensors.push_back(output_tensors[idx]);
                print_layer_info(i, "route_" + to_string(i), previous->getDimensions(),
                                 output_tensors[idx]->getDimensions(), weight_ptr);
            }
        } else if (b_type == "yolo") {
            nvinfer1::Dims grid_dim = previous->getDimensions();
            assert(grid_dim.d[2] == grid_dim.d[1]);
            unsigned int grid_size = grid_dim.d[1];

//            auto yolo_plugin = new YoloLayer(config->get_bboxes(), config->OUTPUT_CLASSES, grid_size);
            auto yolo_plugin = YoloLayerPlugin(config_->get_bboxes(), config_->OUTPUT_CLASSES, grid_size);
            nvinfer1::ILayer *yolo_layer = network->addPluginV2(&previous, 1, yolo_plugin);

            std::string layer_name = "yolo_" + to_string(i);
            if (nullptr == yolo_layer) {
                std::cout << "add " << layer_name << " layer error " << __func__ << ": " << __LINE__ << std::endl;
                return false;
            }

            yolo_layer->setName(layer_name.c_str());

            nvinfer1::ITensor *yolo_output = yolo_layer->getOutput(0);

            //print
            print_layer_info(i, yolo_layer->getName(), previous->getDimensions(),
                             yolo_layer->getOutput(0)->getDimensions(), weight_ptr);

            network->markOutput(*yolo_output);
            yolo_tensors.push_back(yolo_output);
            output_tensors.push_back(yolo_output);

            previous = yolo_output;
            channels = get_num_channels(previous);
            previous->setName(layer_name.c_str());
        } else if (b_type == "upsample") {
            nvinfer1::ILayer *upsample_layer = add_upsample(i, block, weights, trt_weights, weight_ptr, channels,
                                                            previous, network.get());
            if (nullptr == upsample_layer) {
                std::cout << "add upsample_" << to_string(i) << " layer error " << __func__ << ": " << __LINE__
                          << std::endl;
                return false;
            }

            //print
            print_layer_info(i, upsample_layer->getName(), previous->getDimensions(),
                             upsample_layer->getOutput(0)->getDimensions(), weight_ptr);

            previous = upsample_layer->getOutput(0);
            channels = get_num_channels(previous);
            output_tensors.push_back(previous);
            previous->setName(upsample_layer->getName());
        } else if (b_type == "maxpool") {
            // 设置same padding
            if (block.at("size") == "2" && block.at("stride") == "1") {
                tiny_maxpool_padding_formula_->add_same_padding_layer("maxpool_" + std::to_string(i));
            }
            nvinfer1::ILayer *pooling_layer = add_maxpool(i, block, previous, network.get());
            if (nullptr == pooling_layer) {
                std::cout << "add pooling_" << to_string(i) << " layer error " << __func__ << ": " << __LINE__
                          << std::endl;
                return false;
            }

            //print
            print_layer_info(i, pooling_layer->getName(), previous->getDimensions(),
                             pooling_layer->getOutput(0)->getDimensions(), weight_ptr);

            previous = pooling_layer->getOutput(0);
            channels = get_num_channels(previous);
            output_tensors.push_back(previous);
            previous->setName(pooling_layer->getName());
        } else {
            std::cout << "Unsupported layer type --> \"" << blocks.at(i).at("type") << "\""
                      << std::endl;
            return false;
        }

    }

    if (weights.size() != weight_ptr) {
        std::cout << "Number of unused weights left : " << weights.size() - weight_ptr << std::endl;
        std::cout << __func__ << ": " << __LINE__ << std::endl;
        return false;
    }

    // 添加decode plugin
    if (config_->use_cuda_nms_) {
        for (auto &t : yolo_tensors) {
            network->unmarkOutput(*t);
        }

        std::vector<ILayer *> decode_layers;
        std::vector<float> anchors;

        if (config_->get_network_type() == "yolov3-tiny") {
            auto cfg = dynamic_cast<YoloV3TinyCfg *>(config_.get());

            // yolo_layer_1
            for (size_t i = 0; i < cfg->get_bboxes(); i++) {
                anchors.push_back(cfg->ANCHORS[cfg->MASK_1[i] * 2]);
                anchors.push_back(cfg->ANCHORS[cfg->MASK_1[i] * 2 + 1]);
            }
            nvinfer1::ILayer *decode_layer_1 = add_decode(
                    yolo_tensors[0], network.get(), "decode_1",
                    cfg->score_thresh_, anchors,
                    cfg->STRIDE_1,
                    cfg->GRID_SIZE_1,
                    static_cast<int>(cfg->get_bboxes()),
                    cfg->OUTPUT_CLASSES
            );

            anchors.clear();

            for (size_t i = 0; i < cfg->get_bboxes(); i++) {
                anchors.push_back(cfg->ANCHORS[cfg->MASK_2[i] * 2]);
                anchors.push_back(cfg->ANCHORS[cfg->MASK_2[i] * 2 + 1]);
            }
            nvinfer1::ILayer *decode_layer_2 = add_decode(
                    yolo_tensors[1], network.get(), "decode_2",
                    cfg->score_thresh_, anchors,
                    cfg->STRIDE_2,
                    cfg->GRID_SIZE_2,
                    static_cast<int>(cfg->get_bboxes()),
                    cfg->OUTPUT_CLASSES
            );

            decode_layers.push_back(decode_layer_1);
            decode_layers.push_back(decode_layer_2);
        }
        else if (config_->get_network_type() == "yolov3") {
            auto cfg = dynamic_cast<YoloV3Cfg *>(config_.get());
            // yolo_layer_1
            for (size_t i = 0; i < cfg->get_bboxes(); i++) {
                anchors.push_back(cfg->ANCHORS[cfg->MASK_1[i] * 2]);
                anchors.push_back(cfg->ANCHORS[cfg->MASK_1[i] * 2 + 1]);
            }
            nvinfer1::ILayer *decode_layer_1 = add_decode(
                    yolo_tensors[0], network.get(), "decode_1",
                    cfg->score_thresh_, anchors,
                    cfg->STRIDE_1,
                    cfg->GRID_SIZE_1,
                    static_cast<int>(cfg->get_bboxes()),
                    cfg->OUTPUT_CLASSES
            );

            anchors.clear();

            for (size_t i = 0; i < cfg->get_bboxes(); i++) {
                anchors.push_back(cfg->ANCHORS[cfg->MASK_2[i] * 2]);
                anchors.push_back(cfg->ANCHORS[cfg->MASK_2[i] * 2 + 1]);
            }
            nvinfer1::ILayer *decode_layer_2 = add_decode(
                    yolo_tensors[1], network.get(), "decode_2",
                    cfg->score_thresh_, anchors,
                    cfg->STRIDE_2,
                    cfg->GRID_SIZE_2,
                    static_cast<int>(cfg->get_bboxes()),
                    cfg->OUTPUT_CLASSES
            );

            anchors.clear();

            for (size_t i = 0; i < cfg->get_bboxes(); i++) {
                anchors.push_back(cfg->ANCHORS[cfg->MASK_3[i] * 2]);
                anchors.push_back(cfg->ANCHORS[cfg->MASK_3[i] * 2 + 1]);
            }
            nvinfer1::ILayer *decode_layer_3 = add_decode(
                    yolo_tensors[2], network.get(), "decode_3",
                    cfg->score_thresh_, anchors,
                    cfg->STRIDE_3,
                    cfg->GRID_SIZE_3,
                    static_cast<int>(cfg->get_bboxes()),
                    cfg->OUTPUT_CLASSES
            );

            decode_layers.push_back(decode_layer_1);
            decode_layers.push_back(decode_layer_2);
            decode_layers.push_back(decode_layer_3);
        }

        //concat deocode output tensors
        //scores, boxes, classes
        std::vector<nvinfer1::ITensor *> scores, boxes, classes;
        for (auto &l : decode_layers) {
//            l->setPrecision(nvinfer1::DataType::kFLOAT);
            scores.push_back(l->getOutput(0));
            boxes.push_back(l->getOutput(1));
            classes.push_back(l->getOutput(2));

            std::cout << "scores dim : " << scores.back()->getDimensions().d[0] << std::endl;
            std::cout << "boxes dim : " << boxes.back()->getDimensions().d[0] << std::endl;
            std::cout << "classes dim : " << classes.back()->getDimensions().d[0] << std::endl;
        }

        std::vector<nvinfer1::ITensor *> concat;
        for (auto tensor : {scores, boxes, classes}) {
            auto layer = network->addConcatenation(tensor.data(), tensor.size());
            layer->setAxis(0);
//            layer->setPrecision(nvinfer1::DataType::kFLOAT);
            concat.push_back(layer->getOutput(0));
        }
        // add nms plugin
        auto nms_plugin = NMSPlugin(config_->nms_thresh_, config_->max_detection_);
        auto nms_layer = network->addPluginV2(concat.data(), concat.size(), nms_plugin);
//        nms_layer->setPrecision(nvinfer1::DataType::kFLOAT);

        vector<string> names = {"scores", "boxes", "classes"};
        for (int i = 0; i < nms_layer->getNbOutputs(); i++) {
            auto output = nms_layer->getOutput(i);
            network->markOutput(*output);
            output->setName(names[i].c_str());
        }
    }

    builder_config->setMaxWorkspaceSize(1 << 26);
    builder_config->setFlag(BuilderFlag::kGPU_FALLBACK);
    builder_config->setFlag(BuilderFlag::kSTRICT_TYPES);
    if (data_type == nvinfer1::DataType::kHALF) {
        builder_config->setFlag(BuilderFlag::kFP16);
    }
    if (data_type == nvinfer1::DataType::kINT8) {
        assert((calibrator != nullptr) && "Invalid calibrator for INT8 precision");
        builder_config->setFlag(BuilderFlag::kINT8);
        builder_config->setInt8Calibrator(calibrator);
    }
    builder->setMaxBatchSize(batch_size_);

    if(config_->use_dla){
        int nbLayers = network->getNbLayers();
        int layersOnDLA = 0;
        std::cout << "Total number of layers: " << nbLayers << std::endl;
        for (uint i = 0; i < nbLayers; i++) {
            nvinfer1::ILayer *curLayer = network->getLayer(i);
            if (builder->canRunOnDLA(curLayer)) {
                builder->setDeviceType(curLayer, nvinfer1::DeviceType::kDLA);
                layersOnDLA++;
                std::cout << "Set layer " << curLayer->getName() << " to run on DLA" << std::endl;
            }
        }
        std::cout << "Total number of layers on DLA: " << layersOnDLA << std::endl;
    }


    // 创建 engine
    auto cuda_engine = unique_ptr_infer<nvinfer1::ICudaEngine>(
            builder->buildEngineWithConfig(*network, *builder_config));
    if (nullptr == cuda_engine) {
        std::cout << "Build the TensorRT Engine failed !" << __func__ << ": " << __LINE__ << std::endl;
        return false;
    }

    // 保存engine
    save_engine(cuda_engine.get(), planfile_path);

    std::cout << "Serialized plan file cached at location : " << planfile_path << std::endl;

    // deallocate the weights
    for (uint i = 0; i < trt_weights.size(); ++i) {
        free(const_cast<void *>(trt_weights[i].values));
    }

    return true;
}


nvinfer1::ILayer *darknet::Yolo::add_maxpool(int layer_idx, const darknet::Block &block, nvinfer1::ITensor *input,
                                             nvinfer1::INetworkDefinition *network) {
    assert(block.at("type") == "maxpool");
    assert(block.find("stride") != block.end());
    assert(block.find("size") != block.end());

    int w_size = stoi(block.at("size"));
    int stride = stoi(block.at("stride"));
    nvinfer1::IPoolingLayer *pool = network->addPooling(*input, nvinfer1::PoolingType::kMAX,
                                                        nvinfer1::DimsHW(w_size, w_size));
    if (nullptr == pool) {
        return nullptr;
    }
    pool->setStride(nvinfer1::DimsHW{stride, stride});
    std::string layer_name = "maxpool_" + std::to_string(layer_idx);
    pool->setName(layer_name.c_str());

    return pool;
}


nvinfer1::ILayer *
darknet::Yolo::add_conv_bn_leaky(int layer_idx, const darknet::Block &block, std::vector<float> &weight,
                                 std::vector<nvinfer1::Weights> &trt_weights, int &weight_ptr, int &input_channels,
                                 nvinfer1::ITensor *input, nvinfer1::INetworkDefinition *network) {
    assert(block.at("type") == "convolutional");
    assert(block.find("batch_normalize") != block.end());
    assert(block.find("filters") != block.end());
    assert(block.find("size") != block.end());
    assert(block.find("stride") != block.end());
    assert(block.find("pad") != block.end());
    assert(block.find("activation") != block.end());

    int filters = stoi(block.at("filters"));
    int k_size = stoi(block.at("size"));
    int stride = stoi(block.at("stride"));
    int pad = stoi(block.at("pad"));

    pad = pad ? (k_size - 1) / 2 : 0;

    std::vector<float> bn_baises, bn_weights, bn_means, bn_vars;

    for (int i = 0; i < filters; ++i) {
        bn_baises.push_back(weight[weight_ptr++]);
    }

    for (int i = 0; i < filters; ++i) {
        bn_weights.push_back(weight[weight_ptr++]);
    }

    for (int i = 0; i < filters; ++i) {
        bn_means.push_back(weight[weight_ptr++]);
    }

    for (int i = 0; i < filters; ++i) {
        bn_vars.push_back(sqrtf(weight[weight_ptr++] + 1.0e-5));
    }

    nvinfer1::IConvolutionLayer *conv = add_conv(layer_idx, filters, k_size, stride, pad, weight, weight_ptr,
                                                 input_channels, input, network, false);
    if (nullptr == conv) {
        return nullptr;
    }
    trt_weights.push_back(conv->getBiasWeights());
    trt_weights.push_back(conv->getKernelWeights());

    nvinfer1::IScaleLayer *bn = add_bn(layer_idx, filters, bn_baises, bn_weights, bn_means, bn_vars, conv->getOutput(0),
                                       network);
    if (nullptr == bn) {
        return nullptr;
    }
    trt_weights.push_back(bn->getShift());
    trt_weights.push_back(bn->getScale());
    trt_weights.push_back(bn->getPower());

    nvinfer1::ILayer *leaky = add_leakyReLU(layer_idx, bn->getOutput(0), network);
    if (nullptr == leaky) {
        return nullptr;
    }

    return leaky;
}

nvinfer1::ILayer *darknet::Yolo::add_conv_linear(int layer_idx, const darknet::Block &block, std::vector<float> &weight,
                                                 std::vector<nvinfer1::Weights> &trt_weights, int &weight_ptr,
                                                 int &input_channels, nvinfer1::ITensor *input,
                                                 nvinfer1::INetworkDefinition *network) {
    assert(block.at("type") == "convolutional");
    assert(block.find("batch_normalize") == block.end());
    assert(block.find("filters") != block.end());
    assert(block.find("size") != block.end());
    assert(block.find("stride") != block.end());
    assert(block.find("pad") != block.end());
    assert(block.find("activation") != block.end());
    assert(block.at("activation") == "linear");

    int filters = stoi(block.at("filters"));
    int k_size = stoi(block.at("size"));
    int stride = stoi(block.at("stride"));
    int pad = stoi(block.at("pad"));

    pad = pad ? (k_size - 1) / 2 : 0;


    nvinfer1::IConvolutionLayer *conv = add_conv(layer_idx, filters, k_size, stride, pad, weight, weight_ptr,
                                                 input_channels, input, network, true);

    trt_weights.push_back(conv->getBiasWeights());
    trt_weights.push_back(conv->getKernelWeights());

    return conv;
}


nvinfer1::ILayer *darknet::Yolo::add_upsample(int layer_idx, const darknet::Block &block, std::vector<float> &weights,
                                              std::vector<nvinfer1::Weights> &trt_weights, int &weight_ptr,
                                              int &input_channels, nvinfer1::ITensor *input,
                                              nvinfer1::INetworkDefinition *network) {
    assert(block.at("type") == "upsample");
    assert(block.find("stride") != block.end());
    nvinfer1::Dims input_dims = input->getDimensions();
    assert(input_dims.nbDims == 3);

    float stride = stof(block.at("stride"));

    std::vector<float> scales{1.0, 2.0, 2.0};

    nvinfer1::IResizeLayer *upsample_layer = network->addResize(*input);
    upsample_layer->setScales(scales.data(), scales.size());
    upsample_layer->setResizeMode(nvinfer1::ResizeMode::kNEAREST);

    std::string layer_name = "upsample_" + to_string(layer_idx);
    upsample_layer->setName(layer_name.c_str());

    return upsample_layer;
}

nvinfer1::IConvolutionLayer *darknet::Yolo::add_conv(int layer_idx, int filters, int kernel_size, int stride, int pad,
                                                     std::vector<float> &weight, int &weight_ptr, int &input_channels,
                                                     nvinfer1::ITensor *input, nvinfer1::INetworkDefinition *network,
                                                     bool use_biases) {
    float *bias_buff = nullptr;
    if (use_biases) {
        bias_buff = new float[filters];
        for (int i = 0; i < filters; ++i) {
            bias_buff[i] = weight[weight_ptr++];
        }
    }

    int64_t kernel_data_len = (int64_t) kernel_size * kernel_size * filters * input_channels;
    float *weight_buff = new float[kernel_data_len];
    for (size_t i = 0; i < kernel_data_len; i++) {
        weight_buff[i] = weight[weight_ptr++];
    }

    nvinfer1::Weights conv_bias{nvinfer1::DataType::kFLOAT, bias_buff, bias_buff == nullptr ? 0 : filters};
    nvinfer1::Weights conv_weights{nvinfer1::DataType::kFLOAT, weight_buff, kernel_data_len};

    nvinfer1::IConvolutionLayer *conv = network->addConvolution(*input, filters,
                                                                nvinfer1::DimsHW(kernel_size, kernel_size),
                                                                conv_weights, conv_bias);
    if (nullptr == conv) {
        return nullptr;
    }

    conv->setStride(DimsHW(stride, stride));
    conv->setPadding(DimsHW(pad, pad));

    std::string layer_name = "conv_" + to_string(layer_idx);
    conv->setName(layer_name.c_str());

    return conv;
}

/*
bn:
						   x - mean					      γ             mean*γ
		y = γ * ---------------------------- + β = x * -------  + β -  ---------
						  -----------					 var              var
						\|    var^2

trt scale:

		y = ( x * scale + shift) ^ power
*/

nvinfer1::IScaleLayer *
darknet::Yolo::add_bn(int layer_idx, int filters, std::vector<float> &bn_biases, std::vector<float> &bn_weights,
                      std::vector<float> &bn_mean, std::vector<float> &bn_var, nvinfer1::ITensor *input,
                      nvinfer1::INetworkDefinition *network) {
    float *scale_buff = new float[filters];
    float *shift_buff = new float[filters];
    float *power_buff = new float[filters];

    for (int i = 0; i < filters; i++) {
        scale_buff[i] = bn_weights[i] / bn_var[i];
        shift_buff[i] = bn_biases[i] - ((bn_mean[i] * bn_weights[i]) / bn_var[i]);
        power_buff[i] = 1.0;
    }

    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, shift_buff, filters};
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, scale_buff, filters};
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, power_buff, filters};

    nvinfer1::IScaleLayer *bn = network->addScale(*input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    if (nullptr == bn) {
        return nullptr;
    }

    std::string layer_name = "batch_norm_" + to_string(layer_idx);
    bn->setName(layer_name.c_str());

    return bn;
}

nvinfer1::ILayer *
darknet::Yolo::add_leakyReLU(int layer_idx, nvinfer1::ITensor *input, nvinfer1::INetworkDefinition *network) {

    float *data = new float[1];
    data[0] = 0.1;
    nvinfer1::Weights slope{nvinfer1::DataType::kFLOAT, (void *) data, 1};

    Dims slopes_dims{
            input->getDimensions().nbDims,
            {1, 1, 1},
            {DimensionType::kCHANNEL, DimensionType::kSPATIAL, DimensionType::kSPATIAL}};

    auto constLayer = network->addConstant(slopes_dims, slope);

    nvinfer1::ILayer *leaky_relu = network->addParametricReLU(*input, *constLayer->getOutput(0));
    return leaky_relu;
}


nvinfer1::IPluginV2Layer *
darknet::Yolo::add_decode(nvinfer1::ITensor *input, nvinfer1::INetworkDefinition *network, std::string name,
                          float score_thresh, const std::vector<float> anchors, int stride, int gride_size,
                          int num_anchors, int num_classes) {
    auto decode = DecodePlugin(score_thresh, anchors, stride, gride_size, num_anchors, num_classes);
    auto *decode_layer = network->addPluginV2(&input, 1, decode);
    if (nullptr == decode_layer) {
        return nullptr;
    }

    decode_layer->setName(name.c_str());
    return decode_layer;
}

