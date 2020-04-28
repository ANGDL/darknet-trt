#include <assert.h>
#include <iostream>
#include <fstream>
#include <algorithm>  
#include "yolo.h"
#include "darknet_utils.h"
#include "nms_plugin.h"
#include "decode_plugin.h"

darknet::Yolo::Yolo(NetConfig* config, uint batch_size, float confidence_thresh, float nms_thresh) :
	config(config),
	batch_size(batch_size),
	prob_thresh(confidence_thresh),
	nms_thresh(nms_thresh),
	input_index(-1),
	engine(nullptr),
	context(nullptr),
	cuda_stream(nullptr),
	plugin_factory(new PluginFactory),
	bindings(0),
	trt_output_buffers(0),
	tiny_maxpool_padding_formula(new YoloTinyMaxpoolPaddingFormula()),
	is_init(false)
{
	std::string network_type = config->get_network_type();
	assert(network_type == "yolov3" || network_type == "yolov3-tiny");

	std::string precision = config->PRECISION;
	std::string planfile = "./" + network_type + "-" + precision + "-batch-" + to_string(batch_size) + ".engine";
	if (!file_exits(planfile))
	{
		std::cout << "Unable to find cached TensorRT engine for network : " << network_type
			<< " precision : " << precision << " and batch size :" << batch_size
			<< std::endl;
		std::cout << "Creating a new TensorRT Engine" << std::endl;

		if (precision == "kFLOAT") {
			is_init = build(nvinfer1::DataType::kFLOAT, planfile);
		}
		else if (precision == "kHALF") {
			is_init = build(nvinfer1::DataType::kHALF, planfile);
		}
		else if (precision == "KINT8") {
			//TODO
		}
		else {
			std::cout << "Unrecognized precision type " << precision << std::endl;
		}
	}

	if (!is_init && (!file_exits(planfile))) {
		return;
	}
	engine = load_trt_engine(planfile, plugin_factory, logger);
	if (nullptr == engine) {
		is_init = false;
		return;
	}
	context = engine->createExecutionContext();
	if (nullptr == context) {
		is_init = false;
		return;
	}

	input_index = engine->getBindingIndex(config->INPUT_BLOB_NAME.c_str());
	if (input_index == -1) {
		is_init = false;
		return;
	}

	NV_CUDA_CHECK(cudaStreamCreate(&cuda_stream));
	if (cuda_stream == nullptr) {
		is_init = false;
		return;
	}

	auto n_binding = engine->getNbBindings();
	bindings.resize(n_binding, nullptr);
	trt_output_buffers.resize(bindings.size() - 1, nullptr); // 减去一个输入
}


darknet::Yolo::~Yolo()
{
	if (cuda_stream != nullptr) NV_CUDA_CHECK(cudaStreamDestroy(cuda_stream));
	for (auto buffer : trt_output_buffers) NV_CUDA_CHECK(cudaFreeHost(buffer));
	for (auto binding : bindings) NV_CUDA_CHECK(cudaFree(binding));
	if (context != nullptr) {
		context->destroy();
		context = nullptr;
	}
	if (engine != nullptr) {
		engine->destroy();
		engine = nullptr;
	}
	if (plugin_factory != nullptr) {
		plugin_factory->destroy();
		plugin_factory = nullptr;
	}
}

bool darknet::Yolo::good() const
{
	return is_init;
}

bool darknet::Yolo::build(const nvinfer1::DataType data_type, const std::string planfile_path/*, Int8EntropyCalibrator* calibrator*/)
{
	assert(file_exits(config->WEIGHTS_FLIE));
	// 解析网络结构
	const darknet::Blocks& blocks = config->blocks;
	// 读取训练的权重
	std::vector<float> weights = load_weights(config->WEIGHTS_FLIE, config->get_network_type());
	// 
	std::vector<nvinfer1::Weights> trt_weights;
	int weight_ptr = 0;
	int channels = config->INPUT_C;
	// 创建builder
	auto builder = unique_ptr_infer<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
	// 创建network
	auto network = unique_ptr_infer<nvinfer1::INetworkDefinition>(builder->createNetwork());

	if ((data_type == nvinfer1::DataType::kINT8 && !builder->platformHasFastInt8())
		|| (data_type == nvinfer1::DataType::kHALF && !builder->platformHasFastFp16()))
	{
		std::cout << "Platform doesn't support this precision." << __func__ << ": " << __LINE__ << std::endl;
		return false;
	}

	// 添加输出层
	nvinfer1::ITensor* data = network->addInput(
		config->INPUT_BLOB_NAME.c_str(),
		data_type,
		nvinfer1::DimsCHW{
			channels,
			static_cast<int>(config->INPUT_H),
			static_cast<int>(config->INPUT_W) }
	);
	if (nullptr == data) {
		std::cout << "add input layer error " << __func__ << ": " << __LINE__ << std::endl;
		return false;
	}

	// 数据预处理
	// 归一化
	float* div_wights = new float[config->INPUT_SIZE];
	std::fill(div_wights, div_wights + config->INPUT_SIZE, 255.0);
	nvinfer1::Dims div_dim{
		3,
		{static_cast<int>(config->INPUT_C), static_cast<int>(config->INPUT_H), static_cast<int>(config->INPUT_W)},
		{nvinfer1::DimensionType::kCHANNEL, nvinfer1::DimensionType::kSPATIAL, nvinfer1::DimensionType::kSPATIAL}
	};
	nvinfer1::Weights div_weights_trt{ nvinfer1::DataType::kFLOAT, div_wights, static_cast<int64_t>(config->INPUT_SIZE) };

	nvinfer1::IConstantLayer* div_layer = network->addConstant(div_dim, div_weights_trt);
	if (nullptr == div_layer) {
		std::cout << "add constant layer error in  image normalization " << __func__ << ": " << __LINE__ << std::endl;
		return false;
	}

	nvinfer1::IElementWiseLayer* norm_layer = network->addElementWise(
		*data,
		*div_layer->getOutput(0),
		nvinfer1::ElementWiseOperation::kDIV);
	if (nullptr == norm_layer) {
		std::cout << "add norm layer error image normalization " << __func__ << ": " << __LINE__ << std::endl;
		return false;
	}


	nvinfer1::ITensor* previous = norm_layer->getOutput(0);
	std::vector<nvinfer1::ITensor*> output_tensors;
	std::vector<nvinfer1::ITensor*> yolo_tensors;

	//// Set the output dimensions formula for pooling layers
	network->setPoolingOutputDimensionsFormula(tiny_maxpool_padding_formula.get());

	// 构建网络
	for (int i = 0; i < blocks.size(); ++i)
	{
		const Block& block = blocks[i];
		const std::string b_type = block.at("type");
		assert(get_num_channels(previous) == channels);

		if (b_type == "net") {
			// print
			print_layer_info("", "layer", "input_dims", "output_dims", to_string(weight_ptr));
		}
		else if (b_type == "convolutional") {
			nvinfer1::ILayer* conv;
			if (block.find("batch_normalize") == block.end()) {
				conv = add_conv_linear(i, block, weights, trt_weights, weight_ptr, channels, previous, network.get());
			}
			else {
				conv = add_conv_bn_leaky(i, block, weights, trt_weights, weight_ptr, channels, previous, network.get());
			}

			if (nullptr == conv) {
				std::cout << "add convolutional_" + to_string(i) << " layer error " << __func__ << ": " << __LINE__ << std::endl;
				return false;
			}

			//print
			print_layer_info(i, "conv_" + to_string(i), previous->getDimensions(), conv->getOutput(0)->getDimensions(), weight_ptr);

			previous = conv->getOutput(0);
			output_tensors.push_back(previous);

			channels = get_num_channels(previous);
			previous->setName(conv->getName());
		}
		else if (b_type == "shortcut") {
			assert(block.find("from") != block.end());
			assert(block.at("activation") == "linear");

			int from = stoi(block.at("from"));
			assert(output_tensors.size() + from >= 0);
			assert(output_tensors.size() - 1 >= 0);

			nvinfer1::IElementWiseLayer* shortcut_layer = network->addElementWise(
				**(output_tensors.end() - 1),
				**(output_tensors.end() + from),
				nvinfer1::ElementWiseOperation::kSUM
			);
			std::string layer_name = "shortcut_" + to_string(i);

			if (nullptr == shortcut_layer) {
				std::cout << "add " << layer_name << " layer error " << __func__ << ": " << __LINE__ << std::endl;
				return false;
			}

			shortcut_layer->setName(layer_name.c_str());

			//print
			print_layer_info(i, shortcut_layer->getName(), previous->getDimensions(), shortcut_layer->getOutput(0)->getDimensions(), weight_ptr);


			nvinfer1::ITensor* shortcut_out = shortcut_layer->getOutput(0);
			output_tensors.push_back(shortcut_out);
			channels = get_num_channels(shortcut_out);
			previous = shortcut_out;
			previous->setName(layer_name.c_str());
		}
		else if (b_type == "route") {
			assert(block.find("layers") != block.end());
			vector<string> layers_s = split(trim(block.at("layers")), ',');
			if (layers_s.size() == 1) {
				int idx = stoi(layers_s[0]);
				idx = idx < 0 ? idx + output_tensors.size() : idx;
				assert(idx < output_tensors.size() && idx >= 0);

				//print
				print_layer_info(i, "route_" + to_string(i), previous->getDimensions(), output_tensors[idx]->getDimensions(), weight_ptr);

				previous = output_tensors[idx];
				channels = get_num_channels(previous);
				output_tensors.push_back(previous);
			}
			else if (layers_s.size() == 2) {
				int idx_1 = stoi(layers_s[0]);
				int idx_2 = stoi(layers_s[1]);

				idx_1 = idx_1 < 0 ? idx_1 + output_tensors.size() : idx_1;
				idx_2 = idx_2 < 0 ? idx_2 + output_tensors.size() : idx_2;

				assert(idx_1 < output_tensors.size() && idx_1 >= 0);
				assert(idx_2 < output_tensors.size() && idx_2 >= 0);

				nvinfer1::ITensor** concat_input = reinterpret_cast<nvinfer1::ITensor**>(malloc(sizeof(nvinfer1::ITensor*) * 2));
				if (nullptr == concat_input) {
					std::cout << "malloc concat_input memory error!" << __func__ << ": " << __LINE__ << std::endl;
					return false;
				}

				concat_input[0] = output_tensors[idx_1];
				concat_input[1] = output_tensors[idx_2];

				nvinfer1::IConcatenationLayer* route_layer = network->addConcatenation(concat_input, 2);
				std::string layer_name = "route_" + to_string(i);

				if (nullptr == route_layer) {
					std::cout << "add " << layer_name << " layer error " << __func__ << ": " << __LINE__ << std::endl;
					return false;
				}

				route_layer->setName(layer_name.c_str());
				route_layer->setAxis(0);

				//print
				print_layer_info(i, route_layer->getName(), previous->getDimensions(), route_layer->getOutput(0)->getDimensions(), weight_ptr);

				previous = route_layer->getOutput(0);
				output_tensors.push_back(previous);
				channels = get_num_channels(concat_input[0]) + get_num_channels(concat_input[1]);
				previous->setName(layer_name.c_str());
			}
			else {
				std::cout << "error with route layer > 2 !" << std::endl;
				return false;
			}
		}
		else if (b_type == "yolo") {
			nvinfer1::Dims grid_dim = previous->getDimensions();
			assert(grid_dim.d[2] == grid_dim.d[1]);
			unsigned int grid_size = grid_dim.d[1];

			auto yolo_plugin = new YoloLayer(config->get_bboxes(), config->OUTPUT_CLASSES, grid_size);
			nvinfer1::ILayer* yolo_layer = network->addPlugin(&previous, 1, *yolo_plugin);

			std::string layer_name = "yolo_" + to_string(i);
			if (nullptr == yolo_layer) {
				std::cout << "add " << layer_name << " layer error " << __func__ << ": " << __LINE__ << std::endl;
				return false;
			}

			yolo_layer->setName(layer_name.c_str());

			nvinfer1::ITensor* yolo_output = yolo_layer->getOutput(0);

			//print
			print_layer_info(i, yolo_layer->getName(), previous->getDimensions(), yolo_layer->getOutput(0)->getDimensions(), weight_ptr);

			network->markOutput(*yolo_output);
			yolo_tensors.push_back(yolo_output);
			output_tensors.push_back(yolo_output);

			previous = yolo_output;
			channels = get_num_channels(previous);
			previous->setName(layer_name.c_str());
		}
		else if (b_type == "upsample") {
			nvinfer1::ILayer* upsample_layer = add_upsample(i, block, weights, trt_weights, weight_ptr, channels, previous, network.get());
			//nvinfer1::ILayer* upsample_layer = add_upsample2(i, block, weights, channels, previous, network.get());
			if (nullptr == upsample_layer)
			{
				std::cout << "add upsample_" << to_string(i) << " layer error " << __func__ << ": " << __LINE__ << std::endl;
				return false;
			}

			//print
			print_layer_info(i, upsample_layer->getName(), previous->getDimensions(), upsample_layer->getOutput(0)->getDimensions(), weight_ptr);

			previous = upsample_layer->getOutput(0);
			channels = get_num_channels(previous);
			output_tensors.push_back(previous);
			previous->setName(upsample_layer->getName());
		}
		else if (b_type == "maxpool") {
			// 设置same padding
			if (block.at("size") == "2" && block.at("stride") == "1")
			{
				tiny_maxpool_padding_formula->add_same_padding_layer("maxpool_" + std::to_string(i));
			}
			nvinfer1::ILayer* pooling_layer = add_maxpool(i, block, previous, network.get());
			if (nullptr == pooling_layer) {
				std::cout << "add pooling_" << to_string(i) << " layer error " << __func__ << ": " << __LINE__ << std::endl;
				return false;
			}

			//print
			print_layer_info(i, pooling_layer->getName(), previous->getDimensions(), pooling_layer->getOutput(0)->getDimensions(), weight_ptr);

			previous = pooling_layer->getOutput(0);
			channels = get_num_channels(previous);
			output_tensors.push_back(previous);
			previous->setName(pooling_layer->getName());
		}
		else {
			std::cout << "Unsupported layer type --> \"" << blocks.at(i).at("type") << "\""
				<< std::endl;
			return false;
		}

	}

	// 添加decode plugin

	if (config->use_cuda_nms)
	{
		for (auto& t : yolo_tensors)
		{
			network->unmarkOutput(*t);
		}

		std::vector<ILayer*> decode_layers;
		std::vector<float> anchors;

		if (config->get_network_type() == "yolov3-tiny") {
			auto cfg = dynamic_cast<YoloV3TinyCfg*>(config.get());

			// yolo_layer_1
			for (size_t i = 0; i < cfg->get_bboxes(); i++) {
				anchors.push_back(cfg->ANCHORS[cfg->MASK_1[i] * 2]);
				anchors.push_back(cfg->ANCHORS[cfg->MASK_1[i] * 2 + 1]);
			}
			nvinfer1::ILayer* decode_layer_1 = add_decode(
				yolo_tensors[0], network.get(), "decode_1",
				cfg->score_thresh,  anchors,
				cfg->STRIDE_1,
				cfg->GRID_SIZE_1,
				cfg->get_bboxes(),
				cfg->OUTPUT_CLASSES
			);

			anchors.clear();

			for (size_t i = 0; i < cfg->get_bboxes(); i++) {
				anchors.push_back(cfg->ANCHORS[cfg->MASK_2[i] * 2]);
				anchors.push_back(cfg->ANCHORS[cfg->MASK_2[i] * 2 + 1]);
			}
			nvinfer1::ILayer* decode_layer_2 = add_decode(
				yolo_tensors[1], network.get(), "decode_2",
				cfg->score_thresh,  anchors,
				cfg->STRIDE_2,
				cfg->GRID_SIZE_2,
				cfg->get_bboxes(),
				cfg->OUTPUT_CLASSES
			);

			decode_layers.push_back(decode_layer_1);
			decode_layers.push_back(decode_layer_2);
		}
		else if (config->get_network_type() == "yolov3") {
			auto cfg = dynamic_cast<YoloV3Cfg*>(config.get());
			// yolo_layer_1
			for (size_t i = 0; i < cfg->get_bboxes(); i++) {
				anchors.push_back(cfg->ANCHORS[cfg->MASK_1[i] * 2]);
				anchors.push_back(cfg->ANCHORS[cfg->MASK_1[i] * 2 + 1]);
			}
			nvinfer1::ILayer* decode_layer_1 = add_decode(
				yolo_tensors[0], network.get(), "decode_1",
				cfg->score_thresh,  anchors,
				cfg->STRIDE_1,
				cfg->GRID_SIZE_1,
				cfg->get_bboxes(),
				cfg->OUTPUT_CLASSES
			);

			anchors.clear();

			for (size_t i = 0; i < cfg->get_bboxes(); i++) {
				anchors.push_back(cfg->ANCHORS[cfg->MASK_2[i] * 2]);
				anchors.push_back(cfg->ANCHORS[cfg->MASK_2[i] * 2 + 1]);
			}
			nvinfer1::ILayer* decode_layer_2 = add_decode(
				yolo_tensors[1], network.get(), "decode_2",
				cfg->score_thresh,  anchors,
				cfg->STRIDE_2,
				cfg->GRID_SIZE_2,
				cfg->get_bboxes(),
				cfg->OUTPUT_CLASSES
			);

			anchors.clear();

			for (size_t i = 0; i < cfg->get_bboxes(); i++) {
				anchors.push_back(cfg->ANCHORS[cfg->MASK_3[i] * 2]);
				anchors.push_back(cfg->ANCHORS[cfg->MASK_3[i] * 2 + 1]);
			}
			nvinfer1::ILayer* decode_layer_3 = add_decode(
				yolo_tensors[2], network.get(), "decode_3",
				cfg->score_thresh,  anchors,
				cfg->STRIDE_3,
				cfg->GRID_SIZE_3,
				cfg->get_bboxes(),
				cfg->OUTPUT_CLASSES
			);

			decode_layers.push_back(decode_layer_1);
			decode_layers.push_back(decode_layer_2);
			decode_layers.push_back(decode_layer_3);
		}

		//test
		//auto nms_plugin = NMSPlugin(config->nms_thresh, config->max_detection);
		//
		//std::vector<nvinfer1::ITensor*> scores, boxes, classes;
		//for (auto& l : decode_layers) {
		//	std::vector<ITensor*> nms_tensors;
		//	nms_tensors.push_back(l->getOutput(0));
		//	nms_tensors.push_back(l->getOutput(1));
		//	nms_tensors.push_back(l->getOutput(2));
		//	auto nms_layer = network->addPluginV2(nms_tensors.data(), nms_tensors.size(), nms_plugin);
		//	scores.push_back(nms_layer->getOutput(0));
		//	boxes.push_back(nms_layer->getOutput(1));
		//	classes.push_back(nms_layer->getOutput(2));
		//}

		 //concat deocode output tensors
		 //scores, boxes, classes
		std::vector<nvinfer1::ITensor*> scores, boxes, classes;
		for (auto& l : decode_layers) {
			scores.push_back(l->getOutput(0));
			boxes.push_back(l->getOutput(1));
			classes.push_back(l->getOutput(2));

			std::cout << "scores dim : " << scores.back()->getDimensions().d[0] << std::endl;
			std::cout << "boxes dim : " << boxes.back()->getDimensions().d[0] << std::endl;
			std::cout << "classes dim : " << classes.back()->getDimensions().d[0] << std::endl;
		}

		std::vector<nvinfer1::ITensor*> concat;
		for (auto tensor : { scores, boxes, classes })
		{
			auto layer = network->addConcatenation(tensor.data(), tensor.size());
			layer->setAxis(0);
			concat.push_back(layer->getOutput(0));
		}
		// add nms plugin
		auto nms_plugin = NMSPlugin(config->nms_thresh, config->max_detection);
		auto nms_layer = network->addPluginV2(concat.data(), concat.size(), nms_plugin);

		vector<string> names = { "scores", "boxes", "classes" };
		for (int i = 0; i < nms_layer->getNbOutputs(); i++) {
			auto output = nms_layer->getOutput(i);
			network->markOutput(*output);
			output->setName(names[i].c_str());
		}
	}

	if (weights.size() != weight_ptr)
	{
		std::cout << "Number of unused weights left : " << weights.size() - weight_ptr << std::endl;
		std::cout << __func__ << ": " << __LINE__ << std::endl;
		return false;
	}

	//
	builder->setMaxWorkspaceSize(1 << 26);
	builder->setMaxBatchSize(batch_size);

	if (data_type == nvinfer1::DataType::kINT8)
	{
		// TODO
	}
	else if (data_type == nvinfer1::DataType::kHALF)
	{
		builder->setHalf2Mode(true);
	}

	// 创建 engine
	auto cuda_engine = unique_ptr_infer<nvinfer1::ICudaEngine>(builder->buildCudaEngine(*network));
	if (nullptr == cuda_engine)
	{
		std::cout << "Build the TensorRT Engine failed !" << __func__ << ": " << __LINE__ << std::endl;
		return false;
	}

	// 保存engine
	save_engine(cuda_engine.get(), planfile_path);

	std::cout << "Serialized plan file cached at location : " << planfile_path << std::endl;

	// deallocate the weights
	for (uint i = 0; i < trt_weights.size(); ++i)
	{
		free(const_cast<void*>(trt_weights[i].values));
	}

	delete[] div_wights;
	div_wights = nullptr;

	return true;
}


nvinfer1::ILayer* darknet::Yolo::add_maxpool(int layer_idx, const darknet::Block& block, nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
	assert(block.at("type") == "maxpool");
	assert(block.find("stride") != block.end());
	assert(block.find("size") != block.end());

	int w_size = stoi(block.at("size"));
	int stride = stoi(block.at("stride"));
	nvinfer1::IPoolingLayer* pool = network->addPooling(*input, nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW(w_size, w_size));
	if (nullptr == pool) {
		return nullptr;
	}
	pool->setStride(nvinfer1::DimsHW{ stride, stride });
	std::string layer_name = "maxpool_" + std::to_string(layer_idx);
	pool->setName(layer_name.c_str());

	return pool;
}


nvinfer1::ILayer* darknet::Yolo::add_conv_bn_leaky(int layer_idx, const darknet::Block& block, std::vector<float>& weight,
	std::vector<nvinfer1::Weights>& trt_weights, int& weight_ptr, int& input_channels, nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
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

	for (int i = 0; i < filters; ++i)
	{
		bn_baises.push_back(weight[weight_ptr++]);
	}

	for (int i = 0; i < filters; ++i)
	{
		bn_weights.push_back(weight[weight_ptr++]);
	}

	for (int i = 0; i < filters; ++i)
	{
		bn_means.push_back(weight[weight_ptr++]);
	}

	for (int i = 0; i < filters; ++i)
	{
		bn_vars.push_back(sqrtf(weight[weight_ptr++] + 1.0e-5));
	}

	nvinfer1::IConvolutionLayer* conv = add_conv(layer_idx, filters, k_size, stride, pad, weight, weight_ptr, input_channels, input, network, false);
	if (nullptr == conv) {
		return nullptr;
	}
	trt_weights.push_back(conv->getBiasWeights());
	trt_weights.push_back(conv->getKernelWeights());

	nvinfer1::IScaleLayer* bn = add_bn(layer_idx, filters, bn_baises, bn_weights, bn_means, bn_vars, conv->getOutput(0), network);
	if (nullptr == bn) {
		return nullptr;
	}
	trt_weights.push_back(bn->getShift());
	trt_weights.push_back(bn->getScale());
	trt_weights.push_back(bn->getPower());

	nvinfer1::IPluginLayer* leaky = add_leakyRelu(layer_idx, bn->getOutput(0), network);
	if (nullptr == leaky) {
		return nullptr;
	}

	return leaky;
}

nvinfer1::ILayer* darknet::Yolo::add_conv_linear(int layer_idx, const darknet::Block& block, std::vector<float>& weight, std::vector<nvinfer1::Weights>& trt_weights, int& weight_ptr, int& input_channels, nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
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


	nvinfer1::IConvolutionLayer* conv = add_conv(layer_idx, filters, k_size, stride, pad, weight, weight_ptr, input_channels, input, network, true);

	trt_weights.push_back(conv->getBiasWeights());
	trt_weights.push_back(conv->getKernelWeights());

	return conv;
}

nvinfer1::ILayer* darknet::Yolo::add_upsample(int layer_idx, const darknet::Block& block, std::vector<float>& weights, std::vector<nvinfer1::Weights>& trt_weights, int& weight_ptr, int& input_channels, nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
	assert(block.at("type") == "upsample");
	assert(block.find("stride") != block.end());
	nvinfer1::Dims input_dims = input->getDimensions();
	assert(input_dims.nbDims == 3);

	float stride = stof(block.at("stride"));

	nvinfer1::IPlugin* upsample = new UpsampleLayer(stride, input_dims);
	nvinfer1::IPluginLayer* upsample_layer = network->addPlugin(&input, 1, *upsample);
	if (nullptr == upsample)
	{
		return nullptr;
	}

	std::string layer_name = "upsample_" + to_string(layer_idx);
	upsample_layer->setName(layer_name.c_str());

	return upsample_layer;
}

nvinfer1::IConvolutionLayer* darknet::Yolo::add_conv(int layer_idx, int filters, int kernel_size, int stride, int pad,
	std::vector<float>& weight, int& weight_ptr, int& input_channels,
	nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network,
	bool use_biases)
{
	float* bias_buff = nullptr;
	if (use_biases)
	{
		bias_buff = new float[filters];
		for (int i = 0; i < filters; ++i)
		{
			bias_buff[i] = weight[weight_ptr++];
		}
	}

	size_t kernel_data_len = (size_t)kernel_size * kernel_size * filters * input_channels;
	float* weight_buff = new float[kernel_data_len];
	for (size_t i = 0; i < kernel_data_len; i++)
	{
		weight_buff[i] = weight[weight_ptr++];
	}

	nvinfer1::Weights conv_bias{ nvinfer1::DataType::kFLOAT, bias_buff, bias_buff == nullptr ? 0 : filters };
	nvinfer1::Weights conv_weights{ nvinfer1::DataType::kFLOAT, weight_buff, kernel_data_len };

	nvinfer1::IConvolutionLayer* conv = network->addConvolution(*input, filters, nvinfer1::DimsHW(kernel_size, kernel_size), conv_weights, conv_bias);
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
						   x - mean					  γ                  mean*γ
		y = γ * ---------------- + β = x * ------  + β -  ------------
						  -----------					 var                  var
						\|    var^2

trt scale:

		y = ( x * scale + shift) ^ power
*/

nvinfer1::IScaleLayer* darknet::Yolo::add_bn(int layer_idx, int filters, std::vector<float>& bn_biases, std::vector<float>& bn_weights,
	std::vector<float>& bn_mean, std::vector<float>& bn_var, nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
	float* scale_buff = new float[filters];
	float* shift_buff = new float[filters];
	float* power_buff = new float[filters];

	for (int i = 0; i < filters; i++)
	{
		scale_buff[i] = bn_weights[i] / bn_var[i];
		shift_buff[i] = bn_biases[i] - ((bn_mean[i] * bn_weights[i]) / bn_var[i]);
		power_buff[i] = 1.0;
	}

	nvinfer1::Weights shift{ nvinfer1::DataType::kFLOAT, shift_buff, filters };
	nvinfer1::Weights scale{ nvinfer1::DataType::kFLOAT, scale_buff, filters };
	nvinfer1::Weights power{ nvinfer1::DataType::kFLOAT, power_buff, filters };

	nvinfer1::IScaleLayer* bn = network->addScale(*input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
	if (nullptr == bn) {
		return nullptr;
	}

	std::string layer_name = "batch_norm_" + to_string(layer_idx);
	bn->setName(layer_name.c_str());

	return bn;
}

nvinfer1::IPluginLayer* darknet::Yolo::add_leakyRelu(int layer_idx, nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
	nvinfer1::IPlugin* leaky_relu = nvinfer1::plugin::createPReLUPlugin(0.1);
	nvinfer1::IPluginLayer* leaky = network->addPlugin(&input, 1, *leaky_relu);
	if (nullptr == leaky) {
		return nullptr;
	}
	std::string layer_name = "leaky_relu_" + to_string(layer_idx);
	leaky->setName(layer_name.c_str());
	return leaky;
}

nvinfer1::IPluginV2Layer* darknet::Yolo::add_decode(nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network, std::string name, float score_thresh, const std::vector<float> anchors, int stride, int gride_size, int num_anchors, int num_classes)
{
	auto decode = DecodePlugin(score_thresh, anchors, stride, gride_size, num_anchors, num_classes);
	auto* decode_layer = network->addPluginV2(&input, 1, decode);
	if (nullptr == decode_layer) {
		return nullptr;
	}

	decode_layer->setName(name.c_str());
	return decode_layer;
}

nvinfer1::ILayer* darknet::Yolo::add_upsample2(int layer_idx, const darknet::Block& block, std::vector<float>& weights, int& input_channels, nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
	assert(block.at("type") == "upsample");
	assert(block.at("stride") == "2");
	nvinfer1::Dims inpDims = input->getDimensions();
	assert(inpDims.nbDims == 3);
	int h = inpDims.d[1];
	int w = inpDims.d[2];
	// add pre multiply matrix as a constant
	nvinfer1::Dims preDims{ 3,
						   {1, 2 * h, w},
						   {nvinfer1::DimensionType::kCHANNEL, nvinfer1::DimensionType::kSPATIAL,
							nvinfer1::DimensionType::kSPATIAL} };
	int size = 2 * h * w;
	nvinfer1::Weights pre{ nvinfer1::DataType::kFLOAT, nullptr, size };
	float* preWt = new float[size];
	/* (2*h * w)
	[ [1, 0, ..., 0],
	  [1, 0, ..., 0],
	  [0, 1, ..., 0],
	  [0, 1, ..., 0],
	  ...,
	  ...,
	  [0, 0, ..., 1],
	  [0, 0, ..., 1] ]
	*/
	for (int i = 0, idx = 0; i < h; ++i)
	{
		for (int j = 0; j < w; ++j, ++idx)
		{
			preWt[idx] = (i == j) ? 1.0 : 0.0;
		}
		for (int j = 0; j < w; ++j, ++idx)
		{
			preWt[idx] = (i == j) ? 1.0 : 0.0;
		}
	}
	pre.values = preWt;
	nvinfer1::IConstantLayer* preM = network->addConstant(preDims, pre);
	assert(preM != nullptr);
	std::string preLayerName = "pre_" + std::to_string(layer_idx);
	preM->setName(preLayerName.c_str());
	// add post multiply matrix as a constant
	nvinfer1::Dims postDims{ 3,
							{1, h, 2 * w},
							{nvinfer1::DimensionType::kCHANNEL, nvinfer1::DimensionType::kSPATIAL,
							 nvinfer1::DimensionType::kSPATIAL} };
	size = 2 * h * w;
	nvinfer1::Weights post{ nvinfer1::DataType::kFLOAT, nullptr, size };
	float* postWt = new float[size];
	/* (h * 2*w)
	[ [1, 1, 0, 0, ..., 0, 0],
	  [0, 0, 1, 1, ..., 0, 0],
	  ...,
	  ...,
	  [0, 0, 0, 0, ..., 1, 1] ]
	*/
	for (int i = 0, idx = 0; i < h; ++i)
	{
		for (int j = 0; j < 2 * w; ++j, ++idx)
		{
			postWt[idx] = (j / 2 == i) ? 1.0 : 0.0;
		}
	}
	post.values = postWt;
	nvinfer1::IConstantLayer* post_m = network->addConstant(postDims, post);
	assert(post_m != nullptr);
	std::string postLayerName = "post_" + std::to_string(layer_idx);
	post_m->setName(postLayerName.c_str());
	// add matrix multiply layers for upsampling
	nvinfer1::IMatrixMultiplyLayer* mm1 = network->addMatrixMultiply(*preM->getOutput(0), false, *input, false);
	assert(mm1 != nullptr);
	std::string mm1LayerName = "mm1_" + std::to_string(layer_idx);
	mm1->setName(mm1LayerName.c_str());
	nvinfer1::IMatrixMultiplyLayer* mm2
		= network->addMatrixMultiply(*mm1->getOutput(0), false, *post_m->getOutput(0), false);
	assert(mm2 != nullptr);
	std::string mm2LayerName = "mm2_" + std::to_string(layer_idx);
	mm2->setName(mm2LayerName.c_str());
	// switch dimension **types** from kSPATIAL, kCHANNEL, kSPATIAL to kCHANNEL, kSPATIAL, kSPATIAL
	nvinfer1::Dims outDims{ 3,
						   {input_channels, 2 * h, 2 * w},
						   {nvinfer1::DimensionType::kCHANNEL, nvinfer1::DimensionType::kSPATIAL,
							nvinfer1::DimensionType::kSPATIAL} };
	nvinfer1::IShuffleLayer* reshape = network->addShuffle(*mm2->getOutput(0));
	assert(reshape != nullptr);
	std::string reshapeLayerName = "upsample_" + std::to_string(layer_idx);
	reshape->setName(reshapeLayerName.c_str());
	reshape->setReshapeDimensions(outDims);

	return reshape;
}

nvinfer1::ILayer* darknet::Yolo::netAddConvBNLeaky(int layerIdx, const darknet::Block& block,  std::vector<float>& weights, std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr, int& inputChannels, nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
	assert(block.at("type") == "convolutional");
	assert(block.find("batch_normalize") != block.end());
	assert(block.at("batch_normalize") == "1");
	assert(block.at("activation") == "leaky");
	assert(block.find("filters") != block.end());
	assert(block.find("pad") != block.end());
	assert(block.find("size") != block.end());
	assert(block.find("stride") != block.end());

	bool batchNormalize, bias;
	if (block.find("batch_normalize") != block.end())
	{
		batchNormalize = (block.at("batch_normalize") == "1");
		bias = false;
	}
	else
	{
		batchNormalize = false;
		bias = true;
	}
	// all conv_bn_leaky layers assume bias is false
	assert(batchNormalize == true && bias == false);

	int filters = std::stoi(block.at("filters"));
	int padding = std::stoi(block.at("pad"));
	int kernelSize = std::stoi(block.at("size"));
	int stride = std::stoi(block.at("stride"));
	int pad;
	if (padding)
		pad = (kernelSize - 1) / 2;
	else
		pad = 0;

	/***** CONVOLUTION LAYER *****/
	/*****************************/
	// batch norm weights are before the conv layer
	// load BN biases (bn_biases)
	std::vector<float> bnBiases;
	for (int i = 0; i < filters; ++i)
	{
		bnBiases.push_back(weights[weightPtr]);
		weightPtr++;
	}
	// load BN weights
	std::vector<float> bnWeights;
	for (int i = 0; i < filters; ++i)
	{
		bnWeights.push_back(weights[weightPtr]);
		weightPtr++;
	}
	// load BN running_mean
	std::vector<float> bnRunningMean;
	for (int i = 0; i < filters; ++i)
	{
		bnRunningMean.push_back(weights[weightPtr]);
		weightPtr++;
	}
	// load BN running_var
	std::vector<float> bnRunningVar;
	for (int i = 0; i < filters; ++i)
	{
		// 1e-05 for numerical stability
		bnRunningVar.push_back(sqrt(weights[weightPtr] + 1.0e-5));
		weightPtr++;
	}
	// load Conv layer weights (GKCRS)
	int size = filters * inputChannels * kernelSize * kernelSize;
	nvinfer1::Weights convWt{ nvinfer1::DataType::kFLOAT, nullptr, size };
	float* val = new float[size];
	for (int i = 0; i < size; ++i)
	{
		val[i] = weights[weightPtr];
		weightPtr++;
	}
	convWt.values = val;
	trtWeights.push_back(convWt);
	nvinfer1::Weights convBias{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
	trtWeights.push_back(convBias);
	nvinfer1::IConvolutionLayer* conv = network->addConvolution(
		*input, filters, nvinfer1::DimsHW{ kernelSize, kernelSize }, convWt, convBias);
	assert(conv != nullptr);
	std::string convLayerName = "conv_" + std::to_string(layerIdx);
	conv->setName(convLayerName.c_str());
	conv->setStride(nvinfer1::DimsHW{ stride, stride });
	conv->setPadding(nvinfer1::DimsHW{ pad, pad });

	/***** BATCHNORM LAYER *****/
	/***************************/
	size = filters;
	// create the weights
	nvinfer1::Weights shift{ nvinfer1::DataType::kFLOAT, nullptr, size };
	nvinfer1::Weights scale{ nvinfer1::DataType::kFLOAT, nullptr, size };
	nvinfer1::Weights power{ nvinfer1::DataType::kFLOAT, nullptr, size };
	float* shiftWt = new float[size];
	for (int i = 0; i < size; ++i)
	{
		shiftWt[i]
			= bnBiases.at(i) - ((bnRunningMean.at(i) * bnWeights.at(i)) / bnRunningVar.at(i));
	}
	shift.values = shiftWt;
	float* scaleWt = new float[size];
	for (int i = 0; i < size; ++i)
	{
		scaleWt[i] = bnWeights.at(i) / bnRunningVar[i];
	}
	scale.values = scaleWt;
	float* powerWt = new float[size];
	for (int i = 0; i < size; ++i)
	{
		powerWt[i] = 1.0;
	}
	power.values = powerWt;
	trtWeights.push_back(shift);
	trtWeights.push_back(scale);
	trtWeights.push_back(power);
	// Add the batch norm layers
	nvinfer1::IScaleLayer* bn = network->addScale(
		*conv->getOutput(0), nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
	assert(bn != nullptr);
	std::string bnLayerName = "batch_norm_" + std::to_string(layerIdx);
	bn->setName(bnLayerName.c_str());
	/***** ACTIVATION LAYER *****/
	/****************************/
	nvinfer1::IPlugin* leakyRELU = nvinfer1::plugin::createPReLUPlugin(0.1);
	assert(leakyRELU != nullptr);
	nvinfer1::ITensor* bnOutput = bn->getOutput(0);
	nvinfer1::IPluginLayer* leaky = network->addPlugin(&bnOutput, 1, *leakyRELU);
	assert(leaky != nullptr);
	std::string leakyLayerName = "leaky_" + std::to_string(layerIdx);
	leaky->setName(leakyLayerName.c_str());

	return leaky;
}

nvinfer1::ILayer* darknet::Yolo::netAddConvLinear(int layerIdx, const darknet::Block& block, std::vector<float>& weights, std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr, int& inputChannels, nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
	assert(block.at("type") == "convolutional");
	assert(block.find("batch_normalize") == block.end());
	assert(block.at("activation") == "linear");
	assert(block.find("filters") != block.end());
	assert(block.find("pad") != block.end());
	assert(block.find("size") != block.end());
	assert(block.find("stride") != block.end());

	int filters = std::stoi(block.at("filters"));
	int padding = std::stoi(block.at("pad"));
	int kernelSize = std::stoi(block.at("size"));
	int stride = std::stoi(block.at("stride"));
	int pad;
	if (padding)
		pad = (kernelSize - 1) / 2;
	else
		pad = 0;
	// load the convolution layer bias
	nvinfer1::Weights convBias{ nvinfer1::DataType::kFLOAT, nullptr, filters };
	float* val = new float[filters];
	for (int i = 0; i < filters; ++i)
	{
		val[i] = weights[weightPtr];
		weightPtr++;
	}
	convBias.values = val;
	trtWeights.push_back(convBias);
	// load the convolutional layer weights
	int size = filters * inputChannels * kernelSize * kernelSize;
	nvinfer1::Weights convWt{ nvinfer1::DataType::kFLOAT, nullptr, size };
	val = new float[size];
	for (int i = 0; i < size; ++i)
	{
		val[i] = weights[weightPtr];
		weightPtr++;
	}
	convWt.values = val;
	trtWeights.push_back(convWt);
	nvinfer1::IConvolutionLayer* conv = network->addConvolution(
		*input, filters, nvinfer1::DimsHW{ kernelSize, kernelSize }, convWt, convBias);
	assert(conv != nullptr);
	std::string convLayerName = "conv_" + std::to_string(layerIdx);
	conv->setName(convLayerName.c_str());
	conv->setStride(nvinfer1::DimsHW{ stride, stride });
	conv->setPadding(nvinfer1::DimsHW{ pad, pad });

	return conv;
}
