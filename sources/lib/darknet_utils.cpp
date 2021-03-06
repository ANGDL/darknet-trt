﻿#include <functional>
#include <algorithm>
#include <iostream>
#include <iomanip>

#include "darknet_utils.h"
#include "NvInfer.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include "stb_image_resize.h"



bool file_exits(const std::string filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}

static void leftTrim(std::string &s) {
    s.erase(s.begin(), find_if(s.begin(), s.end(), [](int ch) { return !isspace(ch); }));
}

static void rightTrim(std::string &s) {
    s.erase(find_if(s.rbegin(), s.rend(), [](int ch) { return !isspace(ch); }).base(), s.end());
}

std::string trim(std::string s) {
    leftTrim(s);
    rightTrim(s);
    return s;
}

std::vector<std::string> split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<float> load_weights(const std::string weights_path, const std::string network_type) {
    std::vector<float> weights;

    std::ifstream file(weights_path, std::ios_base::binary);
    if (!file.good()) {
        std::cout << "open weight file failed !" << std::endl;
        return weights;
    }

    if (!(network_type == "yolov3" || network_type == "yolov3-tiny")) {
        std::cout << "Invalid network type" << std::endl;
        assert(0);
    }

    // Remove 5 int32 bytes of data from the stream belonging to the header
    file.ignore(4 * 5);

    char float_weight[4];

    while (!file.eof()) {
        file.read(float_weight, 4);
        assert(file.gcount() == 4);
        weights.push_back(*reinterpret_cast<float *> (float_weight));
        if (file.peek() == std::istream::traits_type::eof()) break;
    }

    std::cout << "Total Number of weights read : " << weights.size() << std::endl;
    std::cout << "Loading complete!" << std::endl;

    return weights;
}

int get_num_channels(nvinfer1::ITensor *t) {
    nvinfer1::Dims d = t->getDimensions();
    assert(d.nbDims >= 3);
    if (d.nbDims == 3)
        return d.d[0];
    else if (d.nbDims == 4) {
        return d.d[1];
    }
}

bool save_engine(const nvinfer1::ICudaEngine *engine, const std::string &file_name) {
    std::cout << "Serializing the TensorRT Engine..." << std::endl;
    assert(engine && "Invalid TensorRT Engine");
    auto model_stream = engine->serialize();
    if (!model_stream) {
        std::cout << "Engine serialization failed" << std::endl;
        return false;
    }
    assert(!file_name.empty() && "Enginepath is empty");

    // write data to output file
    std::stringstream gie_model_stream;
    gie_model_stream.seekg(0, std::stringstream::beg);
    gie_model_stream.write(static_cast<const char *>(model_stream->data()), model_stream->size());
    std::ofstream outFile;
    outFile.open(file_name);
    outFile << gie_model_stream.rdbuf();
    outFile.close();

    std::cout << "Serialized plan file cached at location : " << file_name << std::endl;
    return true;
}

nvinfer1::ICudaEngine *
load_trt_engine(const std::string plan_file, nvinfer1::ILogger &logger) {
    // reading the model in memory
    std::cout << "Loading TRT Engine..." << std::endl;
    assert(file_exits(plan_file));
    std::stringstream trt_model_stream;
    trt_model_stream.seekg(0, std::stringstream::beg);
    std::ifstream cache(plan_file);
    assert(cache.good());
    trt_model_stream << cache.rdbuf();
    cache.close();

    // calculating model size
    trt_model_stream.seekg(0, std::ios::end);
    const int model_size = trt_model_stream.tellg();
    trt_model_stream.seekg(0, std::ios::beg);
    void *model_mem = malloc(model_size);
    trt_model_stream.read((char *) model_mem, model_size);

    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine *engine
            = runtime->deserializeCudaEngine(model_mem, model_size);
    free(model_mem);
    runtime->destroy();
    std::cout << "Loading Complete!" << std::endl;

    return engine;
}

std::string dims_to_string(const nvinfer1::Dims d) {
    std::stringstream s;
    assert(d.nbDims >= 1);
    for (int i = 0; i < d.nbDims - 1; ++i) {
        s << std::setw(4) << d.d[i] << " x";
    }
    s << std::setw(4) << d.d[d.nbDims - 1];

    return s.str();
}

void print_layer_info(int layer_idx, std::string layer_name, nvinfer1::Dims input_dims,
                      nvinfer1::Dims output_dims, size_t weight_ptr) {
    std::cout << std::setw(6) << std::left << layer_idx << std::setw(15) << std::left << layer_name;
    std::cout << std::setw(20) << std::left << dims_to_string(input_dims) << std::setw(20) << std::left
              << dims_to_string(output_dims);
    std::cout << std::setw(6) << std::left << weight_ptr << std::endl;
}

void print_layer_info(std::string layer_idx, std::string layer_name, std::string input_dims, std::string output_dims,
                      std::string weight_ptr) {
    std::cout << std::setw(6) << std::left << layer_idx << std::setw(15) << std::left << layer_name;
    std::cout << std::setw(20) << std::left << input_dims << std::setw(20) << std::left
              << output_dims;
    std::cout << std::setw(6) << std::left << weight_ptr << std::endl;
}

float clamp(const float val, const float minVal, const float maxVal) {
    assert(minVal <= maxVal);
    return std::min(maxVal, std::max(minVal, val));
}

float clamp(const float val, const float minVal) {
    return std::max(minVal, val);
}

Tensor2BBoxes::Tensor2BBoxes(
        const unsigned int n_classes,
        const unsigned int n_bboxes,
        const std::vector<float> anchors,
        const int input_w, const int input_h) :
        n_classes(n_classes),
        n_bboxes(n_bboxes),
        anchors(anchors),
        input_w(input_w),
        input_h(input_h) {

}


Tensor2BBoxes::Tensor2BBoxes() {

}

std::vector<BBoxInfo>
Tensor2BBoxes::operator()(const float *detections, const std::vector<int> mask, const unsigned int grid_size,
                          const unsigned int stride, const float confidence_thresh, const int raw_w, const int raw_h) {
    float scale = std::min(static_cast<float>(input_h) / raw_h, static_cast<float>(input_w) / raw_w);
    float dx = (input_w - scale * raw_w) / 2;
    float dy = (input_h - scale * raw_h) / 2;

    std::vector<BBoxInfo> bboxes_info;

    for (unsigned int x = 0; x < grid_size; ++x) {
        for (unsigned int y = 0; y < grid_size; ++y) {
            for (unsigned int b = 0; b < n_bboxes; ++b) {
                const float pw = anchors[mask[b] * 2];
                const float ph = anchors[mask[b] * 2 + 1];

                const int num_girds = grid_size * grid_size;
                const int grid_idx = y * grid_size + x;

                // 立方体
                const float bx = x + detections[grid_idx + num_girds * (b * (5 + n_classes) + 0)];
                const float by = y + detections[grid_idx + num_girds * (b * (5 + n_classes) + 1)];
                const float bw = pw * detections[grid_idx + num_girds * (b * (5 + n_classes) + 2)];
                const float bh = ph * detections[grid_idx + num_girds * (b * (5 + n_classes) + 3)];

                const float obj_score = detections[grid_idx + num_girds * (b * (5 + n_classes) + 4)];


                float confidence_score = 0.0f;
                int label_idx = -1;

                for (int i = 0; i < n_classes; ++i) {
                    float prob = detections[grid_idx + num_girds * (b * (5 + n_classes) + (5 + i))];
                    //if ((int)prob == 1)
                    //{
                    //	std::cout << i << ": " << prob << std::endl;
                    //}
                    if (prob > confidence_score) {
                        confidence_score = prob;
                        label_idx = i;
                    }
                }

                confidence_score *= obj_score;

                if (confidence_score > confidence_thresh) {
                    BBoxInfo binfo;
                    //if (bw >= -1e-9 && bw < 1e-9)
                    //	continue;
                    binfo.box = convert_bbox(bx, by, bw, bh, stride, input_w, input_h);

                    if ((binfo.box.x1 > binfo.box.x2) || (binfo.box.y1 > binfo.box.y2)) {
                        continue;
                    }

                    binfo.box.x1 -= dx;
                    binfo.box.x2 -= dx;
                    binfo.box.y1 -= dy;
                    binfo.box.y2 -= dy;

                    binfo.box.x1 /= scale;
                    binfo.box.x2 /= scale;
                    binfo.box.y1 /= scale;
                    binfo.box.y2 /= scale;

                    binfo.label = label_idx;
                    binfo.prob = confidence_score;

                    bboxes_info.push_back(binfo);
                }
            }
        }
    }

    return bboxes_info;
}

BBox
Tensor2BBoxes::convert_bbox(const float &bx, const float &by, const float &bw, const float &bh,
                            const int &stride, const uint &net_w, const uint &net_h) {
    BBox b;

    float x = bx * stride;
    float y = by * stride;

    b.x1 = x - bw / 2;
    b.x2 = x + bw / 2;

    b.y1 = y - bh / 2;
    b.y2 = y + bh / 2;

    b.x1 = clamp(b.x1, 0, net_w);
    b.x2 = clamp(b.x2, 0, net_w);
    b.y1 = clamp(b.y1, 0, net_h);
    b.y2 = clamp(b.y2, 0, net_h);

    return b;
}


std::vector<BBoxInfo> nms(std::vector<BBoxInfo> &binfo, float nms_thresh) {
    std::vector<BBoxInfo> res;

    if (binfo.size() == 0) {
        return res;
    }

    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min) {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };
    auto computeIoU = [&overlap1D](BBox &bbox1, BBox &bbox2) -> float {
        float overlapX = overlap1D(bbox1.x1, bbox1.x2, bbox2.x1, bbox2.x2);
        float overlapY = overlap1D(bbox1.y1, bbox1.y2, bbox2.y1, bbox2.y2);
        float area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
        float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::stable_sort(binfo.begin(), binfo.end(),
                     [](const BBoxInfo &b1, const BBoxInfo &b2) { return b1.prob > b2.prob; });
    std::vector<BBoxInfo> out;
    for (auto &i : binfo) {
        bool keep = true;
        for (auto &j : out) {
            if (keep) {
                float overlap = computeIoU(i.box, j.box);
                keep = overlap <= nms_thresh;
            } else
                break;
        }
        if (keep) out.push_back(i);
    }
    return out;
}

std::vector<std::string> load_list_from_text_file(const std::string& filename){
    assert(file_exits(filename));
    std::vector<std::string> list;

    std::ifstream f(filename);
    if(!f){
        std::cout << "failed to open " << filename;
        assert(0);
    }

    std::string line;
    while (std::getline(f, line)){
        if (line.empty())
            continue;
        else
            list.push_back(trim(line));
    }

    return list;
}

std::vector<std::string> load_image_list(const::std::string& txt_filename, const std::string& prefix){
    std::vector<std::string> image_list;
    std::vector<std::string> file_list = load_list_from_text_file(txt_filename);

    for(auto& file : file_list){
        if (file_exits(file)){
            image_list.push_back(file);
        }
        else{
            std::string prefixed = prefix + file;
            if(file_exits(prefixed))
                image_list.push_back(prefixed);
            else
                std::cerr << "WARNING: couldn't find: " << prefixed
                          << " while loading: " << txt_filename << std::endl;
        }
    }

    assert(!image_list.empty());
    return image_list;
}

Img read_image(const std::string& filepath) {
    Img image;

    unsigned char* data = stbi_load(filepath.c_str(), &image.w, &image.h, &image.c, 3);
    if(filepath.empty() or data == nullptr){
        return image;
    }
    image.data = std::shared_ptr<float>(new float [image.w * image.h * image.c], [](float *p){delete [] p;});
    for(size_t i = 0; i < image.w * image.h * image.c; ++i){
        image.data.get()[i] = static_cast<float >(data[i]);
    }
    return image;
}

void hwc2chw(const float* src, float* dst, int c, int h, int w) {
    for (int k = 0; k < c; ++k) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int dst_index = x + w * y + w * h * k;
                int src_index = k + c * x + c * w * y;
                dst[dst_index] = src[src_index];
            }
        }
    }
}


Img blob_from_images(const std::vector<Img> &input_images, const int &input_w, const int &input_h) {
    Img blob;
    size_t num_images = input_images.size();

    auto one_image_size = input_w * input_h * 3;
    blob.data = std::shared_ptr<float>(new float [num_images * one_image_size]);
    if(!blob.data){
        assert(0);
    }
    float* ptr = blob.data.get();

    auto resized_data = new float [one_image_size];

    for(auto& im : input_images){
        auto ret = stbir_resize_float(im.data.get(), im.w, im.h, 0, resized_data, input_w, input_h, 0, 3);
        if(ret){
            hwc2chw(resized_data, ptr, 3, input_h, input_w);
            ptr += one_image_size;
        }
    }

    return blob;
}


