#ifndef __DARKNET_UTILS_H__
#define __DARKNET_UTILS_H__

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cassert>
#include <memory>

#include "NvInfer.h"
#include "opencv2/opencv.hpp"

#define NV_CUDA_CHECK(status)                                                                      \
    {                                                                                              \
        if (status != 0)                                                                           \
        {                                                                                          \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) << " in file " << __FILE__ \
                      << " at line " << __LINE__ << std::endl;                                     \
            abort();                                                                               \
        }                                                                                          \
    }

struct BBox {
    float x1;
    float y1;
    float x2;
    float y2;
};

struct BBoxInfo {
    BBox box;
    int label;
    float prob;
};

struct Img{
    std::shared_ptr<float> data = nullptr;
    int w = 0;
    int h = 0;
    int c = 0;

    Img(float* data, int w, int h, int c):
    data(std::shared_ptr<float>(data, [](float *p){delete [] p;})),
    w(w), h(h), c(c){ }

    Img(int w, int h, int c): w(w), h(h), c(c),
    data(std::shared_ptr<float>(new float [c * w * h], [](float *p){delete [] p;})){ }

    Img(): data(nullptr), w(0), h(0), c(0){}

    static float get_pixel(const Img& im, int x, int y, int c) {
        if (x < 0) x = 0;
        if (x >= im.w) x = im.w - 1;
        if (y < 0) y = 0;
        if (y >= im.h) y = im.h - 1;
        if (c < 0) c = 0;
        if (c >= im.c) c = im.c - 1;

        // get the index
        int idx = x + im.w * y + im.w * im.h * c;
        assert(idx < im.w * im.h * im.c && idx >= 0);
        return *(im.data.get() + idx);
    }

    static void set_pixel(Img& im,  int x, int y, int c, float v){
        int idx = x + im.w * y + im.w * im.h * c;
        if (!(idx < im.w * im.h * im.c && idx >= 0)) {
//            printf("%d %d %d\n", x, y, c);
            return;
        }
        //assert(idx < im.w * im.h * im.c && idx >= 0);
        im.data.get()[idx] = v;
    }
};

struct Tensor2BBoxes {
    Tensor2BBoxes();

    Tensor2BBoxes(unsigned int n_classes, unsigned int n_bboxes,
                  std::vector<float> anchors, int input_w, int input_h);

    std::vector<BBoxInfo> operator()(const float *detections, std::vector<int> mask, unsigned int gridSize,
                                     unsigned int stride, float confidence_thresh, int raw_w,
                                     int raw_h);

    BBox convert_bbox(const float &bx, const float &by, const float &bw, const float &bh,
                      const int &stride, const uint &net_w, const uint &net_h);

    unsigned int n_classes;
    unsigned int n_bboxes;
    std::vector<float> anchors;
    int input_w;
    int input_h;
};

std::vector<BBoxInfo> nms(std::vector<BBoxInfo> &bboxes, float nms_thresh);

bool file_exits(std::string filename);

std::string trim(std::string s);

std::vector<std::string> split(const std::string &s, char delimiter);

std::vector<float> load_weights(std::string weights_path, std::string network_type);

int get_num_channels(nvinfer1::ITensor *t);

bool save_engine(const nvinfer1::ICudaEngine *engine, const std::string &file_name);

nvinfer1::ICudaEngine *
load_trt_engine(std::string plan_file, nvinfer1::ILogger &logger);

void print_layer_info(int layer_idx, std::string layer_name, nvinfer1::Dims input_dims,
                      nvinfer1::Dims output_dims, size_t weight_ptr);

void print_layer_info(std::string layer_idx, std::string layer_name, std::string input_dims,
                      std::string output_dims, std::string weight_ptr);

float clamp(float val, float minVal, float maxVal);

float clamp(float val, float minVal);

namespace darknet {
    template<typename T>
    void write(char *&buffer, const T &val) {
        *reinterpret_cast<T *>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T>
    void read(const char *&buffer, T &val) {
        val = *reinterpret_cast<const T *>(buffer);
        buffer += sizeof(T);
    }
}

#define CUDA_ALIGN 256


template<typename T>
inline size_t get_size_aligned(size_t num_elem) {
    size_t size = num_elem * sizeof(T);
    size_t extra_align = 0;
    if (size % CUDA_ALIGN != 0) {
        extra_align = CUDA_ALIGN - size % CUDA_ALIGN;
    }
    return size + extra_align;
}

template<typename T>
inline T *get_next_ptr(size_t num_elem, void *&workspace, size_t &workspace_size) {
    size_t size = get_size_aligned<T>(num_elem);
    if (size > workspace_size) {
        throw std::runtime_error("Workspace is too small!");
    }
    workspace_size -= size;
    T *ptr = reinterpret_cast<T *>(workspace);
    workspace = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(workspace) + size);
    return ptr;
}

std::vector<std::string> load_list_from_text_file(const std::string& filename);

std::vector<std::string> load_image_list(const::std::string& txt_filename, const std::string& prefix);

Img read_image(const std::string& filepath);

Img blob_from_images(const std::vector<Img> &input_images, const int &input_w, const int &input_h);

#endif
