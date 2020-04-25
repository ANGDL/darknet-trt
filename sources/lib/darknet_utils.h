#ifndef __DARKNET_UTILS_H__
#define __DARKNET_UTILS_H__

#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <assert.h>

#include "NvInfer.h"

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

struct BBoxInfo
{
	BBox box;
	int label;
	float prob;
};

struct Tensor2BBoxes
{
	Tensor2BBoxes();
	Tensor2BBoxes(const unsigned int n_classes, const unsigned int n_bboxes,
		const std::vector<float> anchors, const int input_w, const int input_h);
	std::vector<BBoxInfo> operator()(const float* detections, const std::vector<int> mask, const unsigned int gridSize, const unsigned int stride, const float confidence_thresh, const int raw_w, const int raw_h);
	BBox convert_bbox(const float& bx, const float& by, const float& bw, const float& bh, const int& stride);

	unsigned int n_classes;
	unsigned int n_bboxes;
	std::vector<float> anchors;
	int input_w;
	int input_h;
};

std::vector<BBoxInfo> nms(std::vector<BBoxInfo>& bboxes, float nms_thresh);

bool file_exits(const std::string filename);

std::string trim(std::string s);

std::vector<std::string> split(const std::string& s, char delimiter);

std::vector<float> load_weights(const std::string weights_path, const std::string network_type);

int get_num_channels(nvinfer1::ITensor* t);

bool save_engine(const nvinfer1::ICudaEngine* engine, const std::string& file_name);

nvinfer1::ICudaEngine* load_trt_engine(const std::string plan_file, nvinfer1::IPluginFactory* plugin_factory, nvinfer1::ILogger& logger);

void print_layer_info(int layer_idx, std::string layer_name, nvinfer1::Dims input_dims,
	nvinfer1::Dims output_dims, size_t weight_ptr);

void print_layer_info(std::string layer_idx, std::string layer_name, std::string input_dims,
	std::string output_dims, std::string weight_ptr);

float clamp(const float val, const float minVal, const float maxVal);

float clamp(const float val, const float minVal);


namespace darknet {
	template<typename T>
	void write(char*& buffer, const T& val) {
		*reinterpret_cast<T*>(buffer) = val;
		buffer += sizeof(T);
	}

	template <typename T>
	void read(const char*& buffer, T& val) {
		val = *reinterpret_cast<const T*>(buffer);
		buffer += sizeof(T);
	}
}

#endif
