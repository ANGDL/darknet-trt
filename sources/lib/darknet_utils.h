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
#endif
