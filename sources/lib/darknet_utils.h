﻿#ifndef __DARKNET_UTILS_H__
#define __DARKNET_UTILS_H__

#include <filesystem>
#include <iostream>
#include <vector>

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

#endif
