//
// Created by x on 2020/7/24.
//
#include <algorithm>
#include <cuda_runtime.h>

#include "calibrator.h"


Int8EntropyCalibrator::Int8EntropyCalibrator(const uint &batch_size, const std::string &calib_images,
                                             const std::string &calib_image_path,
                                             const std::string &calib_table_file_path, const uint64_t &input_size,
                                             const uint &input_h, const uint &input_w,
                                             const std::string &input_blob_name):
        batch_size_(batch_size),
        input_h_(input_h),
        input_w_(input_w),
        input_size_(input_size),
        input_count_(batch_size * input_size),
        input_blob_name_(input_blob_name),
        calib_table_file_path_(calib_table_file_path)
        {
            if(!file_exits(calib_table_file_path_)){
                image_list_ = load_image_list(calib_images, calib_image_path);
                image_list_.resize(static_cast<int>(image_list_.size() / batch_size) * batch_size);
                std::random_shuffle(image_list_.begin(), image_list_.end(),[](int i){ return rand() % i;});
            }

            NV_CUDA_CHECK(cudaMalloc(&device_input, input_count_ * sizeof(float)));
        }

Int8EntropyCalibrator::~Int8EntropyCalibrator() {
    NV_CUDA_CHECK(cudaFree(device_input));
}

int Int8EntropyCalibrator::getBatchSize() const {
    return batch_size_;
}

bool Int8EntropyCalibrator::getBatch(void **bindings, const char **names, int nbBindings) {
    if (image_index_ + batch_size_ >= image_list_.size())
        return false;

    std::vector<cv::Mat> one_batch_images(batch_size_);
    for(uint j = image_index_; j < image_index_ + batch_size_; ++j){
        one_batch_images.at(j - image_index_) = cv::imread(image_list_.at(j));
    }
    image_index_ += batch_size_;

    cv::Mat trt_input = blob_from_mats(one_batch_images, input_w_, input_h_);
    NV_CUDA_CHECK(cudaMemcpy(
            device_input, trt_input.ptr<float>(0), input_count_*sizeof(float), cudaMemcpyHostToDevice));

    assert(!strcmp(names[0], input_blob_name_.c_str()));
    bindings[0] = device_input;
    return true;
}

const void *Int8EntropyCalibrator::readCalibrationCache(size_t &length) {
    void* output{nullptr};
    calibration_cache_.clear();

    assert(!calib_table_file_path_.empty());
    std::ifstream  input(calib_table_file_path_, std::ios::binary);
    input >> std::noskipws;
    if (read_cache_ && input.good())
        std::copy(std::istreambuf_iterator<char>(input),
                std::istreambuf_iterator<char>(), std::back_inserter(calibration_cache_));

    length = calibration_cache_.size();
    if(length){
        std::cout << "Using cached calibration table to build the engine" << std::endl;
        output = &calibration_cache_[0];
    }
    else{
        std::cout << "New calibration table will be created to build the engine" << std::endl;
        output = nullptr;
    }

    return output;
}

void Int8EntropyCalibrator::writeCalibrationCache(const void *cache, size_t length) {
    assert(!calib_table_file_path_.empty());
    std::ofstream  output(calib_table_file_path_, std::ios::binary);
    output.write(reinterpret_cast<const char* >(cache), length);
    output.close();
}

