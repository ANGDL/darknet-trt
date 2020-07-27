//
// Created by ANG on 2020/7/24.
//

#ifndef DARKNET_TRT_CALIBRATOR_H
#define DARKNET_TRT_CALIBRATOR_H

#include "NvInfer.h"
#include "darknet_utils.h"

#include <string>
#include <vector>


class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator(const uint &batch_size, const std::string &calib_images,
                          const std::string &calib_image_path,
                          const std::string &calib_table_file_path,
                          const uint64_t &input_size,
                          const uint &input_h,
                          const uint &input_w,
                          const std::string &input_blob_name);

    virtual ~Int8EntropyCalibrator();

    int getBatchSize() const override;
    bool getBatch(void* bindings[], const char* names[], int nbBindings) override;
    const void* readCalibrationCache(size_t& length) override;
    void writeCalibrationCache(const void* cache, size_t length) override;

private:
    const uint batch_size_;
    const uint input_w_;
    const uint input_h_;
    const uint64_t input_size_;
    const uint64_t input_count_;
    const std::string input_blob_name_;
    const std::string calib_table_file_path_;
    uint image_index_{0};
    bool read_cache_{true};
    void *device_input{nullptr};
    std::vector<std::string> image_list_;
    std::vector<char> calibration_cache_;
};

#endif
