#include "yolov3-nms.h"

darknet::YoloV3NMS::YoloV3NMS(NetConfig *cfg, uint batch_size) :
        Yolo(cfg, batch_size),
        output_index_1_(-1),
        output_index_2_(-1),
        output_index_3_(-1) {

    //分配内存
    assert(trt_output_buffers_.size() == 3 && bindings_.size() == 4);
    output_index_1_ = engine_->getBindingIndex("scores");
    output_index_2_ = engine_->getBindingIndex("boxes");
    output_index_3_ = engine_->getBindingIndex("classes");

    assert(output_index_1_ != -1);
    assert(output_index_2_ != -1);
    assert(output_index_3_ != -1);

    NV_CUDA_CHECK(cudaMalloc(&bindings_[input_index_], (size_t) batch_size * config_->INPUT_SIZE * sizeof(float)));
    NV_CUDA_CHECK(
            cudaMalloc(&bindings_[output_index_1_], (size_t) batch_size * config_->max_detection_ * sizeof(float)));
    NV_CUDA_CHECK(
            cudaMalloc(&bindings_[output_index_2_], (size_t) batch_size * config_->max_detection_ * sizeof(float4)));
    NV_CUDA_CHECK(
            cudaMalloc(&bindings_[output_index_3_], (size_t) batch_size * config_->max_detection_ * sizeof(float)));

    NV_CUDA_CHECK(
            cudaMallocHost(&trt_output_buffers_[0], (size_t) batch_size * config_->max_detection_ * sizeof(float)));
    NV_CUDA_CHECK(
            cudaMallocHost(&trt_output_buffers_[1], (size_t) batch_size * config_->max_detection_ * sizeof(float4)));
    NV_CUDA_CHECK(
            cudaMallocHost(&trt_output_buffers_[2], (size_t) batch_size * config_->max_detection_ * sizeof(float)));
}

void darknet::YoloV3NMS::infer(const unsigned char *input) {
    NV_CUDA_CHECK(
            cudaMemcpyAsync(bindings_[input_index_], input, (size_t) batch_size_ * config_->INPUT_SIZE * sizeof(float),
                            cudaMemcpyHostToDevice, cuda_stream_));
    context_->enqueue(batch_size_, bindings_.data(), cuda_stream_, nullptr);
    NV_CUDA_CHECK(cudaMemcpyAsync(trt_output_buffers_[0], bindings_[output_index_1_],
                                  (size_t) batch_size_ * config_->max_detection_ * sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  cuda_stream_));
    NV_CUDA_CHECK(cudaMemcpyAsync(trt_output_buffers_[1], bindings_[output_index_2_],
                                  (size_t) batch_size_ * config_->max_detection_ * sizeof(float4),
                                  cudaMemcpyDeviceToHost,
                                  cuda_stream_));
    NV_CUDA_CHECK(cudaMemcpyAsync(trt_output_buffers_[2], bindings_[output_index_3_],
                                  (size_t) batch_size_ * config_->max_detection_ * sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  cuda_stream_));

    cudaStreamSynchronize(cuda_stream_);
}

std::vector<std::vector<BBoxInfo>> darknet::YoloV3NMS::get_detecions(const int image_w, const int image_h) {

    std::vector<std::vector<BBoxInfo>> ret;
    for (int b = 0; b < batch_size_; ++b) {
        float *scores = trt_output_buffers_[0] + b * config_->max_detection_;
        float *boxes = trt_output_buffers_[1] + b * config_->max_detection_ * 4;
        float *classes = trt_output_buffers_[2] + b * config_->max_detection_;

        std::vector<BBoxInfo> ret_b;
        for (int i = 0; i < config_->max_detection_; ++i) {
            if (scores[i] > config_->score_thresh_) {
                BBoxInfo binfo;
                binfo.prob = scores[i];
                binfo.label = classes[i];

                float x1 = boxes[i * 4 + 0];
                float y1 = boxes[i * 4 + 1];
                float x2 = boxes[i * 4 + 2];
                float y2 = boxes[i * 4 + 3];

                x1 = x1 * image_w / config_->INPUT_W;
                x2 = x2 * image_w / config_->INPUT_W;
                y1 = y1 * image_h / config_->INPUT_H;
                y2 = y2 * image_h / config_->INPUT_H;

                x1 = clamp(x1, 0, image_w);
                x2 = clamp(x2, 0, image_w);
                y1 = clamp(y1, 0, image_h);
                y2 = clamp(y2, 0, image_h);

                binfo.box = {x1, y1, x2, y2};

                ret_b.push_back(binfo);
            }
        }

        ret.push_back(ret_b);
    }
    return ret;
}
