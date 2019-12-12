#include "yolov3.h"

darknet::YoloV3::YoloV3(NetConfig* config,
	uint batch_size,
	float confidence_thresh,
	float nms_thresh) :
	Yolo(config, batch_size, confidence_thresh, nms_thresh),
	net_cfg(dynamic_cast<YoloV3Cfg*>(config)),
	output_index_1(-1),
	output_index_2(-1),
	output_index_3(-1)
{
	if (net_cfg == nullptr) {
		std::cout << "cast config to YoloV3Tiny config error!" << std::endl;
		is_init = false;
		return;
	}
	convert_to_bboxes = Tensor2BBoxes(net_cfg->OUTPUT_CLASSES, net_cfg->BBOXES, net_cfg->ANCHORS, net_cfg->INPUT_W, net_cfg->INPUT_H);
	//分配内存
	assert(trt_output_buffers.size() == 3 && bindings.size() == 4);
	output_index_1 = engine->getBindingIndex(net_cfg->OUTPUT_BLOB_NAME_1.c_str());
	output_index_2 = engine->getBindingIndex(net_cfg->OUTPUT_BLOB_NAME_2.c_str());
	output_index_3 = engine->getBindingIndex(net_cfg->OUTPUT_BLOB_NAME_3.c_str());

	assert(output_index_1 != -1);
	assert(output_index_2 != -1);
	assert(output_index_3 != -1);

	NV_CUDA_CHECK(cudaMalloc(&bindings[input_index], (size_t)batch_size * net_cfg->INPUT_SIZE * sizeof(float)));
	NV_CUDA_CHECK(cudaMalloc(&bindings[output_index_1], (size_t)batch_size * net_cfg->OUTPUT_SIZE_1 * sizeof(float)));
	NV_CUDA_CHECK(cudaMalloc(&bindings[output_index_2], (size_t)batch_size * net_cfg->OUTPUT_SIZE_2 * sizeof(float)));
	NV_CUDA_CHECK(cudaMalloc(&bindings[output_index_3], (size_t)batch_size * net_cfg->OUTPUT_SIZE_3 * sizeof(float)));

	NV_CUDA_CHECK(cudaMallocHost(&trt_output_buffers[0], (size_t)batch_size * net_cfg->OUTPUT_SIZE_1 * sizeof(float)));
	NV_CUDA_CHECK(cudaMallocHost(&trt_output_buffers[1], (size_t)batch_size * net_cfg->OUTPUT_SIZE_2 * sizeof(float)));
	NV_CUDA_CHECK(cudaMallocHost(&trt_output_buffers[2], (size_t)batch_size * net_cfg->OUTPUT_SIZE_3 * sizeof(float)));
}

void darknet::YoloV3::infer(const unsigned char* input)
{
	NV_CUDA_CHECK(cudaMemcpyAsync(bindings[input_index], input, (size_t)batch_size * net_cfg->INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, cuda_stream));
	context->enqueue(batch_size, bindings.data(), cuda_stream, nullptr);
	NV_CUDA_CHECK(cudaMemcpyAsync(trt_output_buffers[0], bindings[output_index_1], (size_t)batch_size * net_cfg->OUTPUT_SIZE_1 * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream));
	NV_CUDA_CHECK(cudaMemcpyAsync(trt_output_buffers[1], bindings[output_index_2], (size_t)batch_size * net_cfg->OUTPUT_SIZE_2 * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream));
	NV_CUDA_CHECK(cudaMemcpyAsync(trt_output_buffers[2], bindings[output_index_3], (size_t)batch_size * net_cfg->OUTPUT_SIZE_3 * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream));

	cudaStreamSynchronize(cuda_stream);
}

std::vector<BBoxInfo> darknet::YoloV3::get_detecions(const int image_idx, const int image_w, const int image_h)
{
	std::vector<BBoxInfo> bboxes1 = convert_to_bboxes(
		&trt_output_buffers[0][image_idx * net_cfg->OUTPUT_SIZE_1],
		net_cfg->MASK_1,
		net_cfg->GRID_SIZE_1,
		net_cfg->STRIDE_1,
		prob_thresh,
		image_w,
		image_h
	);

	std::vector<BBoxInfo> bboxes2 = convert_to_bboxes(
		&trt_output_buffers[1][image_idx * net_cfg->OUTPUT_SIZE_2],
		net_cfg->MASK_2,
		net_cfg->GRID_SIZE_2,
		net_cfg->STRIDE_2,
		prob_thresh,
		image_w,
		image_h
	);

	std::vector<BBoxInfo> bboxes3 = convert_to_bboxes(
		&trt_output_buffers[2][image_idx * net_cfg->OUTPUT_SIZE_3],
		net_cfg->MASK_3,
		net_cfg->GRID_SIZE_3,
		net_cfg->STRIDE_3,
		prob_thresh,
		image_w,
		image_h
	);

	std::vector<BBoxInfo> res;
	res.insert(res.end(), bboxes1.begin(), bboxes1.end());
	res.insert(res.end(), bboxes2.begin(), bboxes2.end());
	res.insert(res.end(), bboxes3.begin(), bboxes3.end());

	return nms(res, nms_thresh);
}
