#include "yolov3-nms.h"

darknet::YoloV3NMS::YoloV3NMS(NetConfig* cfg,
	uint batch_size,
	float confidence_thresh,
	float nms_thresh) :
	Yolo(cfg, batch_size, confidence_thresh, nms_thresh),
	output_index_1(-1),
	output_index_2(-1),
	output_index_3(-1)
{

	//分配内存
	assert(trt_output_buffers.size() == 3 && bindings.size() == 4);
	output_index_1 = engine->getBindingIndex("scores");
	output_index_2 = engine->getBindingIndex("boxes");
	output_index_3 = engine->getBindingIndex("classes");

	assert(output_index_1 != -1);
	assert(output_index_2 != -1);
	assert(output_index_3 != -1);

	NV_CUDA_CHECK(cudaMalloc(&bindings[input_index], (size_t)batch_size * config->INPUT_SIZE * sizeof(float)));
	NV_CUDA_CHECK(cudaMalloc(&bindings[output_index_1], (size_t)batch_size * config->max_detection * sizeof(float)));
	NV_CUDA_CHECK(cudaMalloc(&bindings[output_index_2], (size_t)batch_size * config->max_detection * sizeof(float4)));
	NV_CUDA_CHECK(cudaMalloc(&bindings[output_index_3], (size_t)batch_size * config->max_detection * sizeof(float)));

	NV_CUDA_CHECK(cudaMallocHost(&trt_output_buffers[0], (size_t)batch_size * config->max_detection * sizeof(float)));
	NV_CUDA_CHECK(cudaMallocHost(&trt_output_buffers[1], (size_t)batch_size * config->max_detection * sizeof(float4)));
	NV_CUDA_CHECK(cudaMallocHost(&trt_output_buffers[2], (size_t)batch_size * config->max_detection * sizeof(float)));
}

void darknet::YoloV3NMS::infer(const unsigned char* input)
{
	NV_CUDA_CHECK(cudaMemcpyAsync(bindings[input_index], input, (size_t)batch_size * config->INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, cuda_stream));
	context->enqueue(batch_size, bindings.data(), cuda_stream, nullptr);
	NV_CUDA_CHECK(cudaMemcpyAsync(trt_output_buffers[0], bindings[output_index_1], (size_t)batch_size * config->max_detection * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream));
	NV_CUDA_CHECK(cudaMemcpyAsync(trt_output_buffers[1], bindings[output_index_2], (size_t)batch_size * config->max_detection * sizeof(float4), cudaMemcpyDeviceToHost, cuda_stream));
	NV_CUDA_CHECK(cudaMemcpyAsync(trt_output_buffers[2], bindings[output_index_3], (size_t)batch_size * config->max_detection * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream));

	cudaStreamSynchronize(cuda_stream);
}

std::vector<std::vector<BBoxInfo>> darknet::YoloV3NMS::get_detecions(const int image_w, const int image_h)
{
	float scale = std::min(static_cast<float>(config->INPUT_W) / image_w, static_cast<float>(config->INPUT_H) / image_h);
	float dx = (config->INPUT_W - scale * image_w) / 2;
	float dy = (config->INPUT_H - scale * image_h) / 2;

	std::vector<std::vector<BBoxInfo>> ret;
	for (int b = 0; b < batch_size; ++b)
	{
		float* scores = trt_output_buffers[0] + b * config->max_detection;
		float* boxes = trt_output_buffers[1] + b * config->max_detection * 4;
		float* classes = trt_output_buffers[2] + b * config->max_detection;

		std::vector<BBoxInfo> ret_b;
		for (int i = 0; i < config->max_detection; ++i)
		{
			if (scores[i] > config->score_thresh) {
				BBoxInfo binfo;
				binfo.prob = scores[i];
				binfo.label = classes[i];

				float x1 = boxes[i * 4 + 0];
				float y1 = boxes[i * 4 + 1];
				float x2 = boxes[i * 4 + 2];
				float y2 = boxes[i * 4 + 3];

				//x1 -= dx;
				//x2 -= dx;
				//y1 -= dy;
				//y2 -= dy;

				//x1 /= scale;
				//x2 /= scale;
				//y1 /= scale;
				//y2 /= scale;

				x1 = x1 * image_w / config->INPUT_W;
				x2 = x2 * image_w / config->INPUT_W;
				y1 = y1 * image_h / config->INPUT_H;
				y2 = y2 * image_h / config->INPUT_H;

				x1 = clamp(x1, 0, image_w);
				x2 = clamp(x2, 0, image_w);
				y1 = clamp(y1, 0, image_h);
				y2 = clamp(y2, 0, image_h);

				binfo.box = { x1, y1, x2, y2 };

				ret_b.push_back(binfo);
			}
		}

		ret.push_back(ret_b);
	}
	return ret;
}
