#include "decode_kernel.cuh"

#include <cstdint>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/tabulate.h>
#include <thrust/count.h>
#include <thrust/find.h>
#include <thrust/system/cuda/detail/cub/device/device_radix_sort.cuh>
#include <thrust/system/cuda/detail/cub/iterator/counting_input_iterator.cuh>

#include "darknet_utils.h"


int cuda_decode_layer(const void* const* inputs, void** output, int batch_size, float stride,
	size_t grid_size, size_t num_anchors, size_t num_classes, const std::vector<float>& anchors,
	float score_thresh,  void* workspace, size_t workspace_size, cudaStream_t stream)
{

	size_t obj_scores_size = num_anchors * grid_size * grid_size;
	size_t pred_size = num_anchors * (5 + num_classes) * grid_size * grid_size;

	if (!workspace || !workspace_size) {
		workspace_size += get_size_aligned<float>(anchors.size());  // anchors
		return workspace_size;
	}

	auto anchors_d = get_next_ptr<float>(anchors.size(), workspace, workspace_size);
	cudaMemcpyAsync(anchors_d, anchors.data(), anchors.size() * sizeof(float), cudaMemcpyHostToDevice, stream);

	auto on_stream = thrust::cuda::par.on(stream);

	for (int batch = 0; batch < batch_size; ++batch) {
		auto detections = static_cast<const float*>(inputs[0]) + batch * pred_size;

		auto out_scores = static_cast<float*>(output[0]) + batch * grid_size * grid_size * num_anchors;
		auto out_boxes = static_cast<float4*>(output[1]) + batch * grid_size * grid_size * num_anchors ;
		auto out_classes = static_cast<float*>(output[2]) + batch * grid_size * grid_size * num_anchors;


		// 收集boxes
		auto index_iter = thrust::cuda_cub::cub::CountingInputIterator<int>(0);

		thrust::transform(
			on_stream,
			index_iter,
			index_iter + obj_scores_size,
			thrust::make_zip_iterator(
				thrust::make_tuple(out_scores, out_boxes, out_classes)
			),
			[=]__device__(int i) {
			int x = i % grid_size;
			int y = (i / grid_size) % grid_size;
			int b = (i / grid_size / grid_size) % num_anchors;

			assert(b < 3);

			const int num_girds = grid_size * grid_size;
			const int grid_idx = y * grid_size + x;

			const float pw = anchors_d[b * 2];
			const float ph = anchors_d[b * 2 + 1];

			const float bx = x + detections[grid_idx + num_girds * (b * (5 + num_classes) + 0)];
			const float by = y + detections[grid_idx + num_girds * (b * (5 + num_classes) + 1)];
			const float bw = pw * detections[grid_idx + num_girds * (b * (5 + num_classes) + 2)];
			const float bh = ph * detections[grid_idx + num_girds * (b * (5 + num_classes) + 3)];

			const float obj_score = detections[grid_idx + num_girds * (b * (5 + num_classes) + 4)];

			float bbx = bx * stride;
			float bby = by * stride;

			float x1 = bbx - bw / 2;
			float y1 = bby - bh / 2;
			float x2 = bbx + bw / 2;
			float y2 = bby + bh / 2;

			float4 box = float4{
				x1,
				y1,
				x2, 
				y2
			};

			float confidence_score = -9.9f;
			int label_idx = -1;

			for (unsigned int i = 0; i < num_classes; ++i)
			{
				float prob = detections[grid_idx + num_girds * (b * (5 + num_classes) + (5 + i))];

				if (prob > confidence_score)
				{
					confidence_score = prob;
					label_idx = i;
				}
			}

			confidence_score *= obj_score;

			if (confidence_score < score_thresh) {
				confidence_score = -9.9f;
			}
/*			else {
				printf("%f, %f, %f , %f, %f, %d\n", confidence_score, box.x, box.y, box.z, box.w, label_idx);
			}*/		
			return thrust::make_tuple(confidence_score, box, (float)label_idx);
		}
		);
	}
	return 0;
}
