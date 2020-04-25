#include "decode_kernel.cuh"

#include <cstdint>

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


int cuda_decode_layer(const void* input, void** output, int batch_size, float stride,
	size_t grid_size, size_t num_anchors, size_t num_classes, const std::vector<float>& anchors,
	float score_thresh, int top_n, void* workspace, size_t workspace_size, cudaStream_t stream)
{
	int scores_size = num_anchors * num_classes * grid_size * grid_size;
	int boxes_size = num_anchors * 4 * grid_size * grid_size;
	int pred_size = num_anchors * (5 + num_classes) * grid_size * grid_size;

	if (!workspace || !workspace_size) {
		workspace_size = get_size_aligned<int>(pred_size); //  partition flags
		workspace_size += get_size_aligned<float>(scores_size);  //socres
		workspace_size += get_size_aligned<float>(boxes_size);  //boxes
		workspace_size += get_size_aligned<float>(anchors.size());  // anchors
		workspace_size += get_size_aligned<bool>(scores_size);   //flags
		workspace_size += get_size_aligned<int>(scores_size);   // indices
		workspace_size += get_size_aligned<int>(scores_size);   // indices_sorted

		workspace_size += get_size_aligned<float>(scores_size); // socrs_sorted


		// 获取这两步操作需要用到的临时空间
		size_t temp_size_flag = 0;
		thrust::cuda_cub::cub::DeviceSelect::Flagged(
			(void*)nullptr,
			temp_size_flag,  // temp_storage_bytes
			thrust::cuda_cub::cub::CountingInputIterator<int>(scores_size),  // InputIteratorT 
			(bool*)nullptr,  // FlagIterator 
			(int*)nullptr,   // OutputIteratorT
			(int*)nullptr,  // NumSelectedIteratorT 
			scores_size);  // num_items

		size_t temp_size_sort = 0;
		thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(
			(void*)nullptr,
			temp_size_sort,
			(float*)nullptr,
			(float*)nullptr,
			(int*)nullptr,
			(int*)nullptr,
			scores_size);

		workspace_size += max(temp_size_flag, temp_size_sort);
		return workspace_size;
	}

	auto anchors_d = get_next_ptr<float>(anchors.size(), workspace, workspace_size);
	cudaMemcpyAsync(anchors_d, anchors.data(), anchors.size() * sizeof(float), cudaMemcpyHostToDevice, stream);

	auto on_stream = thrust::cuda::par.on(stream);

	auto partition_flags = get_next_ptr<float>(pred_size, workspace, workspace_size);

	auto scores = get_next_ptr<float>(scores_size, workspace, workspace_size);
	auto boxes = get_next_ptr<float>(boxes_size, workspace, workspace_size);

	auto flags = get_next_ptr<bool>(scores_size, workspace, workspace_size);
	auto indices = get_next_ptr<int>(scores_size, workspace, workspace_size);
	auto indices_sorted = get_next_ptr<int>(scores_size, workspace, workspace_size);

	auto scores_sorted = get_next_ptr<float>(scores_size, workspace, workspace_size);


	for (int batch = 0; batch < batch_size; ++batch) {
		auto p = static_cast<const float*>(input) + batch * pred_size;
		auto in_socres = p + 5;

		auto out_scores = static_cast<float*>(output[0]) + batch * top_n;
		auto out_boxes = static_cast<float4*>(output[1]) + batch * top_n;
		auto out_classes = static_cast<float*>(output[2]) + batch * top_n;

		// 分离confidence soces 和 bboxes
		auto index_iter = thrust::cuda_cub::cub::CountingInputIterator<int>(0);
		thrust::transform(
			on_stream,
			index_iter,
			index_iter + pred_size,
			partition_flags,
			thrust::placeholders::_1 % pred_size > 4
		);

		float* in_scores = reinterpret_cast<float*>(scores);
		int* num_partition_selected = reinterpret_cast<int*>(indices);

		thrust::cuda_cub::cub::DeviceSelect::Flagged(
			workspace,
			workspace_size,
			p,
			partition_flags,
			in_scores,
			num_partition_selected,
			scores_size,
			stream
		);

		assert(*num_partition_selected == scores_size);

		thrust::transform(
			on_stream,
			index_iter,
			index_iter + pred_size,
			partition_flags,
			thrust::placeholders::_1 % pred_size < 5
		);

		float* in_boxes = reinterpret_cast<float*>(boxes);
		thrust::cuda_cub::cub::DeviceSelect::Flagged(
			workspace,
			workspace_size,
			p,
			partition_flags,
			in_boxes,
			num_partition_selected,
			boxes_size,
			stream
		);

		assert(*num_partition_selected == boxes_size);

		cudaStreamSynchronize(stream);

		// 使用阈值过滤scores
		thrust::transform(
			on_stream,
			in_scores,
			in_scores + scores_size,
			flags,
			thrust::placeholders::_1 > score_thresh
		);


		int* num_selected = reinterpret_cast<int*>(indices_sorted);
		thrust::cuda_cub::cub::DeviceSelect::Flagged(
			workspace,
			workspace_size,
			index_iter,
			flags,
			indices,
			num_selected,
			scores_size,
			stream
		);

		cudaStreamSynchronize(stream);

		int num_detections = *thrust::device_pointer_cast(num_selected);

		//
		auto indices_filtered = indices;
		if (num_detections > top_n) {
			thrust::gather(
				on_stream,
				indices,
				indices + num_detections,
				in_scores,
				scores
			);
			thrust::cuda_cub::cub::DeviceRadixSort::SortPairsDescending(
				workspace,
				workspace_size,
				scores,
				scores_sorted,
				indices,
				indices_sorted,
				num_detections,
				0,
				sizeof(*scores) * 8,
				stream
			);

			indices_filtered = indices_sorted;
			num_detections = top_n;
		}

		// 收集boxes

		thrust::transform(
			on_stream,
			indices_filtered,
			indices_filtered + num_detections,
			thrust::make_zip_iterator(
				thrust::make_tuple(out_scores, out_boxes, out_classes)
			),
			[=]__device__(int i) {
			int x = i % grid_size;
			int y = (i / grid_size) % grid_size;
			int a = (i / num_classes / grid_size / grid_size) % num_anchors;
			int cls = (i / grid_size / grid_size) % num_classes;

			const int num_girds = grid_size * grid_size;
			const int grid_idx = y * grid_size + x;

			const float pw = anchors_d[a * 2];
			const float ph = anchors_d[a * 2 + 1];

			float4 box = float4{
				in_boxes[grid_idx + num_girds * (a * 4 + 0)],
				in_boxes[grid_idx + num_girds * (a * 4 + 1)],
				in_boxes[grid_idx + num_girds * (a * 4 + 2)],
				in_boxes[grid_idx + num_girds * (a * 4 + 3)]
			};

			float bx = x + box.x;
			float by = y + box.y;
			float bw = pw * box.z;
			float bh = ph * box.w;

			box = float4{
				max(0.0f,  bx - bw / 2),
				max(0.0f, by - bh / 2),
				min(bx + bw / 2, stride * grid_size),
				min(by + bh / 2,  stride * grid_size)
			};

			return thrust::make_tuple(in_scores[i], box, cls);
		}
		);

		if (num_detections < top_n) {
			thrust::fill(on_stream, out_scores + num_detections,
				out_scores + top_n, 0.0f);
			thrust::fill(on_stream, out_classes + num_detections,
				out_classes + top_n, 0.0f);
		}
	}
	return 0;
}
