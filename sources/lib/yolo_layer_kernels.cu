#include <assert.h>
#include "yolo_layer_kernels.cuh"

#include <stdexcept>
#include <algorithm>
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


#define CUDA_ALIGN 256

template <typename T>
inline size_t get_size_aligned(size_t num_elem) {
	size_t size = num_elem * sizeof(T);
	size_t extra_align = 0;
	if (size % CUDA_ALIGN != 0) {
		extra_align = CUDA_ALIGN - size % CUDA_ALIGN;
	}
	return size + extra_align;
}

template <typename T>
inline T* get_next_ptr(size_t num_elem, void*& workspace, size_t& workspace_size) {
	size_t size = get_size_aligned<T>(num_elem);
	if (size > workspace_size) {
		throw std::runtime_error("Workspace is too small!");
	}
	workspace_size -= size;
	T* ptr = reinterpret_cast<T*>(workspace);
	workspace = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(workspace) + size);
	return ptr;
}

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

__device__
inline float d_sigmod(const float& x) {
	return 1.0f / (1.0f + __expf(-x));
}

__global__
void kernel_yolo_layer(
	const float* input,
	float* output,
	const unsigned int grid_size,
	const unsigned int num_classes,
	const unsigned int num_boxes
) {
	unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;;
	unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;

	if ((id_x >= grid_size) || (id_y >= grid_size) || (id_z >= num_boxes)) {
		return;
	}

	unsigned int num_grids = grid_size * grid_size;
	unsigned int grid_idx = id_x + grid_size * id_y;

	output[grid_idx + num_grids * (id_z * (num_classes + 5) + 0)] = d_sigmod(input[grid_idx + num_grids * (id_z * (num_classes + 5) + 0)]);  // sigmod(tx)

	output[grid_idx + num_grids * (id_z * (num_classes + 5) + 1)] = d_sigmod(input[grid_idx + num_grids * (id_z * (num_classes + 5) + 1)]); // sigmod(ty)

	output[grid_idx + num_grids * (id_z * (num_classes + 5) + 2)] = __expf(input[grid_idx + num_grids * (id_z * (num_classes + 5) + 2)]);  // exp(tw)

	output[grid_idx + num_grids * (id_z * (num_classes + 5) + 3)] = __expf(input[grid_idx + num_grids * (id_z * (num_classes + 5) + 3)]);  // exp(th) 

	output[grid_idx + num_grids * (id_z * (num_classes + 5) + 4)] = d_sigmod(input[grid_idx + num_grids * (id_z * (num_classes + 5) + 4)]);  // sigmod(to)

	for (unsigned int i = 0; i < num_classes; ++i)
	{
		output[grid_idx + num_grids * (id_z * (5 + num_classes) + (5 + i))]
			= d_sigmod(input[grid_idx + num_grids * (id_z * (5 + num_classes) + (5 + i))]);
	}
}

cudaError_t cuda_yolo_layer(
	const void* input,
	void* output,
	int batch_size,
	unsigned int grid_size,
	unsigned int num_classes,
	unsigned int num_boxes,
	unsigned int output_size,
	cudaStream_t stream
) {
	assert(num_boxes == 3);
	dim3 threads_per_blocks(16, 16, 4);
	dim3 num_blocks(
		grid_size / threads_per_blocks.x + 1,
		grid_size / threads_per_blocks.y + 1,
		num_boxes / threads_per_blocks.z + 1);

	const float* input_f = reinterpret_cast<const float*>(input);
	float* output_f = reinterpret_cast<float*>(output);

	for (int i = 0; i < batch_size; ++i) {
		kernel_yolo_layer << <num_blocks, threads_per_blocks, 0, stream >> > (
			input_f + (i * output_size), output_f + (i * output_size), grid_size, num_classes, num_boxes);
	}

	return cudaGetLastError();
}


// kernel_upsample
// reference: https://github.com/pjreddie/darknet/blob/master/src/blas_kernels.cu
__global__ void kernel_upsample(size_t N, float* x, int w, int h, int c, int batch, int stride, float scale, float* out)
{
	size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (i >= N) return;
	int out_index = i;
	int out_w = i % (w * stride);
	i = i / (w * stride);
	int out_h = i % (h * stride);
	i = i / (h * stride);
	int out_c = i % c;
	i = i / c;
	int b = i % batch;

	int in_w = out_w / stride;
	int in_h = out_h / stride;
	int in_c = out_c;

	int in_index = b * w * h * c + in_c * w * h + in_h * w + in_w;

	out[out_index] += scale * x[in_index];
}


cudaError_t cuda_upsample_layer(const void* input, void* output, int batch_size, float stride,
	int c, int h, int w, cudaStream_t stream)
{
	unsigned int size = w * h * c * batch_size * stride * stride;
	kernel_upsample << <cuda_gridsize(size), KERNEL_BLOCK, 0, stream >> > (size, (float*)input, w, h, c, batch_size, stride, 1.0, (float*)output);
	return cudaGetLastError();
}

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

		workspace_size += std::max(temp_size_flag, temp_size_sort);
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

		assert(num_partition_selected == scores_size);

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

		assert(num_partition_selected == boxes_size);

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

			float pred_x = x + box.x;
			float pred_y = y + box.y;
			float pred_w = pw * box.z;
			float pred_h = ph * box.w;

			box = float4{
				max(0.0f, pred_x * stride),
				max(0.0f, pred_y * stride),
				max(0.0f, pred_w),
				max(0.0f, pred_h)
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
