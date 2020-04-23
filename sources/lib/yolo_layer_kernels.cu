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
	if (!workspace || !workspace_size) {
		workspace_size = get_size_aligned<float>(anchors.size());  // anchors
		workspace_size += get_size_aligned<bool>(scores_size);   //flags
		workspace_size += get_size_aligned<int>(scores_size);   // indices
		workspace_size += get_size_aligned<int>(scores_size);   // indices_sorted
		workspace_size += get_size_aligned<float>(scores_size);  //socres
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

	auto flags = get_next_ptr<bool>(scores_size, workspace, workspace_size);
	auto indices = get_next_ptr<int>(scores_size, workspace, workspace_size);
	auto indices_sorted = get_next_ptr<int>(scores_size, workspace, workspace_size);
	auto scores = get_next_ptr<float>(scores_size, workspace, workspace_size);
	auto scores = get_next_ptr<float>(scores_size, workspace, workspace_size);

	for (int batch = 0; batch < batch_size; ++batch) {
		auto p = static_cast<const float*>(input) + batch * (grid_size * grid_size * (num_anchors * (5 + num_classes)));
		auto in_socres = static_cast<const float*>(input) + batch * scores_size;
	}
	return cudaError_t();
}
