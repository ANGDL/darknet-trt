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
