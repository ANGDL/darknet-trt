#include <assert.h>
#include "yolo_layer_kernels.cuh"


__device__
float d_sigmod(float x) {
	return 1.0f / (1.0 + __expf(x));
}

__global__
void kernel_yolo_layer(
	const float* input,
	float* output,
	unsigned int grid_size,
	unsigned int num_classes,
	unsigned int num_boxes,
	unsigned int output_size
) {
	unsigned int id_x = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int id_y = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int id_z = threadIdx.z + blockDim.z * blockIdx.z;

	if (id_x >= grid_size || id_y >= grid_size || id_z >= num_boxes) {
		return;
	}

	unsigned int num_grids = grid_size * grid_size;
	unsigned int grid_idx = id_x + grid_size * id_y;

	unsigned int loc_idx = grid_idx + num_grids * (id_z * (num_classes + 5));
	output[loc_idx + 0] = d_sigmod(input[loc_idx + 0]);  // sigmod(tx)
	output[loc_idx + 1] = d_sigmod(input[loc_idx + 1]);  // sigmod(ty)
	output[loc_idx + 2] = __expf(input[loc_idx + 2]);  // exp(tw)
	output[loc_idx + 3] = __expf(input[loc_idx + 3]);  // exp(th)
	output[loc_idx + 4] = d_sigmod(input[loc_idx + 4]);  // sigmod(to)

	for (cuuint32_t i = 0; i < num_boxes; ++i) {
		output[loc_idx + 5 + i] = d_sigmod(input[loc_idx + 5 + i]);  // confidence score
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
	dim3 num_blocks(grid_size / threads_per_blocks.x, grid_size / threads_per_blocks.y, 1);

	const float* input_f = reinterpret_cast<const float*>(input);
	float* output_f = reinterpret_cast<float*>(output);

	for (int i = 0; i < batch_size; ++i) {
		kernel_yolo_layer << <num_blocks, threads_per_blocks >> > (
			input_f, output_f, grid_size, num_classes, num_boxes, output_size);
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
	kernel_upsample << <cuda_gridsize(size), KERNEL_BLOCK >> > (size, (float*)input, w, h, c, batch_size, stride, 1.0, (float*)output);
	return cudaGetLastError();
}
