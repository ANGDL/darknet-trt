#include "upsample_kernel.cuh"


#define KERNEL_BLOCK 512

// cuda_gridsize 
// reference: https://github.com/pjreddie/darknet/blob/master/src/cuda.h
static
dim3 cuda_gridsize(unsigned int n) {
	unsigned int k = (n - 1) / KERNEL_BLOCK + 1;
	unsigned int x = k;
	unsigned int y = 1;
	if (x > 65535) {
		x = static_cast<unsigned int>(ceil(sqrt(k)));
		y = (n - 1) / (x * KERNEL_BLOCK) + 1;
	}
	dim3 d = { x, y, 1 };
	//printf("%ld %ld %ld %ld\n", n, x, y, x*y*KERNEL_BLOCK);
	return d;
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
