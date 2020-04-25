#ifndef  _UP_SAMPLE_PLUGIN_H_
#define _UP_SAMPLE_PLUGIN_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


cudaError_t cuda_upsample_layer(
	const void* input,
	void* output,
	int batch_size,
	float stride,
	int c, int h, int w,
	cudaStream_t stream);


#endif
