#ifndef _DECODE_KERNEL_H_
#define _DECODE_KERNEL_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

int cuda_decode_layer(
	const void* const* inputs,
	void** output,
	int batch_size,
	float stride,
	size_t grid_size,
	size_t num_anchors,
	size_t num_classes,
	const std::vector<float>& anchors,
	float score_thresh,
	void* workspace,
	size_t workspace_size,
	cudaStream_t stream
);

#endif 
