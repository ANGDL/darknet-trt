#ifndef _YOLO_LAYER_KERNELS_CUH_
#define _YOLO_LAYER_KERNELS_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


cudaError_t cuda_yolo_layer(
	const void* input,
	void* output,
	int batch_size,
	unsigned int grid_size,
	unsigned int num_classes,
	unsigned int num_boxes,
	unsigned int output_size,
	cudaStream_t stream
);

#endif
