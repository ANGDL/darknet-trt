#ifndef _YOLO_LAYER_KERNELS_CUH_
#define _YOLO_LAYER_KERNELS_CUH_

#include "NvInfer.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>

#define KERNEL_BLOCK 512

// cuda_gridsize 
// reference: https://github.com/pjreddie/darknet/blob/master/src/cuda.h
static
dim3 cuda_gridsize(unsigned int n) {
    unsigned int k = (n - 1) / KERNEL_BLOCK + 1;
    unsigned int x = k;
    unsigned int y = 1;
    if (x > 65535) {
        x = static_cast<unsigned int>(ceil(sqrt((float) k)));
        y = (n - 1) / (x * KERNEL_BLOCK) + 1;
    }
    dim3 d = {x, y, 1};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*KERNEL_BLOCK);
    return d;
}


cudaError_t cuda_yolo_layer(
        const void *input,
        void *output,
        int batch_size,
        unsigned int grid_size,
        unsigned int num_classes,
        unsigned int num_boxes,
        unsigned int output_size,
        cudaStream_t stream
);

cudaError_t cuda_upsample_layer(
        const void *input,
        void *output,
        int batch_size,
        float stride,
        int c, int h, int w,
        cudaStream_t stream);

#endif
