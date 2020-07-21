#ifndef _NMS_KERNEL_H_
#define _NMS_KERNEL_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

int cuda_nms(int batchSize,
             const void *const *inputs, void **outputs,
             size_t count, int detections_per_im, float nms_thresh,
             void *workspace, size_t workspace_size, cudaStream_t stream);


#endif
