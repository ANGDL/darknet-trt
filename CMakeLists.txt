cmake_minimum_required(VERSION 3.16)

project(darknet_trt LANGUAGES CUDA CXX)

set(CMAKE_CUDA_EXTENSIONS 17)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_CXX_COMPILER, /usr/local/cuda/bin/nvcc)

set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CMAKE_CXX_COMPILER}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set(OPENCV_PATH "/mnt/d/libs/opencv-4.4.0/bin")
set(TRT_PATH "/home/x/TensorRT-7.1.3.4")

include_directories(
        ./sources/lib
        ${OPENCV_PATH}/include/opencv4
        ${CUDA_PATH}/include
        ${TRT_PATH}/include
)

find_library(OPENCV_LIB
        NAMES opencv_core
        PATHS ${OPENCV_PATH}/lib64)

add_library(
        yolov3_nms
        ./sources/lib/darknet_cfg.cpp
        ./sources/lib/darknet_utils.cpp
        ./sources/lib/nms_kernel.cu
        ./sources/lib/decode_kernel.cu
        ./sources/lib/yolo_layer_kernels.cu
        ./sources/lib/yolo.cpp
        ./sources/lib/plugin_factory.cpp
        ./sources/lib/yolov3-nms.cpp
        ./sources/lib/yolov3.cpp
        ./sources/lib/yolo.cpp
)


set_target_properties(
        yolov3_nms
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

add_executable(
        test_yolo
        sources/apps/test.cpp
)

target_link_libraries(
        yolov3_nms PRIVATE
        ${TRT_PATH}/lib/libnvinfer.so
        ${TRT_PATH}/lib/libnvinfer_plugin.so
        ${TRT_PATH}/lib/libnvparsers.so
)

target_link_libraries(
        test_yolo PRIVATE
        stdc++fs
        ${OPENCV_PATH}/lib/libopencv_core.so
        ${OPENCV_PATH}/lib/libopencv_imgproc.so
        ${OPENCV_PATH}/lib/libopencv_highgui.so
        ${OPENCV_PATH}/lib/libopencv_imgcodecs.so
        ${OPENCV_PATH}/lib/libopencv_dnn.so
        yolov3_nms
)