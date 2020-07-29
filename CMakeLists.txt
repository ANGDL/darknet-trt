cmake_minimum_required(VERSION 3.16)

project(darknet_trt LANGUAGES CUDA CXX)

set(CMAKE_CUDA_EXTENSIONS 17)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_CXX_COMPILER, /usr/local/cuda/bin/nvcc)

set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CMAKE_CXX_COMPILER}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-write-strings")

set(OPENCV_PATH "/mnt/d/libs/opencv-4.4.0/bin")
set(TRT_PATH "/home/x/TensorRT-7.0.0.11")
set(CUDA_PATH "/usr/local/cuda")

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
        STATIC
        ./sources/lib/calibrator.cpp
        ./sources/lib/darknet_cfg.cpp
        ./sources/lib/darknet_utils.cpp
        ./sources/lib/nms_kernel.cu
        ./sources/lib/decode_kernel.cu
        ./sources/lib/yolo_layer_kernels.cu
        ./sources/lib/yolo.cpp
        ./sources/lib/yolov3-nms.cpp
        ./sources/lib/stb_image.h
        ./sources/lib/stb_image_resize.h
)

set_target_properties(
        yolov3_nms
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        OUTPUT_NAME "yolov3_nms"
)

SET_TARGET_PROPERTIES(yolov3_nms PROPERTIES CLEAN_DIRECT_OUTPUT 1)


add_executable(
        test_yolo
        sources/apps/test.cpp
)

add_executable(
        create_engine
        sources/apps/create_engine.cpp
)

target_link_libraries(
        yolov3_nms PRIVATE
        ${TRT_PATH}/lib/libnvinfer.so
        ${TRT_PATH}/lib/libnvinfer_plugin.so
        ${TRT_PATH}/lib/libnvparsers.so
)

target_link_libraries(
        test_yolo PUBLIC
        yolov3_nms
        ${OPENCV_PATH}/lib/libopencv_world.so
)

target_link_libraries(
        create_engine PUBLIC
        yolov3_nms
)
