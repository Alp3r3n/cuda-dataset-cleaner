cmake_minimum_required(VERSION 3.24)
project(CudaDatasetCleaner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# Kernel object library compiled by nvcc
add_library(kernels OBJECT
    kernels/grayscale.cu
    kernels/brightness.cu
    kernels/contrast.cu
    kernels/sobel.cu
    kernels/saturation.cu)
target_include_directories(kernels PRIVATE include)

# Main executable
add_executable(cuda-dataset-cleaner
    src/main.cpp
    src/image_loader.cpp
    src/quality_metrics_cpu.cpp
    src/quality_metrics_cuda.cpp
    src/cuda_buffer.cpp
    src/report.cpp
    src/benchmark.cpp
    $<TARGET_OBJECTS:kernels>)

target_include_directories(cuda-dataset-cleaner
    PRIVATE include ${OpenCV_INCLUDE_DIRS})

target_link_libraries(cuda-dataset-cleaner
    PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart)

# quality_metrics_cuda.cpp uses cuda_runtime.h — no <<<>>> syntax, regular C++ compile is fine
