cmake_minimum_required(VERSION 3.10)

# Set the project name and enable CUDA
project(MatMulGPU LANGUAGES CUDA CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)


# Set the minimum required C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find CUDA and cuBLAS
find_package(CUDAToolkit REQUIRED)

# Specify the architecture of your NVIDIA GPU (for example, Compute Capability 8.9 for an NVIDIA 4080)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Automatically include all .cu files in the current directory as source files
file(GLOB CUDA_SOURCES *.cu)

# Specify the header files
file(GLOB HEADERS *.h)

# Create an executable named matmul_gpu
add_executable(matmul_gpu ${CUDA_SOURCES} ${HEADERS})

# Set the compiler flags for CUDA
set_target_properties(matmul_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

# Link the cuBLAS library
target_link_libraries(matmul_gpu PRIVATE CUDA::cublas)

# Optionally set compiler flags for performance optimizations
target_compile_options(matmul_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-extended-lambda
    --use_fast_math
>)

# Print some information about the CUDA environment
message(STATUS "CUDA toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
