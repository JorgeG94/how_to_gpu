cmake_minimum_required(VERSION 3.21)

# Set the project name and version
project(CudaExamples VERSION 1.0 LANGUAGES C CXX CUDA)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

find_package(MPI REQUIRED)
# Set the source directory
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/source)
set(executables ${SOURCE_DIR}/data_transfer_op/data_transfer_op.cu 
                ${SOURCE_DIR}/hello_world/hello_world.cu
                ${SOURCE_DIR}/multi_hello/multi_hello.cu
                ${SOURCE_DIR}/no_streams/no_streams.cu
                ${SOURCE_DIR}/with_streams/with_streams.cu
                )

# Add the executable for data_transfer_op
add_executable(data_transfer_op ${SOURCE_DIR}/data_transfer_op/data_transfer_op.cu)
set_target_properties(data_transfer_op PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add the executable for hello_world
add_executable(hello_world ${SOURCE_DIR}/hello_world/hello_world.cu)
set_target_properties(hello_world PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(no_stream ${SOURCE_DIR}/no_streams/no_streams.cu)
set_target_properties(no_stream PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_executable(with_stream ${SOURCE_DIR}/with_streams/with_streams.cu)
set_target_properties(with_stream PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_executable(multi_gpu_hello_world ${SOURCE_DIR}/multi_hello/multi_hello.cu)
set_target_properties(multi_gpu_hello_world PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(multi_gpu_hello_world PUBLIC MPI::MPI_CXX)
