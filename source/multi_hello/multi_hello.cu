// multi_gpu_hello_world.cu
// A simple CUDA program to demonstrate multi-GPU "Hello, World!" using MPI.

#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Kernel function to print "Hello, World!" from each GPU
__global__ void helloWorldFromGPU(int gpu_id, int rank) {
    printf("Hello, World! from GPU %d, process rank %d, thread %d\n", gpu_id, rank, threadIdx.x);
}

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the number of GPUs
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);

    // Assign a GPU to each MPI process
    int gpu_id = world_rank % num_gpus;
    cudaSetDevice(gpu_id);

    // Print information from each GPU
    helloWorldFromGPU<<<1, 10>>>(gpu_id, world_rank);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}

