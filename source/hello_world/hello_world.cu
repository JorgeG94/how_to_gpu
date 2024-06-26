#include <stdio.h>
#include <cuda_runtime.h>
// Kernel function to print "Hello, World!" from the GPU
__global__ void helloWorldFromGPU() {
  printf("Hello, World! from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {

  // Launch the kernel with 2 block of 5 threads
  helloWorldFromGPU<<<2, 5>>>();
  
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
    
  return 0;
}

