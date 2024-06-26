#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addOne(int* d_array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] += 1;
    }
}

int main() {
    const int arraySize = 10;
    const int arrayBytes = arraySize * sizeof(int);

    // Allocate memory on the host (CPU)
    int h_array[arraySize] = {0};

    // Allocate memory on the device (GPU)
    int* d_array;
    cudaMalloc((void**)&d_array, arrayBytes);

    // Transfer the array from the host to the device
    cudaMemcpy(d_array, h_array, arrayBytes, cudaMemcpyHostToDevice);

    // Launch the kernel with 1 block of 10 threads
    addOne<<<1, arraySize>>>(d_array, arraySize);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Transfer the array from the device back to the host
    cudaMemcpy(h_array, d_array, arrayBytes, cudaMemcpyDeviceToHost);

    // Print the resulting array on the host
    printf("Resulting array:\n");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    // Free the memory allocated on the device
    cudaFree(d_array);

    return 0;
}

