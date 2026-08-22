#include <cstdio>
#include <cstdlib>

__global__ void divergentKernel(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = data[i];
        if (threadIdx.x % 2 == 0) {
            for (int j = 0; j < 500; j++) {
              x = x * 1.0000001f;
            }
        } else {
            for (int j = 0; j < 500; j++) {
              x = x + 1.0000001f;
            }
        }
        data[i] = x;
    }
}

__global__ void nonDivergentKernel(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = data[i];
        if (blockIdx.x % 2 == 0) {
            for (int j = 0; j < 500; j++) {
              x = x * 1.0000001f;
            }
        } else {
            for (int j = 0; j < 500; j++) {
              x = x + 1.0000001f;
            }
        }
        data[i] = x;
    }
}

int main() {
  int N = 10000000;

  int threadsPerBlock = 256;
  int numBlocks = (N + (threadsPerBlock-1) )/ threadsPerBlock;

  size_t size = N * sizeof(float);

  float *h_A, *h_C;
  
  h_A = (float*)malloc(size);
  h_C = (float*)malloc(size);

  float value = 1.0f;
  for (int i = 0; i < N; i++) {
    h_A[i] = value + i;
    h_C[i] = value + i;
  }

  float *d_A, *d_C;
  
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

  divergentKernel<<<numBlocks, threadsPerBlock>>>(d_A, N);
  nonDivergentKernel<<<numBlocks, threadsPerBlock>>>(d_C, N);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  printf("h_A[0] = %f \n", h_A[0]);
  printf("h_A[N-1] = %f \n", h_A[N-1]);
  printf("h_C[0] = %f \n", h_C[0]);
  printf("h_C[N-1] = %f \n", h_C[N-1]);

  cudaFree(d_A); cudaFree(d_C);
  free(h_A); free(h_C);
  return 0;
}

