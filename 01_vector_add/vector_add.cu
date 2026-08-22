#include <cstdio>
#include <cstdlib>

__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
      C[i] = A[i] + B[i];
    }
}

int main() {
  int N = 1000000;
  size_t size = N * sizeof(float);
  float *h_A, *h_B, *h_C;
  
  h_A = (float*)malloc(size);
  h_B = (float*)malloc(size);
  h_C = (float*)malloc(size);
  for (int i = 0; i < N; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }
  float *d_A, *d_B, *d_C;
  
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int numBlocks = (N + (threadsPerBlock-1) )/ threadsPerBlock;
  vectorAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  printf("h_C[0] = %f (atteso 3.0)\n", h_C[0]);
  printf("h_C[N-1] = %f (atteso 3.0)\n", h_C[N-1]);

  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  free(h_A); free(h_B); free(h_C);
  return 0;
}
