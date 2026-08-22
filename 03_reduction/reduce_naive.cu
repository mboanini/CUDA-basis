#include <cstdio>
#include <cstdlib>

__global__ void reduceNaive(float * input, float * output, int n) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2*stride) == 0) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

int main() {
  int N = 16777216;

  int threadsPerBlock = 256;
  int numBlocks = (N + (threadsPerBlock-1) )/ threadsPerBlock;

  size_t sizeA = N * sizeof(float);
  size_t sizeC = numBlocks * sizeof(float);

  float *h_A, *h_C;

  h_A = (float*)malloc(sizeA);
  h_C = (float*)malloc(sizeC);

  float valA = 1.0f;
  for (int i = 0; i < N; i++) {
    h_A[i] = valA + i;
  }

  float *d_A, *d_C;

  cudaMalloc((void**)&d_A, sizeA);
  cudaMalloc((void**)&d_C, sizeC);

  cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);

  reduceNaive<<<numBlocks, threadsPerBlock>>>(d_A, d_C, N);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

  printf("h_C[0] = %f \n", h_C[0]);
  printf("h_C[M*N-1] = %f \n", h_C[numBlocks-1]);

  cudaFree(d_A); cudaFree(d_C);
  free(h_A); free(h_C);
  return 0;
}
