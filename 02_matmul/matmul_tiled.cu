#include <cstdio>
#include <cstdlib>

#define TILE_WIDTH 16

__global__ void matMulTiled(const float *A, const float *B, float *C,
            int M, int K, int N) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    // global idx - related to C
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K  + TILE_WIDTH-1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
      int kA = threadIdx.x + t * TILE_WIDTH;
      if (row < M && kA < K)
        tileA[threadIdx.y][threadIdx.x] = A[row * K + kA];
      else
        tileA[threadIdx.y][threadIdx.x] = 0.0f;

      int kB = threadIdx.y + t * TILE_WIDTH;

      if (kB  < K && col < N)
        tileB[threadIdx.y][threadIdx.x] = B[kB * N + col];
      else
        tileB[threadIdx.y][threadIdx.x] = 0.0f;

      __syncthreads();

      for (int kk = 0; kk < TILE_WIDTH; kk++){

           sum += tileA[threadIdx.y][kk] * tileB[kk][threadIdx.x];
       }
      __syncthreads();

    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }

}

int main() {
  int M = 1024; // 20
  int K = 1024; // 20
  int N = 1024; // 4

  size_t sizeA = M * K * sizeof(float);
  size_t sizeB = K * N * sizeof(float);
  size_t sizeC = M * N * sizeof(float);

  float *h_A, *h_B, *h_C;
  
  h_A = (float*)malloc(sizeA);
  h_B = (float*)malloc(sizeB);
  h_C = (float*)malloc(sizeC);

  float valA = 1.0f;
  float valB = valA + 6.0f;
  for (int i = 0; i < M * K; i++) {
    h_A[i] = valA + i;
  }
  for (int i = 0; i < K * N; i++) {
    h_B[i] = valB + i;
  }

  float *d_A, *d_B, *d_C;
  
  cudaMalloc((void**)&d_A, sizeA);
  cudaMalloc((void**)&d_B, sizeB);
  cudaMalloc((void**)&d_C, sizeC);

  cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH,
                      (M + TILE_WIDTH - 1) / TILE_WIDTH);
  matMulTiled<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

  printf("h_C[0] = %f \n", h_C[0]);
  printf("h_C[M*N-1] = %f \n", h_C[M*N-1]);

  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  free(h_A); free(h_B); free(h_C);
  return 0;
}
