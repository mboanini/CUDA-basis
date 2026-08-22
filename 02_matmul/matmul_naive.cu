#include <cstdio>
#include <cstdlib>

// C = A x B, where A is M x K, B is K x N => C is M x N
// C[row][col] = Σ (per k da 0 a K-1) A[row][k] * B[k][col]

__global__ void matMulNaive(const float *A, const float *B, float *C,
                      int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
  int M = 1024; // 20
  int K = 1024; // 3
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

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((N + 15) / 16, (M + 15) / 16);
  matMulNaive<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);

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
