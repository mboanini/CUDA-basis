# CUDA-basis

CUDA kernels written, tested and profiled. Each pair isolates a single GPU optimization concept and includes profiling numbers collected 
with Nsight Compute (`ncu`).

**Environment**
- Google Colab, T4 GPU (40 SMs, Compute Capability 7.5)
- Compile: `nvcc -o <exe> <file>.cu`
- Profile: `ncu ./<exe>`

**Profiling results**

**Reproducibility**
```
nvcc -o matmul_naive 02_matmul/matmul_naive.cu
nvcc -o matmul_tiled 02_matmul/matmul_tiled.cu
ncu ./matmul_naive
ncu ./matmul_tiled

nvcc -o reduce_naive 03_reduction/reduce_naive.cu
nvcc -o reduce_optimized 03_reduction/reduce_optimized.cu
ncu ./reduce_naive
ncu ./reduce_optimized

nvcc -o divergence 04_warp_divergence/divergence.cu
ncu ./divergence
```

