# CUDA-basis

CUDA kernels written, tested and profiled. Each pair isolates a single GPU optimization concept and includes profiling numbers collected 
with Nsight Compute (`ncu`).

**Environment**
- Google Colab, T4 GPU (40 SMs, Compute Capability 7.5)
- Compile: `nvcc -o <exe> <file>.cu`
- Profile: `ncu ./<exe>`

**Profiling results**
### 1. Matrix Multiplication — Naive vs. Tiled (shared memory)

| Metric | Naive | Tiled | Change |
|---|---|---|---|
| Duration | 9.19 ms | 5.79 ms | **1.59x faster** |
| Total DRAM Elapsed Cycles | 367.2M | 231.6M | -37% (less redundant global memory traffic) |
| Memory Throughput | 62.52% | 74.45% | higher relative utilization over a shorter run |
| Registers/thread | 52 | 39 | fewer, due to tile reuse replacing recomputation |
| Achieved Occupancy | 98.72% | 98.68% | unchanged — the gain isn't from occupancy |

### 2. Parallel Reduction — Naive vs. Optimized addressing

| Metric | Naive | Optimized | Change |
|---|---|---|---|
| Duration | 2.62 ms | 1.60 ms | **1.64x faster** |
| Compute (SM) Throughput | 63.27% | 76.28% | more useful compute per cycle |
| Achieved Occupancy | 94.11% | 90.42% | slightly lower, not the source of the gain |

The naive kernel uses interleaved addressing (`if (tid % (2*stride) == 0)`), which scatters active threads non-contiguously within a warp in early iterations, causing warp divergence. The optimized kernel uses sequential addressing (`if (tid < stride)`, stride halving from `blockDim.x/2`), keeping active threads contiguous and warps either fully active or fully idle — eliminating that divergence.

### 3. Warp Divergence — isolated comparison

| Metric | Divergent (thread-level branch) | Non-divergent (block-level branch) | Change |
|---|---|---|---|
| Duration | 6.77 ms | 3.54 ms | **1.91x faster (non-divergent)** |
| Compute (SM) Throughput | 98.94% | 94.75% | divergent is *higher*, not lower |

Both kernels do the same total arithmetic (500 iterations/thread, multiply vs. add). The divergent version branches on `threadIdx.x % 2`, splitting every warp roughly in half; the non-divergent version branches on `blockIdx.x % 2`, so each warp takes only one path. Divergence forces the warp scheduler to execute both branches serially per warp, masking off the inactive half each time — that's why the divergent kernel shows *higher* compute throughput (it's issuing more total work) while still running slower.

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

