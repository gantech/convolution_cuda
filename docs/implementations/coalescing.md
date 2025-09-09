# Global Memory Coalescing

After our naive implementation, the first optimization we apply is global memory coalescing. This is a fundamental CUDA optimization technique that dramatically improves memory throughput.

## The Problem with Naive Memory Access

In the naive implementation, adjacent threads might access memory locations that are far apart, particularly when traversing rows of a 2D matrix. This leads to multiple memory transactions for a single warp, severely limiting performance.

## Coalesced Memory Access

Memory coalescing means organizing memory accesses so that consecutive threads access consecutive memory addresses. This allows the GPU to combine multiple memory accesses into a single transaction.

## Implementation

```cuda
template <const int BLOCKSIZE>
__global__ void conv2d_global_mem_coalesce(int M, int N, 
                                         const double *A, double *B) {
  // Calculate global thread position
  const int row = blockIdx.y * BLOCKSIZE + threadIdx.y;
  const int col = blockIdx.x * BLOCKSIZE + threadIdx.x;

  // Ensure we don't go out of bounds
  if (row < M && col < N) {
    // Apply filter
    double filter[9] = {-1.0, -1.0, -1.0,
                        -1.0, 8.0, -1.0,
                        -1.0, -1.0, -1.0};
                        
    double tmp = 0.0;
    for (int fi = -1; fi < 2; fi++) {
      for (int fj = -1; fj < 2; fj++) {
        int r = row + fi;
        int c = col + fj;
        
        // Check boundaries and compute
        if (r >= 0 && r < M && c >= 0 && c < N) {
          tmp += A[r * N + c] * filter[(fi + 1) * 3 + (fj + 1)];
        }
      }
    }
    
    // Write result to output
    B[row * N + col] = tmp;
  }
}
```

## Key Differences from Naive Implementation

1. **Thread Block Organization**: We use a 2D thread block that matches the 2D structure of the data
2. **Thread Indexing**: Consecutive threads access consecutive memory locations
3. **Pre-computed Filter**: The filter values are pre-computed rather than calculated on-the-fly
4. **Templated Block Size**: Block size can be tuned at compile time for different GPUs

## Memory Access Pattern Visualization

<div class="chart-container">
    <img src="../assets/coalesced_vs_uncoalesced.png" alt="Coalesced vs Uncoalesced Memory Access">
</div>

## Performance Improvement

This optimization yields a **X% improvement** over the naive implementation, achieving **Y GFLOP/s** on an NVIDIA H100 GPU.

## Limitations

While memory coalescing significantly improves performance, we still face several limitations:

1. **Redundant Memory Access**: Each element is still read multiple times by different threads
2. **Global Memory Latency**: All memory accesses still target high-latency global memory
3. **Control Flow Divergence**: Boundary checks still cause threads to diverge

## Next Steps

To address these limitations, our next optimization will use [Shared Memory Blocking](shared_mem.md) to reduce redundant global memory accesses and take advantage of the much faster shared memory.
