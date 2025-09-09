# Naive Implementation

The journey begins with the simplest possible implementation of a 2D convolution in CUDA.

## The Problem

A 2D convolution applies a filter (or kernel) to an input matrix to produce an output matrix. For each output element, we compute a weighted sum of the input element and its neighbors, with weights defined by the filter.

For our implementations, we use a 3×3 Laplacian filter:

```
-1 -1 -1
-1  8 -1
-1 -1 -1
```

## Naive CUDA Implementation

The naive implementation assigns one thread to each output element:

```cuda
__global__ void conv2d_naive(int M, int N, const double *A, double *B) {
  // Calculate the row and column index
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Check if within bounds
  if (row < M && col < N) {
    double tmp = 0.0;
    
    // Apply the convolution filter
    for (int fi = -1; fi < 2; fi++) {
      for (int fj = -1; fj < 2; fj++) {
        int r = row + fi;
        int c = col + fj;
        
        // Check boundaries
        if (r >= 0 && r < M && c >= 0 && c < N) {
          double filter_value = -1.0;  // Default filter value
          if (fi == 0 && fj == 0) {
            filter_value = 8.0;  // Center of the filter
          }
          tmp += A[r * N + c] * filter_value;
        }
      }
    }
    
    // Write result to output
    B[row * N + col] = tmp;
  }
}
```

## Performance Analysis

### Memory Access Pattern

This implementation suffers from several inefficiencies:

1. **Uncoalesced Memory Access**: Threads in the same warp access non-consecutive memory locations when reading along rows
2. **Redundant Memory Access**: Each element from the input matrix is read multiple times by different threads
3. **Control Flow Divergence**: The boundary checks cause threads within the same warp to take different execution paths
4. **Global Memory Latency**: Every memory access goes directly to global memory with high latency

### Occupancy Issues

The naive implementation also suffers from low occupancy:

1. The kernel uses too many registers per thread
2. No optimization for memory bandwidth utilization
3. Each thread does a small amount of work, making the overhead of thread creation significant

## Benchmarks

On an NVIDIA H100 GPU, this implementation achieves only **X GFLOP/s** for a 4096×4096 matrix, which is less than X% of the theoretical peak performance.

## Roofline Analysis

<div class="chart-container">
    <img src="../assets/roofline_naive.png" alt="Roofline Analysis of Naive Implementation">
</div>

Our naive implementation is heavily memory-bound, limited by the global memory bandwidth.

## Next Steps

The obvious first step for improvement is to address the uncoalesced memory accesses. In the next section, we'll look at [Global Memory Coalescing](coalescing.md) to significantly improve our memory bandwidth utilization.
