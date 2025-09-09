# Shared Memory Blocking

After implementing global memory coalescing, the next major optimization is to use shared memory to reduce redundant global memory accesses.

## Why Shared Memory?

In our previous implementations, each input element is read multiple times from global memory by different threads. This is inefficient because:

1. Global memory has high latency (hundreds of clock cycles)
2. The same data is fetched repeatedly, wasting memory bandwidth
3. There's no reuse of previously fetched data

Shared memory provides a much faster on-chip scratchpad that can be shared among threads in the same block.

## Implementation

```cuda
template <const int BLOCKSIZE>
__global__ void conv2d_shared_mem_block(int M, int N, 
                                       const double *A, double *B) {
  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  double filter[9] = {-1.0, -1.0, -1.0,
          -1.0, 8.0, -1.0,
          -1.0, -1.0, -1.0};

  // allocate buffer for current block including padding in fast shared mem
  __shared__ double As[(BLOCKSIZE + 2) * (BLOCKSIZE + 2)];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // advance pointers to the starting positions
  B += (cRow * BLOCKSIZE) * N + cCol * BLOCKSIZE;

  // Each block loads (BLOCKSIZE+2) x (BLOCKSIZE+2) elements into shared memory
  for (int i = threadIdx.x; i < (BLOCKSIZE+2)*(BLOCKSIZE+2); i += blockDim.x) {
    int smem_row = i / (BLOCKSIZE+2);
    int smem_col = i % (BLOCKSIZE+2);
    int g_row = cRow * BLOCKSIZE + smem_row - 1;
    int g_col = cCol * BLOCKSIZE + smem_col - 1;
    if (g_row >= 0 && g_row < M && g_col >= 0 && g_col < N)
        As[smem_row * (BLOCKSIZE+2) + smem_col] = A[g_row * N + g_col];
    else
        As[smem_row * (BLOCKSIZE+2) + smem_col] = 0.0;
  }

  __syncthreads();
  
  double tmp = 0.0;
  for (int fi = -1 ; fi < 2; fi++) {
    for (int fj = -1; fj < 2; fj++) { 
        tmp += As[(threadRow + fi + 1) * (BLOCKSIZE + 2) + (threadCol + fj + 1)] * filter[(fi + 1) * 3 + (fj + 1)];
    }
  }

  B[threadRow * N + threadCol] = tmp;
}
```

## Key Features

1. **Shared Memory Tile**: We load a (BLOCKSIZE+2) × (BLOCKSIZE+2) tile into shared memory to account for the halo region needed by the convolution filter
2. **Cooperative Loading**: All threads in a block cooperate to load the data into shared memory
3. **Boundary Handling**: Zero-padding for elements outside the matrix bounds
4. **Thread Synchronization**: `__syncthreads()` ensures all loads are completed before computation begins
5. **Reduced Global Memory Access**: Each input element is loaded once into shared memory and reused multiple times

## Visualization of Shared Memory Blocking

<div class="chart-container">
    <img src="../assets/shared_memory_blocking.png" alt="Shared Memory Blocking">
</div>

## Performance Analysis

This optimization achieves a **X% improvement** over the global memory coalescing implementation, reaching **Y GFLOP/s** on an NVIDIA H100 GPU.

### Advantages

1. **Reduced Global Memory Traffic**: Each input element is read only once from global memory
2. **Higher Throughput**: Shared memory has much higher bandwidth and lower latency than global memory
3. **Data Locality**: Once loaded into shared memory, data can be reused efficiently

### Limitations

1. **Limited Shared Memory Size**: The size of the tile is constrained by the available shared memory
2. **Block Synchronization Overhead**: `__syncthreads()` introduces some overhead
3. **Non-optimal Work Distribution**: Some threads might have more work than others

## Next Steps

Our next optimization will focus on [Block Tiling Techniques](tiling.md) to further improve computational efficiency and memory access patterns.
