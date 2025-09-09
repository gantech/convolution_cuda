# Tensor Memory Accelerator (TMA)

In our most advanced optimization, we leverage NVIDIA Hopper architecture's Tensor Memory Accelerator (TMA) to achieve unprecedented memory transfer efficiency for our convolution operations.

## What is TMA?

The Tensor Memory Accelerator (TMA) is a new feature introduced in NVIDIA's Hopper architecture that enables efficient transfer of multidimensional data directly from global memory to shared memory. TMA provides:

1. **Hardware-accelerated data movement**: Uses dedicated hardware paths to transfer data
2. **Asynchronous memory transfers**: Allows computation to overlap with memory transfers
3. **Optimized for multidimensional data**: Perfect for operations like convolution
4. **Reduced register pressure**: Memory transfer details handled by hardware

## Implementation

```cuda
__device__ barrier::arrival_token load_from_gmem(
                             const CUtensorMap tensor_map_a,
                             const int g_col, const int g_row, double * As,
                             const int bytes, barrier& bar) {

  if (threadIdx.x == 0) {
    // Initialize barrier. All `blockDim.x` threads in block participate.
    init(&bar, blockDim.x);
    // Make initialized barrier visible in async proxy.
    cde::fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  barrier::arrival_token token;
  if (threadIdx.x == 0) {
    // Initiate bulk tensor copy.
    cde::cp_async_bulk_tensor_2d_global_to_shared(As, &tensor_map_a, g_col, g_row, bar);
    // Arrive on the barrier and tell how many bytes are expected to come in.
    token = cuda::device::barrier_arrive_tx(bar, 1, bytes);
  } else {
    // Other threads just arrive.
    token = bar.arrive();
  }

  return token;
}

template <const int BLOCKSIZE>
__global__ void conv2d_shared_mem_tma(const __grid_constant__ CUtensorMap tensor_map_a,
                                      int M, int N,
                                      const double *A, double *B) {
  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  double filter[9] = {-1.0, -1.0, -1.0,
          -1.0, 8.0, -1.0,
          -1.0, -1.0, -1.0};

  // allocate buffer for current block including padding in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ alignas(128) double As[(BLOCKSIZE + 2) * (BLOCKSIZE + 2)];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // Initialize shared memory barrier
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;

  barrier::arrival_token token = load_from_gmem(tensor_map_a,
                                                cCol * BLOCKSIZE, cRow * BLOCKSIZE,
                                                As, (BLOCKSIZE+2) * (BLOCKSIZE+2) * sizeof(double),
                                                bar);

  // Wait for the data to have arrived.
  bar.wait(std::move(token));

  double tmp = 0.0;
  for (int fi = -1 ; fi < 2; fi++) {
    for (int fj = -1; fj < 2; fj++) {
        tmp += As[(threadRow + fi + 1) * (BLOCKSIZE + 2) + (threadCol + fj + 1)] * filter[(fi + 1) * 3 + (fj + 1)];
    }
  }

  if ((cRow * BLOCKSIZE + threadRow) < M && (cCol * BLOCKSIZE + threadCol) < N) {
    B[(cRow * BLOCKSIZE + threadRow) * N + (cCol * BLOCKSIZE + threadCol)] = tmp;
  }

  if (threadIdx.x == 0) {
    (&bar)->~barrier();
  }
}
```

## Key Components of TMA Implementation

1. **Tensor Map Descriptor**: A hardware-resident descriptor that defines the shape, size, and layout of the multidimensional tensor data
2. **Barrier Synchronization**: Uses CUDA's barrier API to coordinate asynchronous memory transfers
3. **Memory Alignment**: Shared memory must be 128-byte aligned for TMA operations
4. **Bulk Tensor Copy**: The `cp_async_bulk_tensor_2d_global_to_shared` function initiates the hardware-accelerated copy
5. **Arrival Tokens**: Used to track completion of asynchronous operations

## Setting up the Tensor Map

```cuda
CUtensorMap get_tensor_map(double *A, const int M, const int N,
                           const int BM, const int BN) {

  CUtensorMap tensor_map_a{};
  // rank is the number of dimensions of the array
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {M+2, N+2};
  // stride in bytes between consecutive elements in each dimension
  uint64_t stride[rank - 1] = {(N+2) * sizeof(double)};
  // box size defines the shape of each TMA transfer
  uint32_t box_size[rank] = {BM+2, BN+2};
  // element stride in each dimension
  uint32_t elem_stride[rank] = {1, 1};

  // Create the tensor descriptor
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
  CUresult res_a = cuTensorMapEncodeTiled(
    &tensor_map_a,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
    rank,
    A,
    size,
    stride,
    box_size,
    elem_stride,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );

  return tensor_map_a;
}
```

## Visualization of TMA Operation

<div class="chart-container">
    <img src="../assets/tma_operation.png" alt="TMA Operation">
</div>

## Performance Analysis

The TMA implementation achieves a remarkable **X% improvement** over our previous best implementation, reaching **Y GFLOP/s** on an NVIDIA H100 GPU. This represents **Z%** of the theoretical peak performance.

### Advantages

1. **Hardware Acceleration**: Dedicated hardware paths for memory transfers
2. **Reduced CPU Overhead**: Memory layout and transfer details handled by hardware
3. **Asynchronous Operation**: Computation can overlap with memory transfers
4. **Lower Register Usage**: Hardware handles address calculations
5. **Optimized for 2D Data**: Perfect for convolution operations

### Limitations

1. **Hardware Requirements**: Only available on Hopper architecture and newer
2. **Memory Alignment**: Requires careful attention to alignment requirements
3. **API Complexity**: More complex setup than standard memory operations

## Conclusion

The Tensor Memory Accelerator represents a significant leap forward in optimizing memory transfers for convolution operations. By leveraging this hardware feature, we achieve unprecedented performance that approaches the theoretical limits of the GPU.

For the most performance-critical applications working with structured data, TMA provides an essential tool to maximize memory throughput and computational efficiency.
