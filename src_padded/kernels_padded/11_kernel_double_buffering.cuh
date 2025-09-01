#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// Create an aligned shared memory using a union
union SharedMemory {
  char raw[1];
  __align__(128) double aligned;
};

template <const int BM, const int BN, const int ROWS_PER_BLOCK>
__global__ void conv2dDoubleBuffering(const __grid_constant__ CUtensorMap tensor_map_a,
                                      const __grid_constant__ CUtensorMap tensor_map_b,
                                      int M, int N, const double *A, double *B) {

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  extern __shared__ SharedMemory shared_mem[];

  double* As[2] ;
  As[0] = reinterpret_cast<double*>(shared_mem);
  As[1] = As[0] + 1024;

  double* Bs[2] = {As[0] + 2048,
                   As[0] + 2560};

  // Initialize shared memory barrier with the number of threads participating in the barrier.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar_a[2];
  //__shared__ barrier bar_b[2];
  if (threadIdx.x == 0) {
    for (int i = 0; i < 2; i++) {
      init(&bar_a[i], blockDim.x);
      //init(&bar_b[i], blockDim.x);
    }
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  barrier::arrival_token token[2];
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_global_to_shared(As[0], &tensor_map_a, cCol * BN, cRow * BM, bar_a[0]);
    token[0] = cuda::device::barrier_arrive_tx(bar_a[0], 1, (ROWS_PER_BLOCK+2) * (BN+2) * sizeof(double));
  } else {
    token[0] = bar_a[0].arrive();
  }

  const int threadRow = threadIdx.x / BN;
  const int threadCol = threadIdx.x % BN;

  double filter[9] = {-1.0, -1.0, -1.0,
            -1.0, 8.0, -1.0,
            -1.0, -1.0, -1.0};  

  bar_a[0].wait(std::move(token[0]));

  // TODO: Initiate TMA setup for sending Bs back to global memory
  for (int i = ROWS_PER_BLOCK; i < BM; i += ROWS_PER_BLOCK) {

    int idx = (i / ROWS_PER_BLOCK) % 2;

    if (threadIdx.x == 0) {
      cde::cp_async_bulk_tensor_2d_global_to_shared(As[idx], &tensor_map_a, cCol * BN, cRow * BM + i, bar_a[idx]);
      token[idx] = cuda::device::barrier_arrive_tx(bar_a[idx], 1,  (ROWS_PER_BLOCK+2) * (BN+2) * sizeof(double));
    } else {
      token[idx] = bar_a[idx].arrive();
    }

    int idx_prev = idx ^ 1;

    double tmp = 0.0;
    for (int fi = -1 ; fi < 2; fi++) {
      for (int fj = -1; fj < 2; fj++) { 
        tmp += As[idx_prev][(threadRow + fi + 1) * (BN + 2) + (threadCol + fj + 1)] * filter[(fi + 1) * 3 + (fj + 1)];
      }
    }

    // Store the result into Bs[idx_prev]
    Bs[idx_prev][threadRow * BN + threadCol] = tmp;

    // Wait for shared memory writes to be visible to TMA engine.
    cde::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (threadIdx.x == 0) {
      cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map_b, cCol * BN, cRow * BM + i - ROWS_PER_BLOCK,
                                                    Bs[idx_prev]);
      // Wait for TMA transfer to have finished reading shared memory.
      // Create a "bulk async-group" out of the previous bulk copy operation.
      cde::cp_async_bulk_commit_group();
      // Wait for the group to have completed reading from shared memory.
      cde::cp_async_bulk_wait_group_read<0>();
    }

    // Receive data from global memory
    bar_a[idx].wait(std::move(token[idx]));


  }

  if (threadIdx.x == 0) {
    for (int i = 0; i < 2; i++)
      (&bar_a[i])->~barrier();
  }

}
