#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap
#include <cuda/barrier>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// Create an aligned shared memory using a union
union SharedMemory {
  char raw[1];
  __align__(128) double aligned;
};

template <const int BM, const int BN, const int TM>
__global__ void conv2d1DBlocktiling(const __grid_constant__ CUtensorMap tensor_map_a, 
                                    int M, int N, const double *A, double *B) {

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  extern __shared__ SharedMemory shared_mem[];

  // Now you can use the properly aligned array
  double* As = reinterpret_cast<double*>(shared_mem);

  // Each block loads (BM+2) x (BN+2) elements into shared memory
  for (int i = threadIdx.x; i < (BM+2)*(BN+2); i += blockDim.x) {
    int smem_row = i / (BN+2);
    int smem_col = i % (BN+2);
    int g_row = cRow * BN + smem_row - 1;
    int g_col = cCol * BN + smem_col - 1;
    As[smem_row * (BN+2) + smem_col] = A[g_row * N + g_col];
  }

  // // Initialize shared memory barrier with the number of threads participating in the barrier.
  // #pragma nv_diag_suppress static_var_with_dynamic_init
  // __shared__ barrier bar;

  // if (threadIdx.x == 0) {
  //   // Initialize barrier. All `blockDim.x` threads in block participate.
  //   init(&bar, blockDim.x);
  //   // Make initialized barrier visible in async proxy.
  //   cde::fence_proxy_async_shared_cta();
  // }
  // // Syncthreads so initialized barrier is visible to all threads.
  // __syncthreads();

  // barrier::arrival_token token;
  // if (threadIdx.x == 0) {
  //   // Initiate bulk tensor copy.
  //   cde::cp_async_bulk_tensor_2d_global_to_shared(As, &tensor_map_a, cCol * BN, cRow * BM, bar);
  //   // Arrive on the barrier and tell how many bytes are expected to come in.
  //   token = cuda::device::barrier_arrive_tx(bar, 1,  (BM+2) * (BN+2) * sizeof(double));
  // } else {
  //   // Other threads just arrive.
  //   token = bar.arrive();
  // }
  // // Wait for the data to have arrived.
  // bar.wait(std::move(token));

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;
    
  // allocate thread-local cache for results in registerfile
  double threadResults[TM] = {0.0};

  double filter[9] = {-1.0, -1.0, -1.0,
          -1.0, 8.0, -1.0,
          -1.0, -1.0, -1.0};

  for (int fi = -1 ; fi < 2; fi++) {
    for (int fj = -1; fj < 2; fj++) { 
       for (int ti = 0; ti < TM; ti++) {
          threadResults[ti] += As[(threadRow * TM + ti + fi + 1) * (BN + 2) + (threadCol + fj + 1)] * filter[(fi + 1) * 3 + (fj + 1)];
       }
    }
  }

  // advance pointers to the starting positions
  for (int ti = 0; ti < TM; ti++) {
      B[(cRow * BM + threadRow * TM + ti) * N + cCol * BN + threadCol] = threadResults[ti];
  }

  // if (threadIdx.x == 0) {
  //   (&bar)->~barrier();
  // }    

}
