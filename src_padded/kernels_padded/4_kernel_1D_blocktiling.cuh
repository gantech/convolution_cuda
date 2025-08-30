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

template <const int BM, const int BN, const int TM>
__global__ void conv2d1DBlocktiling(int M, int N, const double *A, double *B) {

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  extern __shared__ double As[];

  // Each block loads (BM+2) x (BN+2) elements into shared memory
  for (int i = threadIdx.x; i < (BM+2)*(BN+2); i += blockDim.x) {
    int smem_row = i / (BN+2);
    int smem_col = i % (BN+2);
    int g_row = cRow * BM + smem_row - 1;
    int g_col = cCol * BN + smem_col - 1;
    As[smem_row * (BN+2) + smem_col] = A[g_row * N + g_col];
  }

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / (BN * TM);
    
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

}
