#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int TM>
__global__ void conv2dVectorize(int M, int N, const double *A, double *B) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  extern __shared__ double As[];

  // Each block loads (BM+2) x (BN+2) elements into shared memory
  if (threadIdx.x <  (BM+2)*(BN+2)/2) {
    for (int i = threadIdx.x; i < (BM+2)*(BN+2)/2; i += blockDim.x) {
      int smem_row = (2 * i) / (BN+2);
      int smem_col = (2 * i) % (BN+2);
      int g_row = cRow * BM + smem_row;
      int g_col = cCol * BN + smem_col;
      double2 tmp =
          reinterpret_cast<const double2 *>(&A[g_row * (N+2) + g_col])[0];
      As[smem_row * (BN+2) + smem_col] = tmp.x;
      As[smem_row * (BN+2) + smem_col + 1] = tmp.y;
    }
  }

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
}
