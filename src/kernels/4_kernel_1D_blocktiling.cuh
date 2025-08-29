#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int TM>
__global__ void conv2d1DBlocktiling(int M, int N, const double *A, double *B) {

  // If we flip x and y here we get ~30% less performance for large matrices.
  // The current, 30% faster configuration ensures that blocks with sequential
  // blockIDs access columns of B sequentially, while sharing the same row of A.
  // The slower configuration would share columns of A, but access into B would
  // be non-sequential. So the faster configuration has better spatial locality
  // and hence a greater L2 hit rate.
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / (BN * TM);

  // allocate buffer for current block including padding in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ double As[(BM + 2) * (BN + 2)];

  // Each block loads (BM+2) x (BN+2) elements into shared memory
  for (int i = threadIdx.x; i < (BM+2)*(BN+2); i += blockDim.x) {
    int smem_row = i / (BN+2);
    int smem_col = i % (BN+2);
    int g_row = cRow * BN + smem_row - 1;
    int g_col = cCol * BN + smem_col - 1;
    if (g_row >= 0 && g_row < M && g_col >= 0 && g_col < N)
        As[smem_row * (BN+2) + smem_col] = A[g_row * N + g_col];
    else
        As[smem_row * (BN+2) + smem_col] = 0.0;
  }

  __syncthreads();
  
    
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
  B += (cRow * BM) * N + cCol * BN;
  for (int ti = 0; ti < TM; ti++) {
      B[(threadRow * TM + ti) * N + threadCol] = threadResults[ti];
  }


}
