#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

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
  // shared mem is shared between all threads in a block
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
