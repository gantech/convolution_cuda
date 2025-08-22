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
  A += (cRow * BLOCKSIZE) * N + cCol * BLOCKSIZE; // row=cRow, col=cCol
  B += (cRow * BLOCKSIZE) * N + cCol * BLOCKSIZE;

  As[(threadRow + 1) * (BLOCKSIZE + 2) + (threadCol + 1)] = A[threadRow * N + threadCol];

  // Now the padding
  if (threadIdx.x == 0) // Bottom left
    As[0] = ( (cRow > 0) && (cCol > 0) ) ? A[-N-1] : 0;

  if (threadIdx.x == 1) // Bottom right
    As[BLOCKSIZE + 1] = ( (cRow > 0) && (cCol * BLOCKSIZE < N) )? A[-N + BLOCKSIZE] : 0;

  if (threadIdx.x == 2) // Top left
    As[(BLOCKSIZE+1) * (BLOCKSIZE + 2)] = (((cRow+1) * BLOCKSIZE < M) && (cCol > 0) )? A[ (BLOCKSIZE * N) - 1] : 0;

  if (threadIdx.x == 3) // Top right
    As[(BLOCKSIZE + 1) * (BLOCKSIZE + 2) + (BLOCKSIZE + 1)] = ( ((cRow+1) * BLOCKSIZE < M) && ((cCol + 1) * BLOCKSIZE < N) ) ? A[ (BLOCKSIZE * N) + BLOCKSIZE] : 0;

  if (threadRow == 0) { // Bottom
    As[threadCol + 1] = (cRow > 0) ? A[-N + threadCol] : 0;
  } else if (threadRow == (BLOCKSIZE - 1) ) { // Top
    As[(BLOCKSIZE + 1) * (BLOCKSIZE + 2) + (threadCol + 1)] = (((cRow+1) * BLOCKSIZE < M) ? A[ threadRow + N + threadCol] : 0);
  }

  if (threadCol == 0) { // Left
    As[(threadRow + 1) * (BLOCKSIZE + 2)] = (cCol > 0) ? A[threadRow * N - 1] : 0;
  } else if (threadCol == (BLOCKSIZE - 1)) { // Right
    As[(threadRow + 1) * (BLOCKSIZE + 2) + (BLOCKSIZE + 1)] = ((cCol+1) * BLOCKSIZE < N) ? A[threadRow * N + BLOCKSIZE] : 0;
  }

  __syncthreads();

  double tmp = 0.0;
  for (int fi = -1 ; fi < 2; fi++) {
    for (int fj = -1; fj < 2; fj++) { 
        tmp += As[(threadRow + fi + 1) * (BLOCKSIZE + 2) + (threadCol + fj + 1)] * filter[(fi + 1) * 3 + (fj + 1)];
    }
  }

  __syncthreads();
  
  B[threadRow * N + threadCol] = tmp;

}