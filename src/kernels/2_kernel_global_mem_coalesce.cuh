#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

template <const uint BLOCKSIZE>
__global__ void conv2d_global_mem_coalesce(int M, int N, const double *A, double *B) {
  const int i = blockIdx.x * BLOCKSIZE + threadIdx.y;
  const int j = blockIdx.y * BLOCKSIZE + threadIdx.x;

  double filter[9] = {-1.0, -1.0, -1.0,
          -1.0, 8.0, -1.0,
          -1.0, -1.0, -1.0};
  double avals[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  A += i * N + j;

  if ( (i < M) && (j < N) ) {

    for (int fi = -1 ; fi < 2; fi++) {
      for (int fj = -1; fj < 2; fj++) {
        if ( (i + fi) >= 0 && (i + fi) < M && (j + fj) >= 0 && (j + fj) < N ) 
          avals[(fi + 1) * 3 + (fj + 1)] = A[fi * N + fj];
      }
    }

  double tmp = 0.0;
  for (int floop = 0; floop < 9; floop++) 
      tmp += avals[floop] * filter[floop];
  B[i * N + j] = tmp;

  }

}
