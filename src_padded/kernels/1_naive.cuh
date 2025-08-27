#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/*

Matrix sizes: MxN

*/

__global__ void conv2d_naive(int M, int N, const double *A,
                            double *B) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  double filter[9] = {-1.0, -1.0, -1.0,
          -1.0, 8.0, -1.0,
          -1.0, -1.0, -1.0};
  double avals[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  A += (i+1) * (N+2) + j+1;

  if ( (i < M) && (j < N) ) {

    for (int fi = -1 ; fi < 2; fi++) {
      for (int fj = -1; fj < 2; fj++) {
        avals[(fi + 1) * 3 + (fj + 1)] = A[fi * (N+2) + fj];
      }
    }

  double tmp = 0.0;
  for (int floop = 0; floop < 9; floop++) 
      tmp += avals[floop] * filter[floop];
  B[i * N + j] = tmp;

  }

  
}
