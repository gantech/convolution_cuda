#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void conv2dResolveBankConflicts(int M, int N, const double *A, double *B) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

}