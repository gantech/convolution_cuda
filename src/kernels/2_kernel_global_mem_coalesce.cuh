#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const uint BLOCKSIZE>
__global__ void conv2d_global_mem_coalesce(int M, int N, const double *A, double *B) {
  const int cRow = blockIdx.x * BLOCKSIZE + threadIdx.y;
  const int cCol = blockIdx.y * BLOCKSIZE + threadIdx.x;

}
