#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

namespace db {

template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(const int N, const int K, double *A, double *B,
                             double *As, double *Bs, const int innerRowA,
                             const int innerColA, const int innerRowB,
                             const int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    double4 tmp = reinterpret_cast<double4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    // transpose A while storing it
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<double4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<double4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(double *regM, double *regN, double *threadResults, const double *As,
                const double *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {

}

} // namespace db

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    conv2dDoubleBuffering(int M, int N, const double *A, double *B) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

}