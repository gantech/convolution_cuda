#include "kernels_padded.cuh"
#include "runner_padded.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>

#define cudaCheck2(err) (cudaCheck(err, __FILE__, __LINE__))

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

float get_sec() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end) { return 1.0e-6 * (end - beg); }

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

void CudaDeviceInfo() {
  int deviceId;

  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
};

void randomize_matrix(double *mat, int M, int N) {
  // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
  // precision is too low and the same random number is generated.
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);

  zero_init_matrix(mat, (M+2)*(N+2));  

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        double tmp = (double)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[(i+1) * (N+2) + (j+1)] = tmp;
    }
  }

}

void range_init_matrix(double *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = i;
  }
}

void zero_init_matrix(double *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = 0.0;
  }
}

void copy_matrix(const double *src, double *dest, int N) {
  int i;
  for (i = 0; src + i && dest + i && i < N; i++)
    *(dest + i) = *(src + i);
  if (i != N)
    printf("copy failed at %d while there are %d elements in total.\n", i, N);
}

void print_matrix(const double *A, int M, int N, std::ofstream &fs) {
  int i;
  fs << std::setprecision(2)
     << std::fixed; // Set doubleing-point precision and fixed notation
  fs << "[";
  for (i = 0; i < M * N; i++) {
    if ((i + 1) % N == 0)
      fs << std::setw(5) << A[i]; // Set field width and write the value
    else
      fs << std::setw(5) << A[i] << ", ";
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N)
        fs << ";\n";
    }
  }
  fs << "]\n";
}

bool verify_matrix(double *matRef, double *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    diff = std::fabs(matRef[i] - matOut[i]);
    if (diff > 0.01) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
             matRef[i], matOut[i], diff, i);
      return false;
    }
  }
  return true;
}

int div_ceil(int numerator, int denominator) {
  std::div_t res = std::div(numerator, denominator);
  return res.rem ? (res.quot + 1) : res.quot;
}

void run_conv2d_naive(int M, int N, double *A, double *B) {
  dim3 gridDim(M/32, N/32);
  dim3 blockDim(32, 32);
  conv2d_naive<<<gridDim, blockDim>>>(M, N, A, B);
}

void run_conv2d_coalesce(int M, int N, double *A, double *B) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  conv2d_global_mem_coalesce<32>
      <<<gridDim, blockDim>>>(M, N, A, B);
}

void run_conv2d_shared_mem_block(int M, int N, double *A, double *B) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  cudaFuncSetAttribute(conv2d_shared_mem_block<32>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  conv2d_shared_mem_block<32>
      <<<gridDim, blockDim>>>(M, N, A, B);
}

// void runConv2d1DBlocktiling(int M, int N, double *A, double *B) {
//   const uint BM = 64;
//   const uint BN = 64;
//   const uint TM = 8;
//   dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//   // There are BM * BN elements to be calculated by this block 
//   // by BM * BN / TM threads such that each thread calculates TM elements.
//   dim3 blockDim((BM * BN) / TM);
//   conv2d1DBlocktiling<BM, BN, TM>
//       <<<gridDim, blockDim>>>(M, N, A, B);
// }

// void runConv2d2DBlocktiling(int M, int N, double *A, double *B) {
//   const uint TM = 8;
//   const uint TN = 8;
//   if (M >= 128 and N >= 128) {
//     const uint BM = 128;
//     const uint BN = 128;
//     dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//     dim3 blockDim((BM * BN) / (TM * TN));
//     // Add before kernel launch:
//     cudaFuncSetAttribute(conv2d2DBlocktiling<128,128,8,8>, 
//                     cudaFuncAttributeMaxDynamicSharedMemorySize, 
//                     135200);  // Increase to almost 132KB if hardware allows
//     conv2d2DBlocktiling<BM, BN, TM, TN>
//         <<<gridDim, blockDim, 135200>>>(M, N, A, B);
//   } else {
//     // this is a hacky solution to the underlying problem
//     // of not having proper bounds checking in the kernel
//     const uint BM = 64;
//     const uint BN = 64;
//     dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//     dim3 blockDim((BM * BN) / (TM * TN));
//     // Add before kernel launch:
//     cudaFuncSetAttribute(conv2d2DBlocktiling<64,64,8,8>, 
//                     cudaFuncAttributeMaxDynamicSharedMemorySize, 
//                     34848);  
//     conv2d2DBlocktiling<BM, BN, TM, TN>
//         <<<gridDim, blockDim, 34848>>>(M, N, A, B);
//   }
// }

// void runConv2dVectorize(int M, int N, double *A, double *B) {
//   const uint BK = 8;
//   const uint TM = 8;
//   const uint TN = 8;
//   if (M >= 128 and N >= 128) {
//     const uint BM = 128;
//     const uint BN = 128;
//     dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//     dim3 blockDim((BM * BN) / (TM * TN));
//     conv2dVectorize<BM, BN, BK, TM, TN>
//         <<<gridDim, blockDim>>>(M, N, A, B);
//   } else {
//     // this is a hacky solution to the underlying problem
//     // of not having proper bounds checking in the kernel
//     const uint BM = 64;
//     const uint BN = 64;
//     dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//     dim3 blockDim((BM * BN) / (TM * TN));
//     conv2dVectorize<BM, BN, BK, TM, TN>
//         <<<gridDim, blockDim>>>(M, N, A, B);
//   }
// }

// void runConv2dResolveBankConflicts(int M, int N, double *A, double *B) {
//   const uint BK = 8;
//   const uint TM = 8;
//   const uint TN = 8;
//   if (M >= 128 and N >= 128) {
//     const uint BM = 128;
//     const uint BN = 128;
//     dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//     dim3 blockDim((BM * BN) / (TM * TN));
//     conv2dResolveBankConflicts<BM, BN, BK, TM, TN>
//         <<<gridDim, blockDim>>>(M, N, A, B);
//   } else {
//     // this is a hacky solution to the underlying problem
//     // of not having proper bounds checking in the kernel
//     const uint BM = 64;
//     const uint BN = 64;
//     dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//     dim3 blockDim((BM * BN) / (TM * TN));
//     conv2dResolveBankConflicts<BM, BN, BK, TM, TN>
//         <<<gridDim, blockDim>>>(M, N, A, B);
//   }
// }

// void runConv2dResolveBankExtraCol(int M, int N, double *A, double *B) {
//   const uint BK = 8;
//   const uint TM = 8;
//   const uint TN = 8;
//   if (M >= 128 and N >= 128) {
//     const uint BM = 128;
//     const uint BN = 128;
//     dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//     dim3 blockDim((BM * BN) / (TM * TN));
//     conv2dResolveBankExtraCol<BM, BN, BK, TM, TN>
//         <<<gridDim, blockDim>>>(M, N, A, B);
//   } else {
//     // this is a hacky solution to the underlying problem
//     // of not having proper bounds checking in the kernel
//     const uint BM = 64;
//     const uint BN = 64;
//     dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
//     dim3 blockDim((BM * BN) / (TM * TN));
//     conv2dResolveBankExtraCol<BM, BN, BK, TM, TN>
//         <<<gridDim, blockDim>>>(M, N, A, B);
//   }
// }

// void runConv2dAutotuned(int M, int N, double *A, double *B) {
//   // A100
//   // const uint K9_BK = 16;
//   // const uint K9_TM = 4;
//   // const uint K9_TN = 4;
//   // const uint K9_BM = 64;
//   // const uint K9_BN = 64;
//   // A6000
//   const uint K9_BK = 16;
//   const uint K9_TM = 8;
//   const uint K9_TN = 8;
//   const uint K9_BM = 128;
//   const uint K9_BN = 128;
//   dim3 blockDim(K9_NUM_THREADS);

//   static_assert(
//       (K9_NUM_THREADS * 4) % K9_BK == 0,
//       "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization issues "
//       "during GMEM->SMEM tiling (loading only parts of the final row of Bs "
//       "during each iteraion)");
//   static_assert(
//       (K9_NUM_THREADS * 4) % K9_BN == 0,
//       "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization issues "
//       "during GMEM->SMEM tiling (loading only parts of the final row of As "
//       "during each iteration)");
//   static_assert(
//       K9_BN % (16 * K9_TN) == 0,
//       "K9_BN must be a multiple of 16*K9_TN to avoid quantization effects");
//   static_assert(
//       K9_BM % (16 * K9_TM) == 0,
//       "K9_BM must be a multiple of 16*K9_TM to avoid quantization effects");
//   static_assert((K9_BM * K9_BK) % (4 * K9_NUM_THREADS) == 0,
//                 "K9_BM*K9_BK must be a multiple of 4*256 to vectorize loads");
//   static_assert((K9_BN * K9_BK) % (4 * K9_NUM_THREADS) == 0,
//                 "K9_BN*K9_BK must be a multiple of 4*256 to vectorize loads");

//   dim3 gridDim(CEIL_DIV(N, K9_BN), CEIL_DIV(M, K9_BM));
//   conv2dAutotuned<K9_BM, K9_BN, K9_BK, K9_TM, K9_TN>
//       <<<gridDim, blockDim>>>(M, N, A, B);
// }

// void runConv2dWarptiling(int M, int N, double *A, double *B) {
//   // Settings for A100
//   // const uint K10_NUM_THREADS = 128;
//   // const uint K10_BN = 128;
//   // const uint K10_BM = 64;
//   // const uint K10_BK = 16;
//   // const uint K10_WN = 64;
//   // const uint K10_WM = 32;
//   // const uint K10_WNITER = 1;
//   // const uint K10_TN = 4;
//   // const uint K10_TM = 4;
//   // Settings for A6000
//   const uint K10_NUM_THREADS = 128;
//   const uint K10_BN = 128;
//   const uint K10_BM = 128;
//   const uint K10_BK = 16;
//   const uint K10_WN = 64;
//   const uint K10_WM = 64;
//   const uint K10_WNITER = 4;
//   const uint K10_TN = 4;
//   const uint K10_TM = 8;
//   dim3 blockDim(K10_NUM_THREADS);

//   constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

//   // warptile in threadblocktile
//   static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
//   static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);

//   // threads in warpsubtile
//   static_assert((K10_WM * K10_WN) % (WARPSIZE * K10_TM * K10_TN * K10_WNITER) ==
//                 0);
//   constexpr uint K10_WMITER =
//       (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
//   // warpsubtile in warptile
//   static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));

//   static_assert((K10_NUM_THREADS * 4) % K10_BK == 0,
//                 "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
//                 "issues during GMEM->SMEM tiling (loading only parts of the "
//                 "final row of Bs during each iteraion)");
//   static_assert((K10_NUM_THREADS * 4) % K10_BN == 0,
//                 "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
//                 "issues during GMEM->SMEM tiling (loading only parts of the "
//                 "final row of As during each iteration)");
//   static_assert(K10_BN % (16 * K10_TN) == 0,
//                 "BN must be a multiple of 16*TN to avoid quantization effects");
//   static_assert(K10_BM % (16 * K10_TM) == 0,
//                 "BM must be a multiple of 16*TM to avoid quantization effects");
//   static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
//                 "BM*BK must be a multiple of 4*256 to vectorize loads");
//   static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
//                 "BN*BK must be a multiple of 4*256 to vectorize loads");

//   dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));
//   conv2dWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
//                   K10_TN, K10_NUM_THREADS>
//       <<<gridDim, blockDim>>>(M, N, A, B);
// }

// void runConv2dDoubleBuffering(int M, int N, double *A, double *B) {
//   // Settings for A100
//   // const uint K11_NUM_THREADS = 256;
//   // const uint K11_BN = 128;
//   // const uint K11_BM = 64;
//   // const uint K11_BK = 16;
//   // const uint K11_WN = 32;
//   // const uint K11_WM = 32;
//   // const uint K11_WNITER = 2;
//   // const uint K11_TN = 4;
//   // const uint K11_TM = 4;
//   // Settings for A6000
//   const uint K11_NUM_THREADS = 256;
//   const uint K11_BN = 256;
//   const uint K11_BM = 128;
//   const uint K11_BK = 16;
//   const uint K11_WN = 32;
//   const uint K11_WM = 128;
//   const uint K11_WNITER = 1;
//   const uint K11_TN = 8;
//   const uint K11_TM = 8;
//   dim3 blockDim(K11_NUM_THREADS);

//   constexpr uint NUM_WARPS = K11_NUM_THREADS / 32;

//   // warptile in threadblocktile
//   static_assert((K11_BN % K11_WN == 0) and (K11_BM % K11_WM == 0));
//   static_assert((K11_BN / K11_WN) * (K11_BM / K11_WM) == NUM_WARPS);

//   // threads in warpsubtile
//   static_assert((K11_WM * K11_WN) % (WARPSIZE * K11_TM * K11_TN * K11_WNITER) ==
//                 0);
//   constexpr uint K11_WMITER =
//       (K11_WM * K11_WN) / (32 * K11_TM * K11_TN * K11_WNITER);
//   // warpsubtile in warptile
//   static_assert((K11_WM % K11_WMITER == 0) and (K11_WN % K11_WNITER == 0));

//   static_assert((K11_NUM_THREADS / 2 * 4) % K11_BK == 0,
//                 "NUM_THREADS*4 must be multiple of BK to avoid quantization "
//                 "issues during GMEM->SMEM tiling (loading only parts of the "
//                 "final row of Bs during each iteraion)");
//   static_assert((K11_NUM_THREADS / 2 * 4) % K11_BN == 0,
//                 "NUM_THREADS*4 must be multiple of BN to avoid quantization "
//                 "issues during GMEM->SMEM tiling (loading only parts of the "
//                 "final row of As during each iteration)");
//   static_assert(K11_BN % (16 * K11_TN) == 0,
//                 "BN must be a multiple of 16*TN to avoid quantization effects");
//   static_assert(K11_BM % (16 * K11_TM) == 0,
//                 "BM must be a multiple of 16*TM to avoid quantization effects");
//   static_assert((K11_BM * K11_BK) % (4 * K11_NUM_THREADS / 2) == 0,
//                 "BM*BK must be a multiple of 4*256 to vectorize loads");
//   static_assert((K11_BN * K11_BK) % (4 * K11_NUM_THREADS / 2) == 0,
//                 "BN*BK must be a multiple of 4*256 to vectorize loads");

//   dim3 gridDim(CEIL_DIV(N, K11_BN), CEIL_DIV(M, K11_BM));
//   // conv2dDoubleBuffering<K11_BM, K11_BN, K11_BK, K11_WM, K11_WN, K11_WNITER,
//   //                      K11_TM, K11_TN, K11_NUM_THREADS>
//   //     <<<gridDim, blockDim>>>(M, N, A, B);
// }

// void runConv2dDoubleBuffering2(int M, int N, double *A, double *B) {
//   // Settings for A6000
//   const uint K12_NUM_THREADS = 128;
//   const uint K12_BN = 128;
//   const uint K12_BM = 128;
//   const uint K12_BK = 16;
//   const uint K12_WN = 64;
//   const uint K12_WM = 64;
//   const uint K12_WNITER = 4;
//   const uint K12_TN = 4;
//   const uint K12_TM = 8;
//   dim3 blockDim(K12_NUM_THREADS);

//   constexpr uint NUM_WARPS = K12_NUM_THREADS / 32;

//   // warptile in threadblocktile
//   static_assert((K12_BN % K12_WN == 0) and (K12_BM % K12_WM == 0));
//   static_assert((K12_BN / K12_WN) * (K12_BM / K12_WM) == NUM_WARPS);

//   // threads in warpsubtile
//   static_assert((K12_WM * K12_WN) % (WARPSIZE * K12_TM * K12_TN * K12_WNITER) ==
//                 0);
//   constexpr uint K12_WMITER =
//       (K12_WM * K12_WN) / (32 * K12_TM * K12_TN * K12_WNITER);
//   // warpsubtile in warptile
//   static_assert((K12_WM % K12_WMITER == 0) and (K12_WN % K12_WNITER == 0));

//   static_assert((K12_NUM_THREADS * 4) % K12_BK == 0,
//                 "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
//                 "issues during GMEM->SMEM tiling (loading only parts of the "
//                 "final row of Bs during each iteraion)");
//   static_assert((K12_NUM_THREADS * 4) % K12_BN == 0,
//                 "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
//                 "issues during GMEM->SMEM tiling (loading only parts of the "
//                 "final row of As during each iteration)");
//   static_assert(K12_BN % (16 * K12_TN) == 0,
//                 "BN must be a multiple of 16*TN to avoid quantization effects");
//   static_assert(K12_BM % (16 * K12_TM) == 0,
//                 "BM must be a multiple of 16*TM to avoid quantization effects");
//   static_assert((K12_BM * K12_BK) % (4 * K12_NUM_THREADS) == 0,
//                 "BM*BK must be a multiple of 4*256 to vectorize loads");
//   static_assert((K12_BN * K12_BK) % (4 * K12_NUM_THREADS) == 0,
//                 "BN*BK must be a multiple of 4*256 to vectorize loads");

//   dim3 gridDim(CEIL_DIV(N, K12_BN), CEIL_DIV(M, K12_BM));
//   // runConv2dDoubleBuffering2<K12_BM, K12_BN, K12_BK, K12_WM, K12_WN, K12_WNITER,
//   //                          K12_TM, K12_TN, K12_NUM_THREADS>
//   //     <<<gridDim, blockDim>>>(M, N, A, B);
// }

void run_kernel(int kernel_num, int M, int N, double *A, double *B) {
  switch (kernel_num) {
  // case 0:
  //   runCuDNNFP64(M, N, A, B);
  //   break;
  case 1:
    run_conv2d_naive(M, N, A, B);
    break;
  case 2:
    run_conv2d_coalesce(M, N, A, B);
    break;  
  case 3:
    run_conv2d_shared_mem_block(M, N,  A, B);
    break;
  // case 4:
  //   runConv2d1DBlocktiling(M, N, A, B);
  //   break;
  // case 5:
  //   runConv2d2DBlocktiling(M, N, A, B);
  //   break;
  // case 6:
  //   runConv2dVectorize(M, N, A, B);
  //   break;
  // case 7:
  //   runConv2dResolveBankConflicts(M, N, A, B);
  //   break;
  // case 8:
  //   runConv2dResolveBankExtraCol(M, N, A, B);
  //   break;
  // case 9:
  //   runConv2dAutotuned(M, N, A, B);
  //   break;
  // case 10:
  //   runConv2dWarptiling(M, N, A, B);
  //   break;
  // case 11:
  //   runConv2dDoubleBuffering(M, N, A, B);
  //   break;
  // case 12:
  //   runConv2dDoubleBuffering2(M, N, A, B);
  //   break;
  default:
    throw std::invalid_argument("Unknown kernel number");
  }
  
}
