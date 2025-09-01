#include "kernels_padded.cuh"
#include "runner_padded.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <cuda/barrier>
#include <cuda_runtime.h>
#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap

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

float run_conv2d_naive(int M, int N, double *A, double *B, int n_repeat) {
  dim3 gridDim(M/32, N/32);
  dim3 blockDim(32, 32);
  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);
  for (int i = 0; i < n_repeat; i++) {
    conv2d_naive<<<gridDim, blockDim>>>(M, N, A, B);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(beg);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, beg, end);
  return elapsed_time;
}

float run_conv2d_coalesce(int M, int N, double *A, double *B, int n_repeat) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);
  for (int i = 0; i < n_repeat; i++) {
    conv2d_global_mem_coalesce<32>
        <<<gridDim, blockDim>>>(M, N, A, B);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(beg);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, beg, end);
  return elapsed_time;
}

float run_conv2d_shared_mem_block(int M, int N, double *A, double *B, int n_repeat) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  cudaFuncSetAttribute(conv2d_shared_mem_block<32>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);
  for (int i = 0; i < n_repeat; i++) {
    conv2d_shared_mem_block<32>
        <<<gridDim, blockDim>>>(M, N, A, B);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(beg);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, beg, end);
  return elapsed_time;
}

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
  // Get pointer to cuTensorMapEncodeTiled
  cudaDriverEntryPointQueryResult driver_status;
  void* cuTensorMapEncodeTiled_ptr = nullptr;
  cudaCheck2(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status));
  assert(driver_status == cudaDriverEntryPointSuccess);

  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}

CUtensorMap get_tensor_map(double *A, const int M, const int N,
                           const int BM, const int BN) {

  CUtensorMap tensor_map_a{};
  // rank is the number of dimensions of the array.
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {M, N};
  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  uint64_t stride[rank - 1] = {(N) * sizeof(double)};
  // The box_size is the size of the shared memory buffer that is used as the
  // destination of a TMA transfer.
  uint32_t box_size[rank] = {BM, BN};
  // The distance between elements in units of sizeof(element). A stride of 2
  // can be used to load only the real component of a complex-valued tensor, for instance.
  uint32_t elem_stride[rank] = {1, 1};

  // Get a function pointer to the cuTensorMapEncodeTiled driver API.
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  // Create the tensor descriptor.
  CUresult res_a = cuTensorMapEncodeTiled(
    &tensor_map_a,                // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
    rank,                       // cuuint32_t tensorRank,
    A,                 // void *globalAddress,
    size,                       // const cuuint64_t *globalDim,
    stride,                     // const cuuint64_t *globalStrides,
    box_size,                   // const cuuint32_t *boxDim,
    elem_stride,                // const cuuint32_t *elementStrides,
    // Interleave patterns can be used to accelerate loading of values that
    // are less than 4 bytes long.
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    // Swizzling can be used to avoid shared memory bank conflicts.
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    // L2 Promotion can be used to widen the effect of a cache-policy to a wider
    // set of L2 cache lines.
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    // Don't set out-of-bounds elements to anything during TMA transfers
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );

  return tensor_map_a;

}

float run_conv2d_shared_mem_tma(int M, int N, double *A, double *B, int n_repeat) {

  const int BM = 32;
  const int BN = 32;

  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);

  cudaFuncSetAttribute(conv2d_shared_mem_tma<32>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);
  for (int i = 0; i < n_repeat; i++) {
    conv2d_shared_mem_tma<32>
        <<<gridDim, blockDim>>>(get_tensor_map(A, M+2, N+2, BM+2, BN+2), M, N, A, B);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(beg);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, beg, end);
  return elapsed_time;
}

float runConv2d1DBlocktiling(int M, int N, double *A, double *B, int n_repeat) {
  const uint BM = 256;
  const uint BN = 64;
  const uint TM = 16;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

  // Add extra storage to ensure 128-byte alignment
  int smem_bytes = (CEIL_DIV((BM + 2) * (BN + 2) , 16) + 1) * 16 * 8;
  // Set max shared memory for the device
  cudaFuncSetAttribute(conv2d1DBlocktiling<BM, BN, TM>, 
      cudaFuncAttributeMaxDynamicSharedMemorySize, 
      smem_bytes);
  // There are BM * BN elements to be calculated by this block 
  // by BM * BN / TM threads such that each thread calculates TM elements.
  dim3 blockDim((BM * BN) / TM);
  assert( blockDim.x < 1025);

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);
  for (int i = 0; i < n_repeat; i++) {
    conv2d1DBlocktiling<BM, BN, TM>
        <<<gridDim, blockDim, smem_bytes>>>(M, N, A, B);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(beg);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, beg, end);
  return elapsed_time;
}

float runConv2dVectorize(int M, int N, double *A, double *B, int n_repeat) {

  const uint BM = 256;
  const uint BN = 64;
  const uint TM = 16;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

  // Add extra storage to ensure 128-byte alignment
  int smem_bytes = (CEIL_DIV((BM + 2) * (BN + 2) , 16) + 1) * 16 * 8;
  // Set max shared memory for the device (200 KB = 204,800 bytes)
  cudaFuncSetAttribute(conv2dVectorize<BM, BN, TM>, 
      cudaFuncAttributeMaxDynamicSharedMemorySize, 
      smem_bytes);
  // There are BM * BN elements to be calculated by this block 
  // by BM * BN / TM threads such that each thread calculates TM elements.
  dim3 blockDim((BM * BN) / TM);
  assert( blockDim.x < 1025);
  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);
  for (int i = 0; i < n_repeat; i++) {
    conv2dVectorize<BM, BN, TM>
        <<<gridDim, blockDim, smem_bytes>>>(M, N, A, B);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(beg);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, beg, end);
  return elapsed_time;
}

float runConv2dDoubleBuffering(int M, int N, double *A, double *B, int n_repeat) {

  const uint BM = 128;
  const uint BN = 128;
  const uint ROWS_PER_BLOCK = 4;

  // Add extra storage to ensure 128-byte alignment
  // int smem_bytes = (CEIL_DIV((ROWS_PER_BLOCK + 2) * (BN + 2) , 16) + 1) * 16 * 8;
  int smem_bytes = 3072 * 8;
  // Set max shared memory for the device
  cudaFuncSetAttribute(conv2d1DBlocktiling<BM, BN, ROWS_PER_BLOCK>, 
      cudaFuncAttributeMaxDynamicSharedMemorySize, 
      smem_bytes);
  // There are BM * BN elements to be calculated by this block. 
  // This will happen in batches of ROWS_PER_BLOCK * BN such that each thread 
  // calculates 1 output element
  dim3 blockDim(ROWS_PER_BLOCK * BN);

  assert( blockDim.x < 1025);
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);
  for (int i = 0; i < n_repeat; i++)
    conv2dDoubleBuffering<BM, BN, ROWS_PER_BLOCK>
      <<<gridDim, blockDim, smem_bytes>>>(get_tensor_map(A, M+2, N+2, ROWS_PER_BLOCK+2, BN+2),
                                          get_tensor_map(B, M, N, ROWS_PER_BLOCK, BN),
                                          M, N, A, B);
  cudaEventRecord(end);
  cudaEventSynchronize(beg);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed_time, beg, end);
  return elapsed_time;
}

float run_kernel(int kernel_num, int M, int N, double *A, double *B, int n_repeat) {
  switch (kernel_num) {
  // case 0:
  //   runCuDNNFP64(M, N, A, B);
  //   break;
  case 1:
    return run_conv2d_naive(M, N, A, B, n_repeat);
  case 2:
    return run_conv2d_coalesce(M, N, A, B, n_repeat);
    break;  
  case 3:
    return run_conv2d_shared_mem_block(M, N,  A, B, n_repeat);
  case 13:
    return run_conv2d_shared_mem_tma(M, N, A, B, n_repeat);
  case 4:
    return runConv2d1DBlocktiling(M, N, A, B, n_repeat);
  // case 5:
  //   runConv2d2DBlocktiling(M, N, A, B);
  //   break;
  case 6:
    return runConv2dVectorize(M, N, A, B, n_repeat);
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
  case 11:
    return runConv2dDoubleBuffering(M, N, A, B, n_repeat);
  // case 12:
  //   runConv2dDoubleBuffering2(M, N, A, B);
  //   break;
  default:
    throw std::invalid_argument("Unknown kernel number");
  }
  
}
