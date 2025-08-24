#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>

#define cudaCheck2(err) (cudaCheck(err, __FILE__, __LINE__))

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

template <const int BLOCKSIZE>
__global__ void conv2d_shared_mem_block(const __grid_constant__ CUtensorMap tensor_map_a, 
                                        int M, int N, 
                                        const double *A, double *B) {
  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;
  
  // Need to include cooperative groups for block synchronization
  namespace cg = cooperative_groups;
  auto block = cg::this_thread_block();

  double filter[9] = {-1.0, -1.0, -1.0,
          -1.0, 8.0, -1.0,
          -1.0, -1.0, -1.0};

  // allocate buffer for current block including padding in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ alignas(128) double As[CEIL_DIV((BLOCKSIZE + 2) * (BLOCKSIZE + 2) , 128) * 128];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // Initialize shared memory barrier with the number of threads participating in the barrier.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;

  if (threadIdx.x == 0) {
    // Initialize barrier. All `blockDim.x` threads in block participate.
    init(&bar, blockDim.x);
    // Make initialized barrier visible in async proxy.
    cde::fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  barrier::arrival_token token;
  if (threadIdx.x == 0) {
    // Initiate bulk tensor copy.
    cde::cp_async_bulk_tensor_2d_global_to_shared(&As, &tensor_map_a, cRow, cCol, bar);
    // Arrive on the barrier and tell how many bytes are expected to come in.
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(As));
  } else {
    // Other threads just arrive.
    token = bar.arrive();
  }
  // Wait for the data to have arrived.
  bar.wait(std::move(token));

  double tmp = 0.0;
  for (int fi = -1 ; fi < 2; fi++) {
    for (int fj = -1; fj < 2; fj++) { 
      tmp += As[(threadRow + fi + 1) * (BLOCKSIZE + 2) + (threadCol + fj + 1)] * filter[(fi + 1) * 3 + (fj + 1)];
    }
  }

  B[(cRow * BLOCKSIZE + threadRow )* N + cCol * BLOCKSIZE + threadCol] = tmp;

}

void randomize_matrix(double *mat, int N) {
  // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
  // precision is too low and the same random number is generated.
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {
    double tmp = (double)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

int main(int argc, char **argv) {

  long M=4096, N=4096;

  double *A = nullptr, *B = nullptr, 
        *B_ref = nullptr; // host matrices
  double *dA = nullptr, *dB = nullptr, 
        *dB_ref = nullptr; // device matrices

  A = (double *)malloc(sizeof(double) * (M+2) * (N+2));
  B = (double *)malloc(sizeof(double) * M * N);
  B_ref = (double *)malloc(sizeof(double) * M * N);

  randomize_matrix(A, (M+2) * (N+2));

  cudaCheck2(cudaMalloc((void **)&dA, sizeof(double) * (M+2) * (N+2)));
  cudaCheck2(cudaMalloc((void **)&dB, sizeof(double) * M * N));
  cudaCheck2(cudaMalloc((void **)&dB_ref, sizeof(double) * M * N));

  cudaCheck2(cudaMemcpy(dA, A, sizeof(double) * (M+2) * (N+2),
                       cudaMemcpyHostToDevice));
  cudaCheck2(cudaMemcpy(dB, B, sizeof(double) * M * N,
                       cudaMemcpyHostToDevice));
  cudaCheck2(cudaMemcpy(dB_ref, B_ref, sizeof(double) * M * N,
                       cudaMemcpyHostToDevice));


  CUtensorMap tensor_map_a{};
  // rank is the number of dimensions of the array.
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {M+2, N+2};
  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  uint64_t stride[rank - 1] = {(N+2) * sizeof(double)};
  // The box_size is the size of the shared memory buffer that is used as the
  // destination of a TMA transfer.
  uint32_t box_size[rank] = {34, 34};
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
    dA,                 // void *globalAddress,
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

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
              
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  cudaFuncSetAttribute(conv2d_shared_mem_block<32>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared);

  cudaEventRecord(beg);
  for (int j = 0; j < 50; j++) {                       
    conv2d_shared_mem_block<32>
      <<<gridDim, blockDim>>>(tensor_map_a, M, N, dA, dB);
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, beg, end);
  elapsedTime /= 1000.0; // Convert to seconds
  printf("Elapsed time: %.2f s\n", elapsedTime);

  long flops = 9 * M * N;
  printf(
      "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
      "(%ld).\n",
      elapsedTime / 50,
      (50 * flops * 1e-9) / elapsedTime, M);
  fflush(stdout);

  // Clean up
  free(A);
  free(B);
  free(B_ref);
  cudaCheck2(cudaFree(dA));
  cudaCheck2(cudaFree(dB));
  cudaCheck2(cudaFree(dB_ref));

  return 0;
}
