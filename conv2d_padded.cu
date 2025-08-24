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
__global__ void conv2d_shared_mem_block(int M, int N, 
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
  __shared__ double As[(BLOCKSIZE + 2) * (BLOCKSIZE + 2)];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // Set up async pipeline with default 2-stage depth
  auto pipe = cuda::make_pipeline();

  // Each threadblock cooperatively loads the entire block in a single TMA operation
  // This is much more efficient than issuing many small transfers
  
  // Since we need to coordinate across the block, we'll use thread 0 to issue the copy
  if (threadIdx.x == 0) {
    // Acquire a slot in the pipeline
    pipe.producer_acquire();
    
    // Calculate source and destination addresses
    const double* src_ptr = &A[cRow * BLOCKSIZE * (N+2) + cCol * BLOCKSIZE];
    double* dst_ptr = &As[0];
    
    // Create a struct to define the 2D memory layout
    struct MemcpyParams {
      // Size in elements (not bytes)
      size_t rows = BLOCKSIZE + 2;
      size_t cols = BLOCKSIZE + 2;
      
      // Stride in elements (not bytes)
      size_t src_stride = N + 2;
      size_t dst_stride = BLOCKSIZE + 2;
    } params;
    
    // Issue a single 2D copy for the entire tile
    // Each row has (BLOCKSIZE+2) elements, and we copy (BLOCKSIZE+2) rows
    cuda::memcpy_async(dst_ptr, src_ptr, 
                      params.cols * sizeof(double), params.rows,
                      params.dst_stride * sizeof(double),
                      params.src_stride * sizeof(double),
                      pipe);
    
    // Commit the copy operation to the pipeline
    pipe.producer_commit();
  }
  
  // Wait for all memory operations to complete
  // Note that even though only one thread issued the copy, all threads need to wait
  pipe.consumer_wait();
  block.sync();  // Ensure all threads see the loaded data
  
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
      <<<gridDim, blockDim>>>(M, N, dA, dB);
    cudaGetLastError(); // Check for async errors during kernel run      
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
