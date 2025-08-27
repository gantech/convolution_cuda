#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <runner_padded.cuh>
#include <vector>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Please select a kernel (range 0 - 12, 0 for NVIDIA cuBLAS)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // get kernel number
  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > 12) {
    std::cerr << "Please enter a valid kernel number (0-12)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // get environment variable for device
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

  // print some device info
  // CudaDeviceInfo();

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // cuBLAS FLOPs ceiling is reached at 8192
  std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};

  long m, n, max_size;
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "Max size: " << max_size << std::endl;
  
  double *A = nullptr, *B = nullptr, 
        *B_ref = nullptr; // host matrices
  double *dA = nullptr, *dB = nullptr, 
        *dB_ref = nullptr; // device matrices

  A = (double *)malloc(sizeof(double) * (max_size+2) * (max_size+2));
  B = (double *)malloc(sizeof(double) * max_size * max_size);
  B_ref = (double *)malloc(sizeof(double) * max_size * max_size);

  randomize_matrix(A, max_size+2, max_size+2);

  cudaCheck(cudaMalloc((void **)&dA, sizeof(double) * (max_size+2) * (max_size+2)));
  cudaCheck(cudaMalloc((void **)&dB, sizeof(double) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB_ref, sizeof(double) * max_size * max_size));

  cudaCheck(cudaMemcpy(dA, A, sizeof(double) * (max_size+2) * (max_size+2),
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(double) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB_ref, B_ref, sizeof(double) * max_size * max_size,
                       cudaMemcpyHostToDevice));

  int repeat_times = 50;
  for (int size : SIZE) {
    m = n = size;

    std::cout << "dimensions(m=n) " << m << std::endl;
    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (kernel_num != 0) {
      run_kernel(0, m, n, dA, dB_ref); // cuDNN
      run_kernel(kernel_num, m, n, dA, dB); // Executes the kernel, modifies the result matrix
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
      cudaMemcpy(B, dB, sizeof(double) * m * n, cudaMemcpyDeviceToHost);
      cudaMemcpy(B_ref, dB_ref, sizeof(double) * m * n, cudaMemcpyDeviceToHost);

      if (!verify_matrix(B_ref, B, m * n)) {
        std::cout
            << "Failed to pass the correctness verification against NVIDIA "
               "cuBLAS."
            << std::endl;
        if (m <= 128) {
          std::cout << " Logging faulty output into " << errLogFile << "\n";
          std::ofstream fs;
          fs.open(errLogFile);
          fs << "A:\n";
          print_matrix(A, m+2, n+2, fs);
          fs << "B:\n";
          print_matrix(B, m, n, fs);
          fs << "Should:\n";
          print_matrix(B_ref, m, n, fs);
        }
        exit(EXIT_FAILURE);
      }
    }

    if (kernel_num == 0)
      run_kernel(kernel_num, m, n, dA, dB);
    else {
      cudaEventRecord(beg);
      for (int j = 0; j < repeat_times; j++) {
        // We don't reset dC between runs to save time
        run_kernel(kernel_num, m, n, dA, dB);
      }
      cudaEventRecord(end);
      cudaEventSynchronize(beg);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&elapsed_time, beg, end);
      elapsed_time /= 1000.; // Convert to seconds

      long flops = 9 * m * n;
      printf(
          "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
          "(%ld).\n",
          elapsed_time / repeat_times,
          (repeat_times * flops * 1e-9) / elapsed_time, m);
      fflush(stdout);
      // make dB and dB_ref equal again (we modified dB while calling our kernel
      // for benchmarking)
      cudaCheck(cudaMemcpy(dB, dB_ref, sizeof(double) * m * n,
                          cudaMemcpyDeviceToDevice));
    }
  }

  // Free up CPU and GPU space
  free(A);
  free(B);
  free(B_ref);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dB_ref);

  return 0;
};
