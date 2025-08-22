#pragma once

#include <iostream>
#include <vector>
#include <numeric> // for std::iota

#include <cuda_runtime.h>
#include <cudnn.h>

// Helper macro for checking CUDA API calls
#define checkCuda(status)                                                                       \
    do {                                                                                        \
        if (status != cudaSuccess) {                                                            \
            std::cerr << "CUDA Error: " << cudaGetErrorString(status) << " at line "            \
                      << __LINE__ << " in file " << __FILE__ << std::endl;                      \
            exit(EXIT_FAILURE);                                                                 \
        }                                                                                       \
    } while(0)

// Helper macro for checking cuDNN API calls
#define checkCUDNN(status)                                                                      \
    do {                                                                                        \
        if (status != CUDNN_STATUS_SUCCESS) {                                                   \
            std::cerr << "cuDNN Error: " << cudnnGetErrorString(status) << " at line "         \
                      << __LINE__ << " in file " << __FILE__ << std::endl;                      \
            exit(EXIT_FAILURE);                                                                 \
        }                                                                                       \
    } while(0)

void runCuDNNFP64(int H, int W, double *d_input, double *d_output) {
    // 1. Initialize cuDNN Handle
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // 2. Define Tensor Descriptors (Input, Filter, Output)
    cudnnTensorDescriptor_t inputDescriptor, outputDescriptor;
    cudnnFilterDescriptor_t filterDescriptor;
    cudnnConvolutionDescriptor_t convolutionDescriptor;

    // Input Tensor: 1 batch, 1 channel, 5x5 image
    int N = 1, C = 1;
    checkCUDNN(cudnnCreateTensorDescriptor(&inputDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(inputDescriptor,
                                          CUDNN_TENSOR_NCHW,  // Layout: NCHW
                                          CUDNN_DATA_DOUBLE,   // Data Type
                                          N, C, H, W));       // Dimensions

    // Filter Tensor: 1 filter, 1 input channel, 3x3 kernel
    int K = 1, R = 3, S = 3;
    checkCUDNN(cudnnCreateFilterDescriptor(&filterDescriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDescriptor,
                                          CUDNN_DATA_DOUBLE,   // Data Type
                                          CUDNN_TENSOR_NCHW,  // Layout
                                          K, C, R, S));       // Dimensions

    // Convolution Descriptor (no padding, stride 1)
    int pad_h = 1, pad_w = 1;
    int stride_h = 1, stride_w = 1;
    int dilation_h = 1, dilation_w = 1;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolutionDescriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolutionDescriptor,
                                               pad_h, pad_w,
                                               stride_h, stride_w,
                                               dilation_h, dilation_w,
                                               CUDNN_CROSS_CORRELATION, // Convolution mode
                                               CUDNN_DATA_DOUBLE));      // Data type

    // Calculate output dimensions directly
    int n_out = N; // Batch size remains the same
    int c_out = K; // Output channels equals the number of filters
    int h_out = ((H - R + 2 * pad_h) / stride_h) + 1;
    int w_out = ((W - S + 2 * pad_w) / stride_w) + 1;

    // Output Tensor
    checkCUDNN(cudnnCreateTensorDescriptor(&outputDescriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(outputDescriptor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_DOUBLE,
                                          n_out, c_out, h_out, w_out));

    // 3. Allocate Host and Device Memory
    size_t inputSize = N * C * H * W * sizeof(double);
    size_t filterSize = K * C * R * S * sizeof(double);
    size_t outputSize = n_out * c_out * h_out * w_out * sizeof(double);

    // Host memory
    std::vector<double> h_filter(K * C * R * S);

    // Initialize filter data (example: a simple edge detection filter)
    h_filter = { -1.0, -1.0, -1.0,
                 -1.0,  8.0, -1.0,
                 -1.0, -1.0, -1.0 };

    // Device memory
    double *d_filter;
    checkCuda(cudaMalloc(&d_filter, filterSize));

    // Copy host data to device
    checkCuda(cudaMemcpy(d_filter, h_filter.data(), filterSize, cudaMemcpyHostToDevice));

    // 4. Choose a Convolution Algorithm (using a specific algorithm for cuDNN 9.12)
    cudnnConvolutionFwdAlgo_t algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM; // A common choice, might need experimentation
                                                                                 // For best performance, consider using the Graph API.

    // 5. Allocate Workspace (if needed)
    size_t workspaceSize = 0;
    // We might still need to call this to get the workspace size for the chosen algorithm
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                      inputDescriptor,
                                                      filterDescriptor,
                                                      convolutionDescriptor,
                                                      outputDescriptor,
                                                      algorithm,
                                                      &workspaceSize));
    void *d_workspace = nullptr;
    if (workspaceSize > 0) {
        checkCuda(cudaMalloc(&d_workspace, workspaceSize));
    }

    // 6. Execute Convolution
    double alpha = 1.0f; // Scaling factor for the result
    double beta = 0.0f;  // Scaling factor for the output (for accumulation)


    cudaEventRecord(beg);
    for (int j = 0; j < 50; j++) {
      // We don't reset dC between runs to save time
      checkCUDNN(cudnnConvolutionForward(cudnn,
                                       &alpha,
                                       inputDescriptor,
                                       d_input,
                                       filterDescriptor,
                                       d_filter,
                                       convolutionDescriptor,
                                       algorithm,
                                       d_workspace,
                                       workspaceSize,
                                       &beta,
                                       outputDescriptor,
                                       d_output));        

    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; // Convert to seconds

    long flops = 9 * H * W;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
        "(%ld).\n",
        elapsed_time / 50,
        (50 * flops * 1e-9) / elapsed_time, m);
    fflush(stdout);

    

}
