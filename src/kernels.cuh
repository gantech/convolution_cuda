#pragma once


#include "kernels/0_cudnn.cuh"
#include "kernels/1_naive_padded.cuh"
#include "kernels/2_kernel_global_mem_coalesce_padded.cuh"
#include "kernels/3_kernel_shared_mem_blocking.cuh"
#include "kernels/3_kernel_shared_mem_tma.cuh"
#include "kernels/4_kernel_1D_blocktiling.cuh"
#include "kernels/6_kernel_vectorize.cuh"
#include "kernels/11_kernel_double_buffering.cuh"

