# CUDA 2D Convolution Optimization Journey

A step-by-step tutorial on optimizing 2D convolution operations in CUDA, from basic implementations to advanced techniques using the latest hardware features on NVIDIA GPUs.

## Introduction

This project explores various optimization techniques for 2D convolution operations in CUDA, a common operation in image processing, computer vision, and computational fluid dynamics. We start with a naive implementation and progressively incorporate optimization techniques including:

1. Global memory coalescing
2. Shared memory blocking
3. Block tiling (1D and 2D)
4. Vectorized memory access
5. Bank conflict resolution
6. Warp-level optimizations
7. Double buffering
8. Tensor Memory Accelerator (TMA) utilization

Each technique is implemented, benchmarked, and analyzed to understand its impact on performance.

## Why Convolution?

Convolution operations have memory access patterns similar to those found in second-order accurate finite volume CFD solvers when assembling discretized Navier-Stokes equations. By optimizing convolution, we develop skills transferable to high-performance CFD applications.

## Hardware Requirements

* CUDA-capable GPU (Hopper architecture H100 recommended for TMA kernels)
* CUDA Toolkit 12.0 or higher
* cuDNN library for baseline comparison

## Performance Comparison

<div class="chart-container">
    <canvas id="performanceChart" width="800" height="400"></canvas>
</div>

## Optimization Journey

This documentation follows our journey through increasingly sophisticated optimization techniques:

1. [Naive Implementation](implementations/naive.md)
2. [Global Memory Coalescing](implementations/coalescing.md)
3. [Shared Memory Blocking](implementations/shared_mem.md)
4. [Block Tiling Techniques](implementations/tiling.md)
5. [Vectorized Memory Access](implementations/vectorization.md)
6. [Bank Conflict Resolution](implementations/bank_conflicts.md)
7. [Warp-Level Optimization](implementations/warp_tiling.md)
8. [Double Buffering](implementations/double_buffering.md)
9. [Tensor Memory Accelerator](implementations/tma.md)

## Installation and Usage

```bash
# Clone the repository
git clone https://github.com/gantech/convolution_cuda.git
cd convolution_cuda

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make

# Run the benchmarks
./conv2d
./conv2d_padded
```

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="js/performance-chart.js"></script>
