# CUDA Convolution Learning Project

This repository documents my journey learning CUDA programming by implementing convolution operations and benchmarking them against cuDNN performance. I'm reusing the template by [Simon Boehm](https://github.com/NVIDIA/cuda-samples).

## Motivation
I've chosen 2D convolution as my focus because its memory access patterns closely resemble those found in second-order accurate finite volume CFD (Computational Fluid Dynamics) solvers when assembling discretized Navier-Stokes equations.

By mastering these convolution operations in CUDA, I aim to develop transferable skills applicable to high-performance CFD applications.