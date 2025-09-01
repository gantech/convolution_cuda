#pragma once
#include <cuda_runtime.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

void cudaCheck(cudaError_t error, const char *file,
               int line); // CUDA error check
void CudaDeviceInfo();    // print CUDA information

void range_init_matrix(double *mat, int N);
void randomize_matrix(double *mat, double * mat_nonpad, int M, int N);
void zero_init_matrix(double *mat, int N);
void copy_matrix(const double *src, double *dest, int N);
void print_matrix(const double *A, int M, int N, std::ofstream &fs);
bool verify_matrix(double *mat1, double *mat2, int N);

float get_current_sec();                        // Get the current moment
float cpu_elapsed_time(float &beg, float &end); // Calculate time difference

float run_kernel(int kernel_num, int m, int n, double *A, double * B, int n_repeat=1);
