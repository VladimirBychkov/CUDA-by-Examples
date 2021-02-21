#pragma once

#include "cuda_includes.cuh"

__device__ int julia_gpu(int x, int y, int p);
__global__ void kernel_gpu(unsigned char* ptr, int* p);
__host__ void run_gpu_mandelbort_kernel(unsigned char* pixels, int i);