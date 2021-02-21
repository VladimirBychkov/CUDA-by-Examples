#pragma once

#include "cuda_includes.cuh"

#define USE_CONSTANT_MEMORY 1

#if USE_CONSTANT_MEMORY
__global__ void ray_tracing_kernel(unsigned char* dev_pixels);
#else
__global__ void ray_tracing_kernel(Sphere* s, unsigned char* dev_pixels);
#endif

void run_constant_memomry_ray_tracing_kernel(unsigned char* pixels);