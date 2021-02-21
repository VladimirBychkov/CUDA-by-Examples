#pragma once

#include "cuda_includes.cuh"


__global__ void kernel(int* a, int* b, int* c);
void test_cuda_streams();