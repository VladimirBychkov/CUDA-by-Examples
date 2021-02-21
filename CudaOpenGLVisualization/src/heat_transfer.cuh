#pragma once

#include "cuda_includes.cuh"
#include "cuda_gl_interop.h"

__global__ void blend_kernel(float* dst, bool dstOut);
__global__ void copy_const_kernel(float* iptr);
__device__ unsigned char value(float n1, float n2, int hue);
__global__ void float_to_color(unsigned char* optr, const float* out_src);

class HeatTransfer {
private:
	GLuint bufferObj;
	cudaGraphicsResource* resource;

	unsigned char* dev_pixels;
	float* dev_heat_src;
	float* dev_in_src;
	float* dev_out_src;
	float total_time;
	int frames;
public:
	uchar4* devPtr;

	void init_heat_src(void);
	void run_heat_transfer(unsigned char* pixels);
	void exit(void);
};
