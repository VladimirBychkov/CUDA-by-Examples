#include "mandelbrot_gpu.cuh"
#include "cuda_includes.cuh"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>


struct cuComplex {
	float   r;
	float   i;
	// cuComplex( float a, float b ) : r(a), i(b)  {}
	__device__ cuComplex(float a, float b) : r(a), i(b) {} // Fix error for calling host function from device
	__device__ float magnitude2(void) {
		return r * r + i * i;
	}
	__device__ cuComplex operator*(const cuComplex& a) {
		return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a) {
		return cuComplex(r + a.r, i + a.i);
	}
};

__device__ int julia_gpu(int x, int y, int p) {
	const float scale = 1.5;
	float jx = scale * (float)(DIM_GPU / 2 - x) / (DIM_GPU / 2);
	float jy = scale * (float)(DIM_GPU / 2 - y) / (DIM_GPU / 2);

	cuComplex c(-0.8 - float(p) / 100000, 0.156 + float(p) / 100000);
	//cuComplex c(-0.70176 * cos(float(p) / 1000), -0.3841 * sin(float(p) / 1000));
	cuComplex a(jx, jy);

	int i = 0;
	for (i = 0; i < 200; i++) {
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}

	return 1;
}

__global__ void kernel_gpu(unsigned char* ptr, int *p) {
	// map from blockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * gridDim.x * blockDim.x;

	// now calculate the value at that position
	int juliaValue = julia_gpu(x, y, *p);
	ptr[offset * 4 + 0] = 125 * juliaValue;
	ptr[offset * 4 + 1] = 255 * juliaValue;
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;
}

__host__ void run_gpu_mandelbort_kernel(unsigned char* pixels, int i) {
	int size = DIM_GPU * DIM_GPU * 4 * sizeof(unsigned char);
	unsigned char* dev_pixels;
	int* p;

	HANDLE_ERROR(cudaMalloc((void**)& dev_pixels, size));
	HANDLE_ERROR(cudaMalloc((void**)& p, sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(p, &i, sizeof(int), cudaMemcpyHostToDevice));

	dim3 grid(DIM_GPU / 10, DIM_GPU / 10);
	dim3 threads(10, 10);
	kernel_gpu<<<grid, threads>>>(dev_pixels, p);

	HANDLE_ERROR(cudaMemcpy(pixels, dev_pixels, size, cudaMemcpyDeviceToHost));
	cudaFree(dev_pixels);
}