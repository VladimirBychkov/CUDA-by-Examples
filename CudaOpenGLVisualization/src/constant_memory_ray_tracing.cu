#include "constant_memory_ray_tracing.cuh"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define rnd(x) (x * rand() / RAND_MAX)
#define SPHERES 20

#if USE_CONSTANT_MEMORY
__constant__ Sphere s[SPHERES];
#else
Sphere *s;
#endif

#if USE_CONSTANT_MEMORY
__global__ void ray_tracing_kernel(unsigned char* dev_pixels) {
#else
__global__ void ray_tracing_kernel(Sphere * s, unsigned char* dev_pixels) {
#endif
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float ox = (x - DIM_GPU / 2);
	float oy = (y - DIM_GPU / 2);

	float r = 0, g = 0, b = 0;
	float maxz = -INF;
	for (int i = 0; i < SPHERES; i++) {
		float n;
		float t = s[i].hit(ox, oy, &n);
		if (t > maxz) {
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
		}
	}

	dev_pixels[offset * 4 + 0] = (int)(r * 255);
	dev_pixels[offset * 4 + 1] = (int)(g * 255);
	dev_pixels[offset * 4 + 2] = (int)(b * 255);
	dev_pixels[offset * 4 + 3] = 255;
}

void run_constant_memomry_ray_tracing_kernel(unsigned char* pixels) {
	int size = DIM_GPU * DIM_GPU * 4 * sizeof(unsigned char);
	unsigned char* dev_pixels;

	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	HANDLE_ERROR(cudaMalloc((void**)& dev_pixels, size));
	HANDLE_ERROR(cudaMalloc((void**)& s, sizeof(Sphere) * SPHERES));

	Sphere* temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
	for (int i = 0; i < SPHERES; i++) {
		temp_s[i].r = rnd(1.0f);
		temp_s[i].g = rnd(1.0f);
		temp_s[i].b = rnd(1.0f);
		temp_s[i].x = rnd(1000.0f) - 500;
		temp_s[i].y = rnd(1000.0f) - 500;
		temp_s[i].z = rnd(1000.0f) - 500;
		temp_s[i].radius = rnd(100.0f) + 20;
	}

#if USE_CONSTANT_MEMORY
	HANDLE_ERROR(cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES));
#else
	HANDLE_ERROR(cudaMemcpy(s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice));
#endif
	free(temp_s);

	dim3 grid(DIM_GPU / 10, DIM_GPU / 10);
	dim3 threads(10, 10);

#if USE_CONSTANT_MEMORY
	ray_tracing_kernel<<<grid, threads>>>(dev_pixels);
#else
	ray_tracing_kernel<<<grid, threads>>>(s, dev_pixels);
#endif

	HANDLE_ERROR(cudaMemcpy(pixels, dev_pixels, size, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float   elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

	printf("Time to generate:  %3.1f ms\n", elapsedTime);

	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));

	cudaFree(dev_pixels);
	cudaFree(s);
}