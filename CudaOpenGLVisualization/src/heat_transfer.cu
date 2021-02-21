#include <GL/glew.h>
#include "heat_transfer.cuh"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED   0.25f

//texture<float, cudaTextureType2D>  tex_heat_src;
//texture<float, cudaTextureType2D>  tex_in;
//texture<float, cudaTextureType2D>  tex_out;

__global__ void blend_kernel(float* out, const float* in) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// no textures
	int left = offset - 1;
	int right = offset + 1;
	if (x == 0) left++;
	if (x == DIM_GPU - 1) right--;

	int top = offset - DIM_GPU;
	int bottom = offset + DIM_GPU;
	if (y == 0) top += DIM_GPU;
	if (y == DIM_GPU - 1) bottom -= DIM_GPU;

	/*float   t, l, c, r, b;
	if (dstOut) {
		t = tex2D(tex_in, x, y - 1);
		l = tex2D(tex_in, x - 1, y);
		c = tex2D(tex_in, x, y);
		r = tex2D(tex_in, x + 1, y);
		b = tex2D(tex_in, x, y + 1);
	}
	else {
		t = tex2D(tex_out, x, y - 1);
		l = tex2D(tex_out, x - 1, y);
		c = tex2D(tex_out, x, y);
		r = tex2D(tex_out, x + 1, y);
		b = tex2D(tex_out, x, y + 1);
	}*/
	//dst[offset] = c + SPEED * (t + b + r + l - 4 * c);

	out[offset] = in[offset] + SPEED * (in[top] + in[bottom] + in[left] + in[right] - 4 * in[offset]);
}

__global__ void copy_const_kernel(float* iptr, const float* heat_src) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	/*float c = tex2D(tex_heat_src, x, y);
	if (c != 0)
		iptr[offset] = c;*/
	if (heat_src[offset] != 0) iptr[offset] = heat_src[offset];
}

__device__ unsigned char value(float n1, float n2, int hue) {
	if (hue > 360)      hue -= 360;
	else if (hue < 0)   hue += 360;

	if (hue < 60)
		return (unsigned char)(255 * (n1 + (n2 - n1) * hue / 60));
	if (hue < 180)
		return (unsigned char)(255 * n2);
	if (hue < 240)
		return (unsigned char)(255 * (n1 + (n2 - n1) * (240 - hue) / 60));
	return (unsigned char)(255 * n1);
}

__global__ void float_to_color(unsigned char* optr,	const float* out_src) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float l = out_src[offset];
	float s = 1;
	int h = (180 + (int)(360.0f * out_src[offset])) % 360;
	float m1, m2;

	if (l <= 0.5f)
		m2 = l * (1 + s);
	else
		m2 = l + s - l * s;
	m1 = 2 * l - m2;

	optr[offset * 4 + 0] = value(m1, m2, h + 120);
	optr[offset * 4 + 1] = value(m1, m2, h);
	optr[offset * 4 + 2] = value(m1, m2, h - 120);
	optr[offset * 4 + 3] = 255;
}

__global__ void float_to_color(uchar4* optr, const float* out_src) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float l = out_src[offset];
	float s = 1;
	int h = (180 + (int)(360.0f * out_src[offset])) % 360;
	float m1, m2;

	if (l <= 0.5f)
		m2 = l * (1 + s);
	else
		m2 = l + s - l * s;
	m1 = 2 * l - m2;

	optr[offset].x = value(m1, m2, h + 120);
	optr[offset].y = value(m1, m2, h);
	optr[offset].z = value(m1, m2, h - 120);
	optr[offset].w = 255;
}

void HeatTransfer::init_heat_src(void) {
	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM_GPU * DIM_GPU * 4, NULL, GL_DYNAMIC_DRAW_ARB);

	HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone));

	int size = DIM_GPU * DIM_GPU * 4 * sizeof(unsigned char);

	HANDLE_ERROR(cudaMalloc((void**)& dev_pixels, size));
	HANDLE_ERROR(cudaMalloc((void**)& dev_in_src, size));
	HANDLE_ERROR(cudaMalloc((void**)& dev_out_src, size));
	HANDLE_ERROR(cudaMalloc((void**)& dev_heat_src, size));

	/*const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	size_t pitch;
	cudaMallocPitch((void**)& dev_heat_src, &pitch, DIM_GPU * sizeof(int), DIM_GPU);
	const textureReference* tex_heat_ref_ptr;
	cudaGetTextureReference(&tex_heat_ref_ptr, &tex_heat_src);
	HANDLE_ERROR(cudaBindTexture2D(NULL, tex_heat_ref_ptr, dev_heat_src, &desc, DIM_GPU, DIM_GPU,
		pitch));

	cudaMallocPitch((void**)& dev_in_src, &pitch, DIM_GPU * sizeof(int), DIM_GPU);
	const textureReference* tex_in_ref_ptr;
	cudaGetTextureReference(&tex_in_ref_ptr, &tex_in);
	HANDLE_ERROR(cudaBindTexture2D(NULL, tex_in_ref_ptr, dev_in_src, &desc, DIM_GPU, DIM_GPU,
		pitch));

	cudaMallocPitch((void**)& dev_out_src, &pitch, DIM_GPU * sizeof(int), DIM_GPU);
	const textureReference* tex_out_ref_ptr;
	cudaGetTextureReference(&tex_out_ref_ptr, &tex_out);
	HANDLE_ERROR(cudaBindTexture2D(NULL, tex_out_ref_ptr,	dev_out_src, &desc, DIM_GPU, DIM_GPU,
		pitch));*/

	float* temp = (float*)malloc(size);
	for (int i = 0; i < DIM_GPU * DIM_GPU; i++) {
		temp[i] = 0;
		int x = i % DIM_GPU;
		int y = i / DIM_GPU;
		if ((x > 300) && (x < 600) && (y > 310) && (y < 601))
			temp[i] = MAX_TEMP;
		if ((x > 400) && (x < 700) && (y > 810) && (y < 901))
			temp[i] = MAX_TEMP;
	}
	temp[DIM_GPU * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
	temp[DIM_GPU * 700 + 100] = MIN_TEMP;
	temp[DIM_GPU * 300 + 300] = MIN_TEMP;
	temp[DIM_GPU * 200 + 700] = MIN_TEMP;
	for (int y = 800; y < 900; y++) {
		for (int x = 400; x < 500; x++) {
			temp[x + y * DIM_GPU] = MIN_TEMP;
		}
	}
	HANDLE_ERROR(cudaMemcpy(dev_heat_src, temp, size, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_in_src, temp, size,	cudaMemcpyHostToDevice));
	free(temp);

}

void HeatTransfer::run_heat_transfer(unsigned char* pixels) {
	size_t  size;

	HANDLE_ERROR(cudaGraphicsMapResources(1, &resource, NULL));
	HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)& devPtr, &size, resource));

	//int size = DIM_GPU * DIM_GPU * 4 * sizeof(unsigned char);
	cudaEvent_t start, stop;

	dim3 blocks(DIM_GPU / 10, DIM_GPU / 10);
	dim3 threads(10, 10);

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	volatile bool dst_out = true;
	for (int i = 0; i < 50; i++) {
		/*float* in, * out;
		if (dst_out) {
			in = dev_in_src;
			out = dev_out_src;
		}
		else {
			out = dev_in_src;
			in = dev_out_src;
		}
		copy_const_kernel<<<blocks, threads>>>(in);
		blend_kernel<<<blocks, threads>>>(out, dst_out);
		dst_out = !dst_out;*/

		copy_const_kernel<<<blocks, threads>>>(dev_in_src, dev_heat_src);
		blend_kernel<<<blocks, threads>>>(dev_out_src, dev_in_src);

		float* temp = dev_in_src;
		dev_in_src = dev_out_src;
		dev_out_src = temp;
	}

	//float_to_color<<<blocks, threads>>>(dev_pixels, dev_in_src);
	float_to_color<<<blocks, threads>>>(devPtr, dev_in_src);

	//HANDLE_ERROR(cudaMemcpy(pixels, dev_pixels, size,	cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float   elapsed_time;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));
	total_time += elapsed_time;
	frames++;
	printf("Average Time per frame:  %3.1f ms\n",	total_time / frames);

	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource, NULL));
}

void HeatTransfer::exit(void) {
	/*cudaUnbindTexture(tex_in);
	cudaUnbindTexture(tex_out);
	cudaUnbindTexture(tex_heat_src);*/
	HANDLE_ERROR(cudaFree(dev_in_src));
	HANDLE_ERROR(cudaFree(dev_out_src));
	HANDLE_ERROR(cudaFree(dev_heat_src));
}
