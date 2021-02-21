#pragma once

#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

#include <math.h>

#define INF 2e10f
#define DIM_GPU 1000

void HandleError(cudaError_t err, const char* file, int line);
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

struct Sphere {
	float r, g, b;
	float radius;
	float x, y, z;

	__device__ float hit(float ox, float oy, float* n) {
		float dx = ox - x;
		float dy = oy - y;
		if (dx * dx + dy * dy < radius * radius) {
			float dz = sqrtf(radius * radius - dx * dx - dy * dy);
			*n = dz / sqrtf(radius * radius);
			return dz + z;
		}

		return -INF;
	}
};
