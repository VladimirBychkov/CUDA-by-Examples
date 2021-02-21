#include "mandelbrot_cpu.h"


struct cuComplex {
	float   r;
	float   i;
	cuComplex(float a, float b) : r(a), i(b) {}
	float magnitude2(void) { return r * r + i * i; }
	cuComplex operator*(const cuComplex& a) {
		return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
	}
	cuComplex operator+(const cuComplex& a) {
		return cuComplex(r + a.r, i + a.i);
	}
};

int julia(int x, int y, int p) {
	const float scale = 1.5;
	float jx = scale * (float)(DIM_CPU / 2 - x) / (DIM_CPU / 2);
	float jy = scale * (float)(DIM_CPU / 2 - y) / (DIM_CPU / 2);

	cuComplex c(-0.8 - float(p) / 10000, 0.156 + float(p) / 10000);
	cuComplex a(jx, jy);

	int i = 0;
	for (i = 0; i < 200; i++) {
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}

	return 1;
}

void run_cpu_madelbrot_kernel(unsigned char* ptr, int i) {
	for (int y = 0; y < DIM_CPU; y++) {
		for (int x = 0; x < DIM_CPU; x++) {
			int offset = x + y * DIM_CPU;

			int juliaValue = julia(x, y, i);
			ptr[offset * 4 + 0] = 125 * juliaValue;
			ptr[offset * 4 + 1] = 255 * juliaValue;
			ptr[offset * 4 + 2] = 0;
			ptr[offset * 4 + 3] = 255;
		}
	}
}