#pragma once

#define DIM_CPU 2000

int julia(int x, int y, int i);
void run_cpu_madelbrot_kernel(unsigned char* ptr, int i);
