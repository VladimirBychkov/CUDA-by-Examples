/*
Visual Studio 2019 settings to build and run.

C/C++
	General - Additional Include Derictories: $(SolutionDir)Dependencies\GLFW\include;$(SolutionDir)Dependencies\GLEW\include;...
	Preprocessor - Preprocessor Definistions: GLEW_STATIC;...

CUDA C/C++
	Device - Code Generation: compute_50,sm_50

Linker
	General - Additional Library Derictories: $(SolutionDir)Dependencies\GLFW\lib-vc2019;$(SolutionDir)Dependencies\GLEW\lib\Release\x64;...
	Input - Additional Dependencies: glew32s.lib;glfw3.lib;opengl32.lib;cudart_static.lib;...
*/

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include "mandelbrot_cpu.h"
#include "mandelbrot_gpu.cuh"
#include "constant_memory_ray_tracing.cuh"
#include "heat_transfer.cuh"
#include "cuda_streams.cuh"
#include "cuda_gl_interop.h"

#define MANDELBROT_ON_GPU 0
#define MANDELBROT_ON_CPU 0
#define CONSTANT_MEMORY_RAY_TRACING 0
#define HEAT_TRANSFER 0
#define CUDA_STREAMS 1

#if MANDELBROT_ON_GPU || HEAT_TRANSFER || CONSTANT_MEMORY_RAY_TRACING || CUDA_STREAMS
const int width = DIM_GPU;
const int height = DIM_GPU;
#endif

#if MANDELBROT_ON_CPU
const int width = DIM_CPU;
const int height = DIM_CPU;
#endif


int main(void)
{
	unsigned char *pixels;
	pixels = (unsigned char*)malloc(width * height * 4 * sizeof(unsigned char));
	if (!pixels) {
		return -1;
	}

	int i = 0;

	cudaDeviceProp prop;
	int dev;

	cudaMemset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 5;
	prop.minor = 0;
	HANDLE_ERROR(cudaChooseDevice(&dev, &prop));

	GLFWwindow* window;

	/* Initialize the library */
	if (!glfwInit())
		return -1;

	/* Create a windowed mode window and its OpenGL context */
	window = glfwCreateWindow(width, height, "Hello World", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return -1;
	}

	/* Make the window's context current */
	glfwMakeContextCurrent(window);

	if (glewInit() != GLEW_OK) {
		std::cout << "Can't init glew!\n";
	}
	std::cout << glGetString(GL_VERSION) << "\n";

#if HEAT_TRANSFER
	HeatTransfer heat_transfer;
	heat_transfer.init_heat_src();
#endif

	/* Loop until the user closes the window */
	while (!glfwWindowShouldClose(window)) {
		/* Render here */
		glClear(GL_COLOR_BUFFER_BIT);
		
#if MANDELBROT_ON_CPU
		run_cpu_madelbrot_kernel(pixels, i);
#endif

#if MANDELBROT_ON_GPU
		run_gpu_mandelbort_kernel(pixels, i);
#endif

#if CONSTANT_MEMORY_RAY_TRACING
		run_constant_memomry_ray_tracing_kernel(pixels);
#endif

#if HEAT_TRANSFER
		heat_transfer.run_heat_transfer(pixels);
#endif

#if CUDA_STREAMS
		test_cuda_streams();
#endif 

		glClearColor(0.0, 0.0, 0.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT);

#if HEAT_TRANSFER
		glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
#else
		glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
#endif

		/* Swap front and back buffers */
		glfwSwapBuffers(window);

		/* Poll for and process events */
		glfwPollEvents();
		
		i++;
	}

	free(pixels);
#if HEAT_TRANSFER
	heat_transfer.exit();
#endif	

	glfwTerminate();
	return 0;
}