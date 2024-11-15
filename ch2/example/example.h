#pragma once
#include <cuda_runtime.h>
#include <iostream>


/**
 * @brief CPU and GPU callable implementation for computing a term in the integral
 * approximation.
 * 
 * @param x start point for the approximation
 * @param terms number of terms in the taylor expansion
 */
__host__ __device__ inline float sinsum(float x, int terms);


/**
 * @brief Parallel implementation for the sin sum. Determines what step
 * to calculate based on the block index, block size, and thread index.
 * 
 * Callable from CPU.
 * 
 * @param sums Pointer to float vector on the GPU allocated to store output.
 * @param steps Total number of steps. Used for bounds checking.
 * @param terms Number of terms in the expansion to calculate.
 * @param step_size Step size to calculate the current position.
 */
__global__ void gpu_sin(float *sums, int steps, int terms, float step_size);


/**
 * @brief CPU entrypoint runner for the gpu sine integral program.
 * 
 * @param start starting value for the integral approximation
 * @param end ending value for the integral approximation
 * @param steps Number of trapezoidal steps under the curve
 * @param terms Number of terms to calculate in the taylor expansion
 * @param step_size Overparameterization. (end-start)/steps
 * @param blocks Number of warps to use?
 * @param thread Number of total trheads to use
 * 
 * @return The double-precision integral approximation.
 * 
 */
double run(float start, float end, int steps, int terms, float step_size,
           int blocks, int threads);
