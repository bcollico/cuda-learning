#include "ch1/example/example.h"
#include <numeric>
#include <vector>

__host__ __device__ inline float sinsum(float x, int terms) {
  float x2 = x * x;
  float term = x;   // first term of series
  float sum = term; // sum of terms so far
  for (int n = 1; n < terms; n++) {
    term *= -x2 / (2 * n * (2 * n + 1)); // build factorial
    sum += term;
  }
  return sum;
}

__global__ void gpu_sin(float *sums, int steps, int terms, float step_size) {
  // Calculate the index for this execution.
  const int step = blockIdx.x * blockDim.x + threadIdx.x;

  // Check to make sure that we're in bounds.
  if (step < steps) {
    // Get the start value as a function of the current step and the step size.
    const float x = step_size * step;
    // Compute the sum and store in the pre-allocated data vector.
    // Note that this is unsafe and assumes allocation ahead of time.
    sums[step] = sinsum(x, terms);
  }
}

double run(float start, float end, int steps, int terms, float step_size,
           int blocks, int threads) {
  // Allocate a vector on GPU for storing the components of the integral
  // approximation.

  std::vector<float> cpu_values(steps);
  float *dptr;
  const size_t num_bytes = sizeof(float) * static_cast<size_t>(steps);
  cudaMalloc(&dptr, num_bytes);

  // Execute the GPU function, passing the GPU pointer and relevant params.
  // These will get copied to each execution of the function.
  gpu_sin<<<blocks, threads>>>(dptr, steps, terms, step_size);

  // Get the output by summing the elements computed on the GPU
  cudaMemcpy(cpu_values.data(), dptr, num_bytes, cudaMemcpyDeviceToHost);
  cudaFree(dptr);
  double gpu_sum = std::accumulate(cpu_values.begin(), cpu_values.end(), 0.0,
                                   std::plus<double>());

  // Trapezoidal Rule Correction
  gpu_sum -= 0.5 * (sinsum(start, terms) + sinsum(end, terms));
  gpu_sum *= step_size;

  return gpu_sum;
}