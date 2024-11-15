
#include "ch1/example/example.h"

int main() {
  // Arguments
  constexpr float start = 0.0f;
  constexpr float end = 3.14159265358979323;
  constexpr int steps = 10000000;
  constexpr int terms = 1000;
  constexpr int threads = 256;
  constexpr int blocks = (steps + threads - 1) / threads;  // round up
  constexpr float step_size = (end - start) / (steps - 1); // NB n-1

  double gpu_sum = run(start, end, steps, terms, step_size, blocks, threads);

  std::cout << "gpu sum: " << gpu_sum << std::endl;
  return 0;
}
