#include <cstdlib>
#include <iostream>
#include "ch2/kernel3d/kernel3d.h"

int main(int argc, char* argv[]) {
    int id = (argc > 1) ? std::atoi(argv[1]) : 12345;

    constexpr int NX = 512;
    constexpr int NY = 512;
    constexpr int NZ = 256;

    run_kernel(NX, NY, NZ, id);

    // Synchronize to get printout.
    cudaDeviceSynchronize();

    return 0;
}