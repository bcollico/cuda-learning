#pragma once
#include <cuda_runtime.h>

/// @brief Compute the number of threads in a single 3D thread block.
__device__ __inline__ int block_size() {
    return blockDim.x * blockDim.y * blockDim.z;
}

/// @brief Compute the total number of threads blocks in the 3D grid.
__device__ __inline__ int grid_size() {
    return gridDim.x * gridDim.y * gridDim.z;
}

/// @brief Compute the linear thread idx within the block. Linear index is computed
/// x-y-z (x-major) order, such that z is the slowest changing.
__device__ __inline__ int thread_rank_in_block() {
    return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
}

/// @brief Compute the block linear idx in the grid. Linear index is computed
/// x-y-z (x-major) order such that z is the slowest changing.
__device__ __inline__ int block_rank_in_grid() {
    return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
}

/// @brief Compute the thread linear index in the grid.
__device__ __inline__ int thread_rank_in_grid() {
    return block_rank_in_grid() * block_size() + thread_rank_in_block();
}