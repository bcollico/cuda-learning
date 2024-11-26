#pragma once
#include <cuda_runtime.h>

__device__ __inline__ int block_size() {
    return blockDim.x * blockDim.y * blockDim.z;
}

__device__ __inline__ int grid_size() {
    return gridDim.x * gridDim.y * gridDim.z;
}

__device__ __inline__ int thread_rank_in_block() {
    return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
}

__device__ __inline__ int block_rank_in_grid() {
    return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
}

__device__ __inline__ int thread_rank_in_grid() {
    return block_rank_in_grid() * block_size() + thread_rank_in_block();
}