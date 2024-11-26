#include <cstdio>
#include "ch2/kernel3d/kernel3d.h"
#include "ch2/kernel3d/utils.h"

__device__ Index3D get_index_3d() {
    Index3D idx(blockIdx.x * blockDim.x + threadIdx.x,
                blockIdx.y * blockDim.y + threadIdx.y,
                blockIdx.z * blockDim.z + threadIdx.z);
    return idx;
}

void run_kernel(int nx, int ny, int nz, int id) {
    dim3 thread3d(32, 8, 2);    // 32*8*2 = 512
    dim3 block3d(16, 64, 128);  // 16*64*128 = 131072

    Index3D grid_extents(nx, ny, nz);

    using TypeA = int;
    using TypeB = float;

    TypeA* ptr_a;
    TypeB* ptr_b;
    cudaMalloc(&ptr_a, grid_extents.prod() * sizeof(TypeA));
    cudaMalloc(&ptr_b, grid_extents.prod() * sizeof(TypeB));

    grid_3d<<<block3d, thread3d>>>(ptr_a, ptr_b, grid_extents, id);
}

__global__ void grid_3d(int* grid_a, float* grid_b, const Index3D grid_extents,
                        int id) {
    const Index3D curr_idx = get_index_3d();
    if (!(curr_idx < grid_extents)) {
        return;
    }

    const int total_threads = block_size() * grid_size();

    const int thread_rank = thread_rank_in_grid();

    grid_a[curr_idx.linear_idx(grid_extents)] = thread_rank;
    grid_b[curr_idx.linear_idx(grid_extents)] =
        std::sqrt(static_cast<float>(thread_rank));

    if (thread_rank == id) {
        printf("array size:\t %d\n", grid_extents.prod());
        printf("thread block:\t %d x %d x %d = %d\n", blockDim.x, blockDim.y,
               blockDim.z, block_size());
        printf("thread grid:\t %d x %d x %d = %d\n", gridDim.x, gridDim.y,
               gridDim.z, grid_size());
        printf("Thread rank in grid:\t %d\n", thread_rank);
        printf("Thread rank in block:\t %d\n", thread_rank_in_block());
        printf("Block rank in grid:\t %d\n", block_rank_in_grid());
    }
}
