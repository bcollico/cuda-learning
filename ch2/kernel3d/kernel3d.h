#pragma once
#include <cuda_runtime.h>

#pragma pack(push, 1)
struct Index3D final {
    int x{};
    int y{};
    int z{};

    __host__ __device__ Index3D(int x, int y, int z) : x(x), y(y), z(z) {}

    __host__ __device__ bool operator==(const Index3D& other) {
        return x == other.x && y == other.y && z == other.z;
    }

    __host__ __device__ bool operator<(const Index3D& other) const {
        return x < other.x && y < other.y && z < other.z;
    }

    __host__ __device__ int prod() const { return x * y * z; }

    __host__ __device__ int linear_idx(const Index3D grid_extents) const {
        return z * grid_extents.x * grid_extents.y + y * grid_extents.x + x;
    }
};
#pragma pack(pop)

static_assert(sizeof(Index3D) == 3 * sizeof(int));

void run_kernel(int nx, int ny, int nz, int id);

__device__ Index3D get_index_3d();

__global__ void grid_3d(int* grid_a, float* grid_b, const Index3D grid_extents,
                        int id);
