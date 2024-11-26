#pragma once
#include <cuda_runtime.h>

#pragma pack(push, 1)
/// @brief Simple container for storing/comparing 3D XYZ indices.
struct Index3D final {
    int x{};
    int y{};
    int z{};

    /// @brief Constructor with XYZ input.
    __host__ __device__ Index3D(int x, int y, int z) : x(x), y(y), z(z) {}

    /// @brief Compare two Index3D coordinates for strict less-then inequality.
    /// @param other Other coordinate to check against.
    /// @return Bool true of this coordinate has XYZ all less than the other coordinate.
    __host__ __device__ bool operator<(const Index3D& other) const {
        return x < other.x && y < other.y && z < other.z;
    }

    /// @brief Compute the product of the XYZ values.
    __host__ __device__ int prod() const { return x * y * z; }

    /// @brief Compute the linear index for this coordinate.
    /// @param grid_extents The 3D extents of the grid.
    /// @return x-major linear index for this 3D coordinate.
    __host__ __device__ int linear_idx(const Index3D grid_extents) const {
        return z * grid_extents.x * grid_extents.y + y * grid_extents.x + x;
    }
};
#pragma pack(pop)

static_assert(sizeof(Index3D) == 3 * sizeof(int));

/**
 * @brief Calculate the 3D index of this thread in the Grid.
 * @return Index3D object containing the XYZ index of this thread in the grid.
 */
__device__ Index3D get_index_3d();

/**
 * @brief Main entrypoint function from the host. Launches the kernel with a specified 3D block size
 * and number of blocks
 * @param nx 3D grid data x-dimension
 * @param ny 3D grid data y-dimension
 * @param nz 3D grid data z-dimension
 * @param id Thread rank in grid (linear index) to print debug message at.
 */
void run_kernel(int nx, int ny, int nz, int id);

/**
 * @brief 3D grid data processing kernel. Each thread stores it's global rank in the integer array
 * and the sqrt of it's rank in the float array. Prints debug at `id`.
 * 
 * @param grid_a Pointer to integer grid data.
 * @param grid_b Pointer to float grid data.
 * @param grid_extents Index3D instance containing the XYZ bounds for the grid data.
 * @param id Thread rank in grid to print debug message at.
 */
__global__ void grid_3d(int* grid_a, float* grid_b, const Index3D grid_extents,
                        int id);
