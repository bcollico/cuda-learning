# Introduction to GPU Kernels and Hardware
- GPU memory is 10x faster than CPU memory. GPU acceleration can be helpful for problems limited by memory bandwidth.

## First Example
- example.cu is a GPU implementation of the trapezoidal rule to evaluate the integral of sin(x).
    - sin(x) is evaluated using the Taylor series expansion up to a given number of terms
- `__host__`: Function can run on CPU
- `__device__`: Function can run on the GPU
- `__global__`: Kernel function. Launched by the host code. Can receive arguments but cannot return values.
- Threads and Blocks
    - Threads are processed in blocks. We defined the number of threads in each block and the number of thread blocks to use
    - The number of threads should be a multiple of 32
    - The number of blocks can be very large

## CPU Architecture
- CPU processing speed is directly proportional to the frequency of the Master Clock. Each pulse of the clock triggers the execution of a single CPU instruction
- The program data and machine code instructions from the compiler are stored in memory.
    - Data from memory is retrieved by aa load/save unit and transferred into a register
    - The Arithmetic Logic Unit (ALU) operates on data in the registers.
- Data and instructions progress through CPU units step by step at each clock cycle -- there is a latency between requesting data and the arrival of that data in a register
    - Typically tens of cycles on a CPU or hundreds of cycles on a GPU
    - Data latency is obfuscated by caching. When data is fetched for a register, a chunk of data around it is also retrieved and stored in a cache so that we can get the next piece of data from cache rather than memory if we are operating on sequential data units.
    - Instruction latency is obfuscated by pipelining instructions from main memory. Branch instructions break this pipeline (and hiding this issue involves complex branch prediction).
- A single chunk of data transferred in and out of a cache is called a "cache line" and usually is 64-128 bytes of contiguous main memory.

## GPU Architecture

- The basic unit is a compute-core capable of float32 and int32 operations. A single core corresponds to a single thread.
- Groups of 32 compute-cores are clustered together and called "32-core processing blocks" or "warp engines"
    - In a CUDA kernel program, threads are grouped into 32-thread groups called **warps** -- a warp is the basic execution unit in CUDA kernel programs
    - Every thread in a warp executes the same instruction on each clock cycle.
    - The warp-engine adds compute resources to it's cores. E.g. Special Function Units (SFUs) for trig/exp/etc. and float64 processing units
- Warp-engines are grouped into two- or four-engine groups called **Symmetric Multiprocessors** (SMs)
    - An SM typically has 128 compute-cores.
    - The threads in a CUDA kernel program are grouped into fixed-size thread blocks,where each block runs on a single SM -- threads within the same block can communicate with each other e.g. via shared memory
    - Additional resources shared between warp-engines:
        - shared memory (~100k)
        - register file (~64k)
        - L1 cache (~48k)
- A number of SMs make up the final GPU. 
    - GTX1080 has 20SMs -> 2560 compute cores
    - L2 cache is shared by all of the SMs (~4g)

### Memory Types
- **Main memory**: accessible from CPU (relatively slow, via the PCI bus) and GPU.Preserved between kernel calls.
    - NB: memory transfers from host to device can be asynchronous with kernel execution. E.g. simultaneously sending the next input while running model inference.
    - Contains Texture and Constant memory
- **Constant Memory**: 64k of main memory reserved for data constants. Dedicated cache so that all threads can read from the same memory location very quickly.
    - Variables can be placed in constant memory with the `constant` or `restricted` keywords or via compiler optimization
    - Explicit usage of constant memory probably isn't necessary
- **Texture Memory**: Array storage (up to three dimensions). Optimized for 2D array addressing. Fast N-linear interpolation via `texND` functions.
- **Local Memory**: Private to each thread. Used for memory that can't be fully contained in a the registers. L1 & L2 caches still apply here.
- **Register File**: Each SM has 64k 32-bit registers shared by thread blocks on the the SM.
    - The upper limit of concurrent warps on a single SM is 64 (~2k threads). 
    - If the compiler allocates more than 32 registers then the maximum number of thread blocks running on the SM is reduced
    - **GPU Occupancy**: The number of threads resident out of the total number of resident threads available. SM occupancy will be reduced in the above example.
- **Shared Memory**: Fastest way for threads (within a thread block) to communicate. 32k-64k memory for an SM. Each concurrent thread block on oan SM gets the same size memory block -- defined at compile time or kernel launch time.
    - The amount of memory allocated to each thread block can also limit occupancy. Max num concurrent thread blocks is bounded by `total_shared_memory / shared_memory_per_block`.

- Similar to CPUs, data latency of main memory access is obfuscated by L1 and L2 caches as well as high occupancy.
    - The most effective cache use is achieved when the 32 threads in a warp access 32-bit variables in up to 32 adjacent memory locations starting aligned with a 32-word memory boundary -- this is known as **memory coalescing**.
    - On modern GPUs, caching is more forgiving, so sticking to adjacent threads operating on adjacent memory is probably sufficient.

## GPU-program Design
- On CPU, choosing `n_threads == n_cores` is sufficient to keep the CPU occupied, but on GPU, you want to choose `n_threads >> n_cores` so that data latencies can be abstracted away by hardware optimizations
    - While waiting for data, the processors can switch between threads to make use of whatever data becomes available for any running thread.
    - While each SM has 2-4 warps (64-128 threads) running at any particular time, it can have a much large number of suspended or **resident threads**.
- Since `n_threads > n_cores`, the threads need to be run in **waves**. The number of threads in a wave is the `n_resident_threads / n_symmetric_multiprocessors`; and the number of waves that run for a given kernel is `n_threads / n_thread_per_wave`.
- The number of SMs, resident threads, and therefore threads per wave vary based on the GPU architecture.
- A group of threads that are batched together and run in the same SM is called a **thread block**.
    - The size of the thread block should be a multiple of the warp size (i.e. 32 threads), up to the hardware maximum
    - Threads in the same block can communicate using shared memory, but threads in different blocks cannot communicate during kernel execution.
    - A CUDA kernel launch configuration consists of the thread block size and the number of thread blocks
        - This is called "launching a grid", where **grid size** is the number of thread blocks.
            - NB: In this text, `threads` and `blocks` will be used to denote the number of threads per block and the number of thread blocks, respectively. 
- Now that we know about resident threads, **Occupancy** is the number of threads actually resident in the SM over the number of possible resident threads.
    - 100% occupancy means that complete waves are running on the SMs. See shared memory discussion above for reasons why 100% occupancy might not be achieved.


### CUDA Built-In Variables
Variable ranges are shown for the 1D case of this Chapter's example.
- `threadIdx`: Thread rank in the thread block. `[0, threads-1]`
- `blockIdx`: Thread block rank in the grid. `[0, blocks-1]`
- `blockDim`: Number of threads in a single thread block. `blockDim == threads`
- `gridDim`: Number of thread blocks in the grid. `gridDim == blocks`
- `warpSize`: Number of threads in a warp.

