# Introduction to GPU Kernels and Hardware
- GPU memory is 10x faster than CPU memory. GPU acceleration can be helpful for problems limited by memory bandwidth.
## First Example
- example.cu is a GPU implementation of the trapezoidal rule to evaluate the integral of sin(x).
    - sin(x) is evaluated using the Taylor series expansion up to a given number of terms
- `__host__`: Function can run on CPU
- `__device__`: Function can run on the GPU
- `__global__`: Kernel function. Launched by the host code.Can receive arguments be cannot return values.

## CPU Architecture
- CPU processing speed is directly proportional to the frequency of the Master Clock. Each pulse of the clock triggers the execution of a single CPU instruction
- The program data and machine code instructions from the compiler are stored in memory.
    - Data from memory is retrieved by a load/save unit and transferred into a register
    - The Arithmetic Logic Unit (ALU) operates on data in the registers.
- Data and instructions progress through CPU units step by step at each clock cycle -- there is a latency between requesting data and the arrival of that data in a register
    - Typically tens of cycles on a CPU or hundreds of cycles on a GPU
    - Data latency is obfuscated by caching. When data is fetched for a register, a chunk of data around it is also retrieved and stored in a cache so that we can get the next piece of data from cache rather than memory if we are operating on sequential data units.
    - Instruction latency is obfuscated by pipelining instructions from main memory. Branch instructures break this pipeline (and hiding this issue involves complex branch prediction).
- A single chunk of data transfered in and out of a cache is called a "cache line" and usually is 64-128 bytes of contigous main memory.

## GPU Architecture

- The basic unit is a compute-core capable of float32 and int32 operations
- Groups of 32 compute-cores are clustered together and called "32-core processing blocks" or "warp engines"
    - In a CUDA kernel program, threads are grouped into 32-thread groups called "warps" -- a warp is the basic execution unit in CUDA kernel programs
    - Every thread in a warp executes the same instruction on each clock cycle.
    - The warp-engine adds compute resources to it's cores. E.g. Special Function Units (SFUs) for trig/exp/etc. and float64 processing units
- Warp-engines are grouped into two- or four-engine groups called Symmetric Multiprocessors (SMs)
    - An SM typically has 128 compute-cores.
    - The threads in a CUDA kernal program are grouped into fixed-size thread blocks,where each block runs on a single SM -- threads within the same block can communicate with each other e.g. via shared memory
    - Additional resources shared between warp-engines:
        - shared memory (~100k)
        - register file (~64k)
        - L1 cache (~48k)
- A number of SMs make up the final GPU. 
    - GTX1080 has 20SMs -> 2560 compute cores
    - L2 cache is shared by all of the SMs (~4g)