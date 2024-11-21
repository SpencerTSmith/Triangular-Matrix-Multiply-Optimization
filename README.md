# PDN Lab 03
This one focused on speeding up a simple matrix multiply operation, with a twist, multiplication is done in a triangular pattern. This means that every i-th row will be multiplied with only j-th > i-th column.

## CPU
In this folder: optimizations through instruction level parallelism with SIMD intrinsic kernels, distributed memory parallelism with MPI, shared memory parallelism with OpenMP, as well as cache aware memory access patterns (blocking, transposing matrix, etc.).

## GPU
In this folder: a CUDA program with a traditional matrix multiply algorithm where every kernel does a dot product.

