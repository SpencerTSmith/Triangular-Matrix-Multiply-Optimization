// -*- mode: c++ -*-
/*
  This is the baseline implementation of a Triangular Matrix Times Matrix
  Multiplication  (TRMM)

  C = AB, where
  A is an MxM lower triangular (A_{i,p} = 0 if p > i) Matrix. It is indexed by i0 and p0
  B is an MxN matrix. It is indexed by p0 and j0.
  C is an MxN matrix. It is indexed by i0 and j0.
  
  
  Parameters:

  m0 > 0: dimension
  n0 > 0: dimension



  float* A_sequential: pointer to original A matrix data
  float* A_distributed: pointer to the input data that you have distributed across
  the system

  float* C_sequential:  pointer to original output data
  float* C_distributed: pointer to the output data that you have distributed across
  the system

  float* B_sequential:  pointer to original weights data
  float* B_distributed: pointer to the weights data that you have distributed across
  the system

  Functions:

  DISTRIBUTED_ALLOCATE_NAME(...): Allocate the distributed buffers.
  DISTRIBUTE_DATA_NAME(...): takes the sequential data and distributes it across the system.
  COMPUTE_NAME(...): Performs the stencil computation.
  COLLECT_DATA_NAME(...): Collect the distributed output and combine it back to the sequential
  one for testing.
  DISTRIBUTED_FREE_NAME(...): Free the distributed buffers that were allocated


  - richard.m.veras@ou.edu

*/

#include <stdio.h>
#include <stdlib.h>


// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}


extern "C" void compute_device( int m0, int n0,
				float *A_distributed,
				float *B_distributed,
				float *C_distributed );


extern "C" void allocate_device( int m0, int n0,
				 float **A_device,
				 float **B_device,
				 float **C_device );

extern "C" void free_device( int m0, int n0,
			     float *A_device,
			     float *B_device,
			     float *C_device );

extern "C" void collect_data_from_device( int m0, int n0,
					  float *C_device,
					  float *C_distributed );

extern "C" void distribute_data_to_device( int m0, int n0,
					   float *A_distributed,
					   float *B_distributed,
					   float *A_device,
					   float *B_device );

/* This is the GPU kernel. */
__global__ void cuda_trmm(int m0, int n0, float *A_device, float *B_device, float *C_device){
  /*
    student_todo: this is where the majority of the work will happen. 
   */

  /*
    Using the convention that row_stride (rs) is the step size you take going down a row,
    column stride (cs) is the step size going down the column.
  */
  // A is column major
  int rs_A = m0;
  int cs_A = 1;

  // B is column major
  int rs_B = m0;
  int cs_B = 1;

  // C is column major
  int rs_C = m0;
  int cs_C = 1;

  int id = blockDim.x * blockIdx.x + threadIdx.x;

  // Only 1 thread will every run this!
  // A better implementation would be to split the work
  // across many threads.
  //if(id == 0)
  if(id < n0)
    {
      int j0 = id;

      //for( int j0 = 0; j0 < n0; ++j0 )
	{
	  for( int i0 = 0; i0 < m0; ++i0 )
	    {
	      float res = 0.0f;
	      for( int p0 = 0; p0 < m0; ++p0 )
		{
		  if( j0 > i0 )
		    {
		      float A_ip = A_device[i0 * cs_A + p0 * rs_A];
		      float B_pj = B_device[p0 * cs_B + j0 * rs_B];
		      
		      res += A_ip*B_pj;
		    }
		}

	      C_device[i0 * cs_C + j0 * rs_C] = res;
	    }
	}

    }
}


void allocate_device( int m0, int n0,
		      float **A_device,
		      float **B_device,
		      float **C_device )
{
  int bytes_A = m0*m0*sizeof(float);
  int bytes_B = m0*n0*sizeof(float);
  int bytes_C = m0*n0*sizeof(float);

  /**/
  
  //printf("GPU Allocate: ");
  gpuErrchk(cudaMalloc(A_device, bytes_A));
  gpuErrchk(cudaMalloc(B_device, bytes_B));
  gpuErrchk(cudaMalloc(C_device, bytes_C));
  //printf("Done\n");
}

void distribute_data_to_device( int m0, int n0,
				float *A_distributed,
				float *B_distributed,
				float *A_device,
				float *B_device )
{
  int bytes_A = m0*m0*sizeof(float);
  int bytes_B = m0*n0*sizeof(float);

  //printf("GPU Distribute: ");

  /* student_todo: you can modify this code if you want to lay out the data in
                   the gpu in a particular way. */

  gpuErrchk(cudaMemcpy(A_device, A_distributed, bytes_A, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(B_device, B_distributed, bytes_B, cudaMemcpyHostToDevice));
  //printf("Done\n");
}


void collect_data_from_device( int m0, int n0,
			       float *C_device,
			       float *C_distributed )
{
  int bytes_C = m0*n0*sizeof(float);

  //printf("GPU Collect: ");

  /* student_todo: you can modify this code if you want to lay out the data in
                   the gpu in a particular way. */

  gpuErrchk(cudaMemcpy(C_distributed, C_device, bytes_C, cudaMemcpyDeviceToHost));
  //printf("Done\n");
}

void free_device( int m0, int n0,
		  float *A_device,
		  float *B_device,
		  float *C_device )
{
  // Free GPU memory
  //printf("GPU Free: ");
  gpuErrchk(cudaFree(A_device));
  gpuErrchk(cudaFree(B_device));
  gpuErrchk(cudaFree(C_device));
  //printf("Done\n");
}

void compute_device( int m0, int n0,
		     float *A_distributed,
		     float *B_distributed,
		     float *C_distributed )

{

  //printf("GPU Compute: ");

  // student_todo: you will sweep through these knobs
  // These are knobs you can tune
  int threads_per_block = n0;
  //int threads_per_block = 1;
  int blocks_per_grid   = 1;

  // Run the kernel
  cuda_trmm<<<blocks_per_grid,threads_per_block>>>(m0, n0, A_distributed, B_distributed, C_distributed);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  //  printf("Done\n");
}
