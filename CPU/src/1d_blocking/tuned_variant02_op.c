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

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef COMPUTE_NAME
#define COMPUTE_NAME baseline
#endif

#ifndef DISTRIBUTE_DATA_NAME
#define DISTRIBUTE_DATA_NAME baseline_distribute
#endif

#ifndef COLLECT_DATA_NAME
#define COLLECT_DATA_NAME baseline_collect
#endif  


#ifndef DISTRIBUTED_ALLOCATE_NAME
#define DISTRIBUTED_ALLOCATE_NAME baseline_allocate
#endif


#ifndef DISTRIBUTED_FREE_NAME
#define DISTRIBUTED_FREE_NAME baseline_free
#endif




void COMPUTE_NAME( int m0, int n0,
		   float *A_distributed,
		   float *B_distributed,
		   float *C_distributed )

{
  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status  status;
  int root_rid = 0;


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
  

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

if (rid == root_rid) {
  int block_size = 32;      // Outer block size
  int sub_block_size = 16;  // Inner block size for 2D blocking

  // Initialize C_distributed matrix to 0
  for (int i0 = 0; i0 < m0; ++i0) {
    for (int j0 = 0; j0 < n0; ++j0) {
      C_distributed[i0 * cs_C + j0 * rs_C] = 0.0f;
    }
  }

  // Process blocks
  for (int i_block = 0; i_block < m0; i_block += block_size) {
    for (int j_block = i_block + 1; j_block < n0; j_block += block_size) {
      for (int p_block = 0; p_block < m0; p_block += block_size) {

        // 2D sub-blocks within each outer block
        for (int ii_block = i_block; ii_block < i_block + block_size && ii_block < m0; ii_block += sub_block_size) {
          for (int jj_block = j_block; jj_block < j_block + block_size && jj_block < n0; jj_block += sub_block_size) {
            for (int pp_block = p_block; pp_block < p_block + block_size && pp_block < m0; pp_block += sub_block_size) {

              // Inner loops for elements within each sub-block
              for (int i0 = ii_block; i0 < ii_block + sub_block_size && i0 < m0; ++i0) {
                for (int j0 = jj_block; j0 < jj_block + sub_block_size && j0 < n0; ++j0) {
                  if (j0 > i0) {  // Maintain triangular matrix constraint
                    float res = 0.0f;

                    // Unrolling the p0 loop by a factor of 4
                    int p0 = pp_block;
                    for (; p0 <= pp_block + sub_block_size - 4 && p0 < m0; p0 += 4) {
                      float A_ip1 = A_distributed[i0 + (p0 + 0) * rs_A];
                      float B_pj1 = B_distributed[(p0 + 0) + j0 * rs_B];
                      res += A_ip1 * B_pj1;

                      float A_ip2 = A_distributed[i0 + (p0 + 1) * rs_A];
                      float B_pj2 = B_distributed[(p0 + 1) + j0 * rs_B];
                      res += A_ip2 * B_pj2;

                      float A_ip3 = A_distributed[i0 + (p0 + 2) * rs_A];
                      float B_pj3 = B_distributed[(p0 + 2) + j0 * rs_B];
                      res += A_ip3 * B_pj3;

                      float A_ip4 = A_distributed[i0 + (p0 + 3) * rs_A];
                      float B_pj4 = B_distributed[(p0 + 3) + j0 * rs_B];
                      res += A_ip4 * B_pj4;
                    }

                    // Handle remaining iterations
                    for (; p0 < pp_block + sub_block_size && p0 < m0; ++p0) {
                      float A_ip = A_distributed[i0 + p0 * rs_A];
                      float B_pj = B_distributed[p0 + j0 * rs_B];
                      res += A_ip * B_pj;
                    }

                    C_distributed[i0 * cs_C + j0 * rs_C] += res;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

  else
    {
      /* STUDENT_TODO: Modify this is you plan to use more
       than 1 rank to do work in distributed memory context. */
    }
}


// Create the buffers on each node
void DISTRIBUTED_ALLOCATE_NAME( int m0, int n0,
				float **A_distributed,
				float **B_distributed,
				float **C_distributed )
{
  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status  status;
  int root_rid = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  if(rid == root_rid )
    {

      *A_distributed=(float *)malloc(sizeof(float)*m0*m0);
      *C_distributed=(float *)malloc(sizeof(float)*m0*n0);
      *B_distributed=(float *)malloc(sizeof(float)*m0*n0);
    }
  else
    {
      /*
	STUDENT_TODO: Modify this is you plan to use more
	than 1 rank to do work in distributed memory context.

	Note: In the original configuration only rank with
	rid == 0 has all of its buffers allocated.
      */

    }
}


void DISTRIBUTE_DATA_NAME( int m0, int n0,
			   float *A_sequential,
			   float *B_sequential,
			   float *A_distributed,
			   float *B_distributed )
{

  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status  status;
  int root_rid = 0;

  // Layout for sequential data
  // A is column major
  int rs_AS = m0;
  int cs_AS = 1;

  // B is column major
  int rs_BS = m0;
  int cs_BS = 1;

  // Note: Here is a perfect opportunity to change the layout
  //       of your data which has the potential to give you
  //       a sizeable performance gain.
  // Layout for distributed data
  // A is column major
  int rs_AD = m0;
  int cs_AD = 1;

  // B is column major
  int rs_BD = m0;
  int cs_BD = 1;

  
  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  if(rid == root_rid )
    {
      // Distribute the inputs
      for( int i0 = 0; i0 < m0; ++i0 )
	for( int p0 = 0; p0 < m0; ++p0 )
	{
	  A_distributed[i0 * cs_AD + p0 * rs_AD] =
	    A_sequential[i0 * cs_AS + p0 * rs_AS];
	}
  
      // Distribute the weights
      for( int p0 = 0; p0 < m0; ++p0 )
	for( int j0 = 0; j0 < n0; ++j0 )
	{
	  B_distributed[p0 * cs_BD + j0 * rs_BD] =
	    B_sequential[p0 * cs_BS + j0 * rs_BS];
	}
    }
  else
    {
      /*
	STUDENT_TODO: Modify this is you plan to use more
	than 1 rank to do work in distributed memory context.

	Note: In the original configuration only rank with
	rid == 0 has all of the necessary data for the computation.
	All other ranks have garbage in their data. This is where
	rank with rid == 0 needs to SEND data to the other nodes
	to RECEIVE the data, or use COLLECTIVE COMMUNICATION to
	distribute the data.
      */

    }
  
}



void COLLECT_DATA_NAME( int m0, int n0,
			float *C_distributed,
			float *C_sequential )
{
  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status  status;
  int root_rid = 0;

  // Layout for sequential data
  // A is column major
  // C is column major
  int rs_CS = m0;
  int cs_CS = 1;

  // Note: Here is a perfect opportunity to change the layout
  //       of your data which has the potential to give you
  //       a sizeable performance gain.
  // Layout for distributed data
  // C is column major
  int rs_CD = m0;
  int cs_CD = 1;

  
  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  if(rid == root_rid )
    {

      // Collect the output
      for( int i0 = 0; i0 < m0; ++i0 )
	for( int j0 = 0; j0 < n0; ++j0 )
	C_sequential[i0 * cs_CS + j0 * rs_CS] =
	  C_distributed[i0 * cs_CD + j0 * rs_CD];
    }
  else
    {
      /*
	STUDENT_TODO: Modify this is you plan to use more
	than 1 rank to do work in distributed memory context.

	Note: In the original configuration only rank with
	rid == 0 performs the computation and copies the
	"distributed" data to the "sequential" buffer that
	is checked by the verifier on rank rid == 0. If the
	other ranks contributed to the computation, then
	rank rid == 0 needs to RECEIVE the contributions that
	the other ranks SEND, or use COLLECTIVE COMMUNICATIONS
	for the same result.
      */

    }
  
}




void DISTRIBUTED_FREE_NAME( int m0, int n0,
			    float *A_distributed,
			    float *B_distributed,
			    float *C_distributed )
{
  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status  status;
  int root_rid = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  if(rid == root_rid )
    {

      free(A_distributed);
      free(B_distributed);
      free(C_distributed);
    }
  else
    {
      /*
	STUDENT_TODO: Modify this is you plan to use more
	than 1 rank to do work in distributed memory context.

	Note: In the original configuration only rank with
	rid == 0 allocates the "distributed" buffers for itself.
	If the other ranks were modified to allocate their own
	buffers then they need to be freed at the end.
      */

    }

}


