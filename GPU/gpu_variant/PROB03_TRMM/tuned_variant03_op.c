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


void compute_device( int m0, int n0,
		     float *A_distributed,
		     float *B_distributed,
		     float *C_distributed );
void allocate_device( int m0, int n0,
		      float **A_device,
		      float **B_device,
		      float **C_device );
void free_device( int m0, int n0,
		  float *A_device,
		  float *B_device,
		  float *C_device );
void collect_data_from_device( int m0, int n0,
			       float *C_device,
			       float *C_distributed );
void distribute_data_to_device( int m0, int n0,
				float *A_distributed,
				float *B_distributed,
				float *A_device,
				float *B_device );



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

  //printf("Message (Rank,Total)=(%i,%i): In compute\n", rid,num_ranks);

  if(rid == root_rid )
    {
      compute_device( m0, n0,
		      A_distributed,
		      B_distributed,
		      C_distributed );
      
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

  //printf("Message (Rank,Total)=(%i,%i): In Allocate.\n", rid,num_ranks);

  if(rid == root_rid )
    {

      /* We can sneak in the allocation of buffers on the
	 gpu in here. The trick is that we will use the 
	 pointers for the "distributed" buffers. */
      allocate_device( m0, n0,
		       A_distributed,
		       B_distributed,
		       C_distributed );
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

  //printf("Message (Rank,Total)=(%i,%i): In Distribute Data.\n", rid,num_ranks);

  if(rid == root_rid )
    {

      /*
	We are going to copy the sequential data on root=0 to the gpu. This
	routine assumes that the original matrices are packed contiguously.
      */
      
      distribute_data_to_device( m0, n0,
				 A_sequential,
				 B_sequential,
				 A_distributed,
				 B_distributed );

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

  //printf("Message (Rank,Total)=(%i,%i): In Collect.\n", rid,num_ranks);

  if(rid == root_rid )
    {

      /*
	We are going to collect the data from the gpu.
       */
      collect_data_from_device( m0, n0,
				C_distributed,
				C_sequential );

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

  //printf("Message (Rank,Total)=(%i,%i): In Free.\n", rid,num_ranks);

  if(rid == root_rid )
    {
      /* We are going to free the buffers on the gpu. */
      free_device( m0, n0,
		   A_distributed,
		   B_distributed,
		   C_distributed );

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


