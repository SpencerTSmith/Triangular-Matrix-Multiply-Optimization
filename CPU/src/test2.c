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
  int rid, num_ranks;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  int rows_per_rank = m0 / num_ranks;
  int extra_rows = m0 % num_ranks;

  int rows_for_this_rank = rows_per_rank + (rid < extra_rows ? 1 : 0);
  int start_row = rid * rows_per_rank + (rid < extra_rows ? rid : extra_rows);

  // Perform matrix multiplication for assigned rows
  for (int i = 0; i < rows_for_this_rank; ++i) {  // Loop over rows assigned to this rank
    for (int j = 0; j < n0; ++j) {                // Loop over columns of B / C
      float result = 0.0f;
      for (int k = 0; k < m0; ++k) {              // Loop over columns of A / rows of B
        result += A_distributed[i * m0 + k] * B_distributed[k * n0 + j];
      }
      // Store the result in the distributed C matrix
      C_distributed[i * n0 + j] = result;
    }
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

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  int rows_per_rank = m0 / num_ranks;
  int extra_rows = m0 % num_ranks;

  int rows_for_this_rank = rows_per_rank + (rid < extra_rows ? 1 : 0);

  *A_distributed=(float *)malloc(sizeof(float)*rows_for_this_rank*m0);
  *B_distributed=(float *)malloc(sizeof(float)*m0*n0);
  *C_distributed=(float *)malloc(sizeof(float)*rows_for_this_rank*n0);
}


void DISTRIBUTE_DATA_NAME(int m0, int n0,
                          float *A_sequential,
                          float *B_sequential,
                          float *A_distributed,
                          float *B_distributed) {
  int rid, num_ranks;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  int rows_per_rank = m0 / num_ranks;
  int extra_rows = m0 % num_ranks;

  // Precompute for this rank
  int rows_for_this_rank = rows_per_rank + (rid < extra_rows ? 1 : 0);
  int start_row = rid * rows_per_rank + (rid < extra_rows ? rid : extra_rows);

  if (rid == 0) {
    // Convert A from column-major to row-major (only rank 0 does this)
    float *A_row_major = (float *)malloc(sizeof(float) * m0 * m0);
    for (int i = 0; i < m0; ++i) {
      for (int j = 0; j < m0; ++j) {
        A_row_major[i * m0 + j] = A_sequential[j * m0 + i]; // Transpose to row-major
      }
    }
    // Distribute rows of A
    for (int r = 0; r < num_ranks; ++r) {
      int rows_to_send = rows_per_rank + (r < extra_rows ? 1 : 0);
      int send_start = r * rows_per_rank + (r < extra_rows ? r : extra_rows);

      if (r == 0) {
        // Use precomputed `rows_for_this_rank` and `start_row` for rank 0
        for (int i = 0; i < rows_for_this_rank * m0; ++i) {
          A_distributed[i] = A_row_major[start_row * m0 + i];
        }
      } else {
        // Send the portion for other ranks
        MPI_Send(&A_row_major[send_start * m0], rows_to_send * m0, MPI_FLOAT, r, 0, MPI_COMM_WORLD);
      }
    }

    free(A_row_major);
  } else {
    // Receive the portion of A directly
    MPI_Recv(A_distributed, rows_for_this_rank * m0, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Broadcast full matrix B to all ranks
  MPI_Bcast(B_distributed, m0 * n0, MPI_FLOAT, 0, MPI_COMM_WORLD);
}


void COLLECT_DATA_NAME( int m0, int n0,
			float *C_distributed,
			float *C_sequential )
{
  int rid, num_ranks;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  int rows_per_rank = m0 / num_ranks;
  int extra_rows = m0 % num_ranks;

  int rows_for_this_rank = rows_per_rank + (rid < extra_rows ? 1 : 0);
  int start_row = rid * rows_per_rank + (rid < extra_rows ? rid : extra_rows);

  if (rid == 0) {
    // Root process: Collect data from all ranks
    for (int r = 0; r < num_ranks; ++r) {
      int rows_to_receive = rows_per_rank + (r < extra_rows ? 1 : 0);
      int receive_start = r * rows_per_rank + (r < extra_rows ? r : extra_rows);

      if (r == 0) {
        // Copy rank 0's data directly into C_sequential
        for (int i = 0; i < rows_to_receive * n0; ++i) {
          C_sequential[receive_start * n0 + i] = C_distributed[i];
        }
      } else {
        // Receive data from rank r
        MPI_Recv(&C_sequential[receive_start * n0], rows_to_receive * n0, MPI_FLOAT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
    float *C_temp = (float *)malloc(sizeof(float) * m0 * n0);
    for (int i = 0; i < m0; ++i) {
      for (int j = 0; j < n0; ++j) {
        C_temp[j * m0 + i] = C_sequential[i * n0 + j];
      }
    }
    // Copy the transposed data back into C_sequential
    for (int i = 0; i < m0 * n0; ++i) {
      C_sequential[i] = C_temp[i];
    }
    free(C_temp);
  } else {
    // Other ranks: Send their data to the root process
    MPI_Send(C_distributed, rows_for_this_rank * n0, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  }
}


void DISTRIBUTED_FREE_NAME(int m0, int n0,
                           float *A_distributed,
                           float *B_distributed,
                           float *C_distributed) {
  int rid;
  int num_ranks;

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  free(A_distributed);
  free(B_distributed);
  free(C_distributed);

}