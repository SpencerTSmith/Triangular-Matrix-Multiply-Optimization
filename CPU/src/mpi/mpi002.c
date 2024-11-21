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

'''In this version, I tried with mpi_scatterv to distribute the data. When debugging, we noted that 
   the data is not well distributed.'''

void COMPUTE_NAME(int m0, int n0, float *A_distributed, float *B_distributed, float *C_distributed) {
    int rid, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    int rows_per_rank = m0 / num_ranks;
    int extra_rows = m0 % num_ranks;
    int rows_for_this_rank = rows_per_rank + (rid < extra_rows ? 1 : 0);

    // Initialize C_distributed to zero (to ensure no garbage values)
    for (int i = 0; i < rows_for_this_rank * n0; ++i) {
        C_distributed[i] = 0.0f;
    }

    // Perform matrix multiplication for the assigned rows
    for (int i = 0; i < rows_for_this_rank; ++i) { // Loop over rows assigned to this rank
        for (int j = 0; j < n0; ++j) {             // Loop over columns of B / C
            float result = 0.0f;                   // Initialize result for C[i, j]
            for (int k = 0; k < m0; ++k) {         // Loop over columns of A / rows of B
                result += A_distributed[i * m0 + k] * B_distributed[k * n0 + j];
            }
            C_distributed[i * n0 + j] = result;    // Store result in the distributed C matrix
        }
    }
}

void DISTRIBUTED_ALLOCATE_NAME(int m0, int n0,
                               float **A_distributed,
                               float **B_distributed,
                               float **C_distributed) {
    int rid, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    int rows_per_rank = m0 / num_ranks;
    int extra_rows = m0 % num_ranks;

    int rows_for_this_rank = rows_per_rank + (rid < extra_rows ? 1 : 0);

    *A_distributed = (float *)malloc(sizeof(float) * rows_for_this_rank * m0);
    *B_distributed = (float *)malloc(sizeof(float) * m0 * n0); // Allocate the full matrix for B
    *C_distributed = (float *)malloc(sizeof(float) * rows_for_this_rank * n0);
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
    int rows_for_this_rank = rows_per_rank + (rid < extra_rows ? 1 : 0);

    int *sendcounts = NULL;
    int *displs = NULL;

    if (rid == 0) {
        sendcounts = (int *)malloc(sizeof(int) * num_ranks);
        displs = (int *)malloc(sizeof(int) * num_ranks);

        int offset = 0;
        for (int r = 0; r < num_ranks; ++r) {
            int rows_to_send = rows_per_rank + (r < extra_rows ? 1 : 0);
            sendcounts[r] = rows_to_send * m0; // Number of elements to send
            displs[r] = offset;
            offset += sendcounts[r];
        }
    }
    MPI_Scatterv(A_sequential, sendcounts, displs, MPI_FLOAT,
                 A_distributed, rows_for_this_rank * m0, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    if (rid == 0) {
        free(sendcounts);
        free(displs);
    }
    if (rid == 0) {
        for (int i = 0; i < m0 * n0; ++i) {
            B_distributed[i] = B_sequential[i];
        }
    }
    MPI_Bcast(B_distributed, m0 * n0, MPI_FLOAT, 0, MPI_COMM_WORLD);

    //Some debugging
    // printf("Rank %d: A_distributed (rows for this rank):\n", rid);
    // for (int i = 0; i < rows_for_this_rank; ++i) {
    //     for (int j = 0; j < m0; ++j) {
    //         printf("%10.2f ", A_distributed[i * m0 + j]);
    //     }
    //     printf("\n");
    // }

    // printf("Rank %d: B_distributed (full matrix):\n", rid);
    // for (int i = 0; i < m0; ++i) {
    //     for (int j = 0; j < n0; ++j) {
    //         printf("%10.2f ", B_distributed[i * n0 + j]);
    //     }
    //     printf("\n");
    // }
}

void COLLECT_DATA_NAME(int m0, int n0,
                       float *C_distributed,
                       float *C_sequential) {
    int rid, num_ranks;

    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    int rows_per_rank = m0 / num_ranks;
    int extra_rows = m0 % num_ranks;
    int rows_for_this_rank = rows_per_rank + (rid < extra_rows ? 1 : 0);

    if (rid == 0) {
        // Root process: Collect data from all ranks
        int offset = 0; // Tracks where to place received rows in C_sequential
        for (int r = 0; r < num_ranks; ++r) {
            int rows_to_receive = rows_per_rank + (r < extra_rows ? 1 : 0);

            if (r == 0) {
                // Copy rank 0's data directly
                for (int i = 0; i < rows_to_receive * n0; ++i) {
                    C_sequential[offset + i] = C_distributed[i];
                }
            } else {
                // Receive data from rank r
                MPI_Recv(&C_sequential[offset], rows_to_receive * n0, MPI_FLOAT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            offset += rows_to_receive * n0; // Update offset for the next rank
        }
        //Here, I am trying to make it such that the sequential is stored col major
        float *C_temp = (float *)malloc(sizeof(float) * m0 * n0);
        for (int i = 0; i < m0; ++i) {
            for (int j = 0; j < n0; ++j) {
                C_temp[j * m0 + i] = C_sequential[i * n0 + j];
            }
        }
        for (int i = 0; i < m0 * n0; ++i) {
            C_sequential[i] = C_temp[i];
        }
        free(C_temp);
    } else {
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
