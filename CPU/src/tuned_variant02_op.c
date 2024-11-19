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

void COMPUTE_NAME(int m0, int n0, float *A_distributed, float *B_distributed,
                  float *C_distributed) {
    int rid, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Calculate the number of rows of `A_distributed` each rank is responsible for
    int rows_per_rank = m0 / num_ranks;
    int start_row = rid * rows_per_rank;
    int end_row = (rid + 1) * rows_per_rank;

    // Ensure C_distributed is initialized to zero
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < n0; ++j) {
            C_distributed[i * m0 + j] = 0.0f;
        }
    }

    // Matrix multiplication
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < n0; ++j) {
            float res = 0.0f;
            for (int p = 0; p < m0; ++p) {
                float A_ip = A_distributed[i + p * m0];
                float B_pj = B_distributed[p + j * m0];
                res += A_ip * B_pj;
            }
            C_distributed[i * m0 + j] = res;
        }
    }
}


// Create the buffers on each node
void DISTRIBUTED_ALLOCATE_NAME(int m0, int n0, float **A_distributed,
                               float **B_distributed, float **C_distributed) {
    int rid;
    int num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Calculate the number of columns each rank will handle in a cyclic distribution
    int cols_per_rank = (n0 + num_ranks - 1) / num_ranks; // ceil(n0 / num_ranks)
    
    // Allocate memory for A_distributed on each rank based on column-cyclic distribution
    *A_distributed = (float *)malloc(sizeof(float) * m0 * cols_per_rank);
    
    // Allocate memory for B_distributed, assuming each rank has a copy of the full matrix
    *B_distributed = (float *)malloc(sizeof(float) * m0 * n0);
    
    // Allocate memory for C_distributed to hold the portion of result for each rank
    *C_distributed = (float *)malloc(sizeof(float) * m0 * cols_per_rank);
}


void DISTRIBUTE_DATA_NAME(int m0, int n0, float *A_sequential,
                          float *B_sequential, float *A_distributed,
                          float *B_distributed) {

    int rid, num_ranks;
    MPI_Status status;
    int root_rid = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Step 1: Scatter `A_sequential` in a column-cyclic manner
    if (rid == root_rid) {
        // Allocate memory to hold a temporary copy for scattering
        float *temp_A = (float *)malloc(m0 * n0 * sizeof(float));

        // Arrange `A_sequential` into a cyclic layout for scattering
        for (int col = 0; col < n0; ++col) {
            int target_rank = col % num_ranks;
            for (int row = 0; row < m0; ++row) {
                temp_A[target_rank * m0 + row] = A_sequential[col * m0 + row];
            }
        }

        // Use MPI_Scatter to distribute columns across ranks
        MPI_Scatter(temp_A, m0 * (n0 / num_ranks), MPI_FLOAT,
                    A_distributed, m0 * (n0 / num_ranks), MPI_FLOAT,
                    root_rid, MPI_COMM_WORLD);

        free(temp_A);
    } else {
        // Non-root ranks receive their parts of `A_sequential`
        MPI_Scatter(NULL, m0 * (n0 / num_ranks), MPI_FLOAT,
                    A_distributed, m0 * (n0 / num_ranks), MPI_FLOAT,
                    root_rid, MPI_COMM_WORLD);
    }

    // Step 2: Broadcast `B_sequential` to all ranks
    MPI_Bcast(B_sequential, m0 * n0, MPI_FLOAT, root_rid, MPI_COMM_WORLD);

    // Now, each rank has its part of `A_distributed` and a full copy of `B_sequential`
}


void COLLECT_DATA_NAME(int m0, int n0, float *C_distributed,
                       float *C_sequential) {
    int rid, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    int rows_per_rank = m0 / num_ranks;  // Each rank has this many rows

    if (rid == 0) {
        // Root rank: allocate buffer to receive all rows from each rank
        float *temp_buffer = (float *)malloc(m0 * n0 * sizeof(float));

        // Gather the rows from all ranks into the temp_buffer
        MPI_Gather(C_distributed, rows_per_rank * n0, MPI_FLOAT,
                   temp_buffer, rows_per_rank * n0, MPI_FLOAT,
                   0, MPI_COMM_WORLD);

        // Reorder `temp_buffer` into column-major order in `C_sequential`
        for (int j = 0; j < n0; ++j) {
            for (int i = 0; i < m0; ++i) {
                C_sequential[j * m0 + i] = temp_buffer[i * n0 + j];
            }
        }

        // Free the temporary buffer
        free(temp_buffer);
    } else {
        // Non-root ranks send their portion of C_distributed to the root
        MPI_Gather(C_distributed, rows_per_rank * n0, MPI_FLOAT,
                   NULL, rows_per_rank * n0, MPI_FLOAT,
                   0, MPI_COMM_WORLD);
    }
}

void DISTRIBUTED_FREE_NAME(int m0, int n0, float *A_distributed,
                           float *B_distributed, float *C_distributed) {
    int rid;
    MPI_Comm_rank(MPI_COMM_WORLD, &rid);

    // Free the memory on each rank, not just the root
    free(A_distributed);
    free(B_distributed);
    free(C_distributed);
}
