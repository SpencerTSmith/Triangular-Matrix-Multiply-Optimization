#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef COMPUTE_NAME
#define COMPUTE_NAME mpi_trmm
#endif

#ifndef DISTRIBUTE_DATA_NAME
#define DISTRIBUTE_DATA_NAME mpi_distribute
#endif

#ifndef COLLECT_DATA_NAME
#define COLLECT_DATA_NAME mpi_collect
#endif  

#ifndef DISTRIBUTED_ALLOCATE_NAME
#define DISTRIBUTED_ALLOCATE_NAME mpi_allocate
#endif

#ifndef DISTRIBUTED_FREE_NAME
#define DISTRIBUTED_FREE_NAME mpi_free
#endif

void COMPUTE_NAME(int m0, int n0,
                  float *A_distributed,
                  float *B_distributed,
                  float *C_distributed) {

    int rid, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    int rows_per_rank = m0 / num_ranks;
    int remainder_rows = m0 % num_ranks;

    // Determine the number of rows and starting row for each rank
    int local_rows = (rid < remainder_rows) ? rows_per_rank + 1 : rows_per_rank;
    int start_row = (rid < remainder_rows) ? rid * (rows_per_rank + 1) : rid * rows_per_rank + remainder_rows;

    // Initialize C_distributed to zero
    for (int i = 0; i < local_rows * n0; i++) {
        C_distributed[i] = 0.0f;
    }

    // Perform the matrix multiplication for assigned rows of C
    for (int i0 = 0; i0 < local_rows; i0++) {          // Loop over each row in this rank’s portion
        int row = start_row + i0;                      // Actual row index in A

        for (int j0 = 0; j0 < n0; j0++) {              // Loop over columns in B and C
            float res = 0.0f;

            for (int p0 = 0; p0 <= row; p0++) {        // Only loop over non-zero elements in the row of A
                float A_ip = A_distributed[i0 * (i0 + 1) / 2 + p0];  // Access non-zero element in lower triangular A
                float B_pj = B_distributed[p0 * n0 + j0];            // Access element in B

                res += A_ip * B_pj;
            }

            // Store the result in C_distributed
            C_distributed[i0 * n0 + j0] = res;
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
    int remainder_rows = m0 % num_ranks;

    // Determine the number of rows and starting row for each rank
    int local_rows = (rid < remainder_rows) ? rows_per_rank + 1 : rows_per_rank;
    int start_row = (rid < remainder_rows) ? rid * (rows_per_rank + 1) : rid * rows_per_rank + remainder_rows;

    // Calculate total non-zero elements for local_rows rows starting from start_row
    int non_zero_elements_A = 0;
    for (int i = 0; i < local_rows; i++) {
        non_zero_elements_A += (start_row + i + 1);  // i-th row contains start_row + i + 1 elements
    }

    if (rid == 0) {
        // Root rank allocates entire A, B, and C for gathering purposes
        *A_distributed = (float *)malloc(sizeof(float) * m0 * (m0 + 1) / 2);  // Full lower triangular A
        *B_distributed = (float *)malloc(sizeof(float) * m0 * n0);             // Full matrix B
        *C_distributed = (float *)malloc(sizeof(float) * m0 * n0);             // Full matrix C
    } else {
        // Other ranks allocate only their portions
        *A_distributed = (float *)malloc(sizeof(float) * non_zero_elements_A);  // Only non-zero elements for A
        *B_distributed = (float *)malloc(sizeof(float) * m0 * n0);              // Full matrix B
        *C_distributed = (float *)malloc(sizeof(float) * local_rows * n0);      // Only required rows of C
    }
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
    int remainder_rows = m0 % num_ranks;

    // Determine the number of rows and starting row for each rank
    int local_rows = (rid < remainder_rows) ? rows_per_rank + 1 : rows_per_rank;
    int start_row = (rid < remainder_rows) ? rid * (rows_per_rank + 1) : rid * rows_per_rank + remainder_rows;

    if (rid == 0) {
        // Rank 0 distributes relevant non-zero parts of A to other ranks
        int offset = 0;  // Tracks position in A_sequential for non-zero elements

        for (int rank = 0; rank < num_ranks; rank++) {
            int rows_to_send = (rank < remainder_rows) ? rows_per_rank + 1 : rows_per_rank;
            int send_pos = 0;
            float *temp_buffer = (float *)malloc(sizeof(float) * rows_to_send * (rows_to_send + 1) / 2);

            for (int i = 0; i < rows_to_send; i++) {
                int row = start_row + i;  // Current row in A
                for (int j = 0; j <= row; j++) {
                    if (rank == 0) {
                        // Rank 0 copies its portion directly into A_distributed
                        A_distributed[send_pos++] = A_sequential[row * m0 + j];
                    } else {
                        // Copy into buffer for sending to other ranks
                        temp_buffer[send_pos++] = A_sequential[row * m0 + j];
                    }
                }
            }

            if (rank != 0) {
                // Send the specific non-zero elements of A to each rank
                MPI_Send(temp_buffer, send_pos, MPI_FLOAT, rank, 0, MPI_COMM_WORLD);
            }

            free(temp_buffer);
            offset += send_pos;  // Move the offset to the next set of rows
        }
    } else {
        // Receive the non-zero elements of A for this rank
        int recv_size = local_rows * (local_rows + 1) / 2;  // Number of non-zero elements
        MPI_Recv(A_distributed, recv_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Broadcast full matrix B to all ranks
    MPI_Bcast(B_distributed, m0 * n0, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void COLLECT_DATA_NAME(int m0, int n0,
                       float *C_distributed,
                       float *C_sequential) {

    int rid, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    int rows_per_rank = m0 / num_ranks;
    int remainder_rows = m0 % num_ranks;

    // Determine the number of rows for each rank
    int local_rows = (rid < remainder_rows) ? rows_per_rank + 1 : rows_per_rank;
    int start_row = (rid < remainder_rows) ? rid * (rows_per_rank + 1) : rid * rows_per_rank + remainder_rows;

    if (rid == 0) {
        // Root rank gathers all parts of C_distributed from each rank into C_sequential
        int offset = 0;  // Tracks the starting position for each rank in C_sequential

        // Copy Rank 0's own portion directly
        for (int i = 0; i < local_rows; i++) {
            for (int j = 0; j < n0; j++) {
                C_sequential[(start_row + i) * n0 + j] = C_distributed[i * n0 + j];
            }
        }

        // Receive the portions from other ranks
        for (int rank = 1; rank < num_ranks; rank++) {
            int recv_rows = (rank < remainder_rows) ? rows_per_rank + 1 : rows_per_rank;
            int recv_size = recv_rows * n0;  // Total elements to receive from this rank

            MPI_Recv(&C_sequential[offset * n0], recv_size, MPI_FLOAT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            offset += recv_rows;
        }
    } else {
        // Each non-root rank sends its computed portion of C_distributed to the root rank
        int send_size = local_rows * n0;
        MPI_Send(C_distributed, send_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
}

void DISTRIBUTED_FREE_NAME(int m0, int n0,
                           float *A_distributed,
                           float *B_distributed,
                           float *C_distributed) {

    int rid, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Free memory allocated on each rank
    free(A_distributed);
    free(B_distributed);
    free(C_distributed);
}