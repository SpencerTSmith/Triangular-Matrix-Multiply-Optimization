#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef COMPUTE_NAME
#define COMPUTE_NAME mpi_trmm_compute
#endif

#ifndef DISTRIBUTE_DATA_NAME
#define DISTRIBUTE_DATA_NAME mpi_trmm_distribute
#endif  

#ifndef COLLECT_DATA_NAME
#define COLLECT_DATA_NAME mpi_trmm_collect
#endif  

#ifndef DISTRIBUTED_ALLOCATE_NAME
#define DISTRIBUTED_ALLOCATE_NAME mpi_trmm_allocate
#endif

#ifndef DISTRIBUTED_FREE_NAME
#define DISTRIBUTED_FREE_NAME mpi_trmm_free
#endif

void compute_counts_displs(int m0, int num_ranks, int *counts, int *displs) {
    int rows_per_rank = m0 / num_ranks;
    int remainder = m0 % num_ranks;
    int offset = 0;
    for (int i = 0; i < num_ranks; i++) {
        counts[i] = rows_per_rank + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += counts[i];
    }
}

void COMPUTE_NAME(int m0, int n0,
                  float *A_distributed,
                  float *B_distributed,
                  float *C_distributed)
{
    int rid, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Compute counts and displacements
    int *counts = (int *)malloc(num_ranks * sizeof(int));
    int *displs = (int *)malloc(num_ranks * sizeof(int));
    compute_counts_displs(m0, num_ranks, counts, displs);

    int local_m = counts[rid];
    int start_row = displs[rid];

    // Matrix strides (assuming column-major order)
    int rs_A = m0;
    int cs_A = 1;

    int rs_B = m0;
    int cs_B = 1;

    int rs_C_local = local_m;
    int cs_C_local = 1;

    for (int i0_local = 0; i0_local < local_m; ++i0_local)
    {
        int i0 = start_row + i0_local;
        for (int j0 = 0; j0 < n0; ++j0)
        {
            float res = 0.0f;
            for (int p0 = 0; p0 <= i0; ++p0)
            {
                float A_ip = A_distributed[i0 * cs_A + p0 * rs_A];
                float B_pj = B_distributed[p0 * cs_B + j0 * rs_B];
                res += A_ip * B_pj;
            }
            C_distributed[i0_local * cs_C_local + j0 * rs_C_local] = res;
        }
    }

    free(counts);
    free(displs);
}

// Create the buffers on each node
void DISTRIBUTED_ALLOCATE_NAME(int m0, int n0,
                               float **A_distributed,
                               float **B_distributed,
                               float **C_distributed)
{
    int rid, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Compute counts and displacements
    int *counts = (int *)malloc(num_ranks * sizeof(int));
    int *displs = (int *)malloc(num_ranks * sizeof(int));
    compute_counts_displs(m0, num_ranks, counts, displs);

    int local_m = counts[rid];

    // All ranks allocate their portion of C
    *C_distributed = (float *)malloc(sizeof(float) * local_m * n0);

    // All ranks allocate full A and B (since we will broadcast them)
    *A_distributed = (float *)malloc(sizeof(float) * m0 * m0);
    *B_distributed = (float *)malloc(sizeof(float) * m0 * n0);

    free(counts);
    free(displs);
}

void DISTRIBUTE_DATA_NAME(int m0, int n0,
                          float *A_sequential,
                          float *B_sequential,
                          float *A_distributed,
                          float *B_distributed)
{
    int rid;
    MPI_Comm_rank(MPI_COMM_WORLD, &rid);

    if (rid == 0)
    {
        // Copy sequential data into distributed buffers on root
        memcpy(A_distributed, A_sequential, sizeof(float) * m0 * m0);
        memcpy(B_distributed, B_sequential, sizeof(float) * m0 * n0);
    }

    // Broadcast A and B to all ranks
    MPI_Bcast(A_distributed, m0 * m0, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B_distributed, m0 * n0, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void COLLECT_DATA_NAME(int m0, int n0,
                       float *C_distributed,
                       float *C_sequential)
{
    int rid, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rid);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Compute counts and displacements
    int *counts = (int *)malloc(num_ranks * sizeof(int));
    int *displs = (int *)malloc(num_ranks * sizeof(int));
    compute_counts_displs(m0, num_ranks, counts, displs);

    // Adjust counts and displacements for gathering C
    int *recvcounts = (int *)malloc(num_ranks * sizeof(int));
    int *displsC = (int *)malloc(num_ranks * sizeof(int));
    for (int i = 0; i < num_ranks; i++)
    {
        recvcounts[i] = counts[i] * n0;
        displsC[i] = displs[i] * n0;
    }

    // Gather the computed parts of C from all ranks
    MPI_Gatherv(C_distributed, counts[rid] * n0, MPI_FLOAT,
                C_sequential, recvcounts, displsC, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    free(counts);
    free(displs);
    free(recvcounts);
    free(displsC);
}

void DISTRIBUTED_FREE_NAME(int m0, int n0,
                           float *A_distributed,
                           float *B_distributed,
                           float *C_distributed)
{
    free(A_distributed);
    free(B_distributed);
    free(C_distributed);
}
