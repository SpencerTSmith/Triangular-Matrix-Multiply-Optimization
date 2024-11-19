#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Include your implementation
#include "test.c"

void print_matrix(const char *name, float *matrix, int rows, int cols) {
  printf("%s:\n", name);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      printf("%10.2f ", matrix[i * cols + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void sequential_matrix_multiply(float *A, float *B, float *C, int m0, int n0) {
  for (int i = 0; i < m0; ++i) {
    for (int j = 0; j < n0; ++j) {
      C[i * n0 + j] = 0.0f;
      for (int k = 0; k < m0; ++k) {
        C[i * n0 + j] += A[i * m0 + k] * B[k * n0 + j];
      }
    }
  }
}

int verify_result(float *C1, float *C2, int m0, int n0) {
  const float epsilon = 1e-5f; // Tolerance for floating-point comparison
  for (int i = 0; i < m0; ++i) {
    for (int j = 0; j < n0; ++j) {
      if (fabs(C1[i * n0 + j] - C2[i * n0 + j]) > epsilon) {
        return 0; // Mismatch found
      }
    }
  }
  return 1; // Matrices are equal
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rid, num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  int m0 = 4; // Rows and columns of A
  int n0 = 2; // Columns of B

  float A_sequential[] = {
    1, 0, 0, 0,
    2, 3, 0, 0,
    4, 5, 6, 0,
    7, 8, 9, 10
  }; // Lower triangular matrix (MxM)

  float B_sequential[] = {
    1, 2,
    3, 4,
    5, 6,
    7, 8
  }; // MxN matrix

  float C_sequential[m0 * n0];
  float *A_distributed, *B_distributed, *C_distributed;

  // Allocate distributed buffers
  DISTRIBUTED_ALLOCATE_NAME(m0, n0, &A_distributed, &B_distributed, &C_distributed);

  // Distribute data
  DISTRIBUTE_DATA_NAME(m0, n0, A_sequential, B_sequential, A_distributed, B_distributed);

  // Perform distributed computation
  COMPUTE_NAME(m0, n0, A_distributed, B_distributed, C_distributed);

  // Collect results
  float C_result[m0 * n0]; // Final gathered result
  COLLECT_DATA_NAME(m0, n0, C_distributed, C_result);

  // Print matrices at rank 0
  if (rid == 0) {
    print_matrix("Matrix A (sequential)", A_sequential, m0, m0);
    print_matrix("Matrix B (sequential)", B_sequential, m0, n0);
    print_matrix("Distributed Matrix C (result)", C_result, m0, n0);

    // Perform sequential computation for verification
    sequential_matrix_multiply(A_sequential, B_sequential, C_sequential, m0, n0);
    print_matrix("Matrix C (sequential verification)", C_sequential, m0, n0);

    // Verify distributed computation
    if (verify_result(C_result, C_sequential, m0, n0)) {
      printf("Verification PASSED: Distributed computation matches sequential computation.\n");
    } else {
      printf("Verification FAILED: Distributed computation does not match sequential computation.\n");
    }
  }

  // Free allocated memory
  DISTRIBUTED_FREE_NAME(m0, n0, A_distributed, B_distributed, C_distributed);

  MPI_Finalize();
  return 0;
}
