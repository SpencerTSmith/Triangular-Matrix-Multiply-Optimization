#!/usr/bin/env bash

######################################
# DO NOT CHANGE THIS FOLLOWING LINE: #
OP_BASELINE_FILE="./src/baseline_op.c"    #
######################################

############################################
# HOWEVER, CHANGE THESE LINES:             #
# Replace the filenames with your variants #
############################################
OP_SUBMISSION_VAR01_FILE="./src/16_simd_blocking.c"
OP_SUBMISSION_VAR02_FILE="./src/16_simd.c"
OP_SUBMISSION_VAR03_FILE="./src/16_simd_blocking_omp.c"

######################################################
# You can even change the compiler flags if you want #
######################################################
CC=mpicc
# CFLAGS="-std=c99 -O2"
CFLAGS="-g -std=c99 -O2 -mavx2 -mfma -fopenmp"

