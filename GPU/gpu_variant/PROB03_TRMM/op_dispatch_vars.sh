#!/usr/bin/env bash

######################################
# DO NOT CHANGE THIS FOLLOWING LINE: #
OP_BASELINE_FILE="baseline_op.c"    #
######################################

############################################
# HOWEVER, CHANGE THESE LINES:             #
# Replace the filenames with your variants #
############################################
OP_SUBMISSION_VAR01_FILE="tuned_variant01_op.c"
OP_SUBMISSION_VAR01_CUDA="tuned_variant01_op.cu"

OP_SUBMISSION_VAR02_FILE="tuned_variant02_op.c"
OP_SUBMISSION_VAR02_CUDA="tuned_variant02_op.cu"

OP_SUBMISSION_VAR03_FILE="tuned_variant03_op.c"
OP_SUBMISSION_VAR03_CUDA="tuned_variant03_op.cu"

######################################################
# You can even change the compiler flags if you want #
######################################################
CC_HOST=mpicc
CC_HOST_CFLAGS="-std=c99 -O2 -mavx2 -mfma"
CC=nvcc
#CFLAGS="-arch=all"
CFLAGS="" # nvcc flags for kepler k20m
LDFLAGS="-lstdc++ -lcudart -lm"

