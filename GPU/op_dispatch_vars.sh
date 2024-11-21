#!/usr/bin/env bash

######################################
# DO NOT CHANGE THIS FOLLOWING LINE: #
OP_BASELINE_FILE="./src/baseline_op.c"    #
######################################

############################################
# HOWEVER, CHANGE THESE LINES:             #
# Replace the filenames with your variants #
############################################
OP_SUBMISSION_VAR01_FILE="./src/good_cuda.c"
OP_SUBMISSION_VAR01_CUDA="./src/good_cuda.cu"

OP_SUBMISSION_VAR02_FILE="./src/gooder_cuda.c"
OP_SUBMISSION_VAR02_CUDA="./src/gooder_cuda.cu"

OP_SUBMISSION_VAR03_FILE="./src/as_good_cuda.c"
OP_SUBMISSION_VAR03_CUDA="./src/as_good_cuda.cu"

######################################################
# You can even change the compiler flags if you want #
######################################################
CC_HOST=mpicc
CC_HOST_CFLAGS="-std=c99 -O2 -mavx2 -mfma"
CC=nvcc
#CFLAGS="-arch=all"
CFLAGS="" # nvcc flags for kepler k20m
LDFLAGS="-lstdc++ -lcudart -lm"

