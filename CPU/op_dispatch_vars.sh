#!/usr/bin/env bash

######################################
# DO NOT CHANGE THIS FOLLOWING LINE: #
OP_BASELINE_FILE="./src/baseline_op.c"    #
######################################

############################################
# HOWEVER, CHANGE THESE LINES:             #
# Replace the filenames with your variants #
############################################
OP_SUBMISSION_VAR01_FILE="./src/tuned_variant01_op.c"
OP_SUBMISSION_VAR02_FILE="./src/tuned_variant02_op.c"
OP_SUBMISSION_VAR03_FILE="./src/tuned_variant03_op.c"

######################################################
# You can even change the compiler flags if you want #
######################################################
CC=mpicc
# CFLAGS="-std=c99 -O2"
CFLAGS="-std=c99 -O2 -mavx2 -mfma"

