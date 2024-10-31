# This Makefile orchestrates the building, verification and timing of
# your implementations.
#
#
# Run modes:
# 1. execute on machine with mpi already installed and loaded.
#    make all-local
#
# 2. If you are on a system with mpi loaded through module, then
#    you can run locally on that node for debugging by typing:
#
#    module load OpenMPI
#    make all-local
#
#
# 3. execute remotely on Schooner.
#    make all-schooner
#

# Increments to use in the local tests
# To expedite development you can adjust these,
# but for performance plots max is the minimum
# size I want to see. (HINT: for a large number
# of ranks you need to up that number to hit
# higher performance numbers)
MIN=64
#MAX=8192 # good large number
MAX=4096 # medium size
#MAX=1024 # small devel size
STEP=64


# This is just for local runs
NUMRANKS=4 

# You should not need to adjust this
MPIVER_SCHOONER="OpenMPI/3.1.4-GCC-8.3.0"
CUDAVER_SCHOONER="CUDA/10.1.243-GCC-8.3.0"
MPIVER="mpi/mpich-x86_64"


# NOTE: https://stackoverflow.com/questions/32217413/error-while-trying-to-load-module-environment-in-makefile
# make will not find "module" because by default it uses "sh" and not "bash"
SHELL:=/bin/bash
# Additionally, we have to use ":\" for each line that we want to use the module environment


clean:
	rm -f *.x *~ *.o

cleanall: clean
	rm -f *.csv *.png


all-schooner: build-verifier-schooner  build-bench-schooner 
	sbatch parallel-prob.sbatch

build-verifier-schooner:
	module load ${MPIVER_SCHOONER}; module load ${CUDAVER_SCHOONER};\
	./build_test_op.sh


build-bench-schooner:
	module load ${MPIVER_SCHOONER}; module load ${CUDAVER_SCHOONER};\
	./build_bench_op.sh

all-local: run-verifier-local run-bench-local

run-verifier-local: build-verifier-local
	mpiexec -n ${NUMRANKS} ./run_test_op_var01.x  ${MIN} ${MAX} ${STEP} 1 1 result_verification_local_op_var01_k${KMEDIUM}.csv
	cat result_verification_local_op_var01_k${KMEDIUM}.csv
	mpiexec -n ${NUMRANKS} ./run_test_op_var02.x  ${MIN} ${MAX} ${STEP} 1 1 result_verification_local_op_var02_k${KMEDIUM}.csv
	cat result_verification_local_op_var02_k${KMEDIUM}.csv
	mpiexec -n ${NUMRANKS} ./run_test_op_var03.x  ${MIN} ${MAX} ${STEP} 1 1 result_verification_local_op_var03_k${KMEDIUM}.csv
	cat result_verification_local_op_var03_k${KMEDIUM}.csv
	echo "Number of FAILS: `grep "FAIL" result_verification_local_op_*.csv|wc -l`"


run-bench-local: build-bench-local
	mpiexec -n ${NUMRANKS} ./run_bench_op_var01.x ${MIN} ${MAX} ${STEP} 1 1  result_bench_local_op_var01_k${KMEDIUM}.csv
	cat result_bench_local_op_var01_k${KMEDIUM}.csv
	mpiexec -n ${NUMRANKS} ./run_bench_op_var02.x ${MIN} ${MAX} ${STEP} 1 1  result_bench_local_op_var02_k${KMEDIUM}.csv
	cat result_bench_local_op_var02_k${KMEDIUM}.csv
	mpiexec -n ${NUMRANKS} ./run_bench_op_var03.x ${MIN} ${MAX} ${STEP} 1 1  result_bench_local_op_var03_k${KMEDIUM}.csv
	cat result_bench_local_op_var03_k${KMEDIUM}.csv


	./plotter_multi.py "Local Results of Stencil Computation" "PLOT_local.png" "result_bench_local_op_var01_k${KMEDIUM}.csv" "result_bench_local_op_var02_k${KMEDIUM}.csv" "result_bench_local_op_var03_k${KMEDIUM}.csv"

build-verifier-local:
	./build_test_op.sh

build-bench-local:
	./build_bench_op.sh

