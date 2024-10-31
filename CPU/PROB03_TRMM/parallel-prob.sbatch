#!/bin/bash
#SBATCH --partition=debug_5min
#SBATCH --exclusive
#SBATCH --nodes=2                   # <--- Nodes for MPI
#SBATCH --ntasks=16                 # <--- Adjust these if you are mixing MPI and OpenMP
#SBATCH --ntasks-per-node=8         # <---
#SBATCH --output=./jobname_%J_stdout.txt
#SBATCH --error=./jobname_%J_stderr.txt
#SBATCH --time=5:00
#SBATCH --job-name=cs4473_cs5473_lab02
#SBATCH --mail-user=youremailaddress@yourinstitution.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=./


# Increments to use in the local tests
MIN=32
MAX=8192 # good large number
#MAX=4096 # medium size
#MAX=1024 # small devel size
STEP=32


module load OpenMPI

# Let's get the stats of the nodes we are on
hostname -f # hostname of the machine
who         # who else is on this system
cat /proc/cpuinfo # low level cpu details


# Verify
mpiexec ./run_test_op_var01.x ${MIN} ${MAX} ${STEP} 1 1  result_verification_op_var01.csv
mpiexec ./run_test_op_var02.x ${MIN} ${MAX} ${STEP} 1 1  result_verification_op_var02.csv
mpiexec ./run_test_op_var03.x ${MIN} ${MAX} ${STEP} 1 1  result_verification_op_var03.csv

echo "Number of FAILS: `grep "FAIL" result_verification_op_*.csv|wc -l`"

# Bench the results
mpiexec ./run_bench_op_var01.x ${MIN} ${MAX} ${STEP} 1 1  result_bench_op_var01.csv
mpiexec ./run_bench_op_var02.x ${MIN} ${MAX} ${STEP} 1 1  result_bench_op_var02.csv
mpiexec ./run_bench_op_var03.x ${MIN} ${MAX} ${STEP} 1 1  result_bench_op_var03.csv


# plot the results
module unload OpenMPI
module load scikit-learn/0.23.1-foss-2020a-Python-3.8.2
module load matplotlib/3.2.1-foss-2019b-Python-3.8.2
./plotter_multi.py "Results of Stencil Computation on Schooner" "PLOT_schooner.png" result_bench_op_var01.csv result_bench_op_var02.csv result_bench_op_var03.csv
