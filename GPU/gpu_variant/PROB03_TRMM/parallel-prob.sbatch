#!/bin/bash
#SBATCH --partition=debug_gpu
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=20          # We want to make sure we are the only ones on the gpu nodes 
#SBATCH --ntasks-per-node=20 # SLURM might try to place two gpu jobs on the same system.
#SBATCH --output=./jobname_%J_stdout.txt
#SBATCH --error=./jobname_%J_stderr.txt
#SBATCH --time=5:00
#SBATCH --job-name=cs4473_cs5473_trmm
#SBATCH --mail-user=youremailaddress@yourinstitution.edu # STUDENT_TODO: Change this
#SBATCH --mail-type=ALL
#SBATCH --chdir=./


# Increments to use in the local tests
MIN=64
MAX=2048
#MAX=32 # something small for a sanity check
STEP=64

# These are sufficiently small

# Load the important modules
module load OpenMPI/3.1.4-GCC-8.3.0
module load CUDA/10.1.243-GCC-8.3.0

# Let's get the stats of the nodes we are on
hostname -f # hostname of the machine
who         # who else is on this system
cat /proc/cpuinfo # low level cpu details
nvidia-smi        # gpu info


# Verify
#mpiexec ./run_test_op_var01.x ${MIN} ${MAX} ${STEP} 1 1  result_verification_op_var01.csv # this runs very slow if you do not modify it.
#mpiexec ./run_test_op_var02.x ${MIN} ${MAX} ${STEP} 1 1  result_verification_op_var02.csv
mpiexec ./run_test_op_var03.x ${MIN} ${MAX} ${STEP} 1 1  result_verification_op_var03.csv

echo "Number of FAILS: `grep "FAIL" result_verification_op_*.csv|wc -l`"

# Bench the results
#mpiexec ./run_bench_op_var01.x ${MIN} ${MAX} ${STEP} 1 1  result_bench_op_var01.csv
#mpiexec ./run_bench_op_var02.x ${MIN} ${MAX} ${STEP} 1 1  result_bench_op_var02.csv
mpiexec ./run_bench_op_var03.x ${MIN} ${MAX} ${STEP} 1 1  result_bench_op_var03.csv


# plot the results
module unload OpenMPI
module load scikit-learn/0.23.1-foss-2020a-Python-3.8.2
module load matplotlib/3.2.1-foss-2019b-Python-3.8.2
./plotter_multi.py "Results of TRMM Computation on Schooner" "PLOT_schooner.png" result_bench_op_var01.csv result_bench_op_var02.csv result_bench_op_var03.csv



