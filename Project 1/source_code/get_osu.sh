#!/bin/bash

# Simple OSU Benchmark Setup
set -e

echo "Setting up OSU benchmarks..."

# Download and extract
wget -q https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.4.tar.gz
tar -xzf osu-micro-benchmarks-7.4.tar.gz
cd osu-micro-benchmarks-7.4

# Load MPI and compile
module load openMPI/4.1.6
./configure CC=mpicc CXX=mpicxx
make -j

echo "OSU benchmarks ready at: $(pwd)/c/mpi/collective/blocking/"
echo "Test with: mpirun -np 2 ./c/mpi/collective/blocking/osu_bcast"

# Define necessary folders
mkdir -p slurm-output
mkdir -p output/{broadcast_fixed, broadcast_var, reduce_fixed, reduce_var}
