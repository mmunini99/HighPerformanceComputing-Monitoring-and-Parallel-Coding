#!/bin/bash

#
set -e

# Define necessary folders
mkdir -p slurm-output
mkdir -p output/{strong_scaling_mpi, strong_scaling_omp, weak_scaling_mpi, weak_scaling_omp}
