#!/bin/bash
#SBATCH --job-name=rsv1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --time=01:59:59
#SBATCH --partition EPYC
#SBATCH --exclusive

module load openMPI/4.1.6

# OSU Benchmark path 
OSU_BENCHMARK_DIR="osu-micro-benchmarks-7.4/c/mpi/collective/blocking"
OSU_REDUCE="$OSU_BENCHMARK_DIR/osu_reduce"

# Benchmarks parameters
N_replica=5000

# Algorithm n 1 --> 
echo "number_processes,Size,Latency" > ../output/reduce_var/reduce_algo1_variable_core.csv

for idx_power in {1..8} # Looping from 2 --> 256
do
	number_processes=$((2**idx_power))
	for size_idx_dim in {1..10} # Looping from 2 --> 1024
	do
		size=$((2**size_idx_dim)) # variable dimension
		result_allreduce1=$(mpirun --map-by core -np $number_processes --mca coll_tuned_use_dynamic_rules true --mca coll_tuned_reduce_algorithm 1 $OSU_REDUCE -m $size -x $N_replica -i $N_replica | tail -n 1 | awk '{print $2}')
		echo "$number_processes,$size,$result_allreduce1" >> ../output/reduce_var/reduce_algo1_variable_core.csv
	done
done	
#  end algo 1

