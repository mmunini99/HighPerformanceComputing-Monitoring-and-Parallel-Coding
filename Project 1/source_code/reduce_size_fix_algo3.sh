#!/bin/bash
#SBATCH --job-name=rsf3
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --time=01:59:59
#SBATCH --partition EPYC
#SBATCH --exclusive

module load openMPI/4.1.6

# OSU Benchmark path 
OSU_BENCHMARK_DIR="osu-micro-benchmarks-7.4/c/mpi/collective/blocking"
OSU_REDUCE="$OSU_BENCHMARK_DIR/osu_reduce"

# Parameters definition
N_replica=5000
dimension_size=4


# Algorithm n 3 -->
echo "idx_process,dimension_size,Latency" > ../output/reduce_fixed/reduce_algo3_fixed_core.csv # CSV file to store results

# Looping over the n of idx_process
for idx_process in {2..256} # from 2 to 256 tasks (128 per node)
do
    # Perform osu_allreduce with current processors, fixed message dimension_size and fixed number of N_replica
    result_allreduce3=$(mpirun --map-by core -np $idx_process --mca coll_tuned_use_dynamic_rules true --mca coll_tuned_reduce_algorithm 4 $OSU_REDUCE -m $dimension_size -x $N_replica -i $N_replica | tail -n 1 | awk '{print $2}') # osu_allreduce with current processors, fixed message dimension_size and fixed number of N_replica
    echo "$idx_process,$dimension_size,$result_allreduce3" >> ../output/reduce_fixed/reduce_algo3_fixed_core.csv # CSV file to store results
done
# end algo 3
