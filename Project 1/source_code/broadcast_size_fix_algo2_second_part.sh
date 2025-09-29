#!/bin/bash
#SBATCH --job-name=bsf2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --time=01:59:59
#SBATCH --partition EPYC
#SBATCH --exclusive

module load openMPI/4.1.6

# OSU Benchmark path 
OSU_BENCHMARK_DIR="osu-micro-benchmarks-7.4/c/mpi/collective/blocking"
OSU_BCAST="$OSU_BENCHMARK_DIR/osu_bcast"

# Parameters definition
N_replica=5000
dimension_size=4



# Algorithm n 2 -->
echo "idx_process,dimension_size,Latency" > ../output/broadcast_fixed/broadcast_algo2_fixed_core_secondpart.csv # CSV file to store results

# Looping over the n of idx_process
for idx_process in {243..256} # from 243 to 256 tasks 
do
    # Perform osu_bcast with current processors, fixed message dimension_size and fixed number of N_replica
    result_broadcast2=$(mpirun --map-by core -np $idx_process --mca coll_tuned_use_dynamic_rules true --mca coll_tuned_bcast_algorithm 3 $OSU_BCAST -m $dimension_size -x $N_replica -i $N_replica | tail -n 1 | awk '{print $2}') # osu_bcast with current processors, fixed message dimension_size and fixed number of N_replica
    echo "$idx_process,$dimension_size,$result_broadcast2" >> ../output/broadcast_fixed/broadcast_algo2_fixed_core_secondpart.csv # CSV file to store results
done
# end algo 2

