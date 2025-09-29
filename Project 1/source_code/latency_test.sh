#!/bin/bash
#SBATCH --job-name=latency
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=0-00:15:00
#SBATCH -A dssc
#SBATCH -p EPYC
#SBATCH --output=latency_%j.out
#SBATCH --error=latency_%j.err

# Load MPI
module purge
module load openMPI/4.1.6

# OSU benchmark path
OSU_BENCHMARK_DIR="osu-micro-benchmarks-7.4/c/mpi/pt2pt/standard"
OSU_LATENCY="$OSU_BENCHMARK_DIR/osu_latency"

echo "Running from: ${SLURM_SUBMIT_DIR}"
cd "${SLURM_SUBMIT_DIR}"

# Create output directory if it doesn't exist
mkdir -p ../output

# Output file
outfile="../output/latency.txt"
echo "# Latency: core 0 vs distant cores (exclusive)" > "$outfile"

# Core to test against core0
for i in 1 4 8 16 32 64 96 112 127; do
    echo "Testing with cores: 0 and $i" | tee -a "$outfile"

    # Create rankfile for explicit core binding
    echo -e "rank 0=localhost slot=0\nrank 1=localhost slot=$i" > rankfile_$i

    # Benchmarking --> tests only 2-byte messages
    # FIXED: Use --rankfile instead of --map-by rankfile:file=
    mpirun --rankfile rankfile_$i \
           $OSU_LATENCY \
           -x 100 -i 1000 -m 2:2 2>&1 | tee -a "$outfile"
    
    # Clean up rankfile
    rm -f rankfile_$i
done