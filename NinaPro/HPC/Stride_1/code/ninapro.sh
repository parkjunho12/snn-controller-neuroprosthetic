#!/bin/bash
#SBATCH --time=4:00:00  # Time allocated for the job to complete (1 hour)
#SBATCH --ntasks=1  # This script will only launch a single SLURM task
#SBATCH --cpus-per-task=10  # Allocate 4 CPUs to this job
#SBATCH --mem=10G  # Allocate 10 gigabytes of memory to this job
#SBATCH --output=./output/ninapro_feature_stride_1.txt  # This is where your output and errors are logged

module load Java/17.0.4
module load Anaconda3/2024.02-1

source activate myspark

spark-submit --executor-memory 10g --driver-memory 10g ./ninapro_stride_1.py
