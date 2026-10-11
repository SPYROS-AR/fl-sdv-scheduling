#!/bin/bash
#SBATCH --job-name=sweep_aggregator
#SBATCH --output=slurm-aggregate-%j.out
#SBATCH --error=slurm-aggregate-%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=yoda

echo "Starting array aggregation..."

python src/compare_sweep.py

echo "Job finished."
