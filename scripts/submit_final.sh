#!/bin/bash
#SBATCH --job-name=fl_final_eval
#SBATCH --partition=yoda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/final_%A_%a.out
#SBATCH --error=logs/final_%A_%a.err
#SBATCH --array=0-5

mkdir -p logs

EXPERIMENTS=(
  "--dynamic_only"
  "--fixed 1"
  "--fixed 3"
  "--fixed 5"
  "--fixed 10"
  "--fixed 20"
)

CURRENT_ARG=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}

echo "Starting array task $SLURM_ARRAY_TASK_ID with argument: $CURRENT_ARG"

python src/main.py --rounds 250 $CURRENT_ARG

echo "Evaluation complete for task $SLURM_ARRAY_TASK_ID"
