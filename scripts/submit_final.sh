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

DIRICHLET_ALPHA=${1:-0.1}
JOB_SUFFIX="a$(echo "$DIRICHLET_ALPHA" | tr -d '.')"

EXPERIMENTS=(
  "--dynamic_only --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}"
  "--fixed 1 --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}"
  "--fixed 5 --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}"
  "--fixed 10 --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}"
  "--fedprox-only --fedprox-epochs 5 --fedprox-mu 0.01 --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}"
  "--dynamic-fedprox --fedprox-mu 0.01 --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}"
)

CURRENT_ARG=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}

echo "Starting array task $SLURM_ARRAY_TASK_ID with argument: $CURRENT_ARG"

python src/main.py --rounds 250 $CURRENT_ARG

echo "Evaluation complete for task $SLURM_ARRAY_TASK_ID"
