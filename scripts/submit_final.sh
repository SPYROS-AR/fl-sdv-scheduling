#!/bin/bash
#SBATCH --job-name=fl_final_eval
#SBATCH --partition=yoda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/final_%A_%a.out
#SBATCH --error=logs/final_%A_%a.err
#SBATCH --array=0-17

set -e

mkdir -p logs


# Ensure preprocessed network dataset is available
if [ ! -f "dataset/data/cleaned_network_dataset.csv" ]; then
  echo "Dataset not preprocessed yet. Initializing dataset..."
  python src/setup_data.py
fi

# CIFAR-10 Dataset (Tasks 0-17)
DATASET="cifar10"

# 3 Dirichlet non-IID distributions: 0.1 (extreme), 0.3 (moderate), 0.5 (mild)
ALPHAS=(0.1 0.3 0.5)
NUM_EXPS=6

# If running inside a Slurm job array, execute only the assigned task ID.
# If launched directly via regular bash, loop sequentially over all 18 tasks
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  TASK_IDS=("$SLURM_ARRAY_TASK_ID")
else
  echo "No SLURM_ARRAY_TASK_ID detected. Running all 18 tasks (0..17) sequentially..."
  TASK_IDS=({0..17})
fi

for TASK_ID in "${TASK_IDS[@]}"; do
  ALPHA_IDX=$(( TASK_ID / NUM_EXPS ))
  EXP_IDX=$(( TASK_ID % NUM_EXPS ))

  DIRICHLET_ALPHA=${ALPHAS[$ALPHA_IDX]}
  ALPHA_STR="a$(echo "$DIRICHLET_ALPHA" | tr -d '.')"
  JOB_SUFFIX="${ALPHA_STR}"

  case $EXP_IDX in
    0) EXP_ARGS="--dataset $DATASET --dynamic_only --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}" ;;
    1) EXP_ARGS="--dataset $DATASET --fixed 1 --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}" ;;
    2) EXP_ARGS="--dataset $DATASET --fixed 5 --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}" ;;
    3) EXP_ARGS="--dataset $DATASET --fixed 10 --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}" ;;
    4) EXP_ARGS="--dataset $DATASET --fedprox-only --fedprox-epochs 5 --fedprox-mu 0.01 --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}" ;;
    5) EXP_ARGS="--dataset $DATASET --dynamic-fedprox --fedprox-mu 0.01 --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}" ;;
  esac

  echo "=========================================================="
  echo "Task ID: $TASK_ID on $(hostname)"
  echo "Dataset: $DATASET | Dirichlet Alpha: $DIRICHLET_ALPHA | Exp Index: $EXP_IDX"
  echo "Job Suffix: $JOB_SUFFIX"
  echo "Command: python src/main.py --rounds 250 --gpus_per_client 0.1 $EXP_ARGS"
  echo "=========================================================="

  python src/main.py --rounds 250 --gpus_per_client 0.1 $EXP_ARGS

  echo "Task $TASK_ID ($DATASET - $JOB_SUFFIX) completed successfully."
done