#!/bin/bash
#SBATCH --job-name=fl_final_eval
#SBATCH --partition=yoda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/final_%A_%a.out
#SBATCH --error=logs/final_%A_%a.err
#SBATCH --array=0-35

set -e

mkdir -p logs

# Environment setup
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
elif [ -f "$HOME/miniconda3/bin/activate" ]; then
  source "$HOME/miniconda3/bin/activate" fl_env
fi

# Ensure preprocessed network trace dataset is available
if [ ! -f "dataset/data/cleaned_network_dataset.csv" ]; then
  echo "Dataset not preprocessed yet. Initializing dataset..."
  python src/setup_data.py
fi

# 2 Datasets: cifar10 (Tasks 0-17) and gtsrb (Tasks 18-35)
DATASETS=(cifar10 gtsrb)

# 3 Dirichlet non-IID distributions: 0.1 (extreme), 0.3 (moderate), 0.5 (mild)
ALPHAS=(0.1 0.3 0.5)
NUM_EXPS=6
NUM_ALPHAS=3
NUM_TASKS_PER_DATASET=$(( NUM_ALPHAS * NUM_EXPS ))

DATASET_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_TASKS_PER_DATASET ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % NUM_TASKS_PER_DATASET ))
ALPHA_IDX=$(( REMAINDER / NUM_EXPS ))
EXP_IDX=$(( REMAINDER % NUM_EXPS ))

DATASET=${DATASETS[$DATASET_IDX]}
DIRICHLET_ALPHA=${ALPHAS[$ALPHA_IDX]}
ALPHA_STR="a$(echo "$DIRICHLET_ALPHA" | tr -d '.')"

if [ "$DATASET" == "cifar10" ]; then
  JOB_SUFFIX="${ALPHA_STR}"
else
  JOB_SUFFIX="gtsrb_${ALPHA_STR}"
fi

case $EXP_IDX in
  0) EXP_ARGS="--dataset $DATASET --dynamic_only --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}" ;;
  1) EXP_ARGS="--dataset $DATASET --fixed 1 --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}" ;;
  2) EXP_ARGS="--dataset $DATASET --fixed 5 --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}" ;;
  3) EXP_ARGS="--dataset $DATASET --fixed 10 --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}" ;;
  4) EXP_ARGS="--dataset $DATASET --fedprox-only --fedprox-epochs 5 --fedprox-mu 0.01 --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}" ;;
  5) EXP_ARGS="--dataset $DATASET --dynamic-fedprox --fedprox-mu 0.01 --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}" ;;
esac

echo "=========================================================="
echo "Slurm Array Task $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "Dataset: $DATASET | Dirichlet Alpha: $DIRICHLET_ALPHA | Exp Index: $EXP_IDX"
echo "Job Suffix: $JOB_SUFFIX"
echo "Command: python src/main.py --rounds 250 --gpus_per_client 0.1 $EXP_ARGS"
echo "=========================================================="

python src/main.py --rounds 250 --gpus_per_client 0.1 $EXP_ARGS

echo "Task $SLURM_ARRAY_TASK_ID ($DATASET - $JOB_SUFFIX) completed successfully."
