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

mkdir -p logs

# Environment setup
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
elif [ -f "$HOME/miniconda3/bin/activate" ]; then
  source "$HOME/miniconda3/bin/activate" fl_env
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 3 Dirichlet non-IID distributions: 0.1 (extreme), 0.3 (moderate), 0.5 (mild)
ALPHAS=(0.1 0.3 0.5)
NUM_EXPS=6

ALPHA_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_EXPS ))
EXP_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_EXPS ))

DIRICHLET_ALPHA=${ALPHAS[$ALPHA_IDX]}
JOB_SUFFIX="a$(echo "$DIRICHLET_ALPHA" | tr -d '.')"

case $EXP_IDX in
  0) EXP_ARGS="--dynamic_only --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}" ;;
  1) EXP_ARGS="--fixed 1 --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}" ;;
  2) EXP_ARGS="--fixed 5 --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}" ;;
  3) EXP_ARGS="--fixed 10 --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}" ;;
  4) EXP_ARGS="--fedprox-only --fedprox-epochs 5 --fedprox-mu 0.01 --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}" ;;
  5) EXP_ARGS="--dynamic-fedprox --fedprox-mu 0.01 --dirichlet-alpha $DIRICHLET_ALPHA --job-id ${JOB_SUFFIX}" ;;
esac

echo "=========================================================="
echo "Slurm Array Task $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "Dirichlet Alpha: $DIRICHLET_ALPHA | Experiment Index: $EXP_IDX"
echo "Command: python src/main.py --rounds 250 --gpus_per_client 0.1 $EXP_ARGS"
echo "=========================================================="

python src/main.py --rounds 250 --gpus_per_client 0.1 $EXP_ARGS

echo "Task $SLURM_ARRAY_TASK_ID completed successfully."
