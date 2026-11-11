#!/bin/bash
#SBATCH --job-name=fl_seeds
#SBATCH --partition=yoda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/seeds_%A_%a.out
#SBATCH --error=logs/seeds_%A_%a.err
#SBATCH --array=0-209%20
#
# Multi-seed evaluation requested by the audit: 14 schemes x 3 Dirichlet alphas x 5 seeds
# = 210 tasks (task = seed-major, then alpha, then scheme). Submit a subset with, e.g.,
#   sbatch --array=0-41 scripts/submit_seeds.sh      # seed 1 only
# Outputs: metrics_<exp><tag>_<aXX>_s<seed>.csv and clients_<...>.csv in the working
# directory; aggregate with
#   python scripts/aggregate_results.py <dir> --plot

set -e
mkdir -p logs

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
elif [ -f "$HOME/miniconda3/bin/activate" ]; then
  source "$HOME/miniconda3/bin/activate" fl_env
fi

if [ ! -f "dataset/data/cleaned_network_dataset.csv" ]; then
  echo "Dataset not preprocessed yet. Initializing dataset..."
  python src/setup_data.py
fi

ROUNDS=${ROUNDS:-250}
SEEDS=(1 2 3 4 5)
ALPHAS=(0.1 0.3 0.5)
REVISED="--stop-rule utility --gain-estimate curve --tag=-rev"
# "<experiment>|<extra arguments>"
SCHEMES=(
  "dynamic|"                                  # submitted configuration
  "dynamic_fedprox|"
  "dynamic|${REVISED}"                        # endogenous gain floor + curve-slope gain
  "dynamic_fedprox|${REVISED}"
  "fixed_1|"
  "fixed_5|"
  "fixed_10|"
  "fedprox_5|"
  "slackfill|"                                # e = floor(rho (1 - U_crit)), always upload
  "fixedcap_5|"                               # E = 5 capped by the slack
  "threshold_5|"                              # slack-capped E = 5, train/upload only if B >= B_th
  "randpart_5_72|"                            # E = 5 for a random 72% of vehicles per round
  "dynamic_nostop|"                           # ablation: no learning-aware stopping
  "dynamic_nocontinue|"                       # ablation: no CONTINUE state
)

N_SCHEMES=${#SCHEMES[@]}
N_ALPHAS=${#ALPHAS[@]}
TASK=${SLURM_ARRAY_TASK_ID:?run through sbatch or set SLURM_ARRAY_TASK_ID}
SEED=${SEEDS[$(( TASK / (N_SCHEMES * N_ALPHAS) ))]}
ALPHA=${ALPHAS[$(( (TASK / N_SCHEMES) % N_ALPHAS ))]}
ENTRY=${SCHEMES[$(( TASK % N_SCHEMES ))]}
EXP=${ENTRY%%|*}
EXTRA=${ENTRY#*|}
JOB_ID="a$(echo "$ALPHA" | tr -d '.')"

CMD="python src/main.py --rounds $ROUNDS --gpus_per_client 0.1 --dataset cifar10 \
  --dirichlet-alpha $ALPHA --fedprox-mu 0.01 --seed $SEED --job-id $JOB_ID \
  --experiments $EXP $EXTRA"

echo "=========================================================="
echo "Task $TASK on $(hostname): seed=$SEED alpha=$ALPHA scheme=$EXP $EXTRA"
echo "$CMD"
echo "=========================================================="
eval "$CMD"
echo "Task $TASK completed."
