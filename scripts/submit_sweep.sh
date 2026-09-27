#!/bin/bash
#SBATCH --job-name=fl_sweep
#SBATCH --partition=yoda
#SBATCH --gres=gpu:1             
#SBATCH --cpus-per-task=20        
#SBATCH --mem=32G                
#SBATCH --array=1-100 # SAME AS NUMBER OF LINES IN params.txt
#SBATCH --time=04:00:00
#SBATCH --output=logs/job_%A_%a.out 

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source ~/miniconda3/bin/activate fl_env
mkdir -p logs

if [ ! -s params.txt ]; then
    echo "Error: params.txt does not exist or is empty!"
    exit 1
fi

# The array task ID directly corresponds to the line number in params.txt
LINE_NUM=$SLURM_ARRAY_TASK_ID

# Prevent running out of bounds if params.txt has fewer lines than the array limit
TOTAL_LINES=$(wc -l < params.txt)
if [ "$LINE_NUM" -gt "$TOTAL_LINES" ]; then
    echo "Task ID $LINE_NUM exceeds the number of parameter sets ($TOTAL_LINES). Exiting."
    exit 0
fi

# Extract parameters for the current line
PARAMS=$(sed -n "${LINE_NUM}p" params.txt)
read -r ALPHA BETA GAMMA DELTA <<< "$PARAMS"

echo "Slurm Array Task ${LINE_NUM} running on $(hostname)"
echo "Launching Sweep: A=$ALPHA B=$BETA G=$GAMMA D=$DELTA"

# Run the single worker directly
python src/sweep_worker.py \
    --alpha $ALPHA \
    --beta $BETA \
    --gamma $GAMMA \
    --delta $DELTA \
    --rounds 50 \
    --job_id ${LINE_NUM}
