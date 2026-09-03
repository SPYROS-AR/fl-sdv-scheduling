#!/bin/bash
#SBATCH --job-name=fl_sweep
#SBATCH --partition=yoda
#SBATCH --gres=gpu:1             
#SBATCH --cpus-per-task=30        
#SBATCH --mem=80G                 
#SBATCH --array=1-4              
#SBATCH --time=04:00:00
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Dynamic Array Math (Distribute total lines evenly among tasks)
TOTAL_LINES=$(wc -l < params.txt)
NUM_TASKS=$SLURM_ARRAY_TASK_MAX 
LINES_PER_TASK=$(( (TOTAL_LINES + NUM_TASKS - 1) / NUM_TASKS ))

START_LINE=$(( (SLURM_ARRAY_TASK_ID - 1) * LINES_PER_TASK + 1 ))
END_LINE=$(( START_LINE + LINES_PER_TASK - 1 ))

if [ "$END_LINE" -gt "$TOTAL_LINES" ]; then
    END_LINE=$TOTAL_LINES
fi

echo "Slurm Array Task ${SLURM_ARRAY_TASK_ID} running on $(hostname)"
echo "File has $TOTAL_LINES lines. Assigned lines $START_LINE to $END_LINE."

# Set strict concurrency limit per node (memory safety)
CONCURRENT_LIMIT=4
ACTIVE_JOBS=0

# Loop through assigned lines with batched execution
for i in $(seq $START_LINE $END_LINE); do
    
    # Extract parameters for the current line
    PARAMS=$(sed -n "${i}p" params.txt)
    read -r ALPHA BETA GAMMA DELTA <<< "$PARAMS"
    
    echo "  -> Launching Sub-task ${i} in background: A=$ALPHA B=$BETA G=$GAMMA D=$DELTA"
    
    # Launch worker in the background
    python src/sweep_worker.py \
        --alpha $ALPHA \
        --beta $BETA \
        --gamma $GAMMA \
        --delta $DELTA \
        --job_id ${i} &
        
    # Increment our active job counter
    ACTIVE_JOBS=$((ACTIVE_JOBS + 1))
    
    # Wait for this batch to finish before continuing
    if [ "$ACTIVE_JOBS" -ge "$CONCURRENT_LIMIT" ]; then
        echo "Waiting for current batch of $CONCURRENT_LIMIT workers to finish..."
        wait
        ACTIVE_JOBS=0 # Reset counter for the next batch
    fi
        
done

# Final wait to catch any leftover jobs if the division wasn't perfectly even
wait

echo "All assigned experiments for Array Task ${SLURM_ARRAY_TASK_ID} have finished."
