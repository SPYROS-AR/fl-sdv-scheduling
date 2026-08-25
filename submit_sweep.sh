#!/bin/bash
#SBATCH --job-name=fl_sweep
#SBATCH --partition=yoda
#SBATCH --gres=gpu:1             
#SBATCH --cpus-per-task=28        
#SBATCH --mem=64G                 
#SBATCH --array=1-4%2              
#SBATCH --time=02:00:00
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err


# Calculate the starting and ending line in params.txt for this specific array task
START_LINE=$(( ($SLURM_ARRAY_TASK_ID - 1) * 4 + 1 ))
END_LINE=$(( $START_LINE + 3 ))

echo "Slurm Array Task ${SLURM_ARRAY_TASK_ID} running on $(hostname)"
echo "Processing params.txt lines ${START_LINE} to ${END_LINE}"

# Loop through the assigned lines
for i in $(seq $START_LINE $END_LINE); do
    
    # Extract parameters for the current line
    PARAMS=$(sed -n "${i}p" params.txt)
    read -r ALPHA BETA GAMMA DELTA <<< "$PARAMS"
    
    echo "  -> Launching Sub-task ${i} in background: A=$ALPHA B=$BETA G=$GAMMA D=$DELTA"
    
    # Run worker in the background
    python src/sweep_worker.py \
        --alpha $ALPHA \
        --beta $BETA \
        --gamma $GAMMA \
        --delta $DELTA \
        --job_id ${i} &
        
done

wait

echo "All 4 concurrent experiments for Array Task ${SLURM_ARRAY_TASK_ID} have finished."