#!/bin/bash
#SBATCH --job-name=fl_final_eval
#SBATCH --partition=yoda
#SBATCH --gres=gpu:1          
#SBATCH --cpus-per-task=20
#SBATCH --time=05:00:00
#SBATCH --mem=32G              
#SBATCH --output=logs/final_%A_%a.out
#SBATCH --error=logs/final_%A_%a.err


source ~/miniconda3/bin/activate fl_env
mkdir -p logs

# Run the python script with the mapped model
python src/main.py --rounds 150

echo "Evaluation complete"
