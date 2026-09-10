#!/bin/bash
#SBATCH --job-name=fl_final_eval
#SBATCH --partition=yoda
#SBATCH --gres=gpu:1             
#SBATCH --cpus-per-task=20
#SBATCH --time=03:00:00
#SBATCH --mem=32G               
#SBATCH --output=logs/final_%j.out
#SBATCH --error=logs/final_%j.err


source ~/miniconda3/bin/activate fl_env

mkdir -p logs

echo "Starting final evaluation (Dynamic vs Fixed baselines) on $(hostname)..."

# Run the main script
python src/main.py \
    --model fl_cnn_balanced
python src/main.py \
    --model fl_cnn

echo "Evaluation complete!"
