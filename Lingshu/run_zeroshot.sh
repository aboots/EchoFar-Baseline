#!/bin/bash
#SBATCH --job-name=lingshu_zeroshot
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/zeroshot_%j.log
#SBATCH --error=logs/zeroshot_%j.err

# Load Conda environment
source /home/mahdi.abootorabi/miniconda3/etc/profile.d/conda.sh
conda activate echofar

# Ensure weights and datasets can be found
export PYTHONPATH=$PYTHONPATH:/home/mahdi.abootorabi/EchoFAR

echo "Starting Lingshu Zero-Shot Inference at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Run the inference script
python /home/mahdi.abootorabi/EchoFAR/Lingshu/inference_zeroshot.py

echo "Finished Lingshu Zero-Shot Inference at $(date)"
