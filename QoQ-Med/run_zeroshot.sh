#!/bin/bash
#SBATCH --job-name=qoq_med_zeroshot
#SBATCH --partition=full
#SBATCH --gres=gpu:b200_full:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mahdi.abootorabi/EchoFAR/QoQ-Med/logs/zeroshot_%j.log
#SBATCH --error=/home/mahdi.abootorabi/EchoFAR/QoQ-Med/logs/zeroshot_%j.err

# Load Conda environment
source /home/mahdi.abootorabi/miniconda3/etc/profile.d/conda.sh
conda activate echofar

# Ensure weights and datasets can be found
export PYTHONPATH=$PYTHONPATH:/home/mahdi.abootorabi/EchoFAR

echo "Starting QoQ-Med Zero-Shot Inference at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Run the inference script
python /home/mahdi.abootorabi/EchoFAR/QoQ-Med/inference_zeroshot.py

echo "Finished QoQ-Med Zero-Shot Inference at $(date)"
