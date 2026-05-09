#!/bin/bash
#SBATCH --job-name=qoq_med_incontext
#SBATCH --partition=full
#SBATCH --gres=gpu:b200_full:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mahdi.abootorabi/EchoFAR/QoQ-Med/logs/incontext_%j.log
#SBATCH --error=/home/mahdi.abootorabi/EchoFAR/QoQ-Med/logs/incontext_%j.err

# Load Conda environment
source /home/mahdi.abootorabi/miniconda3/etc/profile.d/conda.sh
conda activate echofar

# Ensure weights and datasets can be found
export PYTHONPATH=$PYTHONPATH:/home/mahdi.abootorabi/EchoFAR

echo "Starting QoQ-Med In-Context Inference at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Run the inference script
python /home/mahdi.abootorabi/EchoFAR/QoQ-Med/inference_incontext.py

echo "Finished QoQ-Med In-Context Inference at $(date)"
