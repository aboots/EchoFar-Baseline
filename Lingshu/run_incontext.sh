#!/bin/bash
#SBATCH --job-name=lingshu_incontext
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/incontext_%j.log
#SBATCH --error=logs/incontext_%j.err

# Load Conda environment
source /home/mahdi.abootorabi/miniconda3/etc/profile.d/conda.sh
conda activate echofar

# Ensure weights and datasets can be found
export PYTHONPATH=$PYTHONPATH:/home/mahdi.abootorabi/EchoFAR

echo "Starting Lingshu In-Context Inference at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Run the inference script
python /home/mahdi.abootorabi/EchoFAR/Lingshu/inference_incontext.py

echo "Finished Lingshu In-Context Inference at $(date)"
