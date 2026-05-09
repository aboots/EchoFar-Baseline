#!/bin/bash
#SBATCH --job-name=lingshu_ft_infer
#SBATCH --partition=full
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:b200_full:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ft_infer_%j.log
#SBATCH --error=logs/ft_infer_%j.err

source /home/mahdi.abootorabi/miniconda3/etc/profile.d/conda.sh
conda activate echofar

export PYTHONPATH=$PYTHONPATH:/home/mahdi.abootorabi/EchoFAR

echo "Starting Finetuned Lingshu Inference at $(date)"
echo "Job ID: $SLURM_JOB_ID"

python /home/mahdi.abootorabi/EchoFAR/Lingshu/inference_finetuned.py \
    --adapter_path /home/mahdi.abootorabi/EchoFAR/lingshu_finetuned/checkpoint-best \
    --base_model lingshu-medical-mllm/Lingshu-7B \
    --output_path /home/mahdi.abootorabi/EchoFAR/results_lingshu_finetuned.json

echo "Finished Finetuned Lingshu Inference at $(date)"
