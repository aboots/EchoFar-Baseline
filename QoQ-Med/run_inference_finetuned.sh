#!/bin/bash
#SBATCH --job-name=qoq_ft_infer
#SBATCH --partition=full
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:b200_full:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/mahdi.abootorabi/EchoFAR/QoQ-Med/logs/ft_infer_%j.log
#SBATCH --error=/home/mahdi.abootorabi/EchoFAR/QoQ-Med/logs/ft_infer_%j.err

source /home/mahdi.abootorabi/miniconda3/etc/profile.d/conda.sh
conda activate echofar

export PYTHONPATH=$PYTHONPATH:/home/mahdi.abootorabi/EchoFAR

echo "Starting Finetuned QoQ-Med Inference at $(date)"
echo "Job ID: $SLURM_JOB_ID"

python /home/mahdi.abootorabi/EchoFAR/QoQ-Med/inference_finetuned.py \
    --adapter_path /home/mahdi.abootorabi/EchoFAR/qoq_med_finetuned/checkpoint-best \
    --base_model ddvd233/QoQ-Med-VL-7B \
    --output_path /home/mahdi.abootorabi/EchoFAR/results_qoq_med_finetuned.json

echo "Finished Finetuned QoQ-Med Inference at $(date)"
