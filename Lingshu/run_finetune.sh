#!/bin/bash
#SBATCH --job-name=lingshu_finetune
#SBATCH --partition=full
#SBATCH --nodelist=rcl-nv2.ece.ubc.ca
#SBATCH --gres=gpu:b200_full:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=logs/finetune_%j.log
#SBATCH --error=logs/finetune_%j.err

source /home/mahdi.abootorabi/miniconda3/etc/profile.d/conda.sh
conda activate echofar

export PYTHONPATH=$PYTHONPATH:/home/mahdi.abootorabi/EchoFAR

echo "Starting Lingshu Finetuning at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

python /home/mahdi.abootorabi/EchoFAR/Lingshu/finetune_lingshu.py \
    --model_name_or_path lingshu-medical-mllm/Lingshu-7B \
    --output_dir /home/mahdi.abootorabi/EchoFAR/lingshu_finetuned \
    --num_epochs 3 \
    --lr 2e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --grad_accum_steps 8 \
    --max_grad_norm 1.0 \
    --lr_scheduler cosine \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --num_workers 4 \
    --log_every_steps 10 \
    --save_every_steps 500 \
    --val_max_steps 200 \
    --gradient_checkpointing

echo "Finished Lingshu Finetuning at $(date)"
