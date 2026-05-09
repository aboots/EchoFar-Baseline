#!/bin/bash
#SBATCH --job-name=test_lingshu
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_lingshu_%j.log
#SBATCH --error=logs/test_lingshu_%j.err

source /home/mahdi.abootorabi/miniconda3/etc/profile.d/conda.sh
conda activate echofar

export PYTHONPATH=$PYTHONPATH:/home/mahdi.abootorabi/EchoFAR

python /home/mahdi.abootorabi/EchoFAR/Lingshu/test_lingshu_one_sample.py
