#!/bin/bash

#SBATCH --job-name="diffusion_model"
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --account=invest
#SBATCH --partition=gpu-invest
#SBATCH --qos=job_gpu_sznitman
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --output=logs/train_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

source venv/bin/activate
python -m src.train --config configs/train_unconditioned.yaml
python -m src.train --config configs/train_conditioned.yaml
python -m src.train --config configs/train_conditioned_class.yaml