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

# Run the sampling script
python -m src.sample \
    --n 16 \
    --outdir outputs/samples \
    --dataset CelebA \
    --data_root ../data \
    /model_run_directory