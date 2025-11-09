# Project — Image generation with a diffusion model

Project to implement a noise-conditioned diffusion denoiser (Karras et al. style).  
Provides training, sampling (Euler sampler), simple model scaffold and utilities for FashionMNIST / CelebA.

## Repo layout
- src/ — training, sampling, model, data helpers
- configs/ — training configs (configs/train.yaml)
- data/ — datasets (downloaded automatically)
- checkpoints/ — saved model checkpoints
- scripts/ — convenience shell wrappers
- instructions.ipynb — notebook with background + examples

## Requirements (macOS)
- Python 3.9+ (3.10+ recommended)
- PyTorch (matching your CUDA) and torchvision
- Optional: gh (GitHub CLI), pytorch-fid for FID

## Quick start

From project root:

1. Train (small smoke):
```bash
# edit configs/train.yaml for small batch/epochs first
python -m src.train
```

2. Sample (uses latest checkpoint if --ckpt missing):
```bash
python -m src.sample --n 8 --steps 50 --outdir samples
# or specify a checkpoint
python -m src.sample --ckpt checkpoints/model_epoch_1.pth --n 8
```

3. Inspect saved images:
- samples/samples_grid.png
