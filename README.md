# Project — Image generation with a diffusion model

Project to implement a noise-conditioned diffusion denoiser (Karras et al. style).  
Provides training, sampling (Euler sampler), simple model scaffold and utilities for FashionMNIST / CelebA.

## Repo layout
- src/ — training, sampling, model, data helpers
- configs/ — training configs (see configs/train_example.yaml)
- data/ — datasets (downloaded automatically - careful to download issues for CelebA)
- runs/ — saved model run (checkpoints, samples for fids, losses)
- scripts/ — convenience shell wrappers
- instructions.ipynb — notebook with background + examples

## Requirements (macOS)
- see `pyproject.toml`
- install with `pip install -e .` from within the project folder

## Quick start

The entire project was run on UBELIX (the University of Bern computing cluster), but the scripts can also be run locally

From project root:

1. Train (small smoke):
```bash
# edit configs/train_example.yaml for small batch/epochs first
python -m src.train
```

2. Sample (uses latest checkpoint):
```bash
python -m src.sample --n 8 --steps 50 --outdir samples model_dir
```

## Best results achieved

### FashionMNIST

- Provided model, without noise conditioning: `python -m src.train configs/config_fashionmnist_uncond.yaml`
- Provided model, with noise conditioning: `python -m src.train configs/config_fashionmnist_cond.yaml`
- Upgraded model, with noise conditioning: `python -m src.train configs/config_fashionmnist_improved_cond.yaml`
- Provided model, with noise+class conditioning for Classifier-Free Guidance: `python -m src.train configs/config_fashionmnist_cfg.yaml`

### CelebA

- Upgraded model, with noise conditioning and attention layers: `python -m src.train configs/config_celeba_improved_cfg.yaml`
