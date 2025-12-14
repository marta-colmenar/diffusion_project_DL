import argparse
import os
from pathlib import Path

import torch
from torchvision.utils import save_image

from src.common import euler_sample
from src.data import load_dataset_and_make_dataloaders
from src.sigma import build_sigma_schedule
from src.utils import (
    build_model_for_sampling,
    find_latest_checkpoint,
    load_ckpt_into_model,
    to_unit_range,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--outdir", default="samples")
    p.add_argument("--dataset", default="FashionMNIST")
    p.add_argument("--data_root", default="../data")
    p.add_argument("run_dir")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(args.run_dir)

    _, info = load_dataset_and_make_dataloaders(
        dataset_name=args.dataset, root_dir=args.data_root, batch_size=32
    )
    chkp = find_latest_checkpoint(str(run_dir / "checkpoints"))
    model = build_model_for_sampling(str(run_dir / "config.yaml"), device, info)
    model = load_ckpt_into_model(model, chkp, device)

    channels = info.image_channels
    H = info.image_size
    sigma_data = float(info.sigma_data)

    sigmas = build_sigma_schedule(args.steps, rho=args.rho)
    sigmas = sigmas.to(device)

    samples = euler_sample(model, sigmas, args.n, channels, H, sigma_data, device)

    imgs = to_unit_range(samples)
    out_path = os.path.join(args.outdir, "samples_grid.png")
    save_image(imgs, out_path, nrow=min(8, args.n))
    print("Saved samples to", out_path)


if __name__ == "__main__":
    main()
