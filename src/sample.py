import argparse
import os
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from src.common import euler_sample
from src.data import load_dataset_and_make_dataloaders
from src.sigma import build_sigma_schedule
from src.utils import (
    build_model_for_sampling,
    find_latest_checkpoint,
    load_ckpt_into_model,
    to_unit_range,
)

# TODO: should we do FID calculation here after sampling? Probably not, keep separate.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/model_epoch_1.pth")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--outdir", default="samples")
    p.add_argument("--dataset", default="FashionMNIST")
    p.add_argument("--data_root", default="data")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # get dataset info (sigma_data, channels, image size)
    _, info = load_dataset_and_make_dataloaders(
        dataset_name=args.dataset, root_dir=args.data_root, batch_size=1
    )
    channels = info.image_channels
    H = info.image_size
    sigma_data = float(info.sigma_data)

    # decide checkpoint to use: CLI overrides, otherwise use latest in checkpoints/
    ckpt = args.ckpt
    if not os.path.exists(ckpt):
        found = find_latest_checkpoint("checkpoints")
        if found:
            ckpt = found
            print("No ckpt at", args.ckpt, "-> using latest checkpoint:", ckpt)
        else:
            raise FileNotFoundError(
                f"No checkpoint found at {args.ckpt} and no files in checkpoints/"
            )

    model = build_model_for_sampling(info, device)
    model = load_ckpt_into_model(model, ckpt, device)
    sigmas = build_sigma_schedule(args.steps, rho=args.rho)
    sigmas = sigmas.to(device)

    samples = euler_sample(model, sigmas, args.n, channels, H, sigma_data, device)

    imgs = to_unit_range(samples)
    grid = make_grid(imgs, nrow=min(8, args.n))
    out_path = os.path.join(args.outdir, "samples_grid.png")
    # TODO: make_grid is called inside save_image, just pass nrow to it.
    save_image(grid, out_path)
    print("Saved samples to", out_path)


if __name__ == "__main__":
    main()
