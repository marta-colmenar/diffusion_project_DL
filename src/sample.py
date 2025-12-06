import argparse
import os
import sys
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image
import yaml
import glob
from typing import Optional
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data import load_dataset_and_make_dataloaders
from src.sigma import build_sigma_schedule
from src.model import Model


def build_model_for_sampling(info, device, cfg_path="configs/train.yaml"):
    # load optional config for model hyperparams (safe)
    cfg = {}
    cfg_file = os.path.join(_PROJECT_ROOT, cfg_path)
    if os.path.exists(cfg_file):
        try:
            with open(cfg_file, "r") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

    nb_channels = cfg.get("nb_channels", 64)
    num_blocks = cfg.get("num_blocks", 4)
    cond_channels = cfg.get("cond_channels", 64)
    image_channels = getattr(info, "image_channels", 1)
    conditioned=cfg.get("conditioned", True)

    m = Model(
        image_channels=image_channels,
        nb_channels=nb_channels,
        num_blocks=num_blocks,
        cond_channels=cond_channels,
        conditioned=conditioned,
    )

    m.to(device)
    m.eval()
    return m


def c_funcs(sigma: torch.Tensor, sigma_data: float):
    # sigma: (B,) or scalar tensor -> return tensors shaped (B,)
    denom = torch.sqrt(sigma_data ** 2 + sigma ** 2)
    c_in = 1.0 / denom
    c_out = sigma * sigma_data / denom
    c_skip = (sigma_data ** 2) / (sigma_data ** 2 + sigma ** 2)
    c_noise = torch.log(sigma) / 4.0
    return c_in, c_out, c_skip, c_noise


def euler_sample(model, sigmas, n, channels, H, sigma_data, device):
    # sigmas: 1D tensor [sigma_0, sigma_1, ..., sigma_T] (assumed decreasing)
    x = torch.randn(n, channels, H, H, device=device) * sigmas[0].to(device)
    for i, sigma_scalar in enumerate(sigmas):
        sigma_scalar = sigma_scalar.to(device)
        sigma_next = sigmas[i + 1].to(device) if i + 1 < len(sigmas) else torch.tensor(0.0, device=device)
        # use same sigma for all samples in the batch
        sigma_b = sigma_scalar.repeat(n)
        c_in, c_out, c_skip, c_noise = c_funcs(sigma_b, sigma_data)
        cin_x = (c_in.view(-1, 1, 1, 1) * x)
        with torch.no_grad():
            # model expected signature: model(cin_x, c_noise) -> prediction
            pred = model(cin_x, c_noise.to(device))
        x_denoised = c_skip.view(-1, 1, 1, 1) * x + c_out.view(-1, 1, 1, 1) * pred
        d = (x - x_denoised) / sigma_scalar.view(1, 1, 1, 1)
        x = x + d * (sigma_next - sigma_scalar).view(1, 1, 1, 1)
    return x


def load_ckpt_into_model(model, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    state = ck.get("model_state", ck)
    try:
        model.load_state_dict(state)
    except Exception:
        # tolerate wrapped dicts
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            raise
    return model


def find_latest_checkpoint(checkpoints_dir: str = "checkpoints") -> Optional[str]:
    # prefer explicit latest.pt, else pick most recently modified model_epoch_*.pth/.pt
    latest_pt = os.path.join(checkpoints_dir, "latest.pt")
    if os.path.exists(latest_pt):
        return latest_pt
    patterns = [os.path.join(checkpoints_dir, "model_epoch_*.pth"),
                os.path.join(checkpoints_dir, "model_epoch_*.pt"),
                os.path.join(checkpoints_dir, "*.pth"),
                os.path.join(checkpoints_dir, "*.pt")]
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(p))
    if not candidates:
        return None
    # return most recently modified
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return candidates[0]


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
    dl, info = load_dataset_and_make_dataloaders(dataset_name=args.dataset, root_dir=args.data_root, batch_size=1)
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
            raise FileNotFoundError(f"No checkpoint found at {args.ckpt} and no files in checkpoints/")

    model = build_model_for_sampling(info, device)
    model = load_ckpt_into_model(model, ckpt, device)
    sigmas = build_sigma_schedule(args.steps, rho=args.rho)
    sigmas = sigmas.to(device)

    samples = euler_sample(model, sigmas, args.n, channels, H, sigma_data, device)

    # convert from normalized [-1,1] to [0,1]
    imgs = samples.clamp(-1, 1).add(1).div(2)
    grid = make_grid(imgs, nrow=min(8, args.n))
    out_path = os.path.join(args.outdir, "samples_grid.png")
    save_image(grid, out_path)
    print("Saved samples to", out_path)


if __name__ == "__main__":
    main()