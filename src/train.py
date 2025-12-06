import os
import math
import sys
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import subprocess
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset_and_make_dataloaders
from src.model import Model
from src.sigma import sample_sigma, build_sigma_schedule
from src.losses import mse_loss


def c_funcs(sigma: torch.Tensor, sigma_data: float):
    # sigma: (B,)
    sd = sigma_data
    denom = torch.sqrt(sd ** 2 + sigma ** 2)
    c_in = 1.0 / denom
    c_out = sigma * sd / denom
    c_skip = (sd ** 2) / (sd ** 2 + sigma ** 2)
    c_noise = torch.log(sigma) / 4.0
    return c_in, c_out, c_skip, c_noise


def add_noise(y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # y: (B,C,H,W), sigma: (B,)
    eps = torch.randn_like(y)
    return y + sigma.view(-1, 1, 1, 1) * eps


def _get_train_loader(dl):
    # support both dict and simple namespace with .train
    if isinstance(dl, dict) and 'train' in dl:
        return dl['train']
    if (train := getattr(dl, 'train', None)) is not None:
        return train
    return dl  # assume it's already an iterable


def _get_valid_loader(dl):
    if isinstance(dl, dict) and 'valid' in dl:
        return dl['valid']
    if (valid := getattr(dl, 'valid', None)) is not None:
        return valid
    # if only one loader provided, return it (best-effort)
    return dl

# helper to export real images for FID
def export_real_images(valid_loader, outdir: str, n: int = 500):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for batch in valid_loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        imgs = x.clamp(-1, 1).add(1).div(2)  # to [0,1]
        for i in range(imgs.size(0)):
            if saved >= n:
                return saved
            save_image(imgs[i], outdir / f"real_{saved:05d}.png")
            saved += 1
    return saved

# simple Euler sampler used for FID generation inside training
def euler_sample_batch(model, sigmas, n, channels, H, sigma_data, device):
    model.eval()
    with torch.no_grad():
        x = torch.randn(n, channels, H, H, device=device) * sigmas[0].to(device)
        for i, sigma in enumerate(sigmas):
            sigma = sigma.to(device)
            sigma_next = sigmas[i + 1].to(device) if i + 1 < len(sigmas) else torch.tensor(0.0, device=device)
            sigma_b = sigma.repeat(n)
            denom = torch.sqrt(sigma_data ** 2 + sigma_b ** 2)
            c_in = 1.0 / denom
            c_out = sigma_b * sigma_data / denom
            c_skip = (sigma_data ** 2) / (sigma_data ** 2 + sigma_b ** 2)
            c_noise = torch.log(sigma_b) / 4.0
            cin_x = c_in.view(-1, 1, 1, 1) * x
            pred = model(cin_x, c_noise.to(device))
            x_denoised = c_skip.view(-1, 1, 1, 1) * x + c_out.view(-1, 1, 1, 1) * pred
            d = (x - x_denoised) / sigma.view(1, 1, 1, 1)
            x = x + d * (sigma_next - sigma).view(1, 1, 1, 1)
    model.train()
    return x

def train_model(config_path: str = "configs/train.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders, info = load_dataset_and_make_dataloaders(
        dataset_name=cfg.get("dataset_name", "FashionMNIST"),
        root_dir=cfg.get("data_root", "data"),
        batch_size=cfg.get("batch_size", 32),
        num_workers=cfg.get("num_workers", 0),
        pin_memory=cfg.get("pin_memory", False),
    )

    train_loader = _get_train_loader(dataloaders)
    valid_loader = _get_valid_loader(dataloaders)
    sigma_data = float(info.sigma_data)

    model = Model(
        image_channels=getattr(info, "image_channels", 1),
        nb_channels=cfg.get("nb_channels", 64),
        num_blocks=cfg.get("num_blocks", 4),
        cond_channels=cfg.get("cond_channels", 64),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.get("learning_rate", 1e-3))
    num_epochs = cfg.get("num_epochs", 1)
    checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                y = batch[0].to(device)
            else:
                y = batch.to(device)

            b = y.size(0)
            # sample per-sample sigma
            sigma = sample_sigma(b).to(device)  # (B,)
            x = add_noise(y, sigma)

            c_in, c_out, c_skip, c_noise = c_funcs(sigma, sigma_data)
            cin_x = c_in.view(-1, 1, 1, 1) * x

            # forward: model(cin * x, c_noise)
            pred = model(cin_x, c_noise.to(device))

            # target = (y - c_skip * x) / c_out
            target = (y - c_skip.view(-1, 1, 1, 1) * x) / c_out.view(-1, 1, 1, 1)

            loss = mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_loss = loss.item()
            epoch_loss += step_loss
            pbar.set_postfix({"loss": f"{step_loss:.4f}"})

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} avg loss: {avg_epoch_loss:.4f}")

        # save checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save({"epoch": epoch + 1, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # Optional FID evaluation (after checkpoint saved)
        if cfg.get("evaluation_metric", "") == "FID":
            fid_freq = cfg.get("fid_eval_freq", 1)
            if (epoch + 1) % fid_freq == 0:
                model.eval()
                real_dir = cfg.get("fid_real_dir", "data/fid_real")
                fake_dir = os.path.join(checkpoint_dir, "fid_samples", f"epoch_{epoch+1}")
                Path(fake_dir).mkdir(parents=True, exist_ok=True)

                # export real images if missing
                fid_num = cfg.get("fid_num_samples", 500)
                if not Path(real_dir).exists() or not any(Path(real_dir).iterdir()):
                    print("Exporting real images for FID into", real_dir)
                    saved = export_real_images(valid_loader, real_dir, n=fid_num)
                    print(f"Exported {saved} real images for FID")

                # generate fake images and save
                batch_sz = cfg.get("fid_batch_size", 64)
                steps = cfg.get("fid_steps", 50)
                rho = cfg.get("fid_rho", 7.0)
                sigmas = build_sigma_schedule(steps, rho=rho).to(device)

                produced = 0
                while produced < fid_num:
                    b = min(batch_sz, fid_num - produced)
                    xgen = euler_sample_batch(model, sigmas, b, info.image_channels, info.image_size, sigma_data, device)
                    imgs = xgen.clamp(-1, 1).add(1).div(2)  # to [0,1]
                    for i in range(imgs.size(0)):
                        save_image(imgs[i], Path(fake_dir) / f"sample_{produced:05d}.png")
                        produced += 1

                print(f"Saved {produced} generated images to {fake_dir}")

                # call pytorch-fid CLI
                try:
                    subprocess.check_call([
                        "pytorch-fid", str(real_dir), str(fake_dir),
                        "--device", "cuda" if torch.cuda.is_available() else "cpu"
                    ])
                except FileNotFoundError:
                    print("pytorch-fid not installed; install with `pip install pytorch-fid` to compute FID")
                except subprocess.CalledProcessError as e:
                    print("pytorch-fid returned non-zero exit code:", e.returncode)
                model.train()

    return model


if __name__ == "__main__":
    train_model("configs/train.yaml")
