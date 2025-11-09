import os
import sys
import yaml
from pathlib import Path
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset_and_make_dataloaders
from src.model import Model
from src.sigma import sample_sigma
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
    if hasattr(dl, 'train'):
        return dl.train
    return dl  # assume it's already an iterable


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
    sigma_data = float(info.sigma_data)

    model = Model(
        image_channels=getattr(info, "channels", 1),
        nb_channels=cfg.get("nb_channels", 64),
        num_blocks=cfg.get("num_blocks", 4),
        cond_channels=cfg.get("cond_channels", 64),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.get("learning_rate", 1e-3), betas=(0.9, 0.999))
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

        # optional FID evaluation (run every fid_eval_freq epochs)
        if cfg.get("evaluation_metric", "") == "FID":
            fid_freq = cfg.get("fid_eval_freq", 5)
            if (epoch + 1) % fid_freq == 0:
                model.eval()
                outdir = Path(cfg.get("checkpoint_dir", "checkpoints")) / "fid_samples" / f"epoch_{epoch+1}"
                outdir.mkdir(parents=True, exist_ok=True)

                # generate N samples and save to outdir (uses your sampler)
                from src.sample import build_sigma_schedule, euler_sample
                n_samples = cfg.get("fid_num_samples", 500)
                batch = cfg.get("fid_batch_size", 64)
                sigmas = build_sigma_schedule(cfg.get("fid_steps", 50), rho=cfg.get("fid_rho", 7)).to(device)
                generated = []
                with torch.no_grad():
                    for i in range(0, n_samples, batch):
                        b = min(batch, n_samples - i)
                        x = euler_sample(model, sigmas, b, info.image_channels, info.image_size, float(info.sigma_data), device)
                        # convert to [0,1]
                        imgs = x.clamp(-1, 1).add(1).div(2)
                        for j in range(imgs.size(0)):
                            save_image(imgs[j], outdir / f"sample_{i+j:05d}.png")

                model.train()
                # compute FID using pytorch-fid CLI (install: pip install pytorch-fid)
                real_dir = cfg.get("fid_real_dir", "data/fid_real")
                try:
                    subprocess.check_call([
                        "pytorch-fid", str(real_dir), str(outdir),
                        "--device", "cuda" if torch.cuda.is_available() else "cpu"
                    ])
                except FileNotFoundError:
                    print("pytorch-fid not installed; install with `pip install pytorch-fid` to compute FID")

    return model


if __name__ == "__main__":
    train_model("configs/train.yaml")
