import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

from src.common import c_funcs, euler_sample
from src.config import Config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

from src.data import load_dataset_and_make_dataloaders
from src.model import Model
from src.sigma import build_sigma_schedule, sample_sigma


def add_noise(y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # y: (B,C,H,W), sigma: (B,)
    eps = torch.randn_like(y)
    return y + sigma.view(-1, 1, 1, 1) * eps


def _get_train_loader(dl):
    # support both dict and simple namespace with .train
    if isinstance(dl, dict) and "train" in dl:
        return dl["train"]
    if (train := getattr(dl, "train", None)) is not None:
        return train
    return dl  # assume it's already an iterable


def _get_valid_loader(dl):
    if isinstance(dl, dict) and "valid" in dl:
        return dl["valid"]
    if (valid := getattr(dl, "valid", None)) is not None:
        return valid
    # if only one loader provided, return it (best-effort)
    return dl


# helper to export real images for FID
def export_real_images(valid_loader, outdir: Union[Path, str], n: int = 500):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for batch in valid_loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        # TODO: image processing can be unified later, it's repeated
        imgs = x.clamp(-1, 1).add(1).div(2)  # to [0,1]
        for i in range(imgs.size(0)):
            if saved >= n:
                return saved
            save_image(imgs[i], outdir / f"real_{saved:05d}.png")
            saved += 1
    return saved


def train_model(config_path: str = "configs/train.yaml") -> Model:

    cfg = Config.load_from_yaml(Path(config_path))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = timestamp + "_" + cfg.training.run_name
    run_dir = Path("runs") / run_name
    checkpoint_dir = run_dir / "checkpoints"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    shutil.copy2(Path(config_path), run_dir / "config.yaml")

    fid_freq = cfg.training.fid_eval_freq
    checkpoint_freq = cfg.training.checkpoint_freq

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders, info = load_dataset_and_make_dataloaders(
        dataset_name=cfg.data.dataset_name,
        root_dir=str(cfg.data.data_root),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    train_loader = _get_train_loader(dataloaders)
    valid_loader = _get_valid_loader(dataloaders)
    sigma_data = float(info.sigma_data)

    num_classes = info.num_classes if cfg.model.class_conditioned else 0
    model = Model(
        image_channels=getattr(info, "image_channels", 1),
        nb_channels=cfg.model.nb_channels,
        num_blocks=cfg.model.num_blocks,
        cond_channels=cfg.model.cond_channels,
        conditioned=cfg.model.conditioned,
        num_classes=num_classes,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    num_epochs = cfg.training.num_epochs

    model.train()
    for epoch in range(num_epochs):
        # pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        for batch in train_loader:

            y, labels = batch[0].to(device), batch[1].to(device)
            b = y.size(0)
            # sample per-sample sigma
            # TODO: does it make sense that sigma_min and sigma_max are same as sampling?
            sigma = sample_sigma(
                b, sigma_min=cfg.diffusion.sigma_min, sigma_max=cfg.diffusion.sigma_max
            ).to(
                device
            )  # (B,)
            x = add_noise(y, sigma)

            c_in, c_out, c_skip, c_noise = c_funcs(sigma, sigma_data)
            cin_x = c_in.view(-1, 1, 1, 1) * x

            # forward: model(cin * x, c_noise)
            # labels are internally ignored if model is not class-conditioned
            pred = model(cin_x, c_noise.to(device), labels=labels)

            # target = (y - c_skip * x) / c_out
            target = (y - c_skip.view(-1, 1, 1, 1) * x) / c_out.view(-1, 1, 1, 1)

            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_loss = loss.item()
            epoch_loss += step_loss

        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} avg loss: {avg_epoch_loss:.4f}")

        # save checkpoint
        if (epoch + 1) % checkpoint_freq == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                ckpt_path,
            )
            logger.info(f"Saved checkpoint: {ckpt_path}")

        # Optional FID evaluation (after checkpoint saved)
        if cfg.training.evaluation_metric == "FID":
            if (epoch + 1) % fid_freq == 0:
                model.eval()
                real_dir = cfg.data.data_root / (cfg.data.dataset_name + "_fid_real")
                fake_dir = os.path.join(run_dir, "fid_samples", f"epoch_{epoch+1}")
                Path(fake_dir).mkdir(parents=True, exist_ok=True)

                # export real images if missing
                # FIXME: the loop won't enter if fid_num_samples changed after first run
                if not Path(real_dir).exists() or not any(Path(real_dir).iterdir()):
                    logger.info("Exporting real images for FID into", real_dir)
                    saved = export_real_images(
                        valid_loader, real_dir, n=cfg.training.fid_num_samples
                    )
                    logger.info(f"Exported {saved} real images for FID")

                # generate fake images and save
                batch_sz = cfg.data.batch_size
                steps = cfg.training.fid_steps
                rho = 7.0  # common choice from Karras et al.
                sigmas = build_sigma_schedule(
                    steps, rho, cfg.diffusion.sigma_min, cfg.diffusion.sigma_max
                ).to(device)

                produced = 0
                while produced < cfg.training.fid_num_samples:
                    b = min(batch_sz, cfg.training.fid_num_samples - produced)
                    xgen = euler_sample(
                        model,
                        sigmas,
                        b,
                        info.image_channels,
                        info.image_size,
                        sigma_data,
                        device,
                    )
                    imgs = xgen.clamp(-1, 1).add(1).div(2)  # to [0,1]
                    for i in range(imgs.size(0)):
                        save_image(
                            imgs[i], Path(fake_dir) / f"sample_{produced:05d}.png"
                        )
                        produced += 1

                logger.info(f"Saved {produced} generated images to {fake_dir}")

                # call pytorch-fid CLI
                # FIXME: run this in a separate function and avoid training all the time..
                try:
                    subprocess.check_call(
                        [
                            "pytorch-fid",
                            "--device",
                            "cuda" if torch.cuda.is_available() else "cpu",
                            str(real_dir),
                            str(fake_dir),
                        ]
                    )
                except FileNotFoundError:
                    logger.error(
                        "pytorch-fid not installed; install with `pip install pytorch-fid` to compute FID"
                    )
                except subprocess.CalledProcessError as e:
                    logger.error(
                        "pytorch-fid returned non-zero exit code:", e.returncode
                    )
                model.train()

    return model


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/train.yaml")
    args = p.parse_args()

    train_model(args.config)
