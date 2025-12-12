import logging
import os
import shutil
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

from src.common import c_funcs, euler_sample
from src.config import Config
from src.data import load_dataset_and_make_dataloaders
from src.model import get_model
from src.sigma import build_sigma_schedule, sample_sigma
from src.utils import compute_fid, save_fid_real_stats, to_unit_range

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


def add_noise(y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # y: (B,C,H,W), sigma: (B,)
    eps = torch.randn_like(y)
    return y + sigma.view(-1, 1, 1, 1) * eps


def train_model(config_path: str = "configs/train.yaml"):

    cfg = Config.load_from_yaml(Path(config_path))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = timestamp + "_" + cfg.training.run_name
    run_dir = Path("runs") / run_name
    checkpoint_dir = run_dir / "checkpoints"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    shutil.copy2(Path(config_path), run_dir / "config.yaml")

    fid_freq = cfg.training.fid_eval_freq
    checkpoint_freq = cfg.training.checkpoint_freq

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    dataloaders, info = load_dataset_and_make_dataloaders(
        dataset_name=cfg.data.dataset_name,
        root_dir=str(cfg.data.data_root),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    train_loader = dataloaders.train
    valid_loader = dataloaders.valid
    sigma_data = float(info.sigma_data)

    if cfg.training.evaluation_metric == "FID":
        logger.info("Checking for real FID stats...")
        save_fid_real_stats(
            valid_loader,
            cfg.data.dataset_name,
            device=device_str,
            n=len(valid_loader.dataset),
        )

    num_classes = info.num_classes if cfg.model.cf_guidance else 0
    model = get_model(
        model_name=cfg.model.model_name,
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
    epoch_losses = []
    epoch_fids = []
    for epoch in range(num_epochs):

        epoch_loss = 0.0
        for batch in train_loader:

            y, labels = batch[0].to(device), batch[1].to(device)

            # Apply classifier-free guidance dropout
            if cfg.model.cf_guidance:
                labels = labels.clone()
                use_uncond = torch.rand(labels.size(), device=labels.device) < cfg.training.cfg_drop_prob  # type: ignore
                labels[use_uncond] = -1

            b = y.size(0)
            # sample per-sample sigma
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
        epoch_losses.append((epoch, avg_epoch_loss))
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

                zip_path = Path(run_dir) / "fid_samples" / f"epoch_{epoch+1}.zip"
                Path(zip_path.parent).mkdir(parents=True, exist_ok=True)

                # generate fake images and save
                batch_sz = cfg.data.batch_size
                steps = cfg.training.fid_steps
                rho = 7.0  # common choice from Karras et al.
                sigmas = build_sigma_schedule(
                    steps, rho, cfg.diffusion.sigma_min, cfg.diffusion.sigma_max
                ).to(device)

                with zipfile.ZipFile(
                    zip_path, "w", compression=zipfile.ZIP_DEFLATED
                ) as zf:
                    produced = 0
                    while produced < cfg.training.fid_num_samples:
                        b = min(batch_sz, cfg.training.fid_num_samples - produced)
                        # unconditioned generation
                        xgen = euler_sample(
                            model,
                            sigmas,
                            b,
                            info.image_channels,
                            info.image_size,
                            sigma_data,
                            device,
                        )
                        imgs = to_unit_range(xgen)

                        for i in range(imgs.size(0)):
                            buf = BytesIO()
                            save_image(imgs[i], buf, format="PNG")
                            buf.seek(0)
                            zf.writestr(f"sample_{produced:05d}.png", buf.read())
                            produced += 1

                logger.info(f"Saved {produced} generated images to {zip_path}")
                fid = compute_fid(
                    str(zip_path), cfg.data.dataset_name, device=device_str
                )
                # FIXME: if we want to keep it during training, we might want to save FIDs to a file
                logger.info(f"FID: {fid:.4f}")
                epoch_fids.append((epoch, fid))
                model.train()

    with open(run_dir / "losses.txt", "w") as f:
        for epoch, loss in epoch_losses:
            f.write(f"{epoch:04d}, {loss:.6f}\n")

    if len(epoch_fids) > 0:
        with open(run_dir / "fids.txt", "w") as f:
            for epoch, fid in epoch_fids:
                f.write(f"{epoch:04d}, {fid:.6f}\n")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/train.yaml")
    args = p.parse_args()

    train_model(args.config)
