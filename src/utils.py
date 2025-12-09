import glob
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
from pytorch_fid.fid_score import calculate_fid_given_paths, save_fid_stats
from torchvision.utils import save_image

from src.config import Config
from src.data import DataInfo
from src.model import Model

logger = logging.getLogger(__name__)


def build_model_for_sampling(
    cfg_path: str, device: torch.device, ds_info: DataInfo
) -> Model:

    cfg = Config.load_from_yaml(Path(cfg_path))

    nb_channels = cfg.model.nb_channels
    num_blocks = cfg.model.num_blocks
    cond_channels = cfg.model.cond_channels
    image_channels = ds_info.image_channels
    conditioned = cfg.model.conditioned
    label_conditioned = cfg.model.cf_guidance

    m = Model(
        image_channels=image_channels,
        nb_channels=nb_channels,
        num_blocks=num_blocks,
        cond_channels=cond_channels,
        conditioned=conditioned,
        num_classes=ds_info.num_classes if label_conditioned else 0,
    )

    m.to(device)
    m.eval()
    return m


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
    patterns = [
        os.path.join(checkpoints_dir, "model_epoch_*.pth"),
        os.path.join(checkpoints_dir, "model_epoch_*.pt"),
        os.path.join(checkpoints_dir, "*.pth"),
        os.path.join(checkpoints_dir, "*.pt"),
    ]
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(p))
    if not candidates:
        return None
    # return most recently modified
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return candidates[0]


def save_fid_real_stats(valid_loader, output_fname: Path, n: int = 1000, device="cpu"):
    class Done(Exception):
        pass

    with tempfile.TemporaryDirectory(dir="../data") as tempdir:

        try:
            saved = 0
            for batch in valid_loader:

                x = batch[0]
                # TODO: image processing can be unified later, it's repeated
                imgs = x.clamp(-1, 1).add(1).div(2)  # to [0,1]
                for i in range(imgs.size(0)):
                    if saved >= n:
                        raise Done
                    save_image(imgs[i], Path(tempdir) / f"real_{saved:05d}.png")
                    saved += 1

        except Done:
            save_fid_stats([tempdir, output_fname], 50, "cpu", 2048)


def compute_fid(paths, device):
    return calculate_fid_given_paths(paths, 50, device, 2048)
