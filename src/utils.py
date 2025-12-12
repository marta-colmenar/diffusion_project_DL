import glob
import logging
import os
import pathlib
import tempfile
from itertools import islice
from pathlib import Path
from typing import Optional

import torch
from cleanfid import fid
from torchvision.utils import save_image

from src.config import Config
from src.data import DataInfo
from src.model import Model

logger = logging.getLogger(__name__)


def to_unit_range(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-1, 1).add(1).div(2)


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


def save_fid_real_stats(valid_loader, dataset_name: str, n: int = 1_000, device="cpu"):
    dname = dataset_name.lower()

    if fid.test_stats_exists(dname, mode="clean"):
        logger.info(
            f"FID real stats for '{dataset_name}' already exist, skipping computation."
        )
        return

    def image_stream():
        for batch in valid_loader:
            imgs = to_unit_range(batch[0])
            for img in imgs:
                yield img

    with tempfile.TemporaryDirectory(
        dir=pathlib.Path(__file__).parent.resolve() / ".." / "data"
    ) as tempdir:

        for idx, img in enumerate(islice(image_stream(), n)):
            save_image(img, Path(tempdir) / f"real_{idx:05d}.png")
        fid.make_custom_stats(dname, tempdir, mode="clean", device=torch.device(device))

    assert fid.test_stats_exists(dname, mode="clean"), "Failed to save FID real stats."


def compute_fid(gen_path, dataset_name, device, num_workers=4):
    return fid.compute_fid(
        gen_path,
        dataset_name=dataset_name.lower(),
        device=device,
        mode="clean",
        dataset_split="custom",
        num_workers=num_workers,
    )
