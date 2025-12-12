import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from src.model import ModelNameEnum

logger = logging.getLogger(__name__)


class DataConfig(BaseModel):
    """Configuration for data loading."""

    dataset_name: str = "FashionMNIST"
    data_root: Path = Path("data")
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True


class ModelConfig(BaseModel):
    """Configuration for the neural network architecture."""

    model_name: ModelNameEnum = ModelNameEnum.BASIC
    nb_channels: int = 64
    num_blocks: int = 4
    cond_channels: int = 64
    conditioned: bool = True  # Noise conditioning
    cf_guidance: bool = False  # Classifier-free guidance


class DiffusionConfig(BaseModel):
    """Configuration for sigma schedules and noise parameters."""

    sigma_min: float = 0.002
    sigma_max: float = 80.0


class TrainingConfig(BaseModel):
    """Configuration specific to the training loop hyperparameters."""

    fid_steps: int = 20
    fid_num_samples: int = 100

    run_name: str = "fashion_mnist"
    fid_eval_freq: int = 10
    evaluation_metric: Optional[str] = ""
    checkpoint_freq: int = 10
    learning_rate: float = 1e-4
    num_epochs: int = 100
    cfg_drop_prob: Optional[float] = None


class Config(BaseModel):
    """The complete, top-level configuration schema."""

    # Top-level shared parameter (save_model was removed)

    # Nested configurations
    training: TrainingConfig
    data: DataConfig
    model: ModelConfig
    diffusion: DiffusionConfig

    @classmethod
    def load_from_yaml(cls, path: Path) -> "Config":
        """Loads and validates a YAML configuration file."""
        import yaml

        with open(path, "r") as f:
            raw_config = yaml.safe_load(f)
        res = cls(**raw_config)

        if not res.model.cf_guidance and res.training.cfg_drop_prob is not None:
            logger.warning(
                "cfg_drop_prob is set but cf_guidance is False. cfg_drop_prob will be ignored."
            )
        if res.model.cf_guidance and res.training.cfg_drop_prob is None:
            logger.error(
                "cf_guidance is True but cfg_drop_prob is not set. Consider setting cfg_drop_prob for classifier-free guidance dropout."
            )
            raise

        return res
