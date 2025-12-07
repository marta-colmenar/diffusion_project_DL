from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Configuration for data loading."""

    dataset_name: str = "FashionMNIST"
    data_root: Path = Path("data")
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True


class ModelConfig(BaseModel):
    """Configuration for the neural network architecture."""

    nb_channels: int = 64
    num_blocks: int = 4
    cond_channels: int = 64
    conditioned: bool = True  # Noise conditioning
    class_conditioned: bool = False  # Class label conditioning


class DiffusionConfig(BaseModel):
    """Configuration for sigma schedules and noise parameters."""

    sigma_min: float = 0.002
    sigma_max: float = 80.0


class TrainingConfig(BaseModel):
    """Configuration specific to the training loop hyperparameters."""

    # Moved parameters here:
    log_interval: int = 10
    fid_steps: int = 20
    fid_num_samples: int = 100

    run_name: str = "fashion_mnist"
    fid_eval_freq: int = 10
    evaluation_metric: Optional[str] = ""
    checkpoint_freq: int = 10
    learning_rate: float = 1e-4
    num_epochs: int = 100


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
        return cls(**raw_config)

    def save_to_yaml(self, path: Path):
        """Saves the configuration to a YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f)
