from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        image_channels: int,
        nb_channels: int,
        num_blocks: int,
        cond_channels: int,
        conditioned: bool = True,
        num_classes: int = 0,
    ) -> None:
        super().__init__()
        self.conditioned = conditioned
        self.num_classes = num_classes
        self.noise_emb = NoiseEmbedding(cond_channels)

        # Add class embedding if conditioning is used
        if self.conditioned and self.num_classes > 0:
            self.class_emb = ClassEmbedding(num_classes, cond_channels)
        else:
            self.class_emb = None

        self.conv_in = nn.Conv2d(image_channels, nb_channels, kernel_size=3, padding=1)
        self._initialize_blocks(nb_channels, cond_channels, num_blocks, conditioned)
        self.conv_out = nn.Conv2d(nb_channels, image_channels, kernel_size=3, padding=1)

        # According to assignment, we could initialize last conv to zero when conditioned
        if self.conditioned:
            nn.init.zeros_(self.conv_out.weight)

    def _initialize_blocks(
        self, nb_channels, cond_channels, num_blocks, conditioned=True
    ):
        if conditioned:
            self.blocks = nn.ModuleList(
                [
                    CondResidualBlock(nb_channels, cond_channels)
                    for _ in range(num_blocks)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [ResidualBlock(nb_channels) for _ in range(num_blocks)]
            )

    def forward(
        self,
        noisy_input: torch.Tensor,
        c_noise: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        cond = self.noise_emb(c_noise)

        if self.class_emb is not None:
            if labels is not None:
                # labels: (B,)
                # unconditional samples marked with -1
                mask = labels == -1
                # create full embedding, but fill unconditional ones with zeros
                class_cond = torch.zeros_like(cond)
                if (~mask).any():
                    class_cond[~mask] = self.class_emb(labels[~mask])
            else:
                class_cond = torch.zeros_like(cond)

            cond += class_cond

        x = self.conv_in(noisy_input)
        for block in self.blocks:
            if isinstance(block, CondResidualBlock):
                x = block(x, cond)
            else:
                x = block(x)
        return self.conv_out(x)


class NoiseEmbedding(nn.Module):
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer("weight", torch.randn(1, cond_channels // 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 1
        f = 2 * torch.pi * input.unsqueeze(1) @ self.weight  # type: ignore
        return torch.cat([f.cos(), f.sin()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, nb_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(nb_channels)
        self.conv1 = nn.Conv2d(
            nb_channels, nb_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.BatchNorm2d(nb_channels)
        self.conv2 = nn.Conv2d(
            nb_channels, nb_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.relu(self.norm1(x)))
        y = self.conv2(F.relu(self.norm2(y)))
        return x + y


class CondBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, cond_channels: int):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.linear = nn.Linear(cond_channels, 2 * num_features)

        nn.init.zeros_(self.linear.weight)
        # bias = [gamma_bias | beta_bias]
        self.linear.bias.data[:num_features].fill_(1.0)
        self.linear.bias.data[num_features:].zero_()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # cond: (B, cond_channels)
        out = self.bn(x)
        gamma, beta = self.linear(cond)[:, :, None, None].split(
            self.num_features, dim=1
        )  # each (B, C)
        return gamma * out + beta


class CondResidualBlock(nn.Module):
    def __init__(self, nb_channels: int, cond_channels: int):
        super().__init__()
        self.norm1 = CondBatchNorm2d(nb_channels, cond_channels)
        self.conv1 = nn.Conv2d(
            nb_channels, nb_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = CondBatchNorm2d(nb_channels, cond_channels)
        self.conv2 = nn.Conv2d(
            nb_channels, nb_channels, kernel_size=3, stride=1, padding=1
        )

        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.relu(self.norm1(x, cond)))
        h = self.conv2(F.relu(self.norm2(h, cond)))
        return x + h


class ClassEmbedding(nn.Module):
    def __init__(self, num_classes: int, cond_channels: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes, cond_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Input is expected to be a tensor of integer labels (B,)
        assert input.ndim == 1
        return self.embedding(input)
