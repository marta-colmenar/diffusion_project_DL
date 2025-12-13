import logging
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ModelNameEnum(Enum):
    UNET = "unet"
    UNET_ATTENTION = "unet_attention"
    BASIC = "basic"


def get_model(
    model_name: ModelNameEnum,
    image_channels: int,
    nb_channels: int,
    num_blocks: int,
    cond_channels: int,
    conditioned: bool,
    num_classes,
) -> Union["Model", "UNetModel"]:
    if model_name == ModelNameEnum.UNET:
        return UNetModel(
            image_channels=image_channels,
            base_channels=nb_channels,
            channel_mults=[1, 2, 4],
            num_blocks_per_level=num_blocks,
            cond_channels=cond_channels,
            conditioned=conditioned,
            num_classes=num_classes,
        )
    elif model_name == ModelNameEnum.UNET_ATTENTION:
        return UNetModel(
            image_channels=image_channels,
            base_channels=nb_channels,
            channel_mults=[1, 2, 2, 4],
            num_blocks_per_level=num_blocks,
            cond_channels=cond_channels,
            conditioned=conditioned,
            num_classes=num_classes,
            attention_at_blocks=[2],
        )
    elif model_name == ModelNameEnum.BASIC:
        return Model(
            image_channels=image_channels,
            nb_channels=nb_channels,
            num_blocks=num_blocks,
            cond_channels=cond_channels,
            conditioned=conditioned,
            num_classes=num_classes,
        )

    raise ValueError(f"Unknown model name: {model_name}")


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


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        h_norm = self.norm(x)
        q, k, v = self.qkv(h_norm).chunk(3, dim=1)

        q = q.reshape(b, c, h * w).transpose(1, 2)  # (b, hw, c)
        k = k.reshape(b, c, h * w)  # (b, c, hw)
        attn = torch.softmax(q @ k / (c**0.5), dim=-1)

        v = v.reshape(b, c, h * w).transpose(1, 2)
        out = attn @ v
        out = out.transpose(1, 2).reshape(b, c, h, w)

        return x + self.proj(out)


class UNetModel(nn.Module):
    """
    A compact UNet-like model that uses your Residual / CondResidual blocks.
    - image_channels: input/output channels (e.g. 1 or 3)
    - base_channels: number of channels in first layer (e.g. 64)
    - channel_mults: list of multipliers for each down/up stage e.g. [1,2,4]
    - num_blocks_per_level: number of residual blocks per level
    - cond_channels: size of noise embedding
    - conditioned: whether to use conditional blocks
    - num_classes: for class conditioning (optional)
    """

    def __init__(
        self,
        image_channels: int,
        base_channels: int = 64,
        channel_mults: Optional[List[int]] = None,
        num_blocks_per_level: int = 2,
        cond_channels: int = 128,
        conditioned: bool = True,
        num_classes: int = 0,
        attention_at_blocks: List[int] = [],
    ) -> None:
        super().__init__()
        if channel_mults is None:
            channel_mults = [1, 2, 4]

        self.conditioned = conditioned
        self.num_classes = num_classes
        self.noise_emb = NoiseEmbedding(cond_channels)
        self.attention_at_blocks = set(attention_at_blocks)

        if self.conditioned and self.num_classes > 0:
            self.class_emb = ClassEmbedding(num_classes, cond_channels)
        else:
            self.class_emb = None

        self.conv_in = nn.Conv2d(
            image_channels, base_channels, kernel_size=3, padding=1
        )

        self.skip_channels = []
        # encoder / down path
        self.downs = nn.ModuleList()
        in_ch = base_channels
        for ii, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_blocks_per_level):
                blocks.append(
                    CondResidualBlock(in_ch, cond_channels)
                    if conditioned
                    else ResidualBlock(in_ch)
                )
            attn = (
                SelfAttentionBlock(in_ch)
                if ii in self.attention_at_blocks
                else nn.Identity()
            )
            downsample = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
            self.downs.append(
                nn.ModuleDict(
                    {
                        "blocks": blocks,
                        "attn": attn,
                        "down": downsample,
                    }
                )
            )
            self.skip_channels.append(in_ch)
            in_ch = out_ch

        # bottleneck
        self.bottleneck = nn.ModuleList(
            [
                CondResidualBlock(in_ch, cond_channels)
                if conditioned
                else ResidualBlock(in_ch)
                for _ in range(max(1, num_blocks_per_level))
            ]
        )
        self.bottleneck_attn = (
            SelfAttentionBlock(in_ch)
            if len(channel_mults) in self.attention_at_blocks
            else nn.Identity()
        )

        # decoder / up path
        self.ups = nn.ModuleList()
        rev_mults = list(reversed(channel_mults))
        rev_skips = list(reversed(self.skip_channels))
        for up_index, (mult, skip_ch) in enumerate(zip(rev_mults, rev_skips)):
            down_index = len(channel_mults) - 1 - up_index
            out_ch = base_channels * mult
            upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            )
            block_in_ch = skip_ch + out_ch
            # after concatenation with skip, channels = in_ch + skip_ch
            blocks = nn.ModuleList(
                [
                    CondResidualBlock(block_in_ch, cond_channels)
                    if conditioned
                    else ResidualBlock(block_in_ch)
                    for _ in range(num_blocks_per_level)
                ]
            )
            attn = (
                SelfAttentionBlock(block_in_ch)
                if down_index in self.attention_at_blocks
                else nn.Identity()
            )
            self.ups.append(
                nn.ModuleDict({"blocks": blocks, "attn": attn, "up": upsample})
            )
            in_ch = block_in_ch

        self.conv_out = nn.Conv2d(in_ch, image_channels, kernel_size=3, padding=1)
        if self.conditioned:
            nn.init.zeros_(self.conv_out.weight)

    def forward(
        self,
        noisy_input: torch.Tensor,
        c_noise: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        noisy_input: (B, C, H, W)
        c_noise: (B,) scalar timesteps or noise levels
        labels: optional (B,) int labels, -1 for unconditional
        """
        cond = self.noise_emb(c_noise)

        if self.class_emb is not None:
            if labels is not None:
                mask = labels == -1
                class_cond = torch.zeros_like(cond)
                if (~mask).any():
                    class_cond[~mask] = self.class_emb(labels[~mask])
            else:
                class_cond = torch.zeros_like(cond)
            cond = cond + class_cond

        # initial conv
        x = self.conv_in(noisy_input)  # (B, base_channels, H, W)

        skips = []
        for stage in self.downs:
            for block in stage["blocks"]:  # type: ignore
                if isinstance(block, CondResidualBlock):
                    x = block(x, cond)
                else:
                    x = block(x)
            x = stage["attn"](x)  # type: ignore
            skips.append(x)
            x = stage["down"](x)  # type: ignore

        for block in self.bottleneck:
            x = block(x, cond) if self.conditioned else block(x)
        x = self.bottleneck_attn(x)

        for stage in self.ups:
            x = stage["up"](x)  # type: ignore
            skip = skips.pop()

            x = torch.cat([x, skip], dim=1)
            for block in stage["blocks"]:  # type: ignore
                if isinstance(block, CondResidualBlock):
                    x = block(x, cond)
                else:
                    x = block(x)
            x = stage["attn"](x)  # type: ignore

        out = self.conv_out(x)
        return out
