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
    ) -> None:
        super().__init__()
        self.noise_emb = NoiseEmbedding(cond_channels)
        self.conv_in = nn.Conv2d(image_channels, nb_channels, kernel_size=3, padding=1)
        self._initialize_blocks(nb_channels, cond_channels, num_blocks, conditioned)
        self.conv_out = nn.Conv2d(nb_channels, image_channels, kernel_size=3, padding=1)

        nn.init.zeros_(self.conv_out.weight)

    def _initialize_blocks(self, nb_channels, cond_channels, num_blocks, conditioned=True):
        if conditioned:
            self.blocks = nn.ModuleList([CondResidualBlock(nb_channels, cond_channels) for _ in range(num_blocks)])
        else:
            self.blocks = nn.ModuleList([ResidualBlock(nb_channels) for _ in range(num_blocks)])

    def forward(self, noisy_input: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        cond = self.noise_emb(c_noise) # TODO: not used yet
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
        self.register_buffer('weight', torch.randn(1, cond_channels // 2))
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 1
        f = 2 * torch.pi * input.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, nb_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(nb_channels)
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(nb_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.relu(self.norm1(x)))
        y = self.conv2(F.relu(self.norm2(y)))
        return x + y
    
class CondBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, cond_channels: int, eps=1e-5, momentum=0.1):
        super().__init__()
        # Base BatchNorm without affine params (we’ll supply those from cond)
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=False)
        
        # Linear layer maps cond → [gamma, beta]
        self.modulation = nn.Linear(cond_channels, 2 * num_features)
        
        # Initialize modulation so that initially gamma ≈ 1, beta ≈ 0
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        nn.init.constant_(self.modulation.bias[:num_features], 1.) 

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # cond: (B, cond_channels)
        out = self.bn(x)
        params = self.modulation(cond)  # (B, 2 * C)
        B, C = x.shape[0], x.shape[1]
        gamma, beta = params[:, :C], params[:, C:]
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        return gamma * out + beta

class CondResidualBlock(nn.Module):
    def __init__(self, nb_channels: int, cond_channels: int):
        super().__init__()
        self.norm1 = CondBatchNorm2d(nb_channels, cond_channels)
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = CondBatchNorm2d(nb_channels, cond_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)

        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x, cond)))
        h = self.conv2(F.silu(self.norm2(h, cond)))
        return x + h

