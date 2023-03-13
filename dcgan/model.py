import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
        self, latent_dim: int = 100, hidden_size: int = 128, num_channels: int = 3
    ):
        super().__init__()
        self.model = nn.Sequential(
            Generator.block(latent_dim, hidden_size * 8, 4, 1, 0),
            Generator.block(hidden_size * 8, hidden_size * 4, 4, 2, 1),
            Generator.block(hidden_size * 4, hidden_size * 2, 4, 2, 1),
            Generator.block(hidden_size * 2, hidden_size, 4, 2, 1),
            nn.ConvTranspose2d(hidden_size, num_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.model(latent)

    @staticmethod
    def block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 2,
    ) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class Discriminator(nn.Module):
    def __init__(self, hidden_size: int = 128, num_channels: int = 3):
        super().__init__()
        self.model = nn.Sequential(
            Discriminator.block(num_channels, hidden_size, 4, 2, 1),
            Discriminator.block(hidden_size, hidden_size * 2, 4, 2, 1),
            Discriminator.block(hidden_size * 2, hidden_size * 4, 4, 2, 1),
            Discriminator.block(hidden_size * 4, hidden_size * 8, 4, 2, 1),
            nn.Conv2d(
                hidden_size * 8, 1, kernel_size=4, stride=1, padding=0, bias=False
            ),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image).view(-1)

    @staticmethod
    def block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 2,
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )


def weights_init(module: nn.Module):
    module_name = module.__class__.__name__
    if module_name.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif module_name.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)
