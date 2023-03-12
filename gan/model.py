import operator
from functools import reduce

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
        self, latent_dim: int, im_size: tuple[int, ...], hidden_dim: tuple[int, ...]
    ):
        super().__init__()

        im_flatten_size = reduce(operator.mul, im_size, 1)
        dims = [latent_dim] + list(hidden_dim)
        blocks = [
            Generator.block(dim_in, dim_out) for dim_in, dim_out in zip(dims, dims[1:])
        ]
        self.model = nn.Sequential(
            *blocks, nn.Linear(dims[-1], im_flatten_size), nn.Sigmoid()
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.model(latent)

    @staticmethod
    def block(in_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )


class Discriminator(nn.Module):
    def __init__(self, im_size: tuple[int, ...], hidden_dim: tuple[int, ...]):
        super().__init__()

        im_flatten_image = reduce(operator.mul, im_size, 1)
        dims = [im_flatten_image] + list(hidden_dim)
        blocks = [
            Discriminator.block(dim_in, dim_out)
            for dim_in, dim_out in zip(dims, dims[1:])
        ]
        self.model = nn.Sequential(*blocks, nn.Linear(dims[-1], 1), nn.Sigmoid())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image.view(image.size(0), -1))

    @staticmethod
    def block(in_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LeakyReLU(0.2, inplace=True)
        )
