import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as FT


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv0 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        return x


class ResConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        res_crop = FT.center_crop(res, x.shape[-2:])
        x = torch.cat((res_crop, x), dim=1)
        x = self.conv(x)
        return x


class DownsampleBlock(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, 2)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


class UNet(nn.Module):
    """
    Original implementation based on: https://arxiv.org/abs/1505.04597
    """

    def __init__(
        self, in_channels: int = 1, out_channels: int = 2, hidden_channels: int = 64
    ):
        super().__init__()

        self.depth0_conv = ConvBlock(in_channels, hidden_channels)

        self.depth1_down = DownsampleBlock()
        self.depth1_conv = ConvBlock(hidden_channels, hidden_channels * 2)

        self.depth2_down = DownsampleBlock()
        self.depth2_conv = ConvBlock(hidden_channels * 2, hidden_channels * 4)

        self.depth3_down = DownsampleBlock()
        self.depth3_conv = ConvBlock(hidden_channels * 4, hidden_channels * 8)

        self.depth4_down = DownsampleBlock()
        self.depth4_conv = ConvBlock(hidden_channels * 8, hidden_channels * 16)

        self.depth3_up = UpsampleBlock(hidden_channels * 16, hidden_channels * 8)
        self.depth3_res_conv = ResConvBlock(hidden_channels * 16, hidden_channels * 8)

        self.depth2_up = UpsampleBlock(hidden_channels * 8, hidden_channels * 4)
        self.depth2_res_conv = ResConvBlock(hidden_channels * 8, hidden_channels * 4)

        self.depth1_up = UpsampleBlock(hidden_channels * 4, hidden_channels * 2)
        self.depth1_res_conv = ResConvBlock(hidden_channels * 4, hidden_channels * 2)

        self.depth0_up = UpsampleBlock(hidden_channels * 2, hidden_channels)
        self.depth0_res_conv = ResConvBlock(hidden_channels * 2, hidden_channels)

        self.out_conv = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        depth0 = self.depth0_conv(x)
        assert depth0.shape[-1] == 568

        depth1 = self.depth1_conv(self.depth1_down(depth0))
        assert depth1.shape[-1] == 280

        depth2 = self.depth2_conv(self.depth2_down(depth1))
        assert depth2.shape[-1] == 136

        depth3 = self.depth3_conv(self.depth3_down(depth2))
        assert depth3.shape[-1] == 64

        depth4 = self.depth4_conv(self.depth4_down(depth3))
        assert depth4.shape[-1] == 28

        depth3_res = self.depth3_up(depth4)
        depth3_res = self.depth3_res_conv(depth3_res, depth3)
        assert depth3_res.shape[-1] == 52

        depth2_res = self.depth2_up(depth3_res)
        depth2_res = self.depth2_res_conv(depth2_res, depth2)
        assert depth2_res.shape[-1] == 100

        depth1_res = self.depth1_up(depth2_res)
        depth1_res = self.depth1_res_conv(depth1_res, depth1)
        assert depth1_res.shape[-1] == 196

        depth0_res = self.depth0_up(depth1_res)
        depth0_res = self.depth0_res_conv(depth0_res, depth0)
        depth0_res = self.out_conv(depth0_res)
        assert depth0_res.shape[-1] == 388

        return depth0_res


if __name__ == "__main__":
    model = UNet()
    x = torch.rand(1, 1, 572, 572)
    out = model(x)
    assert out.shape == (1, 2, 388, 388)
