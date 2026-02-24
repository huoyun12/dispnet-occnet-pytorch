import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class OccNet(nn.Module):
    def __init__(self, in_channels=7, max_channels=64):
        super(OccNet, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.enc2 = nn.Sequential(
            ResidualBlock2D(32, 32),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.enc3 = nn.Sequential(
            ResidualBlock2D(64, 64),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.enc4 = nn.Sequential(
            ResidualBlock2D(128, 128),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.bottleneck = ResidualBlock2D(256, 256)

        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock2D(128, 128)
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock2D(64, 64)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock2D(32, 32)
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock2D(32, 32)
        )

        self.output = nn.Sequential(
            nn.Conv2d(32, 2, 3, 1, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.dec4(b)
        d4 = torch.cat([d4, e3], dim=1)

        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)

        d1 = self.dec1(d2)

        out = self.output(d1)

        occ_left = out[:, 0:1, :, :]
        occ_right = out[:, 1:2, :, :]

        return occ_left, occ_right


def test_occnet():
    model = OccNet(in_channels=7, max_channels=64)

    B, C, H, W = 2, 7, 256, 256
    x = torch.randn(B, C, H, W)

    occ_left, occ_right = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Occ Left shape: {occ_left.shape}")
    print(f"Occ Right shape: {occ_right.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.3f}M")

    return model


if __name__ == "__main__":
    test_occnet()
