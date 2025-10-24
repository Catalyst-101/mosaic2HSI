import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNReconstruct(nn.Module):
    """
    Simple CNN model for reconstructing hyperspectral images from mosaic inputs.
    Input:  (B, 1, H, W)
    Output: (B, 61, H, W)
    """

    def __init__(self, in_channels=1, out_channels=61, base_channels=64):
        super(CNNReconstruct, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)

        # Bottleneck
        b = self.bottleneck(e2)

        # Decoder
        d1 = self.dec1(b)
        out = self.dec2(d1)

        return out


if __name__ == "__main__":
    # Test model
    model = CNNReconstruct()
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print("Input:", x.shape)
    print("Output:", y.shape)
