import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=61, base_filters=32):
        super().__init__()
        f = base_filters
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DoubleConv(f, f*2)
        self.enc3 = DoubleConv(f*2, f*4)
        self.enc4 = DoubleConv(f*4, f*8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(f*8, f*16)
        self.up4 = nn.ConvTranspose2d(f*16, f*8, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(f*8, f*4, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(f*4, f*2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(f*2, f, 2, stride=2)
        self.dec4 = DoubleConv(f*16, f*8)
        self.dec3 = DoubleConv(f*8, f*4)
        self.dec2 = DoubleConv(f*4, f*2)
        self.dec1 = DoubleConv(f*2, f)
        self.final = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return out
