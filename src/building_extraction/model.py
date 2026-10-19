import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DroneUNet(nn.Module):
    """
    Lightweight, highly performant U-Net architecture optimized for
    high-resolution drone aerial feature segmentation.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_features: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder
        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features, base_features * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 2, base_features * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 4, base_features * 8))

        # Bottleneck
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features * 8, base_features * 16))

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_features * 16, base_features * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_features * 16, base_features * 8)

        self.up2 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_features * 8, base_features * 4)

        self.up3 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_features * 4, base_features * 2)

        self.up4 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(base_features * 2, base_features)

        # Output head
        self.outc = nn.Conv2d(base_features, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        d1 = self.up1(x5)
        d1 = torch.cat([x4, d1], dim=1)
        d1 = self.conv_up1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([x3, d2], dim=1)
        d2 = self.conv_up2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.conv_up3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat([x1, d4], dim=1)
        d4 = self.conv_up4(d4)

        logits = self.outc(d4)
        return logits
