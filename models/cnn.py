import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv2d → BN → ReLU layers (standard UNet building block)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class SpatialAttention(nn.Module):
   
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze along channel axis: [B, C, H, W] → [B, 1, H, W] each
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)   # [B, 2, H, W]
        return self.sigmoid(self.conv(concat))            # [B, 1, H, W]


class FloodUNet(nn.Module):
    """Legacy UNet (kept for backward compatibility, not used in TriModalFloodNet)."""
    def __init__(self, n_channels=5, n_classes=1):
        super().__init__()
        self.inc   = DoubleConv(n_channels, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.up1   = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv1 = DoubleConv(128, 64)
        self.up2   = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv2 = DoubleConv(64, 32)
        self.up3   = nn.ConvTranspose2d(32, 16, 2, 2)
        self.conv3 = DoubleConv(32, 16)
        self.outc  = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x  = self.conv1(torch.cat([x3, self.up1(x4)], dim=1))
        x  = self.conv2(torch.cat([x2, self.up2(x)], dim=1))
        x  = self.conv3(torch.cat([x1, self.up3(x)], dim=1))
        return torch.sigmoid(self.outc(x))
