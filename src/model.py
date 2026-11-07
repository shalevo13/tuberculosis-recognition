"""
TinyTB-Net: A compact, residual convolutional neural network
optimized for detecting Tuberculosis in Chest X-Rays (~307k parameters).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Standard Conv2d -> BatchNorm2d -> ReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # bias=False because BatchNorm subtracts the mean (canceling bias)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) channel attention block.
    Recalibrates channel importance by computing a global channel descriptor
    via adaptive average pooling followed by two 1x1 convolutions (bottleneck)
    and a sigmoid gating activation.
    Designed to have minimal parameter (~5.7k) and zero activation memory overhead
    for ESP32-S3 INT8 deployment.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid_channels = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale


class ResBlock(nn.Module):
    """
    Residual Block with skip connection and optional SE channel attention:
        y = ReLU(x + SE(F(x)))
    If stride > 1, downsamples spatial dimensions and projects channels via a 1x1 conv shortcut.
    """
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

        # Skip shortcut: if shape changes, adapt input via 1x1 conv
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + residual  # The Residual Skip Connection with Channel Attention!
        return self.relu(out)


class TinyTBNet(nn.Module):
    """
    TinyTB-Net architecture for 224x224 Chest X-Ray classification with SE channel attention.
    Total parameters: 313,153 (~305.8 KB INT8 weights).
    Fits within ESP32-S3 internal SRAM budget (~320 KB usable).
    """
    def __init__(self, in_channels=1, num_classes=1, use_se=True):
        super().__init__()
        # 1. Stem: Initial feature extraction (224x224 -> 112x112)
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 16, kernel_size=3, stride=2, padding=1),
        )

        # 2. Residual Stages with Squeeze-and-Excitation
        self.stage1 = ResBlock(16, 16, stride=1, use_se=use_se)   # 112x112, 16 ch
        self.stage2 = ResBlock(16, 32, stride=2, use_se=use_se)   # 56x56,   32 ch
        self.stage3 = ResBlock(32, 64, stride=2, use_se=use_se)   # 28x28,   64 ch
        self.stage4 = ResBlock(64, 128, stride=2, use_se=use_se)  # 14x14,  128 ch

        # 3. Global Average Pooling (14x14 grid -> 1x1 scalar per channel)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # 4. Classification Head (128 -> 1 logit)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: [B, 1, 224, 224]
        x = self.stem(x)        # -> [B, 16, 112, 112]
        x = self.stage1(x)      # -> [B, 16, 112, 112]
        x = self.stage2(x)      # -> [B, 32, 56, 56]
        x = self.stage3(x)      # -> [B, 64, 28, 28]
        x = self.stage4(x)      # -> [B, 128, 14, 14]
        
        x = self.gap(x)         # -> [B, 128, 1, 1]
        x = torch.flatten(x, 1) # -> [B, 128]
        logits = self.classifier(x) # -> [B, 1]
        return logits


def count_parameters(model):
    """Count total and trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TinyTBNet(use_se=True)
    dummy_input = torch.randn(4, 1, 224, 224)
    output = model(dummy_input)
    
    print("=" * 55)
    print("TinyTB-Net (with SE Attention) Architecture Summary")
    print("=" * 55)
    print(f"Input shape:        {dummy_input.shape}")
    print(f"Output shape:       {output.shape} (logits)")
    print(f"Total parameters:   {count_parameters(model):,}")
    print(f"Estimated INT8 size: {count_parameters(model) / 1024:.2f} KB")
    print("=" * 55)
