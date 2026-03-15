import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv2d → (Norm) → Activation
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 norm=nn.InstanceNorm2d, activation=nn.ReLU(inplace=True)):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False))
        if norm is not None:
            layers.append(norm(out_ch))
        if activation is not None:
            layers.append(activation)
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResidualBlock(nn.Module):
    """
    A simple residual block: two conv blocks with skip connection.
    """
    def __init__(self, ch, norm=nn.InstanceNorm2d, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.conv1 = ConvBlock(ch, ch, kernel_size=3, stride=1, padding=1, norm=norm, activation=activation)
        self.conv2 = ConvBlock(ch, ch, kernel_size=3, stride=1, padding=1, norm=norm, activation=None)
        self.act = activation

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + x)


class DownsampleBlock(nn.Module):
    """
    Downsampling by stride-2 conv (or optional choice).
    """
    def __init__(self, in_ch, out_ch, norm=nn.InstanceNorm2d, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, kernel_size=4, stride=2, padding=1, norm=norm, activation=activation)

    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    """
    Upsampling by nearest + conv, or transposed conv.
    """
    def __init__(self, in_ch, out_ch, norm=nn.InstanceNorm2d, activation=nn.ReLU(inplace=True), use_transpose=False):
        super().__init__()
        if use_transpose:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
            self.norm = norm(out_ch) if norm is not None else None
            self.activation = activation
        else:
            # scale up then conv
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
            )
            self.norm = norm(out_ch) if norm is not None else None
            self.activation = activation

    def forward(self, x):
        x = self.up(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
