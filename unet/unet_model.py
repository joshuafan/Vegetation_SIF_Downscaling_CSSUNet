""" 
U-Net model code taken from https://github.com/milesial/Pytorch-UNet

Full assembly of the parts to form the complete network
"""

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, reduced_channels, min_output=None, max_output=None, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Try reducing dimensionality of the channels
        self.dimensionality_reduction = nn.Conv2d(n_channels, reduced_channels, kernel_size=1, stride=1)
        self.inc = DoubleConv(reduced_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        if min_output is not None and max_output is not None:
            self.restrict_output = True
            self.mean_output = (min_output + max_output) / 2
            self.scale_factor = (max_output - min_output) / 2
        else:
            self.restrict_output = False


    def forward(self, x):
        x = self.dimensionality_reduction(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        # Addition: restrict output 
        if self.restrict_output:
            logits = (F.tanh(logits) * self.scale_factor) + self.mean_output
        return logits