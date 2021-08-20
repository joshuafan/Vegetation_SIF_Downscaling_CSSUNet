""" 
U-Net model code taken from https://github.com/milesial/Pytorch-UNet

Full assembly of the parts to form the complete network
"""

import torch.nn.functional as F
import torch.nn as nn
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, reduced_channels, min_output=None, max_output=None, bilinear=True, crop_type_start_idx=12, crop_type_embedding_dim=10):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Create list of crop type indices and non-crop indices
        self.non_crop_type_bands = list(range(0, crop_type_start_idx)) + [n_channels - 1]  # Last channel is the "missing reflectance" mask and is not a crop type
        self.crop_type_bands = list(range(crop_type_start_idx, n_channels - 1))

        # Embedding for each crop type's pixels
        # self.crop_type_embedding = nn.Conv2d(len(self.crop_type_bands), crop_type_embedding_dim, kernel_size=1, stride=1)

        # # Number of channels after embedding crop type
        # channels_after_embedding = n_channels - len(self.crop_type_bands) + crop_type_embedding_dim  # Number of features after embedding crop type
        # self.dimensionality_reduction = nn.Conv2d(channels_after_embedding, reduced_channels, kernel_size=1, stride=1)

        if reduced_channels is not None:
            self.dimensionality_reduction = nn.Conv2d(n_channels, reduced_channels, kernel_size=1, stride=1)
            self.inc = DoubleConv(reduced_channels, 64)
        else:
            self.dimensionality_reduction = None
            self.inc = DoubleConv(n_channels, 64)

        # # Try reducing dimensionality of the channels
        # self.dimensionality_reduction = nn.Conv2d(n_channels, reduced_channels, kernel_size=1, stride=1)
        # self.inc = DoubleConv(reduced_channels, 64)
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
            self.tanh = nn.Tanh()
        else:
            self.restrict_output = False


    def forward(self, x):
        # crop_masks = x[:, self.crop_type_bands, :, :]
        # crop_embeddings = self.crop_type_embedding(crop_masks)
        # # print('crop embeddings', crop_embeddings.shape)

        # # Concatenate crop type embedding with other pixel features
        # x = torch.cat([x[:, self.non_crop_type_bands, :, :], crop_embeddings], dim=1)
        # # print('Combined crop type and other features', x.shape)

        # Embed each pixel. Each pixel's vector should contain semantic information about
        # the crop type + reflectance + other features
        if self.dimensionality_reduction is not None:
            x = self.dimensionality_reduction(x)
            x = F.relu(x)
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
            logits = (self.tanh(logits) * self.scale_factor) + self.mean_output
        return logits, x1


class UNetSmall(nn.Module):
    def __init__(self, n_channels, n_classes, reduced_channels, min_output=None, max_output=None, bilinear=True):
        super(UNetSmall, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Try reducing dimensionality of the channels
        # self.dimensionality_reduction = nn.Conv2d(n_channels, reduced_channels, kernel_size=1, stride=1)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        if min_output is not None and max_output is not None:
            self.restrict_output = True
            self.mean_output = (min_output + max_output) / 2
            self.scale_factor = (max_output - min_output) / 2
            self.tanh = nn.Tanh()
        else:
            self.restrict_output = False


    def forward(self, x):
        # x = self.dimensionality_reduction(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)

        # Addition: restrict output 
        if self.restrict_output:
            logits = (self.tanh(logits) * self.scale_factor) + self.mean_output
        return logits


class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, reduced_channels, min_output=None, max_output=None):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        if reduced_channels is not None:
            self.dimensionality_reduction = nn.Conv2d(n_channels, reduced_channels, kernel_size=1, stride=1)
            self.inc = nn.Conv2d(reduced_channels, 64, kernel_size=1, stride=1)
        else:
            self.dimensionality_reduction = None
            self.inc = nn.Conv2d(n_channels, 64, kernel_size=1, stride=1)
        # self.dropout = nn.Dropout2d()

        # self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 128)
        self.up1 = Up(256, 64, bilinear=True)
        self.up2 = Up(128, 64, bilinear=True)
        self.outc = OutConv(64, n_classes)

        if min_output is not None and max_output is not None:
            self.restrict_output = True
            self.mean_output = (min_output + max_output) / 2
            self.scale_factor = (max_output - min_output) / 2
            self.tanh = nn.Tanh()
        else:
            self.restrict_output = False


    def forward(self, x):
        if self.dimensionality_reduction is not None:
            x = self.dimensionality_reduction(x)
            x = F.relu(x)
        x1 = self.inc(x)
        # x1 = self.dropout(x1)
        x1 = F.relu(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)

        # Addition: restrict output 
        if self.restrict_output:
            logits = (self.tanh(logits) * self.scale_factor) + self.mean_output
        return logits


class UNet2Larger(nn.Module):
    def __init__(self, n_channels, n_classes, reduced_channels, min_output=None, max_output=None):
        super(UNet2Larger, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # self.dimensionality_reduction = nn.Conv2d(n_channels, reduced_channels, kernel_size=1, stride=1)
        self.inc = nn.Conv2d(n_channels, 128, kernel_size=1, stride=1)  # TODO CHange back
        # self.inc = DoubleConv(n_channels, 128)  # TODO CHange back
        self.down1 = Down(128, 256)
        self.down2 = Down(256, 256)
        self.up1 = Up(512, 128, bilinear=True)
        self.up2 = Up(256, 128, bilinear=True)
        self.outc = OutConv(128, n_classes)

        if min_output is not None and max_output is not None:
            self.restrict_output = True
            self.mean_output = (min_output + max_output) / 2
            self.scale_factor = (max_output - min_output) / 2
            self.tanh = nn.Tanh()
        else:
            self.restrict_output = False


    def forward(self, x):
        # x = self.dimensionality_reduction(x)
        # x = F.relu(x)
        x1 = self.inc(x)
        x1 = F.leaky_relu(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)

        # Addition: restrict output 
        if self.restrict_output:
            logits = (self.tanh(logits) * self.scale_factor) + self.mean_output
        return logits


class UNet2CoarseFeatures(nn.Module):
    def __init__(self, n_channels, n_classes, min_output=None, max_output=None, coarse_feature_indices=[9,10,11]):
        super(UNet2, self).__init__()
        n_channels = n_channels - len(coarse_feature_indices)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.coarse_feature_indices = coarse_feature_indices

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 128)
        self.up1 = Up(256, 64, bilinear=True)
        self.up2 = Up(128, 64, bilinear=True)
        self.outc = OutConv(64+len(coarse_feature_indices), n_classes)

        if min_output is not None and max_output is not None:
            self.restrict_output = True
            self.mean_output = (min_output + max_output) / 2
            self.scale_factor = (max_output - min_output) / 2
            self.tanh = nn.Tanh()
        else:
            self.restrict_output = False


    def forward(self, x):
        coarse_features = x[self.coarse_feature_indices, :, :]
        non_coarse_indices = [i for i in range(n_channels) if i not in coarse_features]
        print('Noncoarse indices', non_coarse_indices)
        x = x[non_coarse_indices, :, :]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = torch.concatenate([x, ])
        logits = self.outc(x)

        # Addition: restrict output 
        if self.restrict_output:
            logits = (self.tanh(logits) * self.scale_factor) + self.mean_output
        return logits


class UNet2PixelEmbedding(nn.Module):
    def __init__(self, n_channels, n_classes, reduced_channels, min_output=None, max_output=None):
        super(UNet2PixelEmbedding, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Try reducing dimensionality of the channels
        self.dimensionality_reduction_1 = nn.Conv2d(n_channels, reduced_channels, kernel_size=1, stride=1)
        # self.dimensionality_reduction_2 = nn.Conv2d(n_channels, reduced_channels, kernel_size=1, stride=1)
        # self.relu = nn.ReLU(inplace=True)
        # self.inc = DoubleConv(reduced_channels, 32)
        # self.down1 = Down(32, 64)
        # self.down2 = Down(64, 64)
        # self.up1 = Up(128, 32, bilinear=True)
        # self.up2 = Up(64, 32, bilinear=True)
        # self.outc = OutConv(32, n_classes)

        self.inc = DoubleConv(reduced_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 128)
        self.up1 = Up(256, 64, bilinear=True)
        self.up2 = Up(128, 64, bilinear=True)
        self.outc = OutConv(64, n_classes)

        if min_output is not None and max_output is not None:
            self.restrict_output = True
            self.mean_output = (min_output + max_output) / 2
            self.scale_factor = (max_output - min_output) / 2
            self.tanh = nn.Tanh()
        else:
            self.restrict_output = False


    def forward(self, x):
        x = self.dimensionality_reduction_1(x)
        x = F.relu(x)
        # x = self.dimensionality_reduction_2(x)
        # x = F.relu(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)

        # Addition: restrict output 
        if self.restrict_output:
            logits = (self.tanh(logits) * self.scale_factor) + self.mean_output
        return logits, x1
