""" 
U-Net model code taken from https://github.com/milesial/Pytorch-UNet

Full assembly of the parts to form the complete network
"""

import torch.nn.functional as F
import torch.nn as nn
from .unet_parts import *
from torch.nn.utils.parametrizations import spectral_norm

# Smaller version of U-Net with 2 blocks going up and down. This version is used for the paper's results.
class UNet2(nn.Module):
    # If "reduced_channels" is set, add another 1x1 convolution (basically a pixel-wise nonlinear
    # transform) to the start, to reduce the dimensionality of each pixel.
    # If "min_output" and "max_output" are set, the model's prediction will be constrained by the
    # Tanh function to fall within the range (min_output, max_output).
    def __init__(self, n_channels, n_classes, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, min_output=None, max_output=None):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = nn.Conv2d(n_channels, 64, kernel_size=1, stride=1)
        # self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        self.down2 = Down(128, 128, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        self.up1 = Up(256, 64, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear=True)
        self.up2 = Up(128, 64, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear=True)
        self.outc = OutConv(64, n_classes)

        # Optionally restrict the range of the predictions using Tanh
        if min_output is not None and max_output is not None:
            self.restrict_output = True
            self.mean_output = (min_output + max_output) / 2
            self.scale_factor = (max_output - min_output) / 2
            self.tanh = nn.Tanh()
        else:
            self.restrict_output = False


    def forward(self, x):
        x1 = self.inc(x)
        x1 = F.relu(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)

        # Addition: restrict output
        if self.restrict_output:
            logits = (self.tanh(logits) * self.scale_factor) + self.mean_output
        return logits  #[:, 0:self.n_classes, :, :], logits[:, self.n_classes:, :, :]  # First n_classes entries are predictions. Rest are reconstruction output



class UNet2Contrastive(nn.Module):
    """Smaller U-Net with pixel projection & prediction heads. The projection is normalized to have L2 norm 1.
    """
    def __init__(self, n_channels, n_classes, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, proj_dim=64, proj="linear", min_output=None, max_output=None):
        """Initializes a U-Net contrastive model.
        
        Args:
            n_channels: number of channels in input image
            n_classes: number of output variables
            proj_dim: dimension of pixel projection
            proj: type of projection and prediction head. Can be "mlp" or "linear".
            min_output, max_output: bounds on the output predictions. If these are set, model's predictions
                                    will be constrained by the Tanh function to fall within this range.
                                    Otherwise, there is no constraint on model predictions.
        """
        super(UNet2Contrastive, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = nn.Conv2d(n_channels, 64, kernel_size=1, stride=1)
        self.down1 = Down(64, 128, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        self.down2 = Down(128, 128, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        self.up1 = Up(256, 64, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear=True)
        self.up2 = Up(128, 64, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear=True)
        self.regression_head = PixelRegressionHead(64, output_dim=n_classes, regressor_type=proj, min_output=min_output, max_output=max_output)
        self.projection_head = PixelProjectionHead(64, proj_dim=proj_dim, proj=proj)


    def forward(self, x):
        x1 = self.inc(x)
        x1 = F.relu(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        representations = self.up2(x, x1)
        predictions = self.regression_head(representations)
        projections = self.projection_head(representations)
        return predictions, projections



class UNet2Spectral(nn.Module):
    """Smaller U-Net with pixel projection & prediction heads. The projection is normalized to have L2 norm 1. Layers have spectral normalization.
    """
    def __init__(self, n_channels, n_classes, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, proj_dim=64, proj="linear", min_output=None, max_output=None):
        """Initializes a U-Net contrastive model.
        
        Args:
            n_channels: number of channels in input image
            n_classes: number of output variables
            proj_dim: dimension of pixel projection
            proj: type of projection and prediction head. Can be "mlp" or "linear".
            min_output, max_output: bounds on the output predictions. If these are set, model's predictions
                                    will be constrained by the Tanh function to fall within this range.
                                    Otherwise, there is no constraint on model predictions.
        """
        super(UNet2Spectral, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = spectral_norm(nn.Conv2d(n_channels, 64, kernel_size=1, stride=1))
        self.down1 = spectral_norm(Down(64, 128, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs))
        self.down2 = spectral_norm(Down(128, 128, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs))
        self.up1 = spectral_norm(Up(256, 64, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear=True))
        self.up2 = spectral_norm(Up(128, 64, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear=True))
        self.regression_head = spectral_norm(PixelRegressionHead(64, output_dim=n_classes, regressor_type=proj, min_output=min_output, max_output=max_output))
        self.projection_head = spectral_norm(PixelProjectionHead(64, proj_dim=proj_dim, proj=proj))


    def forward(self, x):
        x1 = self.inc(x)
        x1 = F.relu(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        representations = self.up2(x, x1)
        predictions = self.regression_head(representations)
        projections = self.projection_head(representations)
        return predictions, projections



class UNet2WithReconstruction(nn.Module):
    # If "reduced_channels" is set, add another 1x1 convolution (basically a pixel-wise nonlinear
    # transform) to the start, to reduce the dimensionality of each pixel.
    # If "min_output" and "max_output" are set, the model's prediction will be constrained by the
    # Tanh function to fall within the range (min_output, max_output).
    def __init__(self, n_channels, n_classes, recon_channels, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, min_output=None, max_output=None):
        super(UNet2WithReconstruction, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = nn.Conv2d(n_channels, 64, kernel_size=1, stride=1)
        # self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        self.down2 = Down(128, 128, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        self.up1 = Up(256, 64, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear=True)
        self.up2 = Up(128, 64, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear=True)
        self.outc = OutConv(64, n_classes)

        self.recon1 = nn.Conv2d(n_classes, recon_channels // 2, kernel_size=1, stride=1)
        self.recon2 = nn.Conv2d(recon_channels // 2, recon_channels, kernel_size=1, stride=1)

        # Optionally restrict the range of the predictions using Tanh
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
        x1 = F.relu(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)

        # Addition: restrict output 
        if self.restrict_output:
            logits = (self.tanh(logits) * self.scale_factor) + self.mean_output
        
        recon = self.recon2(F.relu(self.recon1(logits)))
        return logits, recon


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs,
                 min_output=None, max_output=None, bilinear=True, crop_type_start_idx=12, crop_type_embedding_dim=10):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # # Create list of crop type indices and non-crop indices
        # self.non_crop_type_bands = list(range(0, crop_type_start_idx)) + [n_channels - 1]  # Last channel is the "missing reflectance" mask and is not a crop type
        # self.crop_type_bands = list(range(crop_type_start_idx, n_channels - 1))

        # Embedding for each crop type's pixels
        # self.crop_type_embedding = nn.Conv2d(len(self.crop_type_bands), crop_type_embedding_dim, kernel_size=1, stride=1)

        # # Number of channels after embedding crop type
        # channels_after_embedding = n_channels - len(self.crop_type_bands) + crop_type_embedding_dim  # Number of features after embedding crop type
        # self.dimensionality_reduction = nn.Conv2d(channels_after_embedding, reduced_channels, kernel_size=1, stride=1)

        # self.inc = DoubleConv(reduced_channels, 64)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        self.down2 = Down(128, 256, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        self.down3 = Down(256, 512, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        self.up1 = Up(1024, 512 // factor, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear)
        self.up2 = Up(512, 256 // factor, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear)
        self.up3 = Up(256, 128 // factor, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear)
        self.up4 = Up(128, 64, bilinear, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
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

        return logits #logits[:, 0:self.n_classes, :, :], logits[:, self.n_classes:, :, :]  # First n_classes entries are predictions. Rest are reconstruction output


class UNetContrastive(nn.Module):
    """Large U-Net with pixel projection & prediction heads. The projection is normalized to have L2 norm 1.
    """
    def __init__(self, n_channels, n_classes, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, proj_dim=64, proj="linear", min_output=None, max_output=None, bilinear=True):
        """Initializes a U-Net contrastive model.
        
        Args:
            n_channels: number of channels in input image
            n_classes: number of output variables
            proj_dim: dimension of pixel projection
            proj: type of projection and prediction head. Can be "mlp" or "linear".
            min_output, max_output: bounds on the output predictions. If these are set, model's predictions
                                    will be constrained by the Tanh function to fall within this range.
                                    Otherwise, there is no constraint on model predictions.
        """
        super(UNetContrastive, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        self.down1 = Down(64, 128, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        self.down2 = Down(128, 256, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        self.down3 = Down(256, 512, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        self.up1 = Up(1024, 512 // factor, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear)
        self.up2 = Up(512, 256 // factor, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear)
        self.up3 = Up(256, 128 // factor, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear)
        self.up4 = Up(128, 64, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear)
        self.regression_head = PixelRegressionHead(64, output_dim=n_classes, regressor_type=proj, min_output=min_output, max_output=max_output)
        self.projection_head = PixelProjectionHead(64, proj_dim=proj_dim, proj=proj)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        representations = self.up4(x, x1)
        predictions = self.regression_head(representations)
        projections = self.projection_head(representations)
        return predictions, projections


class UNetSmall(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, min_output=None, max_output=None, bilinear=True):
        super(UNetSmall, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Try reducing dimensionality of the channels
        # self.dimensionality_reduction = nn.Conv2d(n_channels, reduced_channels, kernel_size=1, stride=1)
        self.inc = DoubleConv(n_channels, 64, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        self.down1 = Down(64, 128, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        self.down2 = Down(128, 256, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        self.down3 = Down(256, 256, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        self.up1 = Up(512, 128, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear)
        self.up2 = Up(256, 64, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear)
        self.up3 = Up(128, 64, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear)
        self.outc = OutConv(64, n_classes)

        if min_output is not None and max_output is not None:
            self.restrict_output = True
            self.mean_output = (min_output + max_output) / 2
            self.scale_factor = (max_output - min_output) / 2
            self.tanh = nn.Tanh()
        else:
            self.restrict_output = False


    def forward(self, x):
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




class UNet2Larger(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, min_output=None, max_output=None):
        super(UNet2Larger, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # self.dimensionality_reduction = nn.Conv2d(n_channels, reduced_channels, kernel_size=1, stride=1)
        # self.inc = nn.Conv2d(n_channels, 128, kernel_size=1, stride=1)  # TODO CHange back
        self.inc = DoubleConv(n_channels, 128, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)  # TODO CHange back
        self.down1 = Down(128, 256, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        self.down2 = Down(256, 256, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        self.up1 = Up(512, 128, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear=True)
        self.up2 = Up(256, 128, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear=True)
        self.outc = OutConv(128, n_classes)

        if min_output is not None and max_output is not None:
            self.restrict_output = True
            self.mean_output = (min_output + max_output) / 2
            self.scale_factor = (max_output - min_output) / 2
            self.tanh = nn.Tanh()
        else:
            self.restrict_output = False


    def forward(self, x):
        x1 = self.inc(x)
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
    def __init__(self, n_channels, n_classes, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, min_output=None, max_output=None, coarse_feature_indices=[7,8,9]):
        super(UNet2, self).__init__()
        n_channels = n_channels - len(coarse_feature_indices)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.coarse_feature_indices = coarse_feature_indices

        self.inc = nn.Conv2d(n_channels, 64, kernel_size=1, stride=1)
        self.down1 = Down(64, 128, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        self.down2 = Down(128, 128, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        self.up1 = Up(256, 64, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear=True)
        self.up2 = Up(128, 64, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear=True)
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
        non_coarse_indices = [i for i in range(self.n_channels) if i not in coarse_features]
        print('Noncoarse indices', non_coarse_indices)
        x = x[non_coarse_indices, :, :]
        x1 = self.inc(x)
        x1 = F.relu(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = torch.concatenate([x, coarse_features])
        logits = self.outc(x)

        # Addition: restrict output 
        if self.restrict_output:
            logits = (self.tanh(logits) * self.scale_factor) + self.mean_output
        return logits


# class UNet2PixelEmbedding(nn.Module):
#     def __init__(self, n_channels, n_classes, reduced_channels, min_output=None, max_output=None):
#         super(UNet2PixelEmbedding, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes

#         # Try reducing dimensionality of the channels
#         self.dimensionality_reduction_1 = nn.Conv2d(n_channels, reduced_channels, kernel_size=1, stride=1)
#         # self.dimensionality_reduction_2 = nn.Conv2d(n_channels, reduced_channels, kernel_size=1, stride=1)
#         # self.relu = nn.ReLU(inplace=True)
#         # self.inc = DoubleConv(reduced_channels, 32)
#         # self.down1 = Down(32, 64)
#         # self.down2 = Down(64, 64)
#         # self.up1 = Up(128, 32, bilinear=True)
#         # self.up2 = Up(64, 32, bilinear=True)
#         # self.outc = OutConv(32, n_classes)

#         self.inc = DoubleConv(reduced_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 128)
#         self.up1 = Up(256, 64, bilinear=True)
#         self.up2 = Up(128, 64, bilinear=True)
#         self.outc = OutConv(64, n_classes)

#         if min_output is not None and max_output is not None:
#             self.restrict_output = True
#             self.mean_output = (min_output + max_output) / 2
#             self.scale_factor = (max_output - min_output) / 2
#             self.tanh = nn.Tanh()
#         else:
#             self.restrict_output = False


#     def forward(self, x):
#         x = self.dimensionality_reduction_1(x)
#         x = F.relu(x)
#         # x = self.dimensionality_reduction_2(x)
#         # x = F.relu(x)
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x = self.up1(x3, x2)
#         x = self.up2(x, x1)
#         logits = self.outc(x)

#         # Addition: restrict output
#         if self.restrict_output:
#             logits = (self.tanh(logits) * self.scale_factor) + self.mean_output
#         return logits, x1


class PixelNN(nn.Module):
    def __init__(self, input_channels, output_dim, min_output=None, max_output=None):
        super(PixelNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 100, kernel_size=1, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(100, 100, kernel_size=1, stride=1, padding=0)
        # self.conv3 = nn.Conv2d(100, 100, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(100, output_dim, kernel_size=1, stride=1, padding=0)

        if min_output is not None and max_output is not None:
            self.restrict_output = True
            self.mean_output = (min_output + max_output) / 2
            self.scale_factor = (max_output - min_output) / 2
        else:
            self.restrict_output = False

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        if self.restrict_output:
            x = (F.tanh(x) * self.scale_factor) + self.mean_output
        return x, None


