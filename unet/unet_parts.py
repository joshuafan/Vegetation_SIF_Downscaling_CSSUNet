""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    # TODO - add kernel size param
    def __init__(self, in_channels, out_channels, dropout_op=None, dropout_op_kwargs=None, norm_op=None, norm_op_kwargs=None, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if norm_op is None and dropout_op is None:  # For backward compatibility with existing models. Temporary.
            print("No dropout or norm")
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),

                # ATTENTION!!! Kernel size changed to 1
                nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0),
                nn.ReLU(inplace=True)
            )
        else:
            print("Dropout or norm being used")
            if norm_op is None:
                norm_op = nn.Identity()
            if dropout_op is None:
                dropout_op = nn.Identity()
            if dropout_op_kwargs is None:
                dropout_op_kwargs = {'p': 0.1, 'inplace': True}
            if norm_op_kwargs is None:
                norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                dropout_op(**dropout_op_kwargs),
                norm_op(mid_channels, **norm_op_kwargs),
                nn.ReLU(inplace=True),

                # ATTENTION!!! Kernel size changed to 1
                nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0),
                dropout_op(**dropout_op_kwargs),
                norm_op(out_channels, **norm_op_kwargs),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_op, dropout_op_kwargs, norm_op, norm_op_kwargs)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class PixelProjectionHead(nn.Module):
    """Per-pixel projection head
    """
    def __init__(self, dim_in, proj_dim=64, proj='linear'):
        super(PixelProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'mlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


class PixelRegressionHead(nn.Module):
    """Per-pixel regression head
    """
    def __init__(self, dim_in, output_dim=1, regressor_type='linear', min_output=None, max_output=None):
        super(PixelRegressionHead, self).__init__()

        # Define pixel regressor
        if regressor_type == 'linear':
            self.regressor = nn.Conv2d(dim_in, output_dim, kernel_size=1)
        elif regressor_type == 'mlp':
            self.regressor = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_in, output_dim, kernel_size=1)
            )

        # Optionally restrict the range of the predictions using Tanh
        if min_output is not None and max_output is not None:
            self.restrict_output = True
            self.mean_output = (min_output + max_output) / 2
            self.scale_factor = (max_output - min_output) / 2
            self.tanh = nn.Tanh()
        else:
            self.restrict_output = False

    def forward(self, x):
        logits = self.regressor(x)
        if self.restrict_output:
            logits = (self.tanh(logits) * self.scale_factor) + self.mean_output
        return logits
