from src.gan.outer_gan import UpsamplingMode
from typing import Union
import torch.nn as nn

import torchlayers


class ConvTranspose2dBlock(nn.Module):
    '''Convolutional upsampling block. Normalization defaults to BatchNorm but can be customized.'''

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 4,
                 stride: int = 2, padding: int = 1, bias: bool = False,
                 upsampling_factor: int = None,  # must be divisible by 2
                 activation_function=nn.ReLU(True),
                 normalization: Union[bool, nn.Module] = True,
                 upsampling_mode: UpsamplingMode = UpsamplingMode.transposed_conv):

        super(ConvTranspose2dBlock, self).__init__()

        self._set_conv_layer(in_channels, out_channels, kernel_size,
                             stride, padding, bias, upsampling_factor, upsampling_mode)
        self._set_normalization_layer(out_channels, normalization)
        self.activation = activation_function

    def _set_conv_layer(self, in_channels: int, out_channels: int,
                        kernel_size: int = 4,
                        stride: int = 2, padding: int = 1, bias: bool = False,
                        upsampling_factor: int = None,  # must be divisible by 2
                        upsampling_mode: UpsamplingMode = UpsamplingMode.transposed_conv):

        if upsampling_mode == UpsamplingMode.regular_conv and upsampling_factor:
            # From https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190#issuecomment-358546675
            self.conv_layer = nn.Sequential(nn.Upsample(scale_factor=upsampling_factor, mode='nearest'),
                                            nn.ReflectionPad2d(1),
                                            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0))
        elif upsampling_mode == UpsamplingMode.subpixel and upsampling_factor:
            self.conv_layer = torchlayers.upsample.ConvPixelShuffle(
                in_channels, out_channels, kernel_size=kernel_size, upscale_factor=upsampling_factor)
        elif upsampling_mode == UpsamplingMode.transposed_conv:
            if upsampling_factor:
                # This ensures output dimension are scaled up by upsampling_factor
                stride = upsampling_factor
                kernel_size = 2 * upsampling_factor
                padding = upsampling_factor // 2
            self.conv_layer = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def _set_normalization_layer(self, out_channels: int, normalization: Union[bool, nn.Module]):
        if isinstance(normalization, nn.Module):
            # custom normalization layer
            self.normalization = normalization
        elif normalization is True:
            self.normalization = nn.BatchNorm2d(out_channels)
        else:
            self.normalization = None

    def forward(self, x):
        out = self.conv_layer(x)
        if self.normalization:
            out = self.normalization(out)
        return self.activation(out)


class Conv2dBlock(nn.Module):
    '''Convolutional downsampling block. Normalization defaults to BatchNorm but can be customized.'''

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 4,
                 stride: int = 2, padding: int = 1, bias=False,
                 downsampling_factor: int = None,  # must be divisible by 2
                 activation_function=nn.LeakyReLU(
                     0.2, inplace=True),  # from GAN Hacks
                 normalization: Union[bool, nn.Module] = True):

        super(Conv2dBlock, self).__init__()
        if downsampling_factor:
            # This ensures output dimension are scaled down by downsampling_factor
            stride = downsampling_factor
            kernel_size = 2 * downsampling_factor
            padding = downsampling_factor // 2

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if isinstance(normalization, nn.Module):
            # custom normalization layer
            self.normalization = normalization
        elif normalization is True:
            self.normalization = nn.BatchNorm2d(out_channels)
        else:
            self.normalization = None
        self.activation = activation_function

    def forward(self, x):
        out = self.conv_layer(x)
        if self.normalization:
            out = self.normalization(out)
        return self.activation(out)
