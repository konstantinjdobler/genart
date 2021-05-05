from typing import Union
import torch.nn as nn
import torch

from math import log
import torchlayers


class ConvTranspose2dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 4,
                 stride: int = 2, padding: int = 1, bias=False,
                 upsampling_factor: int = None,  # must be divisible by 2
                 activation_function=nn.ReLU(True),
                 batch_norm: bool = Union[bool, nn.Module], smoothing=False):

        super(ConvTranspose2dBlock, self).__init__()
        if upsampling_factor:
            # This ensures output dimension are scaled up by upsampling_factor
            stride = upsampling_factor
            kernel_size = 2 * upsampling_factor
            padding = upsampling_factor // 2
        if smoothing == "regular-upsample" and upsampling_factor:
            # From https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190#issuecomment-358546675
            self.conv_layer = nn.Sequential(nn.Upsample(scale_factor=upsampling_factor, mode='nearest'),
                                            nn.ReflectionPad2d(1),
                                            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0))
        elif smoothing == "subpixel" and upsampling_factor:
            self.conv_layer = torchlayers.upsample.ConvPixelShuffle(
                in_channels, out_channels, upscale_factor=upsampling_factor)
        else:
            self.conv_layer = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if isinstance(batch_norm, nn.Module):
            # custom normalization layer
            self.batch_norm = batch_norm
        elif batch_norm is True:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        else:
            self.batch_norm = None
        self.activation = activation_function

    def forward(self, x):
        out = self.conv_layer(x)
        if self.batch_norm:
            out = self.batch_norm(out)
        return self.activation(out)


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 4,
                 stride: int = 2, padding: int = 1, bias=False,
                 downsampling_factor: int = None,  # must be divisible by 2
                 activation_function=nn.LeakyReLU(
                     0.2, inplace=True),  # from GAN Hacks
                 batch_norm: Union[bool, nn.Module] = True):

        super(Conv2dBlock, self).__init__()
        if downsampling_factor:
            # This ensures output dimension are scaled down by downsampling_factor
            stride = downsampling_factor
            kernel_size = 2 * downsampling_factor
            padding = downsampling_factor // 2

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if isinstance(batch_norm, nn.Module):
            # custom normalization layer
            self.batch_norm = batch_norm
        elif batch_norm is True:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        else:
            self.batch_norm = None
        self.activation = activation_function

    def forward(self, x):
        out = self.conv_layer(x)
        if self.batch_norm:
            out = self.batch_norm(out)
        return self.activation(out)


class cDCGenerator(nn.Module):

    def __init__(self, latent_dim: int, num_features: int, img_shape, n_filters=64, smoothing=False):
        super(cDCGenerator, self).__init__()
        self.num_features = num_features
        # end layer has upsampling=2, first layer outputs 4x4
        num_middle_scaling_layers = int(log(img_shape[-1], 2) - 3)

        # as many scaling layers as necessary to scale to the target image size
        middle_scaling_layers = [ConvTranspose2dBlock(in_channels=n_filters * 2**(i+1),
                                                      out_channels=n_filters * 2**i,
                                                      upsampling_factor=2, smoothing=smoothing) for i in reversed(range(num_middle_scaling_layers))]
        self.main = nn.Sequential(
            ConvTranspose2dBlock(in_channels=latent_dim + num_features,
                                 out_channels=n_filters *
                                 (2**num_middle_scaling_layers),
                                 kernel_size=4, stride=1, padding=0, smoothing=smoothing),
            *middle_scaling_layers,
            ConvTranspose2dBlock(in_channels=n_filters,
                                 out_channels=3, upsampling_factor=2,
                                 activation_function=nn.Tanh(), batch_norm=False, smoothing=smoothing),
        )

    def forward(self, x, attr):
        attr = attr.view(-1, self.num_features, 1, 1)
        x = torch.cat([x, attr], 1)
        return self.main(x)


class cDCGeneratorRegularUpsample(cDCGenerator):
    def __init__(self, latent_dim: int, num_features: int, img_shape, n_filters=64, smoothed="regular-upsample"):
        super(cDCGeneratorRegularUpsample, self).__init__(latent_dim=latent_dim,
                                                          num_features=num_features, img_shape=img_shape, n_filters=n_filters, smoothing="regular-upsample")


class cDCGeneratorSubpixel(cDCGenerator):
    def __init__(self, latent_dim: int, num_features: int, img_shape, n_filters=64, smoothed="subpixel"):
        super(cDCGeneratorSubpixel, self).__init__(latent_dim=latent_dim,
                                                   num_features=num_features, img_shape=img_shape, n_filters=n_filters, smoothing="subpixel")


class cDCDiscriminator(nn.Module):
    def __init__(self, num_features: int, img_shape, n_filters=64):
        super(cDCDiscriminator, self).__init__()
        self.input_image_size = img_shape[-1]
        self.feature_input = nn.Linear(num_features,
                                       self.input_image_size * self.input_image_size)

        # end layer has upsampling=2, first layer outputs 4x4
        num_middle_scaling_layers = int(log(self.input_image_size, 2) - 3)
        # as many scaling layers as necessary to scale to the target image size
        middle_scaling_layers = [Conv2dBlock(in_channels=n_filters * 2**i,
                                             out_channels=n_filters *
                                             2**(i + 1),
                                             downsampling_factor=2) for i in range(num_middle_scaling_layers)]
        self.main = nn.Sequential(
            Conv2dBlock(in_channels=3 + 1, out_channels=n_filters,
                        downsampling_factor=2, batch_norm=False),
            *middle_scaling_layers,
            Conv2dBlock(in_channels=n_filters * 2**num_middle_scaling_layers,
                        out_channels=1, kernel_size=4, stride=1, padding=0,
                        batch_norm=False, activation_function=nn.Identity()),
        )

    def forward(self, x, attr):
        attr = self.feature_input(
            attr).view(-1, 1, self.input_image_size, self.input_image_size)
        x = torch.cat([x, attr], 1)
        return self.main(x).view(-1, 1)
