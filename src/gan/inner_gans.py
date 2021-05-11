from src.gan.outer_gan import ConditionMode, Normalization, UpsamplingMode
from src.gan.blocks import Conv2dBlock, ConvTranspose2dBlock
from typing import Tuple
import torch.nn as nn
import torch

from math import log


class DCGenerator(nn.Module):

    def __init__(self, latent_dim: int, num_features: int,
                 img_shape: Tuple[int], n_filters: int = 64,
                 upsampling_mode: UpsamplingMode = UpsamplingMode.transposed_conv,
                 condition_mode: ConditionMode = ConditionMode.unconditional):
        super(DCGenerator, self).__init__()
        # end layer has upsampling=2, first layer outputs 4x4
        num_middle_scaling_layers = int(log(img_shape[-1], 2) - 3)

        # as many scaling layers as necessary to scale to the target image size
        middle_scaling_layers = [ConvTranspose2dBlock(in_channels=n_filters * 2**(i+1),
                                                      out_channels=n_filters * 2**i,
                                                      upsampling_factor=2, upsampling_mode=upsampling_mode) for i in reversed(range(num_middle_scaling_layers))]

        initial_in_channels = (
            latent_dim + num_features) if condition_mode == ConditionMode.simple_conditioning else latent_dim
        self.main = nn.Sequential(
            ConvTranspose2dBlock(in_channels=initial_in_channels,
                                 out_channels=n_filters *
                                 (2**num_middle_scaling_layers),
                                 kernel_size=4, stride=1, padding=0),
            *middle_scaling_layers,
            ConvTranspose2dBlock(in_channels=n_filters,
                                 out_channels=3, upsampling_factor=2,
                                 activation_function=nn.Tanh(), normalization=False, upsampling_mode=upsampling_mode),
        )

        # Do condition_mode specific setup
        if condition_mode == ConditionMode.simple_conditioning:
            self.num_features = num_features
        # Set appropriate forward hook
        self.forward = getattr(self, f"_{condition_mode.value}_forward")

    def forward(self, x, attr):
        raise NotImplementedError(
            "This should have been replaced with the appropriate forward method in the __init__ call.")

    def _simple_conditioning_forward(self, x, attr):
        attr = attr.view(-1, self.num_features, 1, 1)
        x = torch.cat([x, attr], 1)
        return self.main(x)

    def _unconditional_forward(self, x, attr):
        return self.main(x)


class DCDiscriminator(nn.Module):
    def __init__(self, num_features: int, img_shape: Tuple[int], n_filters=64,
                 condition_mode: ConditionMode = ConditionMode.unconditional,
                 normalization: Normalization = Normalization.batch):
        super(DCDiscriminator, self).__init__()
        self.input_image_size = img_shape[-1]
        # end layer has upsampling=2, first layer outputs 4x4
        num_middle_scaling_layers = int(log(self.input_image_size, 2) - 3)

        middle_scaling_layers = []
        for i in range(num_middle_scaling_layers):
            if normalization is Normalization.batch:
                use_norm = True
                bias = False
            elif normalization is Normalization.no_norm:
                use_norm = False
                bias = True
            elif normalization is Normalization.instance:
                use_norm = nn.InstanceNorm2d(
                    n_filters * 2**(i + 1))
                bias = True
            elif normalization is Normalization.layer:
                # nn.LayerNorm is a bit weird https://github.com/pytorch/pytorch/issues/51455
                # use GroupNorm as in https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch/blob/master/module.py to simulate LayerNorm
                use_norm = nn.GroupNorm(
                    num_groups=1, num_channels=n_filters * 2**(i + 1))
                bias = False

            middle_scaling_layers.append(Conv2dBlock(in_channels=n_filters * 2**i,
                                                     out_channels=n_filters *
                                                     2**(i + 1),
                                                     bias=bias,
                                                     normalization=use_norm,
                                                     downsampling_factor=2))

        image_in_channels = 4 if condition_mode == ConditionMode.simple_conditioning else 3
        self.main = nn.Sequential(
            Conv2dBlock(in_channels=image_in_channels, out_channels=n_filters,
                        downsampling_factor=2, normalization=False),
            *middle_scaling_layers,
            Conv2dBlock(in_channels=n_filters * 2**num_middle_scaling_layers,
                        out_channels=1, kernel_size=4, stride=1, padding=0,
                        normalization=False, activation_function=nn.Identity()),
        )

        self._setup_condition_mode(condition_mode, num_features)

    def _setup_condition_mode(self, condition_mode, num_features):
        if condition_mode == ConditionMode.simple_conditioning:
            self.feature_input = nn.Linear(num_features,
                                           self.input_image_size * self.input_image_size)
         # Set appropriate forward hook
        self.forward = getattr(self, f"_{condition_mode.value}_forward")

    def forward(self, x, attr):
        raise NotImplementedError(
            "This should have been replaced with the appropriate forward method in the __init__ call.")

    def _simple_conditioning_forward(self, x, attr):
        attr = self.feature_input(
            attr).view(-1, 1, self.input_image_size, self.input_image_size)
        x = torch.cat([x, attr], 1)
        return self.main(x).view(-1, 1)

    def _unconditional_forward(self, x, attr):
        '''attr is not used'''

        return self.main(x).view(-1, 1)
