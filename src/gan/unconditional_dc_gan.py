import torch.nn as nn
import torch

from math import log

from gan.conditional_dc_gan import Conv2dBlock, ConvTranspose2dBlock



class DCGenerator(nn.Module):

    def __init__(self, latent_dim: int, num_features: int, img_shape, n_filters=64, smoothed=False):
        super(DCGenerator, self).__init__()
        # end layer has upsampling=2, first layer outputs 4x4
        num_middle_scaling_layers = int(log(img_shape[-1], 2) - 3)

        # as many scaling layers as necessary to scale to the target image size
        middle_scaling_layers = [ConvTranspose2dBlock(in_channels=n_filters * 2**(i+1),
                                                      out_channels=n_filters * 2**i,
                                                      upsampling_factor=2, smoothed=smoothed) for i in reversed(range(num_middle_scaling_layers))]
        self.main = nn.Sequential(
            ConvTranspose2dBlock(in_channels=latent_dim,
                                 out_channels=n_filters *
                                 (2**num_middle_scaling_layers),
                                 kernel_size=4, stride=1, padding=0, smoothed=smoothed),
            *middle_scaling_layers,
            ConvTranspose2dBlock(in_channels=n_filters,
                                 out_channels=3, upsampling_factor=2,
                                 activation_function=nn.Tanh(), batch_norm=False, smoothed=smoothed),
        )

    def forward(self, x, attr):
        return self.main(x)


class DCGeneratorSmoothed(DCGenerator):
    def __init__(self, latent_dim: int, num_features: int, img_shape, n_filters=64, smoothed=True):
        super(DCGeneratorSmoothed, self).__init__(latent_dim=latent_dim,
                                                   num_features=num_features, img_shape=img_shape, n_filters=n_filters, smoothed=True)


class DCDiscriminator(nn.Module):
    def __init__(self, num_features: int, img_shape, n_filters=64):
        super(DCDiscriminator, self).__init__()
        self.input_image_size = img_shape[-1]

        # end layer has upsampling=2, first layer outputs 4x4
        num_middle_scaling_layers = int(log(self.input_image_size, 2) - 3)
        # as many scaling layers as necessary to scale to the target image size
        middle_scaling_layers = [Conv2dBlock(in_channels=n_filters * 2**i,
                                             out_channels=n_filters *
                                             2**(i + 1),
                                             downsampling_factor=2) for i in range(num_middle_scaling_layers)]
        self.main = nn.Sequential(
            Conv2dBlock(in_channels=3, out_channels=n_filters,
                        downsampling_factor=2, batch_norm=False),
            *middle_scaling_layers,
            Conv2dBlock(in_channels=n_filters * 2**num_middle_scaling_layers,
                        out_channels=1, kernel_size=4, stride=1, padding=0,
                        batch_norm=False, activation_function=nn.Sigmoid()),
        )

    def forward(self, x, attr):
        '''attr is not used'''

        return self.main(x).view(-1, 1)
