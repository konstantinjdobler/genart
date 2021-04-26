import wandb
from common.helpers import push_file_to_wandb, randomly_flip_labels
from gan.conditional_dc_gan import cDCGenerator, cDCDiscriminator, cDCGeneratorSmoothed
import torch.nn.functional as F
import torchvision
from torch import nn
import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

from gan.unconditional_dc_gan import DCDiscriminator, DCGenerator, DCGeneratorSmoothed


generator_dict = {
    'cdc-smoothed': cDCGeneratorSmoothed,
    'cdc': cDCGenerator,
    'dc-smoothed': DCGeneratorSmoothed,
    'dc': DCGenerator
}

discriminator_dict = {
    'cdc': cDCDiscriminator,
    'dc': DCDiscriminator
}


class conditionalGAN(pl.LightningModule):

    def __init__(
        self,
        channels,
        width,
        height,
        latent_dim,
        num_features,
        lr,
        batch_size,
        label_flipping_p,
        label_smoothing,
        b1=0.5,
        b2=0.99,
        condition=True,
        generator_type=list(generator_dict.keys())[0],
        discriminator_type=list(discriminator_dict.keys())[0],
        ** kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        data_shape = (channels, width, height)
        self.generator = self._get_generator(
            data_shape, GeneratorClass=generator_dict[generator_type])
        self.discriminator = self._get_discriminator(
            data_shape, DiscriminatorClass=discriminator_dict[discriminator_type])

        self.validation_z = torch.randn(8, self.hparams.latent_dim, 1, 1)

        self.example_input_array = torch.zeros(
            8, self.hparams.latent_dim, 1, 1)
        self.example_feature_array = torch.zeros(
            8, self.hparams.num_features)
        # shift to [-1,1] value range
        self.example_feature_array[self.example_feature_array == 0] = -1

    def set_argparse_config(self, config):
        '''Call before training start'''
        self.argparse_config = config
        return self

    def _get_generator(self, data_shape, GeneratorClass) -> nn.Module:
        print("Using generator", GeneratorClass.__name__)
        generator = GeneratorClass(latent_dim=self.hparams.latent_dim,
                                   num_features=self.hparams.num_features, img_shape=data_shape)
        generator.apply(self._weights_init)
        return generator

    def _get_discriminator(self, data_shape, DiscriminatorClass) -> nn.Module:
        print("Using discriminator", DiscriminatorClass.__name__)

        discriminator = DiscriminatorClass(
            num_features=self.hparams.num_features, img_shape=data_shape)
        discriminator.apply(self._weights_init)
        return discriminator

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname == 'Conv2d' or classname == 'ConvTranspose2d':
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

    def forward(self, z, features=None):
        '''Do a whole pass through the GAN but return only the genrated images. Don't use if performance is key'''
        if features is None:
            features = self.example_feature_array.type_as(z)
        generated_images = self.generator(z, features)
        # Do this step to also show the discriminator in PyTorch-Lightning's automatic model summary
        discriminator_decision = self.discriminator(generated_images, features)
        return generated_images

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def _generator_step(self, real_imgs, features):
        '''Measure generators's ability to generate samples that can fool the discriminator'''

        z = torch.randn(
            real_imgs.shape[0], self.hparams.latent_dim, 1, 1).type_as(real_imgs)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid_ground_truth = torch.ones(
            real_imgs.size(0), 1).type_as(real_imgs)

        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(
            self.discriminator(self.generator(z, features), features), valid_ground_truth)
        self.log('train/g_loss', g_loss, on_epoch=True,
                 on_step=True, logger=True, prog_bar=True)
        return g_loss

    def _discriminator_step(self, real_imgs, features):
        '''Measure discriminator's ability to differntiate between real and generated samples'''
        z = torch.randn(
            real_imgs.shape[0], self.hparams.latent_dim, 1, 1).type_as(real_imgs)

        # ground truth result (ie: all fake)
        # how well can it label as real?
        real_ground_truth = torch.ones(real_imgs.size(0), 1).uniform_(
            self.hparams.label_smoothing, 1)  # label smoothing, if set to 1 nothing changes here
        real_ground_truth = randomly_flip_labels(
            real_ground_truth, p=self.hparams.label_flipping_p).type_as(real_imgs)
        fake_ground_truth = randomly_flip_labels(
            torch.zeros(real_imgs.size(0), 1), p=self.hparams.label_flipping_p).type_as(real_imgs)

        real_predictions = self.discriminator(real_imgs, features)
        real_loss = self.adversarial_loss(real_predictions, real_ground_truth)
        real_detection_accuracy = accuracy(
            real_predictions, torch.ones(real_imgs.size(0), 1, dtype=int, device=self.device))

        fake_predictions = self.discriminator(
            self.generator(z, features).detach(), features)
        fake_loss = self.adversarial_loss(fake_predictions, fake_ground_truth)
        fake_detection_accuracy = accuracy(
            fake_predictions, torch.zeros(real_imgs.size(0), 1, dtype=int, device=self.device))

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log('train/d_loss', d_loss, on_epoch=True,
                 on_step=True, prog_bar=True)
        self.log('train/d_accuracy_fake', fake_detection_accuracy,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/d_accuracy_real', real_detection_accuracy,
                 on_step=False, on_epoch=True, prog_bar=True)
        return d_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, features = batch
        # train discriminator
        if optimizer_idx == 0:
            return self._discriminator_step(real_imgs, features)

        # train generator
        if optimizer_idx == 1:
            return self._generator_step(real_imgs, features)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_d, opt_g], []

    def on_epoch_end(self):
        z = self.validation_z.to(self.device)

        # log sampled images
        sample_imgs = self.generator(z, self.example_feature_array.type_as(z))
        grid = torchvision.utils.make_grid(sample_imgs[:6])
        self.logger.experiment.log({'epoch_generated_images': [
            wandb.Image(grid, caption=f"Samples epoch {self.current_epoch}")]}, commit=False)

        # Save preliminary model to wandb in case of crash
        push_file_to_wandb(f"{self.argparse_config.results_dir}/last.ckpt")
