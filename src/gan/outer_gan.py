from torch.functional import Tensor
import wandb

import torch.nn.functional as F
import torchvision
from torch import nn
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

from src.gan.unconditional_dc_gan import DCDiscriminator, DCGenerator, DCGeneratorSmoothed, WassersteinDiscriminator
from src.common.helpers import push_file_to_wandb, randomly_flip_labels
from src.gan.conditional_dc_gan import cDCGenerator, cDCDiscriminator, cDCGeneratorSmoothed

generator_dict = {
    'cdc-smoothed': cDCGeneratorSmoothed,
    'cdc': cDCGenerator,
    'dc-smoothed': DCGeneratorSmoothed,
    'dc': DCGenerator
}

discriminator_dict = {
    'cdc': cDCDiscriminator,
    'dc': DCDiscriminator,
    'dc-wasserstein': WassersteinDiscriminator
}


class GAN(pl.LightningModule):

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
        self.generator = self._get_generator(data_shape, generator_type)
        self.discriminator = self._get_discriminator(
            data_shape, discriminator_type)

        self.validation_z = torch.randn(8, self.hparams.latent_dim, 1, 1)

        self.example_input_array = torch.zeros(
            8, self.hparams.latent_dim, 1, 1)

        # Create example feature vector in [-1,1]
        self.example_feature_array = torch.randn(
            8, self.hparams.num_features)
        self.example_feature_array[self.example_feature_array <= 0] = -1
        self.example_feature_array[self.example_feature_array > 0] = 1

    def set_argparse_config(self, config):
        '''Call before training start'''
        self.argparse_config = config
        return self

    def _get_generator(self, data_shape, generator_type) -> nn.Module:
        GeneratorClass = generator_dict[generator_type]
        print("Using generator", GeneratorClass.__name__)
        generator = GeneratorClass(latent_dim=self.hparams.latent_dim,
                                   num_features=self.hparams.num_features, img_shape=data_shape)
        generator.apply(self._weights_init)
        return generator

    def _get_discriminator(self, data_shape, discriminator_type) -> nn.Module:
        DiscriminatorClass = discriminator_dict[discriminator_type]
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
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def _generator_step(self, real_imgs, features):
        '''Measure generators's ability to generate samples that can fool the discriminator'''
        batch_size = real_imgs.shape[0]
        z = torch.randn(batch_size, self.hparams.latent_dim,
                        1, 1).type_as(real_imgs)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid_ground_truth = torch.ones(batch_size, 1).type_as(real_imgs)

        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(
            self.discriminator(self.generator(z, features), features), valid_ground_truth)
        self.log('train/g_loss', g_loss, on_epoch=True,
                 on_step=True, logger=True, prog_bar=True)
        return g_loss

    def _discriminator_step(self, real_imgs, features):
        '''Measure discriminator's ability to differntiate between real and generated samples'''
        batch_size = real_imgs.shape[0]

        z = torch.randn(batch_size, self.hparams.latent_dim,
                        1, 1).type_as(real_imgs)

        real_ground_truth_standard, fake_ground_truth_standard = torch.ones(
            batch_size, 1).type_as(real_imgs), torch.zeros(batch_size, 1).type_as(real_imgs)

        # Apply label smoothing if specified
        real_ground_truth = real_ground_truth_standard.uniform_(
            self.hparams.label_smoothing, 1)
        # Apply label flipping if specified
        real_ground_truth = randomly_flip_labels(
            real_ground_truth, p=self.hparams.label_flipping_p)
        fake_ground_truth = randomly_flip_labels(
            fake_ground_truth_standard, p=self.hparams.label_flipping_p)

        # Measure discriminator ability to detect real images
        real_predictions = self.discriminator(real_imgs, features)
        real_loss = self.adversarial_loss(real_predictions, real_ground_truth)
        real_detection_accuracy = accuracy(
            torch.sigmoid(real_predictions), real_ground_truth_standard.int())

        # Measure discriminator ability to detect fake images
        fake_predictions = self.discriminator(
            self.generator(z, features).detach(), features)
        fake_loss = self.adversarial_loss(fake_predictions, fake_ground_truth)
        fake_detection_accuracy = accuracy(
            torch.sigmoid(fake_predictions), fake_ground_truth_standard.int())

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


class WGAN_GP(GAN):
    def adversarial_loss(self, predictions, should_be_real=True):
        '''The discriminator should learn to assign high values (>0, close to 1) to real images and low values (<0, clos to -1) to fake images'''
        return -torch.mean(predictions) if should_be_real else torch.mean(predictions)

    def _get_discriminator(self, data_shape, discriminator_type) -> nn.Module:
        '''Wasserstein GANs cannot use batch norm in discriminator, wo we overwrite here'''
        return super()._get_discriminator(data_shape, discriminator_type + "-wasserstein")

    def _generator_step(self, real_imgs, features):
        batch_size = real_imgs.shape[0]
        z = torch.randn(batch_size, self.hparams.latent_dim,
                        1, 1).type_as(real_imgs)
        predictions = self.discriminator(self.generator(z, features), features)

        # the generator should learn to fool the discriminator
        loss = self.adversarial_loss(predictions, should_be_real=True)
        self.log('train/g_loss', loss, on_epoch=True,
                 on_step=True, logger=True, prog_bar=True)
        return loss

    def compute_gradient_penalty(self, real_samples, fake_samples, features):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Features are hacked, needs more thought

        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random(
            (real_samples.size(0), 1, 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha)
                        * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(self.device)
        d_interpolates = self.discriminator(interpolates, features)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(
            1.0).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def discretize_discriminator_output(self, d_out: Tensor):
        out = d_out.clone().detach()
        out[out <= 0] = 0
        out[out > 0] = 1
        return out

    def _discriminator_step(self, real_imgs, features):
        '''Measure discriminator's ability to differentiate between real and generated samples'''
        batch_size = real_imgs.shape[0]

        z = torch.randn(batch_size, self.hparams.latent_dim,
                        1, 1).type_as(real_imgs)

        # Measure discriminator ability to detect real images
        real_predictions = self.discriminator(real_imgs, features)
        real_loss = self.adversarial_loss(
            real_predictions, should_be_real=True)
        real_detection_accuracy = accuracy(
            self.discretize_discriminator_output(real_predictions), torch.ones_like(real_predictions, dtype=int))

        # Measure discriminator ability to detect fake images
        fake_imgs = self.generator(z, features).detach()
        fake_predictions = self.discriminator(fake_imgs, features)
        fake_loss = self.adversarial_loss(
            fake_predictions, should_be_real=False)

        fake_detection_accuracy = accuracy(
            self.discretize_discriminator_output(fake_predictions), torch.zeros_like(fake_predictions, dtype=int))

        # discriminator loss is the average of these
        gp = self.compute_gradient_penalty(
            real_imgs.data, fake_imgs.data, features)
        d_loss = real_loss + fake_loss + 10 * gp
        self.log('train/d_loss', d_loss, on_epoch=True,
                 on_step=True, prog_bar=True)
        self.log('train/gradient_penalty', gp, on_epoch=True,
                 on_step=False, prog_bar=False)
        self.log('train/d_accuracy_fake', fake_detection_accuracy,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/d_accuracy_real', real_detection_accuracy,
                 on_step=False, on_epoch=True, prog_bar=True)
        return d_loss

    def configure_optimizers(self):
        return super().configure_optimizers()
