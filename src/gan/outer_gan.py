
from enum import Enum
from torch.functional import Tensor
import wandb

import torch.nn.functional as F
import torchvision
from torch import nn
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, hamming_distance

from src.common.helpers import push_file_to_wandb, randomly_flip_labels


class UpsamplingMode(Enum):
    transposed_conv = "transposed_conv"
    subpixel = "subpixel"
    regular_conv = "regular_conv"


class Normalization(Enum):
    batch = "batch"
    instance = "instance"  # TODO: imlpement this
    layer = "layer"
    no_norm = "no_norm"


class ConditionMode(Enum):
    unconditional = "unconditional"
    simple_conditioning = "simple_conditioning"
    simple_embedding = "simple_embedding"  # TODO: implement this
    auxiliary = "auxiliary"  # TODO: implement this


from src.gan.inner_gans import DCGenerator, DCDiscriminator  # nopep8 # avoid cyclical import error

generator_dict = {
    'dc': DCGenerator
}

discriminator_dict = {
    'dc': DCDiscriminator
}


class GAN(pl.LightningModule):

    def __init__(
        self,
        channels: int, width: int, height: int,
        latent_dim: int, num_labels: int, lr: float,
        batch_size: int, label_flipping_p: float,
        label_smoothing: float, b1: float = 0.5, b2: float = 0.99,
        generator_type: str = list(generator_dict.keys())[0],
        discriminator_type: str = list(discriminator_dict.keys())[0],
        condition_mode: ConditionMode = ConditionMode.unconditional,
        upsampling_mode: UpsamplingMode = UpsamplingMode.transposed_conv,
        discriminator_normalization: Normalization = Normalization.batch,
        ** kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        print("Using hyperparameters:\n", self.hparams)

        # networks
        data_shape = (channels, width, height)
        self.generator = self._get_generator(data_shape, generator_type)

        self.discriminator = self._get_discriminator(
            data_shape, discriminator_type)
        self.validation_z = torch.randn(
            batch_size, self.hparams.latent_dim, 1, 1)

        self.example_input_array = torch.zeros(
            batch_size, self.hparams.latent_dim, 1, 1)

        # Create example label vector in [-1,1]
        self.example_label_array = torch.randn(
            batch_size, self.hparams.num_labels)
        self.example_label_array = torch.where(
            self.example_label_array > 0, 1, -1)

    def set_argparse_config(self, config):
        '''Call before training start'''
        self.argparse_config = config
        return self

    def _get_generator(self, data_shape, generator_type) -> DCGenerator:
        GeneratorClass = generator_dict[generator_type]
        generator = GeneratorClass(latent_dim=self.hparams.latent_dim,
                                   num_labels=self.hparams.num_labels, img_shape=data_shape,
                                   condition_mode=self.hparams.condition_mode, upsampling_mode=self.hparams.upsampling_mode)
        generator.apply(self._weights_init)
        return generator

    def _get_discriminator(self, data_shape, discriminator_type) -> DCDiscriminator:
        DiscriminatorClass = discriminator_dict[discriminator_type]
        discriminator = DiscriminatorClass(num_labels=self.hparams.num_labels, img_shape=data_shape,
                                           condition_mode=self.hparams.condition_mode, normalization=self.hparams.discriminator_normalization)
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
        elif classname.find("InstanceNorm") != -1:
            try:
                torch.nn.init.normal_(m.weight, 1.0, 0.02)
                torch.nn.init.zeros_(m.bias)
            except Exception as e:
                print(
                    e, "InstanceNorms have been created without weights. That is okay.")

    def forward(self, z, labels=None):
        '''Do a whole pass through the GAN but return only the genrated images. Don't use if performance is key'''
        if labels is None:
            labels = self.example_label_array.type_as(z)
        generated_images = self.generator(z, labels)
        # Do this step to also show the discriminator in PyTorch-Lightning's automatic model summary
        discriminator_decision = self.discriminator(generated_images, labels)
        return generated_images

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def classification_loss(self, y_hat, y):
        """Used by auxiliary conditioning"""
        shifted_y = torch.where(y == -1, torch.tensor(0.),
                                y)  # shift labels to [0;1]
        return F.binary_cross_entropy_with_logits(y_hat, shifted_y)

    def _generator_step(self, real_imgs, labels):
        '''Measure generators's ability to generate samples that can fool the discriminator'''
        batch_size = real_imgs.shape[0]
        z = torch.randn(batch_size, self.hparams.latent_dim,
                        1, 1).type_as(real_imgs)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid_ground_truth = torch.ones(batch_size, 1).type_as(real_imgs)

        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(
            self.discriminator(self.generator(z, labels), labels), valid_ground_truth)
        self.log('train/g_loss', g_loss, on_epoch=True,
                 on_step=True, logger=True, prog_bar=True)
        return g_loss

    def _discriminator_step(self, real_imgs, labels):
        '''Measure discriminator's ability to differentiate between real and generated samples'''
        batch_size = real_imgs.shape[0]

        z = torch.randn(batch_size, self.hparams.latent_dim,
                        1, 1).type_as(real_imgs)

        real_ground_truth_standard, fake_ground_truth_standard = torch.ones(
            batch_size, 1).type_as(real_imgs), torch.zeros(batch_size, 1).type_as(real_imgs)

        # Apply label smoothing if specified; .clone() to prevent modifying the standard ground truth
        real_ground_truth = real_ground_truth_standard.clone().uniform_(
            self.hparams.label_smoothing, 1)
        # Apply label flipping if specified
        real_ground_truth = randomly_flip_labels(
            real_ground_truth, p=self.hparams.label_flipping_p)
        fake_ground_truth = randomly_flip_labels(
            fake_ground_truth_standard, p=self.hparams.label_flipping_p)

        # Measure discriminator ability to detect real images
        real_predictions = self.discriminator(real_imgs, labels)
        real_loss = self.adversarial_loss(real_predictions, real_ground_truth)
        real_detection_accuracy = accuracy(
            torch.sigmoid(real_predictions), real_ground_truth_standard.int())

        # Measure discriminator ability to detect fake images
        fake_predictions = self.discriminator(
            self.generator(z, labels).detach(), labels)
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
        real_imgs, labels = batch
        # train discriminator
        if optimizer_idx == 0:
            return self._discriminator_step(real_imgs, labels)

        # train generator
        if optimizer_idx == 1:
            return self._generator_step(real_imgs, labels)

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
        # Don't log every epoch, it's too much... maybe a cmd arg later on
        if self.current_epoch % 20 != 0:
            return
        # TODO: do we need to call self.generator.eval() here?
        z = self.validation_z.to(self.device)

        # log sampled images
        sample_imgs = self.generator(z, self.example_label_array.type_as(z))
        grid = torchvision.utils.make_grid(sample_imgs[:16])
        self.logger.experiment.log({'epoch_generated_images': [
            wandb.Image(grid, caption=f"Samples epoch {self.current_epoch}")]}, commit=False)

        # Save preliminary model to wandb in case of crash
        push_file_to_wandb(f"{self.argparse_config.results_dir}/last.ckpt")


class WGAN_GP(GAN):
    '''Based on https://github.com/nocotan/pytorch-lightning-gans/blob/master/models/wgan_gp.py'''

    def __init__(self, *args, b1=0, b2=0.9, discriminator_normalization=Normalization.layer, **kwargs):
        ''' Set betas for Adam as recomended in "Improved Training of Wasserstein GANs"'''
        super().__init__(*args, **kwargs, b1=b1, b2=b2,
                         discriminator_normalization=discriminator_normalization)

    def adversarial_loss(self, predictions, should_be_real=True):
        '''The discriminator should learn to assign high values (>0, close to 1) to real images and low values (<0, clos to -1) to fake images'''
        return -torch.mean(predictions) if should_be_real else torch.mean(predictions)

    def _generator_step(self, real_imgs, labels):
        batch_size = real_imgs.shape[0]
        z = torch.randn(batch_size, self.hparams.latent_dim,
                        1, 1).type_as(real_imgs)

        if self.hparams.condition_mode is ConditionMode.auxiliary:
            predictions, classification = self.discriminator(
                self.generator(z, labels), labels)
        else:
            predictions = self.discriminator(self.generator(z, labels), labels)

        # the generator should learn to fool the discriminator
        loss = self.adversarial_loss(predictions, should_be_real=True)
        if self.hparams.condition_mode is ConditionMode.auxiliary:
            loss += self.classification_loss(classification, labels)

        self.log('train/g_loss', loss, on_epoch=True,
                 on_step=True, logger=True, prog_bar=True)
        self.log('train/g_fake_logits', torch.mean(predictions),
                 on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def compute_gradient_penalty(self, real_samples, fake_samples, labels):
        """Calculates the gradient penalty loss for WGAN GP"""
        # labels are hacked, needs more thought

        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random(
            (real_samples.size(0), 1, 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha)
                        * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(self.device)
        if self.hparams.condition_mode is ConditionMode.auxiliary:
            d_interpolates, _ = self.discriminator(interpolates, labels)
        else:
            d_interpolates = self.discriminator(interpolates, labels)

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

    def _discriminator_step(self, real_imgs, labels):
        '''Measure discriminator's ability to differentiate between real and generated samples'''
        batch_size = real_imgs.shape[0]

        z = torch.randn(batch_size, self.hparams.latent_dim,
                        1, 1).type_as(real_imgs)

        # Measure discriminator ability to detect real images
        if self.hparams.condition_mode is ConditionMode.auxiliary:
            real_predictions, real_classification = self.discriminator(
                real_imgs, labels)
        else:
            real_predictions = self.discriminator(real_imgs, labels)

        real_loss = self.adversarial_loss(
            real_predictions, should_be_real=True)
        real_detection_accuracy = accuracy(
            self.discretize_discriminator_output(real_predictions), torch.ones_like(real_predictions, dtype=int))

        # Measure discriminator ability to detect fake images
        fake_imgs = self.generator(z, labels).detach()
        if self.hparams.condition_mode is ConditionMode.auxiliary:
            fake_predictions, fake_classification = self.discriminator(
                fake_imgs, labels)
        else:
            fake_predictions = self.discriminator(fake_imgs, labels)

        fake_loss = self.adversarial_loss(
            fake_predictions, should_be_real=False)
        fake_detection_accuracy = accuracy(
            self.discretize_discriminator_output(fake_predictions), torch.zeros_like(fake_predictions, dtype=int))

        if self.hparams.condition_mode is ConditionMode.auxiliary:
            real_loss += self.classification_loss(real_classification, labels)
            fake_loss += self.classification_loss(fake_classification, labels)
            self.log('train/d_hamming_real',
                     hamming_distance(torch.sigmoid(real_classification), labels.where(labels == 1, torch.tensor(0.)).int()))
            self.log('train/d_hamming_fake',
                     hamming_distance(torch.sigmoid(fake_classification), labels.where(labels == 1, torch.tensor(0.)).int()))
        gp = self.compute_gradient_penalty(
            real_imgs.data, fake_imgs.data, labels)
        # TODO: fix magic value
        d_loss = real_loss + fake_loss + 10 * gp
        self.log('train/d_loss', d_loss, on_epoch=True,
                 on_step=True, prog_bar=True)
        self.log('train/gradient_penalty', gp, on_epoch=True,
                 on_step=False, prog_bar=False)
        self.log('train/d_accuracy_fake', fake_detection_accuracy,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/d_accuracy_real', real_detection_accuracy,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/d_fake_logits', torch.mean(fake_predictions),
                 on_step=True, on_epoch=True, prog_bar=False)
        self.log('train/d_real_logits', torch.mean(real_predictions),
                 on_step=True, on_epoch=True, prog_bar=False)
        return d_loss

    def configure_optimizers(self):
        """Train discriminator more than generator"""
        # TODO: fix magic values
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return {'optimizer': opt_d, 'frequency': 5}, {'optimizer': opt_g, 'frequency': 1}
