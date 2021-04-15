import pytorch_lightning as pl
import torch
import torchvision
import torch.nn.functional as F
from deep_convolutional_model import Generator, Discriminator
from helpers import randomly_flip_labels
from naive_model import NaiveGenerator, NaiveDiscriminator
import wandb
from optimizers import ExtraAdam


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
        condition=True,
        ** kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        data_shape = (channels, width, height)
        self.generator = Generator(
            latent_dim=self.hparams.latent_dim, num_features=self.hparams.num_features, img_shape=data_shape)
        self.discriminator = Discriminator(
            num_features=self.hparams.num_features, img_shape=data_shape)

        self.validation_z = torch.randn(8, self.hparams.latent_dim, 1, 1)

        self.example_input_array = torch.zeros(
            8, self.hparams.latent_dim, 1, 1)
        self.example_feature_array = torch.zeros(
            8, self.hparams.num_features)

    def forward(self, z, features=None):
        if features is None:
            features = self.example_feature_array.type_as(z)
        return self.discriminator(self.generator(z, features), features)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, features = batch
        # sample noise
        z = torch.randn(
            imgs.shape[0], self.hparams.latent_dim, 1, 1).type_as(imgs)

        # train generator
        if optimizer_idx == 0:
            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(
                self.discriminator(self.generator(z, features), features), valid)

            self.log('train/g_loss', g_loss, on_epoch=True,
                     on_step=True, logger=True, prog_bar=True)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = randomly_flip_labels(valid, p=0.2)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(
                self.discriminator(imgs, features), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = randomly_flip_labels(fake, p=0.2)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(
                self.discriminator(self.generator(z, features).detach(), features), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log('train/d_loss', d_loss, on_epoch=True,
                     on_step=True, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = 0.5  # self.hparams.b1
        b2 = 0.99  # self.hparams.b2

        # This is supposed to be better for GANs than Adam
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        z = self.validation_z.to(self.device)

        # log sampled images
        sample_imgs = self.generator(z, self.example_feature_array.type_as(z))
        grid = torchvision.utils.make_grid(sample_imgs[:6])
        self.logger.experiment.log({'epoch_generated_images': [
            wandb.Image(grid, caption=f"Samples epoch {self.current_epoch}")]}, commit=False)
