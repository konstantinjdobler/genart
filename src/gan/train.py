# Fix imports and prevent formatting
import sys  # nopep8
from os.path import dirname, join, abspath  # nopep8
sys.path.insert(0, abspath(join(dirname(__file__), '../..')))  # nopep8


import wandb
from src.gan.outer_gan import GAN, WGAN_GP
from src.common.helpers import push_file_to_wandb, start_wandb_logging, before_run
from src.common.data_loading import CelebADataModule, WikiArtEmotionsDataModule
from src.common.argparser import get_training_parser, parse_config
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

# https://bugs.python.org/issue43743
# on PowerPC or RedHat Linux there is a bug that sometimes leads to errors when copying files, which wandb does
# recommended fix is below
import shutil
shutil._USE_CP_SENDFILE = False

if __name__ == '__main__':
    # Needed because of multiprocessing error in Google Colab
    __spec__ = None
    parser = get_training_parser()
    config = parse_config(parser)
    before_run(config)

    print("Loading Data")
    if config.celeba:
        dm = CelebADataModule(
            config.data_dir, config.batch_size, config.workers, config.image_resizing, fast_debug=config.fast_debug)
    else:
        dm = WikiArtEmotionsDataModule(
            config.data_dir, config.annotations_file, config.batch_size, config.workers, config.image_resizing, fast_debug=config.fast_debug)

    GANClass = WGAN_GP if config.wasserstein else GAN
    gan_keyword_args = dict(lr=config.lr,
                            batch_size=config.batch_size, latent_dim=config.latent_dim,
                            num_labels=config.num_labels, label_flipping_p=config.label_flipping_p,
                            label_smoothing=config.label_smoothing,
                            generator_type=config.generator_type, discriminator_type=config.discriminator_type,
                            condition_mode=config.condition_mode, upsampling_mode=config.upsampling_mode,
                            discriminator_normalization=config.discriminator_normalization)
    # Filter out None cmd args so that they don't overwrite the default values specified in the implementation
    filtered_keyword_args = {k: v for k,
                             v in gan_keyword_args.items() if v is not None}
    model = GANClass(
        *dm.size(), **filtered_keyword_args).set_argparse_config(config)

    if config.use_checkpoint:
        model = GANClass.load_from_checkpoint(
            config.use_checkpoint).set_argparse_config(config)
    start_wandb_logging(config, model, project=config.wandb_project_name)
    logger = WandbLogger(project=config.wandb_project_name,
                         experiment=wandb.run)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.results_dir, save_last=True)
    trainer = pl.Trainer.from_argparse_args(config, gpus=config.gpus, max_epochs=config.epochs,
                                            progress_bar_refresh_rate=1, logger=logger, callbacks=[checkpoint_callback])

    trainer.fit(model, dm)
    push_file_to_wandb(f"{config.results_dir}/*")
