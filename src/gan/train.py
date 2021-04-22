# Fix imports and prevent formatting
import sys  # nopep8
from os.path import dirname, join, abspath  # nopep8
sys.path.insert(0, abspath(join(dirname(__file__), '..')))  # nopep8


import wandb
from gan.outer_gan import conditionalGAN
from common.helpers import WANDB_PROJECT_NAME, push_file_to_wandb, start_wandb_logging, before_run
from common.data_loading import CelebADataModule, WikiArtEmotionsDataModule
from common.argparser import get_training_parser, parse_config
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl


if __name__ == '__main__':
    # Needed because of multiprocessing error in Google Colab
    __spec__ = None
    parser = get_training_parser()
    config = parse_config(parser)
    before_run(config)

    print("Loading Data")
    dm = WikiArtEmotionsDataModule(
        config.data_dir, config.batch_size, config.workers, config.image_resizing, fast_debug=config.fast_debug)
    model = conditionalGAN(*dm.size(), lr=config.lr,
                           batch_size=config.batch_size, latent_dim=config.latent_dim,
                           num_features=config.num_features, label_flipping_p=config.label_flipping_p,
                           label_smoothing=config.label_smoothing,
                           generator_type=config.generator_type, discriminator_type=config.discriminator_type).set_argparse_config(config)

    start_wandb_logging(config, model, project=WANDB_PROJECT_NAME)
    logger = WandbLogger(project=WANDB_PROJECT_NAME, experiment=wandb.run)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.results_dir, save_last=True)
    trainer = pl.Trainer.from_argparse_args(config, gpus=config.gpus, max_epochs=config.epochs,
                                            progress_bar_refresh_rate=1, logger=logger, callbacks=[checkpoint_callback])

    trainer.fit(model, dm)
    push_file_to_wandb(f"{config.results_dir}/*.ckpt")
