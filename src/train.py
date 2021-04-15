import argparse
from datetime import datetime
import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, wandb
from torch.backends import cudnn
from data_loading import WikiArtEmotionsDataModule
from helpers import WANDB_PROJECT_NAME, push_file_to_wandb, start_wandb_logging

from models import conditionalGAN
import wandb

parser = argparse.ArgumentParser("Parsing training arguments")

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', '-e', type=int, default=20)
parser.add_argument('--workers', '-w', type=int, default=4)
parser.add_argument('--batch-size', '-b', type=int, default=4)
parser.add_argument('--data-dir', '-d', type=str,
                    default="./data/wikiart-emotions")
parser.add_argument('--num-features', '-f', type=int, default=60)
parser.add_argument('--latent-dim', '-l', type=int, default=200)
parser.add_argument('--results-dir', '-c', type=str,
                    default=f"./results/")
parser.add_argument('--training-name', '-n', type=str,
                    help='Name you want to use for this training run, will be used in '
                         'log and model saving.', default=datetime.now().strftime('%d-%m-%Y_%H_%M_%S'))
parser.add_argument('--image-resizing', '-i', type=int, default=256)
parser.add_argument('--no-wandb', action="store_true",
                    help="Disable logging to wandb")
parser.add_argument('--tags', type=str, nargs='+',
                    default=[], help="Tag this run in wandb.")
parser.add_argument('--gpus', type=int, nargs='+', default=-1,
                    help="specify gpus to use. default is to use all available")
parser.add_argument('--cpu', action='store_true',
                    help="use cpu instead of gpu")
parser.add_argument('--fast-debug', action="store_true",
                    help="do a fast run through the code to check for errors")

if __name__ == '__main__':
    # Needed because of multiprocessing error in Google Colab
    __spec__ = None

    config, _ = parser.parse_known_args()
    if config.cpu is True:
        config.gpus = None
    if config.fast_debug is True:
        config.batch_size = 4
        config.epochs = 2
        config.no_wandb = True
    config.results_dir = config.results_dir + config.training_name
    print(f"Results will be saved to {config.results_dir}")
    os.makedirs(config.results_dir, exist_ok=True)

    print("Loading Data")

    # activate options to speed up training
    cudnn.enabled = True
    cudnn.benchmark = True

    dm = WikiArtEmotionsDataModule(
        config.data_dir, config.batch_size, config.workers, config.image_resizing, fast_debug=config.fast_debug)
    model = conditionalGAN(*dm.size(), lr=config.lr,
                           batch_size=config.batch_size, latent_dim=config.latent_dim, num_features=config.num_features)

    if config.no_wandb:
        os.environ["WANDB_MODE"] = "dryrun"
    from dotenv import load_dotenv
    load_dotenv()
    start_wandb_logging(config, model, project=WANDB_PROJECT_NAME)
    logger = WandbLogger(project=WANDB_PROJECT_NAME, log_model=True)
    trainer = pl.Trainer.from_argparse_args(config, gpus=config.gpus, max_epochs=config.epochs,
                                                progress_bar_refresh_rate=1, logger=logger, weights_save_path=wandb.run.dir)
   
    trainer.fit(model, dm)
    # if not config.no_wandb:
    #     push_file_to_wandb("*.ckpt")
