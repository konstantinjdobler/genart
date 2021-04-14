import argparse
from datetime import datetime
import os

import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.backends import cudnn
from data_loading import WikiArtEmotionsDataModule
from helpers import WANDB_PROJECT_NAME, start_wandb_logging

from models import GAN


parser = argparse.ArgumentParser("Parsing training arguments")

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', '-e', type=int, default=20)
parser.add_argument('--workers', '-w', type=int, default=4)
parser.add_argument('--batch-size', '-b', type=int, default=4)
parser.add_argument('--data-dir', '-d', type=str, required=True)
parser.add_argument('--results-dir', '-c', type=str,
                    default=f"./results/")
parser.add_argument('--training-name', '-n', type=str,
                    help='Name you want to use for this training run, will be used in '
                         'log and model saving.', default=datetime.now().strftime('%d-%m-%Y_%H_%M_%S'))
parser.add_argument('--image-resizing', '-i', type=int, default=400)
parser.add_argument('--no-wandb', action="store_true",
                    help="Disable logging to wandb")
parser.add_argument('--tags', type=str, nargs='+',
                    default=[], help="Tag this run in wandb.")
parser.add_argument('--gpus', type=int, nargs='+', default=[0])
parser.add_argument('--cpu', action='store_true',
                    help="use cpu instead of gpu")

if __name__ == '__main__':
    # Needed because of multiprocessing error in Google Colab
    __spec__ = None

    config, _ = parser.parse_known_args()
    if config.cpu is True:
        config.gpus = None
    config.results_dir = config.results_dir + config.training_name
    print(f"Results will be saved to {config.results_dir}")
    os.makedirs(config.results_dir, exist_ok=True)

    print("Loading Data")

    # activate options to speed up training
    cudnn.enabled = True
    cudnn.benchmark = True

    dm = WikiArtEmotionsDataModule(
        config.data_dir, config.batch_size, config.workers, config.image_resizing)
    model = GAN(*dm.size(), lr=config.lr,
                batch_size=config.batch_size, latent_dim=100)

    logger = None
    if not config.no_wandb:
        from dotenv import load_dotenv
        load_dotenv()
        start_wandb_logging(config, model, project=WANDB_PROJECT_NAME)
        logger = WandbLogger(project=WANDB_PROJECT_NAME, log_model=True)

    trainer = pl.Trainer(gpus=config.gpus, max_epochs=config.epochs,
                         progress_bar_refresh_rate=20)
    trainer.fit(model, dm)
