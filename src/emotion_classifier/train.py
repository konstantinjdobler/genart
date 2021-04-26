# Fix imports and prevent formatting
import sys  # nopep8
from os.path import dirname, join, abspath  # nopep8
sys.path.insert(0, abspath(join(dirname(__file__), '..')))  # nopep8


import os
import torch
from torchvision import transforms, datasets
import wandb
from emotion_classifier.emotion_classifier import EmotionClassifier
from common.helpers import push_file_to_wandb, start_wandb_logging, before_run
from common.data_loading import CelebADataModule, WikiArtEmotionsDataModule2
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
    num_classes = 20


# data_dir = 'data/hymenoptera_data'
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x])
#                   for x in ['train', 'val']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=config.batch_size,
#                                               shuffle=True, num_workers=config.workers)
#                for x in ['train', 'val']}
# train_dataloader, val_dataloader = dataloaders["train"], dataloaders["val"]


def filter_image_only(column_name: str, idx: int):
    return column_name.startswith("ImageOnly")


dm = WikiArtEmotionsDataModule2(image_folder=config.image_folder, annotations_file=config.annotations_file, resize_size=256, crop_size=224,
                                batch_size=config.batch_size, num_workers=config.workers, filter_columns_function=filter_image_only, fast_debug=config.fast_debug)

model = EmotionClassifier(num_classes).set_argparse_config(config)

if config.use_checkpoint:
    model = EmotionClassifier.load_from_checkpoint(
        config.use_checkpoint).set_argparse_config(config)
start_wandb_logging(config, model, project=config.wandb_project_name)
logger = WandbLogger(project=config.wandb_project_name, experiment=wandb.run)
checkpoint_callback = ModelCheckpoint(
    dirpath=config.results_dir, save_last=True)

trainer = pl.Trainer.from_argparse_args(config, gpus=config.gpus, max_epochs=config.epochs,
                                        progress_bar_refresh_rate=1, logger=logger, callbacks=[checkpoint_callback])

trainer.fit(model, dm)
push_file_to_wandb(f"{config.results_dir}/*.ckpt")
