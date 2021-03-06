# Fix imports and prevent formatting
import sys  # nopep8
from os.path import dirname, join, abspath  # nopep8
sys.path.insert(0, abspath(join(dirname(__file__), '..')))  # nopep8

import argparse
import torch
import torchvision
import wandb
from emotion_classifier.emotion_classifier import EmotionClassifier
from src.gan.outer_gan import GAN, WGAN_GP
from src.common.condition_helpers import emotions, faces

class_names = {
    "gan": GAN,
    "wgan": WGAN_GP,
}

# celeb
# file_path = "mnt/results/wgan-celeba-layer11-05-2021_17_09_07/last.ckpt"
# run_path = "hpi-genart/genart-spike/2jqs8jz4"

parser = argparse.ArgumentParser("Parsing generate image arguments")

parser.add_argument('--checkpoint', '-c', type=str,
                    default="mnt/results/wgan-layer-norm-continued/last.ckpt", help="Location of checkpoint file.")
parser.add_argument('--wandb', action='store_true',
                    help="Use a checkpoint file from wandb.")
parser.add_argument('--run-path', '-r', type=str,
                    default="hpi-genart/genart-spike/154mkxdc", help="Wandb run path (find in overview).")
parser.add_argument('--class-name', '-cn', default=list(class_names.keys())[0],
                    choices=list(class_names.keys()), help="Specify the type of gan that will be used.")
parser.add_argument('--num-images', '-n', type=int,
                    default=64, help="Number of images to generate.")
parser.add_argument('--output-image', '-o', type=str,
                    default="output/image.png", help="Location of output image.")
parser.add_argument('--nrow', '-nr', type=int,
                    default=8, help="Number of rows in the final grid.")
parser.add_argument('--condition-template', type=str, default="",
                    help="parse conditions from file OR use to specify conditions type: emotions or faces")
parser.add_argument('--conditions', type=str, nargs="+", default=[],
                    help="specify conditions by cmd arg")


def loadAttributes(attributesPath):
    with open(attributesPath) as file:
        lines = [line.rstrip() for line in file]
    attributes = torch.FloatTensor(
        [float(line.split(': ')[-1]) for line in lines])
    attributes[attributes == 0] = -1
    return attributes


if __name__ == "__main__":
    # TODO: seed, otherwise there is no way to reconstruct the sample images
    config = parser.parse_args()
    print(config)

    checkpoint_file = config.checkpoint
    if config.wandb:
        model_file = wandb.restore(checkpoint_file, run_path=config.run_path)
        checkpoint_file = model_file.name

    gan_class = class_names[config.class_name]
    model: GAN = gan_class.load_from_checkpoint(
        checkpoint_file, map_location=torch.device("cpu"))
    model.eval()
    latent_dim, batch_size, num_labels = model.hparams.latent_dim, model.hparams.batch_size, model.hparams.num_labels
    print(model.hparams)

    z = torch.randn(batch_size, latent_dim, 1, 1)
    if config.conditions:
        condition_template = emotions if config.condition_template == "emotions" else faces
        for cond_name in config.conditions:
            condition_template[cond_name] = 1
        labels = torch.cat(config.num_images *
                           [torch.FloatTensor(list(condition_template.values()))])
        if config.output_image == "output/image.png":
            config.output_image = "output/" + \
                "_".join(config.conditions) + ".png"
    elif config.condition_template:
        labels = torch.cat(config.num_images *
                           [loadAttributes(config.condition_template)])
    else:
        print("Using default labels")
        # labels = model.example_label_array
        labels = torch.zeros(batch_size, num_labels, 1, 1)
    with torch.no_grad():
        output = model.generator(z, labels)
        grid = torchvision.utils.make_grid(output[:config.num_images], nrow=config.nrow)
        img = wandb.Image(grid)
        img._image.save(config.output_image)
