# Fix imports and prevent formatting
import sys  # nopep8
from os.path import dirname, join, abspath
from typing import OrderedDict  # nopep8
sys.path.insert(0, abspath(join(dirname(__file__), '..')))  # nopep8

import argparse
import torch
import torchvision
import wandb
from emotion_classifier.emotion_classifier import EmotionClassifier
from src.gan.outer_gan import GAN, WGAN_GP

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
parser.add_argument('--condition-file', type=str,
                    help="parse conditions from file")
parser.add_argument('--conditions', type=str, nargs="+", default=[],
                    help="specify conditions by cmd arg")


def loadAttributes(attributesPath):
    with open(attributesPath) as file:
        lines = [line.rstrip() for line in file]
    attributes = torch.FloatTensor(
        [float(line.split(': ')[-1]) for line in lines])
    attributes[attributes == 0] = -1
    return attributes


emotions = OrderedDict([("agreeableness", -1),
                        ("anger", -1),
                        ("anticipation", -1),
                        ("arrogance", -1),
                        ("disagreeableness", -1),
                        ("disgust", -1),
                        ("fear", -1),
                        ("gratitude", -1),
                        ("happiness", -1),
                        ("humility", -1),
                        ("love", -1),
                        ("optimism", -1),
                        ("pessimism", -1),
                        ("regret", -1),
                        ("sadness", -1),
                        ("shame", -1),
                        ("shyness", -1),
                        ("surprise", -1),
                        ("trust", -1),
                        ("neutral", -1)])

if __name__ == "__main__":
    # TODO: seed
    config = parser.parse_args()
    print(config)

    def loadAttributes(attributesPath):
        with open(attributesPath) as file:
            lines = [line.rstrip() for line in file]
        attributes = torch.FloatTensor(
            [float(line.split(': ')[-1]) for line in lines])
        attributes[attributes == 0] = -1
        return attributes

    checkpoint_file = config.checkpoint
    if config.wandb:
        model_file = wandb.restore(checkpoint_file, run_path=config.run_path)
        checkpoint_file = model_file.name

    gan_class = class_names[config.class_name]
    model: GAN = gan_class.load_from_checkpoint(
        checkpoint_file, map_location=torch.device("cpu"))
    model.eval()
    latent_dim = model.hparams.latent_dim

    z = torch.randn(config.num_images, latent_dim, 1, 1)
    if config.condition_file:
        labels = torch.cat(config.num_images *
                           [loadAttributes(config.condition_file)])
    elif config.conditions:
        for condi in config.conditions:
            emotions[condi] = 1
        labels = torch.cat(config.num_images *
                           [torch.FloatTensor(list(emotions.values()))])
        if config.output_image == "output/image.png":
            config.output_image = "output/" + \
                "_".join(config.conditions) + ".png"
    else:
        labels = model.example_label_array
    output = model.generator(z, labels)
    grid = torchvision.utils.make_grid(output)
    img = wandb.Image(grid)
    img._image.save(config.output_image)
