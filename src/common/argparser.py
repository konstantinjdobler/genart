import argparse
from datetime import datetime
from src.gan.blocks import UpsamplingMode
from src.gan.inner_gans import ConditionMode
from src.gan.outer_gan import Normalization, generator_dict, discriminator_dict
from pathlib import Path


def get_training_parser():
    parser = argparse.ArgumentParser("Parsing training arguments")

    parser.add_argument('--data-dir', '-d', type=str,
                        default="./data/wikiart-emotions")
    parser.add_argument('--image-folder', type=str,
                        default="./data/wikiart-emotions/images")
    parser.add_argument('--annotations-file', type=str)
    parser.add_argument('--gpus', type=int, nargs='+', default=-1,
                        help="specify gpus to use. default is to use all available")
    parser.add_argument('--cpu', action='store_true',
                        help="use cpu instead of gpu")
    parser.add_argument('--workers', '-w', type=int, default=4)
    parser.add_argument('--use-checkpoint', default=None,
                        help="If wanted, specify path to checkpoint file to load")
    ##################### ------------------ #####################
    parser.add_argument('--generator-type', '--gen', default=list(generator_dict.keys())[0],
                        choices=list(generator_dict.keys()), help="Specify the type of generator that will be used.")
    parser.add_argument('--discriminator-type', '--disc', default=list(discriminator_dict.keys())[0],
                        choices=list(discriminator_dict.keys()), help="Specify the type of discriminator that will be used.")
    parser.add_argument('--condition-mode', type=lambda x: ConditionMode(x), default=ConditionMode.unconditional,
                        choices=ConditionMode, help="Specify the conditioning to use / or unconditional.")
    parser.add_argument('--upsampling-mode', type=lambda x: UpsamplingMode(x), default=UpsamplingMode.transposed_conv,
                        choices=UpsamplingMode, help="Specify the upsampling mode.")
    parser.add_argument('--image-resizing', '-i', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--epochs', '-e', type=int, default=20)
    parser.add_argument('--batch-size', '-b', type=int, default=4)
    parser.add_argument('--num-labels', '-f', type=int, default=20)
    parser.add_argument('--latent-dim', '-l', type=int, default=100)
    parser.add_argument('--label-flipping-p', '--lfp', type=float, default=0,
                        help="Probability for label flipping. Set to zero to disable.")
    parser.add_argument('--label-smoothing', '--ls', type=float, default=1.0,
                        help="Use label smoothing trick for discriminator. Argument determines how much labels are smoothed, 1 disables the trick.")
    parser.add_argument('--wasserstein', action='store_true',
                        help="Use Wasserstein-1 distance formulation instead of classic GAN")
    parser.add_argument('--discriminator-normalization', type=lambda x: Normalization(x), default=None, choices=Normalization,
                        help="Type of normalization to use for the discriminator. Default is dependent on the GAN type.")
    ###################### ------------------ #####################
    parser.add_argument('--results-dir', '-c', type=str,
                        default=f"./mnt/results/")
    parser.add_argument('--wandb-project-name', '-wpn', type=str,
                        help='Name of the project in wandb to log to', default='genart-spike')
    parser.add_argument('--training-name', '-n', type=str,
                        help='Name you want to use for this training run, will be used in '
                        'log and model saving.', default=datetime.now().strftime('%d-%m-%Y_%H_%M_%S'))
    parser.add_argument('--no-wandb', action="store_true",
                        help="Disable logging to wandb")
    parser.add_argument('--tags', type=str, nargs='+',
                        default=[], help="Tag this run in wandb.")
    parser.add_argument('--fast-debug', action="store_true",
                        help="do a fast run through the code to check for errors")
    parser.add_argument('--pred-threshold', '-pt', type=float,
                        default=0.5, help="Threshold to make a positive prediction")

    return parser


def parse_config(parser: argparse.ArgumentParser):
    config = parser.parse_args()

    if config.cpu is True:
        config.gpus = None
    if config.fast_debug is True:
        config.batch_size = 4
        config.epochs = 3
        config.no_wandb = True
    config.results_dir = config.results_dir + config.training_name + (
        datetime.now().strftime('%d-%m-%Y_%H_%M_%S') if Path(config.results_dir + config.training_name).exists() else "")

    return config
