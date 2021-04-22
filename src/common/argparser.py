import argparse
from datetime import datetime
from gan.outer_gan import generator_dict, discriminator_dict


def get_training_parser():
    parser = argparse.ArgumentParser("Parsing training arguments")

    parser.add_argument('--data-dir', '-d', type=str,
                        default="./data/wikiart-emotions")
    parser.add_argument('--gpus', type=int, nargs='+', default=-1,
                        help="specify gpus to use. default is to use all available")
    parser.add_argument('--cpu', action='store_true',
                        help="use cpu instead of gpu")
    parser.add_argument('--workers', '-w', type=int, default=4)
    ##################### ------------------ #####################
    parser.add_argument('--generator-type', '--gen', default=list(generator_dict.keys())[0],
                        choices=list(generator_dict.keys()), help="Specify the type of generator that will be used.")
    parser.add_argument('--discriminator-type', '--disc', default=list(discriminator_dict.keys())[0],
                        choices=list(discriminator_dict.keys()), help="Specify the type of discriminator that will be used.")
    parser.add_argument('--image-resizing', '-i', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--epochs', '-e', type=int, default=20)
    parser.add_argument('--batch-size', '-b', type=int, default=4)
    parser.add_argument('--num-features', '-f', type=int, default=60)
    parser.add_argument('--latent-dim', '-l', type=int, default=100)
    parser.add_argument('--label-flipping-p', '--lfp', type=float, default=0,
                        help="Probability for label flipping. Set to zero to disable.")
    parser.add_argument('--label-smoothing', '--ls', type=float, default=1.0,
                        help="Use label smoothing trick for discriminator. Argument determines how much labels are smoothed, 1 disables the trick.")
    ###################### ------------------ #####################
    parser.add_argument('--results-dir', '-c', type=str,
                        default=f"./results/")
    parser.add_argument('--training-name', '-n', type=str,
                        help='Name you want to use for this training run, will be used in '
                        'log and model saving.', default=datetime.now().strftime('%d-%m-%Y_%H_%M_%S'))
    parser.add_argument('--no-wandb', action="store_true",
                        help="Disable logging to wandb")
    parser.add_argument('--tags', type=str, nargs='+',
                        default=[], help="Tag this run in wandb.")
    parser.add_argument('--fast-debug', action="store_true",
                        help="do a fast run through the code to check for errors")

    return parser


def parse_config(parser: argparse.ArgumentParser):
    config, _ = parser.parse_known_args()

    if config.cpu is True:
        config.gpus = None
    if config.fast_debug is True:
        config.batch_size = 4
        config.epochs = 4
        config.no_wandb = True
    config.results_dir = config.results_dir + config.training_name

    return config
