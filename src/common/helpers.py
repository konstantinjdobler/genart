from argparse import Namespace
from enum import Enum
import wandb
import os
import subprocess
import random
from torch.backends import cudnn
from dotenv import load_dotenv
import pandas as pd


def get_wandb_api_key():
    try:
        api_key = os.environ.get("WANDB_API_KEY")
    except Exception as e:
        print(e)
        api_key = None
    return api_key


def try_wandb_login():
    WAND_API_KEY = get_wandb_api_key()
    if WAND_API_KEY:
        try:
            subprocess.run(["wandb", "login", WAND_API_KEY], check=True)
            return True
        except Exception as e:
            print(e)
            return False
    else:
        print("WARNING: No wandb API key found, this run will NOT be logged to wandb.")
        input("Press any key to continue...")
        return False


# WANDB_PROJECT_NAME = "genart-spike"
WANDB_TEAM_NAME = "hpi-genart"


def start_wandb_logging(cfg, model, project):
    if try_wandb_login():
        if cfg.training_name is not None:
            wandb.init(project=project, entity=WANDB_TEAM_NAME,
                       name=cfg.training_name, tags=cfg.tags)
        else:
            wandb.init(project=project, entity=WANDB_TEAM_NAME, tags=cfg.tags)
        wandb.config.update(cfg)
        wandb.watch(model)


def push_file_to_wandb(filepath):
    wandb.save(filepath, os.path.dirname(filepath))


def randomly_flip_labels(labels, p: float = 0.05):
    number_of_labels_to_flip = int(p * labels.shape[0])
    indices_to_flip = random.choices(
        [i for i in range(labels.shape[0])], k=number_of_labels_to_flip)
    # flip chosen labels
    labels[indices_to_flip] = 1 - labels[indices_to_flip]
    return labels


def filter_dataset(config: Namespace):
    df = pd.read_csv(config.annotations_file, sep="\t")
    if config.queries:
        print("Original Dataset Length:", len(df))
        for query in config.queries:
            df = df.query(query)
        print("Filtered Dataset Length:", len(df))
    new_annotations_file = os.path.join(config.results_dir, "dataset.tsv")
    df.to_csv(new_annotations_file, sep="\t", index=False)
    return new_annotations_file


def before_run(config: Namespace):
    print(f"Results will be saved to {config.results_dir}")
    os.makedirs(config.results_dir, exist_ok=True)

    # activate options to speed up training
    cudnn.enabled = True
    cudnn.benchmark = True

    if config.no_wandb:
        os.environ["WANDB_MODE"] = "dryrun"

    load_dotenv()
    
    if not config.celeba and not config.cs_chan and not config.artemis:
        config.annotations_file = filter_dataset(config)


class ExtendedEnum(Enum):

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
