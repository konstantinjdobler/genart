import wandb
import os
import subprocess


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


WANDB_PROJECT_NAME = "genart-spike"
WANDB_TEAM_NAME = "hpi-genart"


def start_wandb_logging(cfg, model, project=WANDB_PROJECT_NAME):
    if try_wandb_login():
        if cfg.training_name is not None:
            wandb.init(project=project, entity=WANDB_TEAM_NAME,
                       name=cfg.training_name, tags=cfg.tags)
        else:
            wandb.init(project=project, entity=WANDB_TEAM_NAME, tags=cfg.tags)
        wandb.config.update(cfg)
        wandb.watch(model)


def push_file_to_wandb(filepath):
    wandb.save(filepath)
