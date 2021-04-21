# GenArt Spike

## Setup

I recommend to setup a virtualenv using Python 3.7.
Then, inside the virtualenv, run: `pip install -r requirements.txt`.

Put the dataset into a root-level `data` folder.

## Training

For example: `python src/gan/train.py -d ./data/wikiart-emotions --no-wandb --cpu -i 64 -b 8 -w 8`

To activate Weights & Biases, either put `WANDB_API_KEY=....` into a root-level `.env` file or supply it via `export WANDB_API_KEY=....` in the command line. Also, omit `--no-wandb` from the run command.

If you get an error like `W ParallelNative.cpp:206] Warning:`, run `export OMP_NUM_THREADS=1` before the training command. (Still need to investigate this issue)
