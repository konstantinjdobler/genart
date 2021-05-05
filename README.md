# GenArt Spike

## Setup

I recommend to setup a virtualenv using Python 3.7.
Then, inside the virtualenv, run: `pip install -r requirements.txt`.

Put the dataset into a root-level `data` folder.

## Training

For example: `python src/gan/train.py -d ./data/wikiart-emotions --no-wandb --cpu -i 64 -b 8 -w 8`

To activate Weights & Biases, either put `WANDB_API_KEY=....` into a root-level `.env` file or supply it via `export WANDB_API_KEY=....` in the command line. Also, omit `--no-wandb` from the run command.

If you get an error like `W ParallelNative.cpp:206] Warning:`, run `export OMP_NUM_THREADS=1` before the training command. (Still need to investigate this issue)

## Score Setup

1. `/scratch/<HPI_USERNAME>` folder auf server erstellen 
   
 ```sh
 cd /scratch
 mkdir <HPI_USERNAME>
 ```
2.  Prepare enroot configuration (https://score.hpi.uni-potsdam.de/ticket/64)
    - Environment variables need to be set before using enroot, paste the following commands into a shell script that you run everytime after you have connected to the server
    - `nano envvars.sh` and paste the following content
    ```sh
    export XDG_DATA_HOME=/scratch/<HPI_USERNAME>/enroot-data
    export XDG_CACHE_HOME=/scratch/<HPI_USERNAME>/enroot-data
    export ENROOT_SQUASH_OPTIONS='-comp lz4 -noD'
    ```
    - `source envvars.sh` to set the environment variables
3. Import docker image from dockerhub (https://github.com/NVIDIA/enroot) (IMPORTANT: you need to have set the env vars in step 2)
    ```sh
        enroot import -o genart-score.sqsh docker://konstantinjdobler/genart:score-dependencies
    ```
- TIP: run this command in your home folder
4. Start a container (https://github.com/NVIDIA/enroot/blob/master/doc/cmd/start.md). Specify `--rw` for write access to the host filesystem. Use `-m`to mount our repository and datasets into the container. 
    ```sh
    enroot start --rw -m .:/mnt genart-score.sqsh
    ```


