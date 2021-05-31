# GenArt Spike

## Setup

I recommend to setup a virtualenv using Python 3.7.
Then, inside the virtualenv, run: `pip install -r requirements.txt`.

Put the dataset into a root-level `data` folder.

## Training

For example: `python src/gan/train.py -d ./data/wikiart-emotions --no-wandb --cpu -i 64 -b 8 -w 8`

To activate Weights & Biases, either put `WANDB_API_KEY=....` into a root-level `.env` file or supply it via `export WANDB_API_KEY=....` in the command line. Also, omit `--no-wandb` from the run command.

If you get an error like `W ParallelNative.cpp:206] Warning:`, run `export OMP_NUM_THREADS=1` before the training command. (Still need to investigate this issue)
### Training tips on score
- Use tmux to run trainings without needing to be connected to the server all the time
- when inside the enroot container, start trainings from _within_ the github repo (that means `cd` into the repo). This way, `wandb` can track the code version of the experiment which is very helpful
- example command to run the training from within the repo: 
```python
 python3 src/gan/train.py -b 256 -i 64 --gpus 2 -w 16 -d /mnt/wikiart_emotions/ -e 10000 -n acwgan-scale --wasserstein --condition-mode auxiliary --lr 0.0001
``` 
- Fix sporadic bug in wandb on centos linux distros:
  - `vi /usr/local/lib/python3.8/site-packages/wandb/filesync/step_checksum.py` when you are in the container
  - Paste the following code snippet under the import statements in that file
    ```python
    import distro
    if distro.id() == "centos":
      print("Falling back on slower copy method because of OS incompatibility")
      shutil._USE_CP_SENDFILE = False
    ```
### Distributed Training (Multi-GPU)
- you need to run `export PL_TORCH_DISTRIBUTED_BACKEND=gloo` before you start the training script
- specify the GPU _indices_ with the `--gpus` flag. E.g. `--gpus 1 3` will start the training on the GPUs with the indices 1 and 3
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


