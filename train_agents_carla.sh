#!/bin/bash
#SBATCH --partition=long                                 # Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 2 CPUs
#SBATCH --gres=gpu:2                                     # Ask for 1 GPU
#SBATCH --mem=32G                                        # Ask for 10 GB of RAM
#SBATCH --time=120:00:00                                 # The job will run for 3 hours
#SBATCH -o /network/scratch/l/lea.cote-turcotte/slurm-%j.out  # Write the log on scratch

# 1. Load the required modules
module --quiet load anaconda/3

# 2. Load your environment
conda activate carla

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
export PYTHONPATH="${PYTHONPATH}:`pwd`"
#export XDG_RUNTIME_DIR="$SLURM_TMPDIR"
Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log & export DISPLAY=:1

export CARLA_ROOT=//home/mila/l/lea.cote-turcotte/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.8-py3.5-linux-x86_64.egg:${CARLA_ROOT}/PythonAPI/carla/agents:${CARLA_ROOT}/PythonAPI/carla

DISPLAY= carla/CarlaUE4.sh -opengl -carla-port=2000 &

### CARLA ###
# VAE
python CDARL/train_agents_carla.py --weather 2 --seed 0 --repr vae --encoder-path <path to encoder> --latent-size 16 

# INVAR CYCLE-VAE
python CDARL/train_agents_carla.py --weather 2 --seed 0 --repr cycle_vae --encoder-path <path to encoder> --latent-size 16

# DISENT CYCLE-VAE
python CDARL/train_agents_carla.py --weather 2 --seed 0 --repr disent --encoder-path <path to encoder> --latent-size 24

# ADAGVAE
python CDARL/train_agents_carla.py --weather 2 --seed 0 --repr adagvae --encoder-path <path to encoder> --latent-size 10

# ILCM
python CDARL/train_agents_carla.py --weather 2 --seed 0 --repr ilcm --encoder-path <path to encoder> --latent-size 16 --reduce_dim_latent_size 32 

# PIXEL-PPO
python CDARL/train_agents_carla.py --weather 2 --seed 1