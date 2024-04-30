#!/bin/bash
#SBATCH --gpus-per-task=rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=30G
#SBATCH --time=20:00:00

echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# CARLA CONDA ENV
salloc --gres=gpu:1 -c 4 --mem=32G -t 30:00:00 --partition=unkillable --constraint=turing

module load anaconda/3 

conda activate carla

# canvas display
export PYTHONPATH="${PYTHONPATH}:`pwd`"
Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log & export DISPLAY=:1

# connect to new terminal to start carla server 
ssh <node> # or open new terminal
export CARLA_ROOT=/home/mila/l/lea.cote-turcotte/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.8-py3.5-linux-x86_64.egg:${CARLA_ROOT}/PythonAPI/carla/agents:${CARLA_ROOT}/PythonAPI/carla

cd carla
# start carla server with 
DISPLAY= ./CarlaUE4.sh -opengl -carla-port=5000
# OR
./CarlaUE4.sh -windowed -carla-port=5000

# Exemple of run with an ilcm representation
python CDARL/train_agents_carla.py --weather 2 --seed 1 --repr ilcm <path to reduce dim encoder ilcm> --ilcm-path <path to ilcm model> --latent-size 10 --reduce_dim_latent_size 16 --port 5000

python CDARL/evaluate/evaluate_carla.py --weather 0 --algo ilcm --model-path <path to policy> --encoder-path <path to reduce dim encoder ilcm> --ilcm-path <path to ilcm model> --latent-size 10 --reduce_dim_latent_size 16 --port 5000 
python CDARL/evaluate/evaluate_carla.py --weather 1 --algo ilcm --model-path <path to policy> --encoder-path <path to reduce dim encoder ilcm> --ilcm-path <path to ilcm model> --latent-size 10 --reduce_dim_latent_size 16 --port 5000  
python CDARL/evaluate/evaluate_carla.py --weather 2 --algo ilcm --model-path <path to policy> --encoder-path <path to reduce dim encoder ilcm> --ilcm-path <path to ilcm model> --latent-size 10 --reduce_dim_latent_size 16 --port 5000 
python CDARL/evaluate/evaluate_carla.py --weather 3 --algo ilcm --model-path <path to policy> --encoder-path <path to reduce dim encoder ilcm> --ilcm-path <path to ilcm model> --latent-size 10 --reduce_dim_latent_size 16 --port 5000 