#!/bin/bash
#SBATCH --gpus-per-task=rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=30G
#SBATCH --time=20:00:00

echo "Date:     $(date)"
echo "Hostname: $(hostname)"

sbatch --gres=gpu:1 -c 4 --mem=48G -t 120:00:00 --partition=main --constraint=turing train_agents.sh
sbatch --gres=gpu:1 -c 4 --mem=32G -t 30:00:00 --partition=main --constraint=turing evaluate_agents.sh
sbatch --gres=gpu:1 -c 4 --mem=32G -t 30:00:00 --partition=long --constraint=turing evaluate_3dshapes.sh
sbatch --gres=gpu:1 -c 4 --mem=32G -t 120:00:00 --partition=long --constraint=turing train_carla_repr.sh
sbatch --gres=gpu:1 -c 4 -t 120:00:00 --partition=long --constraint=turing train_carracing_repr_ilcm.sh
sbatch --gres=gpu:1 -c 4 -t 120:00:00 --partition=long --constraint=turing train_carracing_repr.sh
sbatch --gres=gpu:2 -c 4 --mem=40G -t 120:00:00 --partition=long --constraint=turing train_3dshapes.sh

# LUSR CONDA ENV
salloc --gres=gpu:1 -c 4 --mem=32G -t 32:00:00 --partition=unkillable --constraint=turing

module load anaconda/3 

#module load singularity

conda activate LUSR

# canvas display
export PYTHONPATH="${PYTHONPATH}:`pwd`"
Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log & export DISPLAY=:1

# ray connection - only if running main_carracing
ray start --head --num-cpus=4 --object-store-memory=10000000000 --memory=20000000000

# train cycle-vae for carracing games
python train_cycle_vae.py --random-augmentations True

# train vae for carracing games
python train_vae.py --random-augmentations True

# train weak vae for carracing games
python ADAGVAE/train_weak_vae.py

# train evaluate encoders
python evaluate/evaluate_repr.py --encoder-type adagvae --model-path /home/mila/l/lea.cote-turcotte/CDARL/ADAGVAE/checkpoints/model_32.pt

# ppo training on carracing games
python ppo_agent_stack.py

# evaluate ppo agent on carracing games
python evaluate_stack.py --policy-type ppo

# agents training on carracing
python train_agents.py  --train-epochs 800 --ray-adress 'localhost:46468' --policy-type adagvae --seed 1 --encoder-path /home/mila/l/lea.cote-turcotte/CDARL/ADAGVAE/checkpoints/encoder_adagvae_32.pt --model-save-path /home/mila/l/lea.cote-turcotte/CDARL/checkpoints/policy_adagvae_32.pt

# evaluate the trained agents on carracing games
python evaluate/evaluate_agents.py --policy-type adagvae --model-path /home/mila/l/lea.cote-turcotte/CDARL/checkpoints/policy_adagvae_32.pt

# to connect container to compute node 
rsync -avz /network/scratch/l/lea.cote-turcotte/mujoco.sif $SLURM_TMPDIR

# open container shell to run code
singularity shell --nv -H $HOME:/home/mila/l/lea.cote-turcotte $SLURM_TMPDIR/mujoco.sif
