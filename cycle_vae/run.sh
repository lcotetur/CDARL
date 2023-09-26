#!/bin/bash
#SBATCH --gpus-per-task=rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=30G
#SBATCH --time=20:00:00

echo "Date:     $(date)"
echo "Hostname: $(hostname)"

salloc --gres=gpu:1 -c 4 --mem=32G -t 30:00:00 --partition=unkillable --constraint=turing

module load anaconda/3 

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
python evaluate/evaluate_repr.py --encoder-type adagvae --model-path /home/mila/l/lea.cote-turcotte/LUSR/ADAGVAE/checkpoints/model_32.pt

# ppo training on carracing games
python ppo_agent_stack.py

# evaluate ppo agent on carracing games
python evaluate_stack.py --policy-type ppo

# agents training on carracing
python train_agents.py  --train-epochs 1000 --ray-adress 'localhost:51086' --policy-type adagvae --encoder-path /home/mila/l/lea.cote-turcotte/LUSR/checkpoints/encoder_adagvae_32.pt --model-save-path /home/mila/l/lea.cote-turcotte/LUSR/checkpoints/policy_adagvae_32.pt

# evaluate the trained agents on carracing games
python evaluate.py --policy-type adagvae --model-path /home/mila/l/lea.cote-turcotte/LUSR/checkpoints/policy_adagvae_32.pt
