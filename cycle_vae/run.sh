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

# train lusr for carracing games
python train_lusr.py --data-dir data/carracing_data/  --data-tag car  --num-splitted 10 --beta 10 --class-latent-size 8  --content-latent-size 16  --flatten-size 1024   --num-epochs 2

# ppo training on carracing games
python main_carracing.py   --train-epochs 1000 --ray-adress 'localhost:52926'

# agents training on carracing
python train_agents.py   --train-epochs 1000 --ray-adress 'localhost:46773' --policy-type repr

# evaluate the trained policy on carracing games
python evaluate_carracing.py --model-path checkpoints/policy.pt  --num-episodes 100  --env CarRacing-v0