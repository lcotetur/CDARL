#!/bin/bash
#SBATCH --partition=unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=30G                                        # Ask for 10 GB of RAM
#SBATCH --time=32:00:00                                  # The job will run for 3 hours
#SBATCH -o /network/scratch/l/lea.cote-turcotte/slurm-%j.out  # Write the log on scratch

# 1. Load the required modules
module --quiet load anaconda/3

# 2. Load your environment
conda activate LUSR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
export PYTHONPATH="${PYTHONPATH}:`pwd`"
#export XDG_RUNTIME_DIR="$SLURM_TMPDIR"
Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log & export DISPLAY=:1

### CARRACING ###

#ADAGVAE
#python CDARL/train_agents_ppo.py --seed 0 --repr adagvae --encoder_path /home/mila/l/lea.cote-turcotte/CDARL/representation/ADAGVAE/logs/carracing/2024-03-09/encoder_adagvae.pt --latent_size 10 --num_episodes 10000
#python CDARL/train_agents_ppo.py --seed 1 --repr adagvae --encoder_path /home/mila/l/lea.cote-turcotte/CDARL/representation/ADAGVAE/logs/carracing/2024-03-09/encoder_adagvae.pt --latent_size 10 --num_episodes 10000
#python CDARL/train_agents_ppo.py --seed 2 --repr adagvae --encoder_path /home/mila/l/lea.cote-turcotte/CDARL/representation/ADAGVAE/logs/carracing/2024-03-09/encoder_adagvae.pt --latent_size 10 --num_episodes 10000

#VAE
#python CDARL/train_agents_ppo.py --seed 0 --repr vae --encoder_path /home/mila/l/lea.cote-turcotte/CDARL/representation/VAE/runs/carracing/2024-02-27/encoder_vae_stack.pt --latent_size 32 --num_episodes 10000
#python CDARL/train_agents_ppo.py --seed 1 --repr vae --encoder_path /home/mila/l/lea.cote-turcotte/CDARL/representation/VAE/runs/carracing/2024-02-27/encoder_vae_stack.pt --latent_size 32 --num_episodes 10000
#python CDARL/train_agents_ppo.py --seed 2 --repr vae --encoder_path /home/mila/l/lea.cote-turcotte/CDARL/representation/VAE/runs/carracing/2024-02-27/encoder_vae_stack.pt --latent_size 32 --num_episodes 10000

#CYCLEVAE
#python CDARL/train_agents_ppo.py --seed 0 --repr cycle_vae --encoder_path /home/mila/l/lea.cote-turcotte/CDARL/representation/CYCLEVAE/runs/carracing/2024-02-26/encoder_cycle_vae_stack.pt --latent_size 32 --num_episodes 10000
#python CDARL/train_agents_ppo.py --seed 1 --repr cycle_vae --encoder_path /home/mila/l/lea.cote-turcotte/CDARL/representation/CYCLEVAE/runs/carracing/2024-02-26/encoder_cycle_vae_stack.pt --latent_size 32 --num_episodes 10000
#python CDARL/train_agents_ppo.py --seed 2 --repr cycle_vae --encoder_path /home/mila/l/lea.cote-turcotte/CDARL/representation/CYCLEVAE/runs/carracing/2024-02-26/encoder_cycle_vae_stack.pt --latent_size 32 --num_episodes 10000

#DISENT
#python CDARL/train_agents_ppo.py --seed 0 --repr disent --encoder_path /home/mila/l/lea.cote-turcotte/CDARL/representation/CYCLEVAE/runs/carracing/2024-02-26/encoder_cycle_vae_stack.pt --latent_size 40 --num_episodes 10000
#python CDARL/train_agents_ppo.py --seed 1 --repr disent --encoder_path /home/mila/l/lea.cote-turcotte/CDARL/representation/CYCLEVAE/runs/carracing/2024-02-26/encoder_cycle_vae_stack.pt --latent_size 40 --num_episodes 10000
#python CDARL/train_agents_ppo.py --seed 2 --repr disent --encoder_path /home/mila/l/lea.cote-turcotte/CDARL/representation/CYCLEVAE/runs/carracing/2024-02-26/encoder_cycle_vae_stack.pt --latent_size 40 --num_episodes 10000

#ILCM
python CDARL/train_agents_ppo.py --seed 0 --repr ilcm --ilcm_path /home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carracing/2024-03-25_ilcm/model_step_120000_0.pt --encoder_path /home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carracing/2024-03-17/model_reduce_dim_step_400000.pt --latent_size 10 --reduce_dim_latent_size 16 --num_episodes 10000
python CDARL/train_agents_ppo.py --seed 1 --repr ilcm --ilcm_path /home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carracing/2024-03-25_ilcm/model_step_120000_0.pt --encoder_path /home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carracing/2024-03-17/model_reduce_dim_step_400000.pt --latent_size 10 --reduce_dim_latent_size 16 --num_episodes 10000
python CDARL/train_agents_ppo.py --seed 2 --repr ilcm --ilcm_path /home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carracing/2024-03-25_ilcm/model_step_120000_0.pt --encoder_path /home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carracing/2024-03-17/model_reduce_dim_step_400000.pt --latent_size 10 --reduce_dim_latent_size 16 --num_episodes 10000

#PPO
python CDARL/train_agents_ppo.py --seed 0 --repr None --num_episodes 10000
python CDARL/train_agents_ppo.py --seed 1 --repr None --num_episodes 10000
python CDARL/train_agents_ppo.py --seed 2 --repr None --num_episodes 10000


