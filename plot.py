from utils import Resutls
from torchvision.utils import save_image
from gym.envs.box2d import CarRacing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os

def state_image_preprocess(state_image):
    # crop image
    state_image = state_image[0:84, :, :]
    state_image = state_image.transpose((2,0,1))
    # to torch
    state_image = np.ascontiguousarray(state_image, dtype=np.float32) / 255
    state_image = torch.tensor(state_image.copy(), dtype=torch.float32)
    return state_image.unsqueeze(0).to('cpu')

def generate_plot(title, xlabel, ylabel, logdir, save_path):
    """ Generate plot according to log 
    :param logdir: path to log directories
    :param save_path: Path to save the fig
    """
    columns = [xlabel, ylabel]
    path = os.path.join(logdir, 'progress.csv')
    assert os.path.exists(os.path.join(logdir, 'progress.csv')), f"No csv file in {logdir}"
    df = pd.read_csv(path, usecols=columns)

    fig, ax = plt.subplots()
    ax.plot(df[xlabel], df[ylabel], label=title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel.replace('_', ' '))
    fig.suptitle(title)
    fig.savefig(os.path.join(save_path, f'{ylabel}.png'))


if __name__ == "__main__":
    results_avg = Resutls(title="Moving averaged episode reward", xlabel="episode", ylabel="disentangle_repr_running_score")
    results_mean = Resutls(title="episode reward mean", xlabel="episode", ylabel="disentangle_repr_episode_score")
    results_avg.generate_plot('/home/mila/l/lea.cote-turcotte/LUSR/logs/4', '/home/mila/l/lea.cote-turcotte/LUSR/figures')
    results_mean.generate_plot('/home/mila/l/lea.cote-turcotte/LUSR/logs/4', '/home/mila/l/lea.cote-turcotte/LUSR/figures')
    #results_cycle_vae = Resutls(title="Cycle Consistent VAE loss", xlabel="vae_epoch", ylabel="loss")
    #results_cycle_vae.generate_plot('/home/mila/l/lea.cote-turcotte/LUSR/logs/2', '/home/mila/l/lea.cote-turcotte/LUSR/figures')
    #generate_plot('main carracing moving averaged ep reward', "training_iteration", "episode_reward_mean", '/home/mila/l/lea.cote-turcotte/ray_results/disent_repr_myenv_2023-08-08_10-18-00wj85pq4k', '/home/mila/l/lea.cote-turcotte/LUSR/figures')