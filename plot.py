from utils import Results
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn import manifold
import json
import os

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

def generate_plot_multi_seed_ray(titles, logdirs, save_path):
    fig, ax = plt.subplots()
    columns = ["training_iteration", "episode_reward_mean"]
    for title in titles:
        yfuncs = []
        for logdir in logdirs[title]:
            path = os.path.join(logdir, 'progress.csv')
            assert os.path.exists(os.path.join(logdir, 'progress.csv')), f"No csv file in {logdir}"
            df = pd.read_csv(path, usecols=columns)
            yfuncs.append(df["episode_reward_mean"])

        yfuncs = np.array(yfuncs)
        ymean = yfuncs.mean(axis=0)
        ymin = yfuncs.min(axis=0)
        ymax = yfuncs.max(axis=0)
        yerror = np.stack((ymean-ymin, ymax-ymean))
        x = df["training_iteration"]

        ax.fill_between(x, ymin, ymax, alpha=0.2)
        #plt.errorbar(x, ymean, yerror, color='tab:blue', ecolor='tab:blue',
                    #capsize=3, linewidth=1, label='mean with error bars')
        ax.plot(x, ymean, label=f'{title}')

    ax.legend()
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Episode Reward Mean')
    fig.savefig(os.path.join(save_path, 'Learning Curves With Error Ray.png'))

def generate_plot_multi_seed_carracing(titles, logdirs, save_path, N=442):
    fig, ax = plt.subplots()
    columns = ["episode", "running_score"]
    for title in titles:
        yfuncs = []
        for logdir in logdirs[title]:
            path = os.path.join(logdir, 'results.json')
            assert os.path.exists(os.path.join(logdir, 'results.json')), f"No json file in {logdir}"
            f = open(path)
            df = json.load(f)
            yfuncs.append(np.array(df["running_score"])[:N])

        yfuncs = np.array(yfuncs)
        ymean = yfuncs.mean(axis=0)
        ymin = yfuncs.min(axis=0)
        ymax = yfuncs.max(axis=0)
        yerror = np.stack((ymean-ymin, ymax-ymean))
        x = df["episode"][:N]

        ax.fill_between(x, ymin, ymax, alpha=0.2)
        #plt.errorbar(x, ymean, yerror, color='tab:blue', ecolor='tab:blue',
                    #capsize=3, linewidth=1, label='mean with error bars')
        ax.plot(x, ymean, label=f'{title}')

    ax.legend()
    ax.set_xlabel('Training Episodes')
    ax.set_ylabel('Running Score')
    fig.savefig(os.path.join(save_path, 'Learning Curves With Error PPO Agents on Carracing.png'))

def generate_plot_multi_seed_carla(titles, logdirs, save_path, N=500):
    fig, ax = plt.subplots()
    columns = ["episode", "running average"]
    for title in titles:
        yfuncs = []
        for logdir in logdirs[title]:
            path = os.path.join(logdir, 'logs.json')
            assert os.path.exists(os.path.join(logdir, 'logs.json')), f"No json file in {logdir}"
            f = open(path)
            df = json.load(f)
            yfuncs.append(np.array(df["running average"])[:N])

        yfuncs = np.array(yfuncs)
        ymean = yfuncs.mean(axis=0)
        ymin = yfuncs.min(axis=0)
        ymax = yfuncs.max(axis=0)
        yerror = np.stack((ymean-ymin, ymax-ymean))
        x = df["episode"][:N]

        ax.fill_between(x, ymin, ymax, alpha=0.2)
        #plt.errorbar(x, ymean, yerror, color='tab:blue', ecolor='tab:blue',
                    #capsize=3, linewidth=1, label='mean with error bars')
        ax.plot(x, ymean, label=f'{title}')

    ax.legend()
    ax.set_xlabel('Training Episodes')
    ax.set_ylabel('Running Score')
    fig.savefig(os.path.join(save_path, 'Learning Curves With Error PPO Agents on CARLA.png'))

if __name__ == "__main__":
    #results_avg = Resutls(title="Moving averaged episode reward", xlabel="episode", ylabel="ppo_running_score")
    #results_mean = Resutls(title="episode reward mean", xlabel="episode", ylabel="ppo_episode_score")
    #results_avg.generate_plot('/home/mila/l/lea.cote-turcotte/CDARL/logs/8', '/home/mila/l/lea.cote-turcotte/CDARL/figures')
    #results_mean.generate_plot('/home/mila/l/lea.cote-turcotte/CDARL/logs/8', '/home/mila/l/lea.cote-turcotte/CDARL/figures')
    #results_cycle_vae = Resutls(title="Cycle Consistent VAE loss", xlabel="vae_epoch", ylabel="loss")
    #results_cycle_vae.generate_plot('/home/mila/l/lea.cote-turcotte/CDARL/logs/2', '/home/mila/l/lea.cote-turcotte/CDARL/figures')
    #generate_plot('ilcm moving averaged ep reward', "training_iteration", "episode_reward_mean", '/home/mila/l/lea.cote-turcotte/ray_results/PPO_myenv_2023-11-06_16-11-20qt8vy6z0', '/home/mila/l/lea.cote-turcotte/CDARL/figures')
  
    save_path = '/home/mila/l/lea.cote-turcotte/CDARL/results'
    test_title_ray = ['ILCM', 'AdagVAE', 'LUSR', 'cycle_vae', 'vae']
    test_logdir_ray = {'ILCM': ['/home/mila/l/lea.cote-turcotte/ray_results/PPO_myenv_2024-01-16_12-55-21jn3t_26q_ilcm_0'],
                  'AdagVAE': ['/home/mila/l/lea.cote-turcotte/ray_results/PPO_myenv_2023-09-22_08-53-39hqvly_cp_adagvae',
                              '/home/mila/l/lea.cote-turcotte/ray_results/PPO_myenv_2023-11-15_21-37-22vh1lexc6_adagvae_42'],
                  'LUSR': ['/home/mila/l/lea.cote-turcotte/ray_results/PPO_myenv_2023-09-12_09-47-18fhqmzpyx_invar',
                           '/home/mila/l/lea.cote-turcotte/ray_results/PPO_myenv_2023-11-16_08-03-05zy9zqeum_invar_42'],
                  'cycle_vae': ['/home/mila/l/lea.cote-turcotte/ray_results/PPO_myenv_2023-09-18_14-58-334mjm60yt_disent',
                                '/home/mila/l/lea.cote-turcotte/ray_results/PPO_myenv_2023-11-17_11-40-30xwixjnxi_disent_42'],
                  'vae': ['/home/mila/l/lea.cote-turcotte/ray_results/PPO_myenv_2023-09-16_21-34-15x2xrozpo_repr',
                          '/home/mila/l/lea.cote-turcotte/ray_results/PPO_myenv_2023-11-19_11-25-2538x74yge_repr_42']}
    #generate_plot_multi_seed_ray(test_title_ray, test_logdir_ray, save_path)

    # PPO carracing agents 
    test_title = ['ilcm', 'invar cycle-vae', 'cycle-vae', 'vae', 'adagvae', 'pixel-ppo'] 
    """
    test_logdir = {'ilcm': ['/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/ilcm/2024-02-02_0',
                            '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/ilcm/2024-02-03_1',
                            '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/ilcm/2024-02-03_2'],
                  'disent': ['/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/disent/2024-02-07_0',
                            '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/disent/2024-02-07_1',
                            '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/disent/2024-02-08_2'],
                  'cycle_vae': ['/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/cycle_vae/2024-02-05_0',
                                '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/cycle_vae/2024-02-06_1',
                                '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/cycle_vae/2024-02-06_2'],
                  'vae': ['/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/vae/2024-02-04_0',
                          '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/vae/2024-02-04_1',
                          '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/vae/2024-02-05_2'],
                  'adagvae': ['/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/adagvae/2024-02-08_0',
                              '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/adagvae/2024-02-09_1',
                              '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/adagvae/2024-02-02_2'],
                          }
    """
    test_logdir = {'ilcm': ['/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/ilcm/2024-03-04_0',
                            '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/ilcm/2024-03-03_1',
                            '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/ilcm/2024-03-05_2'],
                  'cycle-vae': ['/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/disent/2024-03-01_0',
                            '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/disent/2024-03-02_1',
                            '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/disent/2024-03-02_2'],
                  'invar cycle-vae': ['/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/cycle_vae/2024-02-29_0',
                                '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/cycle_vae/2024-02-29_1',
                                '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/cycle_vae/2024-03-01_2'],
                  'vae': ['/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/vae/2024-02-27_0',
                          '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/vae/2024-02-28_1',
                          '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/vae/2024-02-29_2'],
                  'adagvae': ['/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/adagvae/2024-03-10_0',
                              '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/adagvae/2024-03-11_1',
                              '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/adagvae/2024-03-11_2'],
                  'pixel-ppo': ['/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/None/2024-03-04_0',
                          '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/None/2024-03-04_1',
                          '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/None/2024-03-04_2']}
    generate_plot_multi_seed_carracing(test_title, test_logdir, save_path)


    # PPO carla agents 
    test_title = ['ilcm', 'cycle_vae', 'vae', 'adagvae']
    test_logdir = {'ilcm': ['/home/mila/l/lea.cote-turcotte/CDARL/carla_logs/ilcm/2024-02-13_0',
                            '/home/mila/l/lea.cote-turcotte/CDARL/carla_logs/ilcm/2024-02-13_1'],
                  'cycle-vae': ['/home/mila/l/lea.cote-turcotte/CDARL/carla_logs/cycle_vae/2024-01-24_0'],
                  'invar cycle-vae': ['/home/mila/l/lea.cote-turcotte/CDARL/carla_logs/cycle_vae/2024-01-24_0'],
                  'vae': ['/home/mila/l/lea.cote-turcotte/CDARL/carla_logs/vae/2024-01-24_0'],
                  'adagvae': ['/home/mila/l/lea.cote-turcotte/CDARL/carla_logs/adagvae/2024-01-25_0'],
                  'pixel-ppo': ['/home/mila/l/lea.cote-turcotte/CDARL/carla_logs/adagvae/2024-01-25_0'],
                          }
    #generate_plot_multi_seed_carla(test_title, test_logdir, save_path)


    
