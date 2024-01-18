from utils import Results
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import json
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

def generate_plot_multi_seed(titles, logdirs, save_path, N=98):
    fig, ax = plt.subplots()
    columns = ["episode", "sac_running_score"]
    for title in titles:
        yfuncs = []
        for logdir in logdirs[title]:
            path = os.path.join(logdir, 'results.json')
            assert os.path.exists(os.path.join(logdir, 'results.json')), f"No json file in {logdir}"
            f = open(path)
            df = json.load(f)
            yfuncs.append(np.array(df["sac_running_score"])[:N])

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
    fig.savefig(os.path.join(save_path, 'Learning Curves With Error SAC Agents.png'))

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
    generate_plot_multi_seed_ray(test_title_ray, test_logdir_ray, save_path)

    test_title = ['ilcm_sac', 'adagvae_sac', 'invar_sac', 'cycle_vae_sac', 'vae_sac', 'svea_sac', 'drq_sac']
    test_logdir = {'ilcm_sac': ['/home/mila/l/lea.cote-turcotte/CDARL/logs/ilcm_sac/2023-12-29_1',
                                '/home/mila/l/lea.cote-turcotte/CDARL/logs/ilcm_sac/2023-12-29_2',
                                '/home/mila/l/lea.cote-turcotte/CDARL/logs/ilcm_sac/2024-01-07_0'],
                  'adagvae_sac': ['/home/mila/l/lea.cote-turcotte/CDARL/logs/adagvae_sac/2024-01-08_0',
                                '/home/mila/l/lea.cote-turcotte/CDARL/logs/adagvae_sac/2024-01-08_1',
                                '/home/mila/l/lea.cote-turcotte/CDARL/logs/adagvae_sac/2024-01-08_2'],
                  'invar_sac': ['/home/mila/l/lea.cote-turcotte/CDARL/logs/invar_sac/2023-12-27_2',
                                '/home/mila/l/lea.cote-turcotte/CDARL/logs/invar_sac/2023-12-28_0',
                                '/home/mila/l/lea.cote-turcotte/CDARL/logs/invar_sac/2023-12-28_1'],
                  'cycle_vae_sac': ['/home/mila/l/lea.cote-turcotte/CDARL/logs/cycle_vae_sac/2024-01-10_0',
                                    '/home/mila/l/lea.cote-turcotte/CDARL/logs/cycle_vae_sac/2024-01-10_1',
                                    '/home/mila/l/lea.cote-turcotte/CDARL/logs/cycle_vae_sac/2024-01-10_2'],
                  'vae_sac': ['/home/mila/l/lea.cote-turcotte/CDARL/logs/vae_sac/2024-01-09_0',
                              '/home/mila/l/lea.cote-turcotte/CDARL/logs/vae_sac/2024-01-09_1',
                              '/home/mila/l/lea.cote-turcotte/CDARL/logs/vae_sac/2024-01-09_2'],
                  'svea_sac': ['/home/mila/l/lea.cote-turcotte/CDARL/logs/svea_sac/2024-01-09_0',
                               '/home/mila/l/lea.cote-turcotte/CDARL/logs/svea_sac/2024-01-10_1',
                               '/home/mila/l/lea.cote-turcotte/CDARL/logs/svea_sac/2024-01-10_2'],
                  'drq_sac': ['/home/mila/l/lea.cote-turcotte/CDARL/logs/drq_sac/2024-01-06_0',
                              '/home/mila/l/lea.cote-turcotte/CDARL/logs/drq_sac/2024-01-07_1',
                              '/home/mila/l/lea.cote-turcotte/CDARL/logs/drq_sac/2024-01-07_2']
                          }
    generate_plot_multi_seed(test_title, test_logdir, save_path)
