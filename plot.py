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

def generate_adaptation_performance(models, data, save_path, title):
    fig, ax = plt.subplots()
    columns = ["Number of Seen Domains", "Average performance"]
    for model in models:
        yfuncs = []
        for result in data[model]:
            yfuncs.append(result)

        yfuncs = np.array(yfuncs)
        ymean = yfuncs.mean(axis=0)
        ymin = yfuncs.min(axis=0)
        ymax = yfuncs.max(axis=0)
        yerror = np.stack((ymean-ymin, ymax-ymean))
        x = ["2", "5", "10"]

        ax.fill_between(x, ymin, ymax, alpha=0.2)
        #plt.errorbar(x, ymean, yerror, color='tab:blue', ecolor='tab:blue',
                    #capsize=3, linewidth=1, label='mean with error bars')
        ax.plot(x, ymean, label=f'{model}')
    
    ax.set_xticks(["2", "5", "10"])  # Set the ticks to show only 10, 5, and 2
    ax.set_xticklabels(['2', '5', '10']) 

    ax.legend()
    ax.set_xlabel('Number of Domains Seen in Training')
    ax.set_ylabel('Average Performance')
    fig.savefig(os.path.join(save_path, title.replace(" ", "_") + "_lines.png"))

def generate_adaptation_violin_plot(data, save_path, title):
    num_domains = 3  # Number of domain counts
    domain_labels = ['10 Domains', '5 Domains', '2 Domains']
    model_names = list(data.keys())
    num_models = len(model_names)

    fig, axs = plt.subplots(1, num_models, figsize=(5 * num_models, 6), constrained_layout=True)
    
    if num_models == 1:
        axs = [axs]  # Ensure axs is iterable for a single model

    for idx, model in enumerate(model_names):
        # Extracting the performance data for each domain
        model_data = np.array(data[model][0]).reshape(-1, num_domains)

        # Creating violin plots for this model for each domain count
        parts = axs[idx].violinplot(model_data, positions=np.arange(num_domains), showmeans=True)
        
        # Customizing the appearance of the violin plot
        for pc in parts['bodies']:
            pc.set_facecolor('skyblue')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        
        axs[idx].set_xticks(np.arange(num_domains))
        axs[idx].set_xticklabels(domain_labels)
        axs[idx].set_title(f'{model}')
    
    plt.suptitle(title)
    
    # Save the figure
    fig.savefig(os.path.join(save_path, title.replace(" ", "_") + ".png"))

def generate_plot_multi_seed_carla(titles, logdirs, save_path, N=500):
    fig, ax = plt.subplots()
    columns = ["done step", "running average"]
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
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Running Reward')
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
    test_title = ['vae', 'invar cycle-vae', 'cycle-vae', 'adagvae', 'ilcm', 'pixel-ppo'] 
    """
    test_logdir_10000 = {'ilcm': ['',
                            '',
                            ''],
                  'cycle-vae': ['/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/disent/2024-03-23_0',
                            '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/disent/2024-03-24_1',
                            '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/disent/2024-03-24_2'],
                  'invar cycle-vae': ['/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/cycle_vae/2024-03-21_0',
                                '//home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/cycle_vae/2024-03-22_1',
                                '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/cycle_vae/2024-03-22_2'],
                  'vae': ['/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/vae/2024-03-18_0',
                          '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/vae/2024-03-19_1',
                          '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/vae/2024-03-20_2'],
                  'adagvae': ['/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/adagvae/2024-03-16_0',
                              '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/adagvae/2024-03-17_1',
                              '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/adagvae/2024-03-17_2'],
                  'pixel-ppo': ['',
                          '',
                          '']}
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
    #generate_plot_multi_seed_carracing(test_title, test_logdir, save_path)

    carracing_adaptation_source_domain = {'vae': [[501.0451283310363, 521.7162773377953, 647.4038364236958], [268.78415753026826, 695.741780375035, ], [487.8478455504944, , ]],
                                        'invar cycle-vae': [[379.64598601207837, 462.30, 416.605610186901], [351.10022306030294, , ], [460.9807129670906, , ]],
                                        'cycle-vae': [[356.9301000258691, 440.72, 316.91095117447304], [451.57177099969687, , ], [546.0215417719102, , ]],
                                        'adagvae': [[4400.2201889001621, 356.3309381814246, 326.0430902993079], [491.7143404098165, 387.51428598825134, ], [350.723763575106, , ]],
                                        'ilcm': [[291.3721461008947, 592.0497209874857, 524.588448543386], [247.12, 1, 1], [195.6373822841058, , ], [254.3443784395661, , ]]}
    carracing_adaptation_seen_domain = {'vae': [[459.0942110171646, 501.7652266730509, 494.2800166329325], [259.67361264548015, 620.27824003914, ], [522.2309316089984, , ]],
                                        'invar cycle-vae': [[410.24668705837576, 414.23,  431.9476550587294], [402.31385627172943, , ], [472.63169812687937, , ]],
                                        'cycle-vae': [[322.58829665680753, 484.94, 478.9422579448911], [495.30793279427184, , ], [482.17418298480777, , ]],
                                        'adagvae': [[414.9448390210974, 386.89369965154214, 319.9743723087566], [462.3298847251641, 361.203046849058, ], [249.63211205272023, , ]],
                                        'ilcm': [[281.3153373742436, 477.77978163780796, 576.4351428785108], [305.1149495910714, , ], [276.5613170735966, , ]]}
    carracing_adaptation_unseen_domain_1 = {'vae': [[523.3508667740612, -64.5514971321253, -139.7921551748219], [279.81704667739615, -22.845668438987218, ], [535.0925329236576, , ]],
                                        'invar cycle-vae': [[366.615245199624, -36.87, -38.84873676546902], [566.4880409066951, , ], [318.255444915345, , ]],
                                        'cycle-vae': [[410.6855677289108, -10.03, -18.95635570635627], [456.5302742511394, , ], [425.8542104721491, , ]],
                                        'adagvae': [[392.6363924943499, 186.96411314871463, -12.470930251670636], [514.3687401336632, 272.1893119641029, ], [380.19376260989344, , ]],
                                        'ilcm': [[356.24953569365067, 492.11963927957237, 130.75246756768786], [315.2119444372653, , ], [391.519029087043, , ]]}
    carracing_adaptation_unseen_domain_2 = {'vae': [[465.0869520982331, 38.51236718614404, -35.5441885652813], [304.4085909940728, 164.95592350881122, ], [505.6349477411844, , ]],
                                        'invar cycle-vae': [[365.05887978018853, 23.25, -46.59069188211057], [412.96615709112115, , ], [486.7593807103897, , ]],
                                        'cycle-vae': [[359.1285685662904, 76.05, -20.806445848754187], [427.9688211937453, , ], [613.5189983299921, , ]],
                                        'adagvae': [[388.1762990809516, -15.898566189935872, -26.516316673817045], [417.1492952836834, -7.781941469372158, ], [302.1656174935559, , ]],
                                        'ilcm': [[278.42223773123374, 383.8413885440494, -27.420792105886786], [338.7661305252416, , ], [284.2942955432852, , ]]}
    titles = ['Adaptation_performance_carracing_models_source.png', 'Adaptation_performance_carracing_models_seen.png', 'Adaptation_performance_carracing_models_unseen.png', 'Adaptation_performance_carracing_models_unseen2.png']
    #generate_adaptation_performance(['vae', 'invar cycle-vae', 'cycle-vae', 'adagvae', 'ilcm'] , carracing_adaptation_seen_domain, save_path, title=titles[1])
    
    
    datasets = {
    'Adaptation on Source Domain (a)': carracing_adaptation_source_domain,
    'Adaptation on Seen Domain (b)': carracing_adaptation_seen_domain,
    'Adaptation on Unseen Domain (c)': carracing_adaptation_unseen_domain_1,
    'Adaptation on Unseen Domain (d)': carracing_adaptation_unseen_domain_2
    }

    # Generate plots for each dataset
    for title, data in datasets.items():
        generate_adaptation_performance(['vae', 'invar cycle-vae', 'cycle-vae', 'adagvae', 'ilcm'] , data, save_path, title)
        generate_adaptation_violin_plot(data, save_path, title)

    # PPO carla agents 
    test_title = ['vae', 'invar cycle-vae', 'cycle-vae', 'adagvae', 'ilcm', 'pixel-ppo'] 
    test_logdir = {'ilcm': ['/home/mila/l/lea.cote-turcotte/CDARL/carla_logs/ilcm/2024-03-23_1',
                            '/home/mila/l/lea.cote-turcotte/CDARL/carla_logs/ilcm/2024-03-24_2'],
                  'cycle-vae': ['/home/mila/l/lea.cote-turcotte/CDARL/carla_logs/disent/2024-03-16_1',
                                '/home/mila/l/lea.cote-turcotte/CDARL/carla_logs/disent/2024-03-17_2'],
                  'invar cycle-vae': ['/home/mila/l/lea.cote-turcotte/CDARL/carla_logs/cycle_vae/2024-03-16_2',
                                      '/home/mila/l/lea.cote-turcotte/CDARL/carla_logs/cycle_vae/2024-03-16_1'],
                  'vae': ['/home/mila/l/lea.cote-turcotte/CDARL/carla_logs/vae/2024-03-14_1',
                          '/home/mila/l/lea.cote-turcotte/CDARL/carla_logs/vae/2024-03-15_2'],
                  'adagvae': ['/home/mila/l/lea.cote-turcotte/CDARL/carla_logs/adagvae/2024-03-13_0',
                              '/home/mila/l/lea.cote-turcotte/CDARL/carla_logs/adagvae/2024-03-17_1'],
                  'pixel-ppo': ['/home/mila/l/lea.cote-turcotte/CDARL/carla_logs/ppo/2024-03-18_1',
                                '/home/mila/l/lea.cote-turcotte/CDARL/carla_logs/ppo/2024-03-18_2'],
                          }
    generate_plot_multi_seed_carla(test_title, test_logdir, save_path)


    
