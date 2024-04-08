from utils import Results
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn import manifold
import seaborn as sns
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

def generate_plot_multi_seed_carracing(titles, logdirs, save_path, N=1000):
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
    ax.set_ylabel('Cumulative Reward')
    fig.savefig(os.path.join(save_path, 'Learning Curves With Error PPO Agents on Carracing.png'))

def generate_adaptation_performance(models, data, save_path, title):
    fig, ax = plt.subplots()
    columns = ["Number of Seen Domains", "Average performance"]
    
    # Define a list of colors to be used for the models
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    for idx, model in enumerate(models):
        # Ensure the model uses a consistent color from the defined list
        model_color = colors[idx % len(colors)]  # Cycle through colors if there are more models than colors
        
        yfuncs = []
        for result in data[model]:
            yfuncs.append(result)

        yfuncs = np.array(yfuncs)
        ymean = yfuncs.mean(axis=0)
        ymin = yfuncs.min(axis=0)
        ymax = yfuncs.max(axis=0)
        yerror = np.stack((ymean-ymin, ymax-ymean))
        x = ["10", "5", "2"]

        ax.fill_between(x, ymin, ymax, alpha=0.2, color=model_color)
        ax.scatter(x, ymean, color=model_color)  # Set the scatter points to use the same color
        ax.plot(x, ymean, label=f'{model}', color=model_color)  # Set the plot to use the same color
    
    ax.set_xticks(["10", "5", "2"])  # Set the ticks to show only 10, 5, and 2
    ax.set_xticklabels(["10", "5", "2"]) 

    ax.legend()
    ax.set_xlabel('Number of Domains Seen in Training')
    ax.set_ylabel('Average Performance')
    fig.savefig(os.path.join(save_path, title.replace(" ", "_") + "_lines.png"))
    plt.close(fig)

def generate_adaptation_violin_plot(data, save_path, title):
    # Assume that domain counts can vary and are not fixed at three
    domain_labels = ['10 Domains', '5 Domains', '2 Domains']
    model_names = list(data.keys())
    num_domains = len(domain_labels)

    # Determine global min and max for y-axis
    all_values = [value for model_data in data.values() for trials in model_data for value in trials]
    global_min = min(all_values) - 10
    global_max = max(all_values) + 10
    y_limit = (global_min, global_max)

    fig, axs = plt.subplots(1, num_models, figsize=(5 * num_models, 6), constrained_layout=True)
    
    if num_models == 1:
        axs = [axs]  # Ensure axs is iterable for a single model

    for idx, model in enumerate(model_names):
        # Collecting all performance data for the model
        model_data = data[model]
        num_domains = len(model_data[0])  # The number of domains is taken from the length of the first data entry

        # Check that all lists within model data are the same length
        assert all(len(lst) == num_domains for lst in model_data)

        # Transposing the data to match the violinplot requirements
        model_data = np.array(model_data)  # Transpose the data

        # Creating violin plots for each domain count
        parts = axs[idx].violinplot(model_data, positions=np.arange(1, num_domains+1), showmeans=True)
        
        # Customizing the appearance of the violin plot
        for pc in parts['bodies']:
            pc.set_facecolor('skyblue')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        
        # Setting model name as title and domain labels on x-axis
        domain_labels = [f'{i} Domains' for i in x_axis_name]
        axs[idx].set_xticks(np.arange(1, num_domains + 1))
        axs[idx].set_xticklabels(domain_labels)
        axs[idx].set_title(model)
    
    # Save the figure to the given path
    fig.savefig(os.path.join(save_path, title.replace(" ", "_") + ".png"))
    plt.close(fig) 

def generate_adaptation_violin_plot2(data, save_path, title):
    # Assume that domain counts can vary and are not fixed at three
    domain_labels = ['10 Domains', '5 Domains', '2 Domains']
    model_names = list(data.keys())
    num_domains = len(domain_labels)

    # Determine global min and max for y-axis
    all_values = [value for model_data in data.values() for trials in model_data for value in trials]
    global_min = min(all_values) - 10
    global_max = max(all_values) + 10
    y_limit = (global_min, global_max)

    fig, axs = plt.subplots(1, num_domains, figsize=(num_domains * 4, 4), constrained_layout=True)

    if num_domains == 1:
        axs = [axs]

    # Using darker colors inspired by Seaborn palettes
    darker_colors = ['royalblue', 'olivedrab', 'firebrick', 'darkgoldenrod', 'mediumpurple']

    for domain_idx in range(num_domains):
        domain_data = []
        for model in model_names:
            model_data_for_domain = [trial[domain_idx] for trial in data[model]]
            domain_data.append(model_data_for_domain)

        parts = axs[domain_idx].violinplot(domain_data, positions=np.arange(1, len(model_names) + 1), showmeans=False)
        
        # Assigning darker colors to each model's violin and making center lines dark
        for pc, color in zip(parts['bodies'], darker_colors):
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.9)

        # Adding dark blue data points for each model
        for i, model_data_for_domain in enumerate(domain_data):
            x = np.random.normal(i + 1, 0.04, size=len(model_data_for_domain))
            axs[domain_idx].plot(x, model_data_for_domain, '.', color='darkblue', alpha=0.9)

        axs[domain_idx].set_xticks(np.arange(1, len(model_names) + 1))
        axs[domain_idx].set_xticklabels(model_names, rotation=45, ha="right")
        axs[domain_idx].set_title(domain_labels[domain_idx])
        
        # Set the same y-axis limits for all subplots
        axs[domain_idx].set_ylim(y_limit)
    
    fig.savefig(os.path.join(save_path, title.replace(" ", "_") + ".png"))
    plt.close(fig)

def generate_plot_multi_seed_carla(titles, logdirs, save_path, N=758):
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
    ax.set_ylabel('Cumulative Reward')
    fig.savefig(os.path.join(save_path, 'Learning Curves With Error PPO Agents on CARLA.png'))
    plt.close(fig) 


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
    titles = ['vae', 'invar cycle-vae', 'cycle-vae', 'adagvae', 'ilcm', 'pixel-ppo'] 
    
    logdir_10000 = {'ilcm': ['/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/ilcm/2024-03-27_0',
                                  '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/ilcm/2024-03-28_1',
                                  '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/ilcm/2024-03-29_2',
                                  '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/ilcm/2024-04-02_100',
                                  '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/ilcm/2024-04-07_4'],
                  'cycle-vae': ['/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/disent/2024-03-23_0',
                                '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/disent/2024-03-24_1',
                                '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/disent/2024-03-24_2',
                                '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/disent/2024-04-04_100',
                                '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/disent/2024-04-08_4'],
                  'invar cycle-vae': ['/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/cycle_vae/2024-03-21_0',
                                      '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/cycle_vae/2024-03-22_1',
                                      '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/cycle_vae/2024-03-22_2',
                                      '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/cycle_vae/2024-04-03_100',
                                      '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/cycle_vae/2024-04-06_4'],
                  'vae': ['/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/vae/2024-03-18_0',
                          '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/vae/2024-03-19_1',
                          '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/vae/2024-03-20_2',
                          '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/vae/2024-04-02_100',
                          '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/vae/2024-04-06_4'],
                  'adagvae': ['/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/adagvae/2024-03-16_0',
                              '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/adagvae/2024-03-17_1',
                              '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/adagvae/2024-03-17_2',
                              '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/adagvae/2024-04-02_100',
                              '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/adagvae/2024-04-05_4'],
                  'pixel-ppo': ['/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/None/2024-03-26_0',
                                '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/None/2024-03-26_1',
                                '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/None/2024-03-27_2',
                                '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/None/2024-04-03_100',
                                '/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/None/2024-04-07_4']}
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
    """
    generate_plot_multi_seed_carracing(titles, logdir_10000, save_path)                                                                                                                                                 [543.4613699242391, 520.5757780474868, 543.1689840218689, 556.3882554884265]

    carracing_adaptation_source_domain = {'vae': [[501.0451283310363, 521.7162773377953, 647.4038364236958], [268.78415753026826, 695.741780375035, 603.2420051037163], [487.8478455504944, 640.0164889157239, 539.8516406134219], [454.1047760486836,551.445909929363,482.5924539346062], [423.45367097124034 ,492.0081536975017,623.9109023103271]],
                                        'invar cycle-vae': [[379.64598601207837, 462.30, 416.605610186901], [351.10022306030294, 343.49689786075936, 765.0662368699636], [460.9807129670906, 347.4744338446765, 739.2417358815339], [386.39162602886216,564.0707812661356,773.7906570523803], [529.1790974289867, 432.75308409806684,]],
                                        'cycle-vae': [[356.9301000258691, 440.72, 316.91095117447304], [451.57177099969687, 457.9887311824981, 280.20231317886424], [546.0215417719102, 386.95264307849703, 307.61959103952], [608.6843242972889,613.1668242428019,332.28720247054355], [, 523.5555149152308, 398.00731346956474]],
                                        'adagvae': [[400.2201889001621, 356.3309381814246, 326.0430902993079], [491.7143404098165, 387.51428598825134, 272.3713592346355], [350.723763575106, 333.4857056198013, 468.8446911583924], [378.5129920759176, 283.37886899448506,301.55779550923097], [410.6269346815547, 386.4116548608833, 327.9523233765831]],
                                        'ilcm': [[423.3520126690767, 592.0497209874857, 546.0347338856408], [402.90410081805317, 275.487447957098, 448.8472654998997], [285.35213590832115, 545.4971823274907, 265.91795379480146], [565.6710099851065, 646.8254443558566,300.77705721065354], [543.4613699242391, 399.24828496790695,]]}
    carracing_adaptation_seen_domain = {'vae': [[459.0942110171646, 501.7652266730509, 494.2800166329325], [259.67361264548015, 620.27824003914, 522.69459863229], [522.2309316089984, 594.2609028837114, 493.3598796251097], [429.2754203074343,428.54882659538265,472.02533961275753], [446.2226868909319, 441.69647134469045,307.587547836092]],
                                        'invar cycle-vae': [[410.24668705837576, 414.23,  431.9476550587294], [402.31385627172943, 313.37262185653196, 744.6827985908405], [472.63169812687937, 334.2154513750566, 702.0490208320807], [418.72182810101464,507.74861073000227,789.6972105681178], [488.64438662962203, 449.00104871729275,]],
                                        'cycle-vae': [[322.58829665680753, 484.94, 478.9422579448911], [495.30793279427184, 533.9296593433327, 139.9269714232044], [482.17418298480777, 520.0313943272965, 357.23546250453575], [533.0603086290462,496.22586928131955,423.76772695747036], [, 520.7282128672947, 448.37526868620256]],
                                        'adagvae': [[414.9448390210974, 386.89369965154214, 319.9743723087566], [462.3298847251641, 361.203046849058, 287.10790768740026], [249.63211205272023, 361.62656488524703, 257.19742702616077], [360.42180870144307,361.46239386935355,279.14478451849436], [355.80412670856924, 393.75868430775466, 274.35691480304286]],
                                        'ilcm': [[434.50279923099333, 477.77978163780796, 583.5290990500209], [418.18808050046624, 289.43644357867066, 152.11626188417517], [352.7949025639728, 508.4392841811646, 457.65949526805724], [612.1042006514148, 607.1622037188698, 240.32009316821117], [520.5757780474868, 408.60227607285736,]],}
    carracing_adaptation_unseen_domain_1 = {'vae': [[523.3508667740612, -64.5514971321253, -139.7921551748219], [279.81704667739615, -22.845668438987218, -73.54261619075886], [535.0925329236576, -52.74907588249488, -74.69673577100157], [493.3677384847762 ,-57.32146493201915,-79.66908863626355], [439.7265097849068, -52.96970584747747,-76.37603250439807]],
                                        'invar cycle-vae': [[366.615245199624, -36.87, -38.84873676546902], [566.4880409066951, -64.63284776627499, -104.78584031862185], [318.255444915345, -79.54379900720838, -42.77277748717356], [91.48093300480315,35.40179744992729,-69.36528609762554], [655.7096153532718,-70.835231879991,]],
                                        'cycle-vae': [[410.6855677289108, -10.03, -18.95635570635627], [456.5302742511394, 36.088660358042745, -49.78357885858853], [425.8542104721491, -60.62690161194058, -27.109233211748297], [534.7283940018575, -45.651151828196525,-21.599228187787688], [, -52.0163594638996, -19.06091592434565]],
                                        'adagvae': [[392.6363924943499, 186.96411314871463, -12.470930251670636], [514.3687401336632, 272.1893119641029, -19.969105185411284], [380.19376260989344, 119.64554959384998, -8.61481345638141], [427.70730209320254,365.36506999251293,-17.488810025729833], [172.48293924500612, 320.67162520654637, -20.31591599884335]],
                                        'ilcm': [[428.73945346005667, 492.11963927957237, 168.87002692018484], [377.9950330275092, 305.98081219735394, -20.143059143536075], [324.8835894737117, 491.98630073932844, 165.563198963097163], [487.45444671013627,618.7489303381193, 116.13924664619216], [543.1689840218689,440.46652949596717,]],}
    carracing_adaptation_unseen_domain_2 = {'vae': [[465.0869520982331, 38.51236718614404, -35.5441885652813], [304.4085909940728, 164.95592350881122, -26.399190881558457], [505.6349477411844, 81.12224896724337, -26.151839612881837], [467.12022645468227,-54.367099263143515,-25.74985386833875], [427.5171866610601, 61.39887293309114,-73.00761540225766]],
                                        'invar cycle-vae': [[365.05887978018853, 23.25, -46.59069188211057], [412.96615709112115, -38.30993367621422, -97.60213229514024], [486.7593807103897, 239.07993194625266, -47.8322378454973], [406.5156794990231, 315.39095276676545,-75.15068411993026], [453.227512330264, 246.77259862031585,]],
                                        'cycle-vae': [[359.1285685662904, 76.05, -20.806445848754187], [427.9688211937453, 657.2620295399505, -42.47962822807222], [613.5189983299921, -32.74407036161279, -39.598693814623026], [664.645026488756,-34.626445429063345, 2.821959883895878], [, -26.178894594675135, 20.114514389203194]],
                                        'adagvae': [[388.1762990809516, -15.898566189935872, -26.516316673817045], [417.1492952836834, -7.781941469372158, -26.110767300398535], [302.1656174935559, -61.574529947763594, -25.37710098467481], [356.3480981534004,-19.98897851621917,-25.533959437676344], [277.59458554468915, -65.11201519885019, -25.96953069292082]],
                                        'ilcm': [[516.4276464388998, 383.8413885440494, -26.112192938464414], [350.8457181634833, 224.73980655403258, -25.31336001246156], [332.5749665156527, 344.8597721152107, -26.67989231540319], [388.2384494330324,295.8126737981538, -25.515286782483876], [556.3882554884265, 380.40535712491845,]],}

    titles = ['Adaptation_performance_carracing_models_source.png', 'Adaptation_performance_carracing_models_seen.png', 'Adaptation_performance_carracing_models_unseen.png', 'Adaptation_performance_carracing_models_unseen2.png']
    generate_adaptation_performance(['vae', 'invar cycle-vae', 'cycle-vae', 'adagvae', 'ilcm'] , carracing_adaptation_seen_domain, save_path, title=titles[1])
    
    
    datasets = {
    'Adaptation on Source Domain (a)': carracing_adaptation_source_domain,
    'Adaptation on Seen Domain (b)': carracing_adaptation_seen_domain,
    'Adaptation on Unseen Domain (c)': carracing_adaptation_unseen_domain_1,
    'Adaptation on Unseen Domain (d)': carracing_adaptation_unseen_domain_2
    }

    # Generate plots for each dataset
    for title, data in datasets.items():
        generate_adaptation_performance(['vae', 'invar cycle-vae', 'cycle-vae', 'adagvae', 'ilcm'] , data, save_path, title)
        generate_adaptation_violin_plot2(data, save_path, title)

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
    #generate_plot_multi_seed_carla(test_title, test_logdir, save_path)


    
