import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import random
import datetime
from datetime import date
import cv2
from torchvision.utils import save_image
from torchvision.transforms import functional as TF

def cat(x, y, axis=0):
    return torch.cat([x, y], axis=0)

def updateloader(args, loader, dataset):
    dataset.loadnext()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return loader

def reparameterize(mu, logsigma):
    std = torch.exp(0.5*logsigma)
    eps = torch.randn_like(std)
    return mu + eps*std

def obs_extract(obs):
    obs = np.transpose(obs['rgb'], (0,3,1,2))
    return torch.from_numpy(obs)

def count_step(i_update, i_env, i_step, num_envs, num_steps):
    step = i_update * (num_steps *  num_envs) + i_env * num_steps + i_step
    return step

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def create_logs(args, algo_name=False, repr=None):
    # create directories
    if algo_name==True:
        if repr is not None:
            log_dir = os.path.join(args.save_dir, args.repr ,str(date.today()) + f'_{args.seed}')
        elif repr is None:
            log_dir = os.path.join(args.save_dir, args.algo ,str(date.today()) + f'_{args.seed}')
    elif algo_name==False:
        log_dir = os.path.join(args.save_dir, str(date.today()) + f'_{args.seed}')
    os.makedirs(log_dir, exist_ok=True)

    #save args
    with open(os.path.join(log_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    return log_dir


# for representation learning
class ExpDataset(Dataset):
    def __init__(self, file_dir, game, num_splitted, transform):
        super(ExpDataset, self).__init__()
        self.file_dir = file_dir
        self.files = [f for f in os.listdir(file_dir) if game in f]
        self.num_splitted = num_splitted
        self.data = []
        self.progress = 0
        self.transform = transform

        self.loadnext()

    def __len__(self):
        assert(len(self.data) > 0)
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.stack([self.transform(d[idx]) for d in self.data])

    def loadnext(self):
        self.data = []
        for file in self.files:
            frames = np.load(os.path.join(self.file_dir, file, '%d.npz' % (self.progress)))['obs']
            self.data.append(frames)

        self.progress = (self.progress + 1) % self.num_splitted


class RandomTransform():
    def __init__(self, imgs):
        self.imgs = imgs

    def apply_random_transformations(self, nb_class=5, value=0.3, output_size=64, random_crop=True):
        m = int(self.imgs.shape[0]/nb_class)

        transforms = []
        for i in range(nb_class):
            if i == 0:
                if random_crop:
                    transforms.append(self.random_crop(self.imgs[i*m:(i+1)*m, :, :, :], output_size))
                else:
                    transforms.append(self.random_jitter(self.imgs[i*m:(i+1)*m, :, :, :], value))
            else:
                if value == None:
                    transforms.append(self.random_jitter(self.imgs[i*m:(i+1)*m, :, :, :]))
                else:
                    transforms.append(self.random_jitter(self.imgs[i*m:(i+1)*m, :, :, :], value))
            if i == nb_class-1:
                return torch.stack(transforms)

    def apply_transformations(self, nb_class=5, value=[0.1, 0.2, 0.3, 0.4, 0.5], output_size=64, random_crop=False):
        m = int(self.imgs.shape[0]/nb_class)

        transforms = []
        for i in range(nb_class):
            if i == 0:
                if random_crop:
                    transforms.append(self.random_crop(self.imgs[i*m:(i+1)*m, :, :, :], output_size))
                else:
                    transforms.append(self.jitter(self.imgs[i*m:(i+1)*m, :, :, :], value=value[0]))
            else:
                transforms.append(self.jitter(self.imgs[i*m:(i+1)*m, :, :, :], value=value[i]))
            if i == nb_class-1:
                return torch.stack(transforms)

    def apply_random_transformations_stack(self, num_frames=3, nb_class=5, value=0.3, output_size=64, random_crop=True):
        m = int(self.imgs.shape[0]/nb_class) * num_frames
        imgs = self.imgs.reshape(-1,3,output_size,output_size)
        transforms = []
        for i in range(nb_class+1):
            if i == 0:
                if random_crop:
                    transformed_img = self.random_crop(imgs[i*m:(i+1)*m, :, :, :], output_size)
                    transforms.append(transformed_img.reshape(-1, 3*num_frames, output_size, output_size))
                else:
                    transformed_img = self.jitter(imgs[i*m:(i+1)*m, :, :, :], value)
                    transforms.append(transformed_img.reshape(-1, 3*num_frames, output_size, output_size))
            if i == nb_class:
                img = torch.stack(transforms).reshape(-1, 3*num_frames, output_size, output_size)
                return torch.stack(transforms)
            elif(i != 0 and i != nb_class):
                if value == None:
                    transformed_img = self.random_jitter(imgs[i*m:(i+1)*m, :, :, :])
                    transforms.append(transformed_img.reshape(-1, 3*num_frames, output_size, output_size))
                else:
                    transformed_img = self.jitter(imgs[i*m:(i+1)*m, :, :, :], value)
                    transforms.append(transformed_img.reshape(-1, 3*num_frames, output_size, output_size))

    def apply_transformations_stack(self, num_frames=3, nb_class=5, value=[0.1, 0.2, 0.3, 0.4, 0.5], output_size=64, random_crop=True):
        m = int(self.imgs.shape[0]/nb_class) * num_frames
        imgs = self.imgs.reshape(-1,3,output_size,output_size)
        transforms = []
        for i in range(nb_class+1):
            if i == 0:
                if random_crop:
                    transformed_img = self.random_crop(imgs[i*m:(i+1)*m, :, :, :], output_size)
                    transforms.append(transformed_img.reshape(-1, 3*num_frames, output_size, output_size))
                else:
                    transformed_img = self.jitter(imgs[i*m:(i+1)*m, :, :, :], value=value[0])
                    transforms.append(transformed_img.reshape(-1, 3*num_frames, output_size, output_size))
            if i == nb_class:
                img = torch.stack(transforms).reshape(-1, 3*num_frames, output_size, output_size)
                return torch.stack(transforms)
            elif(i == 1):
                transformed_img = self.jitter(imgs[i*m:(i+1)*m, :, :, :], value=value[1])
                transforms.append(transformed_img.reshape(-1, 3*num_frames, output_size, output_size))
            elif(i == 2):
                transformed_img = self.jitter(imgs[i*m:(i+1)*m, :, :, :], value=value[2])
                transforms.append(transformed_img.reshape(-1, 3*num_frames, output_size, output_size))
            elif(i == 3):
                transformed_img = self.jitter(imgs[i*m:(i+1)*m, :, :, :], value=value[3])
                transforms.append(transformed_img.reshape(-1, 3*num_frames, output_size, output_size))
            elif(i == 4):
                transformed_img = self.jitter(imgs[i*m:(i+1)*m, :, :, :], value=value[4])
                transforms.append(transformed_img.reshape(-1, 3*num_frames, output_size, output_size))

    def domain_transformation(self, value=0.5, crop=False):
        if value == None:
            return self.imgs
        elif crop == True:
            return self.jitter(self.random_crop(self.imgs), value)
        else:
            return self.jitter(self.imgs, value)

    def domain_transformation_stack(self, value=0.5, crop=False, num_frames=3, output_size=64):
        imgs = self.imgs.reshape(-1,3,output_size,output_size)
        if value == None:
            return self.imgs
        elif crop == True:
            t_img = self.jitter(self.random_crop(imgs), value)
            return t_img.reshape(-1, 3*num_frames, 64, 64).squeeze(0)
        else:
            t_img = self.jitter(imgs, value)
            return t_img.reshape(-1, 3*num_frames, 64, 64).squeeze(0)

    def jitter(self, imgs, value):
        imgs = TF.adjust_hue(imgs, value)
        return imgs

    def random_jitter(self, imgs, value):
        imgs = transforms.ColorJitter(hue=(-value,value))(imgs)
        return imgs

    def random_blur(self, imgs):
        imgs = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(imgs)
        return imgs

    def random_crop(self, imgs, output_size=64):
        imgs = transforms.RandomCrop((output_size, output_size), padding=10)(imgs)
        return imgs

    def change_color_grass_random(self, img):
        obs_randomized = np.copy(img.cpu())
        obs_rand = np.transpose(obs_randomized, (2,1,0))

        color_range = [0.1, 0.6]
        idx = np.random.randint(3)
        this_episode_color = np.random.randint(int(255*color_range[0]), int(255*color_range[1]), 3) / 255
        grass_color = this_episode_color[idx] + 20/255

        obs_rand[np.where((np.isclose(obs_rand[:,:,0], 0.4)) & np.isclose(obs_rand[:,:,1], 0.8) & np.isclose(obs_rand[:,:,2], 0.4))] = this_episode_color
        obs_rand[np.where((np.isclose(obs_rand[:,:,0], 0.4)) & np.isclose(obs_rand[:,:,1], 230/255) & np.isclose(obs_rand[:,:,2], 0.4))] = grass_color

        obs_rand = torch.tensor(obs_rand, dtype=torch.double).to('cuda')
        return obs_rand.permute(2, 1, 0)

class Results():
    def __init__(self, title, xlabel=None, ylabel=None):
        self.logs = {}
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def create_logs(self, labels, init_values):
        for label, value in zip(labels, init_values):
            assert type(value) == list
            self.logs[label] = value

    def update_logs(self, labels, values):
        for label, value in zip(labels, values):
            self.logs[label].append(value)

    def save_logs(self, log_dir, exp_id=None):
        if exp_id == None:
            log_dir = log_dir
        else:
            log_dir = os.path.join(log_dir, exp_id)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "results.json"), "w") as f:
            json.dump(self.logs, f, indent=2)

    def generate_plot(self, logdir, save_path):
        """ Generate plot according to log 
        :param logdir: path to log directories
        :param save_path: Path to save the fig
        """
        data = {}
        json_path = os.path.join(logdir, 'results.json')
        assert os.path.exists(os.path.join(logdir, 'results.json')), f"No json file in {logdir}"
        with open(json_path, 'r') as f:
            data[self.title] = json.load(f)

        fig, ax = plt.subplots()
        ax.plot(data[self.title][self.xlabel], data[self.title][self.ylabel], label=self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel.replace('_', ' '))
        fig.suptitle(self.title)
        fig.savefig(os.path.join(save_path, f'{self.ylabel}.png'))


    def generate_plots(self, list_of_dirs, legend_names, save_path):
        """ Generate plots according to log 
        :param list_of_dirs: List of paths to log directories
        :param legend_names: List of legend names
        :param save_path: Path to save the figs
        """
        assert len(list_of_dirs) == len(legend_names), "Names and log directories must have same length"
        data = {}
        for logdir, name in zip(list_of_dirs, legend_names):
            json_path = os.path.join(logdir, 'results.json')
            assert os.path.exists(os.path.join(logdir, 'results.json')), f"No json file in {logdir}"
            with open(json_path, 'r') as f:
                data[name] = json.load(f)
    
        for ylabel, xlabel in zip(self.ylabel, self.xlabel):
            fig, ax = plt.subplots()
            for name in data:
                ax.plot(data[name][ylabel], label=name)
            ax.legend()
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel.replace('_', ' '))
            fig.savefig(os.path.join(save_path, f'{ylabel}.png'))