import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from skimage.util.shape import view_as_windows
from torchvision import transforms
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import random
import cv2
from torchvision.utils import save_image


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


# referred from https://github.com/MishaLaskin/curl
def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0,:,:, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs

class RandomTransform():
    def __init__(self, imgs):
        self.imgs = imgs

    def apply_random_transformation(self):
        imgs = []
        #n = self.imgs.shape[0]
        for img in self.imgs:
            p = random.uniform(0, 1)
            if p > 0.75:
                imgs.append(self.random_crop(img))
            elif 0.25 < p < 0.75:
                imgs.append(self.random_blur(img))
            else:
                imgs.append(self.random_jitter(img))
        return torch.stack(imgs)

    def apply_transformations(self):
        n = self.imgs.shape[0]
        m = int(n/4)
        imgs_crop = self.random_crop(self.imgs[0:m, :, :, :])
        imgs_blur = self.random_blur(self.imgs[m:2*m, :, :, :])
        imgs_jitter = self.random_jitter(self.imgs[2*m:3*m, :, :, :])
        imgs_norm = self.random_color_grass(self.imgs[3*m:4*m, :, :, :])
        return torch.stack([imgs_crop, imgs_blur, imgs_jitter, imgs_norm])

    def random_jitter(self, imgs):
        imgs = transforms.ColorJitter(brightness=(0.5,1.5), contrast=(1), saturation=(0.5,1.5), hue=(-0.1,0.1))(imgs)
        return imgs

    def random_blur(self, imgs):
        img = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(imgs)
        return imgs

    def random_crop(self, imgs, output_size=64):
        img = transforms.RandomCrop((output_size, output_size), padding=10)(imgs)
        return imgs

    def change_color_grass(self, img, color):
        save_image(img, "/home/mila/l/lea.cote-turcotte/LUSR/figures/test0.png")
        img = np.array(img.permute(2, 1, 0).cpu(), dtype="uint8")
        save_image(torch.tensor(img, dtype=torch.float).permute(2, 1, 0).to('cuda'), "/home/mila/l/lea.cote-turcotte/LUSR/figures/test1.png")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))
        imask_green = mask_green>0
        green = np.zeros_like(img, np.uint8)
        green[imask_green] = img[imask_green]
        save_image(torch.tensor(green, dtype=torch.float).permute(2, 1, 0).to('cuda'), "/home/mila/l/lea.cote-turcotte/LUSR/figures/test5.png")
        green = np.ones_like(img, np.uint8)
        green[imask_green] = color
        save_image(torch.tensor(green, dtype=torch.float).permute(2, 1, 0).to('cuda'), "/home/mila/l/lea.cote-turcotte/LUSR/figures/test2.png")
        img = img*green
        save_image(torch.tensor(img, dtype=torch.float).permute(2, 1, 0).to('cuda'), "/home/mila/l/lea.cote-turcotte/LUSR/figures/test3.png")
        return torch.tensor(img, dtype=torch.float).permute(2, 1, 0).to('cuda')

    def change_color_grass_test(self, img, color):
        img = np.array(img.permute(2, 1, 0).cpu(), dtype="uint8")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))
        imask_green = mask_green>0
        green = np.zeros_like(img, np.uint8)
        green[imask_green] = img[imask_green]
        return torch.tensor(green, dtype=torch.float).permute(2, 1, 0).to('cuda')

    def random_color_grass(self, images):
        imgs = []
        for img in images:
            p = random.uniform(0, 1)
            if p > 0.75:
                imgs.append(self.change_color_grass(img, [0.5,1,0.5]))
                save_image(self.change_color_grass(img, [0.5,1,0.5]), "/home/mila/l/lea.cote-turcotte/LUSR/figures/test4.png")
            elif 0.5 < p < 0.75:
                imgs.append(self.change_color_grass(img, [0.3,1,1]))
                save_image(self.change_color_grass_test(img, [0.5,1,0.5]), "/home/mila/l/lea.cote-turcotte/LUSR/figures/test6.png")
            elif 0.25 < p < 0.5:
                imgs.append(self.change_color_grass(img, [0.3,0.1,1]))
            else:
                imgs.append(img)
        return torch.stack(imgs)

class Resutls():
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

    def save_logs(self, log_dir, exp_id):
        log_dir = os.path.join(log_dir, exp_id)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "args.json"), "w") as f:
            json.dump(self.logs, f, indent=2)

    def generate_plot(self, logdir, save_path):
        """ Generate plot according to log 
        :param logdir: path to log directories
        :param save_path: Path to save the fig
        """
        data = {}
        json_path = os.path.join(logdir, 'args.json')
        assert os.path.exists(os.path.join(logdir, 'args.json')), f"No json file in {logdir}"
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
            json_path = os.path.join(logdir, 'args.json')
            assert os.path.exists(os.path.join(logdir, 'args.json')), f"No json file in {logdir}"
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