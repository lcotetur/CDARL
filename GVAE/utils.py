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

    def apply_transformations(self, nb_class=5):
        m = int(self.imgs.shape[0]/nb_class)

        transforms = []
        for i in range(nb_class+1):
            if i == 0:
                transforms.append(self.random_crop(self.imgs[i*m:(i+1)*m, :, :, :]))
            if i == nb_class:
                return torch.stack(transforms)
            elif(i != 0 and i != nb_class):
                transforms.append(self.random_jitter(self.imgs[i*m:(i+1)*m, :, :, :]))

    def domain_transformation(self, value=0.1):
        if value == None:
            return self.imgs
        else:
            return self.jitter(self.imgs, value)

    def jitter(self, imgs, value):
        imgs = transforms.ColorJitter(hue=(value))(imgs)
        return imgs

    def random_jitter(self, imgs):
        imgs = transforms.ColorJitter(hue=(-0.5,0.5))(imgs)
        return imgs

    def random_blur(self, imgs):
        imgs = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(imgs)
        return imgs

    def random_crop(self, imgs, output_size=64):
        imgs = transforms.RandomCrop((output_size, output_size), padding=10)(imgs)
        return imgs

    def change_color_grass(self, img):
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

    def random_color_grass(self, images):
        imgs = []
        for img in images:
            p = random.uniform(0, 1)
            if p > 0.75:
                imgs.append(self.change_color_grass(img))
            elif 0.5 < p < 0.75:
                imgs.append(self.change_color_grass(img))
            elif 0.25 < p < 0.5:
                imgs.append(self.change_color_grass(img))
            else:
                imgs.append(img)
        return torch.stack(imgs)