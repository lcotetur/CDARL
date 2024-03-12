import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
#from skimage.util.shape import view_as_windows
from torchvision import transforms
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import random
import datetime
from datetime import date
import cv2
import kornia
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

    def change_color_grass(self, img, value):
        obs_randomized = np.copy(img.cpu())
        obs_rand = np.transpose(obs_randomized, (2,1,0))

        #color_range = [0.1, 0.6]
        #idx = np.random.randint(3)
        #print(idx)
        #this_episode_color = np.random.randint(int(255*color_range[0]), int(255*color_range[1]), 3) / 255
        this_episode_color = value / 255
        print(this_episode_color)
        grass_color = this_episode_color[1] + 20/255

        obs_rand[np.where((np.isclose(obs_rand[:,:,0], 0.4)) & np.isclose(obs_rand[:,:,1], 0.8) & np.isclose(obs_rand[:,:,2], 0.4))] = this_episode_color
        obs_rand[np.where((np.isclose(obs_rand[:,:,0], 0.4)) & np.isclose(obs_rand[:,:,1], 230/255) & np.isclose(obs_rand[:,:,2], 0.4))] = grass_color

        obs_rand = torch.tensor(obs_rand, dtype=torch.float).to('cuda')
        print(obs_rand.shape)
        return obs_rand.permute(2, 1, 0)

    def color_grass(self, values=[np.array([0.1, 0.2, 0.8])*255, np.array([0.1, 0.2, 0.8])*255, np.array([0.1, 0.2, 0.8])*255, np.array([0.1, 0.2, 0.8])*255, np.array([102, 204, 102]), np.array([102, 204, 102])]):
        imgs = []
        for index, img in enumerate(self.imgs):
            imgs.append(self.change_color_grass(img, value=values[0]))
        return torch.stack(imgs)

    def random_color_grass_random(self, images):
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


def prefill_memory(obses, capacity, obs_shape):
    """Reserves memory for replay buffer"""
    c, h, w = obs_shape
    for _ in range(capacity):
        frame = np.ones((3, h, w), dtype=np.uint8)
        obses.append(frame)
    return obses

def random_shift(imgs, pad=4):
    """Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
    _,_,h,w = imgs.shape
    imgs = F.pad(imgs, (pad, pad, pad, pad), mode='replicate')
    return kornia.augmentation.RandomCrop((h, w))(imgs)


def random_crop(x, size=84, w1=None, h1=None, return_w1_h1=False):
    """Vectorized CUDA implementation of random crop, imgs: (B,C,H,W), size: output size"""
    assert (w1 is None and h1 is None) or (w1 is not None and h1 is not None), \
        'must either specify both w1 and h1 or neither of them'
    assert isinstance(x, torch.Tensor) and x.is_cuda, \
        'input must be CUDA tensor'
    
    n = x.shape[0]
    img_size = x.shape[-1]
    crop_max = img_size - size

    if crop_max <= 0:
        if return_w1_h1:
            return x, None, None
        return x

    x = x.permute(0, 2, 3, 1)

    if w1 is None:
        w1 = torch.LongTensor(n).random_(0, crop_max)
        h1 = torch.LongTensor(n).random_(0, crop_max)

    windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0,:,:, 0]
    cropped = windows[torch.arange(n), w1, h1]

    if return_w1_h1:
        return cropped, w1, h1

    return cropped

def view_as_windows_cuda(x, window_shape):
    """PyTorch CUDA-enabled implementation of view_as_windows"""
    assert isinstance(window_shape, tuple) and len(window_shape) == len(x.shape), \
        'window_shape must be a tuple with same number of dimensions as x'
    
    slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
    win_indices_shape = [
        x.size(0),
        x.size(1)-int(window_shape[1]),
        x.size(2)-int(window_shape[2]),
        x.size(3)    
    ]

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(x[slices].stride()) + list(x.stride()))

    return x.as_strided(new_shape, strides)

def random_conv(x):
    """Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
    n, c, h, w = x.shape
    for i in range(n):
        weights = torch.randn(3, 3, 3, 3).to(x.device)
        temp_x = x[i:i+1].reshape(-1, 3, h, w)/255.
        temp_x = F.pad(temp_x, pad=[1]*4, mode='replicate')
        out = torch.sigmoid(F.conv2d(temp_x, weights))*255.
        total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
    return total_out.reshape(n, c, h, w)

def transform(obs, nb_class=5, img_stack=3, output_size=96):
    num_frame = obs[0]/3
    frame_batch = obs.reshape(num_frame, img_stack, output_size, output_size)
    for n in range(img_stack):
        frame = RandomTransform(obs[:, 3*n:3*(n+1), :, :]/255).apply_transformations(nb_class, value=0.3, output_size=output_size)
    transformed_obs = transformed_obs.reshape(-1, *transformed_obs.shape[2:])

    return transformed_obs

def transform_carracing(obs, nb_class=5, img_stack=4):
    transform_module = nn.Sequential(ColorJitterLayer(brightness=0.4, 
                                                  contrast=0.4,
                                                  saturation=0.4, 
                                                  hue=0.5, 
                                                  p=1.0, 
                                                  batch_size=128))
    device = torch.device('cuda')
    in_stacked_x = torch.from_numpy(obs).to(device)
    #in_stacked_x= in_stacked_x /255
    in_stacked_x = in_stacked_x.reshape(-1,3,84,84)

    randconv_x = transform_module(in_stacked_x)
    randconv_x = randconv_x.reshape(-1, 9, 84,84)
    return ranconv_x

class ReplayBuffer(object):
    """Buffer to store environment transitions"""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, prefill=True):
        self.capacity = capacity
        self.batch_size = batch_size

        self._obses = []
        if prefill:
            self._obses = prefill_memory(self._obses, capacity, obs_shape)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        obses = (obs, next_obs)
        if self.idx >= len(self._obses):
            self._obses.append(obses)
        else:
            self._obses[self.idx] = (obses)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def _get_idxs(self, n=None):
        if n is None:
            n = self.batch_size
        return np.random.randint(
            0, self.capacity if self.full else self.idx, size=n
        )

    def _encode_obses(self, idxs):
        obses, next_obses = [], []
        for i in idxs:
            obs, next_obs = self._obses[i]
            obses.append(np.array(obs, copy=False))
            next_obses.append(np.array(next_obs, copy=False))
        return np.array(obses), np.array(next_obses)

    def sample_soda(self, n=None):
        idxs = self._get_idxs(n)
        obs, _ = self._encode_obses(idxs)
        return torch.as_tensor(obs).cuda().float()

    def __sample__(self, n=None):
        idxs = self._get_idxs(n)

        obs, next_obs = self._encode_obses(idxs)
        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

        return obs, actions, rewards, next_obs, not_dones

    def sample_curl(self, n=None):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        pos = random_crop(obs.clone())
        obs = random_crop(obs)
        next_obs = random_crop(next_obs)

        return obs, actions, rewards, next_obs, not_dones, pos

    def sample_drq(self, n=None, pad=4):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        obs = random_shift(obs, pad)
        next_obs = random_shift(next_obs, pad)

        return obs, actions, rewards, next_obs, not_dones

    def sample_svea(self, n=None, pad=4):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        obs = random_shift(obs, pad)

        return obs, actions, rewards, next_obs, not_dones

    def sample_sac(self, n=None):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        #obs = transform(obs, 4)
        #next_obs = transform(next_obs, 4)

        return obs, actions, rewards, next_obs, not_dones

    def sample(self, n=None):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        obs = random_crop(obs)
        next_obs = random_crop(next_obs)

        return obs, actions, rewards, next_obs, not_dones