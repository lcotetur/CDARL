from utils import Resutls
from torchvision.utils import save_image
from gym.envs.box2d import CarRacing
import matplotlib.pyplot as plt
import numpy as np
import torch

def state_image_preprocess(state_image):
    # crop image
    state_image = state_image[0:84, :, :]
    state_image = state_image.transpose((2,0,1))
    # to torch
    state_image = np.ascontiguousarray(state_image, dtype=np.float32) / 255
    state_image = torch.tensor(state_image.copy(), dtype=torch.float32)
    return state_image.unsqueeze(0).to('cpu')


if __name__ == "__main__":
    results = Resutls(title="Moving averaged episode reward", xlabel="episode", ylabel="running_score")
    results.generate_plot('/home/mila/l/lea.cote-turcotte/LUSR/logs/1', '/home/mila/l/lea.cote-turcotte/LUSR/figures')
    results_cycle_vae = Resutls(title="Cycle Consistent VAE loss", xlabel="training_step", ylabel="loss")
    results_cycle_vae.generate_plot('/home/mila/l/lea.cote-turcotte/LUSR/logs/2', '/home/mila/l/lea.cote-turcotte/LUSR/figures')