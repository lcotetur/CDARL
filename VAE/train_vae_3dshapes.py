import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import trange

import numpy as np
import argparse
import json
from datetime import date
import os

from CDARL.utils import ExpDataset, reparameterize, RandomTransform, seed_everything
from vae import VAE
from CDARL.data.shapes3d_data import Shape3dDataset

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/home/mila/l/lea.cote-turcotte/CDARL/data/3dshapes.h5', type=str)
parser.add_argument('--save-dir', default="/home/mila/l/lea.cote-turcotte/CDARL/VAE/runs/3dshapes", type=str)
parser.add_argument('--batch-size', default=50, type=int)
parser.add_argument('--num-epochs', default=10000, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--learning-rate', default=0.0001, type=float)
parser.add_argument('--beta', default=1, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--save-freq', default=100, type=int)
parser.add_argument('--latent-size', default=12, type=int)
parser.add_argument('--flatten-size', default=1024, type=int)
parser.add_argument('--verbose', default=True, type=bool)
args = parser.parse_args()

Model = VAE

def vae_loss(x, mu, logsigma, recon_x, beta=1):
    recon_loss = F.mse_loss(x, recon_x, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    kl_loss = kl_loss / torch.numel(x)
    return recon_loss + kl_loss * beta

def compute_loss(x, model, beta):
    mu, logsigma = model.encoder(x)
    latentcode = reparameterize(mu, logsigma)
    recon = model.decoder(latentcode)
    return vae_loss(x, mu, logsigma, recon, beta)

def main():
    seed_everything(args.seed)

    # create directories
    log_dir = os.path.join(args.save_dir, str(date.today()))
    os.makedirs(log_dir, exist_ok=True)

    #save args
    with open(os.path.join(log_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # create dataset
    dataset = Shape3dDataset()
    dataset.load_dataset(file_dir=args.data_dir)

    # create model
    model = Model(latent_size = args.latent_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # do the training
    batch_count = 0
    epoch_generator = trange(args.num_epochs, disable=not args.verbose)
    for i_epoch in epoch_generator:
        imgs = dataset.sample_random_batch(args.batch_size)
        batch_count += args.batch_size

        imgs = dataset.process_obs(imgs, device)
        save_image(imgs, "/home/mila/l/lea.cote-turcotte/CDARL/figures/vae_3dshapes.png", nrow=10)

        optimizer.zero_grad()

        loss = compute_loss(imgs, model, args.beta)

        loss.backward()
        optimizer.step()

        # save image to check and save model 
        if i_epoch % args.save_freq == 0:
            print("%d Epochs, %d Batches is Done." % (i_epoch, batch_count))
            rand_idx = torch.randperm(imgs.shape[0])
            imgs = imgs[rand_idx[:10]]
            with torch.no_grad():
                mu, _ = model.encoder(imgs)
                recon_imgs1 = model.decoder(mu)

            saved_imgs = torch.cat([imgs, recon_imgs1], dim=0)
            save_image(saved_imgs, os.path.join(log_dir, "%d_%d.png" % (i_epoch, batch_count)), nrow=10)

            torch.save(model.state_dict(), os.path.join(log_dir, "model_vae.pt"))
            torch.save(model.encoder.state_dict(), os.path.join(log_dir, "encoder_vae.pt"))


if __name__ == '__main__':
    main()
