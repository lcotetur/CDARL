import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import numpy as np
import argparse
import os

from utils import ExpDataset, reparameterize, RandomTransform
from model import VAE
from CDARL.data.shapes3d_data import Shape3dDataset

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/home/mila/l/lea.cote-turcotte/CDARL/data/3dshapes.h5', type=str)
parser.add_argument('--batch-size', default=50, type=int)
parser.add_argument('--num-epochs', default=10000, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--learning-rate', default=0.0001, type=float)
parser.add_argument('--beta', default=1, type=int)
parser.add_argument('--save-freq', default=100, type=int)
parser.add_argument('--latent-size', default=12, type=int)
parser.add_argument('--flatten-size', default=1024, type=int)
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
    # create direc
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    if not os.path.exists('checkimages'):
        os.makedirs("checkimages")

    # create dataset
    dataset = Shape3dDataset()
    dataset.load_dataset(file_dir=args.data_dir)

    # create model
    model = Model(latent_size = args.latent_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # do the training
    writer = SummaryWriter()
    batch_count = 0
    for i_epoch in range(args.num_epochs):
        imgs = dataset.sample_random_batch(args.batch_size)
        batch_count += args.batch_size

        imgs = dataset.process_obs(imgs, device)
        save_image(imgs, "/home/mila/l/lea.cote-turcotte/CDARL/figures/vae_3dshapes.png", nrow=10)

        optimizer.zero_grad()

        loss = compute_loss(imgs, model, args.beta)

        loss.backward()
        optimizer.step()

        # write log
        writer.add_scalar('loss', loss.item(), batch_count)

                # save image to check and save model 
        if i_epoch % args.save_freq == 0:
            print("%d Epochs, %d Batches is Done." % (i_epoch, batch_count))
            rand_idx = torch.randperm(imgs.shape[0])
            imgs = imgs[rand_idx[:10]]
            with torch.no_grad():
                mu, _ = model.encoder(imgs)
                recon_imgs1 = model.decoder(mu)

            saved_imgs = torch.cat([imgs, recon_imgs1], dim=0)
            save_image(saved_imgs, "/home/mila/l/lea.cote-turcotte/CDARL/checkimages/vae_%d_%d.png" % (i_epoch, batch_count), nrow=10)

            torch.save(model.state_dict(), "/home/mila/l/lea.cote-turcotte/CDARL/checkpoints/model_vae_3dshapes.pt")
            torch.save(model.encoder.state_dict(), "/home/mila/l/lea.cote-turcotte/CDARL/checkpoints/encoder_vae_3dshapes.pt")

    writer.close()

if __name__ == '__main__':
    main()
