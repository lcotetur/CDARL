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

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/home/mila/l/lea.cote-turcotte/LUSR/data/carracing_data', type=str, help='path to the data')
parser.add_argument('--data-tag', default='car', type=str, help='files with data_tag in name under data directory will be considered as collected states')
parser.add_argument('--num-splitted', default=10, type=int, help='number of files that the states from one domain are splitted into')
parser.add_argument('--batch-size', default=10, type=int)
parser.add_argument('--num-epochs', default=2, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--learning-rate', default=0.0001, type=float)
parser.add_argument('--beta', default=10, type=int)
parser.add_argument('--save-freq', default=1000, type=int)
parser.add_argument('--bloss-coef', default=1, type=int)
parser.add_argument('--latent-size', default=32, type=int)
parser.add_argument('--flatten-size', default=1024, type=int)
parser.add_argument('--carla-model', default=False, action='store_true', help='CARLA or Carracing')
args = parser.parse_args()

Model = VAE


def updateloader(loader, dataset):
    dataset.loadnext()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return loader

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

    # create dataset and loader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ExpDataset(args.data_dir, args.data_tag, args.num_splitted, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # create model
    model = Model(latent_size = args.latent_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # do the training
    writer = SummaryWriter()
    batch_count = 0
    for i_epoch in range(args.num_epochs):
        for i_split in range(args.num_splitted):
            for i_batch, imgs in enumerate(loader):

                batch_count += 1
                imgs = imgs.permute(1,0,2,3,4).to(device, non_blocking=True) # from torch.Size([10, 5, 3, 64, 64]) to torch.Size([5, 10, 3, 64, 64])
                imgs = imgs.reshape(-1, *imgs.shape[2:])
                imgs = RandomTransform(imgs).apply_transformations(nb_class=5)
                imgs = imgs.reshape(-1, *imgs.shape[2:]) # from torch.Size([5, 10, 3, 64, 64]) to torch.Size([50, 3, 64, 64])

                optimizer.zero_grad()
                loss = compute_loss(imgs, model, args.beta)

                loss.backward()
                optimizer.step()

                # write log
                writer.add_scalar('loss', loss.item(), batch_count)

                # save image to check and save model 
                if i_batch % args.save_freq == 0:
                    print("%d Epochs, %d Splitted Data, %d Batches is Done." % (i_epoch, i_split, i_batch))
                    rand_idx = torch.randperm(imgs.shape[0])
                    imgs = imgs[rand_idx[:9]]
                    with torch.no_grad():
                        mu, logsigma = model.encoder(imgs)
                        latentcode = reparameterize(mu, logsigma)
                        recon_imgs1 = model.decoder(mu)
                        recon_imgs2 = model.decoder(latentcode)

                    saved_imgs = torch.cat([imgs, recon_imgs1, recon_imgs2], dim=0)
                    save_image(saved_imgs, "/home/mila/l/lea.cote-turcotte/LUSR/checkimages/vae_%d_%d_%d.png" % (i_epoch, i_split, i_batch), nrow=9)

                    torch.save(model.state_dict(), "/home/mila/l/lea.cote-turcotte/LUSR/checkpoints/model_vae.pt")
                    torch.save(model.encoder.state_dict(), "/home/mila/l/lea.cote-turcotte/LUSR/checkpoints/encoder_vae.pt")

            # load next splitted data
            updateloader(loader, dataset)
    writer.close()

if __name__ == '__main__':
    main()
