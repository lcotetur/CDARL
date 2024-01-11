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

from CDARL.data.shapes3d_data import Shape3dDataset
from CDARL.utils import reparameterize, seed_everything
from model import GVAE

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/home/mila/l/lea.cote-turcotte/CDARL/data/3dshapes.h5', type=str)
parser.add_argument('--batch-size', default=50, type=int)
parser.add_argument('--nb-groups', default=5, type=int)
parser.add_argument('--num-epochs', default=10000, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--learning-rate', default=0.0001, type=float)
parser.add_argument('--beta', default=1, type=int)
parser.add_argument('--save-freq', default=100, type=int)
parser.add_argument('--bloss-coef', default=1, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--class-latent-size', default=8, type=int) # 4
parser.add_argument('--content-latent-size', default=16, type=int) # 8
parser.add_argument('--flatten-size', default=1024, type=int)
args = parser.parse_args()

Model = GVAE

def updateloader(loader, dataset):
    dataset.loadnext()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return loader

def kl_loss(mu, logsigma):
    kl_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    return kl_loss

def gvae_loss(x, model, beta=1):
    mu_s, logsigma_s, mu, logvar, recon_x = model(x)

    recon_loss = F.mse_loss(x, recon_x, reduction='mean')
    style_loss = kl_loss(mu_s, logsigma_s) / torch.numel(x)
    content_loss = kl_loss(mu, logvar)

    return recon_loss + (style_loss + content_loss) * beta


def main():
    seed_everything(args.seed)

    # create directory
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    if not os.path.exists('checkimages'):
        os.makedirs("checkimages")

    # create dataset
    dataset = Shape3dDataset()
    dataset.load_dataset(file_dir=args.data_dir)

    # create model
    model = Model(style_dim = args.class_latent_size, class_dim = args.content_latent_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # do the training
    writer = SummaryWriter()
    batch_count = 0
    for i_epoch in range(args.num_epochs):
        groups_size = int(args.batch_size / args.nb_groups)
        imgs = dataset.create_gvae_batch(args.nb_groups, groups_size, device) # torch.Size([5, 10, 3, 64, 64])
        batch_count += args.batch_size

        optimizer.zero_grad()

        loss = 0
        for i_class in range(imgs.shape[0]): # imgs.shape[0] = 5 
            # batch size 10 for each class (5 class)
            image = imgs[i_class]
            loss += gvae_loss(image, model, args.beta)
            save_image(image, "/home/mila/l/lea.cote-turcotte/CDARL/GVAE/checkimages/class_%d.png" % (i_class), nrow=10)
        loss = loss / imgs.shape[0] # divided by the number of classes

        loss.backward()
        optimizer.step()

        # write log
        writer.add_scalar('loss', loss.item(), batch_count)

        # save image to check and save model 
        if i_epoch % args.save_freq == 0:
            print("%d Epochs, %d Batches is Done." % (i_epoch, batch_count))
            imgs = imgs.reshape(-1, *imgs.shape[2:])
            rand_idx = torch.randperm(imgs.shape[0])
            imgs1 = imgs[rand_idx[:10]]
            imgs2 = imgs[rand_idx[-10:]]
            with torch.no_grad():
                mu_s, logsigma_s, mu, logvar = model.encoder(imgs1)
                mu_s2, logsigma_s2, mu2, logvar2 = model.encoder(imgs2)
                latent_code = torch.cat((mu_s, mu), dim=1)
                latent_code2 = torch.cat((mu_s2, mu), dim=1)
                recon_imgs1 = model.decoder(latent_code)
                recon_combined = model.decoder(latent_code2)

            saved_imgs = torch.cat([imgs1, imgs2, recon_imgs1, recon_combined], dim=0)
            save_image(saved_imgs, "/home/mila/l/lea.cote-turcotte/CDARL/GVAE/checkimages/z%d_%d.png" % (i_epoch, batch_count), nrow=10)

            torch.save(model.state_dict(), "/home/mila/l/lea.cote-turcotte/CDARL/GVAE/checkpoints/model_gvae_3dshapes_1ff.pt")
            torch.save(model.encoder.state_dict(), "/home/mila/l/lea.cote-turcotte/CDARL/GVAE/checkpoints/encoder_gvae_3dshapes_1ff.pt")

    writer.close()

if __name__ == '__main__':
    main()