import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image

import numpy as np
import argparse
import os

from CDARL.data.shapes3d_data import Shape3dDataset
from utils import reparameterize
from model import DisentangledVAE

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/home/mila/l/lea.cote-turcotte/CDARL/data/3dshapes.h5', type=str)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--nb-groups', default=2, type=int)
parser.add_argument('--num-epochs', default=10000, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--learning-rate', default=0.0001, type=float)
parser.add_argument('--beta', default=1, type=int)
parser.add_argument('--save-freq', default=100, type=int)
parser.add_argument('--bloss-coef', default=1, type=int)
parser.add_argument('--class-latent-size', default=8, type=int) # 4
parser.add_argument('--content-latent-size', default=16, type=int) # 8
parser.add_argument('--flatten-size', default=1024, type=int)
args = parser.parse_args()

Model = DisentangledVAE

def vae_loss(x, mu, logsigma, recon_x, beta=1):
    recon_loss = F.mse_loss(x, recon_x, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    kl_loss = kl_loss / torch.numel(x)
    return recon_loss + kl_loss * beta

def forward_loss(x, model, beta):
    mu, logsigma, classcode = model.encoder(x)
    contentcode = reparameterize(mu, logsigma)
    shuffled_classcode = classcode[torch.randperm(classcode.shape[0])]

    latentcode1 = torch.cat([contentcode, shuffled_classcode], dim=1)
    latentcode2 = torch.cat([contentcode, classcode], dim=1)

    recon_x1 = model.decoder(latentcode1)
    recon_x2 = model.decoder(latentcode2)

    return vae_loss(x, mu, logsigma, recon_x1, beta) + vae_loss(x, mu, logsigma, recon_x2, beta)


def backward_loss(x, model, device):
    mu, logsigma, classcode = model.encoder(x)
    shuffled_classcode = classcode[torch.randperm(classcode.shape[0])]
    randcontent = torch.randn_like(mu).to(device)

    latentcode1 = torch.cat([randcontent, classcode], dim=1)
    latentcode2 = torch.cat([randcontent, shuffled_classcode], dim=1)

    recon_imgs1 = model.decoder(latentcode1).detach()
    recon_imgs2 = model.decoder(latentcode2).detach()

    cycle_mu1, cycle_logsigma1, cycle_classcode1 = model.encoder(recon_imgs1)
    cycle_mu2, cycle_logsigma2, cycle_classcode2 = model.encoder(recon_imgs2)

    cycle_contentcode1 = reparameterize(cycle_mu1, cycle_logsigma1)
    cycle_contentcode2 = reparameterize(cycle_mu2, cycle_logsigma2)

    bloss = F.l1_loss(cycle_contentcode1, cycle_contentcode2)
    return bloss


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
    model = Model(class_latent_size = args.class_latent_size, content_latent_size = args.content_latent_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # do the training
    writer = SummaryWriter()
    batch_count = 0
    for i_epoch in range(args.num_epochs):
        groups_size = int(args.batch_size / args.nb_groups)
        imgs = dataset.create_cycle_vae_batch(args.nb_groups, groups_size, device) # torch.Size([5, 10, 3, 64, 64])
        batch_count += args.batch_size

        optimizer.zero_grad()

        floss = 0
        for i_class in range(imgs.shape[0]): # imgs.shape[0] = 5 
            # batch size 10 for each class (5 class)
            image = imgs[i_class]
            floss += forward_loss(image, model, args.beta)
        floss = floss / imgs.shape[0] # divided by the number of classes

        # backward circle
        imgs = imgs.reshape(-1, *imgs.shape[2:]) # from torch.Size([5, 10, 3, 64, 64]) to torch.Size([50, 3, 64, 64])
        save_image(imgs, "/home/mila/l/lea.cote-turcotte/CDARL/figures/cycle_vae_groups_3dshapes.png", nrow=10)
        # batch of 50 imagees with mix classes
        bloss = backward_loss(imgs, model, device)

        loss = floss + bloss * args.bloss_coef

        loss.backward()
        optimizer.step()

        # write log
        writer.add_scalar('floss', floss.item(), batch_count)
        writer.add_scalar('bloss', bloss.item(), batch_count)

        # save image to check and save model 
        if i_epoch % args.save_freq == 0:
            print("%d Epochs, %d Batches is Done." % (i_epoch, batch_count))
            rand_idx = torch.randperm(imgs.shape[0])
            imgs1 = imgs[rand_idx[:10]]
            imgs2 = imgs[rand_idx[-10:]]
            with torch.no_grad():
                mu, _, classcode1 = model.encoder(imgs1)
                _, _, classcode2 = model.encoder(imgs2)
                recon_imgs1 = model.decoder(torch.cat([mu, classcode1], dim=1))
                recon_combined = model.decoder(torch.cat([mu, classcode2], dim=1))

            saved_imgs = torch.cat([imgs1, imgs2, recon_imgs1, recon_combined], dim=0)
            save_image(saved_imgs, "/home/mila/l/lea.cote-turcotte/CDARL/checkimages/%d_%d.png" % (i_epoch, batch_count), nrow=10)

            torch.save(model.state_dict(), "/home/mila/l/lea.cote-turcotte/CDARL/checkpoints/model_cvae_3dshapes.pt")
            torch.save(model.encoder.state_dict(), "/home/mila/l/lea.cote-turcotte/CDARL/checkpoints/encoder_cvae_3dshapes.pt")

    writer.close()

if __name__ == '__main__':
    main()
