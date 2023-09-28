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
from weak_vae import ADAGVAE, compute_loss

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/home/mila/l/lea.cote-turcotte/LUSR/data/3dshapes.h5', type=str)
parser.add_argument('--batch-size', default=20, type=int)
parser.add_argument('--num-epochs', default=10000, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--learning-rate', default=0.0001, type=float)
parser.add_argument('--beta', default=1, type=int)
parser.add_argument('--save-freq', default=1000, type=int)
parser.add_argument('--bloss-coef', default=1, type=int)
parser.add_argument('--latent-size', default=12, type=int)
parser.add_argument('--flatten-size', default=1024, type=int)
args = parser.parse_args()

Model = ADAGVAE

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
        imgs = dataset.create_weak_vae_batch(args.batch_size, device)
        '''
        imgs=[]
        for i in range(args.batch_size):
            img_batch = dataset.sample_batch(2)
            img_batch = process_obs(img_batch, device)
            sample_pair = torch.cat([img_batch[0], img_batch[1]], dim=1).unsqueeze(0) # torch.Size([1, 3, 128, 64])
            if imgs == []:
                imgs = sample_pair
            else:
                imgs = torch.cat([imgs, sample_pair], dim=0) # torch.Size([10, 3, 128, 64])

        '''
        batch_count += args.batch_size * 2

        optimizer.zero_grad()

        m = int(imgs.shape[2]/2) # 64
        feature_1 = imgs[:, :, :m, :] # torch.Size([10, 3, 64, 64])
        feature_2 = imgs[:, :, m:, :] # torch.Size([10, 3, 64, 64])
        saved_imgs = torch.cat([feature_1, feature_2], dim=0)
        save_image(saved_imgs, "/home/mila/l/lea.cote-turcotte/LUSR/ADAGVAE/checkimages/model_inputs.png", nrow=10)

        loss = compute_loss(model, feature_1, feature_2, beta=args.beta)

        loss.backward()
        optimizer.step()

        # write log
        writer.add_scalar('loss', loss.item(), batch_count)

        # save image to check and save model 
        if batch_count % args.save_freq == 0:
            print("%d Epochs, %d Batches is Done." % (i_epoch, batch_count))
            imgs1 = feature_1[:10]
            imgs2 = feature_2[:10]
            with torch.no_grad():
                mu, logsigma = model.encoder(imgs1)
                mu2, logsigma2 = model.encoder(imgs2)
                recon_1 = model.decoder(mu)
                recon_2 = model.decoder(mu2)

            saved_imgs = torch.cat([imgs1, imgs2, recon_1, recon_2], dim=0)
            save_image(saved_imgs, "/home/mila/l/lea.cote-turcotte/LUSR/ADAGVAE/checkimages/%d_%d.png" % (i_epoch, batch_count), nrow=10)

            torch.save(model.state_dict(), "/home/mila/l/lea.cote-turcotte/LUSR/ADAGVAE/checkpoints/model_3dshapes_delete.pt")
            torch.save(model.encoder.state_dict(), "/home/mila/l/lea.cote-turcotte/LUSR/ADAGVAE/checkpoints/encoder_3dshapes_delete.pt")

    writer.close()

if __name__ == '__main__':
    main()
