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

from CDARL.utils import ExpDataset, reparameterize, RandomTransform
from weak_vae import ADAGVAE, compute_loss

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/home/mila/l/lea.cote-turcotte/LUSR/data/carracing_data', type=str, help='path to the data')
parser.add_argument('--data-tag', default='car', type=str, help='files with data_tag in name under data directory will be considered as collected states')
parser.add_argument('--num-splitted', default=10, type=int, help='number of files that the states from one domain are splitted into')
parser.add_argument('--batch-size', default=10, type=int)
parser.add_argument('--num-epochs', default=2, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--learning-rate', default=0.0001, type=float)
parser.add_argument('--beta', default=1, type=int)
parser.add_argument('--save-freq', default=1000, type=int)
parser.add_argument('--bloss-coef', default=1, type=int)
parser.add_argument('--latent-size', default=32, type=int)
parser.add_argument('--flatten-size', default=1024, type=int)
parser.add_argument('--carla-model', default=False, action='store_true', help='CARLA or Carracing')
args = parser.parse_args()

Model = ADAGVAE

def updateloader(loader, dataset):
    dataset.loadnext()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return loader

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
                # forward circle
                # Try 
                imgs = imgs.permute(1,0,2,3,4).to(device, non_blocking=True)
                imgs = imgs.reshape(-1, *imgs.shape[2:])
                imgs_repeat = imgs.repeat(2, 1, 1, 1)
                imgs = RandomTransform(imgs_repeat).apply_transformations(nb_class=2, value=None)
                optimizer.zero_grad()

                feature_1 = imgs[0]
                feature_2 = imgs[1]

                loss = compute_loss(model, feature_1, feature_2, beta=args.beta)

                loss.backward()
                optimizer.step()

                # write log
                writer.add_scalar('loss', loss.item(), batch_count)

                # save image to check and save model 
                if i_batch % args.save_freq == 0:
                    print("%d Epochs, %d Splitted Data, %d Batches is Done." % (i_epoch, i_split, i_batch))
                    imgs1 = imgs[0][:10]
                    imgs2 = imgs[1][:10]
                    with torch.no_grad():
                        mu, logsigma = model.encoder(imgs1)
                        mu2, logsigma2 = model.encoder(imgs2)
                        mu_mean = 0.5 * mu + 0.5 * mu2
                        recon_mean1 = model.decoder(mu_mean)
                        recon_1 = model.decoder(mu)
                        recon_2 = model.decoder(mu2)

                    saved_imgs = torch.cat([imgs1, imgs2, recon_mean1, recon_1, recon_2], dim=0)
                    save_image(saved_imgs, "/home/mila/l/lea.cote-turcotte/LUSR/ADAGVAE/checkimages/%d_%d_%d.png" % (i_epoch, i_split,i_batch), nrow=10)

                    torch.save(model.state_dict(), "/home/mila/l/lea.cote-turcotte/LUSR/ADAGVAE/checkpoints/model_32.pt")
                    torch.save(model.encoder.state_dict(), "/home/mila/l/lea.cote-turcotte/LUSR/ADAGVAE/checkpoints/encoder_adagvae_32.pt")

            # load next splitted data
            updateloader(loader, dataset)
    writer.close()

if __name__ == '__main__':
    main()
