import torch
from torch import optim
from torch.nn import functional as F
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import trange

import numpy as np
from datetime import date
import json
import argparse
import os

from CDARL.utils import ExpDataset, reparameterize, RandomTransform, seed_everything
from weak_vae import ADAGVAE, compute_loss

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/home/mila/l/lea.cote-turcotte/CDARL/data/carracing_data', type=str, help='path to the data')
parser.add_argument('--data-tag', default='car', type=str, help='files with data_tag in name under data directory will be considered as collected states')
parser.add_argument('--num-splitted', default=10, type=int, help='number of files that the states from one domain are splitted into')
parser.add_argument('--save-dir', default="/home/mila/l/lea.cote-turcotte/CDARL/representation/ADAGVAE/logs/carracing", type=str)
parser.add_argument('--batch-size', default=10, type=int)
parser.add_argument('--num-epochs', default=2, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--learning-rate', default=0.0001, type=float)
parser.add_argument('--beta', default=1, type=int)
parser.add_argument('--save-freq', default=19000, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--loss', default='mse', type=str)
parser.add_argument('--latent-size', default=10, type=int)
parser.add_argument('--flatten-size', default=1024, type=int)
parser.add_argument('--random-augmentations', default=False, type=bool)
parser.add_argument('--carla-model', default=False, action='store_true', help='CARLA or Carracing')
parser.add_argument('--verbose', default=True, type=bool)
args = parser.parse_args()

Model = ADAGVAE

def updateloader(loader, dataset):
    dataset.loadnext()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return loader

def main():
    seed_everything(args.seed)

    # create directories
    log_dir = os.path.join(args.save_dir, str(date.today()))
    os.makedirs(log_dir, exist_ok=True)

    #save args
    with open(os.path.join(log_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

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
    batch_count = 0
    epoch_generator = trange(args.num_epochs, disable=not args.verbose)
    for i_epoch in epoch_generator:
        for i_split in range(args.num_splitted):
            for i_batch, imgs in enumerate(loader):
                batch_count += 1

                #style interventions
                """
                imgs = imgs.permute(1,0,2,3,4).to(device, non_blocking=True)
                imgs = imgs.reshape(-1, *imgs.shape[2:])
                imgs = imgs.repeat(2, 1, 1, 1)

                imgs = RandomTransform(imgs).apply_transformations(nb_class=2, value=[0, 0.1])
                feature_1 = imgs[0]
                feature_2 = imgs[1]
                """

                imgs = imgs.to(device, non_blocking=True)
                imgs = imgs.reshape(-1, *imgs.shape[2:])
                imgs = imgs.repeat(2, 1, 1, 1)

                imgs = RandomTransform(imgs).apply_transformations(nb_class=2, value=[0, 0.1])
                imgs = imgs.permute(1,0,2,3,4)
                feature_1 = imgs[:25]
                feature_1 = feature_1.reshape(-1, *imgs.shape[2:])
                feature_2 = imgs[25:]
                feature_2 = feature_2.reshape(-1, *imgs.shape[2:])

                save_image(torch.cat([feature_1, feature_2], dim=0), os.path.join(log_dir,'features.png'), nrow=10)

                optimizer.zero_grad()
                
                loss = compute_loss(model, feature_1, feature_2, beta=args.beta, loss=args.loss)

                loss.backward()
                optimizer.step()

                # save image to check and save model 
                if i_batch % args.save_freq == 0:
                    print("%d Epochs, %d Splitted Data, %d Batches is Done." % (i_epoch, i_split, i_batch))
                    imgs1 = feature_1[:10]
                    imgs2 = feature_2[:10]
                    with torch.no_grad():
                        mu, logsigma = model.encoder(imgs1)
                        recon_1 = model.decoder(mu)

                    saved_imgs = torch.cat([imgs1, recon_1], dim=0)
                    save_image(saved_imgs, os.path.join(log_dir,'%d_%d_%d.png' % (i_epoch, i_split, i_batch)), nrow=10)

                    torch.save(model.state_dict(), os.path.join(log_dir, "model.pt"))
                    torch.save(model.encoder.state_dict(), os.path.join(log_dir, "encoder_adagvae.pt"))

            # load next splitted data
            updateloader(loader, dataset)

if __name__ == '__main__':
    main()
