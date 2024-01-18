import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import trange

import numpy as np
from datetime import date
import json
import argparse
import os

from CDARL.utils import ExpDataset, reparameterize, RandomTransform, seed_everything, updateloader
from weak_vae import CarlaADAGVAE, compute_loss

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/home/mila/l/lea.cote-turcotte/CDARL/data/carla_data', type=str, help='path to the data')
parser.add_argument('--data-tag', default='weather', type=str, help='files with data_tag in name under data directory will be considered as collected states')
parser.add_argument('--num-splitted', default=1, type=int, help='number of files that the states from one domain are splitted into')
parser.add_argument('--save-dir', default="/home/mila/l/lea.cote-turcotte/CDARL/representation/ADAGVAE/logs/carla", type=str)
parser.add_argument('--batch-size', default=10, type=int)
parser.add_argument('--num-epochs', default=50, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--learning-rate', default=0.0001, type=float)
parser.add_argument('--beta', default=10, type=int)
parser.add_argument('--save-freq', default=19000, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--bloss-coef', default=1, type=int)
parser.add_argument('--latent-size', default=32, type=int)
parser.add_argument('--flatten-size', default=9216, type=int)
parser.add_argument('--random-augmentations', default=True, type=bool)
parser.add_argument('--carla-model', default=False, action='store_true', help='CARLA or Carracing')
parser.add_argument('--verbose', default=True, type=bool)
args = parser.parse_args()

Model = CarlaADAGVAE

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

                imgs = imgs.permute(1,0,2,3,4).to(device, non_blocking=True)
                imgs = imgs.reshape(-1, *imgs.shape[2:])
                imgs_repeat = imgs.repeat(2, 1, 1, 1)
                if args.random_augmentations:
                    imgs = RandomTransform(imgs_repeat).apply_transformations(nb_class=2, value=0.3, random_crop=False)
                    feature_1 = imgs[0]
                    feature_2 = imgs[1]
                else:
                    n = imgs.shape[0]
                    print(imgs.shape[0])
                    feature_1 = imgs[:int(n/2), :, :, :]
                    print(feature_1.shape)
                    feature_2 = imgs[int(n/2):, :, :, :]
                optimizer.zero_grad()

                loss = compute_loss(model, feature_1, feature_2, beta=args.beta)

                loss.backward()
                optimizer.step()

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
                    save_image(saved_imgs, os.path.join(log_dir,'%d_%d_%d.png' % (i_epoch, i_split, i_batch)), nrow=10)

                    torch.save(model.state_dict(), os.path.join(log_dir, "model.pt"))
                    torch.save(model.encoder.state_dict(), os.path.join(log_dir, "encoder_adagvae.pt"))

            # load next splitted data
            updateloader(loader, dataset)

if __name__ == '__main__':
    main()
