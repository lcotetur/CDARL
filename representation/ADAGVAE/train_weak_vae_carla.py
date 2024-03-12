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
parser.add_argument('--beta', default=1, type=int)
parser.add_argument('--save-freq', default=19000, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--bloss-coef', default=1, type=int)
parser.add_argument('--latent-size', default=10, type=int)
parser.add_argument('--flatten-size', default=9216, type=int)
parser.add_argument('--random-augmentations', default=False, type=bool)
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
                
                # Content
                imgs_content = imgs.permute(1,0,2,3,4).repeat(2, 1, 1, 1, 1)
                m = int(imgs_content.shape[0]/2)
                imgs_content1 = imgs_content[:m, :, :, :, :]
                imgs_content2 = imgs_content[:m, :, :, :, :]
                imgs_content1 = imgs_content1[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]].view(imgs_content1.size())
                imgs_content2 = imgs_content2[[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]].view(imgs_content2.size())

                feature_1 = imgs_content1.reshape(-1, *imgs_content1.shape[2:])
                feature_2 = imgs_content2.reshape(-1, *imgs_content2.shape[2:])
                '''
                # Style
                imgs_style = imgs.repeat(2, 1, 1, 1, 1)
                n = int(imgs_style.shape[0]/2)
                imgs_style1 = imgs_style[:n, :, :, :, :]
                imgs_style2 = imgs_style[n:, :, :, :, :]
                idx = torch.randperm(3)
                imgs_style1 = imgs_style1[idx].view(imgs_style1.size())
                imgs_style1= imgs_style1.reshape(-1, *imgs_style1.shape[2:])
                imgs_style2= imgs_style2.reshape(-1, *imgs_style2.shape[2:])

                feature_1 = torch.cat((imgs_style1, imgs_content1), dim=0)
                feature_2 = torch.cat((imgs_style2, imgs_content2), dim=0)
                '''
                save_image(torch.cat((feature_1, feature_2), dim=0), os.path.join(log_dir,'features.png'), nrow=10)

                optimizer.zero_grad()
                loss = compute_loss(model, feature_1, feature_2, beta=args.beta)

                loss.backward()
                optimizer.step()

                # save image to check and save model 
                if i_batch % args.save_freq == 0:
                    print("%d Epochs, %d Splitted Data, %d Batches is Done." % (i_epoch, i_split, i_batch))
                    imgs1 = feature_1[:10]
                    imgs2 = feature_2[:10]
                    with torch.no_grad():
                        mu, logsigma = model.encoder(imgs1)
                        mu2, logsigma2 = model.encoder(imgs2)
                        recon_1 = model.decoder(mu)
                        recon_2 = model.decoder(mu2)

                    saved_imgs = torch.cat([imgs1, imgs2, recon_1, recon_2], dim=0)
                    save_image(saved_imgs, os.path.join(log_dir,'%d_%d_%d.png' % (i_epoch, i_split, i_batch)), nrow=10)

                    torch.save(model.state_dict(), os.path.join(log_dir, "model.pt"))
                    torch.save(model.encoder.state_dict(), os.path.join(log_dir, "encoder_adagvae.pt"))

            # load next splitted data
            updateloader(args, loader, dataset)

if __name__ == '__main__':
    main()
