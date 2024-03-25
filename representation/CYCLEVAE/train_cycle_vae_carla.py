import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import numpy as np
import argparse
from datetime import date
import json
import os

from CDARL.utils import reparameterize, seed_everything, updateloader
from CDARL.utils import ExpDataset, RandomTransform
from cycle_vae import CarlaDisentangledVAE, forward_loss, backward_loss, vae_loss

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/home/mila/l/lea.cote-turcotte/CDARL/data/carla_data', type=str, help='path to the data')
parser.add_argument('--data-tag', default='weather', type=str, help='files with data_tag in name under data directory will be considered as collected states')
parser.add_argument('--num-splitted', default=1, type=int, help='number of files that the states from one domain are splitted into')
parser.add_argument('--save-dir', default="/home/mila/l/lea.cote-turcotte/CDARL/representation/CYCLEVAE/runs/carla", type=str)
parser.add_argument('--batch-size', default=10, type=int)
parser.add_argument('--num-epochs', default=50, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--learning-rate', default=0.0001, type=float)
parser.add_argument('--beta', default=10, type=int)
parser.add_argument('--save-freq', default=19000, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--bloss-coef', default=1, type=int)
parser.add_argument('--class-latent-size', default=4, type=int)
parser.add_argument('--content-latent-size', default=16, type=int)
parser.add_argument('--flatten-size', default=9216, type=int)
parser.add_argument('--random-augmentations', default=False, type=bool)
parser.add_argument('--carla-model', default=False, action='store_true', help='CARLA or Carracing')
parser.add_argument('--verbose', default=True, type=bool)
args = parser.parse_args()

Model = CarlaDisentangledVAE

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
    model = Model(class_latent_size = args.class_latent_size, content_latent_size = args.content_latent_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # do the training
    batch_count = 0
    for i_epoch in range(args.num_epochs):
        for i_split in range(args.num_splitted):
            for i_batch, imgs in enumerate(loader):
                batch_count += 1
                # forward circle
                imgs = imgs.permute(1,0,2,3,4).to(device, non_blocking=True)
                if args.random_augmentations:
                    imgs = imgs.reshape(-1, *imgs.shape[2:])
                    imgs = RandomTransform(imgs).apply_transformations(nb_class=5, value=0.3, output_size=128)
                optimizer.zero_grad()

                floss = 0
                for i_class in range(imgs.shape[0]):
                    image = imgs[i_class]
                    floss += forward_loss(image, model, args.beta)
                floss = floss / imgs.shape[0]

                # backward circle
                imgs = imgs.reshape(-1, *imgs.shape[2:])
                bloss = backward_loss(imgs, model, device)

                (floss + bloss * args.bloss_coef).backward()
                optimizer.step()

                # save image to check and save model 
                if i_batch % args.save_freq == 0:
                    print("%d Epochs, %d Splitted Data, %d Batches is Done." % (i_epoch, i_split, i_batch))
                    rand_idx = torch.randperm(imgs.shape[0])
                    imgs1 = imgs[rand_idx[:9]]
                    imgs2 = imgs[rand_idx[-9:]]
                    with torch.no_grad():
                        mu, _, classcode1 = model.encoder(imgs1)
                        _, _, classcode2 = model.encoder(imgs2)
                        recon_imgs1 = model.decoder(torch.cat([mu, classcode1], dim=1))
                        recon_combined = model.decoder(torch.cat([mu, classcode2], dim=1))

                    saved_imgs = torch.cat([imgs1, imgs2, recon_imgs1, recon_combined], dim=0)
                    save_image(saved_imgs, os.path.join(log_dir,'%d_%d_%d.png' % (i_epoch, i_split, i_batch)), nrow=9)

                    torch.save(model.state_dict(), os.path.join(log_dir, "model_cycle_vae.pt"))
                    torch.save(model.encoder.state_dict(), os.path.join(log_dir, "encoder_cycle_vae.pt"))

            # load next splitted data
            updateloader(args, loader, dataset)


if __name__ == '__main__':
    main()