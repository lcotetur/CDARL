import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import trange

import numpy as np
import json
from datetime import date
import argparse
import os

from CDARL.utils import ExpDataset, reparameterize, RandomTransform, seed_everything, updateloader, transform
from vae import VAE, compute_loss, vae_loss

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/home/mila/l/lea.cote-turcotte/CDARL/data/carracing_data', type=str, help='path to the data')
parser.add_argument('--data-tag', default='car', type=str, help='files with data_tag in name under data directory will be considered as collected states')
parser.add_argument('--num-splitted', default=10, type=int, help='number of files that the states from one domain are splitted into')
parser.add_argument('--save-dir', default="/home/mila/l/lea.cote-turcotte/CDARL/VAE/runs/carracing", type=str)
parser.add_argument('--batch-size', default=10, type=int)
parser.add_argument('--num-epochs', default=2, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--learning-rate', default=0.0001, type=float)
parser.add_argument('--beta', default=10, type=int)
parser.add_argument('--save-freq', default=19000, type=int)
parser.add_argument('--bloss-coef', default=1, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--latent-size', default=32, type=int)
parser.add_argument('--flatten-size', default=1024, type=int)
parser.add_argument('--stack_frames', default=True, type=bool)
parser.add_argument('--img_stack', default=4, type=int)
parser.add_argument('--random-augmentations', default=True, type=bool)
parser.add_argument('--verbose', default=True, type=bool)
parser.add_argument('--carla-model', default=False, action='store_true', help='CARLA or Carracing')
args = parser.parse_args()

Model = VAE

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
    if args.stack_frames:
        model = Model(latent_size = args.latent_size, img_channel = 3*args.img_stack)
    else:
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
                imgs = imgs.permute(1,0,2,3,4).to(device, non_blocking=True) # from torch.Size([10, 5, 3, 64, 64]) to torch.Size([5, 10, 3, 64, 64])
                imgs = imgs.reshape(-1, *imgs.shape[2:])
                if args.stack_frames:
                    imgs = imgs.repeat(1, args.img_stack, 1, 1)
                    if args.random_augmentations:
                        imgs = RandomTransform(imgs).apply_transformations_stack(num_frames=args.img_stack, nb_class=5, value=0.3)
                else:
                    if args.random_augmentations:
                        imgs = RandomTransform(imgs).apply_transformations_stack(num_frames=1, nb_class=5, value=0.3)
                        #imgs = RandomTransform(imgs).apply_transformations(nb_class=5, value=0.3)
                        imgs = imgs.reshape(-1, *imgs.shape[2:]) # from torch.Size([5, 10, 3, 64, 64]) to torch.Size([50, 3, 64, 64])
                    save_image(imgs, os.path.join(log_dir,'see_transform.png'))

                optimizer.zero_grad()
                loss = compute_loss(imgs, model, args.beta)

                loss.backward()
                optimizer.step()

                # save image to check and save model 
                if i_batch % args.save_freq == 0:
                    print("%d Epochs, %d Splitted Data, %d Batches is Done." % (i_epoch, i_split, i_batch))
                    rand_idx = torch.randperm(imgs.shape[0])
                    imgs = imgs[rand_idx[:9]]
                    with torch.no_grad():
                        mu, logsigma = model.encoder(imgs)
                        recon_imgs1 = model.decoder(mu)

                    saved_imgs = torch.cat([imgs, recon_imgs1], dim=0)
                    if args.stack_frames:
                        save_image(saved_imgs[:, :3, :, :], os.path.join(log_dir,'stack_%d_%d_%d.png' % (i_epoch, i_split, i_batch)), nrow=10)
                        torch.save(model.state_dict(), os.path.join(log_dir, "model_vae_stack.pt"))
                        torch.save(model.encoder.state_dict(), os.path.join(log_dir, "encoder_vae_stack.pt"))
                    else:
                        save_image(saved_imgs, os.path.join(log_dir,'%d_%d_%d.png' % (i_epoch, i_split, i_batch)), nrow=10)
                        torch.save(model.state_dict(), os.path.join(log_dir, "model_vae_test.pt"))
                        torch.save(model.encoder.state_dict(), os.path.join(log_dir, "encoder_vae_test.pt"))

            # load next splitted data
            updateloader(loader, dataset)

if __name__ == '__main__':
    main()
