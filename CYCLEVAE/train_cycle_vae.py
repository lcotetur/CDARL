import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import numpy as np
import argparse
from datetime import date
import json
import os

from CDARL.data.shapes3d_data import Shape3dDataset
from CDARL.utils import reparameterize, seed_everything
from CDARL.utils import ExpDataset, RandomTransform
from cycle_vae import DisentangledVAE

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/home/mila/l/lea.cote-turcotte/CDARL/data/carracing_data', type=str, help='path to the data')
parser.add_argument('--data-tag', default='car', type=str, help='files with data_tag in name under data directory will be considered as collected states')
parser.add_argument('--num-splitted', default=10, type=int, help='number of files that the states from one domain are splitted into')
parser.add_argument('--save-dir', default="/home/mila/l/lea.cote-turcotte/CDARL/CYCLEVAE/runs/carracing", type=str)
parser.add_argument('--batch-size', default=10, type=int)
parser.add_argument('--num-epochs', default=2, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--learning-rate', default=0.0001, type=float)
parser.add_argument('--beta', default=10, type=int)
parser.add_argument('--save-freq', default=19000, type=int)
parser.add_argument('--bloss-coef', default=1, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--class-latent-size', default=8, type=int)
parser.add_argument('--content-latent-size', default=32, type=int)
parser.add_argument('--flatten-size', default=1024, type=int)
parser.add_argument('--stack_frames', default=True, type=bool)
parser.add_argument('--img_stack', default=4, type=int)
parser.add_argument('--random-augmentations', default=True, type=bool)
parser.add_argument('--verbose', default=True, type=bool)
parser.add_argument('--carla-model', default=False, action='store_true', help='CARLA or Carracing')
args = parser.parse_args()

Model = DisentangledVAE


def updateloader(loader, dataset):
    dataset.loadnext()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return loader

def transform(obs, nb_class=5):
	frames=[]
	for n in range(4):
		print(obs[:, 3*n:3*(n+1), :, :].shape)
		frame = RandomTransform(obs[:, 3*n:3*(n+1), :, :]).apply_transformations(nb_class, value=0.3)
		frames.append(frame)
		transformed_obs = torch.cat(frames, dim=2)
	transformed_obs = transformed_obs.reshape(-1, *transformed_obs.shape[2:])
	return transformed_obs

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
        model = Model(class_latent_size = args.class_latent_size, content_latent_size = args.content_latent_size, img_channel = 3*args.img_stack)
    else:
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
                imgs = imgs.permute(1,0,2,3,4).to(device, non_blocking=True) # from torch.Size([10, 5, 3, 64, 64]) to torch.Size([5, 10, 3, 64, 64])
                if args.stack_frames:
                    imgs = imgs.repeat(1, 1, args.img_stack, 1, 1)
                    if args.random_augmentations:
                        imgs = imgs.reshape(-1, *imgs.shape[2:])
                        frames=[]
                        for n in range(4):
                            frame = RandomTransform(imgs[:, 3*n:3*(n+1), :, :]).apply_transformations(nb_class=5, value=0.3)
                            frames.append(frame)
                            transformed_imgs = torch.cat(frames, dim=2)
                        imgs = transformed_imgs
                else:
                    if args.random_augmentations:
                        imgs = imgs.reshape(-1, *imgs.shape[2:])
                        imgs = RandomTransform(imgs).apply_transformations(nb_class=5, value=0.3)
                optimizer.zero_grad()

                floss = 0
                for i_class in range(imgs.shape[0]): # imgs.shape[0] = 5 
                    # batch size 10 for each class (5 class)
                    image = imgs[i_class]
                    floss += forward_loss(image, model, args.beta)
                floss = floss / imgs.shape[0] # divided by the number of classes

                # backward circle
                imgs = imgs.reshape(-1, *imgs.shape[2:]) # from torch.Size([5, 10, 3, 64, 64]) to torch.Size([50, 3, 64, 64])
                # batch of 50 imagees with mix classes
                bloss = backward_loss(imgs, model, device)

                loss = floss + bloss * args.bloss_coef

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

                    if args.stack_frames:
                        save_image(saved_imgs[:, :3, :, :], os.path.join(log_dir,'stack_%d_%d_%d.png' % (i_epoch, i_split, i_batch)), nrow=9)
                        torch.save(model.state_dict(), os.path.join(log_dir, "model_cycle_vae_stack.pt"))
                        torch.save(model.encoder.state_dict(), os.path.join(log_dir, "encoder_cycle_vae_stack.pt"))
                    else:
                        save_image(saved_imgs, os.path.join(log_dir,'%d_%d_%d.png' % (i_epoch, i_split, i_batch)), nrow=9)
                        torch.save(model.state_dict(), os.path.join(log_dir, "model_cycle_vae.pt"))
                        torch.save(model.encoder.state_dict(), os.path.join(log_dir, "encoder_cycle_vae.pt"))

            # load next splitted data
            updateloader(loader, dataset)

if __name__ == '__main__':
    main()
