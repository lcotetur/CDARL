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

from utils import ExpDataset, reparameterize, RandomTransform
from model import GVAE

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/home/mila/l/lea.cote-turcotte/LUSR/data/carracing_data', type=str, help='path to the data')
parser.add_argument('--data-tag', default='car', type=str, help='files with data_tag in name under data directory will be considered as collected states')
parser.add_argument('--num-splitted', default=10, type=int, help='number of files that the states from one domain are splitted into')
parser.add_argument('--batch-size', default=10, type=int)
parser.add_argument('--num-epochs', default=2, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--learning-rate', default=0.0001, type=float)
parser.add_argument('--beta', default=10, type=int)
parser.add_argument('--save-freq', default=1000, type=int)
parser.add_argument('--bloss-coef', default=1, type=int)
parser.add_argument('--class-latent-size', default=8, type=int)
parser.add_argument('--content-latent-size', default=16, type=int)
parser.add_argument('--flatten-size', default=1024, type=int)
args = parser.parse_args()

Model = GVAE

def updateloader(loader, dataset):
    dataset.loadnext()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return loader

def kl_loss(mu, var):
    kl_loss = -0.5 * torch.sum(1 + torch.log(var) - mu.pow(2) - var)
    return kl_loss

def gvae_loss(x, model, beta=1):
    mu_s, var_s = model.encoder_style(x)
    mu_c, var_c = model.encoder_content(x)
    #print((mu_s, var_s))
    #print((mu_c, var_c))

    stylecode = reparameterize(mu_s, var_s)

    mu = torch.mean(mu_c, dim=0)
    var = torch.mean(var_c, dim=0)
    contentcode = reparameterize(mu, var)
    contentcode = contentcode.repeat(stylecode.shape[0], 1)


    latentcode = torch.cat([stylecode, contentcode], dim=1)


    recon_x = model.decoder(latentcode)
    #print(recon_x)

    recon_loss = F.mse_loss(x, recon_x, reduction='mean')
    style_loss = kl_loss(mu_s, var_s) / torch.numel(x)
    content_loss = kl_loss(mu, var)
    #print((recon_loss, style_loss, content_loss))

    return recon_loss - style_loss * beta - content_loss * beta


def main():
    # create directory
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    if not os.path.exists('checkimages'):
        os.makedirs("checkimages")

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
    writer = SummaryWriter()
    batch_count = 0
    for i_epoch in range(args.num_epochs):
        for i_split in range(args.num_splitted):
            for i_batch, imgs in enumerate(loader):

                batch_count += 1
                # forward circle
                # Try 
                imgs = imgs.permute(1,0,2,3,4).to(device, non_blocking=True) # from torch.Size([10, 5, 3, 64, 64]) to torch.Size([5, 10, 3, 64, 64])
                imgs = imgs.reshape(-1, *imgs.shape[2:])
                imgs = RandomTransform(imgs).apply_transformations(nb_class=5)
                #imgs = imgs.permute(1,0,2,3,4).to(device, non_blocking=True) # from torch.Size([10, 5, 3, 64, 64]) to torch.Size([5, 10, 3, 64, 64])
                optimizer.zero_grad()

                loss = 0
                for i_group in range(imgs.shape[0]):
                    # batch size 10 for each class (5 group)
                    image = imgs[i_group]
                    loss += gvae_loss(image, model, args.beta)
                loss = loss / imgs.shape[0] # divided by the number of classes
                loss.backward()
                optimizer.step()

                # write log
                writer.add_scalar('loss', loss.item(), batch_count)

                # save image to check and save model 
                if i_batch % args.save_freq == 0:
                    imgs = imgs.reshape(-1, *imgs.shape[2:])
                    print("%d Epochs, %d Splitted Data, %d Batches is Done." % (i_epoch, i_split, i_batch))
                    rand_idx = torch.randperm(imgs.shape[0])
                    imgs1 = imgs[rand_idx[:9]]
                    imgs2 = imgs[rand_idx[-9:]]
                    print(imgs1.shape)
                    print(imgs2.shape)
                    with torch.no_grad():
                        mu_s, var_s = model.encoder_style(imgs1)
                        mu_c1, var_c1 = model.encoder_content(imgs1)
                        mu_c2, var_c2 = model.encoder_content(imgs2)
                        style_code = reparameterize(mu_s, var_s)
                        content_code1 = reparameterize(mu_c1, var_c1)
                        content_code2 = reparameterize(mu_c2, var_c2)
                        #recon_imgs1 = model.decoder(torch.cat([style_code, content_code1], dim=1))
                        recon_imgs1 = model.decoder(torch.cat([mu_s, mu_c1], dim=1))
                        recon_imgs1 = model.decoder(torch.cat([style_code, content_code1], dim=1))
                        recon_combined = model.decoder(torch.cat([mu_s, mu_c2], dim=1))

                    saved_imgs = torch.cat([imgs1, imgs2, recon_imgs1, recon_combined], dim=0)
                    save_image(saved_imgs, "/home/mila/l/lea.cote-turcotte/LUSR/GVAE/checkimages/%d_%d_%d.png" % (i_epoch, i_split,i_batch), nrow=9)

                    torch.save(model.state_dict(), "/home/mila/l/lea.cote-turcotte/LUSR/GVAE/checkpoints/model.pt")
                    torch.save(model.encoder_content.state_dict(), "/home/mila/l/lea.cote-turcotte/LUSR/GVAE/checkpoints/encoder.pt")

            # load next splitted data
            updateloader(loader, dataset)
    writer.close()

if __name__ == '__main__':
    main()