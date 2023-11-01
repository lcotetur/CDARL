import numpy as np
import torch
import argparse
import time
import os
import json

from metrics import gaussian_total_correlation, gaussian_wasserstein_correlation, gaussian_wasserstein_correlation_norm
from metrics import compute_mig, mutual_information_score, compute_dci
from LUSR.utils import ExpDataset, reparameterize, RandomTransform
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/home/mila/l/lea.cote-turcotte/LUSR/data/carracing_data', type=str, help='path to the data')
parser.add_argument('--data-tag', default='car', type=str, help='files with data_tag in name under data directory will be considered as collected states')
parser.add_argument('--num-splitted', default=10, type=int, help='number of files that the states from one domain are splitted into')
parser.add_argument('--batch-size', default=10, type=int)
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--encoder-type', default='vae', type=str)
parser.add_argument('--num-episodes', default=1, type=int)
parser.add_argument('--eval_steps', default=100, type=int)
parser.add_argument('--supervised', default=False, type=bool)
parser.add_argument('--model-path', default='/home/mila/l/lea.cote-turcotte/LUSR/checkpoints/model_vae.pt', type=str)
parser.add_argument('--latent-size', default=32, type=int)
parser.add_argument('--save-path', default='/home/mila/l/lea.cote-turcotte/LUSR/results', type=str)
parser.add_argument('--work-dir', default='/home/mila/l/lea.cote-turcotte/LUSR', type=str)
args = parser.parse_args()

def updateloader(loader, dataset):
    dataset.loadnext()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return loader

def reconstruction_loss(x, recon_x):
    recon_loss = F.mse_loss(x, recon_x, reduction='mean')
    return recon_loss

######## models ##########
# Encoder download weights cycle_vae
class EncoderD(nn.Module):
    def __init__(self, class_latent_size = 8, content_latent_size = 16, input_channel = 3, flatten_size = 1024):
        super(EncoderD, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )

        self.linear_mu = nn.Linear(flatten_size, content_latent_size)
        self.linear_logsigma = nn.Linear(flatten_size, content_latent_size)
        self.linear_classcode = nn.Linear(flatten_size, class_latent_size)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        logsigma = self.linear_logsigma(x)
        classcode = self.linear_classcode(x)
        return mu, logsigma, classcode

# Encoder download weights vae
class Encoder(nn.Module):
    def __init__(self, latent_size = 32, input_channel = 3, flatten_size = 1024):
        super(Encoder, self).__init__()
        self.latent_size = latent_size

        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )

        self.linear_mu = nn.Linear(flatten_size, latent_size)
        self.linear_logsigma = nn.Linear(flatten_size, latent_size)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        logsigma = self.linear_logsigma(x)

        return mu, logsigma

# Encoder end-to-end
class EncoderE(nn.Module):
    def __init__(self, class_latent_size = 8, content_latent_size = 16, input_channel = 3, flatten_size = 1024):
        super(EncoderE, self).__init__()
        self.class_latent_size = class_latent_size
        self.content_latent_size = content_latent_size
        self.flatten_size = flatten_size

        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )

        self.linear_mu = nn.Linear(flatten_size, content_latent_size)
        self.linear_logsigma = nn.Linear(flatten_size, content_latent_size)
        self.linear_classcode = nn.Linear(flatten_size, class_latent_size) 

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        logsigma = self.linear_logsigma(x)
        classcode = self.linear_classcode(x)
        return mu

    def get_feature(self, x):
        mu, logsigma, classcode = self.forward(x)
        return mu

class Decoder(nn.Module):
    def __init__(self, latent_size = 32, output_channel = 3, flatten_size=1024):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(latent_size, flatten_size)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(flatten_size, 128, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.main(x)
        return x

class MyModel(nn.Module):
    def __init__(self, encoder_type, deterministic_sample=False, latent_size=16):
        nn.Module.__init__(self)
        self.encoder_type = encoder_type

        # evaluate policy with end-to-end training
        if self.encoder_type == 'end_to_end':
            latent_size = 16
            self.encoder = EncoderE(class_latent_size = 8, content_latent_size = 16, input_channel = 3, flatten_size = 1024)
            self.decoder = Decoder(latent_size)

        # evaluate policy entangle representation
        elif self.encoder_type == 'vae':
            latent_size = 32
            self.encoder = Encoder(latent_size=latent_size)
            self.decoder = Decoder(latent_size)

        # evaluate policy invariant representation
        elif self.encoder_type == 'adagvae':
            latent_size = args.latent_size
            self.encoder = Encoder(latent_size=latent_size)
            self.decoder = Decoder(latent_size)

        # evaluate policy disentangled representation
        elif self.encoder_type == 'cycle_vae':
            class_latent_size = 8
            content_latent_size = 32
            latent_size = class_latent_size + content_latent_size
            self.encoder = EncoderD(class_latent_size, content_latent_size)
            self.decoder = Decoder(latent_size)

    def forward(self, x):
        with torch.no_grad():
            if self.encoder_type == 'cycle_vae':
                content, _, style = self.encoder(x)
                features = torch.cat([content, style], dim=1)
                recon = self.decoder(features)
            elif self.encoder_type == 'vae' or self.encoder_type == 'adagvae':
                features, _, = self.encoder(x)
                recon = self.decoder(features)
            elif self.encoder_type == 'cycle_vae_invar':
                features, _, style = self.encoder(x)
                recon = self.decoder(features)
        return features, recon

def evaluate():
    """Loads a representation TFHub module and computes disentanglement metrics.

    Args:
        model_path: String with path to directory where the representation function
        is saved.
        save_path: String with the path where the results should be saved.
        encoder_type: String with the representation type.
        data_dir: String with the path to the data
        evaluation_fn: Function used to evaluate the representation (see metrics/
        for examples).
        random_seed: Integer with random seed used for training.
        name: Optional string with name of the metric (can be used to name metrics).
    """
    model = MyModel(args.encoder_type)
    weights = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(weights)
    device = torch.device('cpu')

    # create dataset and loader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ExpDataset(args.data_dir, args.data_tag, args.num_splitted, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print('data loaded')

    results = {}
    results_mean = {}
    results['recon_loss'] = []
    results['mig_score'] = []
    results['dci_score'] = []
    results['gaussian_total_corr'] = []
    results['gaussian_wasserstein_corr'] = []
    results['mututal_info_score'] = []
    for i_batch, imgs in enumerate(loader):
        imgs = imgs.permute(1,0,2,3,4).to(device, non_blocking=True)
        imgs = imgs.reshape(-1, *imgs.shape[2:])
        imgs = RandomTransform(imgs).apply_transformations(nb_class=5, value=None)
        imgs = imgs.reshape(-1, *imgs.shape[2:])

        features, recon = model(imgs)

        recon_loss = reconstruction_loss(imgs, recon)
        results['recon_loss'].append(recon_loss.item())

        features = features.transpose(0, 1).numpy() # from [50, 16] to [16, 50]

        if args.supervised == True:
            mig_score = compute_mig(features, ground_truth_factor, num_train=50)
            results['mig_score'].append(mig_score)

            dci_score = compute_dci(ground_truth_data, model, random_state, num_train, num_test, batch_size=16)
            results['dci_score'].append(dci_score)

        gaussian_total_corr = gaussian_total_correlation(features)
        results['gaussian_total_corr'].append(gaussian_total_corr)

        gaussian_wasserstein_corr = gaussian_wasserstein_correlation(features)
        results['gaussian_wasserstein_corr'].append(gaussian_wasserstein_corr)

        mututal_info_score = mutual_information_score(features)
        results['mututal_info_score'].append(mututal_info_score)

        saved_imgs = torch.cat([imgs, recon], dim=0)
        save_image(saved_imgs, "/home/mila/l/lea.cote-turcotte/LUSR/evaluate/%s.png" % (args.encoder_type))

        if i_batch == args.eval_steps:
            total_score_gauss = np.mean(results['gaussian_total_corr'])
            print('Evaluate %d batches and achieved %f gaussian total correlation scores' % (i_batch, total_score_gauss))
            results_mean['recon_loss_mean'] = np.mean(results['recon_loss'])
            if args.supervised == True:
                results_mean['mig_score_mean'] = np.mean(results['mig_score'])
                results_mean['dci_score_mean'] = np.mean(results['dci_score'])
            results_mean['gaussian_total_corr_mean'] = total_score_gauss
            results_mean['gaussian_wasserstein_corr_mean'] = np.mean(results['gaussian_wasserstein_corr'])
            results_mean['mututal_info_score_mean'] = np.mean(results['mututal_info_score'])
            print(results_mean)
            break
    with open(os.path.join(args.save_path, 'results_%s.json' % args.encoder_type), 'w') as f:
        json.dump({'results': results, 'results_mean': results_mean}, f)

if __name__ == '__main__':
    evaluate()