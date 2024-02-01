from utils import Results
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn import manifold
import warnings 
from CDARL.utils import ExpDataset, reparameterize, RandomTransform, Results, seed_everything
from CDARL.representation.ILCM.model import MLPImplicitSCM, HeuristicInterventionEncoder, ILCM
from CDARL.representation.ILCM.model import ImageEncoder, ImageDecoder, CoordConv2d, GaussianEncoder
from CDARL.representation.ILCM.model import ImageEncoderCarla, ImageDecoderCarla
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--algo', default='adagvae', type=str)
parser.add_argument('--data', default='carracing', type=str)
parser.add_argument('--data-dir', default='/home/mila/l/lea.cote-turcotte/CDARL/data/carracing_data', type=str, help='path to the data')
#parser.add_argument('--data-dir', default='/home/mila/l/lea.cote-turcotte/CDARL/data/carla_data', type=str, help='path to the data')
parser.add_argument('--data-tag', default='car', type=str, help='files with data_tag in name under data directory will be considered as collected states')
parser.add_argument('--num-splitted', default=10, type=int, help='number of files that the states from one domain are splitted into')
parser.add_argument('--save-path', default="/home/mila/l/lea.cote-turcotte/CDARL/checkimages", type=str)
parser.add_argument('--batch-size', default=60, type=int)
parser.add_argument('--num-workers', default=4, type=int)
args = parser.parse_args()

### CARRACING MODELS
def create_model_reduce_dim():
    # Create model
    scm = create_img_scm()
    encoder, decoder = create_img_encoder_decoder()
    intervention_encoder = create_intervention_encoder()
    model = ILCM(
            scm,
            encoder=encoder,
            decoder=decoder,
            intervention_encoder=intervention_encoder,
            intervention_prior=None,
            averaging_strategy='stochastic',
            dim_z=32,
            )
    return model

def create_img_scm():
    scm = MLPImplicitSCM(
            graph_parameterization='none',
            manifold_thickness=0.01,
            hidden_units=100,
            hidden_layers=2,
            homoskedastic=False,
            dim_z=32,
            min_std=0.2,
        )

    return scm

def create_img_encoder_decoder():
    if args.data == 'carracing':
        encoder = ImageEncoder(
                in_resolution=64,
                in_features=3,
                out_features=32,
                hidden_features=32,
                batchnorm=False,
                conv_class=CoordConv2d,
                mlp_layers=2,
                mlp_hidden=128,
                elementwise_hidden=16,
                elementwise_layers=0,
                min_std=1.e-3,
                permutation=0,
                )
        decoder = ImageDecoder(
                in_features=32,
                out_resolution=64,
                out_features=3,
                hidden_features=32,
                batchnorm=False,
                min_std=1.0,
                fix_std=True,
                conv_class=CoordConv2d,
                mlp_layers=2,
                mlp_hidden=128,
                elementwise_hidden=16,
                elementwise_layers=0,
                permutation=0,
                )
    elif args.data == 'carla':
        encoder = ImageEncoderCarla(
            in_resolution=128,
            in_features=3,
            out_features=32,
            hidden_features=32,
            batchnorm=False,
            conv_class=CoordConv2d,
            mlp_layers=2,
            mlp_hidden=128,
            elementwise_hidden=16,
            elementwise_layers=0,
            min_std=1.e-3,
            permutation=0,
            )
        decoder = ImageDecoderCarla(
            in_features=32,
            out_resolution=128,
            out_features=3,
            hidden_features=32,
            batchnorm=False,
            min_std=1.0,
            fix_std=True,
            conv_class=CoordConv2d,
            mlp_layers=2,
            mlp_hidden=128,
            elementwise_hidden=16,
            elementwise_layers=0,
            permutation=0,
            )
    return encoder, decoder

def create_ilcm():
    """Instantiates a (learnable) VAE model"""

    scm = create_scm()
    encoder, decoder = create_mlp_encoder_decoder()
    intervention_encoder = create_intervention_encoder()
    model = ILCM(
            scm,
            encoder=encoder,
            decoder=decoder,
            intervention_encoder=intervention_encoder,
            intervention_prior=None,
            averaging_strategy='stochastic',
            dim_z=16,
            )

    return model

def create_intervention_encoder():
    intervention_encoder = HeuristicInterventionEncoder()
    return intervention_encoder

def create_mlp_encoder_decoder():
    """Create encoder and decoder"""

    encoder_hidden_layers = 5
    encoder_hidden = [64 for _ in range(encoder_hidden_layers)]
    decoder_hidden_layers = 5
    decoder_hidden = [64 for _ in range(decoder_hidden_layers)]

    encoder = GaussianEncoder(
                hidden=encoder_hidden,
                input_features=32,
                output_features=16,
                fix_std=False,
                init_std=0.01,
                min_std=0.0001,
            )
    decoder = GaussianEncoder(
                hidden=decoder_hidden,
                input_features=16,
                output_features=32,
                fix_std=True,
                init_std=1.0,
                min_std=0.001,
            )

    return encoder, decoder

def create_scm():
    """Creates an SCM"""

    scm = MLPImplicitSCM(
            graph_parameterization='none',
            manifold_thickness=0.01,
            hidden_units=100,
            hidden_layers=2,
            homoskedastic=False,
            dim_z=16,
            min_std=0.2,
        )
    return scm

class EncoderE(nn.Module):
    def __init__(self, class_latent_size = 8, content_latent_size = 32, input_channel = 3, flatten_size = 1024):
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
        return mu, logsigma, classcode 

class Encoder(nn.Module):
    def __init__(self, latent_size = 32, input_channel = 3, flatten_size = 1024):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.flatten_size = flatten_size

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

class CarlaEncoderE(nn.Module):
    def __init__(self, class_latent_size = 8, content_latent_size = 16, input_channel = 3, flatten_size = 9216):
        super(CarlaEncoderE, self).__init__()
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

        return mu, logsigma, classcode

class CarlaEncoder(nn.Module):
    def __init__(self, latent_size = 16, input_channel = 3, flatten_size = 9216):
        super(CarlaEncoder, self).__init__()
        self.latent_size = latent_size
        self.flatten_size = flatten_size

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

def computeTSNEProjectionOfLatentSpace(X, encoder_path, algo=args.algo, data=args.data, ilcm_path='/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carracing/2024-01-15/model_step_180000.pt'):
    # Compute latent space representation
    print("Computing latent space projection...")
    if algo == 'ilcm':
        encoder = create_model_reduce_dim()
        weights_encoder = torch.load(ilcm_path, map_location=torch.device('cuda'))
        for k in list(weights_encoder.keys()):
            if k not in encoder.state_dict().keys():
                del weights_encoder[k]
        encoder.load_state_dict(weights_encoder)
        print("Loaded Weights")

        causal_mlp = create_ilcm()
        weights = torch.load(encoder_path, map_location=torch.device('cuda'))
        for k in list(weights.keys()):
            if k not in causal_mlp.state_dict().keys():
                del weights[k]
        causal_mlp.load_state_dict(weights)
        print("Loaded Weights")
        out_dim = 16

    elif algo == 'vae' or algo == 'adagvae':
        if data == 'carracing':
            encoder = Encoder()
        elif data == 'carla':
            encoder = CarlaEncoder()
        # saved checkpoints could contain extra weights such as linear_logsigma 
        weights = torch.load(encoder_path, map_location=torch.device('cpu'))
        for k in list(weights.keys()):
            if k not in encoder.state_dict().keys():
                del weights[k]
        encoder.load_state_dict(weights)
        print("Loaded Weights")

    elif algo == 'cycle_vae':
        if data == 'carracing':
            encoder = EncoderE()
        elif data == 'carla':
            encoder = CarlaEncoderE()
        # saved checkpoints could contain extra weights such as linear_logsigma 
        weights = torch.load(encoder_path, map_location=torch.device('cpu'))
        for k in list(weights.keys()):
            if k not in encoder.state_dict().keys():
                del weights[k]
        encoder.load_state_dict(weights)
        print("Loaded Weights")

    with torch.no_grad():
        if algo == 'vae' or algo == 'adagvae':
            outputs, _ = encoder(X)
        elif algo == 'ilcm':
            z, _ = encoder.encoder.mean_std(X)
            outputs = causal_mlp.encode_to_causal(z)
        elif algo == 'disent':
            content, _, style = encoder(X)
            outputs = torch.cat([content, style], dim=1)
        elif algo == 'cycle_vae':
            _, _, outputs = encoder(X)

        current_outputs = outputs.cpu().numpy()
        features = np.concatenate((outputs, current_outputs))
        print(features.shape)

    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2)
    X_tsne = tsne.fit_transform(features)
    return X_tsne

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
 
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
 
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range
 
def tsne(tsne, labels, save_path, model_name, data=args.data):
    if data == 'carracing':
        colors_per_class = {'yellow': 'gold', 'blue': 'teal', 'green': 'limegreen', 'mauve': 'slateblue', 'orange': 'coral'}
    elif data == 'carla':
        colors_per_class = {'ClearNoon': 'gold', 'HardRainNoon': 'teal', 'LateEvening': 'slateblue'}
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]
    
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
    
        # convert the class color to matplotlib format
        color = colors_per_class[label]
    
        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label)
    
    # build a legend using the labels we set previously
    ax.legend(loc='best')
 
    # finally, show the plot
    fig.savefig(os.path.join(save_path, model_name))

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    carracing_encoders = {
                'cycle_vae': '/home/mila/l/lea.cote-turcotte/CDARL/representation/CYCLEVAE/runs/carracing/2023-11-20/encoder_cycle_vae.pt',
                'vae': '/home/mila/l/lea.cote-turcotte/CDARL/representation/VAE/runs/carracing/2023-11-20/encoder_vae.pt',
                'adagvae': '/home/mila/l/lea.cote-turcotte/CDARL/representation/ADAGVAE/logs/carracing/2024-01-31/encoder_adagvae.pt',
                'ilcm': '/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carracing/2023-11-21/model_step_188000.pt', 
                'ilcm_reduce_dim': '/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carracing/2024-01-15/model_step_180000.pt'}
    carla_encoders = {
                'cycle_vae': '/home/mila/l/lea.cote-turcotte/CDARL/representation/CYCLEVAE/runs/carla/2024-01-24/encoder_cycle_vae.pt',
                'vae': '/home/mila/l/lea.cote-turcotte/CDARL/representation/VAE/runs/carla/2024-01-24/encoder_vae.pt',
                'adagvae': '',
                'ilcm': '/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carla/2024-01-26/model_step_50000.pt',
                'ilcm_reduce_dim': '/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carla/2024-01-26_1_reduce_dim/model_step_50000.pt'}
    save_path = args.save_path

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ExpDataset(args.data_dir, args.data_tag, args.num_splitted, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    for i_batch, imgs in enumerate(loader):
        imgs = imgs
        break

    imgs = imgs.reshape(-1, *imgs.shape[2:])
    save_image(imgs, os.path.join(save_path ,'tsne_test.png'))
    carracing_labels = [
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange',
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange',
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange',
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange',
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange',
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange',
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange',
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange',
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange',
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange',
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange',
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange']

    carla_labels = ['ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',]
    
    if args.data == 'carracing':
        t_sne = computeTSNEProjectionOfLatentSpace(imgs, carracing_encoders[args.algo], algo=args.algo, ilcm_path=carracing_encoders['ilcm_reduce_dim'])
        tsne(t_sne, carracing_labels, save_path, 't-SNE %s latents carracing' % args.algo)
    elif args.data == 'carla':
        t_sne = computeTSNEProjectionOfLatentSpace(imgs, carla_encoders[args.algo], algo=args.algo, ilcm_path=carla_encoders['ilcm_reduce_dim'])
        tsne(t_sne, carla_labels, save_path, 't-SNE %s latents carla' % args.algo)