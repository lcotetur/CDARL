from utils import Results
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn import manifold
#from CDARL.data.shapes3d_data import Shape3dDataset
import warnings 
from CDARL.utils import ExpDataset, reparameterize, RandomTransform, Results, seed_everything
from CDARL.representation.ILCM.model import MLPImplicitSCM, HeuristicInterventionEncoder, ILCM
from CDARL.representation.ILCM.model import ImageEncoder, ImageDecoder, CoordConv2d, GaussianEncoder
from CDARL.representation.ILCM.model import ImageEncoderCarla, ImageDecoderCarla, Encoder3dshapes, Decoder3dshapes, BasicEncoderCarla, BasicDecoderCarla
import json
import os
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--algo', default='ilcm', type=str)
parser.add_argument('--data', default='carracing', type=str)
parser.add_argument('--seed', default=2, type=int)
parser.add_argument('--ilcm_encoder_type', default='conv', type=str)
parser.add_argument('--data-dir-carracing', default='/home/mila/l/lea.cote-turcotte/CDARL/data/carracing_data', type=str, help='path to the data')
parser.add_argument('--data-dir-carla', default='/home/mila/l/lea.cote-turcotte/CDARL/data/carla_data', type=str, help='path to the data')
parser.add_argument('--data-tag', default='car', type=str, help='files with data_tag in name under data directory will be considered as collected states')
parser.add_argument('--num-splitted', default=10, type=int, help='number of files that the states from one domain are splitted into')
parser.add_argument('--save-path', default="/home/mila/l/lea.cote-turcotte/CDARL/checkimages", type=str)
parser.add_argument('--batch-size', default=120, type=int)
parser.add_argument('--latent-size', default=10, type=int, help='dimension of latent state embedding')
parser.add_argument('--reduce_dim_latent_size', default=16, type=int)
parser.add_argument('--num-workers', default=4, type=int)
args = parser.parse_args()

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
            dim_z=args.reduce_dim_latent_size,
            )
    return model

def create_img_scm():
    scm = MLPImplicitSCM(
            graph_parameterization='none',
            manifold_thickness=0.01,
            hidden_units=100,
            hidden_layers=2,
            homoskedastic=False,
            dim_z=args.reduce_dim_latent_size,
            min_std=0.2,
        )

    return scm

def create_img_encoder_decoder():
    if args.data == 'carracing' or args.data == '3dshapes':
        if args.ilcm_encoder_type == 'resnet':
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
        elif args.ilcm_encoder_type == 'conv':
            encoder = Encoder3dshapes(
                        in_resolution=64,
                        in_features=3,
                        out_features=args.reduce_dim_latent_size,
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
            decoder = Decoder3dshapes(
                        in_features=args.reduce_dim_latent_size,
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
        if args.ilcm_encoder_type == 'resnet':
            encoder = ImageEncoderCarla(
                    in_resolution=128,
                    in_features=3,
                    out_features=8,
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
                    in_features=8,
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
        elif args.ilcm_encoder_type == 'conv':
            encoder = BasicEncoderCarla(
                    in_resolution=128,
                    in_features=3,
                    out_features=args.reduce_dim_latent_size,
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
            decoder = BasicDecoderCarla(
                    in_features=args.reduce_dim_latent_size,
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
            dim_z=args.latent_size,
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
                input_features=args.reduce_dim_latent_size,
                output_features=args.latent_size,
                fix_std=False,
                init_std=0.01,
                min_std=0.0001,
            )
    decoder = GaussianEncoder(
                hidden=decoder_hidden,
                input_features=args.latent_size,
                output_features=args.reduce_dim_latent_size,
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
            dim_z=args.latent_size,
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
        x = x.reshape(x.size(0), -1)
        mu = self.linear_mu(x)
        logsigma = self.linear_logsigma(x)

        return mu, logsigma

class CarlaEncoderE(nn.Module):
    def __init__(self, class_latent_size = 8, content_latent_size = 32, input_channel = 3, flatten_size = 9216):
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
        weights_encoder = torch.load(encoder_path, map_location=torch.device('cpu'))
        for k in list(weights_encoder.keys()):
            if k not in encoder.state_dict().keys():
                del weights_encoder[k]
        encoder.load_state_dict(weights_encoder)
        print("Loaded Weights")

        causal_mlp = create_ilcm()
        weights = torch.load(ilcm_path, map_location=torch.device('cpu'))
        for k in list(weights.keys()):
            if k not in causal_mlp.state_dict().keys():
                del weights[k]
        causal_mlp.load_state_dict(weights)
        print("Loaded Weights")
        out_dim = 6

    elif algo == 'vae' or algo == 'adagvae':
        if data == 'carracing' or data == '3dshapes':
            encoder = Encoder()
        elif data == 'carla':
            encoder = CarlaEncoder(latent_size=10)
        # saved checkpoints could contain extra weights such as linear_logsigma 
        weights = torch.load(encoder_path, map_location=torch.device('cpu'))
        for k in list(weights.keys()):
            if k not in encoder.state_dict().keys():
                del weights[k]
        encoder.load_state_dict(weights)
        print("Loaded Weights")

    elif algo == 'cycle_vae':
        if data == 'carracing' or data == '3dshapes':
            encoder = EncoderE()
        elif data == 'carla':
            encoder = CarlaEncoderE(content_latent_size=16)
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
    elif data == '3dshapes':
        colors_per_class = {'floor_hue': 'gold', 'wall_hue': 'teal', 'object_hue': 'green', 'scale': 'slateblue', 'shape': 'coral', 'orientation': 'red'}
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
        plt.axis("off")
    
    # build a legend using the labels we set previously
    ax.legend(loc='best')
 
    # finally, show the plot
    fig.savefig(os.path.join(save_path, model_name))
    plt.close(fig)

# Saliency Map
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)
transform = transforms.Compose([
    normalize,          
])

def load_weights_saliency_map(model, encoder_path):
    weights_encoder = torch.load(encoder_path, map_location=torch.device('cpu'))
    for k in list(weights_encoder.keys()):
        if k not in model.state_dict().keys():
            del weights_encoder[k]
    model.load_state_dict(weights_encoder)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model

def copy_input(input, info):
    input = copy.deepcopy(input)
    input.requires_grad = True
    info.append(input)
    return input

def saliency_map(X, encoders, algos=['vae', 'cycle_vae', 'adagvae', 'ilcm'], data=args.data, ilcm_path='/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carracing/2024-01-15/model_step_180000.pt'):
    ilcm = create_model_reduce_dim()
    ilcm = load_weights_saliency_map(ilcm, encoders['ilcm'])

    causal_mlp = create_ilcm()
    causal_mlp = load_weights_saliency_map(causal_mlp, encoders['ilcm_causal'])

    if data == 'carracing' or data == '3dshapes':
        vae = Encoder(latent_size=32)
        adagvae = Encoder(latent_size=32)
    elif data == 'carla':
        vae = CarlaEncoder()
        adagvae = CarlaEncoder(latent_size=10)
    # saved checkpoints could contain extra weights such as linear_logsigma 
    vae = load_weights_saliency_map(vae, encoders['vae'])
    adagvae = load_weights_saliency_map(adagvae, encoders['adagvae'])

    if data == 'carracing' or data == '3dshapes':
        cycle_vae = EncoderE()
    elif data == 'carla':
        cycle_vae = CarlaEncoderE(content_latent_size=16)
    cycle_vae = load_weights_saliency_map(cycle_vae, encoders['cycle_vae'])

    input = normalize(X)
    input.unsqueeze(0)

    infos = {'vae':[], 'cycle-vae':[], 'adagvae':[], 'ilcm':[]}
    for algo in algos:
        if algo == 'vae':
            input = copy_input(input, infos['vae'])
            print(input.shape)
            pred, _ = vae(input)
            infos['vae'].append(pred)
        elif algo == 'adagvae':
            input = copy_input(input, infos['adagvae'])
            pred, _ = adagvae(input)
            infos['adagvae'].append(pred)
        elif algo == 'ilcm':
            input = copy_input(input, infos['ilcm'])
            z, _ = ilcm.encoder.mean_std(input)
            pred = causal_mlp.encode_to_causal(z)
            infos['ilcm'].append(pred)
        elif algo == 'cycle_vae':
            input = copy_input(input, infos['cycle-vae'])
            content, _, style = cycle_vae(input)
            pred = torch.cat([content, style], dim=1)
            infos['cycle-vae'].append(pred)
        elif algo == 'cycle_vae content':
            input = copy_input(input, infos['cycle-vae content'])
            pred, _, _ = cycle_vae(input)
            infos['cycle-vae content'].append(pred)

    for key, pred in infos.items():
        score, indices = torch.max(pred[1], 1)

        score.backward()

        slc, _ = torch.max(torch.abs(pred[0].grad[0]), dim=0)
        slc = (slc - slc.min()/(slc.max() - slc.min()))
        infos[key].append(slc)

    with torch.no_grad():
        input_img = inv_normalize(input[0])

    img1 = np.transpose(input_img.detach().numpy(), (1, 2, 0))

    fig = plt.figure(figsize=(30, 10))
    plt.subplot(1, 5, 1)
    plt.imshow(img1)
    plt.xticks([])
    plt.yticks([])
    i=2
    for key, info in infos.items():
        plt.subplot(1, 5, i)
        img2 = info[2].numpy()
        plt.imshow(img2, cmap=plt.cm.hot, interpolation='quadric')
        plt.xticks([])
        plt.yticks([])
        plt.title(key, size=30)
        i = i+1
    fig.savefig(os.path.join(save_path, 'Saliency map'))
    plt.close(fig)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    seed_everything(args.seed)
    carracing_encoders = {
                'cycle_vae_style': '/home/mila/l/lea.cote-turcotte/CDARL/representation/CYCLEVAE/runs/carracing/2024-02-26/encoder_cycle_vae_stack.pt', #/home/mila/l/lea.cote-turcotte/CDARL/representation/CYCLEVAE/runs/carracing/2023-11-20/encoder_cycle_vae.pt',
                'cycle_vae': '/home/mila/l/lea.cote-turcotte/CDARL/representation/CYCLEVAE/runs/carracing/2024-02-26/encoder_cycle_vae_stack.pt', #'/home/mila/l/lea.cote-turcotte/CDARL/representation/CYCLEVAE/runs/carracing/2023-11-20/encoder_cycle_vae.pt',
                'disent': '/home/mila/l/lea.cote-turcotte/CDARL/representation/CYCLEVAE/runs/carracing/2024-02-26/encoder_cycle_vae_stack.pt', #/home/mila/l/lea.cote-turcotte/CDARL/representation/CYCLEVAE/runs/carracing/2023-11-20/encoder_cycle_vae.pt',
                'vae': '/home/mila/l/lea.cote-turcotte/CDARL/representation/VAE/runs/carracing/2024-02-27/encoder_vae_stack.pt', #'/home/mila/l/lea.cote-turcotte/CDARL/representation/VAE/runs/carracing/2023-11-20/encoder_vae.pt',
                'adagvae': '/home/mila/l/lea.cote-turcotte/CDARL/representation/ADAGVAE/logs/carracing/2024-03-07/encoder_adagvae.pt', #/home/mila/l/lea.cote-turcotte/CDARL/representation/ADAGVAE/logs/carracing/2024-02-19/encoder_adagvae.pt', #'/home/mila/l/lea.cote-turcotte/CDARL/representation/ADAGVAE/logs/carracing/2023-11-21/encoder_adagvae.pt',
                'ilcm': '/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carracing/2024-03-26/model_reduce_dim_step_400000.pt', #'/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carracing/2024-02-26/model_reduce_dim_step_330000.pt'
                'ilcm_causal': '/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carracing/2024-03-26_ilcm/model_step_120000_0.pt' } #'/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carracing/2024-02-27/model_step_150000.pt'} #
    #'/home/mila/l/lea.cote-turcotte/CDARL/representation/ADAGVAE/logs/carracing/2024-01-31/encoder_adagvae.pt'
    carla_encoders = {
                'cycle_vae': '/home/mila/l/lea.cote-turcotte/CDARL/representation/CYCLEVAE/runs/carla/2024-01-24/encoder_cycle_vae.pt',
                'vae': '/home/mila/l/lea.cote-turcotte/CDARL/representation/VAE/runs/carla/2024-01-24/encoder_vae.pt',
                'adagvae': '/home/mila/l/lea.cote-turcotte/CDARL/representation/ADAGVAE/logs/carla/2024-03-12/encoder_adagvae.pt',
                'ilcm_causal': '/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carla/2024-04-14_ilcm/model_step_120000.pt',
                'ilcm': '/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carla/2024-04-13_1/model_step_50000.pt'}
    shapes3d_encoders = {
                'cycle_vae style': '/home/mila/l/lea.cote-turcotte/CDARL/representation/CYCLEVAE/runs/3dshapes/2024-02-26/encoder.pt',
                'cycle_vae': '/home/mila/l/lea.cote-turcotte/CDARL/representation/CYCLEVAE/runs/3dshapes/2024-02-26/encoder.pt',
                'vae': '/home/mila/l/lea.cote-turcotte/CDARL/representation/VAE/runs/3dshapes/2024-02-13/encoder_vae.pt',
                'adagvae': '/home/mila/l/lea.cote-turcotte/CDARL/representation/ADAGVAE/logs/3dshapes/2024-02-08_1/encode_3dshapes.pt',
                'ilcm_causal': '/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/3dshapes/2024-02-23/model_step_120.pt',
                'ilcm': '/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/3dshapes/2024-02-22/model_reduce_dim_6000.pt'}
    save_path = args.save_path

    transform = transforms.Compose([transforms.ToTensor()])

    dataset_carracing = ExpDataset(args.data_dir_carracing, 'car', 10, transform)
    loader_carracing = DataLoader(dataset_carracing, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    dataset_carla = ExpDataset(args.data_dir_carla, 'weather', 1, transform)
    loader_carla = DataLoader(dataset_carla, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    for i_batch, imgs in enumerate(loader_carracing):
        imgs_carracing = imgs
        break

    for i_batch, imgs in enumerate(loader_carla):
        imgs_carla = imgs
        break
    
    #imgs = imgs.permute(1,0,2,3,4) 
    imgs_carracing = imgs_carracing.reshape(-1, *imgs_carracing.shape[2:])
    imgs_carla = imgs_carla.reshape(-1, *imgs_carla.shape[2:])
    save_image(imgs_carracing, os.path.join(save_path ,'carracing.png'))
    save_image(imgs_carla, os.path.join(save_path ,'carla.png'))
    carracing_labels_road = [
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange', 
              'yellow', 'blue', 'green','mauve', 'orange',
              'yellow', 'blue', 'green','mauve', 'orange']

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
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',
                    'ClearNoon', 'HardRainNoon', 'LateEvening',]
    labels_3dshapes = ['floor_hue', 'floor_hue', 'floor_hue', 'floor_hue' 'floor_hue', 
                        'floor_hue', 'floor_hue', 'floor_hue', 'floor_hue', 'floor_hue',
                        'wall_hue', 'wall_hue', 'wall_hue', 'wall_hue', 'wall_hue',
                        'wall_hue', 'wall_hue', 'wall_hue', 'wall_hue', 'wall_hue', 
                        'object_hue', 'object_hue', 'object_hue', 'object_hue', 'object_hue',
                        'object_hue', 'object_hue', 'object_hue', 'object_hue', 'object_hue', 
                        'scale', 'scale', 'scale', 'scale', 'scale',
                        'scale', 'scale', 'scale', 'scale', 'scale',
                        'shape', 'shape', 'shape', 'shape', 'shape',
                        'shape', 'shape', 'shape', 'shape', 'shape',
                        'orientation', 'orientation', 'orientation', 'orientation', 'orientation'
                        'orientation', 'orientation', 'orientation', 'orientation', 'orientation']
    
    if args.data == 'carracing':
        t_sne = computeTSNEProjectionOfLatentSpace(imgs_carracing, carracing_encoders[args.algo], algo=args.algo, data='carracing', ilcm_path=carracing_encoders['ilcm_causal'])
        tsne(t_sne, carracing_labels, save_path, 't-SNE %s latents carracing' % args.algo, data='carracing')
        saliency_map(imgs_carracing[0].unsqueeze(0), carracing_encoders, ilcm_path=carracing_encoders['ilcm_causal'])
    elif args.data == 'carla':
        t_sne = computeTSNEProjectionOfLatentSpace(imgs_carla, carla_encoders[args.algo], algo=args.algo, data='carla', ilcm_path=carla_encoders['ilcm_causal'])
        tsne(t_sne, carla_labels, save_path, 't-SNE %s latents carla' % args.algo, data='carla')
        saliency_map(imgs_carla[0].unsqueeze(0), carla_encoders, ilcm_path=carla_encoders['ilcm_causal'])
    #elif args.data == '3dshapes':
        #shapes3d data
        #dataset3dshapes = Shape3dDataset()
        #dataset3dshapes.load_dataset(file_dir='/home/mila/l/lea.cote-turcotte/CDARL/data/3dshapes.h5')
        #imgs1 = dataset3dshapes.create_weak_vae_batch(2, 'cpu', k=1)
        #saliency_map(imgs1[0].unsqueeze(0)[:, :, :64, :], shapes3d_encoders, ilcm_path=shapes3d_encoders['ilcm_causal'])
        #imgs2 = dataset3dshapes.create_batch_plot()
        #imgs2 = imgs2.reshape(-1, *imgs2.shape[2:])
        #save_image(imgs2, os.path.join(save_path ,'tsne_3dshapes.png'))
        #t_sne = computeTSNEProjectionOfLatentSpace(imgs2, shapes3d_encoders[args.algo], algo=args.algo, ilcm_path=shapes3d_encoders['ilcm_causal'])
        #tsne(t_sne, labels_3dshapes, save_path, 't-SNE %s one changes factor 3dshapes' % args.algo)