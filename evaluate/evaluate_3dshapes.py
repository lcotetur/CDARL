import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
import json
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image

import numpy as np
import argparse
import os
import warnings

from metrics import gaussian_total_correlation, gaussian_wasserstein_correlation, gaussian_wasserstein_correlation_norm
from metrics import compute_mig, mutual_information_score, compute_dci
from CDARL.data.shapes3d_data import Shape3dDataset

from CDARL.representation.ILCM.model import MLPImplicitSCM, HeuristicInterventionEncoder, ILCM
from CDARL.representation.ILCM.model import GaussianEncoder, ImageEncoder, ImageDecoder, CoordConv2d, Encoder3dshapes, Decoder3dshapes
from CDARL.utils import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/home/mila/l/lea.cote-turcotte/CDARL/data/3dshapes.h5', type=str)
parser.add_argument('--batch-size', default=16, type=int)
parser.add_argument('--num-train', default=1000, type=int) #10000
parser.add_argument('--num-test', default=500, type=int) #5000
parser.add_argument('--num-workers', default=4, type=int)
parser.add_argument('--encoder-type', default='adagvae', type=str)
parser.add_argument('--num-episodes', default=1, type=int)
parser.add_argument('--eval-steps', default=2, type=int)
parser.add_argument('--supervised', default=True, type=bool)
parser.add_argument('--encoder-path', default='/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/3dshapes/2024-02-22/model_reduce_dim_6000.pt', type=str)
parser.add_argument('--model-path', default='/home/mila/l/lea.cote-turcotte/CDARL/representation/ADAGVAE/logs/3dshapes/2024-03-09_0/encode_3dshapes.pt', type=str)
parser.add_argument('--latent-size', default=10, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--boost-mode', default="sklearn", type=str)
parser.add_argument('--save-path', default='/home/mila/l/lea.cote-turcotte/CDARL/results/3dshapes', type=str)
parser.add_argument('--work-dir', default='/home/mila/l/lea.cote-turcotte/CDARL', type=str)
args = parser.parse_args()

def reconstruction_loss(x, recon_x):
    recon_loss = F.mse_loss(x, recon_x, reduction='mean')
    return recon_loss

######## ilcm model #########
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
    encoder = Encoder3dshapes(
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
    decoder = Decoder3dshapes(
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
            dim_z=6,
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
                output_features=6,
                fix_std=False,
                init_std=0.01,
                min_std=0.0001,
            )
    decoder = GaussianEncoder(
                hidden=decoder_hidden,
                input_features=6,
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
            dim_z=6,
            min_std=0.2,
        )
    return scm

def create_intervention_encoder():
    intervention_encoder = HeuristicInterventionEncoder()
    return intervention_encoder

######## models ##########

# Encoder download weights GVAE
class EncoderG(nn.Module):
    def __init__(self, style_dim = 8, class_dim = 16, input_channel = 3, flatten_size = 1024):
        super(EncoderG, self).__init__()
        self.flatten_size = flatten_size

        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )

        self.style_mu = nn.Linear(in_features=flatten_size, out_features=style_dim)
        self.style_logvar = nn.Linear(in_features=flatten_size, out_features=style_dim)

        # class
        self.class_mu = nn.Linear(in_features=flatten_size, out_features=class_dim)
        self.class_logvar = nn.Linear(in_features=flatten_size, out_features=class_dim)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)

        style_latent_space_mu = self.style_mu(x)
        style_latent_space_logvar = self.style_logvar(x)

        class_latent_space_mu = self.class_mu(x)
        class_latent_space_logvar = self.class_logvar(x)

        return style_latent_space_mu, style_latent_space_logvar, class_latent_space_mu, class_latent_space_logvar

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
    def __init__(self, latent_size = 40, output_channel = 3, flatten_size=1024):
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

class Encoder3dshapes(nn.Module):
    def __init__(self, latent_size = 10, input_channel = 3, flatten_size = 1024):
        super(Encoder3dshapes, self).__init__()
        self.z_total = latent_size

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),  # This was reverted to kernel size 4x4 from 2x2, to match beta-vae paper
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),  # This was reverted to kernel size 4x4 from 2x2, to match beta-vae paper
            nn.Flatten(),
            nn.Linear(in_features=1600, out_features=256),
            nn.ReLU(inplace=True)
        )

        self.linear_mu = nn.Linear(256, self.z_total)
        self.linear_logsigma = nn.Linear(256, self.z_total)

    def forward(self, x):
        z = self.model(x)
        mu, logsigma = self.linear_mu(z), self.linear_logsigma(z)

        return mu, logsigma

class Decoder3dshapes(nn.Module):
    def __init__(self, latent_size = 10, output_channel = 3, flatten_size=1024):
        super(Decoder3dshapes, self).__init__()
        self.z_size=latent_size

        self.model = nn.Sequential(
            nn.Linear(in_features=self.z_size, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(64, 4, 4)),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class MyModel(nn.Module):
    def __init__(self, encoder_type, deterministic_sample=False, latent_size=16):
        nn.Module.__init__(self)
        self.encoder_type = encoder_type

        # evaluate with end-to-end training
        if self.encoder_type == 'end_to_end':
            latent_size = 16
            self.encoder = EncoderE(class_latent_size = 8, content_latent_size = 16, input_channel = 3, flatten_size = 1024)
            self.decoder = Decoder(latent_size)

        # evaluate entangle representation
        elif self.encoder_type == 'vae':
            latent_size = args.latent_size
            self.encoder = Encoder(latent_size=latent_size)
            self.decoder = Decoder(latent_size)

        # evaluate invariant representation
        elif self.encoder_type == 'adagvae':
            latent_size = args.latent_size
            #self.encoder = Encoder3dshapes()
            #self.decoder = Decoder3dshapes() 
            self.encoder = Encoder(latent_size=latent_size)
            self.decoder = Decoder(latent_size)

        # evaluate disentangled representation
        elif self.encoder_type == 'cycle_vae':
            class_latent_size = 8
            content_latent_size = 16
            latent_size = class_latent_size + content_latent_size
            self.encoder = EncoderD(class_latent_size, content_latent_size)
            self.decoder = Decoder(latent_size)

        # evaluate disentangled representation
        elif self.encoder_type == 'gvae':
            class_latent_size = 8
            content_latent_size = 32
            latent_size = class_latent_size + content_latent_size
            self.encoder = EncoderG(class_latent_size, content_latent_size)
            self.decoder = Decoder(latent_size)

        # evaluate representation ilcm
        elif self.encoder_type == 'ilcm_reduce_dim':
            latent_size = 32
            self.model = create_model_reduce_dim()

            weights = torch.load(args.model_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(weights)
            print("Loaded Weights Reduce Dim Ilcm")

        # evaluate representation ilcm
        elif self.encoder_type == 'ilcm':
            latent_size = 6
            self.encoder = create_model_reduce_dim()
            self.model = create_ilcm()

            weights = torch.load(args.encoder_path, map_location=torch.device('cpu'))
            self.encoder.load_state_dict(weights)
            print("Loaded Weights Reduce Dim Ilcm")

            weights = torch.load(args.model_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(weights)
            print("Loaded Weights Ilcm")

    def forward(self, x):
        with torch.no_grad():
            if self.encoder_type == 'cycle_vae':
                content, _, style = self.encoder(x)
                features = torch.cat([content, style], dim=1)
                recon = self.decoder(features)
            elif self.encoder_type == 'vae' or self.encoder_type == 'adagvae':
                features, _, = self.encoder(x)
                recon = self.decoder(features)
            elif self.encoder_type == 'gvae':
                mu_s, _, mu, _ = self.encoder(x)
                features = torch.cat([mu_s, mu], dim=1)
                recon = self.decoder(features)
            elif self.encoder_type == 'ilcm':
                z, _ = self.encoder.encoder.mean_std(x)
                features = self.model.encode_to_causal(z)
                recon = self.encoder.encode_decode(x)
        return features, recon

def evaluate():
    warnings.filterwarnings("ignore")
    print(args.boost_mode)
    seed_everything(args.seed)
    model = MyModel(args.encoder_type)
    if args.encoder_type != 'ilcm':
        weights = torch.load(args.model_path, map_location=torch.device('cpu'))
        model.load_state_dict(weights)
    device = torch.device('cpu')

    # create dataset
    dataset = Shape3dDataset()
    dataset.load_dataset(file_dir=args.data_dir)
    print('data loaded')

    results = {}

    img_batch = dataset.sample_random_batch(args.num_train)
    img_batch = dataset.process_obs(img_batch, device)
    feature, recon = model(img_batch)
    recon_loss = reconstruction_loss(img_batch, recon)
    results['recon_loss'] = recon_loss.item()
    saved_imgs = torch.cat([img_batch[:10], recon[:10]], dim=0)
    save_image(saved_imgs, "/home/mila/l/lea.cote-turcotte/CDARL/evaluate/3dshapes_%s.png" % (args.encoder_type))
    print('recon_loss')

    features, _ = dataset.generate_batch_factor_code(model, num_points=args.num_train, batch_size=args.batch_size)

    gaussian_total_corr = gaussian_total_correlation(features)
    results['gaussian_total_corr'] = gaussian_total_corr 
    print('gaussian_total_corr')

    gaussian_wasserstein_corr = gaussian_wasserstein_correlation(features)
    results['gaussian_wasserstein_corr'] = gaussian_wasserstein_corr
    print('gaussian_wasserstein_corr')

    mututal_info_score = mutual_information_score(features)
    results['mututal_info_score'] = mututal_info_score
    print('mututal_info_score')

    if args.supervised == True:
        mig_score = compute_mig(dataset, model, num_train=args.num_train, batch_size=args.batch_size)
        results['mig_score'] = mig_score["discrete_mig"] 
        print('mig_score')

        dci_score = compute_dci(dataset, model, num_train=args.num_train, num_test=args.num_test, batch_size=args.batch_size, boost_mode=args.boost_mode)
        results['dci_disent_score'] = dci_score["disentanglement"] 
        results['dci_comp_score'] = dci_score["completeness"] 
        results['dci_infotrain_score'] = dci_score["informativeness_train"]
        results['dci_infotest_score'] = dci_score["informativeness_test"] 
        print('dci_disent_score')

    print('Evaluate %d step and achieved %f gaussian total correlation scores' % (args.num_train, results['gaussian_total_corr']))
    print(results)
    with open(os.path.join(args.save_path, 'results_3dshapes_%s.json' % args.encoder_type), 'w') as f:
        json.dump({'results': results}, f)

if __name__ == '__main__':
    evaluate()
