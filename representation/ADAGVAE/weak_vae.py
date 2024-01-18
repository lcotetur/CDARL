import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from CDARL.utils import reparameterize

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

class ADAGVAE(nn.Module):
    def __init__(self, latent_size = 32, img_channel = 3, flatten_size = 1024):
        super(ADAGVAE, self).__init__()
        self.encoder = Encoder(latent_size, img_channel, flatten_size)
        self.decoder = Decoder(latent_size, img_channel, flatten_size)

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        latentcode = reparameterize(mu, logsigma)

        recon_x = self.decoder(latentcode)

        return mu, logsigma, recon_x

# Models for CARLA autonomous driving
class CarlaEncoder(nn.Module):
    def __init__(self, latent_size = 32, input_channel = 3, flatten_size = 9216):
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

    def get_feature(self, x):
        mu, logsigma = self.forward(x)
        return mu

class CarlaDecoder(nn.Module):
    def __init__(self, latent_size = 32, output_channel = 3):
        super(CarlaDecoder, self).__init__()
        self.fc = nn.Linear(latent_size, 9216)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = torch.reshape(x, (-1,256,6,6))
        x = self.main(x)
        return x

class CarlaADAGVAE(nn.Module):
    def __init__(self, latent_size = 32, img_channel = 3, flatten_size=9216):
        super(CarlaADAGVAE, self).__init__()
        self.encoder = CarlaEncoder(latent_size, img_channel, flatten_size)
        self.decoder = CarlaDecoder(latent_size, img_channel)

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        latentcode = reparameterize(mu, logsigma)

        recon_x = self.decoder(latentcode)

        return mu, logsigma, recon_x

def compute_loss(model, feature_1, feature_2, beta=1):
    z_mean, z_logvar = model.encoder(feature_1)
    z_mean_2, z_logvar_2 = model.encoder(feature_2)
    kl_per_point = compute_kl(z_mean, z_mean_2, z_logvar, z_logvar_2)

    # compute average based on group-vae
    new_mean = 0.5 * z_mean + 0.5 * z_mean_2
    var_1 = torch.exp(z_logvar)
    var_2 = torch.exp(z_logvar_2)
    new_log_var = torch.log(0.5*var_1 + 0.5*var_2)

    mu_sample_1, log_var_sample_1 = aggregate(z_mean, z_logvar, new_mean, new_log_var, kl_per_point)
    z_sampled_1 = reparameterize(mu_sample_1, log_var_sample_1)

    mu_sample_2, log_var_sample_2 = aggregate(z_mean_2, z_logvar_2, new_mean, new_log_var, kl_per_point)
    z_sampled_2 = reparameterize(mu_sample_2, log_var_sample_2)

    reconstructions_1 = model.decoder(z_sampled_1)
    reconstructions_2 = model.decoder(z_sampled_2)

    reconstruction_loss_1 = F.mse_loss(feature_1, reconstructions_1, reduction='mean')
    reconstruction_loss_2 = F.mse_loss(feature_2, reconstructions_2, reduction='mean')
    reconstruction_loss = (0.5 * reconstruction_loss_1 + 0.5 * reconstruction_loss_2)

    kl_loss_1 = kl_loss(mu_sample_1, log_var_sample_1) / torch.numel(feature_1)
    kl_loss_2 = kl_loss(mu_sample_2, log_var_sample_2) / torch.numel(feature_2)
    mean_kl_loss = 0.5 * kl_loss_1 + 0.5 * kl_loss_2

    loss = reconstruction_loss + beta * mean_kl_loss
    return loss

def compute_kl(z_1, z_2, logvar_1, logvar_2):
    var_1 = torch.exp(logvar_1)
    var_2 = torch.exp(logvar_2)
    return var_1/var_2 + torch.square(z_2-z_1)/var_2 - 1 + logvar_2 - logvar_1

def aggregate(z_mean, z_logvar, new_mean, new_log_var, kl_per_point):
    """Argmax aggregation with adaptive k.

    The bottom k dimensions in terms of distance are not averaged. K is
    estimated adaptively by binning the distance into two bins of equal width.

    Args:
      z_mean: Mean of the encoder distribution for the original image.
      z_logvar: Logvar of the encoder distribution for the original image.
      new_mean: Average mean of the encoder distribution of the pair of images.
      new_log_var: Average logvar of the encoder distribution of the pair of
        images.
      kl_per_point: Distance between the two encoder distributions.

    Returns:
      Mean and logvariance for the new observation.
    """
    mask = create_mask(kl_per_point)
    z_mean_averaged = torch.where(mask.view(z_mean.size()), z_mean, new_mean)
    z_logvar_averaged = torch.where(mask.view(z_logvar.size()), z_logvar, new_log_var)
    return z_mean_averaged, z_logvar_averaged

def create_mask(x):
    """Discretize a vector in two bins."""
    sep = (torch.max(x, dim=1).values - torch.min(x, dim=1).values) / 2
    index = []
    for i, item in enumerate(x):
        index.append(torch.le(item, sep[i]))
    return torch.stack(index)

def kl_loss(z_mean, z_logvar):
    """Compute KL divergence between input Gaussian and Standard Normal."""
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return kl_loss
