import torch
import torch.nn as nn
import torch.nn.functional as F
from CDARL.utils import reparameterize


def vae_loss(x, mu, logsigma, recon_x, beta=1):
    recon_loss = F.mse_loss(x, recon_x, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    kl_loss = kl_loss / torch.numel(x)
    return recon_loss + kl_loss * beta

def compute_loss(x, model, beta):
    mu, logsigma = model.encoder(x)
    latentcode = reparameterize(mu, logsigma)
    recon = model.decoder(latentcode)
    return vae_loss(x, mu, logsigma, recon, beta)


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
    
class VAE(nn.Module):
    def __init__(self, latent_size = 32, img_channel = 3, flatten_size = 1024):
        super(VAE, self).__init__()
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


class CarlaVAE(nn.Module):
    def __init__(self, latent_size = 32, img_channel = 3, flatten_size=9216):
        super(CarlaVAE, self).__init__()
        self.encoder = CarlaEncoder(latent_size, img_channel, flatten_size)
        self.decoder = CarlaDecoder(latent_size, img_channel)

    def forward(self, x):
        mu, logsigma= self.encoder(x)
        latentcode = reparameterize(mu, logsigma)
        recon_x = self.decoder(latentcode)

        return mu, logsigma, recon_x