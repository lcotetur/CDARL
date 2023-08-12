import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.autograd import Function
from utils import reparameterize


# Models for carracing games
class EncoderStyle(nn.Module):
    def __init__(self, class_latent_size = 3, input_channel = 3, flatten_size = 1024):
        super(EncoderStyle, self).__init__()
        self.class_latent_size = class_latent_size
        self.flatten_size = flatten_size

        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )

        self.linear_mu = nn.Linear(flatten_size, class_latent_size)
        self.linear_var = nn.Softplus(nn.Linear(flatten_size, class_latent_size))

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        var = self.linear_var(x)

        return mu, var

    def get_feature(self, x):
        mu, _ = self.forward(x)
        return mu

# Models for carracing games
class EncoderContent(nn.Module):
    def __init__(self, content_latent_size = 100, input_channel = 3, flatten_size = 1024):
        super(EncoderContent, self).__init__()
        self.content_latent_size = content_latent_size
        self.flatten_size = flatten_size

        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )

        self.linear_mu = nn.Linear(flatten_size, content_latent_size)
        self.linear_var = nn.Softplus(nn.Linear(flatten_size, content_latent_size))

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        var = self.linear_var(x)

        return mu, var

    def get_feature(self, x):
        mu, _ = self.forward(x)
        return mu


class Decoder(nn.Module):
    def __init__(self, latent_size = 103, output_channel = 3, flatten_size=1024):
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

 
class GVAE(nn.Module):
    def __init__(self, class_latent_size = 8, content_latent_size = 16, img_channel = 3, flatten_size = 1024):
        super(GVAE, self).__init__()
        self.encoder_style = EncoderStyle(class_latent_size, img_channel, flatten_size)
        self.encoder_content = EncoderContent(content_latent_size, img_channel, flatten_size)
        self.decoder = Decoder(class_latent_size + content_latent_size, img_channel, flatten_size)

    def forward(self, x):
        mu_s, logsigma_s = self.encoder_style(x)
        mu_c, logsigma_c = self.encoder_content(x)

        stylecode = reparameterize(mu_s, logsigma_s)
        mu = torch.mean(mu_c, dim=0)
        logsigma = torch.mean(logsigma_c, dim=0)
        #change logsigma to variance
        contentcode = reparameterize(mu, logsigma)
        contentcode = contentcode.repeat(stylecode.shape[0], 1)

        latentcode = torch.cat([stylecode, contentcode], dim=1)

        recon_x = self.decoder(latentcode)

        return stylecode, contentcode, recon_x