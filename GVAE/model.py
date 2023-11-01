import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.autograd import Function
from utils import reparameterize

# Models for carracing games
class Encoder(nn.Module):
    def __init__(self, style_dim = 8, class_dim = 16, input_channel = 3, flatten_size = 1024):
        super(Encoder, self).__init__()
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

    def get_feature(self, x):
        _, _, mu, _ = self.forward(x)
        return mu

class Decoder(nn.Module):
    def __init__(self, latent_size = 24, flatten_size=1024):
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
    def __init__(self, style_dim, class_dim, input_channel = 3, flatten_size = 1024):
        super(GVAE, self).__init__()
        self.encoder = Encoder(style_dim, class_dim, input_channel, flatten_size)
        latent_size = style_dim + class_dim
        self.decoder = Decoder(latent_size, flatten_size)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        mu_s, logsigma_s, mu_c, logsigma_c = self.encoder(x)
        mu = torch.sum(mu_c, dim=0)/10
        var = torch.sum(torch.exp(logsigma_c), dim=0)/10
        logvar = torch.log(var)
        mu = mu.repeat(mu_s.shape[0], 1)
        logvar = logvar.repeat(logsigma_s.shape[0], 1)

        stylecode = reparameterize(mu_s, logsigma_s)
        contentcode = reparameterize(mu, logvar)
        latent_code = torch.cat((stylecode, contentcode), dim=1)
        recon_x = self.decoder(latent_code)

        return mu_s, logsigma_s, mu, logvar, recon_x