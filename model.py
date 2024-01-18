import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta
from torch.autograd import Function
from utils import reparameterize

class CarlaLatentPolicy(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_layer=[64,64]):
        super(CarlaLatentPolicy, self).__init__()
        actor_layer_size = [input_dim] + hidden_layer
        actor_feature_layers = nn.ModuleList([])
        for i in range(len(actor_layer_size)-1):
            actor_feature_layers.append(nn.Linear(actor_layer_size[i], actor_layer_size[i+1]))
            actor_feature_layers.append(nn.ReLU())
        self.actor = nn.Sequential(*actor_feature_layers)
        self.alpha_head = nn.Sequential(nn.Linear(hidden_layer[-1], action_dim), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(hidden_layer[-1], action_dim), nn.Softplus())
    
        critic_layer_size = [input_dim] + hidden_layer
        critic_layers = nn.ModuleList([])
        for i in range(len(critic_layer_size)-1):
            critic_layers.append(nn.Linear(critic_layer_size[i], critic_layer_size[i+1]))
            critic_layers.append(nn.ReLU())
        critic_layers.append(nn.Linear(hidden_layer[-1], 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, x, action=None):
        actor_features = self.actor(x)
        alpha = self.alpha_head(actor_features)+1
        beta = self.beta_head(actor_features)+1
        self.dist = Beta(alpha, beta)
        if action is None:
            action = self.dist.sample()
        else:
            action = (action+1)/2
        action_log_prob = self.dist.log_prob(action).sum(-1)
        entropy = self.dist.entropy().sum(-1)
        value = self.critic(x)
        return action*2-1, action_log_prob, value.squeeze(-1), entropy


class CarlaImgPolicy(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_layer=[400,300]):
        super(CarlaImgPolicy, self).__init__()
        self.main_actor = CarlaSimpleEncoder(latent_size = input_dim-1)
        self.main_critic = CarlaSimpleEncoder(latent_size = input_dim-1)
        actor_layer_size = [input_dim] + hidden_layer
        actor_feature_layers = nn.ModuleList([])
        for i in range(len(actor_layer_size)-1):
            actor_feature_layers.append(nn.Linear(actor_layer_size[i], actor_layer_size[i+1]))
            actor_feature_layers.append(nn.ReLU())
        self.actor = nn.Sequential(*actor_feature_layers)
        self.alpha_head = nn.Sequential(nn.Linear(hidden_layer[-1], action_dim), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(hidden_layer[-1], action_dim), nn.Softplus())
    
        critic_layer_size = [input_dim] + hidden_layer
        critic_layers = nn.ModuleList([])
        for i in range(len(critic_layer_size)-1):
            critic_layers.append(nn.Linear(critic_layer_size[i], critic_layer_size[i+1]))
            critic_layers.append(nn.ReLU())
        critic_layers.append(layer_init(nn.Linear(hidden_layer[-1], 1), gain=1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, x, action=None):
        speed = x[:, -1:]
        x = x[:, :-1].view(-1, 3,128,128)  # image size in carla driving task is 128x128
        x1 = self.main_actor(x)
        x1 = torch.cat([x1, speed], dim=1)

        x2 = self.main_critic(x)
        x2 = torch.cat([x2, speed], dim=1)

        actor_features = self.actor(x1)
        alpha = self.alpha_head(actor_features)+1
        beta = self.beta_head(actor_features)+1
        self.dist = Beta(alpha, beta)
        if action is None:
            action = self.dist.sample()
        else:
            action = (action+1)/2
        action_log_prob = self.dist.log_prob(action).sum(-1)
        entropy = self.dist.entropy().sum(-1)
        value = self.critic(x2)
        return action*2-1, action_log_prob, value.squeeze(-1), entropy