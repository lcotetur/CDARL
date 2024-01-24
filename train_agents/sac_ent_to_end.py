import argparse

import numpy as np
import time

import gym
import torch
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import cv2
import tqdm
import os
from datetime import date
import warnings
from torchvision.utils import save_image
from CDARL.config.agents_config import parse_args
from CDARL.representation.ILCM.model import MLPImplicitSCM, HeuristicInterventionEncoder, ILCM
from CDARL.representation.ILCM.model import ImageEncoder, ImageDecoder, CoordConv2d
from CDARL.utils import seed_everything, Results, ReplayBuffer, random_shift, random_crop, random_conv, transform, create_logs, cat, RandomTransform

def evaluate(env, agent, num_episodes=1):
    episode_rewards = []
    for i in range(num_episodes):
        ep_agent = agent
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = ep_agent.select_action(obs)
            next_obs, reward, done, _ = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            episode_reward += reward
            obs = next_obs
        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards)

def updateloader(loader, dataset):
    dataset.loadnext()
    loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=10)
    return loader

def vae_loss(x, mu, logsigma, recon_x, beta=1):
    recon_loss = F.mse_loss(x, recon_x, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    kl_loss = kl_loss / torch.numel(x)
    return recon_loss + kl_loss * beta

def reparameterize(mu, logsigma):
    std = torch.exp(0.5*logsigma)
    eps = torch.randn_like(std)
    return mu + eps*std

def forward_loss(x, model, beta):
    mu, logsigma, classcode = model.encoder(x)
    contentcode = reparameterize(mu, logsigma)
    shuffled_classcode = classcode[torch.randperm(classcode.shape[0])]

    latentcode1 = torch.cat([contentcode, shuffled_classcode], dim=1)
    latentcode2 = torch.cat([contentcode, classcode], dim=1)

    recon_x1 = model.decoder(latentcode1)
    recon_x2 = model.decoder(latentcode2)
    return vae_loss(x, mu, logsigma, recon_x1, beta) + vae_loss(x, mu, logsigma, recon_x2, beta)

def backward_loss(x, model):
    mu, logsigma, classcode = model.encoder(x)
    shuffled_classcode = classcode[torch.randperm(classcode.shape[0])]
    randcontent = torch.randn_like(mu).cuda()

    latentcode1 = torch.cat([randcontent, classcode], dim=1)
    latentcode2 = torch.cat([randcontent, shuffled_classcode], dim=1)

    recon_imgs1 = model.decoder(latentcode1).detach()
    recon_imgs2 = model.decoder(latentcode2).detach()

    cycle_mu1, cycle_logsigma1, cycle_classcode1 = model.encoder(recon_imgs1)
    cycle_mu2, cycle_logsigma2, cycle_classcode2 = model.encoder(recon_imgs2)

    cycle_contentcode1 = reparameterize(cycle_mu1, cycle_logsigma1)
    cycle_contentcode2 = reparameterize(cycle_mu2, cycle_logsigma2)

    bloss = F.l1_loss(cycle_contentcode1, cycle_contentcode2)
    return bloss

class Env():
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self, seed):
        self.env = gym.make('CarRacing-v0')
        self.env.seed(seed)
        self.reward_threshold = self.env.spec.reward_threshold

    def process_obs(self, obs):
        obs = np.ascontiguousarray(obs, dtype=np.float32) / 255
        obs = cv2.resize(obs[:84, :, :], dsize=(64,64), interpolation=cv2.INTER_NEAREST)
        return np.transpose(obs, (2,0,1))

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        processed_obs = self.process_obs(img_rgb)
        self.stack = [processed_obs] * args.img_stack
        img = np.concatenate(self.stack, axis=0)
        save_image(torch.tensor(img[:3, :, :].copy(), dtype=torch.float32), "/home/mila/l/lea.cote-turcotte/CDARL/checkimages/obs.png", nrow=12)
        return img

    def step(self, action):
        total_reward = 0
        for i in range(args.action_repeat):
            img_rgb, reward, die, _ = self.env.step(action)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_rgb = self.process_obs(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_rgb)
        assert len(self.stack) == args.img_stack
        img_rgb = np.concatenate(self.stack, axis=0)
        return img_rgb, total_reward, done, die

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory

def ilcm():
    # Create model
    scm = create_scm()
    encoder, decoder = create_encoder_decoder()
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

def create_scm():
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

def create_encoder_decoder():
    encoder = ImageEncoder(
            in_resolution=63,
            in_features=3,
            out_features=args.latent_size,
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
            in_features=args.latent_size,
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

def create_intervention_encoder():
    intervention_encoder = HeuristicInterventionEncoder()
    return intervention_encoder

class Encoder(nn.Module):
    def __init__(self, input_channel=3):
        super(Encoder, self).__init__()
        self.cnn_base = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )
        self.out_dim = 2*2*256
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(x.size(0), -1)

        return x

class SodaEncoder(nn.Module):
    def __init__(self, projection_dim = 2*2*256, input_channel=12, hidden_dim = 64, out_dim = 64):
        super(Encoder, self).__init__()
        self.out_dim = out_dim
        self.cnn_base = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.cnn_base.apply(self._weights_init)
        self.mlp.apply(weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)

        return x

class VAE(nn.Module):
    def __init__(self, latent_size = 32, input_channel = 12, flatten_size = 1024):
        super(VAE, self).__init__()
        self.latent_size = latent_size

        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )

        self.linear_mu = nn.Linear(flatten_size, latent_size)

    def forward(self, x):
        x = self.main(x)
        x = x.reshape(x.size(0), -1)
        mu = self.linear_mu(x)
        return mu


class CycleVAE(nn.Module):
    def __init__(self, class_latent_size = 8, content_latent_size = 32, input_channel = 9, flatten_size = 1024):
        super(CycleVAE, self).__init__()
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


class AdagVAE(nn.Module):
    def __init__(self, latent_size = 32, input_channel = 3, flatten_size = 1024):
        super(AdagVAE, self).__init__()
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


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability"""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function, see appendix C from https://arxiv.org/pdf/1812.05905.pdf"""
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class Encoder(nn.Module):
    def __init__(self, class_latent_size = 8, content_latent_size = 32, input_channel = 12, flatten_size = 1024):
        super(Encoder, self).__init__()
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
        self.out_dim = content_latent_size

        self.apply(self._weights_init)
        
    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        logsigma = self.linear_logsigma(x)
        classcode = self.linear_classcode(x)

        return mu, logsigma, classcode

    def get_feature(self, x):
        mu, logsigma, classcode = self.forward(x)
        return mu


class Decoder(nn.Module):
    def __init__(self, latent_size=32, output_channel=12, flatten_size=1024):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(latent_size, flatten_size)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(flatten_size, 128, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, output_channel, 6, stride=2), nn.Sigmoid()
        )
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.main(x)
        return x

class DisentangledVAE(nn.Module):
    def __init__(self, class_latent_size = 8, content_latent_size = 32, input_channel = 12, flatten_size = 1024):
        super(DisentangledVAE, self).__init__()
        self.encoder = Encoder(class_latent_size, content_latent_size, input_channel, flatten_size)
        self.decoder = Decoder(class_latent_size + content_latent_size, input_channel, flatten_size)

    def forward(self, x):
        mu, logsigma, classcode = self.encoder(x)
        contentcode = reparameterize(mu, logsigma)
        latentcode = torch.cat([contentcode, classcode], dim=1)

        recon_x = self.decoder(latentcode)

        return mu, logsigma, classcode, recon_x

class Actor(nn.Module):
    def __init__(self, args, encoder, obs_chape, action_shape, hidden_dim, log_std_min, log_std_max):
        super().__init__()
        self.encoder = encoder
        out_dim = self.encoder.out_dim

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )
        self.mlp.apply(weight_init)

    def forward(self, x, compute_pi=True, compute_log_pi=True):
        with torch.no_grad():
            x, _, _ = self.encoder(x)

        mu, log_std = self.mlp(x).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std


class QFunction(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        return self.trunk(torch.cat([obs, action], dim=1))

class Critic(nn.Module):
    def __init__(self, args, encoder, obs_shape, action_shape, hidden_dim):
        super().__init__()
        self.encoder = encoder
        out_dim = self.encoder.out_dim

        self.Q1 = QFunction(out_dim, action_shape[0], hidden_dim)
        self.Q2 = QFunction(out_dim, action_shape[0], hidden_dim)

    def forward(self, x, action):
        with torch.no_grad():
            x, _, _ = self.encoder(x)

        return self.Q1(x, action), self.Q2(x, action)


class SAC(object):

    def __init__(self, args, obs_shape, action_shape):
        self.max_grad_norm = args.max_grad_norm
        self.clip_param = args.clip_param  # epsilon in clipped loss
        self.aux_update_freq = 1
        self.actor_update_freq = args.actor_update_freq
        self.critic_target_update_freq = args.critic_target_update_freq
        self.discount = args.discount
        self.critic_tau = args.critic_tau
        self.encoder_tau = args.encoder_tau
        self.soda_tau = args.soda_tau
        self.aux_beta = args.aux_beta
        self.img_stack = args.img_stack
        self.vae_batchsize = args.vae_batch_size
        self.training_step = 0
        self.vae_batches = 0
        self.vae_epoch = 0

        self.model = DisentangledVAE(input_channel=obs_shape[0]).cuda()
        self.actor = Actor(args, self.model.encoder, obs_shape, action_shape, 1024, -10, 2).cuda()
        self.critic = Critic(args, self.model.encoder, obs_shape, action_shape, 1024).cuda()
        self.critic_target = deepcopy(self.critic)
        self.log_alpha = torch.tensor(np.log(0.1)).cuda()
        self.log_alpha.requires_grad = True

        self.target_entropy = -np.prod(action_shape)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999))
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999))
        self.cycle_vae_optimizer = optim.Adam(self.model.parameters(), lr=args.vae_lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.model.train(training)
        if hasattr(self, 'pad_head'):
            self.pad_head.train(training)
        if hasattr(self, 'soda_predictor'):
            self.soda_predictor.train(training)

    def eval(self):
        self.train(False)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        _obs = torch.tensor(obs).unsqueeze(0).cuda()
        with torch.no_grad():
            mu, _, _, _ = self.actor(_obs, compute_pi=False, compute_log_pi=False)      
        return mu.detach().squeeze().cpu().numpy()
      
    def sample_action(self, obs):
        _obs = torch.tensor(obs).unsqueeze(0).cuda()
        with torch.no_grad():
            mu, pi, _, _ = self.actor(_obs, compute_log_pi=False)
        return pi.detach().squeeze().cpu().numpy()

    def soft_update_critic_target(self):
        self.soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
        self.soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
        self.soft_update_params(self.critic.encoder, self.critic_target.encoder,self.encoder_tau)
        
    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
    def update(self, replay_buffer, algo, logdir, num_updates):
        self.training_step += 1
        for _ in range(num_updates):
            obs, action, reward, next_obs, not_done = replay_buffer.sample_sac()
                
            # update critic SAC
            with torch.no_grad():
                _, policy_action, log_pi, _ = self.actor(next_obs)
                target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
                target_V = torch.min(target_Q1,target_Q2) - self.alpha.detach() * log_pi                
                target_Q = reward + (self.discount * target_V)
                
            current_Q1, current_Q2 = self.critic(obs, action)
            critic_loss = F.mse_loss(current_Q1,target_Q) + F.mse_loss(current_Q2, target_Q)
                    
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
                
            # update actor
            if self.training_step % self.actor_update_freq == 0:
                _, pi, log_pi, log_std = self.actor(obs)
                actor_Q1, actor_Q2 = self.critic(obs, pi)
                    
                actor_Q = torch.min(actor_Q1, actor_Q2)
                actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()
                    
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                    
                self.log_alpha_optimizer.zero_grad()
                alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
                    
                alpha_loss.backward()
                self.log_alpha_optimizer.step()
                
            # update target critic
            if self.training_step % self.critic_target_update_freq == 0:
                self.soft_update_critic_target()

            # update aux task
            if self.training_step % self.aux_update_freq == 0 and self.vae_batches < 100000:
                print('update cycle vae')
                for epoch in range(200):
                    self.vae_batches += 1

                    obs, _, _, _, _ = replay_buffer.sample_sac(n=100)
                    save_image(obs[:, :3, :, :], os.path.join(log_dir, "epoch_%d.png" % epoch), nrow=10)
                    imgs = RandomTransform(obs).apply_transformations_stack(num_frames=self.img_stack, nb_class=5, value=0.3)

                    floss = 0
                    for i_class in range(imgs.shape[0]):
                        image = imgs[i_class]
                        floss += forward_loss(image, self.model, 10)#args.beta
                    floss = floss / imgs.shape[0] # divided by the number of classes

                    # backward circle
                    imgs = imgs.reshape(-1, *imgs.shape[2:]) 
                    bloss = backward_loss(imgs, self.model)

                    self.cycle_vae_optimizer.zero_grad()
                    (floss + bloss * args.bloss_coef).backward()
                    self.cycle_vae_optimizer.step()

                    loss = floss + bloss * args.bloss_coef
                    # save image to check and save model 
                    if self.vae_batches % 4000 == 0:
                        print(f'{self.vae_epoch} Epoch ------ {self.vae_batches} Batches is Done ------ {loss.item()} loss')
                        rand_idx = torch.randperm(imgs.shape[0])
                        imgs1 = imgs[rand_idx[:9]]
                        imgs2 = imgs[rand_idx[-9:]]
                        with torch.no_grad():
                            mu, _, classcode1 = self.model.encoder(imgs1)
                            _, _, classcode2 = self.model.encoder(imgs2)
                            recon_imgs1 = self.model.decoder(torch.cat([mu, classcode1], dim=1))
                            recon_combined = self.model.decoder(torch.cat([mu, classcode2], dim=1))

                        saved_imgs = torch.cat([imgs1, imgs2, recon_imgs1, recon_combined], dim=0)
                        save_image(saved_imgs[:, :3, :, :], os.path.join(log_dir, "stack%d_%d.png" % (self.vae_epoch, self.vae_batches)), nrow=9)
                        torch.save(self.model.encoder.state_dict(), os.path.join(log_dir, "encoder.pt"))
                self.vae_epoch += 1




if __name__ == "__main__":
    args = parse_args()
    print(args.algo)
    if args.algo in ['soda_sac', 'pad_sac']:
        print('not tested yet')
    warnings.filterwarnings("ignore")
    seed_everything(args.seed)
    log_dir = create_logs(args, algo_name=True, repr=args.repr)
    print(log_dir)

    results_sac = Results(title="Moving averaged episode reward", xlabel="episode", ylabel="sac_running_score")
    results_sac.create_logs(labels=["episode", "episode_step", "training_step", "sac_running_score", "score", "training_time", "eval_score"], init_values=[[], [], [], [], [], [], []])
    env = Env(args.seed)
    test_env = Env(args.seed + 1)
    action_shape = (3,)
    obs_shape = (args.img_stack*3, 64, 64)
    agent = SAC(args, obs_shape, action_shape)
    replay_buffer = ReplayBuffer(
        obs_shape=obs_shape,
        action_shape=action_shape,
        capacity=100000,
        batch_size=128
    )

    sac_running_score = 0
    training_time = 0
    state = env.reset()
    start_time = time.time()
    # 100000
    episode, episode_step, score, done, die = 0, 0, 0, True, True
    train_steps = args.training_step
    for step in range(train_steps+1):
        if done or die:
            if step > 1000:
                num_updates = 1
                print('update agent')
                agent.update(replay_buffer, args.algo, log_dir, num_updates)
            
            sac_running_score = sac_running_score * 0.99 + score * 0.01

            if episode % args.log_interval == 0:
                training_time = time.time() - start_time
                #eval_score = evaluate(test_env, agent)
                eval_score = 0
                results_sac.update_logs(["episode", "episode_step", "training_step", "sac_running_score", "score", "training_time", "eval_score"], [episode, episode_step, step, sac_running_score, score, training_time, eval_score])
                print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(episode, score, sac_running_score))
                print('Training time: {:.2f}\t'.format(training_time))
                print('Step: {:.2f}\t'.format(step))
                print('Eval score: {:.2f}\t'.format(eval_score))
                torch.save(agent, os.path.join(log_dir, "policy.pt"))
                results_sac.save_logs(log_dir)
                results_sac.generate_plot(log_dir, log_dir)

            obs = env.reset()
            done = False
            score = 0
            episode_lenght = 0
            episode_step = 0
            episode += 1

        # Take step
        action = agent.sample_action(obs)
        next_obs, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
        done_bool = 0 if episode_step + 1 == 1000 else float(done)
        replay_buffer.add(obs, action, reward, next_obs, done_bool)
        score += reward
        obs = next_obs
        episode_step += 1

        if episode > 10000:
            results_sac.save_logs(log_dir)
            results_sac.generate_plot(log_dir, log_dir)
            break

    print(step)
    results_sac.save_logs(log_dir)
    results_sac.generate_plot(log_dir, log_dir)