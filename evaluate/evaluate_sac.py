import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog, ActionDistribution
from ray.rllib.utils.annotations import override
from video import VideoRecorder
from CDARL.utils import RandomTransform
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Beta
from torch.distributions.kl import kl_divergence
from torchvision.utils import save_image

import gym
import cv2
import os
import numpy as np
import argparse
import warnings
import time

parser = argparse.ArgumentParser()
parser.add_argument('--algo', default='drq_sac', type=str)
parser.add_argument('--repr', default=None, type=str)
parser.add_argument('--deterministic-sample', default=False, action='store_true')
parser.add_argument('--env', default="CarRacing-v0", type=str)
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--num-episodes', default=100, type=int)
parser.add_argument('--model-path', default='/home/mila/l/lea.cote-turcotte/CDARL/logs/drq_sac/2023-12-20_0/policy_sac_stack.pt', type=str)
parser.add_argument('--render', default=False, action='store_true')
parser.add_argument('--latent-size', default=16, type=int)
parser.add_argument('--save-path', default='/home/mila/l/lea.cote-turcotte/CDARL/results', type=str)
parser.add_argument('--action-repeat', default=4, type=int)
parser.add_argument('--img-stack', type=int, default=3, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--save_video', default=True, action='store_true')
parser.add_argument('--work_dir', default='/home/mila/l/lea.cote-turcotte/CDARL', type=str)
parser.add_argument('--nb_domain', default=4, type=int)
args = parser.parse_args()

def process_obs(obs):
    obs = np.ascontiguousarray(obs, dtype=np.float32) / 255
    obs = cv2.resize(obs[:84, :, :], dsize=(64,64), interpolation=cv2.INTER_NEAREST)
    obs = np.transpose(obs, (2,0,1))
    return obs

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.cnn_base = nn.Sequential(
            nn.Conv2d(9, 32, 4, stride=2), nn.ReLU(),
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
    def __init__(self, projection_dim = 2*2*256, hidden_dim = 64, out_dim = 64):
        super(Encoder, self).__init__()
        self.out_dim = out_dim
        self.cnn_base = nn.Sequential(
            nn.Conv2d(9, 32, 4, stride=2), nn.ReLU(),
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
    def __init__(self, latent_size = 32, input_channel = 9, flatten_size = 1024):
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

class Actor(nn.Module):
    def __init__(self, args, action_shape, hidden_dim, log_std_min, log_std_max):
        super().__init__()
        self.repr = args.repr
        if self.repr == None:
            self.encoder = Encoder()
            out_dim = self.encoder.out_dim
        elif self.repr == 'vae_sac':
            self.encoder = VAE()
            weights = torch.load(args.encoder_path, map_location=torch.device('cuda'))
            for k in list(weights.keys()):
                if k not in self.encoder.state_dict().keys():
                    del weights[k]
            self.encoder.load_state_dict(weights)
            print("Loaded Weights")
            out_dim = 32
        elif self.repr == 'cycle_vae_sac':
            self.encoder = CycleVAE()
            weights = torch.load(args.encoder_path, map_location=torch.device('cuda'))
            for k in list(weights.keys()):
                if k not in self.encoder.state_dict().keys():
                    del weights[k]
            self.encoder.load_state_dict(weights)
            print("Loaded Weights")
            out_dim = 32

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )
        self.mlp.apply(weight_init)

    def forward(self, x, compute_pi=True, compute_log_pi=True):
        if self.repr == 'vae_sac':
            x = self.encoder(x)
            x = x.detach()
        elif self.repr == 'cycle_vae_sac':
            x, _, _ = self.encoder(x)
            x = x.detach()
        elif self.repr == None:
            x = self.encoder(x)

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
    def __init__(self, args, action_shape, hidden_dim):
        super().__init__()
        self.repr = args.repr
        if self.repr == None:
            self.encoder = Encoder()
            out_dim = self.encoder.out_dim
        elif self.repr == 'vae_sac':
            self.encoder = VAE()
            weights = torch.load(args.encoder_path, map_location=torch.device('cuda'))
            for k in list(weights.keys()):
                if k not in self.encoder.state_dict().keys():
                    del weights[k]
            self.encoder.load_state_dict(weights)
            print("Loaded Weights")
            out_dim = 32
        elif self.repr == 'cycle_vae_sac':
            self.encoder = CycleVAE()
            weights = torch.load(args.encoder_path, map_location=torch.device('cuda'))
            for k in list(weights.keys()):
                if k not in self.encoder.state_dict().keys():
                    del weights[k]
            self.encoder.load_state_dict(weights)
            print("Loaded Weights")
            out_dim = 32

        self.Q1 = QFunction(out_dim, action_shape[0], hidden_dim)
        self.Q2 = QFunction(out_dim, action_shape[0], hidden_dim)

    def forward(self, x, action):
        if self.repr == 'vae_sac':
            x = self.encoder(x)
            x = x.detach()
        elif self.repr == 'cycle_vae_sac':
            x, _, _ = self.encoder(x)
            x = x.detach()
        elif self.repr == None:
            x = self.encoder(x)

        return self.Q1(x, action), self.Q2(x, action)

class CURLHead(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.W = nn.Parameter(torch.rand(encoder.out_dim, encoder.out_dim))

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

class InverseDynamics(nn.Module):
    def __init__(self, encoder, action_shape, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(2*encoder.out_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_shape[0])
        )
        self.apply(weight_init)

    def forward(self, x, x_next):
        h = self.encoder(x)
        h_next = self.encoder(x_next)
        joint_h = torch.cat([h, h_next], dim=1)
        return self.mlp(joint_h)

class SODAPredictor(nn.Module):
    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.mlp = SODAMLP(
            encoder.out_dim, hidden_dim, encoder.out_dim
        )
        self.apply(weight_init)

    def forward(self, x):
        return self.mlp(self.encoder(x))

def compute_soda_loss(self, x0, x1):
    h0 = self.predictor(x0)
    with torch.no_grad():
        h1 = self.predictor_target.encoder(x1)
    h0 = F.normalize(h0, p=2, dim=1)
    h1 = F.normalize(h1, p=2, dim=1)

    return F.mse_loss(h0, h1)


class SAC(object):
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    sac_epoch = 10
    aux_update_freq = 2
    actor_update_freq = 2
    critic_target_update_freq = 2

    def __init__(self, args, action_shape):
        self.discount = 0.99
        self.critic_tau = 0.01
        self.encoder_tau = 0.05
        self.critic_target_update_freq = 2
        self.counter = 0
        self.training_step = 0

        self.actor = Actor(args, action_shape, 1024, -10, 2).cuda()
        self.critic = Critic(args, action_shape, 1024).cuda()
        self.critic_target = deepcopy(self.critic)
        self.log_alpha = torch.tensor(np.log(0.1)).cuda()
        self.log_alpha.requires_grad = True

        self.target_entropy = -np.prod(action_shape)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4, betas=(0.9, 0.999))

        if args.algo == 'curl_sac':
            self.curl_head = CURLHead(self.critic.encoder).cuda()
            self.curl_optimizer = torch.optim.Adam(self.curl_head.parameters(), lr=1e-3, betas=(0.9, 0.999))

        if args.algo == 'soda_sac':
            soda_encoder = SodaEncoder()
            self.predictor = m.SODAPredictor(soda_encoder, 64).cuda()
            self.predictor_target = deepcopy(self.predictor)

            self.soda_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=args.aux_lr, betas=(args.aux_beta, 0.999))

        if args.algo == 'pad_sac':
            aux_encoder = Encoder()
            self.pad_head = InverseDynamics(aux_encoder, action_shape, args.hidden_dim).cuda()
            self.pad_optimizer = torch.optim.Adam(self.pad_head.parameters(), lr=self.aux_lr, betas=(self.aux_beta, 0.999))

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
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

    def update_soda(self, replay_buffer, L=None, step=None):
        x = replay_buffer.sample_soda(256)
        assert x.size(-1) == 64

        aug_x = x.clone()

        x = augmentations.random_crop(x)
        aug_x = augmentations.random_crop(aug_x)
        aug_x = augmentations.random_overlay(aug_x)

        soda_loss = self.compute_soda_loss(aug_x, x)
        
        self.soda_optimizer.zero_grad()
        soda_loss.backward()
        self.soda_optimizer.step()

        self.soft_update_params(self.predictor, self.predictor_target, 0.005)
            
    def update(self, replay_buffer, algo='sac'):
        self.training_step += 1
        for _ in range(self.sac_epoch):

            if algo == 'drq_sac':
                obs, action, reward, next_obs, not_done = replay_buffer.sample_drq()
            elif algo == 'sac':
                obs, action, reward, next_obs, not_done = replay_buffer.sample_sac()
            elif algo == 'svea_sac':
                obs, action, reward, next_obs, not_done = replay_buffer.sample_svea()
            elif algo == 'pad_sac' or algo == 'soda_sac' or algo == 'rad_sac':
                obs, action, reward, next_obs, not_done = replay_buffer.sample()
            elif algo == 'curl':
                obs, action, reward, next_obs, not_done, pos = replay_buffer.sample_curl()
                
            # update critic SAC
            if algo != 'svea_sac':
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
            # update critic SVEA
            elif algo == 'svea_sac':
                svea_alpha = 0.5
                svea_beta = 0.5

                with torch.no_grad():
                    _, policy_action, log_pi, _ = self.actor(next_obs)
                    target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
                    target_V = torch.min(target_Q1,target_Q2) - self.alpha.detach() * log_pi
                    target_Q = reward + (not_done * self.discount * target_V)

                obs = cat(obs, random_conv(obs.clone()))
                action = cat(action, action)
                target_Q = cat(target_Q, target_Q)

                current_Q1, current_Q2 = self.critic(obs, action)
                critic_loss = (svea_alpha + svea_beta) * \
                        (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
                        
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
            if self.training_step % self.aux_update_freq == 0:
                if algo == 'soda_sac':
                    self.update_soda(replay_buffer)
                elif algo == 'pad_sac':
                    assert obs.shape[-1] == 64 and next_obs.shape[-1] == 64

                    pred_action = self.pad_head(obs, next_obs)
                    pad_loss = F.mse_loss(pred_action, action)

                    self.pad_optimizer.zero_grad()
                    pad_loss.backward()
                    self.pad_optimizer.step()
                elif algo == 'curl_sac':
                    assert obs.size(-1) == 64 and pos.size(-1) == 64

                    z_a = self.curl_head.encoder(obs)
                    with torch.no_grad():
                        z_pos = self.critic_target.encoder(pos)
                    
                    logits = self.curl_head.compute_logits(z_a, z_pos)
                    labels = torch.arange(logits.shape[0]).long().cuda()
                    curl_loss = F.cross_entropy(logits, labels)
                    
                    self.curl_optimizer.zero_grad()
                    curl_loss.backward()
                    self.curl_optimizer.step()


########### Do Evaluation #################
def main():
    warnings.filterwarnings("ignore")
    video_dir = os.path.join(args.work_dir, 'video')
    video = VideoRecorder(video_dir if args.save_video else None)

    env = gym.make(args.env)
    env.seed(args.seed)
    action_shape = (3,)
    agent = SAC(args, action_shape)
    agent = torch.load(args.model_path, map_location=torch.device('cuda'))
    agent.train(False)

    domains_results = []
    domains_means = []
    blur = False

    for domain in range(args.nb_domain):
        if domain == 0:
            color = None
        elif domain == args.nb_domain - 1:
            blur = True
        else:
            color = np.random.randint(4, 5)/10
        print(color, blur)
        results = []
        for i in range(args.num_episodes):
            episode_reward, done, obs = 0, False, env.reset()
            obs = process_obs(obs)
            new_obs = RandomTransform(torch.tensor(obs)).domain_transformation(color, blur)
            stack = [np.array(new_obs)] * args.img_stack
            new_obs = np.concatenate(stack, axis=0)
            video.init(enabled=((i == 0) or (i == 99)))
            while not done:
                obs = torch.from_numpy(new_obs).float()
                action = agent.select_action(obs)
                for _ in range(args.action_repeat):
                    obs, reward, done, info = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
                    obs = process_obs(obs)
                    t_obs = RandomTransform(torch.tensor(obs)).domain_transformation(color, blur)
                    save_image(t_obs, "/home/mila/l/lea.cote-turcotte/CDARL/figures/obs_%d.png" % (domain))
                    stack.pop(0)
                    stack.append(np.array(t_obs))
                    assert len(stack) == args.img_stack
                    new_obs = np.concatenate(stack, axis=0)
                    episode_reward += reward
                    if done:
                        break
                video.record(env)
                if args.render:
                    env.render()
            video.save('%d_%d.mp4' % (domain, i))
            results.append(episode_reward)

        print('Evaluate %d episodes and achieved %f scores in domain %d' % (args.num_episodes, np.mean(results), domain))
        print(results)
        domains_results.append(results)
        domains_means.append(np.mean(results))
    with open(os.path.join(args.save_path, 'results_%s.txt' % args.algo), 'w') as f:
        f.write('score = %s' % domains_results)
        f.write('mean_scores = %s' % domains_means)

if __name__ == '__main__':
    main()