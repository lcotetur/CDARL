import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog, ActionDistribution
from ray.rllib.utils.annotations import override
from video import VideoRecorder
from utils import RandomTransform

import torch
import torch.nn as nn
from torch.distributions import Normal, Beta
from torch.distributions.kl import kl_divergence
from torchvision.utils import save_image

import gym
import cv2
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--deterministic-sample', default=False, action='store_true')
parser.add_argument('--env', default="CarRacing-v0", type=str)
parser.add_argument('--num-episodes', default=100, type=int)
parser.add_argument('--model-path', default='/home/mila/l/lea.cote-turcotte/LUSR/checkpoints/policy_train_v2_offline.pt', type=str)
parser.add_argument('--render', default=False, action='store_true')
parser.add_argument('--latent-size', default=32, type=int)
parser.add_argument('--save-path', default='/home/mila/l/lea.cote-turcotte/LUSR/checkpoints', type=str)
parser.add_argument('--action-repeat', default=4, type=int)
parser.add_argument('--save_video', default=True, action='store_true')
parser.add_argument('--work_dir', default='/home/mila/l/lea.cote-turcotte/LUSR', type=str)
parser.add_argument('--nb_domain', default=4, type=int)
args = parser.parse_args()


######## obs preprocess ###########
def process_obs(obs): # a single frame (96, 96, 3) for CarRacing
    obs = np.ascontiguousarray(obs, dtype=np.float32) / 255
    obs = cv2.resize(obs[:84, :, :], dsize=(64,64), interpolation=cv2.INTER_NEAREST)
    obs = np.transpose(obs, (2,0,1))
    return torch.from_numpy(obs).unsqueeze(0)

######## models ##########
'''
# Encoder download weights
class Encoder(nn.Module):
    def __init__(self, latent_size = 32, input_channel = 3):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )
        self.linear_mu = nn.Linear(2*2*256, latent_size)

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        return mu

'''

# Encoder online
#'''
class Encoder(nn.Module):
    def __init__(self, class_latent_size = 8, content_latent_size = 32, input_channel = 3, flatten_size = 1024):
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
#'''

class MyModel(nn.Module):
    def __init__(self, deterministic_sample=False, latent_size=16):
        nn.Module.__init__(self)

        self.main = Encoder(class_latent_size = 8, content_latent_size = 32, input_channel = 3, flatten_size = 1024)
        #self.main = Encoder(latent_size=latent_size)
        self.critic = nn.Sequential(nn.Linear(latent_size, 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, 1))
        self.actor = nn.Sequential(nn.Linear(latent_size, 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(300, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(300, 3), nn.Softplus())
        self.deterministic_sample = deterministic_sample

    def forward(self, x):
        with torch.no_grad():
            features = self.main(x)
            actor_features = self.actor(features)
            alpha = self.alpha_head(actor_features)+1
            beta = self.beta_head(actor_features)+1
        dist = Beta(alpha, beta)
        if not self.deterministic_sample:
            action = dist.sample().squeeze().numpy()
        else:
            action = dist.mean.squeeze().numpy()
        action[0] = action[0]*2-1
        return action


# TO DO : train an agent without any the encoder representation (on one domain) and evaluate on multiple domains 
        # train an agent with my representation (just content) and evaluate on multiple domains (unseen and seen)
        # train an agent with desentangle representation (so content and style) and evalutate on same domains 

        # Then assess wether it is better to learn an invariant representation or desentangle representation for domain adaptation (2 vs 3)
        # or even if learning those representation help for domain adaptation (1 vs 2 and 3)

########### Do Evaluation #################
def main():

    #video_dir = os.mkdir(os.path.join(args.work_dir, 'video'))
    video_dir = os.path.join(args.work_dir, 'video')
    video = VideoRecorder(video_dir if args.save_video else None)

    results = []
    env = gym.make(args.env)
    model = MyModel(args.deterministic_sample, args.latent_size)
    weights = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(weights)

    for domain in range(args.nb_domain):
        if domain == 0:
            color = None
        else:
            color = np.random.randint(-5, 5)/10
        print(color)
        domain_results = []
        for i in range(args.num_episodes):
            episode_reward, done, obs = 0, False, env.reset()
            obs = process_obs(obs)
            new_obs = RandomTransform(obs).domain_transformation(color)
            save_image(new_obs, "/home/mila/l/lea.cote-turcotte/LUSR/figures/obs_%d.png" % (domain))
            video.init(enabled=((i == 0) or (i == 99)))
            while not done:
                action = model(new_obs)
                for _ in range(args.action_repeat):
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward
                    if done:
                        break
                video.record(env)
                if args.render:
                    env.render()
                obs = process_obs(obs)
                new_obs = RandomTransform(obs).domain_transformation(color)
            video.save('%d_%d.mp4' % (domain, i))
            results.append(episode_reward)

        print('Evaluate %d episodes and achieved %f scores in domain %d' % (args.num_episodes, np.mean(results), domain))
        #file_name = "%s_%d_%s" % (args.env, args.num_episodes, 'results.txt')
        print(results)
        #torch.save(results, os.path.join(args.save_path, file_name))
        domain_results.append(results)
    with open(os.path.join(args.save_path, 'eval_train_v2_offline.py'), 'w') as f:
        f.write('score = %s' % domain_results)

if __name__ == '__main__':
    main()