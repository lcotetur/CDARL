import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog, ActionDistribution
from ray.rllib.utils.annotations import override
from video import VideoRecorder
from CDARL.utils import RandomTransform, seed_everything
import warnings

import torch
import torch.nn as nn
from torch.distributions import Normal, Beta
from torch.distributions.kl import kl_divergence
from torchvision.utils import save_image

from CDARL.representation.ILCM.model import MLPImplicitSCM, HeuristicInterventionEncoder, ILCM
from CDARL.representation.ILCM.model import ImageEncoder, ImageDecoder, CoordConv2d, GaussianEncoder

import gym
import cv2
import os
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--policy-type', default='ilcm', type=str)
parser.add_argument('--deterministic-sample', default=False, action='store_true')
parser.add_argument('--env', default="CarRacing-v0", type=str)
parser.add_argument('--num-episodes', default=100, type=int)
parser.add_argument('--model-path', default='/home/mila/l/lea.cote-turcotte/CDARL/checkpoints/policy_ilcm_0.pt', type=str)
parser.add_argument('--render', default=False, action='store_true')
parser.add_argument('--latent-size', default=8, type=int)
parser.add_argument('--save-path', default='/home/mila/l/lea.cote-turcotte/CDARL/results', type=str)
parser.add_argument('--action-repeat', default=4, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--save_video', default=True, action='store_true')
parser.add_argument('--work_dir', default='/home/mila/l/lea.cote-turcotte/CDARL', type=str)
parser.add_argument('--nb_domain', default=4, type=int)
args = parser.parse_args()


######## obs preprocess ###########
def process_obs(obs): # input single frame (96, 96, 3) for CarRacing
    obs = np.ascontiguousarray(obs, dtype=np.float32) / 255
    obs = cv2.resize(obs[:84, :, :], dsize=(64,64), interpolation=cv2.INTER_NEAREST)
    obs = np.transpose(obs, (2,0,1))
    return torch.from_numpy(obs).unsqueeze(0)

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


######## models ##########
# Encoder download weights invar
class EncoderI(nn.Module):
    def __init__(self, latent_size = 32, input_channel = 3):
        super(EncoderI, self).__init__()
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

# Encoder download weights disent
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

class MyModel(nn.Module):
    def __init__(self, policy_type, deterministic_sample=False, latent_size=16):
        nn.Module.__init__(self)
        self.policy_type = policy_type

        # evaluate policy with end-to-end training
        if self.policy_type == 'end_to_end':
            latent_size = 16
            self.main = EncoderE(class_latent_size = 8, content_latent_size = 16, input_channel = 3, flatten_size = 1024)
        
        # evaluate policy invariant representation
        elif self.policy_type == 'invar':
            latent_size = 32
            self.main = EncoderI(latent_size=latent_size)

        # evaluate policy entangle representation
        elif self.policy_type == 'repr':
            latent_size = 32
            self.main = Encoder(latent_size=latent_size)

        # evaluate policy invariant representation
        elif self.policy_type == 'adagvae':
            latent_size = 16
            self.main = Encoder(latent_size=latent_size)

        # evaluate policy disentangled representation
        elif self.policy_type == 'disent':
            class_latent_size = 8
            content_latent_size = 32
            latent_size = class_latent_size + content_latent_size
            self.main = EncoderD(class_latent_size, content_latent_size)

        # evaluate representation ilcm
        elif self.policy_type == 'ilcm':
            latent_size = 6
            self.encoder = create_model_reduce_dim()
            self.main = create_ilcm()
    
        # evaluate policy no encoder
        elif self.policy_type == 'ppo' or self.policy_type == 'augm':
            latent_size = 2*2*256
            self.cnn_base = nn.Sequential(
                    nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
                    nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                    nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
                    nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
                )

        self.critic = nn.Sequential(nn.Linear(latent_size, 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, 1))
        self.actor = nn.Sequential(nn.Linear(latent_size, 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(300, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(300, 3), nn.Softplus())
        self.deterministic_sample = deterministic_sample

    def forward(self, x):
        with torch.no_grad():
            if self.policy_type == 'ppo' or self.policy_type == 'augm':
                x = self.cnn_base(x)
                features = x.view(x.size(0), -1)
            elif self.policy_type == 'disent':
                content, _, style = self.main(x)
                features = torch.cat([content, style], dim=1)
            elif self.policy_type == 'repr' or self.policy_type == 'adagvae':
                features, _, = self.main(x)
            elif self.policy_type == 'ilcm':
                z, _ = self.encoder.encoder.mean_std(x)
                features = self.main.encode_to_causal(z)
            else:
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
    seed_everything(args.seed)
    warnings.filterwarnings("ignore")

    video_dir = os.path.join(args.work_dir, 'video')
    video = VideoRecorder(video_dir if args.save_video else None)

    env = gym.make(args.env)
    env.seed(args.seed)
    model = MyModel(args.policy_type, args.deterministic_sample, args.latent_size)
    weights = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(weights)

    domains_results = []
    domains_means = []
    crop = False

    for domain in range(args.nb_domain):
        if domain == 0:
            color = None
        elif domain == args.nb_domain - 1:
            crop = True
        elif domain == args.nb_domain - 2:
            color = -0.2
        else:
            color = np.random.randint(3, 6)/10
        print(color, crop)
        results = []
        for i in range(args.num_episodes):
            episode_reward, done, obs = 0, False, env.reset()
            obs = process_obs(obs)
            new_obs = RandomTransform(obs).domain_transformation(color, crop)
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
                new_obs = RandomTransform(obs).domain_transformation(color, crop)
                save_image(new_obs, "/home/mila/l/lea.cote-turcotte/CDARL/checkimages/obs_%d.png" % (domain))
            video.save('%d_%s.mp4' % (domain, args.policy_type))
            results.append(episode_reward)

        print('Evaluate %d episodes and achieved %f scores in domain %d' % (args.num_episodes, np.mean(results), domain))
        print(results)
        domains_results.append(results)
        domains_means.append(np.mean(results))
    with open(os.path.join(args.save_path, 'results_%s.txt' % args.policy_type), 'w') as f:
        f.write('score = %s\n' % domains_results)
        f.write('mean_scores = %s\n' % domains_means)
        f.write('model = %s\n' % args.model_path)

if __name__ == '__main__':
    main()