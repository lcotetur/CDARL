import torch
import torch.nn as nn
import warnings
from torch.distributions import Normal, Beta
from torch.distributions.kl import kl_divergence
from torchvision.utils import save_image

import gym
from gym import spaces
import cv2
import os
import numpy as np
import argparse
import time

#from video import VideoRecorder
from CDARL.representation.VAE.vae import Encoder
from CDARL.representation.CYCLEVAE.cycle_vae import EncoderD
from CDARL.utils import seed_everything, RandomTransform
from CDARL.representation.ILCM.model import MLPImplicitSCM, HeuristicInterventionEncoder, ILCM
from CDARL.representation.ILCM.model import ImageEncoder, ImageDecoder, CoordConv2d, GaussianEncoder, Encoder3dshapes, Decoder3dshapes

parser = argparse.ArgumentParser()
parser.add_argument('--algo', default='clean_rl', type=str)
parser.add_argument('--env_name', default="CarRacing-v0", type=str)
parser.add_argument('--repr', default='ilcm', type=str)
parser.add_argument('--number', default=0, type=int)
parser.add_argument('--use_encoder', default=True, action='store_true')
parser.add_argument('--model_path', default='/home/mila/l/lea.cote-turcotte/CDARL/carracing_logs/ilcm/2024-03-21_0/policy.pt', type=str)
parser.add_argument('--encoder_path', default='/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carracing/2024-03-17/model_reduce_dim_step_400000.pt', type=str)
parser.add_argument('--ilcm_path', default='/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carracing/2024-03-19_ilcm/model.pt', type=str)
parser.add_argument('--ilcm_encoder_type', default='conv', type=str)
parser.add_argument('--save_path', default='/home/mila/l/lea.cote-turcotte/CDARL/results/carracing', type=str)
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--num_episodes', default=100, type=int)
parser.add_argument('--latent_size', default=6, type=int)
parser.add_argument('--reduce_dim_latent_size', default=16, type=int)
parser.add_argument('--action_repeat', default=4, type=int)
parser.add_argument('--img_stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--save_video', default=False, action='store_true')
parser.add_argument('--work_dir', default='/home/mila/l/lea.cote-turcotte/CDARL', type=str)
parser.add_argument('--nb_domain', default=4, type=int)
args = parser.parse_args()

class Env():
    """
    Environment wrapper for CarRacing 
    """
    def __init__(self, env, device, encoder=None, causal=None):
        self.env = env
        self.env.seed(args.seed)
        self.img_stack = args.img_stack
        self.reward_threshold = self.env.spec.reward_threshold
        self.encoder = encoder
        self.causal = causal
        self.device = device
        if self.encoder:
            self.encoder = self.encoder.to(self.device)
        if self.causal:
            self.causal = self.causal.to(self.device)
        if self.encoder:
            self.observation_space = spaces.Box(low=-1000, high=1000, shape=(args.latent_size*self.img_stack,), dtype=np.float64)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(3*self.img_stack,64,64), dtype=np.float64)
        self.action_space = self.env.action_space

    def process_state(self, obs, color, crop, domain):
        obs = np.ascontiguousarray(obs, dtype=np.float32) / 255
        obs = cv2.resize(obs[:84, :, :], dsize=(64,64), interpolation=cv2.INTER_NEAREST)
        obs = np.transpose(obs, (2,0,1))
        obs = RandomTransform(torch.tensor(obs)).domain_transformation(color, crop)
        if self.encoder is None:
            return obs #(3, 64, 64)
        else:
            obs = np.expand_dims(obs, axis=0) #(1, 3, 64, 64)
            save_image(torch.tensor(obs), os.path.join('/home/mila/l/lea.cote-turcotte/CDARL/checkimages', "domain_%s.png" % domain))
            obs = torch.from_numpy(obs).float().to(self.device)
            if args.repr == 'cycle_vae' or args.repr == 'vae' or args.repr == 'adagvae':
                with torch.no_grad():
                    state, _ = self.encoder(obs)
                    state = state.cpu().squeeze().numpy()
            elif args.repr == 'ilcm':
                with torch.no_grad():
                    z, _ = self.encoder.encoder.mean_std(obs)
                    state = self.causal.encode_to_causal(z).cpu().squeeze().numpy()
            elif args.repr == 'disent':
                with torch.no_grad():
                    content, _, style = self.encoder(obs)
                    state = torch.cat([content, style], dim=1).cpu().squeeze().numpy()
        return state

    def reset(self, color, crop, domain):
        self.die = False
        img_rgb = self.env.reset()
        processed_obs = self.process_state(img_rgb, color, crop, domain)
        self.stack = [processed_obs] * self.img_stack
        state = np.concatenate(self.stack, axis=0)
        return state

    def step(self, action, color, crop, domain):
        total_reward = 0
        for i in range(args.action_repeat):
            img_rgb, reward, done, die = self.env.step(action)
            total_reward += reward
            if done:
                break
        obs = self.process_state(img_rgb, color, crop, domain)
        self.stack.pop(0)
        self.stack.append(obs)
        assert len(self.stack) == self.img_stack
        state = np.concatenate(self.stack, axis=0)
        return state, total_reward, done, die

    def render(self, *arg):
        self.env.render(*arg)

######## models ##########
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
            dim_z=args.reduce_dim_latent_size,
            )
    return model

def create_img_scm():
    scm = MLPImplicitSCM(
            graph_parameterization='none',
            manifold_thickness=0.01,
            hidden_units=100,
            hidden_layers=2,
            homoskedastic=False,
            dim_z=args.reduce_dim_latent_size,
            min_std=0.2,
        )

    return scm

def create_img_encoder_decoder():
    if args.ilcm_encoder_type == 'resnet':
        encoder = ImageEncoder(
                in_resolution=64,
                in_features=3,
                out_features=args.reduce_dim_latent_size,
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
                in_features=args.reduce_dim_latent_size,
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
    elif args.ilcm_encoder_type == 'conv':
        encoder = Encoder3dshapes(
                in_resolution=64,
                in_features=3,
                out_features=args.reduce_dim_latent_size,
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
                in_features=args.reduce_dim_latent_size,
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
            dim_z=args.latent_size,
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
                input_features=args.reduce_dim_latent_size,
                output_features=args.latent_size,
                fix_std=False,
                init_std=0.01,
                min_std=0.0001,
            )
    decoder = GaussianEncoder(
                hidden=decoder_hidden,
                input_features=args.latent_size,
                output_features=args.reduce_dim_latent_size,
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
            dim_z=args.latent_size,
            min_std=0.2,
        )
    return scm

class EncoderBase(nn.Module):
    def __init__(self, num_channels):
        super(EncoderBase, self).__init__()
        self.num_channels = num_channels
        self.cnn_base = nn.Sequential(
            nn.Conv2d(self.num_channels, 32, 4, stride=2), nn.ReLU(),
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

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, input_dim, train_encoder=False):
        super().__init__()
        self.input_dim = input_dim
        self.train_encoder = train_encoder
        if self.train_encoder == True:
            self.encoder = EncoderBase(num_channels=12)
            self.input_dim=self.encoder.out_dim

        conv_seqs = [
            nn.ReLU(),
            nn.Linear(self.input_dim, out_features=256),
            nn.ReLU(),
            ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = layer_init(nn.Linear(256, 256), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)
        self.alpha_head = nn.Sequential(nn.Linear(256, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(256, 3), nn.Softplus())

    def get_value(self, x):
        with torch.no_grad():
            x = torch.tensor(x).to(device).unsqueeze(0)
            if self.train_encoder == True:
                x = self.encoder(x)
        return self.critic(self.network(x)) 

    def get_action_and_value(self, x, action=None):
        with torch.no_grad():
            if self.train_encoder == True:
                x = self.encoder(x)
            hidden = self.network(x)
            actor = self.actor(hidden)
            alpha = self.alpha_head(actor) + 1
            beta = self.beta_head(actor) + 1
            probs = Beta(alpha, beta)
            if action is None:
                action = probs.sample()
                action = action.squeeze().cpu().numpy()
                action[0] = action[0]*2-1
        return action 

########### Do Evaluation #################
def main():
    seed_everything(args.seed)
    warnings.filterwarnings("ignore")
    video_dir = os.path.join(args.work_dir, 'video')
    #video = VideoRecorder(video_dir if args.save_video else None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    encoder = None
    main = None
    train_encoder = False
    if args.use_encoder:
        if args.repr == 'cycle_vae' or args.repr == 'vae' or args.repr == 'adagvae':
            encoder = Encoder(latent_size = args.latent_size)
            weights = torch.load(args.encoder_path, map_location=torch.device('cpu'))
            for k in list(weights.keys()):
                if k not in encoder.state_dict().keys():
                    del weights[k]
            encoder.load_state_dict(weights)
        elif args.repr == 'ilcm':
            print('causal')
            encoder = create_model_reduce_dim()
            main = create_ilcm()
            # saved checkpoints could contain extra weights such as linear_logsigma 
            weights = torch.load(args.encoder_path, map_location=torch.device('cpu'))
            for k in list(weights.keys()):
                if k not in encoder.state_dict().keys():
                    del weights[k]
            encoder.load_state_dict(weights)
            print("Loaded Weights Encoder")

            # saved checkpoints could contain extra weights such as linear_logsigma 
            weights = torch.load(args.ilcm_path, map_location=torch.device('cpu'))
            for k in list(weights.keys()):
                if k not in main.state_dict().keys():
                    del weights[k]
            main.load_state_dict(weights)
            print("Loaded Weights Causal")
        elif args.repr == 'disent':
            class_latent_size = 8
            content_latent_size = 32
            encoder = EncoderD(class_latent_size, content_latent_size)
            weights = torch.load(args.encoder_path, map_location=torch.device('cpu'))
            for k in list(weights.keys()):
                if k not in encoder.state_dict().keys():
                    del weights[k]
            encoder.load_state_dict(weights)

    carracing_env = gym.make(args.env_name)
    env = Env(carracing_env, device, encoder, main)
    action_space = env.action_space.shape
    observation_space = env.observation_space.shape

    if encoder == None:
        train_encoder = True
    model = Agent(input_dim=args.latent_size*args.img_stack, train_encoder=train_encoder).to(device)
    weights = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(weights)
    print("Loaded Weights Policy")

    domains_results = []
    domains_means = []
    crop = False

    for domain in range(args.nb_domain):
        if domain == 0:
            color = None
        elif domain == args.nb_domain - 1:
            color = 0.5
        elif domain == args.nb_domain - 2:
            color = -0.3 #Try with -0.4 (rouge)
        else:
            color = -0.2 
        print(color, crop)
        results = []
        for i in range(args.num_episodes):
            episode_reward, done, new_obs = 0, False, env.reset(color, crop, domain)
            #video.init(enabled=((i == 0) or (i == 99)))
            while not done:
                obs = torch.from_numpy(new_obs).to(device).float().unsqueeze(0)
                action = model.get_action_and_value(obs)
                obs, reward, done, die = env.step(action, color, crop, domain)
                new_obs = obs
                episode_reward += reward
                #video.record(env)
            #video.save('%d_%d.mp4' % (domain, i))
            results.append(episode_reward)

        print('Evaluate %d episodes and achieved %f scores in domain %d' % (args.num_episodes, np.mean(results), domain))
        print(results)
        domains_results.append(results)
        domains_means.append(np.mean(results))
    with open(os.path.join(args.save_path, 'results_%s_%d.txt' % (args.repr, args.number)), 'w') as f:
        f.write('score = %s\n' % domains_results)
        f.write('mean_scores = %s\n' % domains_means)
        f.write('model = %s\n' % args.model_path)
if __name__ == '__main__':
    main()