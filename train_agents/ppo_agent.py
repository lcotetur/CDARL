import argparse

import numpy as np
import time

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import cv2
import os
import warnings
from torchvision.utils import save_image
from CDARL.representation.VAE.vae import Encoder
from CDARL.representation.CYCLEVAE.cycle_vae import EncoderD
from CDARL.representation.ILCM.model import MLPImplicitSCM, HeuristicInterventionEncoder, ILCM
from CDARL.representation.ILCM.model import ImageEncoder, ImageDecoder, CoordConv2d, GaussianEncoder
from CDARL.utils import seed_everything, Results, ReplayBuffer, random_shift, random_crop, random_conv, transform, create_logs, cat, RandomTransform

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--algo', default='ilcm_ppo', type=str)
parser.add_argument('--save-dir', default="/home/mila/l/lea.cote-turcotte/CDARL/logs", type=str)
parser.add_argument('--encoder_path', default='/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carracing/2023-11-21/model_step_188000.pt', type=str)
parser.add_argument('--ilcm_path', default='/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carracing/2024-01-15/model_step_180000.pt', type=str)
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--latent-size', default=16, type=int)
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=1, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument(
    '--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
args = parser.parse_args()

transition = np.dtype([('s', np.float64, (3, 64, 64)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (3, 64, 64))])

class Env():
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self):
        self.env = gym.make('CarRacing-v0')
        self.env.seed(args.seed)
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
        return img
        
        #return processed_obs

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

class EncoderBase(nn.Module):
    def __init__(self):
        super(EncoderBase, self).__init__()
        self.cnn_base = nn.Sequential(
            nn.Conv2d(12, 32, 4, stride=2), nn.ReLU(),
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

class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self, args):
        super(Net, self).__init__()
        self.algo = args.algo
        if self.algo in ['ppo', 'curl_ppo', 'drq_ppo', 'rad_ppo', 'aumg_ppo']:
            self.cnn_base = EncoderBase()
            self.out_dim = self.cnn_base.out_dim
        elif self.algo == 'invar_ppo':
            self.cnn_base = EncoderD(class_latent_size = 8, content_latent_size = 32, input_channel = 3, flatten_size = 1024)
            weights = torch.load(args.encoder_path, map_location=torch.device('cuda'))
            for k in list(weights.keys()):
                if k not in self.cnn_base.state_dict().keys():
                    del weights[k]
            self.cnn_base.load_state_dict(weights)
            print("Loaded Weights")
            self.out_dim = 32
        elif self.algo == 'repr_ppo':
            self.cnn_base = Encoder(latent_size = 32, input_channel = 12, flatten_size = 1024)
            weights = torch.load(args.encoder_path, map_location=torch.device('cuda'))
            for k in list(weights.keys()):
                if k not in self.cnn_base.state_dict().keys():
                    del weights[k]
            self.cnn_base.load_state_dict(weights)
            print("Loaded Weights")
            self.out_dim = 32
        elif self.algo == 'ilcm_ppo':
            self.encoder = create_model_reduce_dim()
            weights_encoder = torch.load(args.encoder_path, map_location=torch.device('cuda'))
            for k in list(weights_encoder.keys()):
                if k not in self.encoder.state_dict().keys():
                    del weights_encoder[k]
            self.encoder.load_state_dict(weights_encoder)
            print("Loaded Weights")

            self.causal_mlp = create_ilcm()
            weights = torch.load(args.ilcm_path, map_location=torch.device('cuda'))
            for k in list(weights.keys()):
                if k not in self.causal_mlp.state_dict().keys():
                    del weights[k]
            self.causal_mlp.load_state_dict(weights)
            print("Loaded Weights")
            self.out_dim = 6

        self.critic = nn.Sequential(nn.Linear(self.out_dim, 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, 1))
        self.actor = nn.Sequential(nn.Linear(self.out_dim, 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(300, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(300, 3), nn.Softplus())

    def forward(self, x):
        if self.algo in ['ppo', 'curl_ppo', 'drq_ppo', 'rad_ppo', 'augm_ppo']:
            x = self.cnn_base(x)
        elif self.algo == 'invar_ppo':
            x, _, _ = self.cnn_base(x)
            x = x.detach()
        elif self.algo == 'repr':
            x, _ = self.cnn_base(x)
            x = x.detach()
        elif self.algo == 'ilcm_ppo':
            with torch.no_grad():
                z, _ = self.encoder.encoder.mean_std(x)
                x = self.causal_mlp.encode_to_causal(z)


        v = self.critic(x)
        x = self.actor(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v

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

class Agent():
    """
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 128
    aux_update_freq = 2

    def __init__(self, args):
        self.training_step = 0
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

        self.net = Net(args).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        if args.algo == 'curl_ppo':
            self.curl_head = CURLHead(self.net.cnn_base).cuda()
            self.curl_optimizer = torch.optim.Adam(self.curl_head.parameters(), lr=1e-3, betas=(0.9, 0.999))

    def select_action(self, state):
        state = torch.from_numpy(state).to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def save_param(self, log_dir):
        torch.save(self.net.state_dict(), os.path.join(log_dir, "policy.pt"))

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self, algo='ppo'):
        self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.float).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.float).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.float).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.float).to(device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.float).to(device).view(-1, 1)

        if algo == 'drq_ppo':
            s = random_shift(s)
            s_ = random_shift(s_)
        elif algo == 'rad_ppo':
            s = random_crop(s)
            s_ = random_crop(s_)
        elif algo == 'curl_ppo':
            pos = random_crop(s.clone())
            s = random_crop(s)
            s_ = random_crop(s_)
        elif algo == 'aumg_ppo':
            imgs = RandomTransform(s).apply_transformations_stack(num_frames=args.img_stack, nb_class=4)
            s = imgs.reshape(-1, *imgs.shape[2:]) 
            imgs_ = RandomTransform(s_).apply_transformations_stack(num_frames=args.img_stack, nb_class=4)
            s_ = imgs.reshape(-1, *imgs_.shape[2:]) 

        with torch.no_grad():
            target_v = r + args.gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]
            #adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if self.training_step % 2 == 0:
                    if algo == 'curl_ppo':
                        assert s.size(-1) == 64 and pos.size(-1) == 64

                        z_a = self.curl_head.encoder(s)
                        with torch.no_grad():
                            z_pos = self.critic_target.encoder(pos)
                        
                        logits = self.curl_head.compute_logits(z_a, z_pos)
                        labels = torch.arange(logits.shape[0]).long().cuda()
                        curl_loss = F.cross_entropy(logits, labels)
                        
                        self.curl_optimizer.zero_grad()
                        curl_loss.backward()
                        self.curl_optimizer.step()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = create_logs(args, algo_name=True)

    results_ppo = Results(title="Moving averaged episode reward", xlabel="episode", ylabel="ppo_running_score")
    results_ppo.create_logs(labels=["episode", "ppo_running_score", "ppo_episode_score", "training_time"], init_values=[[], [], [], []])

    agent = Agent(args)
    env = Env()

    ppo_running_score = 0
    training_time = 0
    state = env.reset()
    start_time = time.time()
    # 100000
    for i_ep in range(10000):
        ppo_episode_score = 0
        score = 0
        episode_lenght = 0
        state = env.reset()

        for t in range(2000):
            action, a_logp = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            if agent.store((state, action, a_logp, reward, state_)):
                print('updating')
                agent.update()
            score += reward
            state = state_
            episode_lenght += 1
            if done or die:
                break
        ppo_episode_score = score
        ppo_running_score = ppo_running_score * 0.99 + score * 0.01

        if i_ep % args.log_interval == 0:
            training_time = time.time() - start_time
            results_ppo.update_logs(["episode", "ppo_running_score", "ppo_episode_score", "training_time"], [i_ep, ppo_running_score, ppo_episode_score, training_time])
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, ppo_running_score))
            print('Training time: {:.2f}\t'.format(training_time))
            agent.save_param(log_dir)
            results_ppo.save_logs(log_dir)
            results_ppo.generate_plot(log_dir, log_dir)
        if ppo_running_score > env.reward_threshold:
            results_ppo.save_logs(log_dir)
            results_ppo.generate_plot(log_dir, log_dir)
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(ppo_running_score, score))
            break

    agent.save_param(log_dir)
    results_ppo.save_logs(log_dir)
    results_ppo.generate_plot(log_dir, log_dir)