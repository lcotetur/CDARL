import argparse

import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from utils import Resutls
import cv2
from torchvision.utils import save_image
from utils import RandomTransform, ExpDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--encoder-path', default='/home/mila/l/lea.cote-turcotte/LUSR/checkpoints/encoder_vae.pt')
parser.add_argument('--train-encoder', default=False, type=bool)
parser.add_argument('--learning-rate', default=0.0002, type=float)
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--entropy_coef', default=.01, type=float)
parser.add_argument('--value_loss_coef', default=2., type=float)
parser.add_argument('--latent-size', default=32, type=int)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

transition = np.dtype([('s', np.float64, (3, 64, 64)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (3, 64, 64))])

def reparameterize(mu, logsigma):
    std = torch.exp(0.5*logsigma)
    eps = torch.randn_like(std)
    return mu + eps*std

class Env():
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self):
        self.env = gym.make('CarRacing-v0')
        self.env.seed(args.seed)
        self.reward_threshold = self.env.spec.reward_threshold

    def process_obs(self, obs): # a single frame (96, 96, 3) for CarRacing
        obs = np.ascontiguousarray(obs, dtype=np.float32) / 255
        obs = cv2.resize(obs[:84, :, :], dsize=(64,64), interpolation=cv2.INTER_NEAREST)
        return np.transpose(obs, (2,0,1))

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        processed_obs = self.process_obs(img_rgb)
        return processed_obs

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

class ActorCriticNet(nn.Module):
    def __init__(self, args, latent_size=40):
        nn.Module.__init__(self)
        self.latent_size = args.latent_size
        self.main = Encoder()  

        if args.encoder_path is not None:
            # saved checkpoints could contain extra weights such as linear_logsigma 
            weights = torch.load(args.encoder_path, map_location=torch.device('cuda'))
            for k in list(weights.keys()):
                if k not in self.main.state_dict().keys():
                    del weights[k]
            self.main.load_state_dict(weights)
            print("Loaded Weights")
        else:
            print("No Load Weights")
        

        self.critic = nn.Sequential(nn.Linear(self.latent_size, 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, 1))
        self.actor = nn.Sequential(nn.Linear(self.latent_size, 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(300, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(300, 3), nn.Softplus())
        self.train_encoder = args.train_encoder
        print("Train Encoder: ", self.train_encoder)

    # Version 1
    def forward(self, input):
        features, _ = self.main(input.float())
        if not self.train_encoder:
            features = features.detach() # not train the encoder

        v = self.critic(features)
        actor_features = self.actor(features)
        alpha = self.alpha_head(actor_features) + 1
        beta = self.beta_head(actor_features) + 1

        return (alpha, beta), v

    '''
    # Version 2
    def forward(self, input):
        mu, logsigma, style = self.main(input.float())
        if not self.train_encoder:
            mu = mu.detach()
            logsigma = logsigma.detach()
            style = style.detach()  # not train the encoder

        content = reparameterize(mu, logsigma)
        features = torch.cat([content, style], dim=1)

        v = self.critic(features)
        actor_features = self.actor(features)
        alpha = self.alpha_head(actor_features) + 1
        beta = self.beta_head(actor_features) + 1

        return (alpha, beta), v
    '''


class Agent():
    """
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 128

    def __init__(self, args):
        self.training_step = 0
        self.net = ActorCriticNet(args).to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

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

    def save_param(self):
        torch.save(self.net.state_dict(), '/home/mila/l/lea.cote-turcotte/LUSR/checkpoints/policy_repr.pt')

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self):
        self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.float).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.float).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.float).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.float).to(device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.float).to(device).view(-1, 1)

        #Generalized Advantage Estimation (gae)
        with torch.no_grad():
            target_v = r + args.gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        print('updating agent')
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

                entorpy_loss = torch.mean(dist.entropy())

                loss = action_loss + args.value_loss_coef * value_loss - args.entropy_coef * entorpy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    results_ppo = Resutls(title="Moving averaged episode reward", xlabel="episode", ylabel="repr_running_score")
    results_ppo.create_logs(labels=["episode", "repr_running_score", "repr_episode_score", "training_time"], init_values=[[], [], [], []])

    agent = Agent(args)
    env = Env()

    repr_running_score = 0
    training_time = 0
    state = env.reset()
    start_time = time.time()
    # 100000
    for i_ep in range(10000):
        repr_episode_score = 0
        score = 0
        episode_lenght = 0
        state = env.reset()

        for t in range(2000):
            action, a_logp = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            if agent.store((state, action, a_logp, reward, state_)):
                agent.update()
            score += reward
            state = state_
            episode_lenght += 1
            if done or die:
                break
        repr_episode_score = score
        repr_running_score = repr_running_score * 0.99 + score * 0.01

        if i_ep % args.log_interval == 0:
            training_time = time.time() - start_time
            results_ppo.update_logs(["episode", "repr_running_score", "repr_episode_score", "training_time"], [i_ep, repr_running_score, repr_episode_score, training_time])
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, repr_running_score))
            print('Training time: {:.2f}\t'.format(training_time))
            agent.save_param()
            results_ppo.save_logs('/home/mila/l/lea.cote-turcotte/LUSR/logs', str(7))
        if repr_running_score > env.reward_threshold:
            results_ppo.save_logs('/home/mila/l/lea.cote-turcotte/LUSR/logs', str(7))
            results_ppo.generate_plot('/home/mila/l/lea.cote-turcotte/LUSR/logs/7','/home/mila/l/lea.cote-turcotte/LUSR/figures')
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(repr_running_score, score))
            break
    results_ppo.save_logs('/home/mila/l/lea.cote-turcotte/LUSR/logs', str(7))
    results_ppo.generate_plot('/home/mila/l/lea.cote-turcotte/LUSR/logs/7', '/home/mila/l/lea.cote-turcotte/LUSR/figures')