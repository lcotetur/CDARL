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
from utils import Resutls
import cv2
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--latent-size', default=16, type=int)
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=3, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument(
    '--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

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
        save_image(torch.tensor(img_rgb.copy(), dtype=torch.float32).permute(2, 0, 1), "/home/mila/l/lea.cote-turcotte/LUSR/checkimages/env_ppo_carracing.png", nrow=12)
        processed_obs = self.process_obs(img_rgb)
        save_image(torch.tensor(processed_obs.copy(), dtype=torch.float32), "/home/mila/l/lea.cote-turcotte/LUSR/checkimages/preprocesss_ppo_carracing.png", nrow=12)
        #img_gray = self.rgb2gray(processed_obs)
        #self.stack = [processed_obs] * args.img_stack  # four frames for decision
        #return np.array(self.stack)
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
        #img_gray = self.rgb2gray(img_rgb)
        #self.stack.pop(0)
        #self.stack.append(mg_rgb)
        #assert len(self.stack) == args.img_stack
        #return np.array(self.stack), total_reward, done, die
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

class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self, args):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
        )
        self.critic = nn.Sequential(nn.Linear(2*2*256, 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, 1))
        self.actor = nn.Sequential(nn.Linear(2*2*256, 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(300, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(300, 3), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(x.size(0), -1)
        v = self.critic(x)
        x = self.actor(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v

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
        self.net = Net(args).to(device)
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
        torch.save(self.net.state_dict(), '/home/mila/l/lea.cote-turcotte/LUSR/checkpoints/policy_ppo.pt')

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

        with torch.no_grad():
            target_v = r + args.gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

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
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results_ppo = Resutls(title="Moving averaged episode reward", xlabel="episode", ylabel="ppo_running_score")
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
            agent.save_param()
            results_ppo.save_logs('/home/mila/l/lea.cote-turcotte/LUSR/logs', str(5))
        if ppo_running_score > env.reward_threshold:
            results_ppo.save_logs('/home/mila/l/lea.cote-turcotte/LUSR/logs', str(5))
            results_ppo.generate_plot('/home/mila/l/lea.cote-turcotte/LUSR/logs/5','/home/mila/l/lea.cote-turcotte/LUSR/figures')
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(ppo_running_score, score))
            break
    results_ppo.save_logs('/home/mila/l/lea.cote-turcotte/LUSR/logs', str(5))
    results_ppo.generate_plot('/home/mila/l/lea.cote-turcotte/LUSR/logs/5', '/home/mila/l/lea.cote-turcotte/LUSR/figures')