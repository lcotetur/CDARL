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

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--encoder-path', default='/home/mila/l/lea.cote-turcotte/LUSR/checkpoints/encoder.pt')
parser.add_argument('--train-encoder', default=False, type=bool)
parser.add_argument('--latent-size', default=16, type=int)
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=3, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--vis', action='store_true', help='use visdom')
parser.add_argument(
    '--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cpu")
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

    def process_obs(self, obs): # a single frame (96, 96, 3) for CarRacing
        obs = cv2.resize(obs[:84, :, :], dsize=(64,64), interpolation=cv2.INTER_NEAREST)
        #return np.transpose(obs, (2,0,1))
        return obs

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        save_image(torch.tensor(img_rgb.copy(), dtype=torch.float32).permute(2, 0, 1), "/home/mila/l/lea.cote-turcotte/LUSR/checkimages/env_ppo_carracing.png", nrow=12)
        processed_obs = self.process_obs(img_rgb)
        save_image(torch.tensor(processed_obs.copy(), dtype=torch.float32), "/home/mila/l/lea.cote-turcotte/LUSR/checkimages/preprocesss_ppo_carracing.png", nrow=12)
        #img_gray = self.rgb2gray(processed_obs)
        #self.stack = [img_gray] * args.img_stack  # four frames for decision
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
        #self.stack.append(img_gray)
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
        x = self.main(x/255.0)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        return mu

class MyModel(nn.Module):
    def __init__(self, model_config):
        nn.Module.__init__(self)
        latent_size = model_config['latent_size']

        self.main = Encoder(latent_size=latent_size)
    
        if model_config['encoder_path'] is not None:
            # saved checkpoints could contain extra weights such as linear_logsigma 
            weights = torch.load(model_config['encoder_path'], map_location=torch.device('cpu'))
            for k in list(weights.keys()):
                if k not in self.main.state_dict().keys():
                    del weights[k]
            self.main.load_state_dict(weights)
            print("Loaded Weights")
        else:
            print("No Load Weights")
        
        self.critic = nn.Sequential(nn.Linear(latent_size, 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, 1))
        self.actor = nn.Sequential(nn.Linear(latent_size, 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(300, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(300, 3), nn.Softplus())
        self.train_encoder = model_config['train_encoder']
        self.apply(self._weights_init)
        print("Train Encoder: ", self.train_encoder)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, input):
        features = self.main(input)
        if not self.train_encoder:
            features = features.detach()  # not train the encoder

        v = self.critic(features)
        actor_features = self.actor(features)
        alpha = self.alpha_head(actor_features) + 1
        beta = self.beta_head(actor_features) + 1

        return (alpha, beta), v

class Agent():
    """
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 128

    def __init__(self, model_config):
        self.training_step = 0
        self.net = MyModel(model_config).double().to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def save_param(self):
        torch.save(self.net.state_dict(), 'param/ppo_net_params.pkl')

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

        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)

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
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()


if __name__ == "__main__":
    model_config = {'encoder_path':args.encoder_path, 'train_encoder':args.train_encoder, 'latent_size':args.latent_size}

    agent = Agent(model_config)
    env = Env()
        
    results = Resutls(title="Moving averaged episode reward", xlabel="episode", ylabel="running_score")

    logs = results.create_logs(labels=["episode", "running_score", "score"], init_values=[[], [], []])
    running_score = 0
    state = env.reset()
    # 100000
    for i_ep in range(3500):
        score = 0
        state = env.reset()

        for t in range(1000):
            action, a_logp = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if args.render:
                env.render()
            if agent.store((state, action, a_logp, reward, state_)):
                print('updating')
                agent.update()
            score += reward
            state = state_
            if done or die:
                break
        running_score = running_score * 0.99 + score * 0.01

        if i_ep % args.log_interval == 0:
            logs = results.update_logs(logs, ["episode", "running_score", "score"], [i_ep, running_score, score])
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))
            agent.save_param()
            results.save_logs(logs, '/home/mila/l/lea.cote-turcotte/LUSR/logs', str(1))
        if running_score > env.reward_threshold:
            results.save_logs(logs, '/home/mila/l/lea.cote-turcotte/LUSR/logs', str(1))
            results.generate_plot('/home/mila/l/lea.cote-turcotte/LUSR/logs/1/args.json','/home/mila/l/lea.cote-turcotte/LUSR/figures')
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            break
    results.save_logs(logs, '/home/mila/l/lea.cote-turcotte/LUSR/logs', str(1))
    results.generate_plot('/home/mila/l/lea.cote-turcotte/LUSR/logs/1', '/home/mila/l/lea.cote-turcotte/LUSR/figures')