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
parser.add_argument('--data-dir', default='/home/mila/l/lea.cote-turcotte/LUSR/data/carracing_data', type=str, help='path to the data')
parser.add_argument('--data-tag', default='car', type=str, help='files with data_tag in name under data directory will be considered as collected states')
parser.add_argument('--num-splitted', default=10, type=int, help='number of files that the states from one domain are splitted into')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--encoder-path', default='/home/mila/l/lea.cote-turcotte/LUSR/checkpoints/encoder_offline.pt')
parser.add_argument('--train-encoder', default=False, type=bool)
parser.add_argument('--learning-rate', default=0.0002, type=float)
parser.add_argument('--learning-rate-vae', default=0.0001, type=float)
parser.add_argument('--latent-size', default=32, type=int)
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--vis', action='store_true', help='use visdom')
parser.add_argument('--beta', default=10, type=int)
parser.add_argument('--bloss-coef', default=1, type=int)
parser.add_argument('--class-latent-size', default=8, type=int)
parser.add_argument('--content-latent-size', default=32, type=int)
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='interval between training status logs (default: 10)')
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--entropy_coef', default=.01, type=float)
parser.add_argument('--value_loss_coef', default=2., type=float)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

transition = np.dtype([('s', np.float64, (3, 64, 64)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (3, 64, 64))])

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

def backward_loss(x, model, device):
    mu, logsigma, classcode = model.encoder(x)
    shuffled_classcode = classcode[torch.randperm(classcode.shape[0])]
    randcontent = torch.randn_like(mu).to(device)

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

    '''
    def step(self, action):
        rewards = 0
        for i in range(args.action_repeat):
            obs, reward, done, info = self.env.step(action)
            reward = (-0.1 if reward < 0 else reward)
            rewards += reward
            # maybe try with reward memory (self.av_r)
            if done:
                break
        processed_obs = self.process_obs(obs)
        return processed_obs, rewards, done, info
    '''

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

# Encoder for online training
   # '''
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
    #'''
'''
# Encoder for offline training
class Encoder(nn.Module):
    def __init__(self, latent_size = 16, input_channel = 3):
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

class Decoder(nn.Module):
    def __init__(self, latent_size=32, output_channel=3, flatten_size=1024):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(latent_size, flatten_size)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(flatten_size, 128, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2), nn.Sigmoid()
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
    def __init__(self, class_latent_size = 8, content_latent_size = 32, img_channel = 3, flatten_size = 1024):
        super(DisentangledVAE, self).__init__()
        #online encoder decoder
        self.encoder = Encoder(class_latent_size, content_latent_size, img_channel, flatten_size)
        self.decoder = Decoder(class_latent_size + content_latent_size, img_channel, flatten_size)

        #self.encoder = Encoder(content_latent_size, img_channel)
        #self.decoder = Decoder(class_latent_size + content_latent_size, img_channel, flatten_size)

    def forward(self, x):
        mu, logsigma, classcode = self.encoder(x)
        contentcode = reparameterize(mu, logsigma)
        latentcode = torch.cat([contentcode, classcode], dim=1)

        recon_x = self.decoder(latentcode)

        return mu, logsigma, classcode, recon_x

class ActorCriticNet(nn.Module):
    def __init__(self, args, content_latent_size=32):
        nn.Module.__init__(self)

        self.main = Encoder()  

        '''
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
        '''

        self.critic = nn.Sequential(nn.Linear(content_latent_size, 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, 1))
        self.actor = nn.Sequential(nn.Linear(content_latent_size, 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(300, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(300, 3), nn.Softplus())
        self.train_encoder = args.train_encoder
        #self.apply(self._weights_init)
        print("Train Encoder: ", self.train_encoder)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, input):
        features = self.main.get_feature(input) #for online encoder
        #features = self.main(input.float())
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
    buffer_capacity, batch_size = 2000, 200
    batch_size_vae = 100

    def __init__(self, args, results_vae, vae_batches, vae_epoch):
        self.training_step = 0
        self.vae_epoch = vae_epoch
        self.vae_batches = vae_batches
        self.net = ActorCriticNet(args).to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        self.model = DisentangledVAE(class_latent_size = args.class_latent_size, content_latent_size = args.content_latent_size).to(device)
        self.vae_optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate_vae)

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
        torch.save(self.net.state_dict(), '/home/mila/l/lea.cote-turcotte/LUSR/checkpoints/policy_train_v2_offline.pt')

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

        """

        print('updating cycle vae')
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = ExpDataset(args.data_dir, args.data_tag, 1, transform)
        loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)
        for i_split in range(1):
            for i_batch, imgs in enumerate(loader):
                self.vae_batches += 1
                # forward circle
                # Try 
                imgs = imgs.permute(1,0,2,3,4).to(device, non_blocking=True) # from torch.Size([10, 5, 3, 64, 64]) to torch.Size([5, 10, 3, 64, 64])
                imgs = imgs.reshape(-1, *imgs.shape[2:])
                imgs = RandomTransform(imgs).apply_transformations()
                self.vae_optimizer.zero_grad()

                floss = 0
                for i_class in range(imgs.shape[0]): # imgs.shape[0] = 5 
                    # batch size 10 for each class (5 class)
                    image = imgs[i_class]
                    floss += forward_loss(image, self.model, args.beta)
                    save_image(image, "/home/mila/l/lea.cote-turcotte/LUSR/figures/class_lusr_%d.png" % (i_class))
                floss = floss / imgs.shape[0] # divided by the number of classes

                # backward circle
                imgs = imgs.reshape(-1, *imgs.shape[2:]) # from torch.Size([5, 10, 3, 64, 64]) to torch.Size([50, 3, 64, 64])
                # batch of 50 imgaes with mix classes
                bloss = backward_loss(imgs, self.model, device)

                loss = floss + bloss * args.bloss_coef
                results_vae.update_logs(["vae_batches", "vae_epoch", "loss"], [self.vae_batches, self.vae_epoch, loss.item()])

                (floss + bloss * args.bloss_coef).backward()
                self.vae_optimizer.step()

                # save image to check and save model 
                if i_batch % 1000 == 0:
                    print("%d Splitted Data, %d Batches is Done." % (i_split, i_batch))
                    rand_idx = torch.randperm(imgs.shape[0])
                    imgs1 = imgs[rand_idx[:9]]
                    imgs2 = imgs[rand_idx[-9:]]
                    with torch.no_grad():
                        mu, _, classcode1 = self.model.encoder(imgs1)
                        _, _, classcode2 = self.model.encoder(imgs2)
                        recon_imgs1 = self.model.decoder(torch.cat([mu, classcode1], dim=1))
                        recon_combined = self.model.decoder(torch.cat([mu, classcode2], dim=1))

                    saved_imgs = torch.cat([imgs1, imgs2, recon_imgs1, recon_combined], dim=0)
                    save_image(saved_imgs, "/home/mila/l/lea.cote-turcotte/LUSR/checkimages/y%d_%d.png" % (i_split,i_batch), nrow=9)
                    torch.save(model.encoder.state_dict(), "/home/mila/l/lea.cote-turcotte/LUSR/checkpoints/encoder_data_lusr.pt")
            self.vae_epoch += 1
            # load next splitted data
            updateloader(loader, dataset)

            """
        #'''
        for _ in range(100):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size_vae, False):
                self.vae_batches += 1
                imgs = torch.tensor(self.buffer['s'][index], dtype=torch.float).to(device, non_blocking=True)
                imgs = RandomTransform(imgs).apply_transformations(nb_class=10)

                floss = 0
                for i_class in range(imgs.shape[0]):
                    image = imgs[i_class]
                    floss += forward_loss(image, self.model, args.beta)
                    save_image(image, "/home/mila/l/lea.cote-turcotte/LUSR/figures/class_%d.png" % (i_class))
                floss = floss / imgs.shape[0] # divided by the number of classes

                # backward circle
                imgs = imgs.reshape(-1, *imgs.shape[2:]) 
                bloss = backward_loss(imgs, self.model, device)

                loss = floss + bloss * args.bloss_coef
                results_vae.update_logs(["vae_batches", "vae_epoch", "loss"], [self.vae_batches, self.vae_epoch, loss.item()])

                self.vae_optimizer.zero_grad()
                (floss + bloss * args.bloss_coef).backward()
                self.vae_optimizer.step()

                # save image to check and save model 
                if self.vae_batches % 2000 == 0:
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
                    save_image(saved_imgs, "/home/mila/l/lea.cote-turcotte/LUSR/checkimages/z%d_%d.png" % (self.vae_epoch, self.vae_batches), nrow=9)
                    torch.save(self.model.encoder.state_dict(), "/home/mila/l/lea.cote-turcotte/LUSR/checkpoints/encoder_trainv2_offline.pt")
            self.vae_epoch += 1
            ##'''
        print('updating agent')
        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size_vae, False):

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
        
    results_ppo = Resutls(title="Moving averaged episode reward", xlabel="episode", ylabel="running_score")
    results_ppo.create_logs(labels=["episode", "running_score", "episode_score", "training_time"], init_values=[[], [], [], []])

    results_vae = Resutls(title="Cycle Consistent vae Loss", xlabel="vae_epoch", ylabel="loss")
    results_vae.create_logs(labels=["vae_batches", "vae_epoch", "loss"], init_values=[[], [], []])

    agent = Agent(args, results_vae, 0, 0)
    env = Env()

    running_score = 0
    training_time = 0
    state = env.reset()
    start_time = time.time()
    # 100000
    for i_ep in range(10000):
        episode_score = 0
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
        episode_score = score
        running_score = running_score * 0.99 + score * 0.01

        if i_ep % args.log_interval == 0:
            training_time = time.time() - start_time
            results_ppo.update_logs(["episode", "running_score", "episode_score", "training_time"], [i_ep, running_score, episode_score, training_time])
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))
            print('Training time: {:.2f}\t'.format(training_time))
            agent.save_param()
            results_ppo.save_logs('/home/mila/l/lea.cote-turcotte/LUSR/logs', str(1))
            results_vae.save_logs('/home/mila/l/lea.cote-turcotte/LUSR/logs', str(2))
        if running_score > env.reward_threshold:
            results_ppo.save_logs('/home/mila/l/lea.cote-turcotte/LUSR/logs', str(1))
            results_vae.save_logs('/home/mila/l/lea.cote-turcotte/LUSR/logs', str(2))
            results_ppo.generate_plot('/home/mila/l/lea.cote-turcotte/LUSR/logs/1','/home/mila/l/lea.cote-turcotte/LUSR/figures')
            results_vae.generate_plot('/home/mila/l/lea.cote-turcotte/LUSR/logs/2','/home/mila/l/lea.cote-turcotte/LUSR/figures')
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            break
    results_ppo.save_logs('/home/mila/l/lea.cote-turcotte/LUSR/logs', str(1))
    results_vae.save_logs('/home/mila/l/lea.cote-turcotte/LUSR/logs', str(2))
    results_ppo.generate_plot('/home/mila/l/lea.cote-turcotte/LUSR/logs/1', '/home/mila/l/lea.cote-turcotte/LUSR/figures')
    results_vae.generate_plot('/home/mila/l/lea.cote-turcotte/LUSR/logs/2','/home/mila/l/lea.cote-turcotte/LUSR/figures')