import os
import random
import time
from dataclasses import dataclass
import gym
from gym import spaces
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.distributions import Beta
import torch.optim as optim
import cv2
import argparse
import warnings

from CDARL.representation.VAE.vae import Encoder
from CDARL.representation.CYCLEVAE.cycle_vae import EncoderD
from CDARL.utils import seed_everything, Results, ReplayBuffer, random_shift, random_crop, random_conv, transform, create_logs, cat, RandomTransform
from CDARL.representation.ILCM.model import MLPImplicitSCM, HeuristicInterventionEncoder, ILCM
from CDARL.representation.ILCM.model import ImageEncoderCarla, ImageDecoderCarla, CoordConv2d, GaussianEncoder

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--repr', default='adagvae', type=str)
parser.add_argument('--use-encoder', default=True, action='store_true')
parser.add_argument('--encoder_path', default='/home/mila/l/lea.cote-turcotte/CDARL/representation/ADAGVAE/logs/carracing/2024-01-31/encoder_adagvae.pt', type=str)
parser.add_argument('--ilcm_path', default='/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carracing/2024-01-15/model_step_180000.pt', type=str)
parser.add_argument('--save_dir', default='/home/mila/l/lea.cote-turcotte/CDARL/logs', type=str)
parser.add_argument('--algo', default='clean_rl', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--torch_deterministic', default=True, type=bool)
parser.add_argument('--cuda', default=True, type=bool)
parser.add_argument('--capture_video', default=False, type=bool)
parser.add_argument('--total_timesteps', default=int(25e6), type=int)
parser.add_argument('--learning_rate', default=5e-4, type=float)
parser.add_argument('--num_envs', default=1, type=int)
parser.add_argument('--action_repeat', default=8, type=int)
parser.add_argument('--img_stack', default=4, type=int)
parser.add_argument('--num_steps', default=2000, type=int)
parser.add_argument('--gamma', default=0.999, type=float)
parser.add_argument('--gae_lambda', default=0.95, type=float)
parser.add_argument('--num_minibatches', default=8, type=int)
parser.add_argument('--update_epochs', default=10, type=int)
parser.add_argument('--clip_coef', default=0.2, type=float)
parser.add_argument('--ent_coef', default=0.01, type=float)
parser.add_argument('--vf_coef', default=0.5, type=float)
parser.add_argument('--max_grad_norm', default=0.5, type=float)
parser.add_argument('--anneal_lr', default=False, type=bool)
parser.add_argument('--norm_adv', default=True, type=bool)
parser.add_argument('--clip_vloss', default=True, type=bool)
parser.add_argument('--target_kl', default=None, type=float)
parser.add_argument('--latent-size', default=32, type=int)
parser.add_argument('--log_interval', default=10, type=int)
args = parser.parse_args()

class Env():
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self, env, encoder=None, causal=None):
        self.env = env
        self.env.seed(args.seed)
        self.img_stack = args.img_stack
        self.reward_threshold = self.env.spec.reward_threshold
        self.encoder = encoder
        self.causal = causal
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.encoder:
            self.encoder = self.encoder.to(self.device)
        if self.causal:
            self.causal = self.causal.to(self.device)
        if self.encoder:
            self.observation_space = spaces.Box(low=-1000, high=1000, shape=(args.latent_size*self.img_stack,), dtype=np.float)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(3*self.img_stack,64,64), dtype=np.float)
        self.action_space = self.env.action_space

    def process_obs(self, obs):
        obs = np.ascontiguousarray(obs, dtype=np.float32) / 255
        obs = cv2.resize(obs[:84, :, :], dsize=(64,64), interpolation=cv2.INTER_NEAREST)
        return np.transpose(obs, (2,0,1))

    def process_state(self, obs):
        obs = np.ascontiguousarray(obs, dtype=np.float32) / 255
        obs = cv2.resize(obs[:84, :, :], dsize=(64,64), interpolation=cv2.INTER_NEAREST)
        obs = np.transpose(obs, (2,0,1))
        if self.encoder is None:
            return obs #(3, 64, 64)
        else:
            obs = np.expand_dims(obs, axis=0) #(1, 3, 64, 64)
            save_image(torch.tensor(obs), os.path.join('/home/mila/l/lea.cote-turcotte/CDARL/checkimages', "cleanrl_test.png"))
            obs = torch.from_numpy(obs).float().to(self.device)
            if args.repr == 'cycle_vae' or args.repr == 'vae' or args.repr == 'adagvae':
                with torch.no_grad():
                    state, _ = self.encoder(obs)
                    state = state.cpu().squeeze().numpy()
            elif args.repr == 'ilcm':
                with torch.no_grad():
                    z, _ = self.encoder.encoder.mean_std(obs)
                    obs = self.causal.encode_to_causal(z).cpu().squeeze().numpy()
                    print(obs.shape)
            elif args.repr == 'disent':
                with torch.no_grad():
                    content, _, style = self.encoder(obs)
                    obs = torch.cat([content, style], dim=1).cpu().squeeze().numpy()
                    print(obs.shape)
        return state

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        processed_obs = self.process_state(img_rgb)
        self.stack = [processed_obs] * self.img_stack
        state = np.concatenate(self.stack, axis=0)
        return state
        
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
        obs = self.process_state(img_rgb)
        self.stack.pop(0)
        self.stack.append(obs)
        assert len(self.stack) == self.img_stack
        state = np.concatenate(self.stack, axis=0)
        return state, total_reward, done, die

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

    def close(self):
        self.env.close

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
            self.encoder = EncoderBase(num_channels=3)
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
        x = torch.tensor(x).to(device).unsqueeze(0)
        if self.train_encoder == True:
            x = self.encoder(x)
        return self.critic(self.network(x)) 

    def get_action_and_value(self, x, action=None):
        if self.train_encoder == True:
            x = self.encoder(x)
        hidden = self.network(x)
        actor = self.actor(hidden)
        alpha = self.alpha_head(actor) + 1
        beta = self.beta_head(actor) + 1
        probs = Beta(alpha, beta)
        if action is None:
            action = probs.sample()
        return action.squeeze().cpu().numpy(), probs.log_prob(action).sum(dim=1), probs.entropy().sum(1), self.critic(hidden)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    results = Results(title="Moving averaged episode reward", xlabel="episode", ylabel="running_score")
    results.create_logs(labels=["episode", "running_score", "episodic_return", "training_time"], init_values=[[], [], [], []])

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    log_dir = create_logs(args, algo_name=True, repr=args.repr)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    encoder = None
    main = None
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
            latent_size = 16
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
            print("Loaded Weights Main")
        elif args.repr == 'disent':
            class_latent_size = 16
            content_latent_size = 32
            encoder = EncoderD(class_latent_size, content_latent_size)
            weights = torch.load(args.encoder_path, map_location=torch.device('cpu'))
            for k in list(weights.keys()):
                if k not in encoder.state_dict().keys():
                    del weights[k]
            encoder.load_state_dict(weights)

    carracing_env = gym.make('CarRacing-v0')
    env = Env(carracing_env, encoder, main)
    action_space = env.action_space.shape
    observation_space = env.observation_space.shape

    agent = Agent(input_dim=args.latent_size*args.img_stack).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, ) + observation_space).to(device)
    actions = torch.zeros((args.num_steps, ) + action_space).to(device)
    logprobs = torch.zeros((args.num_steps, )).to(device)
    rewards = torch.zeros((args.num_steps, )).to(device)
    dones = torch.zeros((args.num_steps, )).to(device)
    values = torch.zeros((args.num_steps, )).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.tensor(env.reset()).to(device)
    next_done = 0
    total_reward = 0
    episode_lenght = 0
    episode = 0
    running_score = 0

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                next_obs = torch.tensor(next_obs).to(device).unsqueeze(0)
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = torch.from_numpy(action).to(device)
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            total_reward += reward
            episode_lenght += 1
            rewards[step] = reward
            next_obs, next_done = torch.tensor(next_obs).to(device), torch.tensor(int(done)).to(device)

            if done or die:
                running_score = running_score * 0.99 + total_reward * 0.01
                if episode % args.log_interval == 0:
                    print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(episode, total_reward, running_score))
                    results.update_logs(["episode", "running_score", "episodic_return", "training_time"], [episode, running_score, total_reward, int(global_step / (time.time() - start_time))])
                    torch.save(agent.state_dict(), os.path.join(log_dir, "policy.pt"))
                    results.save_logs(log_dir)
                    results.generate_plot(log_dir, log_dir)

                episode += 1
                total_reward = 0
                episode_lenght = 0
                next_obs = torch.tensor(env.reset()).to(device)
                next_done = 0
                die = 0

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + observation_space)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_space)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        print('update agent')
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y