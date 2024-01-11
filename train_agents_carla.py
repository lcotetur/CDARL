import torch
import torch.nn as nn
import gym
from gym import spaces
import gym_carla
import carla
import numpy as np
import argparse
import random 
import rlcodebase
from rlcodebase.agent import PPOAgent
from rlcodebase.utils import Config, Logger
from torch.utils.tensorboard import SummaryWriter
from model import CarlaLatentPolicy, CarlaImgPolicy

parser = argparse.ArgumentParser()
parser.add_argument('--algo', default='adagvae', type=str)
parser.add_argument('--encoder-path', default='/home/mila/l/lea.cote-turcotte/CDARL/ADAGVAE/logs/carracing/2023-11-22/encoder_adagvae.pt')
parser.add_argument('--model-save-path', default='/home/mila/l/lea.cote-turcotte/CDARL/checkpoints/policy_adagvae_0.pt', type=str)
parser.add_argument('--max-steps', default=int(1e5), type=int)
parser.add_argument('--weather', default=0, type=int)
parser.add_argument('--action-repeat', default=1, type=int)
parser.add_argument('--use-encoder', default=True, action='store_true')
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--port', default=2000, type=int)
parser.add_argument('--tag', default=None, type=str)
parser.add_argument('--latent-size', default=32, type=int, help='dimension of latent state embedding')
args = parser.parse_args()

weathers = [carla.WeatherParameters.ClearNoon, carla.WeatherParameters.HardRainNoon, carla.WeatherParameters(50, 0, 0, 0.35, 0, -40)]
weather = weathers[args.weather]
start_point = (75, -10, 2.25)
end_point = (5, -242, 2.25)

params = {
    'number_of_vehicles': 0,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': False,  # whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': args.port,  # connection port
    'town': 'Town07',  # which town to simulate
    # 'task_mode': 'random',  # removed
    'max_time_episode': 800,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 16,  # observation range (meter)
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,  # threshold for out of lane
    'desired_speed': 5,  # desired speed (m/s)
    'max_ego_spawn_times': 1,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': True,  # whether to output PIXOR observation
    'start_point': start_point,
    'end_point': end_point,
    'weather': weather,
    'ip': 'localhost'
}


class VecGymCarla:
    def __init__(self, env, action_repeat, encoder = None):
        self.env = env
        self.action_repeat = action_repeat
        self.encoder = encoder
        self.action_space = self.env.action_space
        if self.encoder:
            self.observation_space = spaces.Box(low=-1000, high=1000, shape=(32+1,), dtype=np.float)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(3*128*128+1,), dtype=np.uint8)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if self.encoder:
            self.encoder = self.encoder.to(self.device)
        self.episodic_return = 0
        self.episodic_len = 0

    def step(self, action):
        action = np.clip(action, -1, 1)
        action = np.squeeze(action) * self.env.action_space.high
        cum_r = 0
        i = {'episodic_return': None}
        for _ in range(self.action_repeat):
            s,r,d,_ = self.env.step(action)
            cum_r += r
            self.episodic_return += r
            self.episodic_len += 1
            if d:
                s = self.env.reset()
                i = {'episodic_return': self.episodic_return}
                print('Done: ', self.episodic_return, self.episodic_len)
                self.episodic_return, self.episodic_len = 0, 0
                break
        s, cum_r, d, i = self.process_state(s), [cum_r], [d], [i]
        return s, cum_r, d, i

    def reset(self):
        s = self.env.reset()
        self.episodic_return = 0
        return self.process_state(s)

    def process_state(self, s):
        if self.encoder is None:
            obs = np.transpose(s['camera'], (2,0,1)).reshape(-1)
            speed = s['state'][2]
            state = np.append(obs, speed)
            state = np.expand_dims(state, axis=0)
        else:
            obs = np.transpose(s['camera'], (2,0,1))
            obs = np.expand_dims(obs, axis=0)
            obs = torch.from_numpy(obs).float().to(self.device)
            if args.algo == 'cycle_vae' or args.algo == 'vae' or args.algo == 'adagvae':
                with torch.no_grad():
                    obs = self.encoder(obs).cpu().squeeze().numpy()
            elif args.algo == 'ilcm':
                with torch.no_grad():
                    obs = self.encoder.encode_to_causal(obs).cpu().squeeze().numpy()
            elif args.algo == 'disent':
                with torch.no_grad():
                    content, _, style = self.encoder(obs)
                    obs = torch.cat([content, style], dim=1).cpu().squeeze().numpy()

            speed = s['state'][2]
            state = np.expand_dims(np.append(obs, speed), axis=0)
        
        return state

####### ilcm model #########
def create_model():
    # Create model
    scm = create_scm()
    encoder, decoder = create_encoder_decoder()
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

def create_scm():
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

def create_encoder_decoder():
    encoder = ImageEncoder(
            in_resolution=63,
            in_features=3,
            out_features=args.latent_size,
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
            in_features=args.latent_size,
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

def create_intervention_encoder():
    intervention_encoder = HeuristicInterventionEncoder()
    return intervention_encoder

# adagvae, vae and cycle-vae model
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
        self.linear_mu = nn.Linear(9216, latent_size)

    def forward(self, x):
        x = self.main(x/255.0)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        return mu

class EncoderD(nn.Module):
    def __init__(self, class_latent_size = 16, content_latent_size = 32, input_channel = 3, flatten_size = 9216):
        super(EncoderD, self).__init__()
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


def main():
    # prepare env
    encoder = None
    if args.use_encoder:
        if args.algo == 'cycle_vae' or args.algo == 'vae' or args.algo == 'adagvae':
            encoder = Encoder(latent_size = args.latent_size)
            weights = torch.load(args.encoder_path, map_location=torch.device('cpu'))
            for k in list(weights.keys()):
                if k not in encoder.state_dict().keys():
                    del weights[k]
            encoder.load_state_dict(weights)
        elif args.algo == 'ilcm':
            print('causal')
            latent_size = 8
            encoder = create_model()
            if custom_config['encoder_path'] is not None:
                # saved checkpoints could contain extra weights such as linear_logsigma 
                weights = torch.load(custom_config['encoder_path'], map_location=torch.device('cpu'))
                for k in list(weights.keys()):
                    if k not in encoder.state_dict().keys():
                        del weights[k]
                encoder.load_state_dict(weights)
                print("Loaded Weights")
        elif args.algo == 'disent':
            class_latent_size = 16
            content_latent_size = 32
            encoder = EncoderD(class_latent_size, content_latent_size)
            weights = torch.load(args.encoder_path, map_location=torch.device('cpu'))
            for k in list(weights.keys()):
                if k not in encoder.state_dict().keys():
                    del weights[k]
            encoder.load_state_dict(weights)

    carla_env = gym.make('carla-v0', params=params)
    env = VecGymCarla(carla_env, args.action_repeat, encoder)

    # prepare config
    config = Config()
    config.game = 'weather%d' % args.weather
    config.algo = 'ppo'
    config.max_steps = args.max_steps
    config.num_envs = 1
    config.optimizer = 'Adam'
    config.lr = args.lr
    config.discount = 0.99
    config.use_gae = True
    config.gae_lambda = 0.95
    config.use_grad_clip = True
    config.max_grad_norm = 0.5
    config.rollout_length = 128
    config.value_loss_coef = 1
    config.entropy_coef = 0.01
    config.ppo_epoch = 4
    config.ppo_clip_param = 0.2
    config.num_mini_batch = 4
    config.use_gpu = True
    config.save_interval = 10000 
    config.memory_on_gpu = True
    config.after_set()
    print(config)

    # prepare model
    if args.use_encoder:
        Model = CarlaLatentPolicy
        input_dim = args.latent_size+1  # 16+1 in paper
    else:
        Model = CarlaImgPolicy
        input_dim = args.latent_size+1  # 128+1 in paper (16 is too small)
    model = Model(input_dim, 2).to(config.device)

    # create ppo agent and run
    logger =  Logger(SummaryWriter(config.save_path), config.num_echo_episodes)

    # create agent and run
    agent = PPOAgent(config, env, model, logger)
    agent.run()
    torch.save(model.state_dict(), args.model_save_path)

if __name__ == '__main__':
    main()