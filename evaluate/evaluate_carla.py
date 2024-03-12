import torch
import torch.nn as nn
import gym
from gym import spaces
import gym_carla
import carla
import numpy as np
import argparse
import os
import random 
from torchvision.utils import save_image
import rlcodebase
from rlcodebase.agent import PPOAgent
from rlcodebase.utils import Config, Logger
#from torch.utils.tensorboard import SummaryWriter
from CDARL.utils import seed_everything, create_logs 
from CDARL.model import CarlaLatentPolicy, CarlaImgPolicy
from CDARL.representation.ILCM.model import MLPImplicitSCM, HeuristicInterventionEncoder, ILCM
from CDARL.representation.ILCM.model import ImageEncoderCarla, ImageDecoderCarla, CoordConv2d, GaussianEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--weather', default=0, type=int)
parser.add_argument('--action-repeat', default=1, type=int)
parser.add_argument('--algo', default='ilcm', type=str)
parser.add_argument('--model-path', default='/home/mila/l/lea.cote-turcotte/CDARL/carla_logs/ilcm/2024-02-17_0/200000-model.pt', type=str)
parser.add_argument('--use-encoder', default=True, action='store_true')
parser.add_argument('--encoder-path', default='/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carla/2024-02-14_1_reduce_dim/model_step_50000.pt', type=str) 
parser.add_argument('--ilcm-path', default='/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carla/2024-02-15/model_step_50000.pt', type=str)
parser.add_argument('--latent-size', default=16, type=int, help='dimension of latent state embedding')
parser.add_argument('--port', default=2000, type=int)
parser.add_argument('--num-eval', default=10, type=int)
parser.add_argument('--save-path', default='/home/mila/l/lea.cote-turcotte/CDARL/results/carla', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--save-obs', default=False, action='store_true')
parser.add_argument('--save-obs-path', default='./obs', type=str)
parser.add_argument('--save_video', default=True, action='store_true')
parser.add_argument('--work_dir', default='/home/mila/l/lea.cote-turcotte/CDARL', type=str)
parser.add_argument('--nb_domain', default=4, type=int)
args = parser.parse_args()

weathers = [carla.WeatherParameters.ClearNoon, carla.WeatherParameters.HardRainNoon, carla.WeatherParameters.WetCloudySunset, carla.WeatherParameters(50, 0, 0, 0.35, 0, -40), carla.WeatherParameters.HardRainSunset, carla.WeatherParameters.ClearSunset]
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
    def __init__(self, env, action_repeat, encoder=None, causal=None):
        self.env = env
        self.action_repeat = action_repeat
        self.encoder = encoder
        self.causal = causal
        self.action_space = self.env.action_space
        if self.encoder:
            self.observation_space = spaces.Box(low=-1000, high=1000, shape=(16+1,), dtype=np.float)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(3*128*128+1,), dtype=np.uint8)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if self.encoder:
            self.encoder = self.encoder.to(self.device)
        if self.causal:
            self.causal = self.causal.to(self.device)
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
            save_image(torch.tensor(obs)/255, os.path.join('/home/mila/l/lea.cote-turcotte/CDARL/checkimages', "%s.png" % weathers[args.weather]))
            obs = torch.from_numpy(obs).float().to(self.device)
            if args.algo == 'cycle_vae' or args.algo == 'vae' or args.algo == 'adagvae':
                with torch.no_grad():
                    obs = self.encoder(obs).cpu().squeeze().numpy()
            elif args.algo == 'ilcm':
                with torch.no_grad():
                    z, _ = self.encoder.encoder.mean_std(obs/255)
                    obs = self.causal.encode_to_causal(z).cpu().squeeze().numpy()
            elif args.algo == 'disent':
                with torch.no_grad():
                    content, _, style = self.encoder(obs)
                    obs = torch.cat([content, style], dim=1).cpu().squeeze().numpy()

            speed = s['state'][2]
            state = np.expand_dims(np.append(obs, speed), axis=0)
        
        return state


####### ilcm model #########
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
    encoder = ImageEncoderCarla(
            in_resolution=128,
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
    decoder = ImageDecoderCarla(
            in_features=32,
            out_resolution=128,
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
            dim_z=16,
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
                output_features=16,
                fix_std=False,
                init_std=0.01,
                min_std=0.0001,
            )
    decoder = GaussianEncoder(
                hidden=decoder_hidden,
                input_features=16,
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
            dim_z=16,
            min_std=0.2,
        )
    return scm

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
        mu = self.linear_mu(x/255)
        logsigma = self.linear_logsigma(x)
        classcode = self.linear_classcode(x)

        return mu, logsigma, classcode

def main():
    seed_everything(args.seed)

    #video_dir = os.path.join(args.work_dir, 'video')
    #video = VideoRecorder(video_dir if args.save_video else None)
    # prepare env
    encoder = None
    main = None
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
    carla_env.seed(args.seed)
    env = VecGymCarla(carla_env, args.action_repeat, encoder, main)

    # prepare model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if args.use_encoder:
        Model = CarlaLatentPolicy
        input_dim = args.latent_size+1  # 16+1 in paper
    else:
        Model = CarlaImgPolicy
        input_dim = args.latent_size+1  # 128+1 in paper (16 is too small)
    model = Model(input_dim, 2)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cuda')))
    model = model.to(device)

    res = []
    state = env.reset()
    while(len(res) < args.num_eval):
        action, _, _, _ = model(torch.from_numpy(state).float().to(device))
        state, _, done, info = env.step(action.cpu().numpy())
        for i in info:
            if i['episodic_return'] is not None:
                res.append(i['episodic_return'])

    print("Average Score", np.mean(res))
    with open(os.path.join(args.save_path, 'weather_%s_results_%s.txt' % (args.weather, args.algo)), 'w') as f:
        f.write('score = %s\n' % res)
        f.write('mean_scores = %s\n' % np.mean(res))
        f.write('model = %s\n' % args.model_path)
        f.write('weather = %s\n' % weathers[args.weather])


if __name__ == '__main__':
    # evaluate for weather 0, 1 and 2
    main()