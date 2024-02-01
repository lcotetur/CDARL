import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog, ActionDistribution
from ray.rllib.utils.annotations import override
import torch
import torch.nn as nn
from torch.distributions import Beta
import gym
from gym.spaces import Box
import cv2
import numpy as np
import argparse
from torchvision.utils import save_image
from tqdm import trange

from CDARL.representation.VAE.vae import Encoder

from CDARL.representation.ILCM.model import MLPImplicitSCM, HeuristicInterventionEncoder, ILCM
from CDARL.representation.ILCM.model import ImageEncoder, ImageDecoder, CoordConv2d, GaussianEncoder
from CDARL.utils import seed_everything


######## Args Setting ##########
parser = argparse.ArgumentParser()
parser.add_argument('--policy-type', default='ilcm', type=str)
parser.add_argument('--ray-adress', default='auto', type=str)
parser.add_argument('--save-freq', default=100, type=int)
parser.add_argument('--train-epochs', default=5000, type=int)
parser.add_argument('--encoder-path', default='/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carracing/2023-11-21/model_step_188000.pt')
parser.add_argument('--ilcm-path', default='/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/runs/carracing/2024-01-15/model_step_180000.pt')
parser.add_argument('--model-save-path', default='/home/mila/l/lea.cote-turcotte/CDARL/checkpoints/policy_ilcm_0.pt', type=str)
parser.add_argument('--train-encoder', default=False, type=bool)
parser.add_argument('--num-workers', default=1, type=int)
parser.add_argument('--num-envs-per-worker', default=2, type=int)
parser.add_argument('--num-gpus', default=1, type=float)
parser.add_argument('--use-gae', default=True, type=bool)
parser.add_argument('--batch-mode', default='truncate_episodes', type=str)
parser.add_argument('--vf-loss-coeff', default=1, type=int)
parser.add_argument('--vf-clip-param', default=1000, type=int)
parser.add_argument('--reward-wrapper', default=True, type=bool, help='whether using reward wrapper so that avoid -100 penalty')
parser.add_argument('--lr', default=0.0002, type=float)
parser.add_argument('--kl-coeff', default=0, type=float)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num-sgd-iter', default=10, type=int)
parser.add_argument('--sgd-minibatch-size', default=200, type=int)
parser.add_argument('--grad-clip', default=0.1, type=float, help='other implementations may refer as max_grad_norm')
parser.add_argument('--rollout-fragment-length', default=250, type=int)
parser.add_argument('--train-batch-size', default=2000, type=int)
parser.add_argument('--clip-param', default=0.1, type=float, help='other implementations may refer as clip_ratio')
parser.add_argument('--action-repeat', default=4, type=int)
parser.add_argument('--latent-size', default=32, type=int)
parser.add_argument('--verbose', default=True, type=bool)
args = parser.parse_args()


######## Env Setting ###########
#def process_obs(obs): # a single frame (96, 96, 3) for CarRacing
    #obs = cv2.resize(obs[:84, :, :], dsize=(64,64), interpolation=cv2.INTER_NEAREST)
    #return np.transpose(obs, (2,0,1))

def process_obs(obs): # a single frame (96, 96, 3) for CarRacing
    obs = np.ascontiguousarray(obs, dtype=np.float32) / 255
    obs = cv2.resize(obs[:84, :, :], dsize=(64,64), interpolation=cv2.INTER_NEAREST)
    return np.transpose(obs, (2,0,1))

def choose_env(worker_index):
    return 'CarRacing-v0'

class MyEnvRewardWrapper(gym.Env):
    def __init__(self, env_config):
        self.env = gym.make(choose_env(env_config.worker_index))
        self.env.seed(args.seed)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.observation_space = Box(low=0, high=255, shape=(3,64,64), dtype=self.env.observation_space.dtype)

    def reset(self):
        obs = self.env.reset()
        processed_obs = process_obs(obs)
        return processed_obs

    def step(self, action):
        action[0] = action[0]*2-1
        rewards = 0
        for i in range(args.action_repeat):
            obs, reward, done, info = self.env.step(action)
            reward = (-0.1 if reward < 0 else reward)
            rewards += reward
            if done:
                break
        processed_obs = process_obs(obs)
        return processed_obs, rewards, done, info

register_env("myenv", lambda config: MyEnvRewardWrapper(config))

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



######## Model Setting ##########
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
    def __init__(self, class_latent_size = 8, content_latent_size = 32, input_channel = 3, flatten_size = 1024):
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

class MyModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        custom_config = model_config["custom_options"]
        self.policy_type = custom_config['policy_type']
        latent_size = custom_config['latent_size']

        # train policy with end-to-end training
        if self.policy_type == 'end_to_end':
            print('end-to-end')
            latent_size = 32
            self.main = EncoderE(class_latent_size = 8, content_latent_size = 16, input_channel = 3, flatten_size = 1024)

            if custom_config['encoder_path'] is not None:
                # saved checkpoints could contain extra weights such as linear_logsigma 
                weights = torch.load(custom_config['encoder_path'], map_location=torch.device('cpu'))
                for k in list(weights.keys()):
                    if k not in self.main.state_dict().keys():
                        del weights[k]
                self.main.load_state_dict(weights)
                print("Loaded Weights")
            else:
                print("No Load Weights")
        
        # train policy invariant representation
        elif self.policy_type == 'invar':
            print('invariant')
            latent_size = 32
            self.main = EncoderI(latent_size=latent_size)

            if custom_config['encoder_path'] is not None:
                # saved checkpoints could contain extra weights such as linear_logsigma 
                weights = torch.load(custom_config['encoder_path'], map_location=torch.device('cpu'))
                for k in list(weights.keys()):
                    if k not in self.main.state_dict().keys():
                        del weights[k]
                self.main.load_state_dict(weights)
                print("Loaded Weights")
            else:
                print("No Load Weights")

        # train policy invariant representation
        elif self.policy_type == 'adagvae':
            print('adagvae')
            latent_size = 32
            self.main = Encoder(latent_size=latent_size)

            if custom_config['encoder_path'] is not None:
                # saved checkpoints could contain extra weights such as linear_logsigma 
                weights = torch.load(custom_config['encoder_path'], map_location=torch.device('cpu'))
                for k in list(weights.keys()):
                    if k not in self.main.state_dict().keys():
                        del weights[k]
                self.main.load_state_dict(weights)
                print("Loaded Weights")
            else:
                print("No Load Weights")

        # train policy entangle representation
        elif self.policy_type == 'repr':
            print('entangle')
            latent_size = 32
            self.main = Encoder(latent_size=latent_size)

            if custom_config['encoder_path'] is not None:
                # saved checkpoints could contain extra weights such as linear_logsigma 
                weights = torch.load(custom_config['encoder_path'], map_location=torch.device('cpu'))
                for k in list(weights.keys()):
                    if k not in self.main.state_dict().keys():
                        del weights[k]
                self.main.load_state_dict(weights)
                print("Loaded Weights")
            else:
                print("No Load Weights")

        # train policy disentangled representation
        elif self.policy_type == 'disent':
            print('disentangle')
            class_latent_size = 8
            content_latent_size = 32
            latent_size = class_latent_size + content_latent_size
            self.main = EncoderD(class_latent_size, content_latent_size)

            if custom_config['encoder_path'] is not None:
                # saved checkpoints could contain extra weights such as linear_logsigma 
                weights = torch.load(custom_config['encoder_path'], map_location=torch.device('cpu'))
                for k in list(weights.keys()):
                    if k not in self.main.state_dict().keys():
                        del weights[k]
                self.main.load_state_dict(weights)
                print("Loaded Weights")
            else:
                print("No Load Weights")
            
       # train policy causal representation
        elif self.policy_type == 'ilcm_reduce_dim':
            print('ilcm reduce dim')
            latent_size = 32
            self.main = create_model()

            if custom_config['encoder_path'] is not None:
                # saved checkpoints could contain extra weights such as linear_logsigma 
                weights = torch.load(custom_config['encoder_path'], map_location=torch.device('cpu'))
                for k in list(weights.keys()):
                    if k not in self.main.state_dict().keys():
                        del weights[k]
                self.main.load_state_dict(weights)
                print("Loaded Weights")
            else:
                print("No Load Weights")

       # train policy causal representation
        elif self.policy_type == 'ilcm':
            print('causal')
            latent_size = 6
            self.encoder = create_model_reduce_dim()
            self.main = create_ilcm()

            if custom_config['encoder_path'] is not None:
                # saved checkpoints could contain extra weights such as linear_logsigma 
                weights = torch.load(custom_config['encoder_path'], map_location=torch.device('cpu'))
                for k in list(weights.keys()):
                    if k not in self.encoder.state_dict().keys():
                        del weights[k]
                self.encoder.load_state_dict(weights)
                print("Loaded Weights Encoder")
            else:
                print("No Load Weights")

            if custom_config['ilcm_path'] is not None:
                # saved checkpoints could contain extra weights such as linear_logsigma 
                weights = torch.load(custom_config['ilcm_path'], map_location=torch.device('cpu'))
                for k in list(weights.keys()):
                    if k not in self.main.state_dict().keys():
                        del weights[k]
                self.main.load_state_dict(weights)
                print("Loaded Weights Main")
            else:
                print("No Load Weights")
    
        # train policy no encoder
        elif self.policy_type == 'ppo'  or self.policy_type == 'augm':
            print('ppo')
            latent_size = 2*2*256
            self.cnn_base = nn.Sequential(
                    nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
                    nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                    nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
                    nn.Conv2d(128, 256, 4, stride=2), nn.ReLU()
                )
            self.apply(self._weights_init)

        '''
        self.main = Encoder(latent_size=latent_size)
    
        if custom_config['encoder_path'] is not None:
            # saved checkpoints could contain extra weights such as linear_logsigma 
            weights = torch.load(custom_config['encoder_path'], map_location=torch.device('cpu'))
            for k in list(weights.keys()):
                if k not in self.main.state_dict().keys():
                    del weights[k]
            self.main.load_state_dict(weights)
            print("Loaded Weights")
        else:
            print("No Load Weights")
        '''
        
        self.critic = nn.Sequential(nn.Linear(latent_size, 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU(), nn.Linear(300, 1))
        self.actor = nn.Sequential(nn.Linear(latent_size, 400), nn.ReLU(), nn.Linear(400, 300), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(300, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(300, 3), nn.Softplus())
        self._cur_value = None
        self.train_encoder = custom_config['train_encoder']
        print("Train Encoder: ", self.train_encoder)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        if self.policy_type == 'ppo' or self.policy_type == 'augm':
            x = self.cnn_base(input_dict['obs'].float())
            features = x.view(x.size(0), -1)
        elif self.policy_type == 'disent':
            content, _, style = self.main(input_dict['obs'].float())
            features = torch.cat([content, style], dim=1)
            if not self.train_encoder:
                features = features.detach() 
        elif self.policy_type == 'repr':
            features, _, = self.main(input_dict['obs'].float())
            if not self.train_encoder:
                features = features.detach() 
        elif self.policy_type == 'adagvae':
            features, _, = self.main(input_dict['obs'].float())
            if not self.train_encoder:
                features = features.detach() 
        elif self.policy_type == 'invar':
            features = self.main(input_dict['obs'].float())
            if not self.train_encoder:
                features = features.detach() 
        elif self.policy_type == 'ilcm_reduce_dim':
            features, _ = self.main.mean_std(input_dict['obs'].float())
            if not self.train_encoder:
                features = features.detach() 
        elif self.policy_type == 'ilcm':
            with torch.no_grad():
                z, _ = self.encoder.encoder.mean_std(input_dict['obs'].float())
                features = self.main.encode_to_causal(z)
        '''
        features = self.main(input_dict['obs'].float())
        if not self.train_encoder:
            features = features.detach()  # not train the encoder
        '''
        actor_features = self.actor(features)
        alpha = self.alpha_head(actor_features)+1
        beta = self.beta_head(actor_features)+1
        logits = torch.cat([alpha, beta], dim=1)
        self._cur_value = self.critic(features).squeeze(1)

        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, 'Must call forward() first'
        return self._cur_value

ModelCatalog.register_custom_model("mymodel", MyModel)

############ Distribution Setting ##############
class MyDist(ActionDistribution):
    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return 6

    def __init__(self, inputs, model):
        super(MyDist, self).__init__(inputs, model)
        self.dist = Beta(inputs[:, :3], inputs[:, 3:])

    def sample(self):
        self.sampled_action = self.dist.sample()
        return self.sampled_action

    def deterministic_sample(self):
        return self.dist.mean

    def sampled_action_logp(self):
        return self.logp(self.sampled_action)

    def logp(self, actions):
        return self.dist.log_prob(actions).sum(-1)

    # refered from https://github.com/pytorch/pytorch/blob/master/torch/distributions/kl.py
    def kl(self, other):
        p, q = self.dist, other.dist
        sum_params_p = p.concentration1 + p.concentration0
        sum_params_q = q.concentration1 + q.concentration0
        t1 = q.concentration1.lgamma() + q.concentration0.lgamma() + (sum_params_p).lgamma()
        t2 = p.concentration1.lgamma() + p.concentration0.lgamma() + (sum_params_q).lgamma()
        t3 = (p.concentration1 - q.concentration1) * torch.digamma(p.concentration1)
        t4 = (p.concentration0 - q.concentration0) * torch.digamma(p.concentration0)
        t5 = (sum_params_q - sum_params_p) * torch.digamma(sum_params_p)
        return (t1 - t2 + t3 + t4 + t5).sum(-1)

    def entropy(self):
        return self.dist.entropy().sum(-1)



ModelCatalog.register_custom_action_dist("mydist", MyDist)


########### Do Training #################
def main():
    seed_everything(args.seed)
    print(f'policy type is {args.policy_type}, do not forget to change the name of model_save_path')
    print(f'ray adress is {args.ray_adress}')
    ray.init(address=args.ray_adress, redis_password='5241590000000000')
    
    #  Hyperparameters of PPO are not well tuned. Most of them refer to https://github.com/xtma/pytorch_car_caring/blob/master/train.py
    trainer = PPOTrainer(env="myenv", config={
        "use_pytorch": True,
        "model":{"custom_model":"mymodel", 
                "custom_options":{'encoder_path':args.encoder_path, 'ilcm_path':args.ilcm_path,'train_encoder':args.train_encoder, 'latent_size':args.latent_size, 'policy_type':args.policy_type},
                "custom_action_dist":"mydist",
                },
        "env_config":{'game':'CarRacing'},
        "num_workers":args.num_workers,
        "num_envs_per_worker":args.num_envs_per_worker,
        "num_gpus":args.num_gpus,
        "use_gae":args.use_gae,
        "batch_mode":args.batch_mode,
        "vf_loss_coeff":args.vf_loss_coeff,
        "vf_clip_param":args.vf_clip_param,
        "lr":args.lr,
        "kl_coeff":args.kl_coeff,
        "num_sgd_iter":args.num_sgd_iter,
        "grad_clip":args.grad_clip,
        "clip_param":args.clip_param,
        "rollout_fragment_length":args.rollout_fragment_length,
        "train_batch_size":args.train_batch_size,
        "sgd_minibatch_size":args.sgd_minibatch_size,
        'seed': args.seed 
        })

    for i in range(args.train_epochs):
        trainer.train()
        if i %  args.save_freq == 0:
            print('Ep {}'.format(i))
            weights = trainer.get_policy().get_weights()
            torch.save(weights, args.model_save_path)

    weights = trainer.get_policy().get_weights()
    torch.save(weights, args.model_save_path)
    trainer.stop()

if __name__ == '__main__':
    main()