import argparse
import numpy as np


def parse_args():
	parser = argparse.ArgumentParser()

	# algorithm
	parser.add_argument('--repr', default='adagvae_sac', type=str)
	parser.add_argument('--algo', default='sac', type=str)
	parser.add_argument('--img_stack', default=4, type=int)
	parser.add_argument('--action_repeat', default=8, type=int)
	parser.add_argument('--encoder_path', default=None, type=str)
	parser.add_argument('--training_step', default=500000, type=int)	
	
	# agent
	parser.add_argument('--sac_epoch', default=10, type=str)
	parser.add_argument('--max-grad-norm', default=0.5, type=float)
	parser.add_argument('--clip-param', default=0.1, type=float)
	parser.add_argument('--gamma', default=0.99, type=float)
	parser.add_argument('--discount', default=0.99, type=float)

	# actor
	parser.add_argument('--actor_lr', default=1e-3, type=float)
	parser.add_argument('--actor_beta', default=0.9, type=float)
	parser.add_argument('--actor_log_std_min', default=-10, type=float)
	parser.add_argument('--actor_log_std_max', default=2, type=float)
	parser.add_argument('--actor_update_freq', default=2, type=int)

	# critic
	parser.add_argument('--critic_lr', default=1e-3, type=float)
	parser.add_argument('--critic_beta', default=0.9, type=float)
	parser.add_argument('--critic_tau', default=0.01, type=float)
	parser.add_argument('--critic_target_update_freq', default=2, type=int)

	# architecture
	parser.add_argument('--num_shared_layers', default=11, type=int)
	parser.add_argument('--num_head_layers', default=0, type=int)
	parser.add_argument('--num_filters', default=32, type=int)
	parser.add_argument('--projection_dim', default=100, type=int)
	parser.add_argument('--encoder_tau', default=0.05, type=float)
	parser.add_argument('--latent_size', default=32, type=int)
	
	# entropy maximization
	parser.add_argument('--init_temperature', default=0.1, type=float)
	parser.add_argument('--alpha_lr', default=1e-4, type=float)
	parser.add_argument('--alpha_beta', default=0.5, type=float)

	# auxiliary tasks
	parser.add_argument('--aux_lr', default=1e-3, type=float)
	parser.add_argument('--aux_beta', default=0.9, type=float)
	parser.add_argument('--aux_update_freq', default=2, type=int)

	# soda
	parser.add_argument('--soda_batch_size', default=256, type=int)
	parser.add_argument('--soda_tau', default=0.005, type=float)

	# svea
	parser.add_argument('--svea_alpha', default=0.5, type=float)
	parser.add_argument('--svea_beta', default=0.5, type=float)

	# eval
	parser.add_argument('--eval_episodes', default=100, type=int)

	# general
	parser.add_argument('--seed', default=None, type=int)
	parser.add_argument('--save_dir', default='/home/mila/l/lea.cote-turcotte/CDARL/logs', type=str)
	parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')

	args = parser.parse_args()

	assert args.algo in {'sac', 'drq_sac', 'rad_sac', 'curl_sac', 'pad_sac', 'soda_sac', 'svea_sac'}, f'specified algorithm "{args.algo}" is not supported'
	assert args.repr in {'cycle_vae_sac', 'invar_sac', 'ilcm_sac', 'vae_sac', 'adagvae_sac'}, f'specified representation "{args.repr}" is not supported'

	assert args.seed is not None, 'must provide seed for experiment'

	if args.repr is not None:
		args.algo = 'sac'
		args.img_stack = 1
		if args.repr=='vae_sac':
			args.encoder_path = '/home/mila/l/lea.cote-turcotte/CDARL/representation/VAE/runs/carracing/2023-11-20/encoder_vae.pt'
			args.latent_size = 32
		elif args.repr=='ilcm_sac':
			args.encoder_path = '/home/mila/l/lea.cote-turcotte/CDARL/representation/ILCM/checkpoints/model_step_130000_carracing.pt'
			args.latent_size = 8
		elif args.repr=='cycle_vae_sac':
			args.encoder_path = '/home/mila/l/lea.cote-turcotte/CDARL/representation/CYCLEVAE/runs/carracing/2023-11-20/encoder_cycle_vae.pt'
			args.latent_size = 32
		elif args.repr=='invar_sac':
			args.encoder_path = '/home/mila/l/lea.cote-turcotte/CDARL/representation/CYCLEVAE/runs/carracing/2023-11-20/encoder_cycle_vae.pt'
			args.latent_size = 40
		elif args.repr=='adagvae_sac':
			args.encoder_path = '/home/mila/l/lea.cote-turcotte/CDARL/representation/ADAGVAE/checkpoints/encoder_adagvae_32.pt'
			args.latent_size = 32
	
	return args
