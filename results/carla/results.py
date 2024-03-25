import numpy as np
import torch

# source domain hard rain
agent_trained_on_invariant_representation = [[1738.665000215607, 88.56449534224649, 116.32442731473732, 36.90594397626156], [1351.7631143196413, 82.81167122483245, 91.04721908468554, 50.04272035856382]]
agent_trained_on_disentangle_representation = [[1780.9270207450154, 72.08302958012432, 44.151138717159135, 29.63560030277359], [2105.3396638848935, 71.1405826980555, 83.05177880841003, 36.01636373358205]]
agent_trained_on_entangle_representation = [[2157.56372396254, 489.02630165441934, 267.30350950939976, 181.30485129898904],[1951.8215431010212, 305.23375774259097, 167.85704567162105, 248.99695476713015]]
agent_trained_on_adagvae_representation = [[1506.5921473207986, 986.0629600077498, 387.2703936392961, 413.3403345053072], [1718.7956843421628, 424.6309663285627, 194.72208543701035, 139.30106167742377]]
agent_trained_on_causal_representation = [[324.1002844282449, 101.80736463537313, 89.99827821136361, 258.045628846431], [105.99867010307771, 165.7840636030468, 148.40737595378317, 189.81035298077649]]
agent_trained_on_ppo = [[131.76821565174578, -141.01029408910796, 153.4283462852548, 168.69280872866676],  [-181.8142862790729, -181.8142862790729, -181.8142862790729, -181.8142862790729]]


if __name__ == '__main__':
    #print( f'data augmenation results : {agent_trained_on_data_augmentation.mean()}, normal ppo results : {agent_trained_on_original_domain.mean()}, invariant representation results : {agent_trained_on_invariant_representation.mean()}, 
    #disentangle representation results : {agent_trained_on_disentangle_representation.mean()}, entangle representation results : {agent_trained_on_entangle_representation.mean()},
    #disentangle adagvae results: {agent_trained_on_adagvae_representation}, disentangle ilcm results: {agent_trained_on_ilcm_representation}')
    print(torch.tensor(agent_trained_on_invariant_representation).shape)
    print('invariant', torch.mean(torch.tensor(agent_trained_on_invariant_representation), axis=0), torch.std(torch.tensor(agent_trained_on_invariant_representation), axis=0))
    print('cycle_vae',torch.mean(torch.tensor(agent_trained_on_disentangle_representation), axis=0), torch.std(torch.tensor(agent_trained_on_disentangle_representation), axis=0))
    print('vae', torch.mean(torch.tensor(agent_trained_on_entangle_representation), axis=0), torch.std(torch.tensor(agent_trained_on_entangle_representation), axis=0))
    print('adagvae', torch.mean(torch.tensor(agent_trained_on_adagvae_representation), axis=0), torch.std(torch.tensor(agent_trained_on_adagvae_representation), axis=0))
    print('ilcm', torch.mean(torch.tensor(agent_trained_on_causal_representation), axis=0), torch.std(torch.tensor(agent_trained_on_causal_representation), axis=0))
    print('ppo', torch.mean(torch.tensor(agent_trained_on_ppo), axis=0), torch.std(torch.tensor(agent_trained_on_ppo), axis=0))