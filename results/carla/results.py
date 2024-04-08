import numpy as np
import torch

# source domain hard rain
agent_trained_on_invariant_representation = [[1738.665000215607, 88.56449534224649, 116.32442731473732, 36.90594397626156], [1351.7631143196413, 82.81167122483245, 91.04721908468554, 50.04272035856382]] #[]
agent_trained_on_disentangle_representation = [[1780.9270207450154, 72.08302958012432, 44.151138717159135, 29.63560030277359], [2105.3396638848935, 71.1405826980555, 83.05177880841003, 36.01636373358205]]#[]
agent_trained_on_entangle_representation = [[2157.56372396254, 489.02630165441934, 267.30350950939976, 181.30485129898904],[1951.8215431010212, 305.23375774259097, 167.85704567162105, 248.99695476713015]]#[]
agent_trained_on_adagvae_representation = [[1506.5921473207986, 986.0629600077498, 387.2703936392961, 413.3403345053072], [1718.7956843421628, 424.6309663285627, 194.72208543701035, 139.30106167742377]]#[]
#03_12
agent_trained_on_causal_representation = [[324.1002844282449, 101.80736463537313, 89.99827821136361, 258.045628846431], [105.99867010307771, 165.7840636030468, 148.40737595378317, 189.81035298077649]]#[]
#03_21
#agent_trained_on_causal_representation = [[], []]
#04_03
#agent_trained_on_causal_representation = [[76.7068422216544, 77.17725139341802, 87.16133156658006, 1103.401052146129], [2129.461302508318, 924.5570563120298, 170.30396057323225, 46.16495900065591]]

agent_trained_on_ppo = [[131.76821565174578, -141.01029408910796, 153.4283462852548, 168.69280872866676],  [-181.8142862790729, -181.8142862790729, -181.8142862790729, -181.8142862790729]]#[]

# source domain noon
agent_trained_on_invariant_representation = [[1813.146348788629, 333.3114258456634, 306.2129624333659, 69.20140163472834], [66.12712123565186, 141.72906459724817, 59.81889097822896, 41.45545226946983]]#[]
agent_trained_on_disentangle_representation = [[2092.971309231617, 240.05829570065958, 185.70536239581452, 50.36683102973813], [803.1944894627438, 403.64455802378103, 101.1926651740672, 54.431964108888124]]#[]
agent_trained_on_entangle_representation = [[1996.0767486648049, 135.4759650304513, 108.24318653416269, 195.55332566555268],[791.5067159410327, 130.4741208633065, 122.12633966012456, 195.24493068308703]]#[]
agent_trained_on_adagvae_representation = [[2254.1652002620203, 100.34116485875003, 168.645677081323, 632.4638828308055], [1679.5133024925258, 55.84065487292209, 61.13751232996178, 59.35390677087978]]#[]
#03_21
#agent_trained_on_causal_representation = [[506.1483524795235, 14.70690778822322, 92.21690331595087, -127.96511863973294], [709.3569440229955, 59.3744682088932, 55.7086080696856, 2.7712970330728846]]
#03_12
agent_trained_on_causal_representation = [[254.43216536676442, 33.69502321255962, -149.1749713352421, 310.34795239814054], [404.2411228229642, 41.452963659221986, 60.67085367211571, 270.596470945285]]#[]
#04_03
#agent_trained_on_causal_representation = [[1723.8371466488757, 143.39086922175775, 84.35656086629729, 186.67062069123813], [550.8030150660315, 140.48141519255748, 78.45405592479113, 69.02652643557946]]

agent_trained_on_ppo = [[],  []]#[]

#source domain night
agent_trained_on_invariant_representation = [[928.1401088912744, 129.6066502630199, 221.00904907453105, 355.3845198608319], [476.7030372517571, 102.1609486628648, 137.9726633083428, -50.84985200292231]]#[]
agent_trained_on_disentangle_representation = [[828.7323031711488, 94.88233414010861, 178.5321079369406, 164.66325907856566], [1417.3950211636052, 85.74646338273878, 112.81905975312131, 136.57222304384754]]#[]
agent_trained_on_entangle_representation = [[539.247805924939, 324.5736447185336, 395.8880304441573, 170.0267535177955], [986.5439696420981, 769.7168710113046, 1339.751197510794, 897.6312238209655]]#[]
agent_trained_on_adagvae_representation = [[621.5955977879174, 660.970751429195, 53.179785026351304, 83.84675048105035], [879.2114899903265, 660.970751429195,  64.20757466245848, 134.56383954700465]]#[]
#03_21
#agent_trained_on_causal_representation = [[256.5556888855548, 72.98889080370684, -17.575069883078065, -41.9584961902375], [ 434.7638746702859, 6.150767133141157, -37.442090304629055, -36.78323797590134]]
#03_12
agent_trained_on_causal_representation = [[233.37877486418884, 64.35500230643495, 207.44216138131134, 156.03400081338185], [315.6632242665308, 49.88030215206465, -41.39768903001605, 124.12316641556556]]#[]
#04_03
#agent_trained_on_causal_representation = [[461.91699609610816, 8.15258142267178, -42.74783605661101 , -14.894773174224108], []]
agent_trained_on_ppo = [[],  []]#[]


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