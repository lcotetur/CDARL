import numpy as np
import torch

# Have not been trained on the same number of epoch so consider only adaptation to domain
agent_trained_on_data_augmentation = [-24.620903048672982, -25.41818601086693, -26.945765081590313, -25.321616931391972]
agent_trained_on_original_domain = [721.9011222061245, 80.24276590317983, -7.210216459967674, -13.8644701205344]

# Not train with ray (Random Augmentation) - test domain [None, 0.3, -0.2, 0.1] - All train for 5000 epoch
agent_trained_on_invariant_representation = [[379.64598601207837, 410.24668705837576, 366.615245199624, 365.05887978018853], [351.10022306030294, 402.31385627172943, 566.4880409066951, 412.96615709112115], [460.9807129670906, 472.63169812687937, 318.255444915345, 486.7593807103897]]
agent_trained_on_disentangle_representation = [[356.9301000258691, 322.58829665680753, 410.6855677289108, 359.1285685662904], [451.57177099969687, 495.30793279427184, 456.5302742511394, 427.9688211937453], [546.0215417719102, 482.17418298480777, 425.8542104721491, 613.5189983299921]]
agent_trained_on_entangle_representation = [[501.0451283310363, 459.0942110171646, 523.3508667740612, 465.0869520982331],[268.78415753026826, 259.67361264548015, 279.81704667739615, 304.4085909940728],[487.8478455504944, 522.2309316089984, 535.0925329236576, 505.6349477411844]]
agent_trained_on_adagvae_representation = [[400.2201889001621, 414.9448390210974, 392.6363924943499, 388.1762990809516], [491.7143404098165, 462.3298847251641, 514.3687401336632, 417.1492952836834], [350.723763575106, 249.63211205272023, 380.19376260989344, 302.1656174935559]]
agent_trained_on_causal_representation = [[291.3721461008947, 281.3153373742436, 356.24953569365067, 278.42223773123374], [195.6373822841058, 305.1149495910714, 315.2119444372653, 338.7661305252416], [254.3443784395661, 276.5613170735966, 391.519029087043, 284.2942955432852]]
agent_trained_on_ppo = [[-69.38893662364192, -73.93506873615584, -74.79133125679466, -76.21802250638243],  [-77.74525918857994, -81.20739632175321, -82.90510908399439, -82.77109370938422], [-62.77417529840343, -72.13280232516735, -56.58609557859183, -52.03487983154636]]

"""
agent_trained_on_ppo_curl 

# Not train with ray (No Random Augmentation - 10 domain) - test domain []
agent_trained_on_invariant_representation 
agent_trained_on_disentangle_representation 
agent_trained_on_entangle_representation 
agent_trained_on_causal_representation 
agent_trained_on_ppo_drq 
agent_trained_on_ppo_curl 

# results with ray
agent_trained_on_invariant_representation = [761.0100383665064, 775.3111107572336, 741.9157497145322, 500.55210311850203]
agent_trained_on_disentangle_representation = [695.5063339513786, 665.1377370916099, 663.785677322686, 497.6832238785365]
agent_trained_on_entangle_representation = [802.8633645461862, 536.285681591151, 508.06234930118137, 222.1848365990127]
agent_trained_on_adagvae_representation = [716.4183771345885, 212.26305866858613, 220.83392546937844, 0.8568826616382027]
agent_trained_on_causal_representation = [414.35831732334265, 430.3686266021296, 400.1234243744409, 67.54746816494185]
agent_trained_on_ppo = [721.9011222061245, 80.24276590317983, -7.210216459967674, -13.8644701205344]
"""

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