import numpy as np
import torch

# Have not been trained on the same number of epoch so consider only adaptation to domain
agent_trained_on_data_augmentation = [-24.620903048672982, -25.41818601086693, -26.945765081590313, -25.321616931391972]
agent_trained_on_original_domain = [721.9011222061245, 80.24276590317983, -7.210216459967674, -13.8644701205344]

# Not train with ray (Random Augmentation) - test domain [None, 0.3, -0.2, 0.1] - All train for 5000 epoch
agent_trained_on_invariant_representation = [[357.49002642250804, 379.1604469035519, 408.78447829162286, 402.9745414247528], [429.1342660014802, 412.01305146731494, 436.4789614855995, 393.33662905509846], [419.49502089309533, 490.33056816503455, 486.7159041327404, 498.3617580697298]]
agent_trained_on_disentangle_representation = [[366.1882744313045, 407.19786031622453, 377.44117265147986, 377.66466877080876], [461.63074122390856, 487.9204626275899, 502.710596168382, 492.1700470285273], [534.2856234210471, 564.442597412412, 515.6513746734075, 554.3395585879105]]
agent_trained_on_entangle_representation = [[523.6397410819818, 557.4604699440828, 506.0635103432316, 484.24722307810424],[273.682779524141, 61.85807601709133, 258.91913396523523, 342.3876199636137],[515.2663774822036, 554.1272241368389, 512.9561272832136, 494.1961392506356]]
agent_trained_on_adagvae_representation = [[239.8011840723293, -26.299627172654432, 178.9234954445809, 277.7391986676965], [491.7143404098165, 462.3298847251641, 514.3687401336632, 417.1492952836834], [350.723763575106, 249.63211205272023, 380.19376260989344, 302.1656174935559]]
agent_trained_on_causal_representation = [[311.8282810337814, 196.86379916337674, 354.8195476009928, 313.39014291419784], [204.79623211958503, 49.26789178866119, 308.83362175230974, 155.78020032526092], [256.9801482623451, 28.824649882790574, 368.56832716191514, 128.44584777674942]]
agent_trained_on_ppo = [[-73.02260773938087, -68.69456422341625, -72.48020238593239, -68.88445774603609], [-77.58777584326201, -80.62909644177121, -81.78209684354833, -78.32366193590235], [-74.11349247989143, -60.75666137539662, -59.91206769652971, -63.32724684245841]]

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