import numpy as np
import torch

# Have not been trained on the same number of epoch so consider only adaptation to domain
agent_trained_on_data_augmentation = [-24.620903048672982, -25.41818601086693, -26.945765081590313, -25.321616931391972]
agent_trained_on_original_domain = [721.9011222061245, 80.24276590317983, -7.210216459967674, -13.8644701205344]

# Not train with ray (Random Augmentation) - test domain [None, 0.3, -0.2, 0.1] - All train for 5000 epoch
agent_trained_on_invariant_representation = [[379.64598601207837, 410.24668705837576, 366.615245199624, 365.05887978018853], [351.10022306030294, 402.31385627172943, 566.4880409066951, 412.96615709112115], [460.9807129670906, 472.63169812687937, 318.255444915345, 486.7593807103897], [386.39162602886216, 418.72182810101464, 91.48093300480315, 406.5156794990231], [529.1790974289867, 488.64438662962203, 655.7096153532718, 453.227512330264]]
agent_trained_on_disentangle_representation = [[356.9301000258691, 322.58829665680753, 410.6855677289108, 359.1285685662904], [451.57177099969687, 495.30793279427184, 456.5302742511394, 427.9688211937453], [546.0215417719102, 482.17418298480777, 425.8542104721491, 613.5189983299921], [608.6843242972889, 533.0603086290462, 534.7283940018575, 664.645026488756]]
agent_trained_on_entangle_representation = [[501.0451283310363, 459.0942110171646, 523.3508667740612, 465.0869520982331],[268.78415753026826, 259.67361264548015, 279.81704667739615, 304.4085909940728],[487.8478455504944, 522.2309316089984, 535.0925329236576, 505.6349477411844], [454.1047760486836, 429.2754203074343, 493.3677384847762, 467.12022645468227], [423.45367097124034, 446.2226868909319, 439.7265097849068, 427.5171866610601]]
agent_trained_on_adagvae_representation = [[400.2201889001621, 414.9448390210974, 392.6363924943499, 388.1762990809516], [491.7143404098165, 462.3298847251641, 514.3687401336632, 417.1492952836834], [350.723763575106, 249.63211205272023, 380.19376260989344, 302.1656174935559], [378.5129920759176, 360.42180870144307, 427.70730209320254, 356.3480981534004], [410.6269346815547, 355.80412670856924, 172.48293924500612, 277.59458554468915]]
agent_trained_on_causal_representation = [[423.3520126690767, 434.50279923099333, 428.73945346005667, 516.4276464388998], [402.90410081805317, 418.18808050046624, 377.9950330275092, 350.8457181634833], [285.35213590832115, 352.7949025639728, 324.8835894737117, 332.5749665156527], [565.6710099851065, 612.1042006514148, 487.45444671013627, 388.2384494330324], [543.4613699242391, 520.5757780474868, 543.1689840218689, 556.3882554884265]]
agent_trained_on_ppo = [[212.15913917688184, -24.84890110296091, -24.844899574738903, -23.601004408009818], [126.92468711480707, -24.853528583704485, -26.324572356606772, -27.601451436070473], [496.0666727824995, -52.32877577864299, -69.74587449951927, -71.15153974656208], [51.20935830888539, -27.53812033243108, -28.474502881505277, -25.669540975185555], [108.69023327058736, -23.520085148790194, -27.09961977570937, -24.36157736501678]]

"""
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