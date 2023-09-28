# Have not been trained on the same number of epoch so consider only adaptation to domain
agent_trained_on_data_augmentation = [-24.620903048672982, -25.41818601086693, -26.945765081590313, -25.321616931391972]
agent_trained_on_original_domain = [721.9011222061245, 80.24276590317983, -7.210216459967674, -13.8644701205344]

# Not train with ray (less consistent on each run)
agent_trained_on_invariant_representation = [684.4934927062735, 680.8247638617838, 633.1407370048859, 645.798978265601]
agent_trained_on_disentangle_representation = [600.158572301402, 577.058912171658, 651.7723555025151, 587.2040231131028]
agent_trained_on_entangle_representation = [416.885245474459, 361.56662124266273, 258.2663828142932, 162.11691966133102]

# results with ray
agent_trained_on_invariant_representation = [761.0100383665064, 775.3111107572336, 741.9157497145322, 500.55210311850203]
agent_trained_on_disentangle_representation = [695.5063339513786, 665.1377370916099, 663.785677322686, 497.6832238785365]
agent_trained_on_entangle_representation = [802.8633645461862, 536.285681591151, 508.06234930118137, 222.1848365990127]
agent_trained_on_adagvae_representation = [716.4183771345885, 212.26305866858613, 220.83392546937844, 0.8568826616382027]

if __name__ == '__main__':
    print( f'data augmenation results : {agent_trained_on_data_augmentation.mean()}, normal ppo results : {agent_trained_on_original_domain.mean()}, invariant representation results : {agent_trained_on_invariant_representation.mean()}, disentangle representation results : {agent_trained_on_disentangle_representation.mean()}, entangle representation results : {agent_trained_on_entangle_representation.mean()}')