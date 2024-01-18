# Have not been trained on the same number of epoch so consider only adaptation to domain
agent_trained_on_data_augmentation = [-24.620903048672982, -25.41818601086693, -26.945765081590313, -25.321616931391972]
agent_trained_on_original_domain = [721.9011222061245, 80.24276590317983, -7.210216459967674, -13.8644701205344]

# Not train with ray (less consistent on each run)
agent_trained_on_invariant_representation = [684.4934927062735, 680.8247638617838, 633.1407370048859, 645.798978265601]
agent_trained_on_disentangle_representation = [600.158572301402, 577.058912171658, 651.7723555025151, 587.2040231131028]
agent_trained_on_entangle_representation = [416.885245474459, 361.56662124266273, 258.2663828142932, 162.11691966133102]
agent_trained_on_causal_representation = [328.4191859448607, 355.5630750734049, 307.9752179094669, 71.01303073421175]
agent_trained_on_ppo_drq = [547.6708894925528, 4.95862531090973, 8.734605977916523, -2.3886857452170664]
agent_trained_on_ppo_curl = [461.48702168096935, -30.144663675603393, -42.82130370925729, -92.59999037860217]

# results with ray
agent_trained_on_invariant_representation = [761.0100383665064, 775.3111107572336, 741.9157497145322, 500.55210311850203]
agent_trained_on_disentangle_representation = [695.5063339513786, 665.1377370916099, 663.785677322686, 497.6832238785365]
agent_trained_on_entangle_representation = [802.8633645461862, 536.285681591151, 508.06234930118137, 222.1848365990127]
agent_trained_on_adagvae_representation = [716.4183771345885, 212.26305866858613, 220.83392546937844, 0.8568826616382027]
agent_trained_on_causal_representation = [414.35831732334265, 430.3686266021296, 400.1234243744409, 67.54746816494185]
agent_trained_on_ppo = [721.9011222061245, 80.24276590317983, -7.210216459967674, -13.8644701205344]

# evaluate repr (latent dim 32)
vae :  {"recon_loss_mean": 0.005601471077908974, "gaussian_total_corr_mean": 25.645086785087102, "gaussian_wasserstein_corr_mean": 0.6171508199911979, "mututal_info_score_mean": 1.2639963384136323}
cycle_vae: {"recon_loss_mean": 0.00386427358797945, "gaussian_total_corr_mean": 47.27601413264951, "gaussian_wasserstein_corr_mean": 1.3021744208824015, "mututal_info_score_mean": 1.4382852184078718}
adagvae:  {"recon_loss_mean": 0.01800146347631027, "gaussian_total_corr_mean": 52.34963919778118, "gaussian_wasserstein_corr_mean": 19.308044050745732, "mututal_info_score_mean": 1.471662063740292}
ilcm:  {"recon_loss_mean": 0.01894488625947644, "gaussian_total_corr_mean": 38.60906868349538, "gaussian_wasserstein_corr_mean": 0.40966980325322205, "mututal_info_score_mean": 1.5066403540341613}

if __name__ == '__main__':
    print( f'data augmenation results : {agent_trained_on_data_augmentation.mean()}, normal ppo results : {agent_trained_on_original_domain.mean()}, invariant representation results : {agent_trained_on_invariant_representation.mean()}, 
    disentangle representation results : {agent_trained_on_disentangle_representation.mean()}, entangle representation results : {agent_trained_on_entangle_representation.mean()},
    disentangle adagvae results: {agent_trained_on_adagvae_representation}, disentangle ilcm results: {agent_trained_on_ilcm_representation}')