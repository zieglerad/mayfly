import numpy as np
import matplotlib.pyplot as plt
import os
import mayfly as mf
import pickle as pkl


template_name = '210604_epa_template_grid'
signal_name = '210609_epa_fixed_pa_test_2'

results_path = '/home/az396/project/mayfly/analysis/results'
result_name = '210609_epa_template_bank_sim_fixed_pa_test.pkl'

result = mf.utils.ComputeTemplateBank(signal_name, template_name, N_batch=5)

#result = {'sig_meta': signal_meta, 'temp_meta': template_meta, 'mf_score': result}


with open(os.path.join(results_path, result_name), 'wb') as outfile:
    pkl.dump(result, outfile)
