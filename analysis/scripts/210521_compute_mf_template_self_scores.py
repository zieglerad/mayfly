import numpy as np
import matplotlib.pyplot as plt
import os
import mayfly as mf
import pickle as pkl


simulation_name = '210610_epa_grid'
results_path = '/home/az396/project/mayfly/analysis/results'
result_name = '210610_epa_grid_self_scores.pkl'

result = mf.utils.ComputeSelf(simulation_name)

#result = {'sig_meta': signal_meta, 'temp_meta': template_meta, 'mf_score': result}


with open(os.path.join(results_path, result_name), 'wb') as outfile:
    pkl.dump(result, outfile)
    
    






