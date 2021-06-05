import numpy as np
import matplotlib.pyplot as plt
import os
import matchfilter as mf
import pickle as pkl


simulation_name = '21520_variable_epaz'
results_path = '/home/az396/project/matchedfiltering/analysis/results'
result_name = '210521_variable_epaz_self_scores.pkl'

result = mf.utils.ComputeSelf(simulation_name)

#result = {'sig_meta': signal_meta, 'temp_meta': template_meta, 'mf_score': result}


with open(os.path.join(results_path, result_name), 'wb') as outfile:
    pkl.dump(result, outfile)
    
    






