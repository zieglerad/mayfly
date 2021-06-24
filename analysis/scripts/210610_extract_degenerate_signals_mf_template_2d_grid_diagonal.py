import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os
import pandas as pd

working_dir = '/home/az396/project/mayfly'
signal_path = 'data/signals'
template_path = 'data/templates'
simulation_name = '210604_epa_template_grid'

#signal_metadata = 

analysis_date = '210610'
plot_path = '/home/az396/project/mayfly/analysis/plotting/plots'


result_path = '/home/az396/project/mayfly/analysis/results'
result_date = '210604'
result_name = 'epa_template_grid_self_scores'

path2signals = f'{working_dir}/{signal_path}/{simulation_name}'

result_file = f'{result_date}_{result_name}.pkl'

new_result_name = f'{analysis_date}_{result_name}_diagonal_on_grid_degeneracies.pkl'

with open(os.path.join(result_path, result_file), 'rb') as infile:
    result = pd.DataFrame(pkl.load(infile))
    

    
def GetDegenerateSignals2DMFScore(template_x_key, template_y_key, fix_sig_x_key, fix_sig_y_key, signal_index, result, path2signals):

    h_x_list = result[template_x_key].unique()
    h_y_list = result[template_y_key].unique()
    x_x_list = result[fix_sig_x_key].unique()
    x_y_list = result[fix_sig_y_key].unique()
    
    grid_x_size = h_x_list.size
    grid_y_size = h_y_list.size
    grid = np.zeros((grid_y_size, grid_x_size))
    
    signal_x = x_x_list[signal_index]
    signal_y = x_y_list[signal_index]
    
    #print(signal_x, signal_y)
    
    selected_result = result[
                        (result[fix_sig_x_key] == signal_x) & 
                        (result[fix_sig_y_key] == signal_y) 
                            ]

    for i, h_x in enumerate(h_x_list):
        for j, h_y in enumerate(h_y_list):

            grid[i, j] = selected_result[
                        (result[template_x_key] == h_x) & 
                        (result[template_y_key] == h_y) 
                        ]['T']
    degenerate_templates = np.argwhere(grid >= 6)
    
    degenerate_template_x = h_x_list[degenerate_templates[:, 0]]
    degenerate_template_y = h_y_list[degenerate_templates[:, 1]]
    
    #print(degenerate_templates, degenerate_template_x, degenerate_template_y)



    degenerate_signal_templates = {'signal': {}, 'template': []}
    for i in range(len(degenerate_template_x)):
        template_info = {}
        degenerate_result = result[
                                (result[template_x_key] == degenerate_template_x[i]) & 
                                (result[template_y_key] == degenerate_template_y[i]) & 
                                (result[fix_sig_x_key] == signal_x) & 
                                (result[fix_sig_y_key] == signal_y) 
                                ]
        if i == 0:
            signal_params = []
            for sig_key in ['x_r', 'x_pa', 'x_E', 'x_z']:
                #print(degenerate_result[sig_key][0])
                signal_params.append(degenerate_result[sig_key].iloc[0])
                degenerate_signal_templates['signal'].update({sig_key: degenerate_result[sig_key].iloc[0]})
                
            signal_file = f'herc_sim_angle{signal_params[1]:2.4f}_rad{signal_params[0]:1.3f}_energy{signal_params[2]:5.2f}_axial{signal_params[3]:1.3f}.pkl'
            with open(os.path.join(path2signals, signal_file), 'rb') as infile:
                signal_ts = pkl.load(infile)
                degenerate_signal_templates['signal'].update({'x': signal_ts})
        
        template_params = []
        for temp_key in ['h_r', 'h_pa', 'h_E', 'h_z']:
            #print(degenerate_result[temp_key])
            template_params.append(degenerate_result[temp_key].iloc[0])
            template_info.update({temp_key: degenerate_result[temp_key].iloc[0]})
        template_file = f'herc_sim_angle{template_params[1]:2.4f}_rad{template_params[0]:1.3f}_energy{template_params[2]:5.2f}_axial{template_params[3]:1.3f}.pkl'
        
        with open(os.path.join(path2signals, template_file), 'rb') as infile:
                template_ts = pkl.load(infile)
                template_info.update({'h': template_ts})
        degenerate_signal_templates['template'].append(template_info)
        
    return degenerate_signal_templates

#print(result['x_pa'].unique().shape)
list_of_degenerate_signal_template_pairs = []
for i in range(result['x_pa'].unique().shape[0]):
    print(i)
    degenerate_signal_template_pairs = GetDegenerateSignals2DMFScore('h_E', 'h_pa', 'x_E', 'x_pa', i, result, path2signals)
    print(len(degenerate_signal_template_pairs['template']))
    list_of_degenerate_signal_template_pairs.append(degenerate_signal_template_pairs)
    
with open(os.path.join(result_path, new_result_name), 'wb') as outfile:
    pkl.dump(list_of_degenerate_signal_template_pairs, outfile)

