import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os
import pandas as pd

working_dir = '/home/az396/project/mayfly'
signal_path = 'data/signals'
template_path = 'data/templates'
simulation_name = '210610_epa_grid'

#signal_metadata = 

analysis_date = '210611'
plot_path = '/home/az396/project/mayfly/analysis/plotting/plots'


result_path = '/home/az396/project/mayfly/analysis/results'
result_date = '210610'
result_name = 'epa_grid_self_scores'

result_file = f'{result_date}_{result_name}.pkl'

new_result_name = f'{analysis_date}_{result_name}_animation_matrices.pkl'

with open(os.path.join(result_path, result_file), 'rb') as infile:
    result = pd.DataFrame(pkl.load(infile))
    

    
def Compute2DMFScoreGrid(template_x_key, template_y_key, fix_sig_x_key, fix_sig_y_key, signal_index, result):

    h_x_list = result[template_x_key].unique()
    h_y_list = result[template_y_key].unique()
    x_x_list = result[fix_sig_x_key].unique()
    x_y_list = result[fix_sig_y_key].unique()
    
    grid_x_size = h_x_list.size
    grid_y_size = h_y_list.size
    grid = np.zeros((grid_y_size, grid_x_size))
    
    signal_x = x_x_list[signal_index]
    signal_y = x_y_list[signal_index]
    
    print(signal_x, signal_y)
    
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
                        
    return grid, signal_x, signal_y

#print(result['x_pa'].unique().shape)
grid_list = []
signal_x_list = []
signal_y_list = []
for i in range(result['x_pa'].unique().shape[0]):
    print(i)
    grid, signal_x, signal_y = Compute2DMFScoreGrid('h_E', 'h_pa', 'x_E', 'x_pa', i, result)
    grid_list.append(grid)
    signal_x_list.append(signal_x)
    signal_y_list.append(signal_y)
with open(os.path.join(result_path, new_result_name), 'wb') as outfile:
    pkl.dump({'grids': grid_list, 'sig_x': signal_x_list, 'sig_y': signal_y_list}, outfile)

