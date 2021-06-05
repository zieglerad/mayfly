import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os
import pandas as pd

working_dir = '/home/az396/project/matchedfiltering'
signal_path = 'data/signals'
template_path = 'data/templates'
simulation_name = '21520_variable_epaz'

#signal_metadata = 

plot_date = '210526'
plot_path = '/home/az396/project/matchedfiltering/analysis/plotting/plots'


result_path = '/home/az396/project/matchedfiltering/analysis/results'
result_date = '210521'
result_name = 'variable_epaz_self_scores'

result_file = f'{result_date}_{result_name}.pkl'

with open(os.path.join(result_path, result_file), 'rb') as infile:
    result = pd.DataFrame(pkl.load(infile))
#print(result['x_pa'].unique())
#print(result[(result['x_pa'] == 89.50) & (result['h_pa'] == 89.55)])

def PlotMFGrid(x_key, h_key, result, save_path, T_threshold = 5.0):

    x_list = result[x_key].unique()
    h_list = result[h_key].unique()
    grid_size = h_list.size
    grid = np.zeros((grid_size, grid_size))
    
    for i, x in enumerate(x_list):
        for j, h in enumerate(h_list):
            if i == j:
                N_total = result[(result[x_key] == x) & (result[h_key] == h)].shape[0]
                grid[i, j] = result[(result[x_key] == x) & (result[h_key] == h) & (result['T'] >= T_threshold)].shape[0] / N_total
                #grid[i, j] = result[(result[x_key] == x) & (result[h_key] == h) & (result['T'] >= T_threshold)]['T'].mean()
            elif i >= j:
                N_total = result[(result[x_key] == x) & (result[h_key] == h)].shape[0]
                grid[j, i] = grid[i, j] = result[(result[x_key] == x) & (result[h_key] == h) & (result['T'] >= T_threshold)].shape[0] / N_total
                #grid[j, i] = grid[i, j] = result[(result[x_key] == x) & (result[h_key] == h) & (result['T'] >= T_threshold)]['T'].mean()

    fig = plt.figure(figsize=(8,8))

    ax1 = plt.subplot(111)

    im1 = ax1.imshow(grid.T, cmap = 'inferno_r', extent = (h_list[0], h_list[-1], x_list[-1], x_list[0]))
    cbar = plt.colorbar(im1)
    ax1.set_title(f'Fraction of Signal-Template Combinations\n above MF Score T = {T_threshold}', size=16)
    ax1.set_ylabel(x_key, size=16)
    ax1.set_xlabel(h_key, size=16)
    cbar.ax.tick_params(labelsize = 14)
    #cbar.set_label('', size=16)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #ax1.set_ylim(18598, 18595)

    plt.savefig(os.path.join(save_path))

plot_name1 = f'{plot_date}_mf_trigger_pa_grid_variable_epaz.png'
plot_name2 = f'{plot_date}_mf_trigger_E_grid_variable_epaz.png'
T = 6
PlotMFGrid('x_pa', 'h_pa', result, os.path.join(plot_path,plot_name1), T_threshold=T)
PlotMFGrid('x_E', 'h_E', result, os.path.join(plot_path,plot_name2), T_threshold=T)


#print(result[(result['h_pa'] == 89.5) & (result['T'] >= 4.5)])
#image = matrices.mean(axis=0)
#print(np.diagonal(image))
#fig = plt.figure(figsize=(7,7))

#ax1 = plt.subplot(111)
#extent = (0.0, 0.02, 0.02, 0.0)
#img = ax1.imshow(image, extent=extent, origin='upper', cmap='inferno_r')

#ax1.set_xlabel('Signal Axial Position (m)')
#ax1.set_ylabel('Template Axial Position (m)')
#ax1.set_title(f'MF Score for Variable Axial Position Electrons\n Simulated On-axis, E={18600} eV, Angle={pa} deg')
#ax1.set_xticks(xticks)
#ax1.set_xticklabels(xticklabels)
#ax1.set_yticks(yticks)
#ax1.set_yticklabels(yticklabels)

#ax1in = ax1.inset_axes([0.6, 0.6, 0.35, 0.35])
#imgin = ax1in.imshow(image, extent=extent, origin='upper', cmap='inferno_r')
#ax1in.set_xlim(89.0,89.05)
#ax1in.set_ylim(89.05, 89.0)
#ax1in.spines['top'].set_color('white')
#ax1in.spines['bottom'].set_color('white')
#ax1in.spines['left'].set_color('white')
#ax1in.spines['right'].set_color('white')
#ax1in.tick_params(axis='y', color='white')
#ax1in.tick_params(axis='x', color='white')
#ax1in.xaxis.label.set_color('white')
#ax1.indicate_inset_zoom(ax1in, edgecolor='black')
#fig.colorbar(img)

#fig.savefig(os.path.join(plot_path, name))


