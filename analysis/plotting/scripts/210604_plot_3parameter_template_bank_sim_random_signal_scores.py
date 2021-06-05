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

plot_date = '210604'
plot_path = '/home/az396/project/matchedfiltering/analysis/plotting/plots'


result_path = '/home/az396/project/matchedfiltering/analysis/results'
result_date = '210521'
result_name = 'random_epaz_template_bank_sim'
result_file = f'{result_date}_{result_name}.pkl'

plot_name = f'{plot_date}_random_epaz_1signal_mf_scores.png'

with open(os.path.join(result_path, result_file), 'rb') as infile:
    result = pd.DataFrame(pkl.load(infile))


unique_signals = result[['x_r', 'x_E', 'x_pa', 'x_z']].drop_duplicates()
N_signals = unique_signals.shape[0]
for isignal in range(N_signals):
    print(isignal)
    signal = unique_signals.iloc[isignal]
    #template_z = signal['x_z']
    #print(signal)
    selected_results = result[
                                (result['x_r'] == signal['x_r']) &
                                (result['x_E'] == signal['x_E']) &
                                (result['x_pa'] == signal['x_pa']) &
                                (result['x_z'] == signal['x_z'])
                             ] 

    #print(selected_results[['h_pa', 'h_E', 'h_z', 'h_r']])

    template_z = selected_results['h_z'].unique()

    for n, h_z in enumerate(template_z):
        
        plot_name = f'{plot_date}_random_epaz_1signal_mf_scores_signal{isignal}_template_z{h_z}.png'
        
        plot_result = selected_results[selected_results['h_z'] == h_z]

        N_pa = plot_result['h_pa'].unique().size
        N_E = plot_result['h_E'].unique().size

        plot_grid = np.zeros((N_E, N_pa))

        for irow, E in enumerate(plot_result['h_E'].unique()):
            for icol, pa in enumerate(plot_result['h_pa'].unique()):
                plot_grid[irow, icol] = plot_result[
                                                        (plot_result['h_E'] == E) &
                                                        (plot_result['h_pa'] == pa)
                                                        ]['T'] 

        fig = plt.figure(figsize=(10,8))
        ax1 = plt.subplot(111)

        pa0 = plot_result['h_pa'].unique()[0]
        pa1 = plot_result['h_pa'].unique()[-1]

        E0 = plot_result['h_E'].unique()[0]
        E1 = plot_result['h_E'].unique()[-1]
        img = ax1.imshow(
                        plot_grid, 
                        extent = (pa0, pa1, E1, E0), 
                        aspect = 'auto', 
                       interpolation = 'none', 
                        cmap='viridis'
                        )
        cbar = plt.colorbar(img)

        ax1.tick_params(size=4)

        ax1.set_yticks(np.linspace(E1, E0, 7))
        ax1.set_yticklabels(np.linspace(E1, E0, 7), size=12)
        ax1.set_xticks(np.linspace(pa0, pa1, 6))
        ax1.set_xticklabels(np.round(np.linspace(pa0, pa1, 6), 2), size=12)
        ax1.set_xlabel('Template Pitch Angle (deg)', size=14)
        ax1.set_ylabel('Tempalate Energy (eV)', size=14)
        ax1.set_title(f'MF Scores for a Randomly Generated, Off-grid Signal\n Template Axial Positions = {h_z*1e3} mm', size=14)

        cbar.set_label('MF Score', size=14)
    #print(x_pa)
        x_pa = signal['x_pa']
        x_E = signal['x_E']
        x_z = signal['x_z']
        ax1.text(89.53, 18597.38, f'Signal Parameters', size=14, color='w')
        ax1.text(89.53, 18597.5, f'Pitch Angle: {x_pa} deg', size=14, color='w')
        ax1.text(89.53, 18597.62, f'Energy : {x_E} eV', size=14, color='w')
        ax1.text(89.53, 18597.74, f'Axial Pos. : {x_z*1e3} mm', size=14, color='w')


        ax1.plot(x_pa, x_E, 'r*', markersize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, plot_name))

#unique_signals = result[['x_r', 'x_E', 'x_pa', 'x_z']].drop_duplicates()
#unique_templates = result[['h_r', 'h_E', 'h_pa', 'h_z']].drop_duplicates()

#grid = np.zeros((unique_signals.shape[0], unique_templates.shape[0]))
##print(unique_templates)
#for i in range(unique_signals.shape[0]):
#    sig_params = unique_signals.iloc[i]
#    loop_result1 = result[(result['x_r'].isin(sig_params)) & 
#                        (result['x_E'].isin(sig_params)) & 
#                        (result['x_pa'].isin(sig_params)) & 
#                        (result['x_z'].isin(sig_params))
#                        ]
    #print(loop_result1)
    #input()
#    for j in range(unique_templates.shape[0]):
#        template_params = unique_templates.iloc[j]
#        loop_result2 = loop_result1[(loop_result1['h_r'].isin(template_params)) & 
#                        (loop_result1['h_E'].isin(template_params)) & 
#                        (loop_result1['h_pa'].isin(template_params)) & 
#                        (loop_result1['h_z'].isin(template_params))
#                        ]
#        grid[i, j] = loop_result2.iloc[-1]['T']
#        print(grid[i, j])
#        #input()

#fig = plt.figure(figsize=(10,8))
#ax1 = plt.subplot(111)

#img = plt.imshow(result, extent = (0, 1323, 100, 0), aspect = 'auto', interpolation = 'none', cmap='inferno_r')
#cbar = plt.colorbar(img)

#ax1.set_xlabel('Template', size=14)
#ax1.set_ylabel('Signal', size=14)
#ax1.set_title('Tempalate Bank Simulation with Random Signals', size=14)
#cbar.set_label('MF Score', size=14)

#plt.savefig(os.path.join(plot_path, plot_name))
#signal_params = []
#for key in ['x_r', 'x_E', 'x_pa', 'x_z']:
#    signal_params.append(result[key].unique())
#    print(result[key].unique().size)
    
#print(np.array(signal_params).)

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
#PlotMFGrid('x_pa', 'h_pa', result, os.path.join(plot_path,plot_name1), T_threshold=T)
#PlotMFGrid('x_E', 'h_E', result, os.path.join(plot_path,plot_name2), T_threshold=T)


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


