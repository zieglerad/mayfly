import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os
import pandas as pd

working_dir = '/home/az396/project/mayfly'
signal_path = 'data/signals'
template_path = 'data/templates'
simulation_name = '21520_variable_epaz'

#signal_metadata = 

plot_date = '210610'
plot_path = '/home/az396/project/mayfly/analysis/plotting/plots'


result_path = '/home/az396/project/mayfly/analysis/results'
result_date = '210610'
result_name = 'epa_template_grid_self_scores_diagonal_on_grid_degeneracies'
result_file = f'{result_date}_{result_name}.pkl'



with open(os.path.join(result_path, result_file), 'rb') as infile:
    result = pkl.load(infile)
    
for i in range(len(result)):

    if i < 101:
        fig = plt.figure(figsize=(8,6))
        ax = plt.subplot(1,1,1)
        
        
        signal_ts = result[i]['signal']['x']['x']
        signal_pa = result[i]['signal']['x_pa']
        signal_E = result[i]['signal']['x_E']
        signal_fft = np.fft.fftshift(np.fft.fft(signal_ts)) / np.sqrt(signal_ts.size)
        
        freq = np.fft.fftfreq(signal_ts.size, 1/200e6)
        
        ax.plot(freq, abs(signal_fft) ** 2, label = f'sig_angle{signal_pa}_energy{signal_E}')
        
        for j in range(len(result[i]['template'])):
            
            template_ts = result[i]['template'][j]['h']['x']
            
            template_fft = np.fft.fftshift(np.fft.fft(template_ts)) / np.sqrt(template_ts.size)
            
            template_pa = result[i]['template'][j]['h_pa']
            template_E = result[i]['template'][j]['h_E']
            ax.plot(freq, abs(template_fft) ** 2, label = f'temp{j}_angle{template_pa}_energy{template_E}')
            
        plot_name = f'{plot_date}_signal_and_degenerate_templates{i}.png'
        plt.legend(loc=1)
        plt.savefig(os.path.join(plot_path, plot_name))
        plt.close()



