import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os
import pandas as pd

working_dir = '/home/az396/project/mayfly'
signal_path = 'data/signals'
template_path = 'data/templates'

simulation_name = '210604_epa_template_grid'

pitch = 89.
energy = 18595.
z = 0.0
r = 0.0

signal_name = f'herc_sim_angle{pitch:2.4f}_rad{r:1.3f}_energy{energy:5.2f}_axial{z:1.3f}.pkl'
template_name = f'template_herc_sim_angle{pitch:2.4f}_rad{r:1.3f}_energy{energy:5.2f}_axial{z:1.3f}.pkl'



plot_date = '210610'
plot_path = '/home/az396/project/mayfly/analysis/plotting/plots'


result_path = '/home/az396/project/mayfly/analysis/results'
result_date = '210610'
result_name = 'template'
result_file = f'{result_date}_{result_name}.pkl'



with open(os.path.join(working_dir, signal_path, simulation_name, signal_name), 'rb') as infile:
    signal = pkl.load(infile)
    
with open(os.path.join(working_dir, template_path, simulation_name, template_name), 'rb') as infile:
    template = pkl.load(infile)
    
N_trials = 512
plot_name = f'{plot_date}_example_mf_score_histogram_10K_noise.png'

var = 1.38e-23 * 10 * 50 * 200e6

mf_scores = np.zeros(N_trials)
for n in range(N_trials):
    
    noise = np.random.multivariate_normal([0,0], np.eye(2) * var / 2, 60 * 8192)
    noise = noise[:, 0] + 1j * noise[:, 1]
    
    noisy_signal = signal['x'] + noise
    
    score = abs(np.vdot(noisy_signal, template['h']))
    
    mf_scores[n] = score
    
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(1,1,1)
mf_var = np.var(mf_scores)
ax.hist(mf_scores, 32)
ax.set_title(f'Variance = {mf_var}')

plt.savefig(os.path.join(plot_path, plot_name))


