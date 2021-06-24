import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.animation import FuncAnimation
import seaborn as sns

working_dir = '/home/az396/project/mayfly'
signal_path = 'data/signals'
template_path = 'data/templates'


plot_date = '210611'
plot_path = '/home/az396/project/mayfly/analysis/plotting/plots'


result_path = '/home/az396/project/mayfly/analysis/results'
result_date = '210611'
result_name = 'epa_grid_self_scores_animation_matrices'

result_file = f'{result_date}_{result_name}.pkl'

plot_name = f'{plot_date}_{result_name}_animation.gif'

with open(os.path.join(result_path, result_file), 'rb') as infile:
    result = pd.DataFrame(pkl.load(infile))

fig = plt.figure(figsize=(8,8))

ax1 = plt.subplot(111)
cmap = sns.color_palette(palette='rocket', as_cmap = True)
im1 = ax1.imshow(result['grids'][0].T, cmap = cmap, extent = (18590, 18600, 88., 87.), aspect=10.)
cbar = plt.colorbar(im1)

ax1.set_title(f'Matched Filter Scores for On-grid Signals', size=16)
ax1.set_ylabel('Template Pitch Angle (deg)', size=16)
ax1.set_xlabel('Template Energy (eV)', size=16)

cbar.ax.tick_params(labelsize = 14)
ax1.tick_params(size=4)

ax1.set_xticks(np.linspace(18590, 18600, 6))
ax1.set_xticklabels(np.linspace(18590, 18600, 6), size=12)

ax1.set_yticks(np.linspace(87., 88., 6))
ax1.set_yticklabels(np.linspace(87., 88., 6), size=12)

line1, = ax1.plot(result['sig_x'][0], result['sig_y'][0], 'r*', markersize=18)

plt.tight_layout()

def animate(i):
    im1.set_data(result['grids'][i].T)
    line1.set_data(result['sig_x'][i], result['sig_y'][i])
    cbar.update_normal(im1)
    return im1, line1,
    
animation = FuncAnimation(fig, animate, frames=np.arange(0, 41, 1), blit=True, interval=500)
animation.save(os.path.join(plot_path, plot_name))


