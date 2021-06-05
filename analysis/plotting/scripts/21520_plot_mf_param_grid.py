import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os

data_path = '/home/az396/project/matchedfiltering/analysis/results'
plot_path = '/home/az396/project/matchedfiltering/analysis/plotting/plots'

pa = 90.0
energy = 18600
date = '21520'
#xticks = np.arange(0, 440, 40)
#xticklabels = np.linspace(87, 88, 401)[xticks]
#yticks = xticks
#yticklabels = xticklabels


name = f'{date}_mf_axial_grid_pa{pa}.png'

data_file = f'21520_pa{pa}_mf_axial_grid.pkl'

with open(os.path.join(data_path, data_file), 'rb') as infile:
    matrices = pkl.load(infile)
    
image = matrices.mean(axis=0)
#print(np.diagonal(image))
fig = plt.figure(figsize=(7,7))

ax1 = plt.subplot(111)
extent = (0.0, 0.02, 0.02, 0.0)
img = ax1.imshow(image, extent=extent, origin='upper', cmap='inferno_r')

ax1.set_xlabel('Signal Axial Position (m)')
ax1.set_ylabel('Template Axial Position (m)')
ax1.set_title(f'MF Score for Variable Axial Position Electrons\n Simulated On-axis, E={18600} eV, Angle={pa} deg')
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
fig.colorbar(img)

fig.savefig(os.path.join(plot_path, name))


