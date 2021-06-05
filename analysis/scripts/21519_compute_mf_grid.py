import numpy as np 
import matchfilter as mf
import matplotlib.pyplot as plt
import pickle as pkl
import os

r = 0.0
angles = np.linspace(89.50, 89.55, 21)
energy = np.linspace(18595, 18598, 21)
z_pos = [0.000, 0.003, 0.005]
T = 10
results = '/home/az396/project/matchedfiltering/analysis/results'

N = 100

for pa in angles:

    #plot_angles = np.linspace(pa, pa + 1, 1 )
    plot_angles = np.linspace(pa, pa, 1)
    #plot_angles = angles
    signals = '/home/az396/project/matchedfiltering/data/signals/21520_variable_epaz'
    signal_data, signal_meta = mf.utils.Load(r, plot_angles, energy, z_pos, signals)
    print(signal_meta)

    #templates = '/home/az396/project/matchedfiltering/data/templates/21519_axial_position'
    #template_data, template_meta = mf.utils.Load(r, plot_angles, energy, z_pos, templates)

    #matrices = []
    #for n in range(N):
    #    print(n+1)
    #    noisy_signals = mf.utils.AddNoise(signal_data, T)
    #    matrices.append(abs(np.matmul(noisy_signals, template_data.conjugate().T)))

    #matrices = np.array(matrices)
    #with open(os.path.join(results, f'21520_pa{pa}_mf_axial_grid.pkl'), 'wb') as outfile:
    #    pkl.dump(matrices, outfile)
#fig1 = plt.figure(figsize=(8,8))
#ax1 = plt.subplot(111)
#img = ax1.imshow(np.mean(matrices, axis = 0))
#ticks = np.arange(0, 450, 50)
#plt.colorbar(img)
#ax1.set_xticks(ticks)
#ax1.set_yticks(ticks)
#ax1.set_xticklabels(pa[ticks])
#ax1.set_yticklabels(pa[ticks])
#ax1.set_xlabel('Template Pitch Angle (deg)')
#ax1.set_ylabel('Signal Pitch Angle (deg)')
#ax1.set_title('MF Score for On-axis Electrons in 10K Noise')

#plt.savefig('test.png')

