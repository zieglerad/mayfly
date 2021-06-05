import numpy as np 
import matchfilter as mf
import matplotlib.pyplot as plt
import pickle as pkl

r = 0.0
angles = np.linspace(90, 90, 1)
energy = np.linspace(18550, 18600, 201)
T = 10

N = 100

for pa in angles:

    signals = '/home/az396/project/matchedfiltering/data/signals/21518_variable_energy'
    signal_data, signal_meta = mf.utils.Load(r, np.array([pa]), energy, signals)

    templates = '/home/az396/project/matchedfiltering/data/templates/21518_variable_energy'
    template_data, template_meta = mf.utils.Load(r, np.array([pa]), energy, templates)

    matrices = []
    for n in range(N):
        print(n+1)
        noisy_signals = mf.utils.AddNoise(signal_data, T)
        matrices.append(abs(np.matmul(noisy_signals, template_data.conjugate().T)))

    matrices = np.array(matrices)
    with open(f'21519_pa{pa}_mf_energy_grid.pkl', 'wb') as outfile:
        pkl.dump(matrices, outfile)
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

