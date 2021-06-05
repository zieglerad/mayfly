import numpy as np 
import matchfilter as mf
import matplotlib.pyplot as plt
import pickle as pkl
import os

r = 0.0
angles = np.linspace(90,90, 1)
energy = np.linspace(18600, 18600, 1)
z_pos = np.linspace(0.0, 0.02, 2)

T = 10
date = '21520'
plots = '/home/az396/project/matchedfiltering/analysis/plotting/plots'
name = f'{date}_signals_axial_position.png'

N = 100

for pa in angles:

    #plot_angles = np.linspace(pa, pa + 1, 1 )
    plot_angles = np.linspace(pa, pa, 1)
    signals = '/home/az396/project/matchedfiltering/data/signals/21519_axial_position'
    signal_data, signal_meta = mf.utils.Load(r, plot_angles, energy, z_pos, signals)
    #print(signal_meta)
    fig = plt.figure(figsize=(8,6))
    ax1 = plt.subplot(111)
    
    for i in range(z_pos.size):
        #ax1.plot(np.unwrap(np.arctan2(signal_data[i][0:1024].imag, signal_data[i][0:1024].real)))
        ax1.plot(abs(np.fft.fft(signal_data[i][0:8192])))
        
    plt.savefig(os.path.join(plots, name))
    plt.close()

    

    
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

