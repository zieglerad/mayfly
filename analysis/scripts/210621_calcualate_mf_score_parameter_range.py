import numpy as np
import matplotlib.pyplot as plt
import os
import mayfly as mf
import h5py
import pandas as pd

mayflypath = '/home/az396/project/mayfly'
setdate = '210621'
setname = 'mf3'

# parameter ranges
e_min = 18500
e_max = 18602

angle_min = 86
angle_max = 90



# check the metadata

h5file = h5py.File(os.path.join(mayflypath, f'data/datasets/{setdate}_{setname}.h5'), 'r')
metadata = h5file['meta']
signals = h5file['signal']
#templates = h5file['templates']

metapd = pd.DataFrame({'E': metadata['E'][:], 'angle':metadata['angle'][:], 'r': metadata['r'][:], 'z': metadata['z'][:]})
#print(metapd)
keys_selected = metapd[
                        (metapd['angle']>=angle_min) & 
                        (metapd['angle']<=angle_max) &
                        (metapd['E']>=e_min) & 
                        (metapd['E']<=e_max) 
                    ].index.values
                    
sort_ind = np.argsort(metapd[
                        (metapd['angle']>=angle_min) & 
                        (metapd['angle']<=angle_max) &
                        (metapd['E']>=e_min) & 
                        (metapd['E']<=e_max) 
                    ]['E'])

print(metapd[
                        (metapd['angle']>=angle_min) & 
                        (metapd['angle']<=angle_max) &
                        (metapd['E']>=e_min) & 
                        (metapd['E']<=e_max) 
                    ])

#keys_selected = keys_selected[np.argsort(metapd[(metapd['angle']>=angle_min) & (metapd['angle']<=angle_max)]['angle'])]

keys_selected = keys_selected[sort_ind]
#print(keys_selected)
#print(keys_selected)
#angles = np.argsort(metapd[(metapd['angle']>=angle_min) & (metapd['angle']<=angle_max)]['angle'])
#print(angles)

var = 1.38e-23 * 200e6 * 10 * 50
grid = np.zeros((keys_selected.size, keys_selected.size))
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(1,1,1)
for row, ikey_signal in enumerate(keys_selected):
    #print(ikey_signal)
    for col, ikey_template in enumerate(keys_selected):
        scores = np.zeros(10)
        for n in range(10):
            
            noise = np.random.multivariate_normal([0, 0], np.eye(2) * var / 2, 8192)
            noise = noise[:, 0] + 1j * noise[:, 1]

            scores[n] = abs(np.vdot(signals[f'{ikey_signal}'][:] + noise, templates[f'{ikey_template}'][:]))
    #print(scores.mean())
        grid[row, col] = scores.mean()
        


    #ax.plot(abs(np.fft.fft(signals[f'{ikey_signal}'][:])))
img = ax.imshow(grid)
cb = plt.colorbar(img)
plt.savefig('test.png')
h5file.close() 






