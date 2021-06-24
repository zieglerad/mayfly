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

xaxis_param = 'angle'
yaxis_param = 'E'



# check the metadata

h5file = h5py.File(os.path.join(mayflypath, f'data/datasets/{setdate}_{setname}.h5'), 'r')
metadata = h5file['meta']
signals = h5file['signal']
#templates = h5file['templates']

metapd = pd.DataFrame({'E': metadata['E'][:], 'angle':metadata['angle'][:], 'r': metadata['r'][:], 'z': metadata['z'][:]})

metapd = metapd[[xaxis_param, yaxis_param]].drop_duplicates()

Nsignals = metapd.shape[0]

xparam_list = metapd[xaxis_param].unique()
yparam_list = metapd[yaxis_param].unique()
index = metapd.index.values

randomsignal_ind = index[np.random.randint(0, Nsignals, 1)][0]

grid = np.zeros((yparam_list.size, xparam_list.size))

#signal = signals[f'{randomsignal_ind}'][:].reshape(signals[f'{randomsignal_ind}'][:].size)
signal = signals[f'{randomsignal_ind}'][:].sum(0)

var = 1.38e-23 * 200e6 * 10 * 50
gridmax = 0
besttemplate = -1
for row, yparam in enumerate(yparam_list):
    for col, xparam in enumerate(xparam_list):
        print(row, col)
        itemplate = metapd[(abs(metapd[yaxis_param] - yparam) < 1e-6) & (abs(metapd[xaxis_param] - xparam) < 1e-6)].index.values[0]
        
        noise = np.random.multivariate_normal([0, 0], np.eye(2) * var / 2, signal.size)
        noise = noise[:, 0] + 1j * noise[:, 1]
        
        #template = signals[f'{itemplate}'][:].reshape(signals[f'{itemplate}'][:].size)
        template = signals[f'{itemplate}'][:].sum(0)
        
        grid[row, col] = abs((1 / np.sqrt(var * np.vdot(template, template))) * np.vdot(signal + noise, template))
        if grid[row, col] > gridmax:
            besttemplate = itemplate
            gridmax = grid[row, col]
            
print(metapd.iloc[randomsignal_ind])
for item in signals[f'{besttemplate}'].attrs.items():
    print(item[0], item[1])
'''
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



'''
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(1,1,1)

img = ax.imshow(grid)
cb = plt.colorbar(img)
plt.savefig('test.png')
h5file.close() 
