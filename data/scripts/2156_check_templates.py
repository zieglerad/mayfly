import numpy as np 
import matplotlib.pyplot as plt
import os
import pickle as pkl
import h5py

mayflypath = '/home/az396/project/mayfly'

dataset_date = '210621'
dataset_name = 'mf1'

h5file = h5py.File(os.path.join(mayflypath, f'data/datasets/{dataset_date}_{dataset_name}.h5'), 'r')

keys = np.array(list(h5file['signal'].keys()))
print(keys)
random_keys = keys[np.random.randint(0, len(keys), 4)]


fig = plt.figure(figsize=(10,10))
for n, key in enumerate(random_keys):
    ax = plt.subplot(2,2,n+1)
    
    #ax.plot(h5file['signal'][key][:].real, label = np.mean(abs(h5file['signal'][key][:])**2)/50)
    #ax.plot(h5file['signal'][key][:].imag)
    print(np.mean(abs(h5file['signal'][key][:])**2)/50)
    print(np.sum(0.02 * abs(np.fft.fft(h5file['signal'][key][:]) / 8192) ** 2))
    ax.plot(0.02 * abs(np.fft.fft(h5file['signal'][key][:]) / 8192) ** 2, label = h5file['signal'][key].attrs['angle'])
    
    #ax.set_xlim(0, 200)
    plt.legend(loc=1)
plt.savefig('test1.png')

h5file.close()
