import numpy as np 
import matplotlib.pyplot as plt
import os
import pickle as pkl

template_path = '/home/az396/project/matchedfiltering/data/templates'
signal_path = '/home/az396/project/matchedfiltering/data/raw_data'


N_files = len(os.listdir(template_path))
noise_var = 4 * 1.38e-23 * 100e6 * 50 * 10

for n in range(N_files):

    with open(os.path.join(template_path, 'templates' + str(n) + '.pkl'), 'rb') as infile:
        templates = pkl.load(infile)
    with open(os.path.join(signal_path, 'raw_data' + str(n) + '.pkl'), 'rb') as infile:
        signals = pkl.load(infile)
        
    noise = np.random.multivariate_normal([0, 0], np.eye(2) * noise_var / 2, templates['h'].size)
    noise = (noise[:, 0] + 1j * noise[:, 0]).reshape(templates['h'].shape)
    
    print(abs(np.matmul(templates['h'].conjugate(), np.transpose(signals['x'], axes = (0, 2, 1))))[0, :, :])
    print(abs(np.matmul(templates['h'].conjugate(), np.transpose(noise, axes = (0, 2, 1))))[0, :, :])
