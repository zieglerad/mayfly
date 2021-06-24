import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
import mayfly as mf

path_to_eggs = '/home/az396/project/sims/sim_data/210610_epa_grid'
name = '210610_epa_grid'
v_range = 5.5e-8
slice_num = 0
N_start = 1500
N_slice = 8192

list_sim = os.listdir(path_to_eggs)
sims_per_file = 100

#if len(list_sim) % sims_per_file == 0:
#	N_files = len(list_sim) // sims_per_file
#	remainder = 0
#else:
#	N_files = len(list_sim) // sims_per_file + 1
#	remainder = len(list_sim) % sims_per_file
	
# get simulation parameters (pitch angle, pos, etc.) cv


n = 0
#for i, sim in enumerate(list_sim):
for i in range(len(list_sim)):
    if os.path.isdir(os.path.join(path_to_eggs, list_sim[i])):
        sim = list_sim[i]
        angle = sim.split('angle')[-1].split('_')[0]
        rad = sim.split('rad')[-1].split('_')[0]
        #energy = sim.split('energy')[-1]
        energy = sim.split('energy')[-1].split('_')[0]
        axial = sim.split('axial')[-1]
        egg_file = os.path.join(path_to_eggs, sim, 'angle' + angle + '_rad' + rad + '_energy' + energy + '_axial' + axial + '_locust.egg')
        try:
            parsed_egg = mf.utils.EggReader(egg_file)
        except:
            print(f'Resubmit: {sim}')
            continue
        egg_time_series = parsed_egg[:, N_start : N_start + N_slice]
        
        egg_time_series = egg_time_series.reshape(egg_time_series.size)

        data = {'pa': float(angle), 'rad': float(rad), 'E': float(energy), 'z': float(axial), 'x': np.array(egg_time_series)} 
        with open(f'/home/az396/project/mayfly/data/signals/{name}/{sim}.pkl', 'wb') as outfile:
            pkl.dump(data, outfile)
        if i % 50 == 49:
            print('Done with %d' % i)

