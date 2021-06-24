import h5py
import os
import numpy as np

simpath = '/home/az396/project/sims'
mayflypath = '/home/az396/project/mayfly'

combineddata_date = '210621'
combineddata_name = 'mf3'

## create a list of simulation datasets to combine
simdata_date = '210621'
simdata_name = 'mf_run3'

'''
nsimdatarun = 1
simdatalist = []
for i in range(nsimdatarun):
    simdatalist.append(f'{simdata_date}_{simdata_name}{i + 1}.h5')
'''
simdatalist = [
            f'{simdata_date}_{simdata_name}.h5'
            ]

####

assumed_noise_temp = 10 # warning assuming diagonal noise covariace matrix
noise_var = 1.38e-23 * 50 * 200e6 * assumed_noise_temp

## open destination h5 file, iterate through list of signals copying each one

h5combined = h5py.File(os.path.join(mayflypath, f'data/datasets/{combineddata_date}_{combineddata_name}.h5'), 'w')
combinedsignals = h5combined.create_group('signal')
#combinedtemplates = h5combined.create_group('templates')
#combinedtemplates.attrs.create('T', assumed_noise_temp)
combinedmetadata = h5combined.create_group('meta')

#energy_list, angle_list, rad_list, z_list = [], [], [], []
meta_list = [[], [], [], [], []]
ncombine = 0
for simdata in simdatalist:
    h5simdata = h5py.File(os.path.join(simpath,f'datasets/{simdata}'), 'r')
    simdatakeys = list(h5simdata['signal'].keys())
    print(simdata)
    for key in simdatakeys:
    
        signal = h5simdata['signal'][key][:]
        simdataattrs = h5simdata['signal'][key].attrs.items()
        signaldset = combinedsignals.create_dataset(
                                                f'{ncombine}', 
                                                data = signal
                                                )
        #templatedset = combinedtemplates.create_dataset(
        #                                        f'{ncombine}', 
        #                                        data = signal * 1 / (np.sqrt(noise_var * np.vdot(signal, signal)))
        #                                        )
        #print(combineddset)
        
        meta_list[0].append(ncombine)
        for n, item in enumerate(simdataattrs):
            meta_list[n + 1].append(item[1])
            for dset in [signaldset]:
                dset.attrs.create(item[0], item[1])

                
        ncombine += 1
    
    h5simdata.close()

combinedmetadata.create_dataset('ind', data = np.array(meta_list[0]))
for ikey, key in enumerate(['E', 'angle', 'r', 'z']):
    combinedmetadata.create_dataset(
                                    key, 
                                    data = np.array(meta_list[ikey + 1])
                                    )

h5combined.close()

####
'''
if diagonal_cov_mat == True:
        a = signal['x']
        b = 1 / np.sqrt(noise_cov_mat[0, 0] * np.vdot(a, a))
        
        h = b * a
'''
