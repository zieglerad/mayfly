import numpy as np
import matplotlib.pyplot as plt
import os 
import pickle as pkl
import h5py
import pandas as pd

def Load(path, metadata):
    r = [0.0]
    signal_path = path
    
    signal_list = os.listdir(signal_path)
    meta_rad = np.unique(metadata[:, 0])
    meta_E = np.unique(metadata[:, 1])
    meta_pa = np.unique(metadata[:, 2])
    meta_z = np.unique(metadata[:, 3])
    
    signals_to_load = []
    for signal in signal_list:
        if (np.isin(float(signal.split('rad')[-1].split('_')[0]), meta_rad) and 
        np.isin(float(signal.split('energy')[-1].split('_')[0]), meta_E) and 
        np.isin(float(signal.split('angle')[-1].split('_')[0]), meta_pa) and 
        np.isin(float(signal.split('axial')[-1].split('.pkl')[0]), meta_z)):
            
            signals_to_load.append(signal)
    
    data = []
    for i, signal in enumerate(signals_to_load):
        with open(os.path.join(signal_path, signal), 'rb') as infile:
            x = pkl.load(infile)
            try:
                data.append(list(x['x']))
            except:
                data.append(list(x['h']))
        
    data = np.array(data)

    return data

def RecursiveSort(data, metadata, i):
    for n in np.unique(metadata[:, i]):
        data[np.where(metadata[:, i] == n)[0], :] = (data[np.where(metadata[:, i] == n)[0], :])[
        np.argsort(metadata[np.where(metadata[:, i] == n)[0], i+1]), :]
        
        metadata[np.where(metadata[:, i] == n)[0], :] = (metadata[np.where(metadata[:, i] == n)[0], :])[
        np.argsort(metadata[np.where(metadata[:, i] == n)[0], i+1]), :]
        
    if i>1:
        #print(data, metadata)
        return data, metadata
    else:
        i += 1
        data, metadata = RecursiveSort(data, metadata, i)
        
    return data, metadata
    
#def RecursiveSortMeta(metadata, i):

#    for n in np.unique(metadata[:, i]):
        
#        metadata[np.where(metadata[:, i] == n)[0], :] = (metadata[np.where(metadata[:, i] == n)[0], :])[
#        np.argsort(metadata[np.where(metadata[:, i] == n)[0], i+1]), :]
        
#    if i>1:
        #print(data, metadata)
#        return metadata
#    else:
#        i += 1
#        print(metadata[0:20,:])
#        metadata = RecursiveSortMeta(metadata, i)
        
#    return metadata
    
def LoadMeta(path):

    signal_path = path
    
    signal_list = os.listdir(signal_path)
    meta_list = []
    for signal in signal_list:
        entry = []
        entry.append(float(signal.split('rad')[-1].split('_')[0]))
        entry.append(float(signal.split('energy')[-1].split('_')[0]))
        entry.append(float(signal.split('angle')[-1].split('_')[0]))
        entry.append(float(signal.split('axial')[-1].split('.pkl')[0]))
        meta_list.append(entry)
    metadata = np.array(meta_list)
    
    sorted_metadata = np.zeros(metadata.shape)
    
    meta_r = np.unique(metadata[:, 0])
    meta_E = np.unique(metadata[:, 1])
    meta_pa = np.unique(metadata[:, 2])
    meta_z = np.unique(metadata[:, 3])
    sorting_indexes = []
    for i in meta_r:
        x = metadata[np.where(metadata[:, 0] == i)[0]]
        x_ind = np.where(metadata[:, 0] == i)[0]
        for j in meta_E:
            y = x[np.where(x[:, 1] == j)[0]]
            y_ind = x_ind[np.where(x[:, 1] == j)[0]]
            for k in meta_pa:
                z = y[np.where(y[:, 2] == k)[0]]
                z_ind = y_ind[np.where(y[:, 2] == k)[0]]
                #print(z[np.argsort(z[:, 3])], z_ind[np.argsort(z[:, 3])])
                sorting_indexes.extend(list(z_ind[np.argsort(z[:, 3])]))
                
    sorted_metadata = metadata[np.asarray(sorting_indexes), :]
    sorted_signals = np.asarray(signal_list)[np.asarray(sorting_indexes)]
  
    return sorted_metadata, sorted_signals
    
def ComputeSelf(name):

    working_dir = '/home/az396/project/matchedfiltering'
    signal_path = 'data/signals'
    template_path = 'data/templates'
    
    signal_metadata, sorted_signals = LoadMeta(os.path.join(working_dir, signal_path, name))
    template_metadata, sorted_templates = LoadMeta(os.path.join(working_dir, template_path, name))
    
    
    result = []
    
    N_batch = 500
    batched_signals = np.array_split(sorted_signals, N_batch)
    batched_templates = np.array_split(sorted_templates, N_batch)
    batched_signal_metadata = np.array_split(signal_metadata, N_batch, axis=0)
    batched_template_metadata = np.array_split(template_metadata, N_batch, axis=0)
    
    
    for i, signal_batch in enumerate(batched_signals):
        for j, template_batch in enumerate(batched_templates):
            if i >=j:
                loop_signals = []
                loop_templates = []
                for signal in signal_batch:
                    with open(os.path.join(working_dir, signal_path, name, signal), 'rb') as infile:
                        loop_signals.append(pkl.load(infile)['x'])
                for template in template_batch:
                    with open(os.path.join(working_dir, template_path, name, template), 'rb') as infile:
                        loop_templates.append(pkl.load(infile)['h'])
                noisy_signals = AddNoise(np.array(loop_signals), 10)
                print(i,j)
                #print(noisy_signals.shape, np.array(loop_templates).conjugate().T.shape)
                #print(np.matmul(noisy_signals, np.array(loop_templates).conjugate().T))
                mf_scores = abs(np.matmul(noisy_signals, np.array(loop_templates).conjugate().T))
                print(mf_scores)
                print(batched_signal_metadata[i])
                print(batched_template_metadata[j])
                for n in range(len(batched_signal_metadata[i][:, 0])):
                    for m in range(len(batched_template_metadata[j][:, 0])):
                        #if n >= m
                        result.append({'T':mf_scores[n, m], 
                                        'x_r': batched_signal_metadata[i][n, 0],
                                        'x_E': batched_signal_metadata[i][n, 1],
                                        'x_pa': batched_signal_metadata[i][n, 2],
                                        'x_z': batched_signal_metadata[i][n, 3],
                                        'h_r': batched_template_metadata[j][m, 0],
                                        'h_E': batched_template_metadata[j][m, 1],
                                        'h_pa': batched_template_metadata[j][m, 2],
                                        'h_z': batched_template_metadata[j][m, 3]
                                        })
    return result
    
def ComputeTemplateBank(signal_name, template_name):

    working_dir = '/home/az396/project/matchedfiltering'
    signal_path = 'data/signals'
    template_path = 'data/templates'
    
    signal_metadata, sorted_signals = LoadMeta(os.path.join(working_dir, signal_path, signal_name))
    template_metadata, sorted_templates = LoadMeta(os.path.join(working_dir, template_path, template_name))
    
    result = []
    
    N_batch = 20
    #batched_signals = np.array_split(sorted_signals, N_batch)
    batched_templates = np.array_split(sorted_templates, N_batch)
    #batched_signal_metadata = np.array_split(signal_metadata, N_batch, axis=0)
    batched_template_metadata = np.array_split(template_metadata, N_batch, axis=0)
    
    for i, signal in enumerate(sorted_signals):
        with open(os.path.join(working_dir, signal_path, signal_name, signal), 'rb') as infile:
            loop_signal = pkl.load(infile)['x']
            noisy_signal = AddNoise(np.array(loop_signal), 10)
        for j, template_batch in enumerate(batched_templates):
            loop_templates = []
            for template in template_batch:
                with open(os.path.join(working_dir, template_path, template_name, template), 'rb') as infile:
                    loop_templates.append(pkl.load(infile)['h'])
            noisy_signals = noisy_signal.reshape((1, noisy_signal.size)).repeat(len(loop_templates), axis=0)
            print(i,j)
            #print(noisy_signals.shape)
            mf_scores = np.diagonal(abs(np.matmul(noisy_signals, np.array(loop_templates).conjugate().T)))
            for m in range(len(batched_template_metadata[j][:, 0])):
                #if n >= m
                result.append({'T':mf_scores[m], 
                                'x_r': signal_metadata[i, 0],
                                'x_E': signal_metadata[i, 1],
                                'x_pa': signal_metadata[i, 2],
                                'x_z': signal_metadata[i, 3],
                                'h_r': batched_template_metadata[j][m, 0],
                                'h_E': batched_template_metadata[j][m, 1],
                                'h_pa': batched_template_metadata[j][m, 2],
                                'h_z': batched_template_metadata[j][m, 3]
                                })
    return result

def AddNoise(data, T):

    size = data.size
    shape = data.shape
    var = 1.38e-23 * 100e6 * 50 * T
    
    noise = np.random.multivariate_normal([0, 0], np.eye(2) * var / 2, size)
    noise = noise[:, 0] + 1j * noise[:, 1]
    
    return data + noise.reshape(shape)  
    
    
def EggReader(pathtoegg, Vrange=5.5e-8, nbit=8):

    f=h5py.File(pathtoegg,'r')
    dset=f['streams']['stream0']['acquisitions']['0']
    channels=list(f['channels'].keys())

    Nsamp=dset.shape[1]//(2*len(channels))
    ind=[]
    for ch in channels:
        ind.append(int(ch.split('l')[1]))
        #print(ch.split('l'))
    ind=np.array(ind)

    data=dset[0,:].reshape(ind.size,2*Nsamp)

    Idata=np.float64(data[:,np.arange(0,2*Nsamp,2)])
    Qdata=np.float64(data[:,np.arange(1,2*Nsamp,2)])

    for i in range(len(channels)):
        Idata[i,:]-=np.mean(Idata[i,:])
        Qdata[i,:]-=np.mean(Qdata[i,:])

    for i in range(len(channels)):
        Idata[i,:]*=Vrange/(2**nbit)
        Qdata[i,:]*=Vrange/(2**nbit)

    complexdata=Idata+1j*Qdata

    f.close()

    return complexdata

    
    
    
    

