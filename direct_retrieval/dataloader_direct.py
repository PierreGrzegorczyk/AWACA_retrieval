import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import random
import json
import time

# DEFAULT_KEYS = ['a_mass_size', 'alpha_area_size', 'aspect_ratio', 'b_mass_size', 'beta_area_size', 'dmean', 'lwc', 'noise_level_W', 'windW']
# TURB_KEYS = ['SIGMA_TURB']
# DEFAULT_TARGET = ['spectrum_W_256']

    
    
class direct_dataset(Dataset):

    def __init__(self, target_data_file, spectrum_data_file, i_start_ds = 0, i_end_ds=-1, normalize_spectra = True, normalize_target = True, mean_normalization_json = None, std_normalization_json = None, mean_normalization_npy = None, std_normalization_npy = None):
        super(Dataset, self).__init__()
        self.i_start_ds = i_start_ds
        self.i_end_ds = i_end_ds
        
        self.spectrum_W = h5py.File(spectrum_data_file, 'r')['spectrum_W_256']
        self.spectrum_X = h5py.File(spectrum_data_file, 'r')['spectrum_X_256']
        self.target_dataset = np.load(target_data_file)[i_start_ds:i_end_ds]
        self.norm_file = mean_normalization_json
        self.norm_spectra = normalize_spectra
        self.norm_target = normalize_target
        if not(mean_normalization_json is None):
            with open(mean_normalization_json) as mn:
                mean_norm = json.load(mn)
                self.spectra_mean_for_normalization_W = mean_norm['spectrum_W_256']
                self.spectra_mean_for_normalization_X = mean_norm['spectrum_X_256']
            with open(std_normalization_json) as sn:
                std_norm = json.load(sn)
                self.spectra_std_for_normalization_W = std_norm['spectrum_W_256']
                self.spectra_std_for_normalization_X = std_norm['spectrum_X_256']
            self.target_mean_for_normalization = torch.from_numpy(np.load(mean_normalization_npy))
            self.target_std_for_normalization = torch.from_numpy(np.load(std_normalization_npy))
            
    def __len__(self):
        return self.target_dataset.shape[0]


    def __getitem__(self, idx):

        target_data = torch.from_numpy(self.target_dataset[idx,:])
        spectrum_W = torch.from_numpy(self.spectrum_W[idx+self.i_start_ds,:])
        spectrum_X = torch.from_numpy(self.spectrum_X[idx+self.i_start_ds,:])
            
        if (not(self.norm_file is None) & self.norm_target) :
            target_data = (target_data-self.target_mean_for_normalization)/self.target_std_for_normalization
        if (not(self.norm_file is None) & self.norm_spectra):
            spectrum_data_W = (spectrum_W-self.spectra_mean_for_normalization_W)/self.spectra_std_for_normalization_W
            spectrum_data_X = (spectrum_X-self.spectra_mean_for_normalization_X)/self.spectra_std_for_normalization_X
            
        spectrum_data = torch.stack((spectrum_data_W, spectrum_data_X),axis=0)
        return (spectrum_data, target_data)


if __name__ == '__main__':
    import json
    tic = time.time()
    
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    
    config = json.load(open('config.json'))
    ## Prepare dataset
    train_dataset = direct_dataset(config['target_array'], config['spectrum_h5'], 
                                               i_start_ds = config['i_start_ds'], i_end_ds = config['i_end_ds'],
                                               mean_normalization_json = config['mean_normalization_json'],
                                               std_normalization_json = config['std_normalization_json'],
                                               mean_normalization_npy = config['mean_normalization_npy'],
                                               std_normalization_npy = config['std_normalization_npy'],
                                               normalize_spectra = config['normalize_spectra'],
                                               normalize_target = config['normalize_target'])
    
#     train_dataset = direct_dataset('../input_newNoiseRound_w_minspec_log.npy', '/data/acbr_spectra_dbcreation/newNoiseRound_all_wind_ts.h5')
    
    N = train_dataset.spectrum_W.shape[0] 
    print(N)
    N2 = len(train_dataset)
    print(N2)
    ts_size_per_epoch = 100000
    epoch=0
    inds = torch.arange((ts_size_per_epoch*epoch)%N, min(N,(ts_size_per_epoch*epoch)%N + ts_size_per_epoch))
#     inds = torch.arange((int(ts_size_per_epoch/2)*epoch)%(int(N/2)), (int(ts_size_per_epoch/2)*epoch)%(int(N/2)) + int(ts_size_per_epoch/2)).tolist() + torch.arange(-(int(ts_size_per_epoch/2)*epoch)%(int(N/2)) - int(ts_size_per_epoch/2), -(int(ts_size_per_epoch/2)*epoch)%(int(N/2))).tolist()
    train_dataloader = DataLoader(train_dataset, num_workers=32, batch_size=250, sampler = torch.utils.data.SubsetRandomSampler(inds), pin_memory=True) #
    print(len(train_dataloader))
    tac = time.time()
    for batch_idx, data in enumerate(train_dataloader):
        print(batch_idx)
#         if batch_idx==500:
#             print(time.time()-tic)
#             print(time.time()-tac)
#             break