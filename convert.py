import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
from sigpy.mri import app
import torch
from sigpy.mri import samp

def ground_k(file):
     with h5py.File(file, 'r') as f:
          kspace = f['kspace'][:]
          print('kspace:', kspace.shape)
     return kspace

def get_ground(k):
     recon = sp.rss(sp.ifft(k, axes=[-2, -1]), axes=(-3))
     combine_recon = sp.resize(recon, oshape = [k.shape[0], 384, 384]) 
     combine_recon = combine_recon / np.linalg.norm(combine_recon) * 100
     return combine_recon

def get_coil_sens(k):
     device = sp.Device(0) if torch.cuda.is_available() else sp.cpu_device
     print('> device: ', device)
     kspace_dev = sp.to_device(k, device=device)
     cs = []
     for s in range(k.shape[0]):
          k = kspace_dev[s]
          c =app.EspiritCalib(k, device=device).run()
          # crop
          c_shape = c.shape[:]
          start_X = (c_shape[2] - 384) // 2
          start_y = (c_shape[1] - 384) // 2
          crop_c = c[:, start_y : start_y + 384, start_X : start_X + 384]
          cs.append(sp.to_device(crop_c))
     return np.array(cs)

def undersample_mask(ground, acc):        
     _, _, y, x = ground.shape
     mask_cartes = np.zeros([y, x])
     mask_cartes[::4, :] = 1
     # common_mask = samp.poisson([y, x], acc)
     mask_height, mask_width = mask_cartes.shape
     start_y = (mask_height - 384) // 2
     start_x = (mask_width - 384) // 2
     cropped_mask = mask_cartes[start_y:start_y + 384, start_x:start_x + 384]
     us_mask = np.tile(cropped_mask, (ground.shape[0], 1, 1))
     return us_mask

def get_undersample(k, mask):
     kspace = mask * k
     return kspace

def process_file(file_list):
     all_ground_recon = []
     all_csm = []
     all_us_mask = []
    
     for idx, file in enumerate(file_list):
        print('> idx %4d, file %s' % (idx, file))
        full_path = os.path.join('./processedData/processed', file)
        ground_kspace = ground_k(full_path)

        if ground_kspace.shape[1] == 16:
            k = ground_kspace
            ground_recon = get_ground(k).astype(np.complex64)
            csm = get_coil_sens(k).astype(np.complex64)
            us_mask = undersample_mask(k, 2).astype(np.int8)
            
            all_ground_recon.append(ground_recon)
            all_csm.append(csm)
            all_us_mask.append(us_mask)

     return np.concatenate(all_ground_recon, axis=0), np.concatenate(all_csm, axis=0), np.concatenate(all_us_mask, axis=0)

path = './processedData/processed'
processed_dir = './processedData'
processed_file = 'fastMRI_4_cart.h5'
processed_path = os.path.join(processed_dir, processed_file)
files = os.listdir(path)
train_files = files[:25]
test_files = files[25:50]

train_ground_recon, train_csm, train_us_mask = process_file(train_files)
test_ground_recon, test_csm, test_us_mask = process_file(test_files)

with h5py.File(processed_path, 'w') as hf:
    hf.create_dataset('trnCsm', data=train_csm)
    hf.create_dataset('trnMask', data=train_us_mask)
    hf.create_dataset('trnOrg', data=train_ground_recon) 
    hf.create_dataset('tstCsm', data=test_csm)
    hf.create_dataset('tstMask', data=test_us_mask)
    hf.create_dataset('tstOrg', data=test_ground_recon)
   
                         