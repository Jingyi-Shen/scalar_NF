# from numpy.testing._private.utils import decorate_methods
import pandas as pd
import numpy as np
import random
from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import hflip, to_tensor
import torch.nn.functional as F

import struct
import json
# import scipy.ndimage
import os
import math
from skimage.transform import rescale, resize

import vtk
from vtkmodules.all import *
from vtkmodules.util import numpy_support

from scipy.spatial import KDTree
from models.modules.utils.attr_kdtree import KDTree as AKDTree

from utils.utils_base import min_max
from utils.utils_reg import read_vortex_data, read_combustion_data, read_nyx_data, read_data_bin

min_max_nyx = {
    'min': [8.773703], 
    'max': [12.799037]
}

# min_max_nyx_emsenble = {
#     'min': [8.649637], 
#     'max': [13.584945]
# }

# min_max_fpm = {
# 	'min': [0, -5.63886223e+01, -3.69567909e+01, -7.22953186e+01],
# 	'max': [357.19000244, 38.62746811, 48.47133255, 50.60621262]
# }

# min_max_fpm = {
# 	'min': [0, -6.8942547, -5.760195, -6.0264626],
# 	'max': [357.19000244, 8.238125, 8.214644, 6.751672]
# }

min_max_fpm = {
	'min': [0],
	'max': [357.19000244]
}

# mean:  [-1.34590493e-02  3.07447077e-02  4.86026144e+00  2.37552388e+01  1.87237223e-02 -2.11847946e-02  1.94381937e-02]
# min:  [-4.99989   -4.9999275  0.         0.        -6.8942547 -5.760195  -6.0264626]
# max:  [4.9999995   4.999923   10.        357.19     8.238125    8.214644  6.751672 ]

min_max_coord_fpm = {
    'min': [-4.99989, -4.9999275, 0.],
	'max': [4.9999995, 4.999923, 10.]
}

min_max_cos = {
	'min': [-2466, -2761, -2589, -17135.6, -20040, -20096, -6928022],
	'max': [2.7808181e+03, 2.9791230e+03, 2.6991892e+03, 1.9324572e+04, 2.0033873e+04, 1.7973633e+04, 6.3844562e+05]
}

# min_max_jet3b = {
# 	'min': [-1.50166025e+01, 1.47756422e+00],
# 	'max': [1.24838667e+01, 1.00606432e+01]
# }

min_max_jet3b = {
	'min': [-1.50166025e+01],
	'max': [1.24838667e+01]
}

# min_max_coord_jet3b = {
# 	'min': [-410, -410, -410],
# 	'max': [410, 410, 410]
# }

min_max_vortex = {
    'min': [0.005305924], 
	'max': [10.520865]
}


def normalize_zscore(d, mean_std):
    return (d - mean_std['mean']) / mean_std['std']

def denormalize_zscore(d, mean_std):
    return d * mean_std['std'] + mean_std['mean']
 
def normalize_max_min(d, min_max): 
    return np.divide((np.subtract(d, min_max['min'])), (np.subtract(min_max['max'], min_max['min'])))
    # return (d - min_max['min']) / (min_max['max'] - min_max['min'])

def denormalize_max_min(d, min_max):
    return np.add( np.multiply(d, np.subtract(min_max['max'], min_max['min'])), min_max['min'])
    # return d * (min_max['max'] - min_max['min']) + min_max['min']

def numpy_to_torch(block_d):
    return torch.from_numpy(block_d.astype(np.float32))


def collect_file(directory, mode, shuffle=True):
    file_list = []
    if mode == "nyx_ensemble":
        dirnames = os.listdir(directory)
        dirnames.sort()
        dirnames = dirnames[:80]
        # dirnames = [dirname for dirname in dirnames if np.abs(float(dirname.split('_')[-1])-0.6) < 0.08][:8]
        # print(dirnames, len(dirnames))
        for idx, dirname in enumerate(dirnames):
            inp = os.path.join(directory, dirname, 'Raw_plt256_00200', 'density.bin') 
            file_list.append(inp)
    else:
        for (dirpath, dirnames, filenames) in os.walk(directory):
            for filename in sorted(filenames):
                if mode == "fpm":
                    if filename.endswith(".vtu") and filename != "000.vtu": 
                        inp = os.sep.join([dirpath, filename])
                        file_list.append(inp)
                elif mode == "cos":
                    if "ds14" in filename.split('_'):
                        inp = os.sep.join([dirpath, filename])
                        file_list.append(inp)
                elif mode == "jet3b":
                    if "run3g" in filename.split('_'):
                        inp = os.sep.join([dirpath, filename])
                        file_list.append(inp)
                elif mode == "vortex":
                    if filename.endswith(".data"): 
                        inp = os.sep.join([dirpath, filename])
                        file_list.append(inp)
                elif mode == "nyx" or mode == "nyx_ensemble":
                    if filename.endswith(".bin"): 
                        inp = os.sep.join([dirpath, filename])
                        file_list.append(inp)
    if shuffle:
        random.shuffle(file_list)
    else:
        file_list.sort()
    if mode == "fpm":
        print(file_list[80])
        return [file_list[80]]
    elif mode == "jet3b":
        print(file_list[30])
        return [file_list[30]]
    elif mode == 'vortex':
        print(file_list[:5])
        return file_list[:5]
        # print('/fs/ess/PAS0027/vortex_data/vortex/vorts01.data')
        # return ['/fs/ess/PAS0027/vortex_data/vortex/vorts01.data']
    elif mode == 'nyx':
        print(file_list[0])
        return [file_list[0]]
    elif mode == 'nyx_ensemble':
        print(file_list)
        # return [file_list[0]]
        return file_list


def read_data_from_file(dataname, data_path):
    if dataname == 'fpm':
        file_list = collect_file(os.path.join(data_path, "fpm"), dataname, shuffle=True)
    elif dataname == 'cos':
        file_list = collect_file(os.path.join(data_path, "ds14_scivis_0128/raw"), dataname, shuffle=True)
    elif dataname == 'jet3b':
        file_list = collect_file(os.path.join(data_path, "jet3b"), dataname, shuffle=True)
    elif dataname == 'nyx':
        file_list = collect_file(os.path.join(data_path, "nyx/256/256CombineFiles/raw"), dataname, shuffle=True)
    elif dataname == "vortex":
        file_list = collect_file(os.path.join(data_path, "vortex_data/vortex"), dataname, shuffle=True)
    elif dataname == 'nyx_ensemble':
        # file_list = collect_file('./data/density/nyx/raw', dataname, shuffle=True)
        file_list = collect_file(os.path.join(data_path, "nyx/256/output"), dataname, shuffle=True)
    return file_list


def data_reader(filename, dataname, normalize=True, padding=0, scale=1, order=3):
    if dataname == 'cos':
        data, name = read_data_bin(filename, scale=scale, order=order)
        attr_min_max = min_max_cos
        # mean = [30.4, 32.8, 32.58, 0, 0, 0, 0, 0, 0, -732720]
        # std = [18.767, 16.76, 17.62, 197.9, 247.2, 193.54, 420.92, 429, 422.3, 888474]
    elif dataname == 'fpm':
        data, name = read_data_bin(filename, scale=scale, order=order)
        attr_min_max = min_max_fpm
        # mean = [0, 0, 5, 23.9, 0, 0, 0.034]
        # std = [2.68, 2.68, 3.09, 55.08, 0.3246, 0.3233, 0.6973]
    elif dataname =='jet3b':
        data, name = read_data_bin(filename, scale=scale, order=order)
        attr_min_max = min_max_jet3b
    elif dataname == 'vortex':
        data, name = read_vortex_data(filename, padding=padding, scale=scale, order=order)
        attr_min_max = min_max_vortex
    elif dataname == 'nyx' or dataname == 'nyx_ensemble':
        data, name = read_nyx_data(filename, padding=padding, scale=scale, order=order)
        attr_min_max = min_max_nyx
        if dataname == 'nyx_ensemble': # for data direcltly read from /fs/ess/PAS0027/nyx/256/output/
            data = np.log10(data)
    elif dataname == 'na':
        # for nyx
        shape = np.array([256, 256, 256]) // scale
        data, name = read_data_bin(filename, padding=padding, order=order, shape=shape)
        attr_min_max = min_max_nyx

    if normalize:
        data = normalize_max_min(data, attr_min_max)
      
    return np.float32(data), attr_min_max #convert data to normalized float32


class ScalarData(Dataset):
    def __init__(self, data, mode='train', scale_factor=2, blocksize=24, padding=4, sampler=600, p=0.3, shape=None):
        self.sf = scale_factor 
        self.p = p
        
        self.padding = padding
        self.blocksize = blocksize
        self.blocksize_low = blocksize // scale_factor
        self.padding_low   = padding // scale_factor
        
        self.data = data
        self.size = data.shape # train need self.size to be high res, test needs self.size to be low res
        self.mode = mode
        
        if self.mode == 'train':
            self.data_length = sampler
        elif self.mode == 'eval':
            self.x_cnt = int(shape[2]/(self.blocksize-2*self.padding))  
            self.y_cnt = int(shape[1]/(self.blocksize-2*self.padding)) 
            self.z_cnt = int(shape[0]/(self.blocksize-2*self.padding))
            self.data_length = self.x_cnt * self.y_cnt * self.z_cnt
        elif self.mode == "test":
            self.x_cnt = int(shape[2]/(self.blocksize_low-2*self.padding_low))  
            self.y_cnt = int(shape[1]/(self.blocksize_low-2*self.padding_low)) 
            self.z_cnt = int(shape[0]/(self.blocksize_low-2*self.padding_low))
            self.data_length = self.x_cnt * self.y_cnt * self.z_cnt
        # print(f'[{mode}set] {self.data_length} blocks')

    def __getitem__(self, idx):
        if self.mode == 'train':
            # print(self.size[2], 2*self.padding, self.blocksize)
            if self.size[2] - 2*self.padding - self.blocksize < 1:
                ii = 0
                jj = 0
                kk = 0
            else:
                ii = random.randint(0, self.size[2] - 2*self.padding - self.blocksize)
                jj = random.randint(0, self.size[1] - 2*self.padding - self.blocksize)
                kk = random.randint(0, self.size[0] - 2*self.padding - self.blocksize)
            # if self.mode == 'train': # get corresponding low res
            #     sample = random.random()
            #     0: nearest; 1: bilinear; 2: trilinear?Bi-quadratic?
            #     if sample < self.p:
            #         block_low = rescale(block_high, [1./self.sf, 1./self.sf, 1./self.sf], order=0)
            #     elif sample < 2 * self.p:
            #         block_low = rescale(block_high, [1./self.sf, 1./self.sf, 1./self.sf], order=1)
            #     else:
            #         block_low = rescale(block_high, [1./self.sf, 1./self.sf, 1./self.sf], order=2)
            block_high = self.data[kk:kk+self.blocksize, jj:jj+self.blocksize, ii:ii+self.blocksize]
            block_low = rescale(block_high, [1./self.sf, 1./self.sf, 1./self.sf], order=2)

        elif self.mode == 'eval': # take high as input
            kk = int(idx / (self.x_cnt * self.y_cnt))
            jj = int((idx - kk * (self.x_cnt * self.y_cnt)) / self.x_cnt)
            ii = int(idx - kk * (self.x_cnt * self.y_cnt) - jj * self.x_cnt)
            bs = self.blocksize - 2*self.padding
            block_high = self.data[kk*bs:kk*bs+self.blocksize, jj*bs:jj*bs+self.blocksize, ii*bs:ii*bs+self.blocksize]
            block_low = rescale(block_high, [1./self.sf, 1./self.sf, 1./self.sf], order=2)
        
        elif self.mode == 'test': # take low as input
            kk = int(idx / (self.x_cnt * self.y_cnt))
            jj = int((idx - kk * (self.x_cnt * self.y_cnt)) / self.x_cnt)
            ii = int(idx - kk * (self.x_cnt * self.y_cnt) - jj * self.x_cnt)
            bs = self.blocksize_low - 2*self.padding_low
            block_low = self.data[kk*bs:kk*bs+self.blocksize_low, jj*bs:jj*bs+self.blocksize_low, ii*bs:ii*bs+self.blocksize_low]
            block_high = rescale(block_low, [self.sf, self.sf, self.sf], order=2)
        
        # print('data', self.data.shape, block_low.shape, block_high.shape)
        # print('size', self.size)
        # print(block_high.shape, block_low.shape )
        
        # block high is residual between lerp(low) and high
        # lerp_high = rescale(block_low, [self.sf, self.sf, self.sf], order=3)
        # block_high = block_high - lerp_high

        block_low = block_low[None, ...] # add channel dimension
        block_high = block_high[None, ...]

        if self.mode == 'eval' or  self.mode == 'test':
            return numpy_to_torch(block_high), numpy_to_torch(block_low), kk, jj, ii
        else:
            return numpy_to_torch(block_high), numpy_to_torch(block_low)
        
    def __len__(self):
        return self.data_length


# they have (fixed) mean low resolution 
class ScalarEnsembleData(Dataset):
    def __init__(self, data, data_low, mode='train', scale_factor=2, blocksize=24, padding=4, sampler=600, p=0.3, shape=None):
        self.sf = scale_factor 
        self.p = p
        
        self.padding = padding
        self.blocksize = blocksize
        self.blocksize_low = blocksize // scale_factor
        self.padding_low   = padding // scale_factor
        
        self.data = data
        self.size = data.shape if data is not None else shape
        self.mode = mode
        self.data_low = data_low # read_data_bin(low_res_path)
        
        self.x_cnt = int(self.size[2]/(self.blocksize_low-2*self.padding_low))  
        self.y_cnt = int(self.size[1]/(self.blocksize_low-2*self.padding_low)) 
        self.z_cnt = int(self.size[0]/(self.blocksize_low-2*self.padding_low))

        if self.mode == 'train':
            self.data_length = sampler
        elif self.mode == 'eval':
            self.data_length = self.x_cnt * self.y_cnt * self.z_cnt
        # print(f'[{mode}set] {self.data_length} blocks')
        # print('self.x_cnt', self.x_cnt, self.y_cnt, self.z_cnt)

    def __getitem__(self, idx):
        if self.mode == 'train':
            # print(self.size[2], 2*self.padding, self.blocksize)
            if self.size[2]//2 - 2*self.padding_low - self.blocksize_low < 1:
                ii = 0
                jj = 0
                kk = 0
            else:
                ii = random.randint(0, self.size[2]//2 - 2*self.padding_low - self.blocksize_low)
                jj = random.randint(0, self.size[1]//2 - 2*self.padding_low - self.blocksize_low)
                kk = random.randint(0, self.size[0]//2 - 2*self.padding_low - self.blocksize_low)
            # print(ii, jj, kk)
            block_low = self.data_low[kk:kk+self.blocksize_low, jj:jj+self.blocksize_low, ii:ii+self.blocksize_low]
            block_high = self.data[kk*self.sf:kk*self.sf+self.blocksize, jj*self.sf:jj*self.sf+self.blocksize, ii*self.sf:ii*self.sf+self.blocksize]
            block_low = block_low[None, ...] # add channel dimension
            block_high = block_high[None, ...]
            # print('data', self.data.shape, block_low.shape)
            
        elif self.mode == 'eval': # take low as input
            kk = int(idx / (self.x_cnt * self.y_cnt))
            jj = int((idx - kk * (self.x_cnt * self.y_cnt)) / self.x_cnt)
            ii = int(idx - kk * (self.x_cnt * self.y_cnt) - jj * self.x_cnt)
            bs = self.blocksize_low - 2*self.padding_low
            block_low = self.data_low[kk*bs:kk*bs+self.blocksize_low, jj*bs:jj*bs+self.blocksize_low, ii*bs:ii*bs+self.blocksize_low]
            block_low = block_low[None, ...]

        if self.mode == 'eval':
            return numpy_to_torch(block_low), numpy_to_torch(block_low), kk, jj, ii
        else:
            return numpy_to_torch(block_high), numpy_to_torch(block_low)
        
    def __len__(self):
        return self.data_length


def get_size(dataname, sf=2, level=0):
    ds_scale = np.power(sf, level)
    if dataname == 'isabel':
        size = [96 // ds_scale, 512 // ds_scale, 512 // ds_scale] #z, y, x 
    elif dataname == 'vortex':
        size = [128 // ds_scale, 128 // ds_scale, 128 // ds_scale]
    elif dataname == 'combustion':
        size = [128 // ds_scale, 720 // ds_scale, 480 // ds_scale]
    elif dataname == 'nyx' or dataname == 'nyx_ensemble':
        size = [256 // ds_scale, 256 // ds_scale, 256 // ds_scale]
    return size


def get_eval_dataloader(args, dataname, filepath, normalize=True, blocksize=24, padding=0, batch_size_test=128, level=0, order=3):
    ds_scale = np.power(args.sf, level) # size of GT high res
    test_high_data, attr_min_max = data_reader(filepath, dataname, normalize=normalize, padding=padding, scale=ds_scale, order=order) # normalized
    test_dataset = ScalarData(test_high_data, mode='eval', scale_factor=args.sf, blocksize=blocksize, padding=padding, shape=get_size(dataname, level=level))                
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, drop_last=False, num_workers=0)
    test_gt_data, _ = data_reader(filepath, dataname, normalize=normalize, padding=padding, scale=1) # (ds_scale//2)
    return test_dataloader, attr_min_max, test_high_data, test_gt_data
    
def get_test_dataloader(args, dataname, filepath, normalize=True, blocksize=24, padding=0, batch_size_test=128, level=0, order=3):
    ds_scale = np.power(args.sf, level) # size of GT low res
    test_low_data, attr_min_max = data_reader(filepath, dataname, normalize=normalize, padding=padding//args.sf, scale=ds_scale, order=order) # normalized
    test_dataset = ScalarData(test_low_data, mode='test', scale_factor=args.sf, blocksize=blocksize, padding=padding, shape=get_size(dataname, level=level))                
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, drop_last=False, num_workers=0)
    test_gt_data, _ = data_reader(filepath, dataname, normalize=normalize, padding=padding, scale=1) # (ds_scale//2)
    return test_dataloader, attr_min_max, test_low_data, test_gt_data

def get_test_nyxensemble_loader(args, dataname, filepath, normalize=True, blocksize=24, padding=0, batch_size_test=128, level=0, order=3):
    ds_scale = np.power(args.sf, level) 
    f_low = f'./data/density/nyx/low_res/nyx_low_level{level}.bin' # 
    test_data, attr_min_max = data_reader(f_low, 'na', normalize=normalize, padding=padding//args.sf, scale=ds_scale, order=order) # normalized
    test_dataset = ScalarEnsembleData(data=None, data_low=test_data, mode='eval', scale_factor=args.sf, blocksize=blocksize, padding=padding, shape=get_size('nyx_ensemble', level=level))                
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, drop_last=False, num_workers=0)
    test_gt_data, attr_min_max = data_reader(filepath, dataname, normalize=normalize, padding=padding, scale=ds_scale//args.sf, order=order) # (ds_scale*2)
    return test_dataloader, attr_min_max, test_data, test_gt_data



if __name__ == "__main__":
    # from_sdf_to_vtu()
    # generate_low_res()
    print('ScalarData Class')
    

    
    


