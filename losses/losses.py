import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.tensor as Tensor
from  torch import Tensor

import sys 
import os
sys.path.append("..")
from dataset import denormalize_zscore, denormalize_max_min

from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points

class ReconLoss(nn.Module):
    def __init__(self, beta=1, dataname='vortex'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.beta = beta
        self.cos_sim =  F.cosine_similarity
        self.dataname = dataname
        
    def forward(self, output, target):
        if self.dataname == 'tornado':
            recon_loss = self.l1(output, target)
        else:
            recon_loss = self.mse(output, target)
        
        out_loss = self.beta * recon_loss
        
        if self.dataname == 'tornado':
            cos_loss = 1 - self.cos_sim(output, target, dim=1)
            out_loss = out_loss + torch.mean(cos_loss)

        return out


class Metrics(nn.Module):
    def __init__(self, norm_method='mean_std', mean_std_norm=None, min_max_norm=None):
        super().__init__()
        import pdb
        pdb.set_trace()
        self.diff = min_max_norm['max']-min_max_norm['min']
        self.mean_std_norm = mean_std_norm
        self.min_max_norm = min_max_norm
        self.norm_method = norm_method
        
    def MSE(self, x, y):
        # MSE or Weighted Mean Squared Error (WMSE)
        # x, y: 5D # 4D [0, 1]
        if self.norm_method == 'mean_std':
            x = denormalize_zscore(x, self.mean_std_norm)
            y = denormalize_zscore(y, self.mean_std_norm)
        elif self.norm_method == 'min_max':
            x = denormalize_max_min(x, self.min_max_norm)
            y = denormalize_max_min(y, self.min_max_norm)
        return torch.mean((x - y) ** 2, dim=[1, 2, 3, 4])

    def PSNR(self, x, y):
        # x, y: 5D [0, 1]
        mse = self.MSE(x, y)
        psnr = 10 * torch.log10(self.diff ** 2. / mse)  # (B,)
        return torch.mean(psnr)

    def forward(self, output, target):
        psnr = self.PSNR(output, target)
        return psnr

    
# def masked_mse_loss(output, target, mask):
#     '''
#         output: (B,N,D)
#         target: (B,N,D)
#         mask: (B,N,1)
#     '''
#     _,_,D = output.shape
#     out = torch.sum(((output-target)*mask)**2.0)  / (torch.sum(mask)*D)
#     return out


# from typing import List

# from kaolin.metrics.pointcloud import chamfer_distance as history_chamfer_distance
# from metric.emd.emd_module import emdFunction
# from pytorch3d.loss import chamfer_distance




