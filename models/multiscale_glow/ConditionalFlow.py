import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.multiscale_glow import thops
from models.multiscale_glow.Basic import Conv3d, Conv3dZeros, GaussianDiag, DenseBlock, RRDB, FCN, RRDB_v2
from models.multiscale_glow.FlowStep import FlowStep

import functools
from models.multiscale_glow import module_util as mutil

class ConditionalFlow(nn.Module):
    def __init__(self, num_channels, num_channels_split, n_flow_step=0, num_levels_condition=0, SR=True, RRDB_nb=[3, 3], RRDB_nf=16, 
                    flow_actNorm='actNorm3d', flow_permutation='invconv', flow_coupling='Affine',
                    kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.SR = SR

        # number of levels of RRDB features. One level of conditional feature is enough for image rescaling
        # num_features_condition = 2 if self.SR else 1
        num_features_condition = 1

        # feature extraction
        # RRDB_nb = [5, 5] # opt_get(opt, ['RRDB_nb'], [5, 5])
        # RRDB_nf = RRDB_nf # opt_get(opt, ['RRDB_nf'], 64) # 16
        RRDB_gc = 16 # opt_get(opt, ['RRDB_gc'], 32)
        
        RRDB_f = functools.partial(RRDB, nf=RRDB_nf, gc=RRDB_gc) 
        # RRDB_f = functools.partial(RRDB_v2, nf=RRDB_nf, gc=RRDB_gc) 
        self.conv_first = nn.Conv3d(num_channels_split + RRDB_nf*num_features_condition*num_levels_condition, RRDB_nf, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.trunk_conv1 = nn.Conv3d(RRDB_nf, RRDB_nf, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.f = Conv3dZeros(RRDB_nf*num_features_condition, (num_channels-num_channels_split)*2)
        
        self.RRDB_trunk0 = mutil.make_layer(RRDB_f, RRDB_nb[0])
        self.RRDB_trunk1 = mutil.make_layer(RRDB_f, RRDB_nb[1])
        
        # conditional flow
        self.additional_flow_steps = nn.ModuleList()
        for k in range(n_flow_step):
            self.additional_flow_steps.append(FlowStep(in_channels=num_channels-num_channels_split,
                                                                  cond_channels=RRDB_nf*num_features_condition, # cond is not none
                                                                  flow_actNorm=flow_actNorm,
                                                                  flow_permutation=flow_permutation,  #opt['flow_permutation'],
                                                                  flow_coupling=flow_coupling)) #opt['flow_coupling']))

    def forward(self, z, u, eps_std=None, logdet=0, reverse=False, training=True): # u: condition
        # for SR
        if not reverse:
            # [B, 16, 128] --split--> [B, 8, 128] u, z
            # output: conditional_feature [B, 32, 128]
            conditional_feature = self.get_conditional_feature_SR(u)

            for layer in self.additional_flow_steps:
                z, logdet = layer(z, u=conditional_feature, logdet=logdet, reverse=False)
            
            # [B, 32, 128] --> [B, 16, 128]
            h = self.f(conditional_feature)
            mean, logs = thops.split_feature(h, "cross")
            logdet += GaussianDiag.logp(mean, logs, z)

            return z, logdet, conditional_feature

        else:
            conditional_feature = self.get_conditional_feature_SR(u)

            h = self.f(conditional_feature)
            mean, logs = thops.split_feature(h, "cross")

            if z is None:
                z = GaussianDiag.sample(mean, logs, eps_std)
            logdet -= GaussianDiag.logp(mean, logs, z)

            for layer in reversed(self.additional_flow_steps):
                z, logdet = layer(z, u=conditional_feature, logdet=logdet, reverse=True)

            return z, logdet, conditional_feature
    
    def get_conditional_feature_SR(self, u):
        u_feature_first = self.conv_first(u)
        u_feature = self.trunk_conv1(self.RRDB_trunk1(self.RRDB_trunk0(u_feature_first))) + u_feature_first
        return u_feature
    
    # def get_conditional_feature_SR(self, u):
    #     import pdb
    #     pdb.set_trace()
    #     u_feature_first = self.conv_first(u)
    #     u_feature1 = self.RRDB_trunk0(u_feature_first)
    #     u_feature2 = self.trunk_conv1(self.RRDB_trunk1(u_feature1)) + u_feature_first
    #     return torch.cat([u_feature1, u_feature2], 1)

    # def get_conditional_feature_Rescaling(self, u):
    #     u_feature_first = self.conv_first(u)
    #     u_feature = self.trunk_conv1(self.RRDB_trunk1(self.RRDB_trunk0(u_feature_first))) + u_feature_first
    #     return u_feature

