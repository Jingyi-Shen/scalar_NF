import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

from models.multiscale_glow import Basic
from models.multiscale_glow.FlowStep import FlowStep, FlowAssembly
from models.multiscale_glow.ConditionalFlow import ConditionalFlow
from models.multiscale_glow.layers_util import GDN3d, BasicBlockEncoder


class FlowNet(nn.Module):
    def __init__(self, C=1, K=8, L=1, splitOff=[4,4,4], RRDB_nb=[3, 3], RRDB_nf=16, preK=3, factor=4, ks_s_pad=[1, 1, 0]):
        super().__init__()
        # self.C, H, W = data_shape
        self.C = C
        self.L = L #opt_get(opt, ['network_G', 'flowDownsampler', 'L'])
        self.K = K #opt_get(opt, ['network_G', 'flowDownsampler', 'K']) 8
        self.preK = preK
        if isinstance(self.K, int): 
            self.K = [self.K] * (self.L + 1)

        flow_actNorm='actNorm3d'
        # flow_actNorm="none"
        flow_permutation = 'invconv' # opt_get(self.opt, ['network_G', 'flowDownsampler', 'flow_permutation'], 'invconv')
        flow_coupling = 'Affine' # opt_get(self.opt, ['network_G', 'flowDownsampler', 'flow_coupling'], 'Affine')
        print(flow_actNorm, flow_permutation, flow_coupling)
        
        cond_channels = None # opt_get(self.opt, ['network_G', 'flowDownsampler', 'cond_channels'], None)
        # cond_channels_up = [self.C]*self.preK + [self.C * factor] * (self.K[0] - splitOff[0])
        enable_splitOff = True # opt_get(opt, ['network_G', 'flowDownsampler', 'splitOff', 'enable'], False)
        after_splitOff_flowStep = splitOff #opt_get(opt, ['network_G', 'flowDownsampler', 'splitOff', 'after_flowstep'], 0) [5, 5, 5]
        # squeeze: haar # better than squeeze2d
        # flow_permutation: none # bettter than invconv
        # flow_coupling: Affine3shift # better than affine
        if isinstance(after_splitOff_flowStep, int): 
            after_splitOff_flowStep = [after_splitOff_flowStep] * (self.L + 1)

        # construct flow
        self.layers = nn.ModuleList()
        # self.output_shapes = []
        # print('after_splitOff_flowStep', after_splitOff_flowStep)
        for level in range(self.L):
            # coupling layers
            if factor > 0:
                # 1. Squeeze
                # re-organize N*N points into image-like NxN points with 3 channels
                self.layers.append(Basic.SqueezeLayer(factor)) # may need a better way for squeezing
                self.C = self.C * (factor**3) #, D // 2, H // 2, W // 2
                
            # self.output_shapes.append([-1, self.C, H, W])
            # 2. main FlowSteps (unconditional flow)
            for k in range(self.K[level]-after_splitOff_flowStep[level]):
                self.layers.append(FlowStep(in_channels=self.C, cond_channels=cond_channels,
                                                       flow_actNorm=flow_actNorm,
                                                       flow_permutation=flow_permutation,
                                                       flow_coupling=flow_coupling,
                                                       ks_s_pad=ks_s_pad))

            # 3. additional FlowSteps (split + conditional flow)
            if enable_splitOff:
                if level == 0:
                    # self.layers.append(Basic.Split(num_channels_split=self.C // 2 if level < self.L-1 else 3, level=level))
                    self.layers.append(Basic.Split(num_channels_split=self.C//2, level=level)) # self.C // 2
                    self.level0_condFlow = ConditionalFlow(num_channels=self.C,
                                                    num_channels_split=self.C//2,
                                                    n_flow_step=after_splitOff_flowStep[level],
                                                    num_levels_condition=0, SR=True, RRDB_nb=RRDB_nb, RRDB_nf=RRDB_nf,
                                                    flow_actNorm = flow_actNorm,
                                                    flow_permutation=flow_permutation,
                                                    flow_coupling=flow_coupling, 
                                                    kernel_size=ks_s_pad[0], stride=ks_s_pad[1], padding=ks_s_pad[2])
                                                    # num_levels_condition=1, SR=True)
                elif level == 1:
                    self.layers.append(Basic.Split(num_channels_split=self.C//2, level=level))
                    self.level1_condFlow = ConditionalFlow(num_channels=self.C,
                                                    num_channels_split=self.C//2,
                                                    n_flow_step=after_splitOff_flowStep[level],
                                                    num_levels_condition=0, SR=True, RRDB_nb=RRDB_nb, RRDB_nf=RRDB_nf,
                                                    flow_actNorm = flow_actNorm,
                                                    flow_permutation=flow_permutation,
                                                    flow_coupling=flow_coupling,
                                                    kernel_size=ks_s_pad[0], stride=ks_s_pad[1], padding=ks_s_pad[2])
                self.C = self.C // 2 # if level < self.L-1 else 3

    def forward(self, hr=None, z=None, u=None, eps_std=None, reverse=False, training=True, z1=None, z2=None):
        if hr is not None:
            B = hr.shape[0]
            device = hr.device
        elif z is not None:
            B = z.shape[0]
            device = z.device
        logdet = torch.zeros((B,), device=device)
        if not reverse:
            return self.normal_flow(hr, u=u, logdet=logdet, training=training)
        else:
            return self.reverse_flow(z, u=u, logdet=logdet, eps_std=eps_std, training=training, z1=z1, z2=z2)

    '''
    hr->y1+z1->y2+z2
    '''
    def normal_flow(self, z, u=None, logdet=None, training=True):
        # for layer, shape in zip(self.layers, self.output_shapes): 
        for layer in self.layers:
            # print('normal', layer, z.shape)
            if isinstance(layer, FlowStep):
                z, logdet = layer(z, u, logdet=logdet, reverse=False)
            elif isinstance(layer, Basic.SqueezeLayer):
                z, logdet = layer(z, logdet=logdet, reverse=False)
            elif isinstance(layer, Basic.Split):
                z, a1 = layer(z, reverse=False)
                y1 = z.clone()
                z1, logdet, _ = self.level0_condFlow(a1, z, logdet=logdet, reverse=False, training=training) 
                z2 = None
                # if layer.level == 0:
                #     z, a1 = layer(z, reverse=False)
                #     y1 = z.clone()
                #     print('z, a1', z.shape, a1.shape)
                # elif layer.level == 1:
                #     z, a2 = layer(z, reverse=False)  # z --> y2  # y2
                #     print('z, a2', z.shape, a2.shape)
                #     z2, logdet, conditional_feature2 = self.level1_condFlow(a2, z, logdet=logdet, reverse=False, training=training) # z2
                #     conditional_feature1 = torch.cat([y1, F.interpolate(conditional_feature2, scale_factor=2, mode='nearest')],1)
                #     z1, logdet, _ = self.level0_condFlow(a1, conditional_feature1, logdet=logdet, reverse=False, training=training)
        return z, logdet, z1, z2

    '''
    y2+z2->y1+z1->hr
    '''
    def reverse_flow(self, z, u=None, logdet=None, eps_std=None, training=True, z1=None, z2=None):
        # for layer, shape in zip(reversed(self.layers), reversed(self.output_shapes)):
        for layer in reversed(self.layers): 
            if isinstance(layer, FlowStep):
                z, logdet = layer(z, u, logdet=logdet, reverse=True)
            elif isinstance(layer, Basic.SqueezeLayer):
                z, logdet = layer(z, logdet=logdet, reverse=True)
            elif isinstance(layer, Basic.Split):
                a1, logdet, _ = self.level0_condFlow(z1, z, eps_std=eps_std, logdet=logdet, reverse=True, training=training)
                z = layer(z, a1, reverse=True)
                # if layer.level == 1:
                #     a2, logdet, conditional_feature2 = self.level1_condFlow(z2, z, eps_std=eps_std, logdet=logdet, reverse=True, training=training)
                #     z = layer(z, a2, reverse=True)
                # elif layer.level == 0:
                #     conditional_feature1 = torch.cat([z, F.interpolate(conditional_feature2, scale_factor=2, mode='nearest')],1)
                #     a1, logdet, _ = self.level0_condFlow(z1, conditional_feature1, eps_std=eps_std, logdet=logdet, reverse=True, training=training)
                #     z = layer(z, a1, reverse=True)
        return z
        
    def sample(self, z, u=None, logdet=0, eps_std=0.9, training=True):
        return self.reverse_flow(z, u=u, logdet=logdet, eps_std=eps_std, training=training, z1=None, z2=None)



class FlowNetInjectCond(nn.Module):
    def __init__(self, data_shape, C=1, K=8, L=1, splitOff=[4,4,4], RRDB_nb=[3, 3], RRDB_nf=16, preK=3, factor=4, ks_s_pad=[1, 1, 0]):
        super(FlowNetInjectCond, self).__init__()
        self.C = C
        self.L = L #opt_get(opt, ['network_G', 'flowDownsampler', 'L'])
        self.K = K #opt_get(opt, ['network_G', 'flowDownsampler', 'K']) 8
        self.preK = preK
        if isinstance(self.K, int): 
            self.K = [self.K] * (self.L + 1)
        
        feat_channels = [self.C, 16, 32] + [64] * (K - 2)
        hidden_channels = [8, 16] + [32] * (K - 2)
        cond_channels = [16, 32] + [64] * (K - 2)

        flow_actNorm='actNorm3d'
        # flow_actNorm="none"
        flow_permutation = 'invconv' # opt_get(self.opt, ['network_G', 'flowDownsampler', 'flow_permutation'], 'invconv')
        flow_coupling = 'Affine' # opt_get(self.opt, ['network_G', 'flowDownsampler', 'flow_coupling'], 'Affine')
        print(flow_actNorm, flow_permutation, flow_coupling)
        
        enable_splitOff = True # opt_get(opt, ['network_G', 'flowDownsampler', 'splitOff', 'enable'], False)
        after_splitOff_flowStep = splitOff #opt_get(opt, ['network_G', 'flowDownsampler', 'splitOff', 'after_flowstep'], 0) [5, 5, 5]
        # squeeze: haar # better than squeeze2d
        # flow_permutation: none # bettter than invconv
        # flow_coupling: Affine3shift # better than affine
        if isinstance(after_splitOff_flowStep, int): 
            after_splitOff_flowStep = [after_splitOff_flowStep] * (self.L + 1)

        # construct conditions # TBD
        self.feat_convs = nn.ModuleList()
        for i in range(self.K[0]-after_splitOff_flowStep[0]):
            feat_conv = nn.Sequential(
                BasicBlockEncoder(feat_channels[i], feat_channels[i+1]),
                nn.BatchNorm3d(feat_channels[i+1]),
                nn.LeakyReLU(0.1, inplace=True)
            )
            self.feat_convs.append(feat_conv)

        # construct flow
        self.layers = nn.ModuleList()
        for level in range(self.L):
            # coupling layers
            if factor > 0:
                # 1. Squeeze
                # re-organize N*N points into image-like NxN points with 3 channels
                self.layers.append(Basic.SqueezeLayer(factor)) # may need a better way for squeezing
                self.C = self.C * (factor**3) #, D // 2, H // 2, W // 2
            
            # main FlowSteps (conditional flow)
            for k in range(self.K[level]-after_splitOff_flowStep[level]):
                self.layers.append(FlowStep(in_channels=self.C, cond_channels=cond_channels[k],
                                            flow_actNorm = flow_actNorm,
                                            flow_permutation=flow_permutation,
                                            flow_coupling=flow_coupling,
                                            ks_s_pad=ks_s_pad))
            if enable_splitOff:
                self.layers.append(Basic.Split(num_channels_split=self.C//2, level=level)) # self.C // 2
                self.level0_condFlow = ConditionalFlow(num_channels=self.C,
                                            num_channels_split=self.C//2,
                                            n_flow_step=after_splitOff_flowStep[level],
                                            num_levels_condition=0, SR=True, RRDB_nb=RRDB_nb, RRDB_nf=RRDB_nf,
                                            flow_actNorm = flow_actNorm,
                                            flow_permutation=flow_permutation,
                                            flow_coupling=flow_coupling, 
                                            kernel_size=ks_s_pad[0], stride=ks_s_pad[1], padding=ks_s_pad[2])
                                            # num_levels_condition=1, SR=True)

    def feat_extract(self, x): # extract conditional features from x low
        for i in range(len(self.feat_convs)):
            x = self.feat_convs[i](x)
            conds.append(x)
        return conds

    def forward(self, hr=None, z=None, eps_std=None, reverse=False, training=True, z1=None, z2=None, x_low=None):
        if hr is not None:
            B = hr.shape[0]
            device = hr.device
        elif z is not None:
            B = z.shape[0]
            device = z.device
        
        logdet = torch.zeros((B,), device=device)
        conds = None
        if x_low is not None:
            conds = self.feat_extract(x_low)
        if not reverse:
            return self.normal_flow(hr, u=conds, logdet=logdet, training=training)
        else:
            return self.reverse_flow(z, u=conds, logdet=logdet, eps_std=eps_std, training=training, z1=z1, z2=z2)

    '''
    hr->y1+z1->y2+z2
    '''
    def normal_flow(self, z, u=None, logdet=None, training=True):
        # for layer, shape in zip(self.layers, self.output_shapes): 
        idx_cond = 0
        for layer in self.layers:
            if isinstance(layer, FlowStep):
                import pdb
                pdb.set_trace()
                c = u[idx_cond] if idx_cond<len(u) else None
                # c = F.interpolate(c, size=z.shape[-1], mode='nearest')
                idx_cond += 1
                z, logdet = layer(z, c, logdet=logdet, reverse=False)
            elif isinstance(layer, Basic.SqueezeLayer):
                z, logdet = layer(z, logdet=logdet, reverse=False)
            elif isinstance(layer, Basic.Split):
                z, a1 = layer(z, reverse=False)
                y1 = z.clone()
                z1, logdet, _ = self.level0_condFlow(a1, z, logdet=logdet, reverse=False, training=training) 
                z2 = None
            # print('encoder', idx_cond, z.shape)
        return z, logdet, z1, z2

    '''
    y2+z2->y1+z1->hr
    '''
    def reverse_flow(self, z, u=None, logdet=None, eps_std=None, training=True, z1=None, z2=None):
        # for layer, shape in zip(reversed(self.layers), reversed(self.output_shapes)):
        idx_cond = 1
        for layer in reversed(self.layers): 
            if isinstance(layer, FlowStep):
                import pdb
                pdb.set_trace()
                c = u[-idx_cond] if idx_cond<=len(u) else None
                # c = F.interpolate(c, size=z.shape[-1], mode='nearest')
                idx_cond += 1
                z, logdet = layer(z, c, logdet=logdet, reverse=True)
            elif isinstance(layer, Basic.SqueezeLayer):
                z, logdet = layer(z, logdet=logdet, reverse=True)
            elif isinstance(layer, Basic.Split):
                a1, logdet, _ = self.level0_condFlow(z1, z, eps_std=eps_std, logdet=logdet, reverse=True, training=training)
                z = layer(z, a1, reverse=True)
        return z
        
    def sample(self, z, logdet=0, eps_std=0.9, training=True, x_low=None):
        if x_low is not None:
            conds = self.feat_extract(x_low)
        return self.reverse_flow(z, u=conds, logdet=logdet, eps_std=eps_std, training=training, z1=None, z2=None)


    
class FlowNet_FlowAssembly(nn.Module):
    def __init__(self, C=1, K=8, L=1, splitOff=[4,4,4], RRDB_nb=[3, 3], RRDB_nf=16, preK=3, factor=4):
        super(FlowNet_FlowAssembly, self).__init__()
        self.C = C
        self.L = L #opt_get(opt, ['network_G', 'flowDownsampler', 'L'])
        self.K = K #opt_get(opt, ['network_G', 'flowDownsampler', 'K']) 8
        self.preK = preK
        if isinstance(self.K, int): 
            self.K = [self.K] * (self.L + 1)

        # flow_actNorm='actNorm1d'
        flow_actNorm="none"
        flow_permutation = 'invconv1d' # opt_get(self.opt, ['network_G', 'flowDownsampler', 'flow_permutation'], 'invconv')
        flow_coupling = 'Affine1d'
        # flow_coupling = 'Affine1d_gf'
        print(flow_actNorm, flow_permutation, flow_coupling)
        
        cond_channels = None # opt_get(self.opt, ['network_G', 'flowDownsampler', 'cond_channels'], None)
        # cond_channels_up = [self.C]*self.preK + [self.C * factor] * (self.K[0] - splitOff[0])
        enable_splitOff = True # opt_get(opt, ['network_G', 'flowDownsampler', 'splitOff', 'enable'], False)
        after_splitOff_flowStep = splitOff #opt_get(opt, ['network_G', 'flowDownsampler', 'splitOff', 'after_flowstep'], 0) [5, 5, 5]
        if isinstance(after_splitOff_flowStep, int): 
            after_splitOff_flowStep = [after_splitOff_flowStep] * (self.L + 1)

        # construct flow
        self.layers = nn.ModuleList()
        # self.output_shapes = []
        # print('after_splitOff_flowStep', after_splitOff_flowStep)
        for level in range(self.L): # single level
            for k in range(self.preK):
                self.layers.append(FlowAssembly(in_channels=self.C, cond_channels=cond_channels,
                                                flow_actNorm=flow_actNorm,
                                                l_id=k))
            if factor > 0:
                # 1. Squeeze
                self.layers.append(Basic.SqueezeLayer(factor))
                self.C = self.C * (factor**3) # D // 2, H // 2, W // 2
                
        #     # self.output_shapes.append([-1, self.C, H, W])
        #     # 2. main FlowSteps (unconditional flow)
            for k in range(self.K[level]-after_splitOff_flowStep[level]):
                self.layers.append(FlowAssembly(in_channels=self.C, cond_channels=cond_channels,
                                                flow_actNorm=flow_actNorm,
                                                l_id=k+self.preK))

            # 3. additional FlowSteps (split + conditional flow)
            if enable_splitOff:
                if level == 0:
                    # self.layers.append(Basic.Split(num_channels_split=self.C // 2 if level < self.L-1 else 3, level=level))
                    self.layers.append(Basic.Split(num_channels_split=self.C//2, level=level)) # self.C // 2
                    self.level0_condFlow = ConditionalFlow(num_channels=self.C,
                                                    num_channels_split=self.C//2,
                                                    n_flow_step=after_splitOff_flowStep[level],
                                                    num_levels_condition=0, SR=True, RRDB_nb=RRDB_nb, RRDB_nf=RRDB_nf,
                                                    flow_actNorm = flow_actNorm,
                                                    flow_permutation=flow_permutation,
                                                    flow_coupling=flow_coupling)
                                                    # num_levels_condition=1, SR=True)
                elif level == 1:
                    self.layers.append(Basic.Split(num_channels_split=self.C//2, level=level))
                    self.level1_condFlow = ConditionalFlow(num_channels=self.C,
                                                    num_channels_split=self.C//2,
                                                    n_flow_step=after_splitOff_flowStep[level],
                                                    num_levels_condition=0, SR=True, RRDB_nb=RRDB_nb, RRDB_nf=RRDB_nf,
                                                    flow_actNorm = flow_actNorm,
                                                    flow_permutation=flow_permutation,
                                                    flow_coupling=flow_coupling)
                self.C = self.C // 2 # if level < self.L-1 else 3


        self.H = H
        self.W = W
        # self.scaleH = data_shape[0] / H
        # self.scaleW = data_shape[1] / W

    def forward(self, hr=None, z=None, u=None, eps_std=None, reverse=False, training=True, z1=None, z2=None):
        if hr is not None:
            B = hr.shape[0]
            device = hr.device
        elif z is not None:
            B = z.shape[0]
            device = z.device
        
        logdet = torch.zeros((B,), device=device)
        if not reverse:
            return self.normal_flow(hr, u=u, logdet=logdet, training=training)
        else:
            return self.reverse_flow(z, u=u, logdet=logdet, eps_std=eps_std, training=training, z1=z1, z2=z2)

    '''
    hr->y1+z1->y2+z2
    '''
    def normal_flow(self, z, u=None, logdet=None, training=True):
        # for layer, shape in zip(self.layers, self.output_shapes): 
        for layer in self.layers:
            # print('normal', layer, z.shape)
            if isinstance(layer, FlowAssembly):
                z, logdet = layer(z, u, logdet=logdet, reverse=False)
            elif isinstance(layer, Basic.SqueezeLayer):
                z, logdet = layer(z, logdet=logdet, reverse=False)
            elif isinstance(layer, Basic.Split):
                z, a1 = layer(z, reverse=False)
                y1 = z.clone()
                z1, logdet, _ = self.level0_condFlow(a1, z, logdet=logdet, reverse=False, training=training) 
                z2 = None
                # if layer.level == 0:
                #     z, a1 = layer(z, reverse=False)
                #     y1 = z.clone()
                #     print('z, a1', z.shape, a1.shape)
                # elif layer.level == 1:
                #     z, a2 = layer(z, reverse=False)  # z --> y2  # y2
                #     print('z, a2', z.shape, a2.shape)
                #     z2, logdet, conditional_feature2 = self.level1_condFlow(a2, z, logdet=logdet, reverse=False, training=training) # z2
                #     conditional_feature1 = torch.cat([y1, F.interpolate(conditional_feature2, scale_factor=2, mode='nearest')],1)
                #     z1, logdet, _ = self.level0_condFlow(a1, conditional_feature1, logdet=logdet, reverse=False, training=training)
        return z, logdet, z1, z2

    '''
    y2+z2->y1+z1->hr
    '''
    def reverse_flow(self, z, u=None, logdet=None, eps_std=None, training=True, z1=None, z2=None):
        # for layer, shape in zip(reversed(self.layers), reversed(self.output_shapes)):
        for layer in reversed(self.layers): 
            if isinstance(layer, FlowAssembly):
                z, logdet = layer(z, u, logdet=logdet, reverse=True)
            elif isinstance(layer, Basic.SqueezeLayer):
                z, logdet = layer(z, logdet=logdet, reverse=True)
            elif isinstance(layer, Basic.Split):
                a1, logdet, _ = self.level0_condFlow(z1, z, eps_std=eps_std, logdet=logdet, reverse=True, training=training)
                z = layer(z, a1, reverse=True)
                # if layer.level == 1:
                #     a2, logdet, conditional_feature2 = self.level1_condFlow(z2, z, eps_std=eps_std, logdet=logdet, reverse=True, training=training)
                #     z = layer(z, a2, reverse=True)
                # elif layer.level == 0:
                #     conditional_feature1 = torch.cat([z, F.interpolate(conditional_feature2, scale_factor=2, mode='nearest')],1)
                #     a1, logdet, _ = self.level0_condFlow(z1, conditional_feature1, eps_std=eps_std, logdet=logdet, reverse=True, training=training)
                #     z = layer(z, a1, reverse=True)
        return z
        
    def sample(self, z, u=None, logdet=0, eps_std=0.9, training=True):
        return self.reverse_flow(z, u=u, logdet=logdet, eps_std=eps_std, training=training, z1=None, z2=None)
   


# encoder for low res
class Encoder_y2(nn.Module):
    def __init__(self, inchannel=1, outchannel=4, hidden_channel=64):
        super(Encoder_y2, self).__init__()
        
        self.conv3d_0 = nn.Conv3d(inchannel, hidden_channel, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.gdn_0 = GDN3d(channels=hidden_channel)
        self.conv3d_1 = nn.Conv3d(hidden_channel, hidden_channel*2, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.gdn_1 = GDN3d(channels=hidden_channel*2)
        self.conv3d_2 = nn.Conv3d(hidden_channel*2, hidden_channel, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.gdn_2 = GDN3d(channels=hidden_channel)
        self.conv3d_3 = nn.Conv3d(hidden_channel, outchannel, kernel_size=3, stride=1, padding=1, padding_mode='replicate')

    def encoder(self, x):
        x = self.conv3d_0(x)
        x = self.gdn_0(x)
        x = self.conv3d_1(x)
        x = self.gdn_1(x)
        x = self.conv3d_2(x)
        x = self.gdn_2(x)
        x = self.conv3d_3(x)
        return x

    def forward(self, x):
        z = self.encoder(x)
        return z


class Encoder_y2_V2(nn.Module):
    def __init__(self, inchannel=1, outchannel=4, hidden_channel=64):
        super(Encoder_y2_V2, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv3d(inchannel, hidden_channel//4, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(hidden_channel//4, hidden_channel//2, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(hidden_channel//2, hidden_channel, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(hidden_channel, hidden_channel//2, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(hidden_channel//2, hidden_channel//4, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(hidden_channel//4, outchannel, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, x):
        z = self.encoder(x)
        return z


class Encoder_y2_V3_resnet(nn.Module):
    def __init__(self, inchannel=1, outchannel=4, hidden_channel=64, upsample=True):
        super(Encoder_y2_V3_resnet, self).__init__()
        # self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm3d(mid_planes)
        # self.conv2 = nn.Conv3d(mid_planes, mid_planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm3d(mid_planes)
        # self.conv3 = nn.Conv3d(mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        # self.relu = nn.ReLU(inplace=True)
        # self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)
        self.encoder = nn.Sequential(
            BasicBlockEncoder(inchannel, hidden_channel * 8),
            BasicBlockEncoder(hidden_channel * 8, hidden_channel * 4),
            BasicBlockEncoder(hidden_channel * 4, hidden_channel * 2),
            BasicBlockEncoder(hidden_channel * 2, hidden_channel),
            nn.BatchNorm3d(hidden_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(hidden_channel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x



class Encoder_y2_V4_resnet(nn.Module):
    def __init__(self, inchannel=1, outchannel=4, hidden_channel=64, upsample=True):
        super(Encoder_y2_V4_resnet, self).__init__()
        # self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm3d(mid_planes)
        # self.conv2 = nn.Conv3d(mid_planes, mid_planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm3d(mid_planes)
        # self.conv3 = nn.Conv3d(mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        # self.relu = nn.ReLU(inplace=True)
        # self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)
        self.encoder = nn.Sequential(
            BasicBlockEncoder(inchannel, hidden_channel * 4),
            BasicBlockEncoder(hidden_channel * 4, hidden_channel * 2),
            BasicBlockEncoder(hidden_channel * 2, hidden_channel),
            nn.BatchNorm3d(hidden_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(hidden_channel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x

