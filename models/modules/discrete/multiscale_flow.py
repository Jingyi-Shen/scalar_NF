import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.autograd import Variable

from typing import List
# from pytorch3d.ops import knn_gather, knn_points
# from pytorch3d.ops import sample_farthest_points


from models.modules.flows.coupling import AffineCouplingLayer, AffineInjectorLayer
from models.modules.flows.coupling import AffineSpatialCouplingLayer
from models.modules.flows.normalize import ActNorm
from models.modules.flows.permutate import Permutation
from models.modules.flows.linear import SoftClampling

from models.modules.utils.probs import GaussianDistribution

from models.modules.utils.template import get_template

import scipy
import numpy as np

import itertools

# -------------------------------------------------------------------------------------------------------------------------------
class LinearA1D(nn.Module):
    def __init__(self, dim_in: int, dim_h: int, dim_out: int, dim_c=None, device='cuda'):
        super(LinearA1D, self).__init__()
        linear_zero = nn.Linear(dim_h, dim_out, bias=True)
        linear_zero.weight.data.zero_()
        linear_zero.bias.data.zero_()

        in_channel = dim_in if dim_c is None else dim_in + dim_c
        self.in_channel = in_channel
        # self.dim_in = dim_in
        # self.dim_c = dim_c
        
        self.layers = nn.Sequential(
            nn.Linear(in_channel, dim_h, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim_h, dim_h, bias=True),
            nn.LeakyReLU(inplace=True),
            linear_zero).to(device)

    def forward(self, h: Tensor, c: Tensor=None):
        if c is not None:
            h = torch.cat([h, c], dim=-1)
        h = self.layers(h)
        return h


class LinearUnit(nn.Module):
    def __init__(self, dim_in: int, dim_h: int, dim_out: int, dim_c=None, n_block=3, batch_norm=False, device='cuda'):
        super(LinearUnit, self).__init__()
        
        in_channel = dim_in if dim_c is None else dim_in + dim_c
        
        layers = []
        for _ in range(n_block):
            linear   = nn.Conv1d(in_channel, dim_h, kernel_size=1)
            activate = nn.ReLU(inplace=True)
            # activate = nn.LeakyReLU(inplace=True)
            
            if batch_norm: # if batch_norm is None:
                batchnorm = nn.BatchNorm1d(dim_h)
                layers.extend([linear, activate, batchnorm])
            else:
                layers.extend([linear, activate])
            # batchnorm = nn.BatchNorm1d(dim_h)
            
            in_channel = dim_h

        out_conv = nn.Conv1d(dim_h, dim_out, kernel_size=1)
        nn.init.normal_(out_conv.weight, mean=0.0, std=0.05)
        nn.init.zeros_(out_conv.bias)
        layers.append(out_conv)

        self.linears = nn.Sequential(*layers).to(device)

    def forward(self, h: Tensor, c: Tensor=None):
        """
        h: [B, N, C]
        """
        if c is not None:
            h = torch.cat([h, c], dim=-1)
        h = torch.transpose(h, 1, 2)
        h = self.linears(h)
        h = torch.transpose(h, 1, 2)
        return h


# -------------------------------------------------------------------------------------------------------------------------------
class FlowBlock(nn.Module):

    def __init__(self, idim, hdim, cdim, is_even, device='cuda'):
        super(FlowBlock, self).__init__()

        self.actnorm    = ActNorm(idim, dim=2, device=device)
        self.permutate1 = Permutation('inv1x1', idim, dim=2, device=device)
        self.permutate2 = Permutation('reverse', idim, dim=2, device=device)
        
        self.idim = idim # 7
        self.cdim = cdim # 8
        # 7/2 + 8 = 11
        
        channel1 = idim - idim // 2 #4
        channel2 = idim // 2 #3
        # if idim == 3:
        #     tdim = 1 if is_even else 2
        #     self.coupling1 = AffineSpatialCouplingLayer('additive', LinearA1D, split_dim=2, is_even=is_even, clamp=None,
        #         params={ 'dim_in' : tdim, 'dim_h': hdim, 'dim_out': idim - tdim, 'dim_c': cdim })
        # else:
        #     self.coupling1 = AffineCouplingLayer('additive', LinearA1D, split_dim=2, clamp=None,
        #         params={ 'dim_in' : idim // 2, 'dim_h': hdim, 'dim_out': idim - idim // 2, 'dim_c': cdim })
        
        if idim < 5:  # if id < 3 or id % 3 == 0:
        # last dimension as channel
            self.coupling1 = AffineCouplingLayer('affine', LinearUnit, split_dim=2, clamp=SoftClampling(),
                params={ 'dim_in': channel1, 'dim_h': hdim, 'dim_out': channel2, 'dim_c': cdim, 'device': device})
            self.coupling2 = AffineCouplingLayer('affine', LinearUnit, split_dim=2, clamp=SoftClampling(),
                params={ 'dim_in': channel1, 'dim_h': hdim, 'dim_out': channel2, 'dim_c': cdim, 'device': device})
        else:
            self.coupling1 = AffineCouplingLayer('affine', LinearUnit, split_dim=2, clamp=SoftClampling(),
                params={ 'dim_in': channel1, 'dim_h': hdim, 'dim_out': channel2, 'dim_c': cdim, 'batch_norm': True, 'device': device})
            self.coupling2 = AffineCouplingLayer('affine', LinearUnit, split_dim=2, clamp=SoftClampling(),
                params={ 'dim_in': channel1, 'dim_h': hdim, 'dim_out': channel2, 'dim_c': cdim, 'batch_norm': True, 'device': device})
        
    def forward(self, x: Tensor, c: Tensor=None):
        x, _log_det0 = self.actnorm(x)
        x, _log_det1 = self.permutate1(x, c)
        x, _log_det2 = self.coupling1(x, c)
        x, _         = self.permutate2(x, c)
        x, _log_det4 = self.coupling2(x, c)
        return x, _log_det0 + _log_det1 + _log_det2 + _log_det4

    def inverse(self, z: Tensor, c: Tensor=None):
        z, _log_det4 = self.coupling2.inverse(z, c)
        z, _         = self.permutate2.inverse(z, c)
        z, _log_det2 = self.coupling1.inverse(z, c)
        z, _log_det1 = self.permutate1.inverse(z, c)
        z, _log_det0 = self.actnorm.inverse(z)
        return z, _log_det0 + _log_det1 + _log_det2 + _log_det4


# -------------------------------------------------------------------------------------------------------------------------------
class Multi_Scale_FlowBlock(nn.Module):

    def __init__(self, idim, hdim, cdim, is_even):
        super(Multi_Scale_FlowBlock, self).__init__()

        self.actnorm    = ActNorm(idim, dim=2)
        self.permutate1 = Permutation('inv1x1', idim, dim=2)
        self.permutate2 = Permutation('reverse', idim, dim=2)
        
        self.idim = idim # 7
        self.cdim = cdim # 8
        # 7/2 + 8 = 11
        
        if idim == 3:
            tdim = 1 if is_even else 2
            self.coupling1 = AffineSpatialCouplingLayer('additive', LinearA1D, split_dim=2, is_even=is_even, clamp=None,
                params={ 'dim_in' : tdim, 'dim_h': hdim, 'dim_out': idim - tdim, 'dim_c': cdim })
        else:
            self.coupling1 = AffineCouplingLayer('additive', LinearA1D, split_dim=2, clamp=None,
                params={ 'dim_in' : idim // 2, 'dim_h': hdim, 'dim_out': idim - idim // 2, 'dim_c': cdim })
        
        if self.cdim is not None:
            self.coupling2 = AffineInjectorLayer('affine', LinearA1D, split_dim=2, clamp=None,
                params={ 'dim_in': cdim, 'dim_h': hdim, 'dim_out': idim, 'dim_c': None })
        else: # new added, without condition
            self.coupling2 = AffineCouplingLayer('affine', LinearA1D, split_dim=2, clamp=None,
                params={ 'dim_in' : idim // 2, 'dim_h': hdim, 'dim_out': idim - idim // 2, 'dim_c': cdim })
        
    def forward(self, x: Tensor, c: Tensor=None):
        x, _log_det0 = self.actnorm(x)
        x, _log_det1 = self.permutate1(x, c)
        x, _         = self.coupling1(x, c)
        x, _         = self.permutate2(x, c)
        x, _log_det4 = self.coupling2(x, c)
        return x, _log_det0 + _log_det1 + _log_det4

    def inverse(self, z: Tensor, c: Tensor=None):
        z, _log_det4 = self.coupling2.inverse(z, c)
        z, _         = self.permutate2.inverse(z, c)
        z, _         = self.coupling1.inverse(z, c)
        z, _log_det1 = self.permutate1.inverse(z, c)
        z, _log_det0 = self.actnorm.inverse(z)
        return z, _log_det0 + _log_det1 + _log_det4


# -----------------------------------------------------------------------------------------
# class DistanceEncoder(nn.Module):

#     def __init__(self, dim_in: int, dim_out: int, k: int):
#         super(DistanceEncoder, self).__init__()

#         self.k = k
#         self.mlp = nn.Sequential(
#             nn.Conv2d(dim_in * 3 + 1, 64, [1, 1]),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(64, 64, [1, 1]),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(64, dim_out, [1, 1]))

#     def distance_vec(self, xyz: Tensor):
#         B, N, C = xyz.shape
#         idxb = torch.arange(B).view(-1, 1)

#         _, idx_knn, _ = knn_points(xyz, xyz, K=self.k, return_nn=False, return_sorted=False)  # [B, N, k]
#         idx_knn = idx_knn.view(B, -1) # [B, N * k]
#         neighbors = xyz[idxb, idx_knn].view(B, N, self.k, C)  # [B, N, k, C]

#         # pt_raw = torch.unsqueeze(xyz, dim=2).repeat(1, 1, self.k, 1)  # [B, N, k, C]
#         pt_raw = torch.unsqueeze(xyz, dim=2).expand_as(neighbors)  # [B, N, k, C]
#         neighbor_vector = pt_raw - neighbors   # [B, N, k, C]
#         distance = torch.sqrt(torch.sum(neighbor_vector ** 2, dim=-1, keepdim=True))

#         f_distance = torch.cat([pt_raw, neighbors, neighbor_vector, distance], dim=-1)
#         f_distance = f_distance.permute(0, 3, 1, 2)
#         return f_distance, idx_knn  # [B, C, N, k]

#     def forward(self, xyz: Tensor):
#         f_dist, idx_knn = self.distance_vec(xyz)  # [B, C, N, k]
#         f = self.mlp(f_dist)
#         return f, idx_knn  # [B, C_out, N, k]


# -----------------------------------------------------------------------------------------
# class WeightEstimationUnit(nn.Module):

#     def __init__(self, feat_dim: int):
#         super(WeightEstimationUnit, self).__init__()    
        
#         self.r_max = 32

#         self.mlp = nn.Sequential(
#             nn.Conv2d(feat_dim, 128, kernel_size=[1, 1]),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(128, 64, kernel_size=[1, 1]),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(64, self.r_max, kernel_size=[1, 1]))

#     def forward(self, context: Tensor):
#         """
#         context: [B, C, N, K]
#         return: [B, R, N, K]
#         """
#         f = self.mlp(context)         # [B, R, N, K]
#         return f.permute(0, 2, 1, 3)  # [B, N, R, K]


# -----------------------------------------------------------------------------------------
# class InterpolationModule(nn.Module):

#     def __init__(self, input_dim, k: int):
#         super(InterpolationModule, self).__init__()
#         k = 8
#         self.k = k
#         self.knn_context = KnnContextEncoder(input_dim, k)
#         self.weight_unit = WeightEstimationUnit(feat_dim=256)

#     def forward(self, z: Tensor, xyz: Tensor, upratio: int):
#         (B, N, C), device = z.shape, z.device
#         idxb1 = torch.arange(B, device=device).view(-1, 1)

#         # Learn interpolation weight for each point
#         context, idx_knn = self.knn_context(xyz)  # [B, C, N, k], [B, N * k]
#         weights = self.weight_unit(context)       # [B, N, upratio, k]
#         weights = F.softmax(weights[:, :, :upratio], dim=-1)

#         # Interpolation
#         nei_prior = z[idxb1, idx_knn].view(B, N, self.k, C)  # [B, N, k, C]
#         nei_prior = nei_prior.permute(0, 1, 3, 2)            # [B, N, C, k]
#         intep_prior = torch.einsum('bnck, bnrk->bncr', nei_prior, weights)
#         return intep_prior


# -----------------------------------------------------------------------------------------
class FeatureExtractVAEModule(nn.Module):

    def __init__(self, inchannel: int, latchannel: int, hchannel: int, ddim: int):
        super(FeatureExtractVAEModule, self).__init__()

        # self.num_conv = (odim // growth_width)
        self.inchannel = inchannel
        self.latchannel = latchannel
        self.ddim = ddim
        
        """
        # self.convs = nn.ModuleList()
        self.conv_1 = nn.Sequential(*[
            nn.Conv1d(inchannel, hchannel, kernel_size=1), # 3, 1
            nn.BatchNorm1d(hchannel),
            nn.LeakyReLU(0.1, inplace=True),
        ])
        # self.convs.append(conv_1)
        
        self.conv_2 = nn.Sequential(*[
            nn.Conv1d(hchannel, hchannel*2, kernel_size=1), # N/2, attr
            nn.BatchNorm1d(hchannel*2),
            nn.LeakyReLU(0.1, inplace=True),
        ])
        # self.convs.append(conv_2)
        
        self.conv_3 = nn.Sequential(*[
            nn.Conv1d(hchannel*2, hchannel, kernel_size=1), # N/4, attr
            nn.BatchNorm1d(hchannel),
            nn.LeakyReLU(0.1, inplace=True),
        ])
        # self.convs.append(conv_3)
        
        self.conv_4 = nn.Sequential(*[
            nn.Conv1d(hchannel, latchannel, kernel_size=1), # N/4, attr
            nn.BatchNorm1d(latchannel),
            nn.LeakyReLU(0.1, inplace=True),
        ])
        # self.convs.append(conv_4)
        
        self.enc_mu = nn.Linear(ddim*latchannel, ddim*latchannel)
        self.enc_var = nn.Linear(ddim*latchannel, ddim*latchannel)
        
        self.deconv_1 = nn.Sequential(*[
            nn.ConvTranspose1d(latchannel, hchannel, kernel_size=1), # 3, 1
            nn.BatchNorm1d(hchannel),
            nn.LeakyReLU(0.1, inplace=True),
        ])
        
        self.deconv_2 = nn.Sequential(*[
            nn.ConvTranspose1d(hchannel, hchannel*2, kernel_size=1), # N/2, attr
            nn.BatchNorm1d(hchannel*2),
            nn.LeakyReLU(0.1, inplace=True),
        ])
        
        self.deconv_3 = nn.Sequential(*[
            nn.ConvTranspose1d(hchannel*2, hchannel, kernel_size=1), # N/4, attr
            nn.BatchNorm1d(hchannel),
            nn.LeakyReLU(0.1, inplace=True),
        ])
        
        self.deconv_4 = nn.Sequential(*[
            nn.ConvTranspose1d(hchannel, inchannel, kernel_size=1), # N/4, attr
            nn.BatchNorm1d(inchannel),
            nn.LeakyReLU(0.1, inplace=True),
        ])
        """
        # for i in range(self.num_conv - 1):
        #     in_channel = growth_width * (i + 1) + idim
        #     conv = nn.Sequential(*[
        #         nn.Conv2d(in_channel, growth_width, kernel_size=[1, 1], bias=True),
        #         nn.BatchNorm2d(growth_width),
        #         nn.LeakyReLU(0.05, inplace=True),
        #     ])
        #     self.convs.append(conv)
        # self.conv_out = nn.Conv2d(, odim, kernel_size=[1, 1], bias=True)
        
        """
        # self.conv_1 = nn.Sequential(*[
        #     nn.Conv2d(inchannel, hchannel, 1),
        #     nn.BatchNorm2d(hchannel),
        #     nn.LeakyReLU(0.1, inplace=True),
        # ])
        
        # self.conv_2 = nn.Sequential(*[
        #     nn.Conv2d(hchannel, hchannel*2, 1),
        #     nn.BatchNorm2d(hchannel*2),
        #     nn.LeakyReLU(0.1, inplace=True),
        # ])
        
        # self.conv_3 = nn.Sequential(*[
        #     nn.Conv2d(hchannel*2, hchannel, 1),
        #     nn.BatchNorm2d(hchannel),
        #     nn.LeakyReLU(0.1, inplace=True),
        # ])
        
        # self.conv_4 = nn.Sequential(*[ # latchannel
        #     nn.Conv2d(hchannel, hchannel//2, 1),
        #     nn.BatchNorm2d(hchannel//2), # latchannel
        #     nn.LeakyReLU(0.1, inplace=True),
        # ])
        
        # self.enc_mu = nn.Sequential(*[
        #     nn.Conv2d(latchannel, latchannel, 1),
        #     nn.BatchNorm2d(latchannel),
        #     nn.LeakyReLU(0.1, inplace=True),
        # ])
        
        # self.enc_var = nn.Sequential(*[
        #     nn.Conv2d(latchannel, latchannel, 1),
        #     nn.BatchNorm2d(latchannel),
        #     nn.LeakyReLU(0.1, inplace=True),
        # ])
        
        # self.deconv_1 = nn.Sequential(*[
        #     nn.ConvTranspose2d(latchannel, hchannel, kernel_size=1), 
        #     nn.BatchNorm2d(hchannel),
        #     nn.LeakyReLU(0.1, inplace=True),
        # ])
        
        # self.deconv_2 = nn.Sequential(*[
        #     nn.ConvTranspose2d(hchannel, hchannel*2, kernel_size=1),
        #     nn.BatchNorm2d(hchannel*2),
        #     nn.LeakyReLU(0.1, inplace=True),
        # ])
        
        # self.deconv_3 = nn.Sequential(*[
        #     nn.ConvTranspose2d(hchannel*2, hchannel, kernel_size=1),
        #     nn.BatchNorm2d(hchannel),
        #     nn.LeakyReLU(0.1, inplace=True),
        # ])
        
        # self.deconv_4 = nn.Sequential(*[
        #     nn.ConvTranspose2d(hchannel, inchannel, kernel_size=1),
        #     nn.BatchNorm2d(inchannel),
        #     nn.LeakyReLU(0.1, inplace=True),
        # ])
        """
        self.conv_1 = nn.Sequential(*[
            nn.Conv1d(inchannel, hchannel, 1),
            nn.BatchNorm1d(hchannel),
            nn.LeakyReLU(0.1, inplace=True),
        ])
        
        self.conv_2 = nn.Sequential(*[
            nn.Conv1d(hchannel, hchannel*2, 1),
            nn.BatchNorm1d(hchannel*2),
            nn.LeakyReLU(0.1, inplace=True),
        ])
        
        self.conv_3 = nn.Sequential(*[
            nn.Conv1d(hchannel*2, hchannel, 1),
            nn.BatchNorm1d(hchannel),
            nn.LeakyReLU(0.1, inplace=True),
        ])
        
        self.conv_4 = nn.Sequential(*[
            nn.Conv1d(hchannel, latchannel, 1),
            nn.BatchNorm1d(latchannel),
            nn.LeakyReLU(0.1, inplace=True),
        ])
        
        self.enc_mu = nn.Sequential(*[
            nn.Conv1d(hchannel//2, latchannel, 1),
            nn.BatchNorm1d(latchannel),
            nn.LeakyReLU(0.1, inplace=True),
        ])
        
        self.enc_var = nn.Sequential(*[
            nn.Conv1d(hchannel//2, latchannel, 1),
            nn.BatchNorm1d(latchannel),
            nn.LeakyReLU(0.1, inplace=True),
        ])
        
        self.deconv_1 = nn.Sequential(*[
            nn.ConvTranspose1d(latchannel, hchannel, kernel_size=1), 
            nn.BatchNorm1d(hchannel),
            nn.LeakyReLU(0.1, inplace=True),
        ])
        
        self.deconv_2 = nn.Sequential(*[
            nn.ConvTranspose1d(hchannel, hchannel*2, kernel_size=1),
            nn.BatchNorm1d(hchannel*2),
            nn.LeakyReLU(0.1, inplace=True),
        ])
        
        self.deconv_3 = nn.Sequential(*[
            nn.ConvTranspose1d(hchannel*2, hchannel, kernel_size=1),
            nn.BatchNorm1d(hchannel),
            nn.LeakyReLU(0.1, inplace=True),
        ])
        
        self.deconv_4 = nn.Sequential(*[
            nn.ConvTranspose1d(hchannel, inchannel, kernel_size=1),
            nn.BatchNorm1d(inchannel),
            nn.LeakyReLU(0.1, inplace=True),
        ])
        
        # self.encoder = FoldNet_Encoder(feat_dims=latchannel, k=16)
        # self.decoder = FoldNet_Decoder(feat_dims=latchannel, shape='plane', out_dims=7) 
        
    # def reparameterize(self, mu, logvar):
    def reparameterize(self, mu, sigma):
        # std = torch.exp(0.5*logvar) #logvar.mul(0.5).exp_()
        # eps = torch.randn_like(std)
        # return mu + eps*std
        z = mu + sigma * torch.randn_like(mu, device=sigma.device)
        z = Variable(z)
        return z
    
    def encode(self, x: Tensor):
        ##x: [B, 16, 7] or x: [B, 16, 7, 1]
        # import pdb
        # pdb.set_trace()
        num_points = x.shape[1]
        x = x.transpose(2, 1) #x: [B, 7, 16]
        x = self.conv_1(x)
        x = self.conv_2(x)    #x: [B, h, 16]
        # local_features = x #x: [B, 16, h]
        x = self.conv_3(x)
        x = self.conv_4(x)    #x: [B, h*4, 16]
        # print(x.shape)
        # exit()
        return x
        # x = x.transpose(2, 1)
        # x = torch.max(x, 1, keepdim=True)[0]
        # x = nn.MaxPool1d(num_points)(x) #x: [B, h*4, 1]
        # x = x.transpose(2, 1)
        # mu, logvar = self.enc_mu(x), self.enc_var(x)
        # print(mu.shape, logvar.shape)
        # return x, local_features, num_points
        # return mu, logvar
    
    def decode(self, z: Tensor, num_points=32, local_features=None):
        ##x: [B, D, N, 1]
        # z = z.view(z.size(0), self.latchannel, -1) # latdim, k
        # z = z.repeat(1, 1, num_points)  #x: [B, h*4, 16]
        # z = torch.cat([z, local_features], 1)  #x: [B, h*4+h, 16]
        x = self.deconv_1(z)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.deconv_4(x)
        x = x.transpose(2, 1)
        return x
    
    def encode_z(self, x: Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) # 512, 4, 7
        # z = z.squeeze(-1)
        return z
        
    def forward(self, x):
        """
        Input: input particle data, [B, C, N]  N: particle attributes
        Output: 
            VAE encoded, [B, D, N]
            decoded data, [B, C, N]
        """
        # z, local_features, num_points = self.encode(x)
        # x_hat = self.decode(z, num_points, local_features)
        
        z  = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
        
        # mu, logvar = self.encode(x)
        # z = self.reparameterize(mu, logvar) 
        # x_hat = self.decode(z)
        # return x_hat, z, mu, logvar
        
        # x = x.unsqueeze(-1) # [B, C, N, 1]
        # mu, logvar = self.encode(x) # [B, D, N, 1]
        # z = self.reparameterize(mu, logvar) # [B, D, N, 1]
        # x_hat = self.decode(z) # [B, C, N, 1]
        # return x_hat, z, mu, logvar
        # return x_hat.squeeze(-1), z.squeeze(-1), mu.squeeze(-1), logvar.squeeze(-1)
        
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # return x_hat, z


######################################################### folding net #########################################################
def local_maxpool(x, idx):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()                      # (batch_size, num_points, num_dims)
    x = x.view(batch_size*num_points, -1)[idx, :]           # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    x = x.view(batch_size, num_points, -1, num_dims)        # (batch_size, num_points, k, num_dims)
    x, _ = torch.max(x, dim=2)                              # (batch_size, num_points, num_dims)

    return x


def local_cov(pts, idx):
    batch_size = pts.size(0)
    num_points = pts.size(2)
    pts = pts.view(batch_size, -1, num_points)              # (batch_size, 3, num_points)
 
    _, num_dims, _ = pts.size()

    x = pts.transpose(2, 1).contiguous()                    # (batch_size, num_points, 3)
    x = x.view(batch_size*num_points, -1)[idx, :]           # (batch_size*num_points*2, 3)
    x = x.view(batch_size, num_points, -1, num_dims)        # (batch_size, num_points, k, 3)

    x = torch.matmul(x[:,:,0].unsqueeze(3), x[:,:,1].unsqueeze(2))  # (batch_size, num_points, 3, 1) * (batch_size, num_points, 1, 3) -> (batch_size, num_points, 3, 3)
    # x = torch.matmul(x[:,:,1:].transpose(3, 2), x[:,:,1:])
    x = x.view(batch_size, num_points, num_dims*num_dims).transpose(2, 1)   # (batch_size, 9, num_points)
    
    x = torch.cat((pts, x), dim=1)                          # (batch_size, num_dims^2+num_dims, num_points)
    return x


def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]            # (batch_size, num_points, k)

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points
    else:
        idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    return idx

class FoldNet_Encoder(nn.Module):
    def __init__(self, feat_dims, k=16):
        super(FoldNet_Encoder, self).__init__()
        # self.n = 2048   # input point cloud size
        self.k = k
        self.mlp1 = nn.Sequential( # 7*7 + 7
            nn.Conv1d(56, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        
        # self.mlp2 = nn.Sequential(
        #     nn.Conv1d(128, 64, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(64, feat_dims, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(feat_dims, feat_dims, 1),
        # )
        
        self.linear1 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.linear2 = nn.Linear(128, 128)
        self.conv2 = nn.Conv1d(128, feat_dims, 1) # 128, 1024, 1
        
        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024, feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dims, feat_dims, 1),
        )

    def graph_layer(self, x, idx):           
        x = local_maxpool(x, idx)    
        x = self.linear1(x)  
        x = x.transpose(2, 1)                                     
        x = F.relu(self.conv1(x))                            
        x = local_maxpool(x, idx)  
        x = self.linear2(x) 
        x = x.transpose(2, 1)                                   
        x = self.conv2(x)                       
        return x

    def forward(self, pts):
        pts = pts.transpose(2, 1)               # (batch_size, 3, num_points)
        idx = knn(pts, k=self.k)
        x = local_cov(pts, idx)                 # (batch_size, 3, num_points) -> (batch_size, 12, num_points])            
        x = self.mlp1(x)                        # (batch_size, 12, num_points) -> (batch_size, 64, num_points])
        x = self.graph_layer(x, idx)            # (batch_size, 64, num_points) -> (batch_size, 1024, num_points)
        # x = torch.max(x, 2, keepdim=True)[0]    # (batch_size, 1024, num_points) -> (batch_size, 1024, 1)
        # x = self.mlp2(x)                        # (batch_size, 1024, 1) -> (batch_size, feat_dims, 1)
        feat = x.transpose(2,1)                 # (batch_size, feat_dims, 1) -> (batch_size, 1, feat_dims)
        return feat           


class FoldNet_Decoder(nn.Module):
    def __init__(self, feat_dims=3, shape='plane', out_dims=7):
        super(FoldNet_Decoder, self).__init__()
        # self.num_points = num_points  ## 45 * 45.
        self.shape = shape
        self.meshgrid = [[-0.3, 0.3, 8], [-0.3, 0.3, 8]]
        # self.meshgrid = [[-0.3, 0.3, 4], [-0.3, 0.3, 4]]
        self.sphere = np.load("./data/pos/sphere.npy")
        self.gaussian = np.load("./data/pos/gaussian.npy")
        if self.shape == 'plane':
            self.folding1 = nn.Sequential(
                nn.Conv1d(feat_dims, feat_dims, 1), # +2
                nn.ReLU(),
                nn.Conv1d(feat_dims, feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(feat_dims, 7, 1),
            )
        else:
            self.folding1 = nn.Sequential(
                nn.Conv1d(feat_dims+3, feat_dims, 1), # +3
                nn.ReLU(),
                nn.Conv1d(feat_dims, feat_dims, 1),
                nn.ReLU(),
                nn.Conv1d(feat_dims, 7, 1),
            )  
        self.folding2 = nn.Sequential(
            nn.Conv1d(feat_dims+7, feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dims, feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dims, out_dims, 1),
        )

    def build_grid(self, batch_size):
        if self.shape == 'plane':
            x = np.linspace(*self.meshgrid[0])
            y = np.linspace(*self.meshgrid[1])
            points = np.array(list(itertools.product(x, y)))
        elif self.shape == 'sphere':
            points = self.sphere
        elif self.shape == 'gaussian':
            points = self.gaussian
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float()

    def forward(self, x, num_points=16):
        # x = x.transpose(1, 2)#.repeat(1, 1, num_points)      # (batch_size, feat_dims, num_points)
        # points = self.build_grid(x.shape[0]).transpose(1, 2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)
        # if x.get_device() != -1:
        #     points = points.cuda(x.get_device())
        # cat1 = torch.cat((x, points), dim=1)            # (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)
        folding_result1 = self.folding1(x)           # (batch_size, 3, num_points)
        cat2 = torch.cat((x, folding_result1), dim=1)   # (batch_size, feat_dims+3, num_points)
        folding_result2 = self.folding2(cat2)           # (batch_size, 3, num_points)
        return folding_result2.transpose(1, 2)          # (batch_size, num_points ,3)


######################################################### folding net #########################################################

# ######################################################### DGCNN #########################################################
# def get_graph_feature(x, k=16, idx=None):
#     batch_size = x.size(0)
#     num_points = x.size(2)
#     x = x.view(batch_size, -1, num_points)      # (batch_size, num_dims=7, num_points)
#     if idx is None:
#         idx = knn(x, k=k)                       # (batch_size, num_points, k)
 
#     _, num_dims, _ = x.size()

#     x = x.transpose(2, 1).contiguous()          # (batch_size, num_points, num_dims=7)
#     feature = x.view(batch_size*num_points, -1)[idx, :]                 # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
#     feature = feature.view(batch_size, num_points, k, num_dims)         # (batch_size, num_points, k, num_dims)
#     x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (batch_size, num_points, k, num_dims)
    
#     feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)      # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)
  
#     return feature                              # (batch_size, 2*num_dims, num_points, k)


# class DGCNN(nn.Module):
#     def __init__(self, in_dim=7, feat_dims=128, k=16):
#         super(DGCNN, self).__init__()
#         self.k = k
        
#         self.bn1 = nn.BatchNorm2d(32)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(16)
#         self.bn4 = nn.BatchNorm2d(16)
#         self.bn5 = nn.BatchNorm1d(feat_dims)

#         self.conv1 = nn.Sequential(nn.Conv2d(in_dim*2, 32, kernel_size=1, bias=False),
#                                   self.bn1,
#                                   nn.LeakyReLU(negative_slope=0.2))
#         self.conv2 = nn.Sequential(nn.Conv2d(32*2, 64, kernel_size=1, bias=False),
#                                   self.bn2,
#                                   nn.LeakyReLU(negative_slope=0.2))
#         self.conv3 = nn.Sequential(nn.Conv2d(64*2, 16, kernel_size=1, bias=False),
#                                   self.bn3,
#                                   nn.LeakyReLU(negative_slope=0.2))
#         self.conv4 = nn.Sequential(nn.Conv2d(16*2, 16, kernel_size=1, bias=False),
#                                   self.bn4,
#                                   nn.LeakyReLU(negative_slope=0.2))
#         self.conv5 = nn.Sequential(nn.Conv1d(32+64+16+16, feat_dims, kernel_size=1, bias=False),
#                                   self.bn5,
#                                   nn.LeakyReLU(negative_slope=0.2))
        
#         self.decoder = FoldNet_Decoder(feat_dims=feat_dims, shape='plane', out_dims=7)
                                   
        
#     def forward(self, x):        
#         # encode
#         x = x.transpose(2, 1)  # [16, 64, 7] --> [16, 7, 64]
#         batch_size = x.size(0)
#         x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
#         x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
#         x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

#         x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
#         x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
#         x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

#         x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
#         x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
#         x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

#         x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
#         x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
#         x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

#         x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 512, num_points)

#         x0 = self.conv5(x)                      # (batch_size, 512, num_points) -> (batch_size, feat_dims, num_points)
#         # x = x0.max(dim=-1, keepdim=False)[0]    # (batch_size, feat_dims, num_points) -> (batch_size, feat_dims)
#         # feat = x.unsqueeze(1)                   # (batch_size, feat_dims) -> (batch_size, 1, feat_dims)
#         # print('latent', x0.shape)
#         x_hat = self.decoder(x0)
#         return x_hat, x0

######################################################### pointnet #########################################################
class STNkd(nn.Module):
    def __init__(self, indim=64, k=7):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(indim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
        

class PointNetfeat(nn.Module):
    def __init__(self, indim=4, latdim=16, n_pts=64, input_transform=False, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.indim = indim
        self.latdim = latdim
        self.input_transform = input_transform
        self.feature_transform = feature_transform
        
        if self.input_transform:
            self.stn = STNkd(indim=n_pts, k=self.indim) # depends on number of points
        self.conv1 = torch.nn.Conv1d(indim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, latdim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(latdim)
        
        if self.feature_transform:
            self.fstn = STNkd(indim=n_pts, k=latdim)
        
        self.conv4 = torch.nn.Conv1d(latdim, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 64, 1)
        self.conv6 = torch.nn.Conv1d(64, indim, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(indim)
        
        self.leakyrelu = nn.LeakyReLU(0.1)

    def encode(self, x):
        n_pts = x.size()[1]
        if self.input_transform:
            trans = self.stn(x)
            x = torch.bmm(x, trans)
        
        x = x.transpose(2, 1) # [B, 64, 4] --> # [B, 4, 64]
        x = self.leakyrelu(self.bn1(self.conv1(x))) # F.relu
        
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None
        
        # pointfeat = x
        x = self.leakyrelu(self.bn2(self.conv2(x)))
        x = self.leakyrelu(self.bn3(self.conv3(x)))
        x = x.transpose(2, 1)
        # x = torch.max(x, 2, keepdim=True)[0]
        # x = x.view(-1, self.latdim)
        # x = x.view(-1, self.latdim, 1).repeat(1, 1, n_pts)
        # x = torch.cat([x, pointfeat], 1) #, trans, trans_feat
        return x
        
    def decode(self, z):
        z = z.transpose(2, 1)
        x = self.leakyrelu(self.bn4(self.conv4(z)))
        x = self.leakyrelu(self.bn5(self.conv5(x)))
        x = self.bn6(self.conv6(x))
        x = x.transpose(2, 1)
        return x
    
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class Mapping2Dto3D(nn.Module):
    """
    Takes batched points as input and run them through an MLP.
    Note : the MLP is implemented as a torch.nn.Conv1d with kernels of size 1 for speed.
    Note : The latent vector is added as a bias after the first layer. This is strictly identical
    as concatenating each input point with the latent vector but saves memory and speeed.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, dim_template, latdim=64, ddim=7, n_pts=128, num_layers=3, hidden_neurons=64, use_grid=True):
        self.latdim = latdim
        self.input_size = dim_template
        self.dim_output = ddim
        self.hidden_neurons = hidden_neurons
        self.num_layers = num_layers
        self.n_pts = n_pts
        self.use_grid = use_grid
        super(Mapping2Dto3D, self).__init__()
        
        # self.conv1 = torch.nn.Conv1d(self.input_size, self.latdim, 1) # 2 --> latdim
        self.conv0 = torch.nn.Conv1d(self.input_size, self.dim_output, 1) # [1, 2, 121] --> [1, 7, 121]
        self.conv1 = torch.nn.Conv1d(self.n_pts, self.n_pts, 1)
        
        self.conv2 = torch.nn.Conv1d(self.latdim, self.hidden_neurons, 1)
        self.conv_list = nn.ModuleList(
            [torch.nn.Conv1d(self.hidden_neurons, self.hidden_neurons, 1) for i in range(self.num_layers)])
        self.last_conv = torch.nn.Conv1d(self.hidden_neurons, self.n_pts, 1)
        
        # self.bn1 = torch.nn.BatchNorm1d(self.latdim)
        self.bn1 = torch.nn.BatchNorm1d(self.n_pts)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_neurons)
        self.bn3 = torch.nn.BatchNorm1d(self.n_pts)
        self.bn_list = nn.ModuleList([torch.nn.BatchNorm1d(self.hidden_neurons) for i in range(self.num_layers)])

        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x, latent):
        if self.use_grid:
            x = self.conv0(x).transpose(2,1) # [1, 121, 7]
        latent = self.leakyrelu(self.bn2(self.conv2(latent)))
        for i in range(self.num_layers):
            latent = self.leakyrelu(self.bn_list[i](self.conv_list[i](latent)))
        # v1
        # latent = self.last_conv(latent) # [B, 121, 7]
        # if self.use_grid:
        #     latent = latent + x # latent + grid --> deform into point clouds
        #     latent = self.leakyrelu(self.bn1(self.conv1(latent)))
        # v2
        latent = self.leakyrelu(self.bn3(self.last_conv(latent))) # [B, 121, 7]
        if self.use_grid:
            latent = latent + x # latent + grid --> deform into point clouds
            latent = self.bn1(self.conv1(latent))
        return latent


class PointNetfeatPoint(nn.Module):
    def __init__(self, indim=7, latdim=8, n_pts=256, input_transform=False, feature_transform=False, template_type='square', device='cpu'):
        super(PointNetfeatPoint, self).__init__()
        self.indim = indim
        self.latdim = latdim
        self.input_transform = input_transform
        self.feature_transform = feature_transform
        self.n_pts = n_pts
        self.template_type = template_type
        # self.nb_pts_in_primitive = n_pts // opt.nprimitives
        
        if self.input_transform:
            self.stn = STNkd(indim=n_pts, k=self.indim) # depends on number of points
        self.conv1 = torch.nn.Conv1d(n_pts, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, latdim, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(latdim)
        
        self.lin1 = nn.Linear(latdim, latdim*4)
        self.lin2 = nn.Linear(latdim*4, latdim)
        self.bnl1 = nn.BatchNorm1d(latdim*4)
        self.bnl2 = nn.BatchNorm1d(latdim)
        
        if self.feature_transform:
            self.fstn = STNkd(indim=n_pts, k=latdim)
        
        # type 1 decoder
        self.conv4 = torch.nn.Conv1d(latdim, 64, 1)
        self.conv5 = torch.nn.Conv1d(64, 128, 1)
        self.conv6 = torch.nn.Conv1d(128, n_pts, 1)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(n_pts)
        
        # type 2 decoder
        self.template = get_template(template_type, device=device)
        self.deform_decoder = Mapping2Dto3D(self.template.dim, latdim=latdim, ddim=indim, n_pts=n_pts, num_layers=4, hidden_neurons=128, use_grid=True) # hidden_neurons=64  
        
        self.leakyrelu = nn.LeakyReLU(0.1)

    def encode(self, x, pooling=False):
        n_pts = x.size()[1]
        if self.input_transform:
            trans = self.stn(x)
            x = torch.bmm(x, trans)

        x = self.leakyrelu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = torch.bmm(x, trans_feat)
        else:
            trans_feat = None
        
        x = self.leakyrelu(self.bn2(self.conv2(x)))
        x = self.leakyrelu(self.bn3(self.conv3(x)))
        
        if pooling:
            # x, _ = torch.max(x, 2)
            x = torch.mean(x, 2)
            x = x.view(-1, self.latdim)
            x = self.leakyrelu(self.bnl1(self.lin1(x).unsqueeze(-1)))
            x = self.leakyrelu(self.bnl2(self.lin2(x.squeeze(2)).unsqueeze(-1))).squeeze(2)
        return x
        
    def decode(self, z):
        x = self.leakyrelu(self.bn4(self.conv4(z)))
        x = self.leakyrelu(self.bn5(self.conv5(x)))
        x = self.bn6(self.conv6(x))
        return x
    
    def forward(self, x):
        # z = self.encode(x)
        # x_hat = self.decode(z)
        
        z = self.encode(x, pooling=False)
        if self.training:
            input_points = self.template.get_random_points(
                torch.Size((1, self.template.dim, self.n_pts)), z.device)
        else:
            input_points = self.template.get_regular_points(self.n_pts, z.device)
        # x_hat = self.deform_decoder(input_points, z.unsqueeze(2)).transpose(2,1)
        x_hat = self.deform_decoder(input_points, z)
        # if not self.training:
        #     print(x.shape, x_hat.shape, z.shape, input_points.shape)
        return x_hat, z
######################################################### pointnet END #########################################################



######################################################### Discriminator_point #########################################################
class Discriminator_point(nn.Module):
    def __init__(self, indim=7, latdim=2, n_pts=16, device='cpu'):
        super(Discriminator_point, self).__init__()
        self.encoder = PointNetfeat(indim=indim, latdim=latdim, n_pts=n_pts).to(device)
        self.n_pts = n_pts
        self.latdim = latdim
        
        self.linearblock = nn.Sequential(
            nn.Linear(self.latdim * self.n_pts, 4 * self.latdim),
            nn.LeakyReLU(),
            nn.Linear(4 * self.latdim, 1),
        )

    def forward(self, input):
        z = self.encoder.encode(input)
        z = z.view(-1, self.latdim * self.n_pts)
        z = self.linearblock(z)
        return z
######################################################### Discriminator_point END #########################################################


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, h_channel=32, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channel, h_channel, 1, padding=0) 
        self.conv2 = nn.Conv1d(h_channel, out_channel, 1, padding=0) 
        # zero initialize 
        self.conv1.weight.data.zero_()
        self.conv1.bias.data.zero_()
        self.conv2.weight.data.zero_()
        self.conv2.bias.data.zero_()
        
        # self.scale = nn.Parameter(torch.zeros(1, out_channel, 1))
        # self.relu = nn.LeakyReLU(0.1, inplace=True)
        # self.relu = nn.ReLU(inplace=True)
        self.LeakyReLU = nn.LeakyReLU(0.1)

    def forward(self, input):
        # input: 512, 4, 7
        # out: 512, 8, 7
        # h = h * torch.exp(self.scale * 3) # logscale_factor=3
        # h = self.LeakyReLU(self.conv(input))
        h = self.LeakyReLU(self.conv1(input))
        h = self.LeakyReLU(self.conv2(h))
        return h
            
            
class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, h_channel=32, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, h_channel, 1, padding=0) 
        self.conv2 = nn.Conv2d(h_channel, out_channel, 1, padding=0) 
        # zero initialize 
        self.conv1.weight.data.zero_()
        self.conv1.bias.data.zero_()
        self.conv2.weight.data.zero_()
        self.conv2.bias.data.zero_()
        
        # self.scale = nn.Parameter(torch.zeros(1, out_channel, 1))
        # self.relu = nn.LeakyReLU(0.1, inplace=True)
        # self.relu = nn.ReLU(inplace=True)
        self.LeakyReLU = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input):
        # input: 512, 4, 7, 1
        # out: 512, 8, 7, 1
        # h = h * torch.exp(self.scale * 3) # logscale_factor=3
        # h = self.LeakyReLU(self.conv(input))
        input = input.unsqueeze(-1) # [512, 4, 7, 1]
        h = self.LeakyReLU(self.conv1(input))
        h = self.LeakyReLU(self.conv2(h))
        return h.squeeze(-1)


# -----------------------------------------------------------------------------------------
class vaeParticleFlow(nn.Module):

    def __init__(self, input_dim: int, num_blocks=6, k=16, lr_latdim=16, vae_latdim=4, device='cuda'):
        super(vaeParticleFlow, self).__init__()

        self.num_blocks = num_blocks
        self.k = k
        self.lr_latdim = lr_latdim
        self.vae_latdim = vae_latdim
        
        self.dist_stdGaussian = GaussianDistribution(input_dim, mu=0.0, vars=1.0, temperature=1.0, device=device)
        self.dist_Gaussian = GaussianDistribution(input_dim, mu=0.0, vars=1.0, temperature=1.0, device=device)
 
        feat_channels = [input_dim, 8, 16] + [32] * (self.num_blocks - 2)
        growth_widths = [4, 8] + [16] * (self.num_blocks - 2)
        cond_channels = [8, 16] + [32] * (self.num_blocks - 2)

        # self.feat_convs = nn.ModuleList()
        # for i in range(self.num_blocks):
        #     feat_conv = FeatureExtractUnit(feat_channels[i], feat_channels[i + 1], self.k, growth_widths[i], is_dynamic=False)
        #     self.feat_convs.append(feat_conv)

        # feature to condition
        # self.merge_convs = nn.ModuleList()
        # for i in range(self.num_blocks):
        #     merge_unit = FeatMergeUnit(feat_channels[i + 1], cond_channels[i])
        #     self.merge_convs.append(merge_unit)

        self.flow_blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            # step = FlowBlock(idim=input_dim, hdim=16, cdim=cond_channels[i], is_even=(i % 2 == 0))
            step = FlowBlock(idim=input_dim, hdim=16, cdim=None, is_even=(i % 2 == 0), device=device)
            # step = FlowBlock(idim=input_dim, hdim=8, cdim=None, is_even=(i % 2 == 0), device=device)
            self.flow_blocks.append(step)
        
        # for latent upsample (latdim: 4 --> input_dim=7)
        self.hidden_neuron = 128
        self.folding = nn.Sequential(
                nn.Conv1d(self.vae_latdim, self.hidden_neuron, 1),
                nn.BatchNorm1d(self.hidden_neuron),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv1d(self.hidden_neuron, self.hidden_neuron, 1),
                nn.BatchNorm1d(self.hidden_neuron),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv1d(self.hidden_neuron, input_dim, 1),
                nn.BatchNorm1d(input_dim),
        )
        
        # for gaussian 
        # self.prior = ZeroConv1d(self.lr_latdim, self.lr_latdim*2)
        self.prior = ZeroConv2d(self.lr_latdim, self.lr_latdim*2)
    
    def f(self, xyz: Tensor, cs: List[Tensor]):
        (B, _, _), device = xyz.shape, xyz.device
        log_det_J = torch.zeros((B,), device=device)
        # log_det_abs = torch.zeros((B,), device=device)
        
        p = xyz
        for i in range(self.num_blocks):
            p, _log_det_J = self.flow_blocks[i](p, cs[i])
            if _log_det_J is not None:
                log_det_J += _log_det_J
                # log_det_abs += torch.abs(_log_det_J)
        return p, log_det_J

    # added reverse flow
    def g(self, z: Tensor, cs: List[Tensor]):
        (B, _, _), device = z.shape, z.device
        log_det_J = torch.zeros((B,), device=device)
        
        for i in reversed(range(self.num_blocks)):
            # c = torch.repeat_interleave(cs[i], upratio, dim=1) # (input, repeats, dim=None, *, output_size=None)
            z, _log_det_J = self.flow_blocks[i].inverse(z, c=cs[i])
            if _log_det_J is not None:
                log_det_J += _log_det_J
        return z, log_det_J
    
    def forward(self, xyz: Tensor, z_vae=None, resample=False, temperature=None, mode='standard'):
        p = xyz
        z_vae_folded = None
        x_hat_folded = None
        
        # cs = self.feat_extract(p, knn_idx)  3 cs can be importance information
        cs = [None for i in range(len(xyz))]
        # z, mu, logvar, log_det_J, logp_z = self.log_prob(p, cs) # (z.shape)[512, 32, 7]
        z, log_det_J, logp_z = self.log_prob(p, cs) # (z.shape)[512, 32, 7]
        if resample:
            if mode == 'standard':
                z_ = self.dist_stdGaussian.standard_sample(z.shape, z.device)
                z = z_
                logp_z= self.dist_stdGaussian.standard_logp(z).to(z.device)
            elif mode == 'uniform':
                z_ = self.dist_stdGaussian.rand_sample(z.shape, z.device, min_=-1, max_=1)
                z = z_
                logp_z= self.dist_stdGaussian.standard_logp(z).to(z.device)
            elif mode == 'tail':
                z_ = scipy.stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size=z.shape)
                z = torch.from_numpy(z_.astype(np.float32)).to(z.device)
                logp_z= self.dist_stdGaussian.standard_logp(z).to(z.device)
        
        # Reconstruction 
        x_hat, log_det_J_rev = self.g(z, cs)
        
        if z_vae is not None:
            z_vae_folded = self.folding(z_vae).transpose(1, 2)
            z_replaced = z.detach().clone()
            z_replaced[:, :self.lr_latdim, :] = z_vae_folded
            x_hat_folded, _ = self.g(z_replaced, cs)
        return z, x_hat, z_vae_folded, x_hat_folded, log_det_J, logp_z
    
    def log_prob(self, xyz: Tensor, cs: List[Tensor]): # modified x-->z
        z, log_det_J = self.f(xyz, cs)
        
        # standard gaussian probs
        # z split into two parts, z_vae: [512, lowres_latdim, 7]
        # mu, logvar = self.prior(z[:, :self.lr_latdim, :]).chunk(2, 1)
        # logp_z_1 = self.dist_Gaussian.logp(mu, logvar, z=z[:, :self.lr_latdim, :]).to(z.device)
        # logp_z_2 = self.dist_stdGaussian.standard_logp(z=z[:, self.lr_latdim:, :]).to(z.device)
        # logp_z = logp_z_1 + logp_z_2
        # return z, mu, logvar, log_det_J, logp_z
        
        logp_z = self.dist_stdGaussian.standard_logp(z).to(z.device)
        return z, log_det_J, logp_z
    
    def sr(self, z_vae, zshape=[24, 7], xyz=None):
        cs = [None for i in range(len(z_vae))]
        zshape = [z_vae.shape[0]] + zshape
        # z_sampled = self.dist_stdGaussian.standard_sample(zshape, z_vae.device)
        z, _ = self.f(xyz, cs)
        """
        z_vae_folded = self.folding(z_vae).transpose(1, 2) # folded has loss
        # z = torch.cat([z_vae_folded, z_sampled], dim=1).to(z_vae.device)
        z[:, :self.lr_latdim] = z_vae_folded
        """
        z[:, self.lr_latdim:] = torch.zeros(z[:, self.lr_latdim:].shape).to(z_vae.device)
        x_hat, log_det_J_rev = self.g(z, cs)
        # logp_z_1 = self.dist_Gaussian.logp(mu, logvar, z=z_vae.reshape(z_vae.shape[0], -1)).to(z_vae.device)
        # logp_z_2 = self.dist_stdGaussian.standard_logp(z=z_sampled.reshape(z_vae.shape[0], -1)).to(z_vae.device)
        # logp_z_1 = self.dist_Gaussian.logp(mu, logvar, z=z_vae).to(z_vae.device)
        # logp_z_2 = self.dist_stdGaussian.standard_logp(z=z_sampled).to(z_vae.device)
        logp_z = self.dist_stdGaussian.standard_logp(z=z).to(z_vae.device)
        return x_hat, logp_z-log_det_J_rev #logp_z_1, logp_z_2, logp_z_1 * logp_z_2 - log_det_J_rev
    
    # def sample(self, sparse: Tensor, upratio=4):
    #     dense, _ = self(sparse, upratio) # pos, upratio
    #     return dense
    #     full_x = self.g(z, idxes)
    #     clean_x = full_x[..., :self.in_channel]  # [B, N, 3]
    #     return clean_x
    
    def sample(self, z, shape=[128, 24, 7]):
        cs = [None for i in range(len(z))]
        z_ = self.dist_stdGaussian.standard_sample(shape, z.device)
        z[:, self.lr_latdim:] = z_
        # z_ = torch.cat([z_vae_folded, z_], dim=1)
        x_hat, _ = self.g(z, cs)
        return x_hat 
    
    def init_as_trained_state(self):
        """Set the network to initialized state, needed for evaluation(significant performance impact)"""
        for i in range(self.num_blocks):
            self.flow_blocks[i].actnorm._initialized = torch.tensor(True)
        
# -----------------------------------------------------------------------------------------
