import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from typing import List
# from pytorch3d.ops import knn_gather, knn_points
# from pytorch3d.ops import sample_farthest_points

from models.modules.flows.coupling import AffineCouplingLayer, AffineInjectorLayer
from models.modules.flows.coupling import AffineSpatialCouplingLayer
from models.modules.flows.normalize import ActNorm
from models.modules.flows.permutate import Permutation
from models.modules.flows.linear import SoftClampling

from models.modules.utils.probs import GaussianDistribution

import scipy
import numpy as np

# -----------------------------------------------------------------------------------------
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
            # layers.extend([linear, activate, batchnorm])
            # layers.extend([linear, activate])
            
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
        
# -----------------------------------------------------------------------------------------
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
        
        """
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
        """
        
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
    
    # def forward(self, x: Tensor, c: Tensor=None, knn_idx=None):
    #     x, _log_det0 = self.permutate1(x, c)
    #     x, _log_det1 = self.actnorm(x)
    #     x, _log_det2 = self.coupling1(x, c)
    #     x, _         = self.permutate2(x, c)
    #     x, _log_det4 = self.coupling2(x, c)
    #     return x, _log_det0 + _log_det1 + _log_det2 + _log_det4
    
    # def inverse(self, z: Tensor, c: Tensor=None, knn_idx=None):
    #     z, _log_det4 = self.coupling2.inverse(z, c)
    #     z, _         = self.permutate2.inverse(z, c)
    #     z, _log_det2 = self.coupling1.inverse(z, c)
    #     z, _log_det1 = self.actnorm.inverse(z)
    #     z, _log_det0 = self.permutate1.inverse(z, c)
    #     return z, _log_det0 + _log_det1 + _log_det2 + _log_det4



# -----------------------------------------------------------------------------------------
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
# class KnnContextEncoder(nn.Module):

#     def __init__(self, input_dim, k):
#         super(KnnContextEncoder, self).__init__()
#         self.distance_encoder = DistanceEncoder(dim_in=input_dim, dim_out=32, k=k)
#         self.feat_conv = FeatureExtractUnit(input_dim, 32, k=k, growth_width=16, is_dynamic=False)

#     def forward(self, xyz: Tensor):
#         B, N, _ = xyz.shape
#         dist, idx_knn = self.distance_encoder(xyz)  # [B, C, N, k]
#         feat = self.feat_conv(xyz, idx_knn.view(B, N, -1), is_pooling=False)  # [B, C, N, K]
#         return torch.cat([dist, feat], dim=1), idx_knn

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
class FeatureExtractUnit(nn.Module):

    def __init__(self, idim: int, odim: int, k: int, growth_width: int, is_dynamic: bool):
        super(FeatureExtractUnit, self).__init__()
        assert (odim % growth_width == 0)

        self.k = k
        self.is_dynamic = is_dynamic
        self.num_conv = (odim // growth_width)

        idim = idim * 3
        self.convs = nn.ModuleList()

        conv_first = nn.Sequential(*[
            nn.Conv2d(idim, growth_width, kernel_size=[1, 1]),
            # nn.BatchNorm2d(growth_width),
            nn.LeakyReLU(0.05, inplace=True),
        ])
        self.convs.append(conv_first)

        for i in range(self.num_conv - 1):
            in_channel = growth_width * (i + 1) + idim

            conv = nn.Sequential(*[
                nn.Conv2d(in_channel, growth_width, kernel_size=[1, 1], bias=True),
                # nn.InstanceNorm1d(growth_width),
                # nn.BatchNorm2d(growth_width),
                nn.LeakyReLU(0.05, inplace=True),
            ])
            self.convs.append(conv)

        self.conv_out = nn.Conv2d(growth_width * self.num_conv + idim, odim, kernel_size=[1, 1], bias=True)

    def derive_edge_feat(self, x: Tensor, knn_idx: Tensor or None):
        """
        x: [B, N, C]
        """
        if knn_idx is None and self.is_dynamic:
            _, knn_idx, _ = knn_points(x, x, K=self.k, return_nn=False, return_sorted=False)  # [B, N, K]
        knn_feat = knn_gather(x, knn_idx)                         # [B, N, K, C]
        x_tiled = torch.unsqueeze(x, dim=-2).expand_as(knn_feat)  # [B, N, K, C]
        return torch.cat([x_tiled, knn_feat, knn_feat - x_tiled], dim=-1)  # [B, N, K, C * 3]

    def forward(self, x: Tensor, knn_idx: Tensor or None, is_pooling=True):
        f = self.derive_edge_feat(x, knn_idx)  # [B, N, K, C]
        f = f.permute(0, 3, 1, 2)              # [B, C, N, K]

        for i in range(self.num_conv):
            _f = self.convs[i](f)
            f = torch.cat([f, _f], dim=1)

        f = self.conv_out(f)  # [B, C, N, K]

        if is_pooling:
            f, _ = torch.max(f, dim=-1, keepdim=False)
            return torch.transpose(f, 1, 2)   # [B, N, C]
        else:
            return f

# -----------------------------------------------------------------------------------------
class FeatMergeUnit(nn.Module):
    def __init__(self, idim: int, odim: int):
        super(FeatMergeUnit, self).__init__()
        self.conv1 = nn.Linear(idim, idim // 2, bias=True)
        self.conv2 = nn.Linear(idim // 2, odim, bias=False)

    def forward(self, x: Tensor):
        return self.conv2(F.relu(self.conv1(x)))


# -----------------------------------------------------------------------------------------
class ParticleFlow(nn.Module):

    def __init__(self, input_dim: int, num_blocks=6, num_neighbors=16, device='cuda'):
        super(ParticleFlow, self).__init__()

        self.num_blocks = num_blocks
        self.num_neighbors = num_neighbors
        
        self.dist = GaussianDistribution(input_dim, mu=0.0, vars=1.0, temperature=1.0, device=device)
        # self.interp = InterpolationModule(input_dim=3, k=self.num_neighbors)

        feat_channels = [input_dim, 8, 16] + [32] * (self.num_blocks - 2)
        growth_widths = [4, 8] + [16] * (self.num_blocks - 2)
        cond_channels = [8, 16] + [32] * (self.num_blocks - 2)

        self.feat_convs = nn.ModuleList()
        for i in range(self.num_blocks):
            feat_conv = FeatureExtractUnit(feat_channels[i], feat_channels[i + 1], self.num_neighbors, growth_widths[i], is_dynamic=False)
            self.feat_convs.append(feat_conv)

        self.merge_convs = nn.ModuleList()
        for i in range(self.num_blocks):
            merge_unit = FeatMergeUnit(feat_channels[i + 1], cond_channels[i])
            self.merge_convs.append(merge_unit)

        self.flow_blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            # step = FlowBlock(idim=input_dim, hdim=16, cdim=cond_channels[i], is_even=(i % 2 == 0))
            step = FlowBlock(idim=input_dim, hdim=16, cdim=None, is_even=(i % 2 == 0), device=device)
            # step = FlowBlock(idim=input_dim, hdim=8, cdim=None, is_even=(i % 2 == 0), device=device)
            self.flow_blocks.append(step)

    def feat_extract(self, xyz: Tensor, knn_idx: Tensor):
        cs = []
        c = xyz

        for i in range(self.num_blocks):
            c = self.feat_convs[i](c, knn_idx=knn_idx)
            _c = self.merge_convs[i](c)
            cs.append(_c)
        return cs

    # def set_to_initialized_state(self):
    #     for i in range(self.num_blocks):
    #         self.flow_blocks[i].actnorm.is_inited = True
    
    def f(self, xyz: Tensor, cs: List[Tensor]):
        (B, _, _), device = xyz.shape, xyz.device
        log_det_J = torch.zeros((B,), device=device)
        log_det_abs = torch.zeros((B,), device=device)
        
        p = xyz
        for i in range(self.num_blocks):
            # print('block i', i)
            p, _log_det_J = self.flow_blocks[i](p, cs[i])
            if _log_det_J is not None:
                log_det_J += _log_det_J
                log_det_abs += torch.abs(_log_det_J)
        # exit()
        return p, log_det_J, log_det_abs

    # added reverse flow
    def g(self, z: Tensor, cs: List[Tensor]):
        # z = torch.flatten(z.transpose(2, 3), 1, 2)
        (B, _, _), device = z.shape, z.device
        log_det_J = torch.zeros((B,), device=device)
        
        for i in reversed(range(self.num_blocks)):
            # c = torch.repeat_interleave(cs[i], upratio, dim=1) # (input, repeats, dim=None, *, output_size=None)
            z, _log_det_J = self.flow_blocks[i].inverse(z, c=cs[i])
            if _log_det_J is not None:
                log_det_J += _log_det_J
        return z, log_det_J
    
    def forward(self, xyz: Tensor, resample=False, temperature=None, mode='standard'):
        # _, knn_idx, _ = knn_points(xyz, xyz, K=self.num_neighbors, return_nn=False, return_sorted=False)  # [B, N, K]
        p = xyz
        
        # cs = self.feat_extract(p, knn_idx) 
        cs = [None for i in range(len(xyz))]
        z, log_det_J, logp_z_, log_det_abs = self.log_prob(p, cs) # (z.shape)[512, 32, 7]
        
        if resample:
            if mode == 'standard':
                z_ = self.dist.standard_sample(z.shape, z.device)
                z = z_
                logp_z_= self.dist.standard_logp(z).to(z.device)
            elif mode == 'uniform':
                z_ = self.dist.rand_sample(z.shape, z.device, min_=-1, max_=1)
                z = z_
                logp_z_= self.dist.standard_logp(z).to(z.device)
            elif mode == 'tail':
                # torch.normal(mean=torch.zeros(shape), std=(torch.ones(shape) * temp)).to(device)
                # z_ = self.dist.standard_sample(z.shape, z.device)
                # z = z_
                # import pdb
                # pdb.set_trace()
                # https://blog.csdn.net/CSDNBigBoy/article/details/97034126
                z_ = scipy.stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size=z.shape)
                z = torch.from_numpy(z_.astype(np.float32)).to(z.device)
                logp_z_= self.dist.standard_logp(z).to(z.device)
        
        # Reconstruction 
        x_hat, log_det_J_rev = self.g(z, cs)
        return z, x_hat, log_det_J, logp_z_, log_det_J_rev, log_det_abs
    
    def log_prob(self, xyz: Tensor, cs: List[Tensor]): # modified x-->z
        z, log_det_J, log_det_abs = self.f(xyz, cs)

        # standard gaussian probs
        logp_z_ = self.dist.standard_logp(z).to(z.device)
        # logp_x = -torch.mean(logp_z_ + log_det_J)
        return z, log_det_J, logp_z_, log_det_abs

    # def sample(self, sparse: Tensor, upratio=4):
    #     dense, _ = self(sparse, upratio) # pos, upratio
    #     return dense
    #     full_x = self.g(z, idxes)
    #     clean_x = full_x[..., :self.in_channel]  # [B, N, 3]
    #     return clean_x
    
    def init_as_trained_state(self):
        """Set the network to initialized state, needed for evaluation(significant performance impact)"""
        for i in range(self.num_blocks):
            # self.flow_blocks[i].actnorm.is_inited = True
            self.flow_blocks[i].actnorm._initialized = torch.tensor(True)
        
# -----------------------------------------------------------------------------------------
