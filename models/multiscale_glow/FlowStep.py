import torch
from torch import nn as nn

from models.multiscale_glow import ActNorms, Permutations, AffineCouplings
from models.multiscale_glow.Basic import SoftClampling, IdentityLayer

class FlowStep(nn.Module):
    def __init__(self, in_channels, cond_channels=None, flow_actNorm='actNorm2d', flow_permutation='invconv', flow_coupling='Affine', LRvsothers=True,
                 actnorm_scale=1.0, LU_decomposed=False, split_dim=1, ks_s_pad=[1, 1, 0]):
        super().__init__()
        self.flow_actNorm = flow_actNorm
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling

        # 1. actnorm
        if self.flow_actNorm == 'actNorm3d':
            self.actnorm = ActNorms.ActNorm3d(in_channels, actnorm_scale)
        elif self.flow_actNorm == "none":
            self.actnorm = None

        # 2. permute # todo: maybe hurtful for downsampling; presever the structure of downsampling
        if self.flow_permutation == "invconv":
            self.permute = Permutations.InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
        elif self.flow_permutation == "none":
            self.permute = None

        # 3. coupling
        if self.flow_coupling == "AffineInjector":
            self.affine = AffineCouplings.AffineCouplingInjector(in_channels=in_channels, cond_channels=cond_channels)
        elif self.flow_coupling == "noCoupling":
            pass
        elif self.flow_coupling == "Affine":
            self.affine = AffineCouplings.AffineCoupling(in_channels=in_channels, cond_channels=cond_channels)
        elif self.flow_coupling == "Affine3shift":
            self.affine = AffineCouplings.AffineCoupling3shift(in_channels=in_channels, cond_channels=cond_channels, LRvsothers=LRvsothers)
        
    def forward(self, z, u=None, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(z, u, logdet)
        else:
            return self.reverse_flow(z, u, logdet)

    def normal_flow(self, z, u=None, logdet=None):
        # 1. actnorm
        if self.actnorm is not None:
            z, logdet = self.actnorm(z, logdet=logdet, reverse=False)
        # 2. permute
        if self.permute is not None:
            z, logdet = self.permute(z, logdet=logdet, reverse=False)
        # 3. coupling
        z, logdet = self.affine(z, u=u, logdet=logdet, reverse=False)
        return z, logdet

    def reverse_flow(self, z, u=None, logdet=None):
        # 1.coupling
        z, _ = self.affine(z, u=u, logdet=logdet, reverse=True)
        # 2. permute
        if self.permute is not None:
            z, _ = self.permute(z, logdet=logdet, reverse=True)
        # 3. actnorm
        if self.actnorm is not None:
            z, _ = self.actnorm(z, logdet=logdet, reverse=True)
        return z, logdet



class FlowAssembly(nn.Module):
    def __init__(self, in_channels, cond_channels=None, flow_actNorm='actNorm2d', split_dim=1, l_id=0, ks_s_pad=[1, 1, 0]):
        super(FlowAssembly, self).__init__()
        self.flow_actNorm = flow_actNorm
        # self.flow_permutation = flow_permutation
        # self.flow_coupling = flow_coupling

        # 1. actnorm
        if self.flow_actNorm == 'actNorm2d':
            self.actnorm = ActNorms.ActNorm2d(in_channels)
        elif self.flow_actNorm == "none":
            self.actnorm = None

        # 2. permute # todo: maybe hurtful for downsampling; presever the structure of downsampling
        self.permute1 = Permutations.InvertibleConv1x1(in_channels)
        self.permute2 = Permutations.InvertibleConv1x1(in_channels)
        
        # 3. coupling
        self.affine1 = AffineCouplings.AffineCoupling(in_channels=in_channels, cond_channels=cond_channels)
        self.affine2 = AffineCouplings.AffineCoupling(in_channels=in_channels, cond_channels=cond_channels)
        
    def forward(self, z, u=None, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(z, u, logdet)
        else:
            return self.reverse_flow(z, u, logdet)

    def normal_flow(self, z, u=None, logdet=None):
        if self.actnorm is not None:
            z, logdet = self.actnorm(z, logdet=logdet, reverse=False)
        z, logdet = self.permute1(z, logdet=logdet, reverse=False)
        z, logdet = self.affine1(z, u=u, logdet=logdet, reverse=False)
        z, logdet = self.permute2(z, logdet=logdet, reverse=False)
        z, logdet = self.affine2(z, u=u, logdet=logdet, reverse=False)
        return z, logdet

    def reverse_flow(self, z, u=None, logdet=None):
        z, _ = self.affine2(z, u=u, logdet=logdet, reverse=True)
        z, _ = self.permute2(z, logdet=logdet, reverse=True)
        z, _ = self.affine1(z, u=u, logdet=logdet, reverse=True)
        z, _ = self.permute1(z, logdet=logdet, reverse=True)
        if self.actnorm is not None:
            z, _ = self.actnorm(z, logdet=logdet, reverse=True)
        return z, _
