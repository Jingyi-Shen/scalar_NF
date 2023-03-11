import torch
from torch import nn as nn
import torch.nn.functional as F

from models.multiscale_glow import thops
from models.multiscale_glow.Basic import DenseBlock, FCN, RDN, IdentityLayer

from pytorch3d.ops import knn_gather, knn_points

class AffineCoupling(nn.Module):
    def __init__(self, in_channels, cond_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = 32 # 32 # 64 # opt_get(opt, ['hidden_channels'], 64)
        self.n_hidden_layers = 4 #4
        self.cond_channels = cond_channels
        f_in_channels = self.in_channels//2 if cond_channels is None else self.in_channels//2 + cond_channels
        f_out_channels = (self.in_channels - self.in_channels//2) * 2
        # print(f_in_channels, f_out_channels)
        nn_module = 'FCN' # opt_get(opt, ['nn_module'], 'FCN')
        # nn_module = 'DenseBlock'
        if nn_module == 'DenseBlock':
            self.f = DenseBlock(in_channels=f_in_channels, out_channels=f_out_channels, gc=self.hidden_channels)
        elif nn_module == 'FCN':
            self.f = FCN(in_channels=f_in_channels, out_channels=f_out_channels, hidden_channels=self.hidden_channels,
                         n_hidden_layers=self.n_hidden_layers)

    def forward(self, z, u=None, y=None, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(z, u, y, logdet)
        else:
            return self.reverse_flow(z, u, y, logdet)


    def normal_flow(self, z, u=None, y=None, logdet=None):
        z1, z2 = thops.split_feature(z, "split")
        # print('z1.shape, z2.shape', z1.shape, z2.shape)
        h = self.f(z1) if self.cond_channels is None else self.f(thops.cat_feature(z1, u))
        shift, scale = thops.split_feature(h, "cross")
        # adding 1e-4 is crucial for torch.slogdet(), as used in Glow (leads to black rect in experiments).
        # see https://github.com/didriknielsen/survae_flows/issues/5 for discussion.
        # or use `torch.exp(2. * torch.tanh(s / 2.)) as in SurVAE (more unstable in practice).

        # version 1, srflow (use FCN)
        scale = torch.sigmoid(scale + 2.) + 1e-4
        z2 = (z2 + shift) * scale
        logdet += thops.sum(torch.log(scale), dim=[1, 2, 3, 4])

        # version2, survae
        # logscale = 2. * torch.tanh(scale / 2.)
        # z2 = (z2+shift) * torch.exp(logscale) # as in glow, it's shift+scale!
        # logdet += thops.sum(logscale, dim=[1, 2, 3])

        # version3, FrEIA, now have problem with FCN, but densenet is ok. (use FCN2/Denseblock)
        # logscale = 0.5 * 0.636 * torch.atan(scale / 0.5) # clamp it to be between [-0.5,0.5]
        # logscale = 0.318 * torch.atan(2 * scale)
        # z2 = (z2 + shift) * torch.exp(logscale)
        
        z = thops.cat_feature(z1, z2)

        return z, logdet

    def reverse_flow(self, z, u=None, y=None, logdet=None):
        z1, z2 = thops.split_feature(z, "split")

        h = self.f(z1) if self.cond_channels is None else self.f(thops.cat_feature(z1, u))
        shift, scale = thops.split_feature(h, "cross")

        # version1, srflow
        scale = torch.sigmoid(scale + 2.) + 1e-4
        z2 = (z2 / scale) -shift

        # version2, survae
        # logscale = 2. * torch.tanh(scale / 2.)
        # z2 = z2 * torch.exp(-logscale) - shift

        # version3, FrEIA
        # logscale = 0.5 * 0.636 * torch.atan(scale / 0.5)
        # logscale = 0.318 * torch.atan(2 * scale)
        # z2 = z2 * torch.exp(-logscale) - shift
        
        logdet -= thops.sum(torch.log(scale), dim=[1, 2, 3, 4])

        z = thops.cat_feature(z1, z2)

        return z, logdet


'''3 channel conditional on the rest channels, or vice versa. only shift LR.
   used in image rescaling to divide the low-frequencies and the high-frequencies apart from early flow layers.'''
class AffineCoupling3shift(nn.Module):
    def __init__(self, in_channels, cond_channels=None, LRvsothers=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = 32 # opt_get(opt, ['hidden_channels'], 64)
        self.n_hidden_layers = 1
        self.kernel_hidden = 1
        self.cond_channels = cond_channels
        self.LRvsothers = LRvsothers
        if LRvsothers:
            f_in_channels = 3 if cond_channels is None else 3 + cond_channels
            f_out_channels = (self.in_channels - 3) * 2
        else:
            f_in_channels = self.in_channels - 3 if cond_channels is None else self.in_channels - 3 + cond_channels
            f_out_channels = 3
        nn_module = 'FCN' # opt_get(opt, ['nn_module'], 'FCN')

        if nn_module == 'DenseBlock':
            self.f = DenseBlock(in_channels=f_in_channels, out_channels=f_out_channels, gc=self.hidden_channels)
        elif nn_module == 'FCN':
            self.f = FCN(in_channels=f_in_channels, out_channels=f_out_channels, hidden_channels=self.hidden_channels,
                          kernel_hidden=self.kernel_hidden, n_hidden_layers=self.n_hidden_layers)


    def forward(self, z, u=None, y=None, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(z, u, y, logdet)
        else:
            return self.reverse_flow(z, u, y, logdet)

    def normal_flow(self, z, u=None, y=None, logdet=None):
        if self.LRvsothers:
            z1, z2 = z[:, :3, ...], z[:, 3:, ...]
            h = self.f(z1) if self.cond_channels is None else self.f(thops.cat_feature(z1, u))
            shift, scale = thops.split_feature(h, "cross")
            # logscale = 0.318 * torch.atan(2 * scale)
            # z2 = (z2 + shift) * torch.exp(logscale)
            scale = torch.sigmoid(scale + 2.) + 1e-4
            z2 = (z2 + shift) * scale
            if logdet is not None:
                # logdet += thops.sum(logscale, dim=[1, 2, 3])
                logdet += thops.sum(torch.log(scale), dim=[1, 2, 3])
        else:
            z2, z1 = z[:, :3, ...], z[:, 3:, ...]
            shift = self.f(z1) if self.cond_channels is None else self.f(thops.cat_feature(z1, u))
            z2 = z2 + shift

        if self.LRvsothers:
            z = thops.cat_feature(z1, z2)
        else:
            z = thops.cat_feature(z2, z1)

        return z, logdet

    def reverse_flow(self, z, u=None, y=None, logdet=None):
        if self.LRvsothers:
            z1, z2 = z[:, :3, ...], z[:, 3:, ...]
            h = self.f(z1) if self.cond_channels is None else self.f(thops.cat_feature(z1, u))
            shift, scale = thops.split_feature(h, "cross")
            # logscale = 0.318 * torch.atan(2 * scale)
            # z2 = z2 * torch.exp(-logscale) - shift
            scale = torch.sigmoid(scale + 2.) + 1e-4
            z2 = z2 / scale - shift
            logdet -= thops.sum(torch.log(scale), dim=[1, 2, 3])
        else:
            z2, z1 = z[:, :3, ...], z[:, 3:, ...]
            shift = self.f(z1)
            z2 = z2 - shift

        if self.LRvsothers:
            z = thops.cat_feature(z1, z2)
        else:
            z = thops.cat_feature(z2, z1)

        return z, logdet


''' srflow's affine injector + original affine coupling, not used in this project'''
class AffineCouplingInjector(nn.Module):
    def __init__(self, in_channels, cond_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = 32 # opt_get(opt, ['hidden_channels'], 64)
        self.n_hidden_layers = 1
        self.kernel_hidden = 1
        self.cond_channels = cond_channels
        f_in_channels = self.in_channels//2 if cond_channels is None else self.in_channels//2 + cond_channels
        f_out_channels = (self.in_channels - self.in_channels//2) * 2
        nn_module = 'FCN' # opt_get(opt, ['nn_module'], 'FCN')
        if nn_module == 'DenseBlock':
            self.f = DenseBlock(in_channels=f_in_channels, out_channels=f_out_channels, gc=self.hidden_channels)
            self.f_injector = DenseBlock(in_channels=cond_channels, out_channels=self.in_channels*2, gc=self.hidden_channels)
        elif nn_module == 'FCN':
            self.f = FCN(in_channels=f_in_channels, out_channels=f_out_channels, hidden_channels=self.hidden_channels,
                          kernel_hidden=self.kernel_hidden, n_hidden_layers=self.n_hidden_layers)
            self.f_injector = FCN(in_channels=cond_channels, out_channels=self.in_channels*2, hidden_channels=self.hidden_channels,
                          kernel_hidden=self.kernel_hidden, n_hidden_layers=self.n_hidden_layers)

    def forward(self, z, u=None, y=None, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(z, u, y, logdet)
        else:
            return self.reverse_flow(z, u, y, logdet)

    def normal_flow(self, z, u=None, y=None, logdet=None):
        # overall-conditional
        h = self.f_injector(u)
        shift, scale = thops.split_feature(h, "cross")
        # logscale = 0.318 * torch.atan(2 * scale) # clamp it to be between [-5,5]
        # z = (z + shift) * torch.exp(logscale)
        # logdet += thops.sum(logscale, dim=[1, 2, 3])
        scale = torch.sigmoid(scale + 2.) + 1e-4
        z = (z + shift) * scale
        logdet += thops.sum(torch.log(scale), dim=[1, 2, 3])

        # self-conditional
        z1, z2 = thops.split_feature(z, "split")
        h = self.f(z1) if self.cond_channels is None else self.f(thops.cat_feature(z1, u))
        shift, scale = thops.split_feature(h, "cross")
        # logscale = 0.318 * torch.atan(2 * scale) # clamp it to be between [-5,5]
        # z2 = (z2 + shift) * torch.exp(logscale)
        # logdet += thops.sum(logscale, dim=[1, 2, 3])
        scale = torch.sigmoid(scale + 2.) + 1e-4
        z2 = (z2 + shift) * scale
        logdet += thops.sum(torch.log(scale), dim=[1, 2, 3])
        
        z = thops.cat_feature(z1, z2)
        return z, logdet

    def reverse_flow(self, z, u=None, y=None, logdet=None):
        # self-conditional
        z1, z2 = thops.split_feature(z, "split")
        h = self.f(z1) if self.cond_channels is None else self.f(thops.cat_feature(z1, u))
        shift, scale = thops.split_feature(h, "cross")
        # logscale = 0.318 * torch.atan(2 * scale)
        # z2 = z2 * torch.exp(-logscale) - shift
        scale = torch.sigmoid(scale + 2.) + 1e-4
        z2 = z2 / scale - shift
        logdet -= thops.sum(torch.log(scale), dim=[1, 2, 3])
        z = thops.cat_feature(z1, z2)

        # overall-conditional
        h = self.f_injector(u)
        shift, scale = thops.split_feature(h, "cross")
        # logscale = 0.318 * torch.atan(2 * scale)
        # z = z * torch.exp(-logscale) - shift
        scale = torch.sigmoid(scale + 2.) + 1e-4
        z = z / scale - shift
        logdet -= thops.sum(torch.log(scale), dim=[1, 2, 3])
        return z, logdet
       
 
# # -----------------------------------------------------------------------------------------

# class AffineCoupling1d(nn.Module):
#     def __init__(self, in_channels, clamp=None, cond_channels=None, ks_s_pad=[1, 1, 0], nn_module='double'):
#         super(AffineCoupling1d, self).__init__()
#         self.in_channels = in_channels
#         self.hidden_channels = 64 # 32/64 # opt_get(opt, ['hidden_channels'], 64)
#         self.n_hidden_layers = 1
#         self.kernel_hidden = 1
#         self.cond_channels = cond_channels
#         f_in_channels = self.in_channels//2 if cond_channels is None else self.in_channels//2 + cond_channels
#         f_out_channels = (self.in_channels - self.in_channels//2) * 2
#         f_out_channel = (self.in_channels - self.in_channels//2)
#         # nn_module = 'LinearUnit'
#         # nn_module = 'FCN1D'
#         # nn_module = 'FCN1D_v2'
#         # nn_module='double'
#         # nn_module='knnConvUnit'
#         if nn_module == 'DenseBlock':
#             self.f = DenseBlock(in_channels=f_in_channels, out_channels=f_out_channels, gc=self.hidden_channels)
#         elif nn_module == 'FCN':
#             self.f = FCN(in_channels=f_in_channels, out_channels=f_out_channels, hidden_channels=self.hidden_channels,
#                           kernel_hidden=self.kernel_hidden, n_hidden_layers=self.n_hidden_layers)
#         elif nn_module == 'LinearUnit':
#             self.f = LinearUnit(in_channels=f_in_channels, out_channels=f_out_channels, hidden_channels=self.hidden_channels, n_block=3, batch_norm=False)
#         elif nn_module == 'FCN1D':
#             self.f = FCN1D(in_channels=f_in_channels, out_channels=f_out_channels, hidden_channels=self.hidden_channels, kernel_size=ks_s_pad[0], stride=ks_s_pad[1], padding=ks_s_pad[2])
#         elif nn_module == 'FCN1D_v2':
#             self.f = FCN1D_v2(in_channels=f_in_channels, out_channels=f_out_channels, hidden_channels=self.hidden_channels)
#         elif nn_module == 'double':
#             self.f = FCN1D(in_channels=f_in_channels, out_channels=f_out_channel, hidden_channels=self.hidden_channels, kernel_size=ks_s_pad[0], stride=ks_s_pad[1], padding=ks_s_pad[2])
#             self.g = FCN1D(in_channels=f_in_channels, out_channels=f_out_channel, hidden_channels=self.hidden_channels, kernel_size=ks_s_pad[0], stride=ks_s_pad[1], padding=ks_s_pad[2])
#         elif nn_module == 'knnConvUnit':
#             self.f = KnnConvUnit(in_channels=f_in_channels, out_channels=f_out_channel, hidden_channels=self.hidden_channels)
#             self.g = KnnConvUnit(in_channels=f_in_channels, out_channels=f_out_channel, hidden_channels=self.hidden_channels)
        
#         self.clamping = clamp or IdentityLayer()
#         self.nn_module = nn_module
        
#     def forward(self, z, u=None, y=None, logdet=None, reverse=False):
#         if not reverse:
#             return self.normal_flow(z, u, y, logdet)
#         else:
#             return self.reverse_flow(z, u, y, logdet)

#     def normal_flow(self, z, u=None, y=None, logdet=None):
#         z1, z2 = thops.split_feature(z, "split")
        
#         if self.nn_module == 'double':
#             shift = self.f(z1) if self.cond_channels is None else self.f(thops.cat_feature(z1, u))
#             scale = self.g(z1) if self.cond_channels is None else self.g(thops.cat_feature(z1, u))
#         elif self.nn_module == 'knnConvUnit':
#             # knn_idx based on z1
#             _, knn_idx, _ = knn_points(p1=z1.permute(0,2,1).contiguous(), p2=z1.permute(0,2,1).contiguous(), K=16, return_sorted=False) 
#             shift = self.f(z1, knn_idx=knn_idx) if self.cond_channels is None else self.f(thops.cat_feature(z1, u), knn_idx=knn_idx)
#             scale = self.g(z1, knn_idx=knn_idx) if self.cond_channels is None else self.g(thops.cat_feature(z1, u), knn_idx=knn_idx)
#         else:
#             h = self.f(z1) if self.cond_channels is None else self.f(thops.cat_feature(z1, u))
#             shift, scale = thops.split_feature(h, "cross")

#         # adding 1e-4 is crucial for torch.slogdet(), as used in Glow (leads to black rect in experiments).
#         # see https://github.com/didriknielsen/survae_flows/issues/5 for discussion.
#         # or use `torch.exp(2. * torch.tanh(s / 2.)) as in SurVAE (more unstable in practice).

#         # version 1, srflow (use FCN)
#         scale = torch.sigmoid(scale + 2.) + 1e-4
#         # scale = self.clamping(scale + 2.) + 1e-4
#         z2 = (z2 + shift) * scale
        
#         # version2, survae
#         # logscale = 2. * torch.tanh(scale / 2.)
#         # z2 = (z2+shift) * torch.exp(logscale) # as in glow, it's shift+scale!
        
#         logdet += thops.sum(torch.log(scale), dim=[1, 2])
#         z = thops.cat_feature(z1, z2)
#         return z, logdet

#     def reverse_flow(self, z, u=None, y=None, logdet=None):
#         z1, z2 = thops.split_feature(z, "split")

#         if self.nn_module == 'double':
#             shift = self.f(z1) if self.cond_channels is None else self.f(thops.cat_feature(z1, u))
#             scale = self.g(z1) if self.cond_channels is None else self.g(thops.cat_feature(z1, u))
#         elif self.nn_module == 'knnConvUnit':
#             # knn_idx based on z1
#             _, knn_idx, _ = knn_points(p1=z1.permute(0,2,1).contiguous(), p2=z1.permute(0,2,1).contiguous(), K=16, return_sorted=False) 
#             shift = self.f(z1, knn_idx=knn_idx) if self.cond_channels is None else self.f(thops.cat_feature(z1, u), knn_idx=knn_idx)
#             scale = self.g(z1, knn_idx=knn_idx) if self.cond_channels is None else self.g(thops.cat_feature(z1, u), knn_idx=knn_idx)
#         else:
#             h = self.f(z1) if self.cond_channels is None else self.f(thops.cat_feature(z1, u))
#             shift, scale = thops.split_feature(h, "cross")

#         # version1, srflow
#         scale = torch.sigmoid(scale + 2.) + 1e-4
#         # scale = self.clamping(scale + 2.) + 1e-4
#         z2 = (z2 / scale) -shift

#         # version2, survae
#         # logscale = 2. * torch.tanh(scale / 2.)
#         # z2 = z2 * torch.exp(-logscale) - shift
        
#         logdet -= thops.sum(torch.log(scale), dim=[1, 2])
#         z = thops.cat_feature(z1, z2)
#         return z, logdet
    
#     # def channel_split(self, x):
#     #     return torch.chunk(x, 2, dim=self.dim)
    
#     # def channel_cat(self, z1, z2):
#     #     return torch.cat([z1, z2], dim=self.dim)

