import torch
import torch.nn as nn

import sys 
import os
sys.path.append("..")
from models.pointnet2.pointnet2_modules import PointnetSAModule, PointnetFPModule
# import models.pointnet2.pointnet2_utils as pointnet_utils
import models.pointnet2.pytorch_utils as pt_utils
from typing import List


def get_model(npoint=1024, up_ratio=2, latdim=4, use_normal=False, use_bn=False, use_res=False):
    return PUNet(npoint, up_ratio, use_normal, use_bn, use_res)

class PUNet(nn.Module):
    def __init__(self, npoint=1024, up_ratio=2, latdim=4, use_normal=False, use_bn=False, use_res=False, device='cpu'):
        super().__init__()

        self.npoint = npoint
        self.use_normal = use_normal
        self.up_ratio = up_ratio
        self.latdim = latdim

        self.npoints = [
            npoint, 
            npoint // 2, 
            npoint // 4, 
            npoint // 8
        ]

        mlps = [
            # [32, 32, 64],
            # [64, 64, 128],
            # [128, 128, 256],
            # [256, 256, 512]
            [16, 16, 32],
            [32, 32, 64],
            [64, 64, 128],
            [128, 128, 256]
        ]

        radius = [0.05, 0.1, 0.2, 0.3]

        nsamples = [32, 32, 32, 32]

        ## for 4 downsample layers
        # in_ch = 0 if not use_normal else 3
        in_ch = 7-3
        self.SA_modules = nn.ModuleList()
        for k in range(len(self.npoints)):
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=self.npoints[k],
                    radius=radius[k],
                    nsample=nsamples[k],
                    mlp=[in_ch] + mlps[k], # output : 32, 64, 128, 256...
                    use_xyz=True,
                    use_res=use_res,
                    bn=use_bn).to(device))
            in_ch = mlps[k][-1]

        ## upsamples for layer 2 ~ 4
        self.FP_Modules = nn.ModuleList()
        for k in range(len(self.npoints) - 1):
            self.FP_Modules.append(
                PointnetFPModule(
                    mlp=[mlps[k + 1][-1], 32],  # output : 32
                    bn=use_bn))
        
        ## feature Expansion
        in_ch = len(self.npoints) * 32 + 3 # 4 layers + input xyz
        self.FC_Modules = nn.ModuleList()
        for k in range(up_ratio):
            self.FC_Modules.append(
                pt_utils.SharedMLP(
                    [in_ch, 32, self.latdim],
                    bn=use_bn))

        ## coordinate & attribute reconstruction
        in_ch = self.latdim # ori: 128, default: 4,
        
        self.pcd_layer = nn.Sequential(
            pt_utils.SharedMLP([in_ch, 64], bn=use_bn),
            pt_utils.SharedMLP([64, 7], activation=None, bn=False)) 


    def forward(self, points, npoint=None):
        if npoint is None:
            npoints = [None] * len(self.npoints)
        else:
            npoints = []
            for k in range(len(self.npoints)):
                npoints.append(npoint // 2 ** k)
        
        ## points: bs, N, 3/6
        xyz = points[..., :3].contiguous()
        feats = points[..., 3:].transpose(1, 2).contiguous() # if self.use_normal else None
        
        ## downsample
        l_xyz, l_feats = [xyz], [feats]
        for k in range(len(self.SA_modules)):
            # print(l_xyz[k].device, l_feats[k].device, next(self.SA_modules[k].parameters()).device, npoints[k])
            lk_xyz, lk_feats = self.SA_modules[k](l_xyz[k], l_feats[k], npoint=npoints[k])
            l_xyz.append(lk_xyz)
            l_feats.append(lk_feats)

        ## upsample
        up_feats = []
        for k in range(len(self.FP_Modules)):
            upk_feats = self.FP_Modules[k](xyz, l_xyz[k + 2], None, l_feats[k + 2])
            up_feats.append(upk_feats)
        
        ## aggregation
        # [xyz, l0, l1, l2, l3] 3, 32, 32, 32, 32
        feats = torch.cat([
            xyz.transpose(1, 2).contiguous(),
            l_feats[1],
            *up_feats], dim=1).unsqueeze(-1) # bs, mid_ch, N, 1

        ## expansion or aggregation 
        r_feats = []
        for k in range(len(self.FC_Modules)):
            feat_k = self.FC_Modules[k](feats) # bs, mid_ch, N, 1
            r_feats.append(feat_k)
        
        r_feats = torch.cat(r_feats, dim=2) # bs, mid_ch, r * N, 1
        # print(r_feats.shape) # [1, 4 ,npts, 1]
        # import pdb
        # pdb.set_trace()
        ## reconstruction
        output = self.pcd_layer(r_feats) # bs, 3, r * N, 1
        return output.squeeze(-1).transpose(1, 2).contiguous(), r_feats # bs, 3, r * N


if __name__ == '__main__':
    model = PUNet(npoint=64, up_ratio=1, latdim=4).cuda()
    points = torch.randn([1, 64, 7]).float().cuda()
    output, lat = model(points)
    print(output.shape)
    print(lat.shape)
