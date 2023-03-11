import torch
import torch.nn as nn
from torch import Tensor

logabs = lambda x: torch.log(torch.abs(x))

# -----------------------------------------------------------------------------------------
class ActNorm(nn.Module):
    """ActNorm implementation for Point Cloud."""

    def __init__(self, channel: int, dim=1, use_exp=False, device='cuda'):
        super(ActNorm, self).__init__()

        assert dim in [-1, 1, 2]
        self.dim = 2 if dim == -1 else dim

        if self.dim == 1:
            self.logs = nn.Parameter(torch.zeros((1, channel, 1)))  # log sigma
            self.bias = nn.Parameter(torch.zeros((1, channel, 1)))
            self.Ndim = 2
        if self.dim == 2:
            # self.scale = nn.Parameter(torch.ones((1, 1, channel)))
            # self.bias = nn.Parameter(torch.zeros((1, 1, channel)))
            # self.logs = nn.Parameter(torch.zeros((1, 1, channel)))
            # self.loc = nn.Parameter(torch.ones((1, 1, channel)))
            size       = (1, 1, channel) #voxels, #channels
            # self.logs  = torch.nn.Parameter(torch.zeros(size, dtype=torch.float, device=device, requires_grad=True))
            # self.b     = torch.nn.Parameter(torch.zeros(size, dtype=torch.float, device=device, requires_grad=True))
            
            # self.loc = nn.Parameter(torch.zeros(size, device=device)) # BxCxDxHxW
            # self.scale = nn.Parameter(torch.ones(size, device=device))
            
            self.logs  = torch.nn.Parameter(torch.zeros(size,dtype=torch.float,device=device,requires_grad=True))
            self.b     = torch.nn.Parameter(torch.zeros(size,dtype=torch.float,device=device,requires_grad=True))
            self.Ndim = 1
        
        self.eps = 1e-6
        self.register_buffer("_initialized", torch.tensor(False).to(device))

    # def forward(self, x: Tensor, _: Tensor=None):
    #     if not self.is_inited:
    #         self.__initialize(x)

    #     z = x * torch.exp(self.logs) + self.bias
    #     logdet = torch.sum(self.logs) * x.shape[self.Ndim]
    #     return z, logdet
    
    def forward(self, x: Tensor, _: Tensor=None):
        """
            logs is log_std of `mean of channels`
            so we need to multiply by number of voxels
        """
        if not self._initialized:
            self.__initialize(x)
            self._initialized = torch.tensor(True)
        
        # z = x * torch.exp(self.logs) + self.bias
        # z = (x - self.bias) / self.scale
        # logdet = - self.scale.abs().log().sum() * x.shape[self.Ndim]
        # z = (x + self.bias) * torch.exp(self.logs)
        # logdet = torch.sum(self.logs) * x.shape[self.Ndim]
        z = (x + self.b) * torch.exp(self.logs)
        logdet = torch.sum(self.logs) * x.shape[self.Ndim]
        
        # s = torch.tanh(self.scale)
        # z = s * (x + self.loc)
        # log_abs = logabs(self.scale)
        # logdet = torch.sum(log_abs) * x.shape[self.Ndim]
        return z, logdet

    def inverse(self, z: Tensor, _: Tensor=None):
        # x = z * self.scale + self.bias
        # logdet = self.scale.abs().log().sum() * x.shape[self.Ndim]
        # x = (z - self.bias) * torch.exp(-self.logs)
        # x = z * torch.exp(-self.logs) - self.bias
        # logdet = -torch.sum(self.logs) * x.shape[self.Ndim]
        
        x = z * torch.exp(-self.logs) - self.b
        logdet = -torch.sum(self.logs) * x.shape[self.Ndim]
        
        # s = torch.tanh(self.scale)
        # x = z / s - self.loc
        # log_abs = logabs(self.scale)
        # logdet = -torch.sum(log_abs) * x.shape[self.Ndim]
        return x, logdet

    def __initialize(self, x: Tensor):
        if not self.training:
            return
        with torch.no_grad():
            dims = [0, 1, 2]
            dims.remove(self.dim)

            # bias = -torch.mean(x.detach(), dim=dims, keepdim=True)
            # logs = -torch.log(torch.std(x.detach(), dim=dims, keepdim=True) + self.eps)
            # var = torch.mean((x.detach() + bias) ** 2, dim=dims, keepdim=True)
            # logs = torch.log(self.scale / (torch.sqrt(var) + self.eps))
            # self.bias.data.copy_(bias.data)
            # self.logs.data.copy_(logs.data)
            
            # bias = torch.mean(x.detach(), dim=dims, keepdim=True)
            # scale = torch.std(x.detach(), dim=dims, keepdim=True) + self.eps
            # self.bias.data.copy_(bias.data)
            # self.scale.data.copy_(scale.data)
            # self.loc.data = torch.mean(x.detach(), dim=dims)
            
            b_    = torch.mean(x.clone(), dim=dims, keepdim=True)
            s_    = torch.mean((x.clone() - b_)**2, dim=dims, keepdim=True)
            b_    = -1 * b_
            logs_ = -1 * torch.log(torch.sqrt(s_) + self.eps)
            self.logs.data.copy_(logs_.data)
            self.b.data.copy_(b_.data)
            
            # mean = torch.mean(x.clone(), dim=dims, keepdim=True)
            # std = torch.std(x.clone(), dim=dims, keepdim=True)
            # self.loc.data.copy_(-mean)
            # self.scale.data.copy_(1 / (std + 1e-6))
            
            # if self.use_exp:
            #     self.scale.data = torch.log(torch.std(x, dim=dims) + self.eps)
            # else:
            #     self.scale.data = torch.log(torch.exp(torch.std(x, dim=dims)) + self.eps)
            
            # self.bias.squeeze().data.copy_(x.transpose(0,1).flatten(1).mean(1)).view_as(self.scale)
            # self.scale.squeeze().data.copy_(x.transpose(0,1).flatten(1).std(1, False) + 1e-6).view_as(self.bias)
            
# -----------------------------------------------------------------------------------------