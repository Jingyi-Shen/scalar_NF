import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch import nn

import numpy as np


# # Model
# class ParticleNF(nn.Module):
#     def __init__(self, args, input_dim, lat_dim):
#         # ParticleNF(input_dim=input_dim, lat_dim=config['lat_dim'])
#         super(ParticleNF, self).__init__()
#         self.input_dim = input_dim
#         self.lat_dim = lat_dim
#         self.use_latent_flow = args.use_latent_flow
#         self.use_deterministic_encoder = args.use_deterministic_encoder
#         self.prior_weight = args.prior_weight
#         self.recon_weight = args.recon_weight
#         self.entropy_weight = args.entropy_weight
#         self.distributed = args.distributed
#         self.truncate_std = None
#         self.encoder = Encoder(
#                 lat_dim=lat_dim, 
#                 input_dim=input_dim,
#                 use_deterministic_encoder=args.use_deterministic_encoder)
#         self.point_cnf  = get_point_cnf(args)
#         self.latent_cnf = get_latent_cnf(args) if args.use_latent_flow else nn.Sequential()

#     @staticmethod
#     def sample_gaussian(size, truncate_std=None, gpu=None):
#         y = torch.randn(*size).float()
#         y = y if gpu is None else y.cuda(gpu)
#         if truncate_std is not None:
#             truncated_normal(y, mean=0, std=1, trunc_std=truncate_std)
#         return y

#     @staticmethod
#     def reparameterize_gaussian(mean, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn(std.size()).to(mean)
#         return mean + std * eps

#     @staticmethod
#     def gaussian_entropy(logvar):
#         const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
#         ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
#         return ent

#     def multi_gpu_wrapper(self, f):
#         self.encoder = f(self.encoder)
#         self.point_cnf = f(self.point_cnf)
#         # self.latent_cnf = f(self.latent_cnf)

#     def make_optimizer(self, args):
#         def _get_opt_(params):
#             if args.optimizer == 'adam':
#                 optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2),
#                                       weight_decay=args.weight_decay)
#             elif args.optimizer == 'sgd':
#                 optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
#             else:
#                 assert 0, "args.optimizer should be either 'adam' or 'sgd'"
#             return optimizer
#         opt = _get_opt_(list(self.encoder.parameters()) + list(self.point_cnf.parameters())
#                         + list(list(self.latent_cnf.parameters())))
#         return opt

#     def forward(self, x, opt, step, writer=None):
#         opt.zero_grad()
#         batch_size = x.size(0)
#         num_points = x.size(1)
#         z_mu, z_sigma = self.encoder(x)
#         if self.use_deterministic_encoder:
#             z = z_mu + 0 * z_sigma
#         else:
#             z = self.reparameterize_gaussian(z_mu, z_sigma)

#         # Compute H[Q(z|X)]
#         if self.use_deterministic_encoder:
#             entropy = torch.zeros(batch_size).to(z)
#         else:
#             entropy = self.gaussian_entropy(z_sigma)

#         # Compute the prior probability P(z)
#         if self.use_latent_flow:
#             w, delta_log_pw = self.latent_cnf(z, None, torch.zeros(batch_size, 1).to(z))
#             log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(1, keepdim=True)
#             delta_log_pw = delta_log_pw.view(batch_size, 1)
#             log_pz = log_pw - delta_log_pw
#         else:
#             log_pz = torch.zeros(batch_size, 1).to(z)

#         # Compute the reconstruction likelihood P(X|z)
#         z_new = z.view(*z.size())
#         z_new = z_new + (log_pz * 0.).mean()
#         y, delta_log_py = self.point_cnf(x, z_new, torch.zeros(batch_size, num_points, 1).to(x))
#         log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
#         delta_log_py = delta_log_py.view(batch_size, num_points, 1).sum(1)
#         log_px = log_py - delta_log_py

#         # Loss
#         entropy_loss = -entropy.mean() * self.entropy_weight
#         recon_loss = -log_px.mean() * self.recon_weight
#         prior_loss = -log_pz.mean() * self.prior_weight
#         loss = entropy_loss + prior_loss + recon_loss
#         loss.backward()
#         opt.step()

#         # LOGGING (after the training)
#         if self.distributed:
#             entropy_log = reduce_tensor(entropy.mean())
#             recon = reduce_tensor(-log_px.mean())
#             prior = reduce_tensor(-log_pz.mean())
#         else:
#             entropy_log = entropy.mean()
#             recon = -log_px.mean()
#             prior = -log_pz.mean()

#         recon_nats = recon / float(x.size(1) * x.size(2))
#         prior_nats = prior / float(self.zdim)

#         if writer is not None:
#             writer.add_scalar('train/entropy', entropy_log, step)
#             writer.add_scalar('train/prior', prior, step)
#             writer.add_scalar('train/prior(nats)', prior_nats, step)
#             writer.add_scalar('train/recon', recon, step)
#             writer.add_scalar('train/recon(nats)', recon_nats, step)

#         return {
#             'entropy': entropy_log.cpu().detach().item()
#             if not isinstance(entropy_log, float) else entropy_log,
#             'prior_nats': prior_nats,
#             'recon_nats': recon_nats,
#         }

#     def encode(self, x):
#         z_mu, z_sigma = self.encoder(x)
#         if self.use_deterministic_encoder:
#             return z_mu
#         else:
#             return self.reparameterize_gaussian(z_mu, z_sigma)

#     def decode(self, z, num_points, truncate_std=None):
#         # transform points from the prior to a point cloud, conditioned on a shape code
#         y = self.sample_gaussian((z.size(0), num_points, self.input_dim), truncate_std)
#         x = self.point_cnf(y, z, reverse=True).view(*y.size())
#         return y, x

#     def sample(self, batch_size, num_points, truncate_std=None, truncate_std_latent=None, gpu=None):
#         assert self.use_latent_flow, "Sampling requires `self.use_latent_flow` to be True."
#         # Generate the shape code from the prior
#         w = self.sample_gaussian((batch_size, self.zdim), truncate_std_latent, gpu=gpu)
#         z = self.latent_cnf(w, None, reverse=True).view(*w.size())
#         # Sample points conditioned on the shape code
#         y = self.sample_gaussian((batch_size, num_points, self.input_dim), truncate_std, gpu=gpu)
#         x = self.point_cnf(y, z, reverse=True).view(*y.size())
#         return z, x

#     def reconstruct(self, x, num_points=None, truncate_std=None):
#         num_points = x.size(1) if num_points is None else num_points
#         z = self.encode(x)
#         _, x = self.decode(z, num_points, truncate_std)
#         return x
        
# # ------------------------------------------------------------------------------------------------------------------------
# def get_hop_distance(num_node, edge, max_hop=1):
#     A = np.zeros((num_node, num_node))
#     max_hop = int(max_hop)
#     for i, j in edge:
#         A[j, i] = 1
#         A[i, j] = 1

#     hop_dis = np.zeros((num_node, num_node)) + np.inf
#     transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
#     arrive_mat = (np.stack(transfer_mat) > 0)
#     for d in range(max_hop, -1, -1):
#         hop_dis[arrive_mat[d]] = d
#     return hop_dis

# def normalize_digraph(A):
#     Dl = np.sum(A, 0)
#     num_node = A.shape[0]
#     Dn = np.zeros((num_node, num_node))
#     for i in range(num_node):
#         if Dl[i] > 0:
#             Dn[i, i] = Dl[i]**(-1)
#     AD = np.dot(A, Dn)
#     return AD

# def normalize_undigraph(A):
#     Dl = np.sum(A, 0)
#     num_node = A.shape[0]
#     Dn = np.zeros((num_node, num_node))
#     for i in range(num_node):
#         if Dl[i] > 0:
#             Dn[i, i] = Dl[i]**(-0.5)
#     DAD = np.dot(np.dot(Dn, A), Dn)
#     return DAD


# class Graph:
#     def __init__(self, 
#                  layout='locomotion',
#                  scale:int=1):
#         self.scale = int(scale)
#         self.get_edge(layout)
#         self.hop_dis = get_hop_distance(self.num_node, self.edge, self.scale) 
#         self.get_adjacency()

#     def get_edge(self, layout):
#         if layout == 'locomotion':
#             self.num_node = 21
#             self_link = [(i, i) for i in range(self.num_node)]
#             neighbor_link = [(1, 0), (2, 1), (3, 2), (4, 3), 
#                              (5, 0), (6, 5), (7, 6), (8, 7), 
#                              (9, 0), (10, 9), (11, 10), (12, 11), 
#                              (13, 11), (14, 13), (15, 14), (16, 15), 
#                              (17, 11), (18, 17), (19, 18), (20, 19)]
#             self.edge = self_link + neighbor_link
#             self.center = 10
#         else:
#             raise ValueError('This layout is not supported!')
    
#     def get_adjacency(self):
#         valid_hop = range(0, self.scale+1, 1)
#         adjacency = np.zeros((self.num_node, self.num_node))
#         for hop in valid_hop:
#             adjacency[self.hop_dis == hop] = 1
#         normalize_adjacency = normalize_digraph(adjacency)
        
#         A = []
#         for hop in valid_hop:
#             a_root = np.zeros((self.num_node, self.num_node))
#             a_close = np.zeros((self.num_node, self.num_node))
#             a_further = np.zeros((self.num_node, self.num_node))
#             for i in range(self.num_node):
#                 for j in range(self.num_node):
#                     if self.hop_dis[j, i] == hop:
#                         if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
#                             a_root[j, i] = normalize_adjacency[j, i]
#                         elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
#                             a_close[j, i] = normalize_adjacency[j, i]
#                         else:
#                             a_further[j, i] = normalize_adjacency[j, i]
#             if hop == 0:
#                 A.append(a_root)
#             else:
#                 A.append(a_root + a_close)
#                 A.append(a_further)
#         self.A = np.stack(A)


# class LinearZeros(nn.Linear):
#     def __init__(self, in_channels, out_channels):
#         super().__init__(in_channels, out_channels)
#         self.weight.data.zero_()
#         self.bias.data.zero_()
        

# class Conv2dZeros(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, 
#                  kernel_size: int=3, 
#                  stride: int=1, 
#                  padding='same',
#                  logscale_factor=3):
#         kernel_size = [kernel_size, kernel_size]
#         stride = [stride, stride]
#         pad_dict = {
#             "same": lambda kernel, stride: [((k - 1) * s + 1) // 2 for k, s in zip(kernel, stride)],
#             "valid": lambda kernel, stride: [0 for _ in kernel]
#         }
#         padding = pad_dict[padding](kernel_size, stride)
        
#         super().__init__(in_channels, out_channels, kernel_size, stride, padding)
#         self.logscale_factor = logscale_factor
#         self.logs = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
#         self.weight.data.zero_()
#         self.bias.data.zero_()
        
#     def forward(self, x):
#         out = super().forward(x)
#         return out * torch.exp(self.logs * self.logscale_factor)


# class GraphConvolution(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size,
#                  t_kernel_size=1,
#                  t_stride=1,
#                  t_padding=0,
#                  t_dilation=1,
#                  bias=True):
#         super().__init__()
        
#         self.kernel_size = kernel_size
#         self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, 
#                               kernel_size=(t_kernel_size, 1), 
#                               padding=(t_padding, 0), 
#                               stride=(t_stride, 1), 
#                               dilation=(t_dilation, 1),
#                               bias=bias)
    
#     def forward(self, x, A):
#         x = self.conv(x)
#         n, kc, t, v = x.size()
#         x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
#         x = torch.einsum('nkctv,kvw->nctw', (x, A))

#         return x.contiguous()


# class st_gcn(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size,
#                  stride=1, dropout=0.0, residual=False, cond_res=False):
#         super().__init__()
#         assert len(kernel_size) == 2
#         assert kernel_size[0] % 2 == 1
        
#         padding = ((kernel_size[0] - 1) // 2, 0)
#         self.gcn = GraphConvolution(in_channels, out_channels, kernel_size[1])
#         self.tcn = nn.Sequential(nn.BatchNorm2d(out_channels),
#                                  nn.ReLU(inplace=True),
#                                  nn.Conv2d(out_channels,
#                                           out_channels,
#                                           (kernel_size[0], 1),
#                                           (stride, 1),
#                                           padding),
#                                  nn.BatchNorm2d(out_channels),
#                                  nn.Dropout(dropout, inplace=True)
#                                  )
        
#         if not residual:
#             self.resudial = lambda x: 0
#         elif (in_channels == out_channels) and (stride == 1):
#             self.residual = lambda x: x
#         else:
#             self.residual = nn.Sequential(nn.Conv2d(in_channels,
#                                                     out_channels,
#                                                     kernel_size=1,
#                                                     stride=(stride, 1)),
#                                         #   nn.BatchNorm2d(out_channels)
#                                           )
#         self.relu = nn.ReLU(inplace=True)
#         self.cond_res = cond_res
#         self.cond_residual = nn.Sequential(nn.Conv1d(in_channels,
#                                                     out_channels,
#                                                     kernel_size=1,
#                                                     stride=stride),
#                                         #   nn.BatchNorm1d(out_channels)
#                                           )
    
#     def forward(self, x, A):
#         # res = self.residual(x)
#         x = self.gcn(x, A)
#         x = self.tcn(x) #+ res
        
#         # if self.cond_res:
#         #     cond_res = self.cond_residual(cond).unsqueeze(-1).repeat(1, 1, 1, 21)
#         #     x = x + cond_res
#         return self.relu(x)


# class STGCN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, 
#                  layout='locomotion', graph_scale=1,
#                  edge_importance_weighting=True):
#         super().__init__()
        
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
        
#         graph = Graph(layout=layout, scale=graph_scale)
#         A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        
#         self.register_buffer('A', A)
        
#         spatial_kernel_size = A.size(0)
#         temporal_kernel_size = 9
        
#         kernel_size = (temporal_kernel_size, spatial_kernel_size)
        
#         self.gcn_networks = nn.ModuleList((
#             st_gcn(input_dim, hidden_dim, kernel_size, cond_res=True),
#             st_gcn(hidden_dim, hidden_dim, kernel_size)
#         ))
    
#         if edge_importance_weighting:
#             self.edge_importance = nn.ParameterList([
#                 nn.Parameter(torch.ones(A.size()))
#                 for i in self.gcn_networks
#             ])
#         else:
#             self.edge_importance = [1] * len(self.gcn_networks)
            
#         self.fcn = Conv2dZeros(hidden_dim, output_dim)
        
#     def forward(self, x):
        
#         N, C, V, T = x.size() # input x: N, C, V, T
#         x = x.permute(0, 1, 3, 2).contiguous() # N, C, T, V

#         for gcn, importance in zip(self.gcn_networks, self.edge_importance):
#             x = gcn(x, self.A * importance)

#         y = self.fcn(x).permute(0, 1, 3, 2)
        
#         return y


# def split_feature(tensor, type="split"):
#     """
#     type = ["split", "cross"]
#     """
#     C = tensor.size(1)
#     if type == "split":
#         return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
#     elif type == "cross3":
#         return tensor[:, 0::3, ...], tensor[:, 1::3, ...], tensor[:, 2::3, ...]
#     elif type == "cross":
#         return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


# class ActNorm(nn.Module):
#     def __init__(self, num_channels:int, scale:float=1.0):
#         super().__init__()
#         size = [1, num_channels, 1, 1] # N, C, V, T
#         self.num_channels = num_channels
#         self.inited = False
#         self.scale = scale
#         self.bias = nn.Parameter(torch.zeros(*size))
#         self.logs = nn.Parameter(torch.zeros(*size))
        
#     def _check_input_dim(self, x):
#         assert len(x.size()) == 4
#         assert x.size(1) == self.num_channels
    
#     def initialize_parameters(self, x):
#         if not self.training:
#             return
#         assert x.device == self.bias.device
#         self.initialize_bias(x)
#         self.initialize_logs(x)
#         self.inited = True
    
#     def initialize_bias(self, x):
#         with torch.no_grad():
#             bias = torch.mean(x, dim=[0, 2, 3], keepdim=True) * -1.0
#             self.bias.data.copy_(bias.data)
        
#     def initialize_logs(self, x):
#         with torch.no_grad():
#             vars = torch.mean((x+self.bias) ** 2, dim=[0, 2, 3], keepdim=True)
#             logs = torch.log(self.scale/(torch.sqrt(vars)+1e-6))
#             # std = torch.std(x, dim=[0, 2, 3], keepdim=True)
#             # scale = self.scale / (std+1e-6)
#             # logs = torch.log(scale)
#             self.logs.data.copy_(logs.data)
    
#     def _scale(self, x, logdet=None, reverse=False):
#         logs = self.logs
#         if not reverse:
#             y = x * torch.exp(logs)
#         else:
#             y = x * torch.exp(-logs)
        
#         if logdet is not None:
#             num_pixels = x.size(2) * x.size(3)
#             dlogdet = torch.sum(logs) * num_pixels
#             if reverse:
#                 dlogdet *= -1
#             logdet = logdet + dlogdet
            
#         return y, logdet

#     def _center(self, x, reverse=False):
#         if not reverse:
#             return x + self.bias
#         else:
#             return x - self.bias
    
#     def forward(self, x, logdet=None, reverse=False):
#         if not self.inited:
#             self.initialize_parameters(x)
#         self._check_input_dim(x)
        
#         if not reverse:
#             x = self._center(x, reverse)
#             y, logdet = self._scale(x, logdet, reverse)
#             return y, logdet
#         else:
#             x, logdet = self._scale(x, logdet, reverse)
#             y = self._center(x, reverse)
#             return y


# class InvertibleConv1x1(nn.Module):
#     def __init__(self, num_channels, LU_decomposed=True):
#         super().__init__()
#         self.num_channels = num_channels
#         self.LU_decomposed = LU_decomposed
#         weight_shape = [num_channels, num_channels]
#         weight, _ = torch.qr(torch.randn(*weight_shape))
        
#         if not self.LU_decomposed:
#             self.weight = nn.Parameter(weight)
#         else:
#             weight_lu, pivots = torch.lu(weight)
#             w_p, w_l, w_u = torch.lu_unpack(weight_lu, pivots)
#             w_s = torch.diag(w_u)
#             sign_s = torch.sign(w_s)
#             log_s = torch.log(torch.abs(w_s))
#             w_u = torch.triu(w_u, 1)
            
#             u_mask = torch.triu(torch.ones_like(w_u), 1)
#             l_mask = u_mask.T.contiguous()
#             eye = torch.eye(l_mask.shape[0])
            
#             self.register_buffer('p', w_p)
#             self.register_buffer('sign_s', sign_s)
#             self.register_buffer('eye', eye)
#             self.register_buffer('u_mask', u_mask)
#             self.register_buffer('l_mask', l_mask)
#             self.l = nn.Parameter(w_l)
#             self.u = nn.Parameter(w_u)
#             self.log_s = nn.Parameter(log_s)
    
#     def forward(self, x, logdet=None, reverse=False):
#         num_pixels = x.size(2) * x.size(3)
#         if not self.LU_decomposed:
#             dlogdet = torch.slogdet(self.weight)[1] * num_pixels
#             if not reverse:
#                 weight = self.weight.unsqueeze(2).unsqueeze(3)
#                 y = F.conv1d(x, weight)
#                 if logdet is not None:
#                     logdet = logdet + dlogdet
#                 return y, logdet
#             else:
#                 weight = torch.inverse(self.weight.double()).float().unsqueeze(2).unsqueeze(3)
#                 y = F.conv1d(x, weight)
#                 if logdet is not None:
#                     logdet = logdet - dlogdet
#                 return y
#         else:
#             l = self.l * self.l_mask + self.eye
#             u = self.u * self.u_mask + torch.diag(self.sign_s * torch.exp(self.log_s))
#             dlogdet = torch.sum(self.log_s) * num_pixels
#             if not reverse:
#                 weight = torch.matmul(self.p, torch.matmul(l, u)).unsqueeze(2).unsqueeze(3)
#                 y = F.conv2d(x, weight)
#                 if logdet is not None:
#                     logdet = logdet + dlogdet
#                 else:
#                     logdet = dlogdet
#                 return y, logdet
#             else:
#                 l = torch.inverse(l.double()).float()
#                 u = torch.inverse(u.double()).float()
#                 weight = torch.matmul(u, torch.matmul(l, self.p.inverse())).unsqueeze(2).unsqueeze(3)
#                 y = F.conv2d(x, weight)
#                 if logdet is not None:
#                     logdet = logdet - dlogdet
#                 else:
#                     logdet = dlogdet
#                 return y


# class Permute(nn.Module):
#     def __init__(self, num_channels, shuffle=False):
#         super().__init__()
#         self.num_channels = num_channels
#         self.indices = np.arange(self.num_channels - 1, -1,-1).astype(np.long)
#         self.indices_inverse = np.zeros((self.num_channels), dtype=np.long)
#         if shuffle:
#             np.random.shuffle(self.indices)
#         for i in range(self.num_channels):
#             self.indices_inverse[self.indices[i]] = i

#     def forward(self, x, reverse=False):
#         assert len(x.size()) == 4
#         if not reverse:
#             return x[:, self.indices, :, :]
#         else:
#             return x[:, self.indices_inverse, :, :]
        

# class AffineCoupling(nn.Module):
#     def __init__(self, in_channels, hidden_size=512, net_type='gcn', graph_scale=1.0, layout='locomotion', affine=True):
#         super().__init__()
#         self.affine = affine
#         self.net_type = net_type
#         if net_type == 'gcn':
#             self.net = STGCN(input_dim=in_channels//2,
#                           hidden_dim=hidden_size,
#                           output_dim=2*(in_channels-in_channels//2),
#                           layout=layout,
#                           graph_scale=graph_scale)

#     def forward(self, x, logdet=None, reverse=False):
#         if not reverse:
#             # import pdb
#             # pdb.set_trace()
#             # cond1, cond2 = split_feature(cond, 'split')

#             # step 1
#             x1, x2 = split_feature(x, 'split')
#             h = x1 #self.net(x1)
#             shift, scale = split_feature(h, "cross")
#             scale = torch.sigmoid(scale + 2.) + 1e-6
#             x2 = (x2 + shift) * scale
#             y = torch.cat([x1, x2], 1)
            
#             # step 2
#             # x1, x2 = split_feature(y, 'split')
#             h = x2 #self.net(x2)
#             shift, scale = split_feature(h, "cross")
#             scale = torch.sigmoid(scale + 2.) + 1e-6
#             x1 = (x1 + shift) * scale
#             y = torch.cat([x1, x2], 1)
            
#             logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
#             return y, logdet
        
#         else:
#             # cond1, cond2 = split_feature(cond, 'split')
            
#             # step 1
#             x1, x2 = split_feature(x, "split")
#             h = self.net(x2)
#             shift, scale = split_feature(h, "cross")
#             scale = torch.sigmoid(scale + 2.) + 1e-6
#             x1 = x1 / scale - shift
#             # x = torch.cat([x1, x2], 1)
            
#             # step 2
#             # x1, x2 = split_feature(x, "split")
#             h = self.net(x1)
#             shift, scale = split_feature(h, "cross")
#             scale = torch.sigmoid(scale + 2.) + 1e-6
#             x2 = x2 / scale - shift
#             y = torch.cat([x1, x2], 1)
            
#             return y


import torch
import torch.nn as nn
import torch.nn.functional as F


class ActNorm(nn.Module):
    """
    https: // arxiv.org / pdf / 1807.03039.pdf
    """
    def __init__(self, x_dim, use_exp=False):
        super(ActNorm, self).__init__()
        self.loc = nn.Parameter(torch.ones(x_dim))
        self.scale = nn.Parameter(torch.ones(x_dim))
        self.register_buffer("initialised", torch.tensor(False))
        self.use_exp = use_exp

    def inverse(self, z: torch.tensor) -> (torch.tensor, torch.tensor):
        if self.initialised == False:
            self.loc.data = torch.mean(z, dim=0)
            if self.use_exp:
                self.scale.data = torch.log(torch.std(z, dim=0))
            else:
                self.scale.data = torch.log(torch.exp(torch.std(z, dim=0)) - 1.0)
            self.initialised.data = torch.tensor(True).to(self.loc.device)
        if self.use_exp:
            out = (z - self.loc) / torch.exp(self.scale)
            log_det = -torch.sum(self.scale)
        else:
            s = torch.nn.functional.softplus(self.scale)
            out = (z - self.loc) / s
            log_det = -torch.sum(torch.log(s))
        return out, log_det


    def forward(self, x: torch.tensor) -> (torch.tensor, torch.tensor):
        if self.use_exp:
            out = x*torch.exp(self.scale) + self.loc
            log_det = torch.sum(self.scale)
        else:
            s = torch.nn.functional.softplus(self.scale)
            out = x*s + self.loc
            log_det = torch.sum(torch.log(s))
        return out, log_det



if __name__ == '__main__':
    x = torch.randn(10, 2)
    actnorm = ActNorm(2)
    z, log_det = actnorm.inverse(x)
    print(torch.std(x, dim=0), torch.mean(x, dim=0))
    assert actnorm.initialised == True
    x_, log_det_ = actnorm(z)


