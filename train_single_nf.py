import argparse
from ctypes import Structure
import random
import sys
import os
from datetime import datetime
import numpy as np
import struct
import scipy.ndimage
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from models.multiscale_glow.FlowNet_SR_x4 import FlowNet, Encoder_y2, FlowNet_FlowAssembly, Encoder_y2_V3_resnet

from dataset import get_test_dataloader, ScalarData, get_size
# from dataset import min_max_nyx, min_max_vortex, min_max_fpm, min_max_cos, min_max_jet3b
from utils.utils_base import init, Logger, load_checkpoint, save_checkpoint, AverageMeter, get_duration

from losses.losses import Metrics, ReconLoss
from dataset import denormalize_zscore, denormalize_max_min, read_data_from_file, data_reader

import pdb
# torch.manual_seed(0)

def parse_args(argv):
    parser = argparse.ArgumentParser(description='NormFlow')
    parser.add_argument('--train', action='store_true', help='Is true if is training.')
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--name', help='result dir name', default=datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), type=str)
    parser.add_argument('--resume_nf', help='NF model snapshot path', type=str)
    
    parser.add_argument('--seed', help='seed number', default=123, type=int)
    parser.add_argument('--sample_size', type=int, help='sample size')
    parser.add_argument('--sample_size_list', nargs='+', type=int, help='sample size list')
    parser.add_argument('--sample_type', help='sample type (single, multi)', default=128, type=str)
    
    parser.add_argument('--sample_loss', type=float, default=0, help='sample_loss (default: 0)')
    parser.add_argument('--cycle_loss', type=float, default=0, help='cycle_loss (default: 0)')
    parser.add_argument('--sf', dest='sf', type=int, default=4, help='scale factor for sr')
    
    args = parser.parse_args(argv)

    if not args.config:
        if args.resume:
            assert args.resume.startswith('./')
            dir_path = '/'.join(args.resume.split('/')[:-2])
            args.config = os.path.join(dir_path, 'config.yaml')
        else:
            args.config = './configs/config.yaml'
    return args


# def get_metric(dataname):
#     if dataname == 'fpm':
#         metric = Metrics(norm_method='min_max', min_max_norm=min_max_fpm)
#     elif dataname == 'cos':
#         metric = Metrics(norm_method='min_max', min_max_norm=min_max_cos)
#     elif dataname == 'jet3b':
#         metric = Metrics(norm_method='min_max', min_max_norm=min_max_jet3b)
#     elif dataname == 'vortex':
#         metric = Metrics(norm_method='min_max', min_max_norm=min_max_vortex)
#     elif dataname == 'nyx':
#         metric = Metrics(norm_method='min_max', min_max_norm=min_max_nyx)
#     return metric


def train(args, config, base_dir, snapshot_dir, output_dir, log_dir):
    device = torch.device("cuda" if config['cuda'] else "cpu") 
    # device = 'cpu'
    print('device', device)
    
    file_list = read_data_from_file(config['dataname'], config['data_path'])
    
    logger = Logger(config, base_dir, snapshot_dir, output_dir, log_dir)
    
    nfmodel = FlowNet(C=1, K=config['flows'], L=config['L'], 
                    splitOff=config['splitOff'], RRDB_nb=config['RRDB_nb'], RRDB_nf=config['RRDB_nf'], 
                    preK=config['preK'], factor=config['factor'], ks_s_pad=config['ks_s_pad']) 
    print(nfmodel)
    nfmodel = nfmodel.to(device)
    
    torch.autograd.set_detect_anomaly(True)
    nfmodel.train()
    
    params = list(nfmodel.parameters())
    optimizer = optim.Adam(params, lr=config['lr'])
    
    if args.resume_nf:
        itr, nfmodel = load_checkpoint(args.resume_nf, nfmodel, only_net=True)
        logger.load_itr(itr)
    
    if config['set_lr']:
        lr_prior = optimizer.param_groups[0]['lr']
        for g in optimizer.param_groups:
            g['lr'] = float(config['set_lr'])
        print(f'[set lr] {lr_prior} -> {optimizer.param_groups[0]["lr"]}')
    
    mse_best = 1e10
    while logger.itr < config['max_itr']:
        # print('logger.itr', logger.itr)
        epoch_loss = 0
        epoch_sample_loss = 0
        epoch_logpx_loss = 0
        epoch_cycle_loss = 0

        MAE = nn.L1Loss()
        MSE = nn.MSELoss()
        # MAE_none = nn.L1Loss(reduction='none')
        
        count = 0
        for ind_f, f in enumerate(file_list):
            # print(ind_f, "===> File in process: ", f)
            train_data, attr_min_max = data_reader(f, config['dataname'], padding=config['padding'])
            
            if args.sample_type == 'single':
                train_dataset = ScalarData(train_data, mode='train', scale_factor=args.sf, blocksize=config['blocksize'], padding=config['padding'], sampler=args.sample_size)
            elif args.sample_type == 'multi':
                if not isinstance(args.sample_size_list, list):
                    sample_size = [args.sample_size_list] * 5
                else:
                    sample_size = args.sample_size_list
                train_dataset = ScalarData(train_data, mode='train', scale_factor=args.sf, blocksize=config['blocksize'], padding=config['padding'], sampler=sample_size[0])
                for level in range(1, len(sample_size)):
                    ds_scale = np.power(args.sf, level)
                    train_data, attr_min_max = data_reader(f, config['dataname'], padding=config['padding'], scale=ds_scale)
                    train_dataset_2 = ScalarData(train_data, mode='train', scale_factor=args.sf, blocksize=config['blocksize'], padding=config['padding'], sampler=sample_size[level])
                    train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_2])
            
            train_dataloader = DataLoader(train_dataset, batch_size=config['batchsize'], shuffle=True, drop_last=False, num_workers=0)
            
            train_x = []
            for x, x_low in train_dataloader:
                x = x.to(device) 
                x_low = x_low.to(device) 

                # import pdb
                # pdb.set_trace()

                x_nf_z, logdet, z1, z2 = nfmodel(x)
                loss_x = (-logdet / (math.log(2) * x.shape[2])).mean()
                loss = loss_x
                epoch_logpx_loss += loss_x.item()

                if args.sample_loss > 0:
                    x_hat_sample = nfmodel.sample(z=x_nf_z, eps_std=0.8)
                    mae_dist = MSE(x, x_hat_sample)
                    # cham_dist, _ = f_chamfer(x, x_hat_sample, batch_reduction=None)
                    sample_loss = torch.mean(mae_dist)
                    loss += args.sample_loss*sample_loss 
                    epoch_sample_loss += args.sample_loss*sample_loss.item()
                
                if args.cycle_loss > 0: # actually is still sample loss
                    x_hat_nf = nfmodel(z=x_nf_z, eps_std=0, reverse=True, training=False, z1=None, z2=None)
                    cycle_loss = MSE(x, x_hat_nf)
                    loss += args.cycle_loss*cycle_loss 
                    epoch_cycle_loss += args.cycle_loss*cycle_loss.item()
                
                if torch.isnan(loss).detach().cpu().item():
                    loss.data = torch.ones_like(loss) * 0.1
                    print('Encounter Nan', loss, "===> File in process: ", f)
                
                optimizer.zero_grad()
                loss.backward()
                count += len(x)
                optimizer.step()

                epoch_loss += loss.item()
            
        epoch_loss /= count
        epoch_sample_loss /= count
        epoch_logpx_loss /= count
        epoch_cycle_loss /= count
        # log_dict
        logger.update({
            'loss': epoch_loss,
            'logp_x_loss': epoch_logpx_loss,
            'sample_loss': epoch_sample_loss,
            'cycle_loss': epoch_cycle_loss
        })
        if logger.itr % config['log_itr'] == 0:
            print('==> Epoch {:02d} average loss: {:.6f}'.format(logger.itr, epoch_loss))
            print('logp_x_loss..: ', epoch_logpx_loss) # .detach().cpu().item()
            if args.sample_loss > 0:
                print('sample_loss..: ', epoch_sample_loss)
            if args.cycle_loss > 0:
                print('cycle_loss..: ', epoch_cycle_loss)
            
            logger.write()
            logger.init()

        # test and save model snapshot
        if logger.itr % config['test_itr'] == 0 or logger.itr % config['snapshot_save_itr'] == 0:
            mse = test(args, config['dataname'], config['test_data_file_path'], logger, nfmodel, blocksize=config['blocksize'], padding=config['padding'], batchsize_test=config['batchsize_test'], save_latent=False)
            print('[test] loss, epoch_loss, loss_best: ', mse , epoch_loss, mse_best)
            
            if epoch_loss < mse_best:
                print('Best!')
                save_checkpoint(os.path.join(snapshot_dir, 'best_nf.pt'), logger.itr, nfmodel)
                mse_best = epoch_loss

            if logger.itr % config['snapshot_save_itr'] == 0:
                save_checkpoint(os.path.join(snapshot_dir, f'{logger.itr:05}_{mse:.8f}_nf.pt'),
                                logger.itr, nfmodel)
        
        # lr scheduling
        if logger.itr % config['lr_shedule_step'] == 0:
            lr_prior = optimizer.param_groups[0]['lr']
            for g in optimizer.param_groups:
                g['lr'] *= config['lr_shedule_scale']
            print(f'[lr scheduling] {lr_prior} -> {optimizer.param_groups[0]["lr"]}')

 
def PSNR_MSE(x, y, diff=1, istorch=False):
    if istorch:
        mse = torch.mean((x - y) ** 2, dim=[1, 2, 3, 4])
        psnr = 10 * torch.log10(self.diff ** 2. / mse)
    else:
        mse = np.mean((x - y) ** 2)
        psnr = 20 * np.log10(diff / np.sqrt(mse))
    return psnr, mse


def test(args, dataname, val_path, logger, nfmodel, blocksize=24, padding=4, batchsize_test=128, save_latent=False):
    nfmodel.eval()
    device = next(nfmodel.parameters()).device
    test_dataloader, attr_min_max, all_data_gt = get_test_dataloader(args, dataname, val_path, blocksize=blocksize, padding=padding, batch_size_test=batchsize_test, level=0)
    
    centers = []
    loss = 0
    cnt = 0
    act_size = blocksize - 2 * padding

    with torch.no_grad():
        logger.init()

        result_size = get_size(dataname, sf=args.sf, level=0)
        result = np.zeros((result_size))
        
        start = datetime.now()
        for x, x_low, z_ind, y_ind, x_ind in test_dataloader:
            x = x.to(device)
            x_low = x_low.to(device)
            
            x_nf_z, logdet, z1, z2 = nfmodel(x)
            
            loss_x = (-logdet / (math.log(2) * x.shape[2])).mean() 
            loss += loss_x.detach().cpu().item() 
            cnt += len(x)

            # import pdb
            # pdb.set_trace()
            # reconstruction and sample
            x_hat_nf = nfmodel(z=x_nf_z, eps_std=0, reverse=True, training=False, z1=None, z2=None)
            x_hat_nf = x_hat_nf.detach().cpu().numpy()
            for ii in range(len(z_ind)):
                if padding > 0:
                    result[int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += x_hat_nf[ii].reshape(blocksize, blocksize, blocksize)[padding:-padding, padding:-padding, padding:-padding]
                else:
                    result[int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += x_hat_nf[ii].reshape(blocksize, blocksize, blocksize)
        
        del x, x_low, z_ind, y_ind, x_ind
        h, m, s = get_duration(start, datetime.now())
        time_str = 'Test Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)  
        print(time_str)
        
        loss /= cnt 
        
        result = np.array(result)
        all_data_gt = np.array(all_data_gt)
        if padding > 0:
            all_data_gt = all_data_gt[padding:-padding, padding:-padding, padding:-padding]

        result = denormalize_max_min(result, attr_min_max).reshape(-1)
        all_data_gt = denormalize_max_min(all_data_gt, attr_min_max).reshape(-1)

        recon_psnr, recon_mse = PSNR_MSE(all_data_gt, result, attr_min_max['max'][0]-attr_min_max['min'][0])
        
        print('max_gt, max_recon', all_data_gt.shape, all_data_gt.shape, np.min(all_data_gt), np.max(all_data_gt), np.min(result), np.max(result))
        with open(os.path.join(logger.output_dir, dataname[:3]+f'_mse{recon_mse:.4f}_psnr{recon_psnr:.4f}_loss{loss:.4f}.bin'), 'wb') as f:
            f.write(struct.pack('<%df' % len(result), *result))
        
        with open(os.path.join(logger.output_dir, dataname[:3]+f'_gt.bin'), 'wb') as f:
            f.write(struct.pack('<%df' % len(all_data_gt), *all_data_gt))
        
        print(f'[ Test ]==>>> {dataname}, recon_mse{recon_mse:.4f}, recon_psnr{recon_psnr:.4f}, loss{loss:.4f}')
        
    logger.init()
    nfmodel.train()
    return recon_mse #+ loss


def main(argv):
    args = parse_args(argv)
    config, base_dir, snapshot_dir, output_dir, log_dir = init(args)
    if args.seed is not None:
        seed = args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU

    print('[PID]', os.getpid())
    print('[config]', args.config)
    msg = f'======================= {args.name} ======================='
    print(msg)
    for k, v in config.items():
        if k in {'lr', 'set_lr', 'p'}:
            print(f' *{k}: ', v)
        else:
            print(f'  {k}: ', v)
    print('=' * len(msg))
    print()
    
    if args.train:
        train(args, config, base_dir, snapshot_dir, output_dir, log_dir)
    else:
        # device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        device = torch.device("cuda" if config['cuda'] else "cpu") 
        val_path = config['test_data_file_path']
        logger = Logger(config, base_dir, snapshot_dir, output_dir, log_dir)
        
        nfmodel = FlowNet(C=1, K=config['flows'], L=config['L'], 
                            splitOff=config['splitOff'], RRDB_nb=config['RRDB_nb'], RRDB_nf=config['RRDB_nf'], 
                            preK=config['preK'], factor=config['factor'], ks_s_pad=config['ks_s_pad'])
        
        if args.resume_nf:
            itr, nfmodel = load_checkpoint(args.resume_nf, nfmodel, only_net=True)
        logger.load_itr(itr)
        
        nfmodel = nfmodel.to(device)
        nfmodel.eval()
        test(args, config['dataname'], val_path, logger, nfmodel, blocksize=config['blocksize'], padding=config['padding'], batchsize_test=config['batchsize_test'], save_latent=False)


if __name__ == '__main__':
    main(sys.argv[1:])

