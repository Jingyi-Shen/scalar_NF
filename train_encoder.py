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
    parser.add_argument('--resume_ae', help='NF_y2 ae model snapshot path', type=str)
    
    parser.add_argument('--seed', help='seed number', default=123, type=int)
    
    parser.add_argument('--sample_size', type=int, help='sample size')
    parser.add_argument('--sample_size_list', nargs='+', type=int, help='sample size list')
    parser.add_argument('--sample_type', help='sample type from [random, uniform]', default='random', type=str)
    
    parser.add_argument("--cycle_loss", type=float, default=0, help="invertible cycle consistent loss (default: 0)")
    parser.add_argument("--cycle_recon_loss", type=float, default=0, help="invertible cycle consistent loss (default: 0)")
    parser.add_argument('--sample_loss', type=float, default=0, help='sample_loss (default: 0)')
    parser.add_argument('--latent_loss', type=float, default=0, help='latent_loss in latent space (default: 0)')
    parser.add_argument('--ae_recon_loss', type=float, default=0, help='ae_recon_loss for low res data (default: 0)')
    parser.add_argument('--feature_loss', type=float, default=0, help='feature_loss for reconstructed high res data (default: 0)')

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



def train_encoder_fixed_nf(args, config, base_dir, snapshot_dir, output_dir, log_dir, dim='2D'):
    device = torch.device("cuda" if config['cuda'] else "cpu") 
    # device = 'cpu'
    print('device', device)
    
    file_list = read_data_from_file(config['dataname'], config['data_path'])
    
    logger = Logger(config, base_dir, snapshot_dir, output_dir, log_dir)
    
    nfmodel = FlowNet(C=1, K=config['flows'], L=config['L'], 
                    splitOff=config['splitOff'], RRDB_nb=config['RRDB_nb'], RRDB_nf=config['RRDB_nf'], 
                    preK=config['preK'], factor=config['factor'], ks_s_pad=config['ks_s_pad']) 
    
    # aemodel_y2 = Encoder_y2(inchannel=1, outchannel=4, hidden_channel=16)
    aemodel_y2 = Encoder_y2_V3_resnet(inchannel=1, outchannel=4, hidden_channel=8)

    print(nfmodel)
    print(aemodel_y2)

    nfmodel = nfmodel.to(device)
    aemodel_y2 = aemodel_y2.to(device)
    
    torch.autograd.set_detect_anomaly(True)
    params = list(aemodel_y2.parameters())
    optimizer = optim.Adam(params, lr=config['lr'], weight_decay=1e-4)
    
    if args.resume_nf:
        itr, nfmodel = load_checkpoint(args.resume_nf, nfmodel, only_net=True)
    if args.resume_ae:
        itr, aemodel_y2 = load_checkpoint(args.resume_ae, aemodel_y2, only_net=True)
    
    # NF model is fixed
    aemodel_y2.train()
    nfmodel.eval()

    mse_best = 1e10
    while logger.itr < config['max_itr']:
        # print('logger.itr', logger.itr)
        epoch_loss = 0
        epoch_cycle_loss = 0
        epoch_latent_loss = 0
        
        MAE = nn.L1Loss()
        MSE = nn.MSELoss()
        MAE_none = nn.L1Loss(reduction='none')
        MSE_none = nn.MSELoss(reduction='none')
        
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
            
            for x, x_low in train_dataloader:
                x = x.to(device) 
                x_low = x_low.to(device) 
                
                x_nf_z, logdet, z1, z2 = nfmodel(x)
                y2_ae_z = aemodel_y2(x=x_low)

                loss = torch.zeros(1).to(device)
                # import pdb
                # pdb.set_trace()

                if args.cycle_loss > 0:
                    x_hat_sr = nfmodel(z=y2_ae_z, eps_std=0, reverse=True, training=False, z1=z1, z2=z2)
                    # x_hat_sr = nfmodel(z=z, eps_std=0, reverse=True, training=False, z1=z1, z2=z2, x_low=x_low)
                    # cycle_loss = chamfer_distance_attr_mae_loss(x_hat_sr.permute(0,2,1).contiguous(), x.permute(0,2,1).contiguous())
                    cycle_loss = MSE(x, x_hat_sr)
                    loss += args.cycle_loss*cycle_loss 
                    epoch_cycle_loss += args.cycle_loss*cycle_loss.item()
                
                if args.latent_loss > 0:
                    latent_loss = MAE(x_nf_z, y2_ae_z)
                    loss += args.latent_loss*latent_loss
                    epoch_latent_loss += args.latent_loss*latent_loss.item()
                
                if torch.isnan(loss).detach().cpu().item():
                    loss.data = torch.ones_like(loss) * 0.1
                    print('Encounter Nan', loss)
                    print(ind_f, "===> File in process: ", f)
                
                optimizer.zero_grad()
                loss.backward()
                count += len(x)
                optimizer.step()
                
                epoch_loss += loss.item()
            # print("===> File processed: {}/{}".format(i+1, len(file_list)))
        
        epoch_loss /= count
        epoch_cycle_loss /= count
        epoch_latent_loss /= count
        
        # logging
        logger.update({
            'loss': epoch_loss,
            'cycle_loss': epoch_cycle_loss,
            'latent_loss': epoch_latent_loss
        })
        logger.update({'loss': epoch_loss})
        if logger.itr % config['log_itr'] == 0:
            print('==> Epoch {:02d} average loss: {:.6f}'.format(logger.itr, epoch_loss))
            if args.latent_loss > 0:
                print('latent_loss..: ', epoch_latent_loss)
            if args.cycle_loss > 0:
                print('cycle_loss..: ', epoch_cycle_loss)
            
            logger.write()
            logger.init()

        # test and save model snapshot
        if logger.itr % config['test_itr'] == 0 or logger.itr % config['snapshot_save_itr'] == 0:
            mse = test(args, config['dataname'], config['test_data_file_path'], logger, nfmodel, aemodel_y2, blocksize=config['blocksize'], padding=config['padding'], batchsize_test=config['batchsize_test'], save_latent=False)
            print('[test] loss, epoch_loss, loss_best: ', mse, epoch_loss, mse_best)
            
            if epoch_loss < mse_best:
                print('Best!')
                save_checkpoint(os.path.join(snapshot_dir, 'best_nf.pt'), logger.itr, nfmodel)
                save_checkpoint(os.path.join(snapshot_dir, 'best_aey2.pt'), logger.itr, aemodel_y2)
                mse_best = epoch_loss

            if logger.itr % config['snapshot_save_itr'] == 0:
                save_checkpoint(os.path.join(snapshot_dir, f'{logger.itr:05}_{mse:.8f}_aey2.pt'),
                                logger.itr, aemodel_y2)
        
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


def test(args, dataname, val_path, logger, model, model_y2, blocksize=24, padding=4, batchsize_test=128, save_latent=False):
    model.eval()
    model_y2.eval()
    device = next(model.parameters()).device
    test_dataloader, attr_min_max, all_data_gt = get_test_dataloader(args, dataname, val_path, blocksize=blocksize, padding=padding, batch_size_test=batchsize_test, level=0)

    loss = 0
    cnt = 0
    act_size = blocksize - 2 * padding
    act_size_low = act_size // args.sf
    blocksize_low = blocksize // args.sf
    padding_low = padding // args.sf

    with torch.no_grad():
        logger.init()
        
        result_size = get_size(dataname, sf=args.sf, level=0)
        result = np.zeros((result_size))
        result_low_res = np.zeros(( np.array(result_size) // args.sf))
        
        start = datetime.now()
        for x, x_low, z_ind, y_ind, x_ind in test_dataloader:    
            x = x.to(device)
            x_low = x_low.to(device)
            
            x_nf_z, logdet, z1, z2 = model(x)
            y2_z = model_y2(x=x_low)

            loss_x = (-logdet / (math.log(2) * x.shape[2])).mean() 
            loss = loss_x.detach().cpu().item()
            cnt += len(x)
            
            x_hat_sr = model(z=y2_z, eps_std=0, reverse=True, training=False, z1=None, z2=None)
            # x_hat_nf = model(z=x_nf_z, eps_std=0, reverse=True, training=False, z1=None, z2=None)
            x_hat_sr = x_hat_sr.detach().cpu().numpy()
            x_low = x_low.detach().cpu().numpy()
            for ii in range(len(z_ind)):
                if padding > 0:
                    result[int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += x_hat_sr[ii].reshape(blocksize, blocksize, blocksize)[padding:-padding, padding:-padding, padding:-padding]
                    result_low_res[int(z_ind[ii])*act_size_low:int(z_ind[ii])*act_size_low+act_size_low, int(y_ind[ii])*act_size_low:int(y_ind[ii])*act_size_low+act_size_low, int(x_ind[ii])*act_size_low:int(x_ind[ii])*act_size_low+act_size_low] += x_low[ii].reshape(blocksize_low, blocksize_low, blocksize_low)[padding_low:-padding_low, padding_low:-padding_low, padding_low:-padding_low]
                else:
                    result[int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += x_hat_sr[ii].reshape(blocksize, blocksize, blocksize)
                    result_low_res[int(z_ind[ii])*act_size_low:int(z_ind[ii])*act_size_low+act_size_low, int(y_ind[ii])*act_size_low:int(y_ind[ii])*act_size_low+act_size_low, int(x_ind[ii])*act_size_low:int(x_ind[ii])*act_size_low+act_size_low] += x_low[ii].reshape(blocksize_low, blocksize_low, blocksize_low)
                
        del x, x_low, z_ind, y_ind, x_ind
        h, m, s = get_duration(start, datetime.now())
        time_str = 'Test Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)  
        print(time_str)
        
        loss /= cnt 
        
        result = np.array(result)
        all_data_gt = np.array(all_data_gt)
        if padding > 0:
            all_data_gt = all_data_gt[padding:-padding, padding:-padding, padding:-padding]
        
        # # add lerp to redisual result
        # lerp_result = scipy.ndimage.zoom(result_low_res, [args.sf, args.sf, args.sf], order=3)
        # result = lerp_result + result
        
        result = denormalize_max_min(result, attr_min_max).reshape(-1)
        all_data_gt = denormalize_max_min(all_data_gt, attr_min_max).reshape(-1)
        result_low_res = denormalize_max_min(result_low_res, attr_min_max).reshape(-1)

        recon_psnr, recon_mse = PSNR_MSE(all_data_gt, result, attr_min_max['max'][0]-attr_min_max['min'][0])
        
        print('max_gt, max_recon', all_data_gt.shape, all_data_gt.shape, np.min(all_data_gt), np.max(all_data_gt), np.min(result), np.max(result))
        with open(os.path.join(logger.output_dir, dataname[:3]+f'_mse{recon_mse:.4f}_psnr{recon_psnr:.4f}_loss{loss:.4f}.bin'), 'wb') as f:
            f.write(struct.pack('<%df' % len(result), *result))
        
        if not os.path.isfile(os.path.join(logger.output_dir, dataname[:3]+f'_gt.bin')):
            with open(os.path.join(logger.output_dir, dataname[:3]+f'_gt.bin'), 'wb') as f:
                f.write(struct.pack('<%df' % len(all_data_gt), *all_data_gt))
        
        # with open(os.path.join(logger.output_dir, dataname[:3]+f'_low_res_mse{recon_mse:.4f}.bin'), 'wb') as f:
        #     f.write(struct.pack('<%df' % len(result_low_res), *result_low_res))
        
        print(f'[ Test ]==>>> {dataname}, recon_mse{recon_mse:.4f}, recon_psnr{recon_psnr:.4f}, loss{loss:.4f}')
        
    logger.init()
    model_y2.train()
    model.eval()
    return recon_mse


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
        train_encoder_fixed_nf(args, config, base_dir, snapshot_dir, output_dir, log_dir)
    else:
        # device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        device = torch.device("cuda" if config['cuda'] else "cpu") 
        nfmodel = FlowNet(C=1, K=config['flows'], L=config['L'], 
                            splitOff=config['splitOff'], RRDB_nb=config['RRDB_nb'], RRDB_nf=config['RRDB_nf'], 
                            preK=config['preK'], factor=config['factor'], ks_s_pad=config['ks_s_pad'])
        
        aemodel_y2 = Encoder_y2(inchannel=1, outchannel=4, hidden_channel=16)

        nfmodel = nfmodel.to(device)
        aemodel_y2 = aemodel_y2.to(device)
        
        val_path = config['test_data_file_path']
        # low_res_path = config['test_low_res_file_path']
        # high_res_path = config['test_high_res_file_path']
        
        logger = Logger(config, base_dir, snapshot_dir, output_dir, log_dir)
        
        if args.resume_nf:
            itr, nfmodel = load_checkpoint(args.resume_nf, nfmodel, only_net=True)
            logger.load_itr(itr)
        if args.resume_ae:
            itr, aemodel_y2 = load_checkpoint(args.resume_ae, aemodel_y2, only_net=True)
            logger.load_itr(itr)
        
        nfmodel.eval()
        aemodel_y2.eval()

        test(args, config['dataname'], val_path, logger, nfmodel, aemodel_y2, blocksize=['blocksize'], padding=['padding'], batchsize_test=config['batchsize_test'], save_latent=False)


if __name__ == '__main__':
    main(sys.argv[1:])

