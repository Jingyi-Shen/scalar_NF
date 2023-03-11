import argparse
from ctypes import Structure
import random
import sys
import os
from datetime import datetime
import numpy as np
import struct
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from models.baseline.esrgan import ESRGAN, Discriminator

from dataset import get_test_dataloader, ScalarData, ScalarEnsembleData, get_size, get_test_nyxensemble_loader
# from dataset import min_max_nyx, min_max_vortex, min_max_fpm, min_max_cos, min_max_jet3b
from utils.utils_base import init, Logger, load_checkpoint, save_checkpoint, AverageMeter, get_duration

from losses.losses import Metrics, ReconLoss
from dataset import denormalize_zscore, denormalize_max_min, read_data_from_file, data_reader, normalize_max_min

import scipy.ndimage
from skimage.transform import rescale, resize

import pdb
# torch.manual_seed(0)

def parse_args(argv):
    parser = argparse.ArgumentParser(description='NormFlow')
    parser.add_argument('--train', action='store_true', help='Is true if is training.')
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--name', help='result dir name', default=datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), type=str)
    parser.add_argument('--resume_G', help='generator model snapshot path', type=str)
    parser.add_argument('--resume_D', help='discriminator model snapshot path', type=str) 
    parser.add_argument('--seed', help='seed number', default=123, type=int)
    
    parser.add_argument('--sample_size', type=int, help='sample size')
    parser.add_argument('--sample_size_list', nargs='+', type=int, help='sample size list')
    parser.add_argument('--sample_type', help='sample type (single, multi)', default='single', type=str)
    parser.add_argument('--ensemble', action='store_true', help='ensemble or not')
    
    parser.add_argument('--lr_D', type=float, default=4e-4, metavar='LR', help='learning rate of D')
    parser.add_argument('--lr_G', type=float, default=1e-4, metavar='LR', help='learning rate of G')
    parser.add_argument('--lambda_adv', type=float, default=1e-3, metavar='W', help='weight of adversarial loss')
    parser.add_argument('--lambda_mse', type=float, default=1, metavar='W', help='weight of content loss')
    parser.add_argument('--lambda_percep', type=float, default=5e-2, metavar='W', help='weight of perceptual loss')
    
    parser.add_argument('--sf', dest='sf', type=int, default=2, help='scale factor for sr')
    
    args = parser.parse_args(argv)

    if not args.config:
        if args.resume:
            assert args.resume.startswith('./')
            dir_path = '/'.join(args.resume.split('/')[:-2])
            args.config = os.path.join(dir_path, 'config.yaml')
        else:
            args.config = './configs/config.yaml'
    return args


def get_dataset(config, high_res, low_res=None, mode='train', scale_factor=2, sampler=100):
    if low_res is None:
        dataset = ScalarData(high_res, mode=mode, scale_factor=scale_factor, blocksize=config['blocksize'], padding=config['padding'], sampler=sampler)
    else:
        dataset = ScalarEnsembleData(high_res, low_res, mode=mode, scale_factor=scale_factor, blocksize=config['blocksize'], padding=config['padding'], sampler=sampler)
    return dataset


def train(args, config, base_dir, snapshot_dir, output_dir, log_dir):
    device = torch.device("cuda" if config['cuda'] else "cpu") 
    # device = 'cpu'
    print('device', device)
    
    file_list = read_data_from_file(config['dataname'], config['data_path'])
    
    logger = Logger(config, base_dir, snapshot_dir, output_dir, log_dir)
    
    G = ESRGAN(in_channels=1, out_channels=1, nf=32, gc=16, scale_factor=args.sf, n_basic_block=10)# 23, 18
    Ds = Discriminator(num_conv_block=2)
    print(G)
    print(Ds)
    G = G.to(device)
    Ds = Ds.to(device)
    
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr_G, betas=(0.9,0.999))
    optimizer_s_D = optim.Adam(Ds.parameters(), lr=args.lr_D, betas=(0.5,0.999))
    fake = 0
    real = 1
    criterion = nn.MSELoss()
    critic = 1
    
    torch.autograd.set_detect_anomaly(True)
    G.train()
    Ds.train()
    
    mse_best = 1e10
    adversarial_criterion = nn.BCEWithLogitsLoss().to(device)
    content_criterion = nn.L1Loss().to(device)
    # perception_criterion = PerceptualLoss().to(device)

    while logger.itr < config['max_itr']:
        # print('logger.itr', logger.itr)
        epoch_loss = 0
        epoch_mse_loss = 0
        epoch_Ds_loss = 0

        count = 0
        for ind_f, f in enumerate(file_list):
            # print(ind_f, "===> File in process: ", f)
            train_data, attr_min_max = data_reader(f, config['dataname'], padding=config['padding'])
            
            if args.ensemble:
                name_low = f.split('.')[0]
                f_low = './data/density/nyx/low_res/nyx_low_level1.bin'
                train_data_low, attr_min_max = data_reader(f_low, 'na', padding=config['padding']//args.sf, scale=args.sf)
            
            if args.sample_type == 'single':
                if args.ensemble:
                    train_dataset = get_dataset(config, train_data, train_data_low, mode='train', scale_factor=args.sf, sampler=args.sample_size)
                else:
                    train_dataset = get_dataset(config, train_data, low_res=None, mode='train', scale_factor=args.sf, sampler=args.sample_size)
                    # train_dataset = ScalarData(train_data, mode='train', scale_factor=args.sf, blocksize=config['blocksize'], padding=config['padding'], sampler=args.sample_size)
            
            elif args.sample_type == 'multi':
                if not isinstance(args.sample_size_list, list):
                    sample_size = [args.sample_size_list] * 5
                else:
                    sample_size = args.sample_size_list
                # train_dataset = ScalarData(train_data, mode='train', scale_factor=args.sf, blocksize=config['blocksize'], padding=config['padding'], sampler=sample_size[0])
                if args.ensemble:
                    # print('level 0')
                    train_dataset = get_dataset(config, train_data, train_data_low, mode='train', scale_factor=args.sf, sampler=sample_size[0])
                else:
                    train_dataset = get_dataset(config, train_data, low_res=None, mode='train', scale_factor=args.sf, sampler=sample_size[0])
                
                for level in range(1, len(sample_size)):
                    # print('level ', level)
                    ds_scale = np.power(args.sf, level)
                    train_data, attr_min_max = data_reader(f, config['dataname'], padding=config['padding'], scale=ds_scale)
                    # train_dataset_2 = ScalarData(train_data, mode='train', scale_factor=args.sf, blocksize=config['blocksize'], padding=config['padding'], sampler=sample_size[level])
                    if args.ensemble:
                        f_low = f'./data/density/nyx/low_res/nyx_low_level{level+1}.bin'
                        train_data_low, attr_min_max = data_reader(f_low, 'na', padding=config['padding']//args.sf, scale=ds_scale*args.sf) # xsf downsample
                        train_dataset_2 = get_dataset(config, train_data, train_data_low, mode='train', scale_factor=args.sf, sampler=sample_size[level])
                    else:
                        train_dataset_2 = get_dataset(config, train_data, low_res=None, mode='train', scale_factor=args.sf, sampler=sample_size[level])
                    train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_2])
            
            train_dataloader = DataLoader(train_dataset, batch_size=config['batchsize'], shuffle=True, drop_last=False, num_workers=0)
            
            for x, x_low in train_dataloader:
                x = x.to(device) 
                x_low = x_low.to(device) 

                real_labels = torch.ones((x.size(0), 1)).to(device)
                fake_labels = torch.zeros((x.size(0), 1)).to(device)
                
                #################################
                # Update G network
                #################################
                for p in G.parameters():
                    p.requires_grad = True
                for p in Ds.parameters():
                    p.requires_grad = False
                
                optimizer_G.zero_grad()

                fake_data = G(x_low)

                score_fake = Ds(fake_data)
                score_real = Ds(x)
                # adversarial loss
                discriminator_rf = score_real - score_fake.mean()
                discriminator_fr = score_fake - score_real.mean()

                adversarial_loss_rf = adversarial_criterion(discriminator_rf, fake_labels)
                adversarial_loss_fr = adversarial_criterion(discriminator_fr, real_labels)
                L_adv = 0.5*(adversarial_loss_fr + adversarial_loss_rf)
                
                # perception loss
                # L_p = perception_criterion(x, fake_data)
                # content loss
                L_c = content_criterion(fake_data, x)
                
                loss = args.lambda_adv*(L_adv)+args.lambda_mse*L_c # +args.lambda_percep*L_p
                
                loss.backward()
                epoch_mse_loss += loss.item() 
                optimizer_G.step()

                for p in Ds.parameters():
                    p.requires_grad = True

                ################################
                # Update D network
                ################################
                for p in G.parameters():
                    p.requires_grad = False
                
                for j in range(1,critic+1):
                    optimizer_s_D.zero_grad()
                    
                    # train with real
                    score_real = Ds(x)
                    score_fake = Ds(fake_data.detach())
                    discriminator_rf = score_real - score_fake.mean()
                    discriminator_fr = score_fake - score_real.mean()
                    
                    adversarial_loss_rf = adversarial_criterion(discriminator_rf, real_labels)
                    adversarial_loss_fr = adversarial_criterion(discriminator_fr, fake_labels)
                    loss_gan = 0.5*(adversarial_loss_fr + adversarial_loss_rf)
                    
                    loss_gan.backward()
                    epoch_Ds_loss += loss_gan.mean().item()
                    optimizer_s_D.step()

                
                count += len(x)
                
                epoch_loss += (loss.item() + loss_gan.mean().item())
                # epoch_loss += loss.item()


        epoch_loss /= count
        epoch_mse_loss /= count
        epoch_Ds_loss /= count
        # logging
        # log_dict
        logger.update({
            'loss': epoch_loss,
            'sample_loss': epoch_Ds_loss,
            'cycle_loss': epoch_mse_loss
        })
        if logger.itr % config['log_itr'] == 0:
            print('==> Epoch {:02d} average loss: {:.6f}'.format(logger.itr, epoch_loss))
            print('epoch_mse_loss..', epoch_mse_loss)
            print('epoch_Ds_loss..', epoch_Ds_loss)
            logger.write()
            logger.init()

        # test and save model snapshot
        if logger.itr % config['test_itr'] == 0 or logger.itr % config['snapshot_save_itr'] == 0:
            if logger.itr % config['snapshot_save_itr'] == 0:
                mse = test(args, config['dataname'], config['test_data_file_path'], logger, G, Ds, blocksize=config['blocksize'], padding=config['padding'], batchsize_test=config['batchsize_test'], save_test=True)
            else:
                mse = test(args, config['dataname'], config['test_data_file_path'], logger, G, Ds, blocksize=config['blocksize'], padding=config['padding'], batchsize_test=config['batchsize_test'], save_test=False)
            print('[test] loss, epoch_loss, loss_best: ', mse , epoch_loss, mse_best)
            
            if epoch_loss < mse_best:
                print('Best!')
                save_checkpoint(os.path.join(snapshot_dir, 'best_G.pt'), logger.itr, G)
                save_checkpoint(os.path.join(snapshot_dir, 'best_Ds.pt'), logger.itr, Ds)
                mse_best = epoch_loss

            if logger.itr % config['snapshot_save_itr'] == 0:
                save_checkpoint(os.path.join(snapshot_dir, f'{logger.itr:05}_{mse:.8f}_G.pt'),
                                logger.itr, G)
                save_checkpoint(os.path.join(snapshot_dir, f'{logger.itr:05}_{mse:.8f}_Ds.pt'),
                                logger.itr, Ds)
        
        # # lr scheduling
        # if logger.itr % config['lr_shedule_step'] == 0:
        #     lr_prior = optimizer.param_groups[0]['lr']
        #     for g in optimizer.param_groups:
        #         g['lr'] *= config['lr_shedule_scale']
        #     print(f'[lr scheduling] {lr_prior} -> {optimizer.param_groups[0]["lr"]}')


def PSNR_MSE(x, y, diff=1, istorch=False):
    if istorch:
        mse = torch.mean((x - y) ** 2, dim=[1, 2, 3, 4])
        psnr = 10 * torch.log10(self.diff ** 2. / mse)
    else:
        mse = np.mean((x - y) ** 2)
        psnr = 20 * np.log10(diff / np.sqrt(mse))
    return psnr, mse


def test(args, dataname, val_path, logger, G, Ds, blocksize=24, padding=4, batchsize_test=128, mode='sr', save_test=False):
    G.eval()
    Ds.eval()
    device = next(G.parameters()).device

    test_dataloader, attr_min_max, all_data_gt, _ = get_test_dataloader(args, dataname, val_path, blocksize=blocksize, padding=padding, batch_size_test=batchsize_test, level=1)
    # test_dataloader, attr_min_max, all_data_gt = get_test_nyxensemble_loader(args, dataname, val_path, blocksize=blocksize, padding=padding, batch_size_test=batchsize_test, level=1)
    
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
        # result_low_res = np.zeros(( np.array(result_size) // args.sf))
        
        start = datetime.now()
        for x, x_low, z_ind, y_ind, x_ind in test_dataloader:
        # for x_low, z_ind, y_ind, x_ind in test_dataloader:
            x_low = x_low.to(device)
            
            x_hat = G(x_low)
            
            x_hat = x_hat.detach().cpu().numpy()
            for ii in range(len(z_ind)):
                if padding > 0:
                    result[int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += x_hat[ii].reshape(blocksize, blocksize, blocksize)[padding:-padding, padding:-padding, padding:-padding]
                else:
                    result[int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += x_hat[ii].reshape(blocksize, blocksize, blocksize)
        
        del x_low, z_ind, y_ind, x_ind
        h, m, s = get_duration(start, datetime.now())
        time_str = 'Test Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)  
        print(time_str)
        
        # loss /= cnt 
        result = np.array(result)
        all_data_gt = np.array(all_data_gt)
        if padding > 0:
            all_data_gt = all_data_gt[padding:-padding, padding:-padding, padding:-padding]
        
        # # add lerp to redisual result
        # lerp_result = rescale(result_low_res, [args.sf, args.sf, args.sf], order=3)
        # print('min max residual', np.min(result), np.max(result))
        # print('(gt) min max residual', np.min(all_data_gt-lerp_result),  np.max(all_data_gt-lerp_result))
        # result = lerp_result + result
        
        result = denormalize_max_min(result, attr_min_max).reshape(-1)
        all_data_gt = denormalize_max_min(all_data_gt, attr_min_max).reshape(-1)
        # result_low_res = denormalize_max_min(result_low_res, attr_min_max).reshape(-1)

        recon_psnr, recon_mse = PSNR_MSE(all_data_gt, result, attr_min_max['max'][0]-attr_min_max['min'][0])
        
        # print('max_gt, max_recon', all_data_gt.shape, all_data_gt.shape, np.min(all_data_gt), np.max(all_data_gt), np.min(result), np.max(result))
        # with open(os.path.join(logger.output_dir, dataname[:3]+f'_{mode}_mse{recon_mse:.4f}_psnr{recon_psnr:.4f}_loss{loss:.4f}.bin'), 'wb') as f:
        #     f.write(struct.pack('<%df' % len(result), *result))
        if save_test:
            with open(os.path.join(logger.output_dir, dataname[:3]+f'_{mode}_mse{recon_mse:.4f}_psnr{recon_psnr:.4f}.bin'), 'wb') as f:
                f.write(struct.pack('<%df' % len(result), *result))
        
        # if not os.path.isfile(os.path.join(logger.output_dir, dataname[:3]+f'_gt.bin')):
        #     with open(os.path.join(logger.output_dir, dataname[:3]+f'_gt.bin'), 'wb') as f:
        #         f.write(struct.pack('<%df' % len(all_data_gt), *all_data_gt))
        
        # with open(os.path.join(logger.output_dir, dataname[:3]+f'_low_res_mse{recon_mse:.4f}.bin'), 'wb') as f:
        #     f.write(struct.pack('<%df' % len(result_low_res), *result_low_res))
        print(f'[ Test ]==>>> {dataname}, recon_mse{recon_mse:.4f}, recon_psnr{recon_psnr:.4f}')
    
    logger.init()
    G.train()
    Ds.train()
    return recon_mse

    
def test_recursive(args, dataname, val_path_high, val_path_low, logger, nfmodel, model_y2, sr_steps=3, blocksize=24, padding=4, batchsize_test=128, return_latent=False, STD=0, test_everystep=False):
    device = next(nfmodel.parameters()).device
    nfmodel.eval()
    model_y2.eval()
    # name = low_res_path.split('/')[-1].split('.')[0]
    
    act_size = blocksize - 2 * padding
    act_size_low = act_size // args.sf
    blocksize_low = blocksize // args.sf
    padding_low = padding // args.sf

    # low-res data
    # ds_scale = np.power(args.sf, level)
    data_high, attr_min_max = data_reader(val_path_high, dataname, padding=padding, scale=1) # normalized
    test_dataloader, attr_min_max, data_low = get_test_dataloader(args, dataname, val_path_high, 
                            blocksize=blocksize_low, padding=padding_low, batch_size_test=batchsize_test, 
                            level=sr_steps, order=2) # order: 2 (default)
    # test_dataloader, attr_min_max, data_low, data_high = get_low_res_test_dataloader(args, dataname, val_path_high, blocksize=blocksize_low, padding=padding_low, batch_size_test=batchsize_test, level=sr_steps)
    # test_dataloader, attr_min_max, data_high = get_test_dataloader(args, dataname, val_path_high, blocksize=blocksize, padding=padding, batch_size_test=batchsize_test, level=0)
    print('initial low resolution shape ', data_low.shape, data_high.shape)

    start = datetime.now()
    with torch.no_grad():
        logger.init()
        
        for sr_step in range(sr_steps):
            result_size = get_size(dataname, sf=args.sf, level=sr_steps-sr_step-1)
            result = np.zeros((result_size))
            # result_low_res = np.zeros(( np.array(result_size) // args.sf))

            for x, _, z_ind, y_ind, x_ind in test_dataloader:   
                x = x.to(device)
                y2_z = model_y2(x=x) # x_low [8,8,8]
                # import pdb
                # pdb.set_trace()
                x_hat_sr = nfmodel(z=y2_z, eps_std=STD, reverse=True, training=False, z1=None, z2=None)
                x_hat_sr = x_hat_sr.detach().cpu().numpy()
                # x = x.detach().cpu().numpy()
                for ii in range(len(z_ind)):
                    if padding > 0:
                        result[int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += x_hat_sr[ii].reshape(blocksize, blocksize, blocksize)[padding:-padding, padding:-padding, padding:-padding]
                        # result_low_res[int(z_ind[ii])*act_size_low:int(z_ind[ii])*act_size_low+act_size_low, int(y_ind[ii])*act_size_low:int(y_ind[ii])*act_size_low+act_size_low, int(x_ind[ii])*act_size_low:int(x_ind[ii])*act_size_low+act_size_low] += x[ii].reshape(blocksize_low, blocksize_low, blocksize_low)[padding_low:-padding_low, padding_low:-padding_low, padding_low:-padding_low]
                    else:
                        result[int(z_ind[ii])*act_size:int(z_ind[ii])*act_size+act_size, int(y_ind[ii])*act_size:int(y_ind[ii])*act_size+act_size, int(x_ind[ii])*act_size:int(x_ind[ii])*act_size+act_size] += x_hat_sr[ii].reshape(blocksize, blocksize, blocksize)
                        # result_low_res[int(z_ind[ii])*act_size_low:int(z_ind[ii])*act_size_low+act_size_low, int(y_ind[ii])*act_size_low:int(y_ind[ii])*act_size_low+act_size_low, int(x_ind[ii])*act_size_low:int(x_ind[ii])*act_size_low+act_size_low] += x[ii].reshape(blocksize_low, blocksize_low, blocksize_low)
            # cal PSNR
            # lerp_result = rescale(result_low_res, [args.sf, args.sf, args.sf], order=3)
            # print('(gt) min max residual', np.min(all_data_gt-lerp_result),  np.max(all_data_gt-lerp_result))
            # result = lerp_result + result

            if test_everystep:
                ds_scale = np.power(args.sf, sr_steps-sr_step-1) # current high res
                if padding > 0:
                    data_sr_gt = rescale(data_high[padding:-padding, padding:-padding, padding:-padding], [1./ds_scale, 1./ds_scale, 1./ds_scale], order=3)
                else:
                    data_sr_gt = rescale(data_high, [1./ds_scale, 1./ds_scale, 1./ds_scale], order=3)
            
                result = denormalize_max_min(result, attr_min_max)
                data_sr_gt = denormalize_max_min(data_sr_gt, attr_min_max)
                
                recon_psnr, recon_mse = PSNR_MSE(data_sr_gt, result, attr_min_max['max'][0]-attr_min_max['min'][0])
                print(sr_step, result.shape, f'----- [psnr, mse], {recon_psnr:.4f} , {recon_mse:.4f}')

            if sr_step < sr_steps-1:
                if test_everystep:
                    result = normalize_max_min(result, attr_min_max)
                result = np.pad(result, ((padding_low, padding_low), (padding_low, padding_low), (padding_low, padding_low)), 'edge')
                test_dataset = ScalarData(result, mode='eval', scale_factor=args.sf, blocksize=blocksize_low, padding=padding_low)                
                test_dataloader = DataLoader(test_dataset, batch_size=batchsize_test, shuffle=False, drop_last=False, num_workers=0)

        h, m, s = get_duration(start, datetime.now())
        time_str = 'Test Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)  
        print(time_str)
        
        if padding_low > 0:
            data_low = data_low[padding_low:-padding_low, padding_low:-padding_low, padding_low:-padding_low]
            data_high = data_high[padding:-padding, padding:-padding, padding:-padding]
            
        if not test_everystep:
            result = denormalize_max_min(result, attr_min_max)
            data_high = denormalize_max_min(data_high, attr_min_max)
            recon_psnr, recon_mse = PSNR_MSE(data_high, result, attr_min_max['max'][0]-attr_min_max['min'][0])
            print(sr_step, result.shape, f'----- [psnr, mse], {recon_psnr:.4f} , {recon_mse:.4f}')
                
        data_low = denormalize_max_min(data_low, attr_min_max)
        if return_latent:
            print('return low, GT, and sr, with shape: ', data_low.shape, data_high.shape, result.shape)
            return data_low, data_high, result
        else:
            result = result.reshape(-1)
            data_high = data_high.reshape(-1)
            data_low = data_low.reshape(-1)
            with open(os.path.join(logger.output_dir, dataname[:3]+f'_sr{sr_steps}_mse{recon_mse:.4f}_psnr{recon_psnr:.4f}.bin'), 'wb') as f:
                f.write(struct.pack('<%df' % len(result), *result))
            
            with open(os.path.join(logger.output_dir, dataname[:3]+f'_sr{sr_steps}_gt.bin'), 'wb') as f:
                f.write(struct.pack('<%df' % len(data_high), *data_high))
            
            with open(os.path.join(logger.output_dir, dataname[:3]+f'_sr{sr_steps}_gt_low.bin'), 'wb') as f:
                f.write(struct.pack('<%df' % len(data_low), *data_low))


def compute_uncertainty(args, dataname, val_path_high, val_path_low, logger, nfmodel, model_y2, sr_steps=3, blocksize=24, padding=4, batchsize_test=128, eval_lerp=True, STD=0.5, test_everystep=False):
    result_size = get_size(dataname, sf=args.sf, level=0)
    result_size.insert(0, 5)
    result_all = np.zeros(result_size)
    for i in range(5):
        print('---'*8, i)
        data_low, _, result = test_recursive(args, dataname, val_path_high, val_path_low, logger, nfmodel, model_y2, sr_steps, blocksize, padding, batchsize_test, return_latent=True, STD=STD, test_everystep=test_everystep)
        result_all[i] = result
    # cal std at each voxel
    print(result_all.shape)
    
    # cal mean at each voxel, compute psnr and mse, save
    result_all_mean = np.mean(result_all, axis=0).reshape(-1)
    data_sr_gt, attr_min_max = data_reader(val_path_high, dataname, normalize=False, padding=0, scale=1) # normalized
    data_sr_gt = data_sr_gt.reshape(-1)
    recon_psnr, recon_mse = PSNR_MSE(data_sr_gt, result_all_mean, attr_min_max['max'][0]-attr_min_max['min'][0])
    print(f'averaged----- [psnr, mse] {recon_psnr:.4f} , {recon_mse:.4f}')

    # with open(os.path.join(logger.output_dir, dataname[:3]+f'_sr{sr_steps}_mean_mse{recon_mse:.4f}_psnr{recon_psnr:.4f}.bin'), 'wb') as f:
    #     f.write(struct.pack('<%df' % len(result_all_mean), *result_all_mean))

    result_all_std = np.std(result_all, axis=0).reshape(-1)
    # with open(os.path.join(logger.output_dir, dataname[:3]+f'_sr{sr_steps}_STD{STD}_mse{recon_mse:.4f}_psnr{recon_psnr:.4f}.bin'), 'wb') as f:
    #     f.write(struct.pack('<%df' % len(result_all_std), *result_all_std))
    
    # evaluate lerp
    if eval_lerp:
        lerp_names = ['nearest', 'linear', 'cubic']
        ds_scale = np.power(args.sf, sr_steps)
        for i in range(3):
            # lerp_result = scipy.ndimage.zoom(data_low, [ds_scale, ds_scale, ds_scale], order=i, grid_mode=True, mode='grid-constant').reshape(-1)
            lerp_result = rescale(data_low, [ds_scale, ds_scale, ds_scale], order=i).reshape(-1)
            recon_psnr_lerpi, recon_mse_lerpi = PSNR_MSE(data_sr_gt, lerp_result, attr_min_max['max'][0]-attr_min_max['min'][0])
            print(f'[ Test {lerp_names[i]}]==>>> {dataname}, recon_mse{recon_mse_lerpi:.4f}, recon_psnr{recon_psnr_lerpi:.4f}')
            # with open(os.path.join(logger.output_dir, dataname[:3]+f'_sr{sr_steps}_lerp{i}_mse{recon_mse_lerpi:.4f}_psnr{recon_psnr_lerpi:.4f}.bin'), 'wb') as f:
            #     f.write(struct.pack('<%df' % len(lerp_result), *lerp_result))
    
    # data_low = data_low.reshape(-1)
    # with open(os.path.join(logger.output_dir, dataname[:3]+f'_low{sr_steps}_rescale.bin'), 'wb') as f:
    #     f.write(struct.pack('<%df' % len(data_low), *data_low))


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
        
        # print(nfmodel)
        # print(aemodel_y2)
        val_path = config['test_data_file_path']
        # low_res_path = config['test_low_res_file_path']
        # high_res_path = config['test_high_res_file_path']
        
        logger = Logger(config, base_dir, snapshot_dir, output_dir, log_dir)
        
        if args.resume_G:
            itr, G = load_checkpoint(args.resume_G, G, only_net=True)
        
        G = G.to(device)
        G.eval()
        
        sr_steps = 3 # high_res block level, 1(x2) 2(x4) 3(x8)

        # test(args, config['dataname'], val_path, logger, nfmodel, aemodel_y2, blocksize=config['blocksize'], padding=config['padding'], batchsize_test=config['batchsize_test'], mode='sr')
        # test_recursive(args, config['dataname'], val_path, None, logger, nfmodel, aemodel_y2, sr_steps=sr_steps, blocksize=config['blocksize'], padding=config['padding'], batchsize_test=config['batchsize_test'], test_everystep=False)
        compute_uncertainty(args, config['dataname'], val_path, None, logger, nfmodel, aemodel_y2, sr_steps=sr_steps, blocksize=config['blocksize'], padding=config['padding'], batchsize_test=config['batchsize_test'], eval_lerp=True, STD=0.8, test_everystep=False)
    

if __name__ == '__main__':
    main(sys.argv[1:])

