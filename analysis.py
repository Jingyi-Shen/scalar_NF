import numpy as np
import struct
import matplotlib.pyplot as plt
# import seaborn as sns
import os
import vtk
from vtkmodules.all import *
from vtkmodules.util import numpy_support

# from utils.octree import build_octree
from dataset import min_max_fpm, min_max_cos, min_max_jet3b, min_max_nyx, min_max_vortex
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from utils.utils_reg import read_vortex_data, read_combustion_data, read_nyx_data, read_data_bin
from train_vaenf import PSNR_MSE
import scipy.ndimage

from skimage.metrics import structural_similarity as ssim_fun
from skimage import io
from utils.sdf import SDFRead

from dataset import  get_size
from skimage.transform import rescale

import scipy.stats as st
import random

# def proj_2D():
#     from sklearn.manifold import TSNE
#     import plotly.express as px
#     import plotly.graph_objects as go
    
#     # read z and prob
    
#     print('TSNE.... ', X.shape, Y.shape)
#     tsne = TSNE(n_components=2, random_state=0)
#     projections = tsne.fit_transform(Y)
    
#     import pandas as pd
#     df = pd.DataFrame(projections, columns = ['0', '1'])
#     df['h'] = X_str
#     df['v'] = values

#     # pdb.set_trace()
#     # fig = go.Figure()
#     # fig.add_trace(go.Scatter(x=df['0'], y=df['1'],
#     #                 mode='lines+markers+text',
#     #                 name='lines+markers+text',
#     #                 # size=[3 for i in range(len(df['v']))],
#     #                 # color='v', text='v', hover_data='h')
#     #                 ))
#     fig = px.scatter(
#         df, x='0', y='1', size=[3 for i in range(len(df['v']))], color='v', text='v', hover_data='h', 
#     )
#     # # px.line()
#     fig.update_traces(textposition="bottom right")
#     fig.show()

# def fpm_reader(filename):
#     reader = vtk.vtkXMLPolyDataReader()
#     reader.SetFileName(filename)
#     reader.Update()
#     vtk_data = reader.GetOutput()
#     coord = numpy_support.vtk_to_numpy(vtk_data.GetPoints().GetData())
#     logp_x = numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetArray(0))
#     logp_z = numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetArray(1))
#     # velocity = numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetArray(1))
#     # point_data = np.concatenate((coord, concen, velocity), axis=-1)
#     return logp_x, logp_z

# def corr_px_pz():
#     # filename1 = './results/2022-11-20_13_23_58_flow5/outputs/fpm_run44_080_stdsampledz_coord_Prob_corrd16.7199_attr4980.2534.vtu'
#     # filename1 = './results/2022-11-20_13_23_58_flow5/outputs/fpm_run44_080_unisampledz_coord_Prob_corrd21.4698_attr6768.2656.vtu'
#     # filename1 = './results/2022-11-20_13_23_58_flow5/outputs/fpm_run44_080_sampledz_n1_1_coord_Prob_corrd15.2006_attr5246.0908.vtu'
    
#     # filename2 = './results/2022-11-21_12_02_47_reg1/outputs/fpm_run44_080_stdsampledz_coord_Prob_corrd165772.4844_attr2372720.7500.vtu'
#     # filename2 = './results/2022-11-21_12_02_47_reg1/outputs/fpm_run44_080_unisampledz_coord_Prob_corrd1721594.2500_attr20255688.0000.vtu'
#     # filename2 = './results/2022-11-21_12_02_47_reg1/outputs/fpm_run44_080_stdsampledz_n1_1_coord_Prob_corrd16297.3291_attr399782.8438.vtu'
    
#     # filename3 = './results/2022-11-21_12_03_39_reg05/outputs/fpm_run44_080_stdsampledz_coord_Prob_corrd17.1953_attr4870.6587.vtu'
#     # filename3 = './results/2022-11-21_12_03_39_reg05/outputs/fpm_run44_080_unisampledz_coord_Prob_corrd29.2237_attr9098.0986.vtu'
#     # filename3 = './results/2022-11-21_12_03_39_reg05/outputs/fpm_run44_080_stdsampledz_n1_1_coord_Prob_corrd16.5409_attr5172.9639.vtu'
    
#     # filename1 = './results/2022-11-20_13_23_58_flow5/outputs/fpm_run44_020_Prob_corrd0.0000_attr0.0000_loss-0.0183.vtu'
#     # filename2 = './results/2022-11-21_12_02_47_reg1/outputs/fpm_run44_020_Prob_corrd0.0000_attr0.0000_loss0.0180.vtu'
#     # filename3 = './results/2022-11-21_12_03_39_reg05/outputs/fpm_run44_020_Prob_corrd0.0000_attr0.0000_loss-0.0195.vtu'
    
#     # filename1 = './results/2022-11-20_13_23_58_flow5/outputs/fpm_run44_080_Prob_corrd0.0000_attr0.0000_loss-0.0117.vtu'
#     # filename2 = './results/2022-11-21_12_02_47_reg1/outputs/fpm_run44_080_Prob_corrd0.0000_attr0.0000_loss0.0179.vtu'
#     # filename3 = './results/2022-11-21_12_03_39_reg05/outputs/fpm_run44_080_Prob_corrd0.0000_attr0.0000_loss-0.0168.vtu'
    
#     # filename1 = './results/2022-11-20_13_23_58_flow5/outputs/fpm_run44_120_Prob_corrd0.0000_attr0.0000_loss0.0388.vtu'
#     # filename2 = './results/2022-11-21_12_02_47_reg1/outputs/fpm_run44_120_Prob_corrd0.0000_attr0.0000_loss0.0178.vtu'
#     # filename3 = './results/2022-11-21_12_03_39_reg05/outputs/fpm_run44_120_Prob_corrd0.0000_attr0.0000_loss0.0075.vtu'
    
#     # uniform in [-1, 1]
#     filename1 = './results/2022-11-20_13_23_58_flow5/outputs/fpm_run44_120_unisampledz_n1_1_coord_Prob_corrd15.3475_attr4494.7886.vtu'
#     filename2 = './results/2022-11-21_12_02_47_reg1/outputs/fpm_run44_120_unisampledz_n1_1_coord_Prob_corrd21808.5586_attr529535.3125.vtu'
#     filename3 = './results/2022-11-21_12_03_39_reg05/outputs/fpm_run44_120_unisampledz_n1_1_coord_Prob_corrd16.4590_attr4037.2449.vtu'
    
#     from scipy.stats import pearsonr
#     logp_x1, logp_z1 = fpm_reader(filename1)
#     logp_x2, logp_z2 = fpm_reader(filename2)
#     logp_x3, logp_z3 = fpm_reader(filename3)
#     print(len(logp_x3))
#     corr, _ = pearsonr(logp_x1, logp_z1)
#     print('(w/o reg) Pearsons correlation: %.3f' % corr)
    
#     corr, _ = pearsonr(logp_x3, logp_z3)
#     print('(0.5 reg) Pearsons correlation: %.3f' % corr)
    
#     corr, _ = pearsonr(logp_x2, logp_z2)
#     print('(1 reg) Pearsons correlation: %.3f' % corr)
    
#     # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
#     # plt.scatter(logp_x3, logp_z3, label='0.5 reg', s=1)
#     plt.scatter(logp_x1, logp_z1, c='#d62728', label='w/o reg', s=1)#, alpha=0.8)
#     plt.scatter(logp_x3, logp_z3, c='#1f77b4', label='0.5 reg', s=1)#, alpha=0.8)
#     plt.scatter(logp_x2, logp_z2, c='#ff7f0e', label='1 reg', s=1)
#     plt.legend()
#     plt.show()
    

def particle_uncertainty_analysis(data_all, data_name='fpm'):
    resolution = [1, 1, 1]
    min_max_coord = None
    if data_name == 'fpm':
        min_max_coord = {
            'min': [-4.99989, -4.9999275, 0.],
            'max': [4.9999995, 4.999923, 10.]
        }
    # create a 3D mesh grid
    xs = np.arange(min_max_coord['min'][0], min_max_coord['max'][0], resolution[0])
    ys = np.arange(min_max_coord['min'][1], min_max_coord['max'][1], resolution[1])
    zs = np.arange(min_max_coord['min'][2], min_max_coord['max'][2], resolution[2])
    X, Y, Z = np.meshgrid(xs, ys, zs)
    print(f"My 3d meshgrid array (X): \n {X}")
    print(X.shape)

    # Define the grid spacing, uniform
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    dz = zs[1] - zs[0]

    # # for each grid. find data in grid and cal distances between point sets
    # # Calculate the eight corners of every grid in the mesh grid
    # X_corners = np.array([X - dx/2, X + dx/2, X + dx/2, X - dx/2, X - dx/2, X + dx/2, X + dx/2, X - dx/2])
    # Y_corners = np.array([Y - dy/2, Y - dy/2, Y + dy/2, Y + dy/2, Y - dy/2, Y - dy/2, Y + dy/2, Y + dy/2])
    # Z_corners = np.array([Z - dz/2, Z - dz/2, Z - dz/2, Z - dz/2, Z + dz/2, Z + dz/2, Z + dz/2, Z + dz/2])

    # # Transpose the corner arrays to match the shape of the mesh grid
    # X_corners = np.transpose(X_corners, (1, 2, 3, 0))
    # Y_corners = np.transpose(Y_corners, (1, 2, 3, 0))
    # Z_corners = np.transpose(Z_corners, (1, 2, 3, 0))

    uncertainty_array = np.zeros((len(xs)-1, len(ys)-1, len(zs)-1, 1))

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]): # indices of the grid to test
                # Calculate the bounds of the grid
                xmin, xmax = x[i] - dx/2, x[i] + dx/2
                ymin, ymax = y[j] - dy/2, y[j] + dy/2
                zmin, zmax = z[k] - dz/2, z[k] + dz/2

                # Find all the points in the grid
                all_points_in_grid = []
                for data_idx in range(len(data_all)):
                    data = data_all[data_idx]
                    print(data_idx, ' data shape', data.shape)
                    import pdb
                    pdb.set_trace()
                    points_in_grid = []
                    for p in data:
                        x_test, y_test, z_test = point
                        if xmin <= x_test <= xmax and ymin <= y_test <= ymax and zmin <= z_test <= zmax:
                            points_in_grid.append(point)
                    print("Found {} points in the grid ({}, {}, {}).".format(len(points_in_grid), i, j, k))
                    all_points_in_grid.append(points_in_grid)
                # cal uncertainty of this grid using all_points_in_grid
                # all_points_in_grid, chamfer
                pdb.set_trace()
                chamfer = 0
                uncertainty_array[i, j, k] += chamfer
                # save uncertainty
    
    # plot and save uncertainty array
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(X, Y, Z, c=data.flatten(), cmap='viridis')
    # ax.scatter3D(X, Y, Z, c=mean_points.flatten(), cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('uncertainty')
    plt.show()

def read_text_data(filename):
    # read Nyx particles
    # fh = open(os.path.join(root, "r1.txt"))
    # filenames = []
    # for line in fh:
    #     line = line.strip("\r\n")
    #     filenames.append(line)
    return None


################################################################################################
# data reader
def fpm_reader(filename, vel=True):
    if vel:
        reader = vtk.vtkXMLUnstructuredGridReader()
    else: # recon data, polydata format
        reader = vtk.vtkXMLPolyDataReader()
    
    reader.SetFileName(filename)
    reader.Update()
    vtk_data = reader.GetOutput()
    coord = numpy_support.vtk_to_numpy(vtk_data.GetPoints().GetData())
    concen = numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetArray(0))[:,None]

    if vel:
        velocity = numpy_support.vtk_to_numpy(vtk_data.GetPointData().GetArray(1))
        if len(velocity.shape) < 2:
            velocity = velocity[:,None]
        point_data = np.concatenate((coord, concen, velocity), axis=-1)
    else:
        point_data = np.concatenate((coord, concen), axis=-1)
    return point_data


def sdf_reader(filename):
    particles = SDFRead(filename)
    h_100 = particles.parameters['h_100']
    cosmo_a = particles.parameters['a']
    kpc_to_Mpc = 1./1000
    convert_to_cMpc = lambda proper: (proper ) * h_100 * kpc_to_Mpc / cosmo_a + 31.25
    numpy_data = np.array(list(particles.values())[2:-1]).T # coord, velocity, acceleration, phi
    numpy_data[:,:3] = convert_to_cMpc(numpy_data[:,:3])
    return numpy_data

def jet3b_reader(filename):
    ds = SDFRead(filename)
    rho = np.array(ds['rho'])
    temp = np.array(ds['temp'])
    rho = np.log10(rho)
    temp = np.log10(temp)
    # data = np.stack([ds['x'], ds['y'], ds['z'], rho, temp], axis=1) # density and temp
    data = np.stack([ds['x'], ds['y'], ds['z'], rho], axis=1) # density and temp
    return data

################################################################################################

def particle_to_density():
    dataname = 'fpm'
    filename = '/fs/ess/PAS0027/2016_scivis_fpm/0.44/run41/081.vtu'
    # /fs/ess/PAS0027/jet3b/run3g_50Am_jet3b_sph.3000
    if dataname == 'fpm':
        data = fpm_reader(filename)
    elif dataname == 'jet3b':
        data = jet3b_reader(filename)
    
    min_max_coord = {
        'min': data[:, :3].min(0),
        'max': data[:, :3].max(0)
    }
    min_max_attr = {
        'min': data[:, 3].min(),
        'max': data[:, 3].max()
    }
    print('coord min max', min_max_coord)
    print('attr min max', min_max_attr)
    resolution = [50, 50, 50]
    # divide volume into grid and compute density of points (#/V) in this grid
    # xs = np.arange(min_max_coord['min'][0], min_max_coord['max'][0], resolution[0])
    # ys = np.arange(min_max_coord['min'][1], min_max_coord['max'][1], resolution[1])
    # zs = np.arange(min_max_coord['min'][2], min_max_coord['max'][2], resolution[2])
    xs = np.linspace(min_max_coord['min'][0], min_max_coord['max'][0], num=resolution[0])
    ys = np.linspace(min_max_coord['min'][1], min_max_coord['max'][1], num=resolution[1])
    zs = np.linspace(min_max_coord['min'][2], min_max_coord['max'][2], num=resolution[2])
    # attrs = np.linspace(min_max_attr['min'], min_max_attr['max'], num=resolution[2])
    X, Y, Z = np.meshgrid(xs, ys, zs)
    # print(f"My 3d meshgrid array (X): \n {X}")
    # print(X.shape) # [101, 101, 101]
    # import pdb
    # pdb.set_trace()
    values = np.vstack([data[:, 0], data[:, 1]])
    values = np.vstack([values, data[:, 2]])
    kernel = st.gaussian_kde(values)
    
    # Evaluate the KDE on a regular grid (positions)...
    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    print('computing density...')
    density = np.reshape(kernel(positions).T, X.shape)
    print(density.shape)
    density = density.reshape(-1)
    name = filename.split('/')[-1].split('.')[0]
    with open(os.path.join('./data/density/',f'{dataname}/{name}_res_{resolution[0]}_{resolution[1]}_{resolution[2]}.bin'), 'wb') as f:
        f.write(struct.pack('<%df' % len(density), *density))
        

def particle_to_scalar():
    dataname = 'fpm'
    filename = '/fs/ess/PAS0027/2016_scivis_fpm/0.44/run41/081.vtu'
    # /fs/ess/PAS0027/jet3b/run3g_50Am_jet3b_sph.3000
    if dataname == 'fpm':
        data = fpm_reader(filename)
    elif dataname == 'jet3b':
        data = jet3b_reader(filename)
    
    min_max_coord = {
        'min': data[:, :3].min(0),
        'max': data[:, :3].max(0)
    }
    min_max_attr = {
        'min': data[:, 3].min(),
        'max': data[:, 3].max()
    }
    print('coord min max', min_max_coord)
    print('attr min max', min_max_attr)
    resolution = np.array([64, 64, 64])
    voxel_size = (min_max_coord['max'] - min_max_coord['min'] + 0.5) / resolution
    # Calculate voxel indices for each point
    # transform to positive grid
    data[:,:3] -= data[:,:3].min(0)
    voxel_indices = np.floor(data[:,:3] / voxel_size).astype(int)
    # Calculate the number of points in each voxel
    voxel_attrs = np.zeros((resolution[0], resolution[1], resolution[2]))
    voxel_counts = np.zeros((resolution[0], resolution[1], resolution[2]))
    print('computing density...')
    for i in range(len(data)):
        voxel_attrs[voxel_indices[i, 0], voxel_indices[i, 1], voxel_indices[i, 2]] += data[i,3]
        voxel_counts[voxel_indices[i, 0], voxel_indices[i, 1], voxel_indices[i, 2]] += 1
    # Calculate the average attribuate of each voxel
    voxel_density = voxel_attrs / (voxel_counts+0.1) #/ (voxel_size[0]*voxel_size[1]*voxel_size[2])
    # Flatten the voxel density array to a 1D array of point densities
    # point_density = voxel_density[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]]
    # # Print the density of the first 10 points
    # print(point_density[:10])

    # print(point_density.shape)
    # point_density = point_density.reshape(-1)
    voxel_density = voxel_density.reshape(-1)
    name = filename.split('/')[-1].split('.')[0]
    with open(os.path.join('./data/density/',f'{dataname}/{name}_attr_res_{resolution[0]}_{resolution[1]}_{resolution[2]}.bin'), 'wb') as f:
        f.write(struct.pack('<%df' % len(voxel_density), *voxel_density))
        


def compute_ssim():
    result = 'img_vortex_AENF_500_15f_5cond_3nb_64nf_sz2_1lat_38.1098'
    root = f"./results/images/{result}/4/"
    
    tfs = 1
    recon_N = 4
    gt_img = io.imread(os.path.join(root, "r1_gt.png"))
    ssims = np.zeros(recon_N)
    names = ['flow', 'lerp0', 'lerp1', 'lerp2']
    # for i in range(len(filenames)):
        # filename = filenames[i]
    for i in range(recon_N):
        print(names[i])
        img_recon = io.imread(os.path.join(root, f"r1_{names[i]}.png"))
        # ssim = ssim_fun(gt_img, img_recon, data_range=255, multichannel=True)
        ssim1 = ssim_fun(gt_img[:,:,0], img_recon[:,:,0], data_range=255)
        ssim2 = ssim_fun(gt_img[:,:,1], img_recon[:,:,1], data_range=255)
        ssim3 = ssim_fun(gt_img[:,:,2], img_recon[:,:,2], data_range=255)
        ssims[i] = (ssim1 + ssim2 + ssim3) / 3.
    # np.save(os.path.join(args.root, "res", "ssim_" + args.mode + ".npy"), ssims)
    ssim_avg = ssims.mean()
    print(ssims, ssim_avg)
    # 1: 0.94007169 0.96610703 0.9508873  0.96220694
    # 2: 0.93297222 0.94479468 0.95137095 0.96591406
    # 3: 0.94237336 0.94960458 0.95738539 0.97020612

    # 1: 0.81115315 0.81352924 0.84528984 0.86839209
    # 2: 0.7244913  0.72545611 0.7734341  0.81679825
    # 3: 0.75103752 0.74447638 0.80157843 0.83147355
    # 4: 0.73539254 0.70286758 0.79381382 0.82735074
    # flow, lerp 0, lerp 1, lerp 2
    

def test_lerp():
    name = 'nyx'
    if name == 'vortex':
        attr_min_max = min_max_vortex 
        filename = '/fs/ess/PAS0027/vortex_data/vortex/vorts01.data'
        data_high, name = read_vortex_data(filename, padding=0, scale=1)
        data_low, name = read_vortex_data(filename, padding=0, scale=8)
    
    elif name == 'nyx':
        attr_min_max = min_max_nyx
        filename = '/fs/project/PAS0027/nyx/256/256CombineFiles/raw/4.bin'
        data_high, name = read_nyx_data(filename, padding=0, scale=1)
        data_low, name = read_nyx_data(filename, padding=0, scale=8)
    
    for i in range(3):
        lerp_result = scipy.ndimage.zoom(data_low, [8, 8, 8], order=i)
        recon_psnr, recon_mse = PSNR_MSE(lerp_result, data_high, attr_min_max['max'][0]-attr_min_max['min'][0])
        print('recon_psnr, recon_mse', recon_psnr, recon_mse)
            
def test_seam():
    # should ue rescale instead of zoom, zoom: assume pixels are in the center of image cell
    from skimage.transform import rescale
    filename = '/fs/ess/PAS0027/vortex_data/vortex/vorts01.data'
    blocksize = 24
    padding = 4
    
    data_high, name = read_vortex_data(filename, padding=padding, scale=1)
    
    x_cnt = int(128/(blocksize-2*padding))  
    y_cnt = int(128/(blocksize-2*padding)) 
    z_cnt = int(128/(blocksize-2*padding))
    print(x_cnt, y_cnt, z_cnt)
    bs = blocksize - 2*padding
    bs_low = bs // 2
    padding_low = padding // 2
    
    # as testing, high block-wise dowsample to low
    result = np.zeros(([64, 64, 64]))
    for idx in range(x_cnt * y_cnt * z_cnt):
        kk = int(idx / (x_cnt * y_cnt))
        jj = int((idx - kk * (x_cnt * y_cnt)) / x_cnt)
        ii = int(idx - kk * (x_cnt * y_cnt) - jj * x_cnt)
        block_high = data_high[kk*bs:kk*bs+blocksize, jj*bs:jj*bs+blocksize, ii*bs:ii*bs+blocksize]
        block_low = rescale(block_high, [1./2, 1./2, 1./2], order=3)
        print(block_low.shape)
        result[kk*bs_low:kk*bs_low+bs_low, jj*bs_low:jj*bs_low+bs_low, ii*bs_low:ii*bs_low+bs_low] += block_low[padding_low:-padding_low, padding_low:-padding_low, padding_low:-padding_low]
    
    # result = result.reshape(-1)
    # with open(os.path.join('./results/', name[:3]+f'_highblock2low.bin'), 'wb') as f:
    #     f.write(struct.pack('<%df' % len(result), *result))
    
    # as inference, high to low
    print(data_high.shape)
    data_low = rescale(data_high, [1./2, 1./2, 1./2], order=3)
    data_low = data_low[padding_low:-padding_low, padding_low:-padding_low, padding_low:-padding_low]
    # data_low = data_low.reshape(-1)
    # with open(os.path.join('./results/', name[:3]+f'_highwhole2low.bin'), 'wb') as f:
    #     f.write(struct.pack('<%df' % len(data_low), *data_low))
    result = np.abs(data_low-result)
    result = result.reshape(-1)
    with open(os.path.join('./results/', name[:3]+f'_rescale_dff_o3_nopad.bin'), 'wb') as f:
        f.write(struct.pack('<%df' % len(result), *result))

    
def find_high_res_nyx_ensemble():
    directory = '/fs/ess/PAS0027/nyx/256/output/'
    dirnames = os.listdir(directory)
    dirnames.sort()
    dirnames = dirnames[:800]
    # random.shuffle(dirnames)
    # find data with similar Hubble constant 
    dirnames = [dirname for dirname in dirnames if np.abs(float(dirname.split('_')[-1])-0.6) < 0.08][:8]
    print(dirnames, len(dirnames))
    min_ = 1e5
    max_ = -1e5
    sf = 2
    # std in high res
    # result_size = [256, 256, 256]
    # result_size.insert(0, 8)
    # result_all = np.zeros(result_size)

    for level in range(1, 5):
        ds_scale = np.power(sf, level) 
        for idx, dirname in enumerate(dirnames):
            filename = os.path.join(directory, dirname, 'Raw_plt256_00200', 'density.bin') 
            data, _ = read_nyx_data(filename)
            name = dirname
            data = np.log10(data)
            data_low = rescale(data, [1./ds_scale, 1./ds_scale, 1./ds_scale], order=2)
            print(idx, data.shape, data_low.shape)
            # print(name, np.min(data), np.max(data))
            # result_all[idx] = data
            data = data.reshape(-1)
            data_low = data_low.reshape(-1)
            with open(os.path.join('./data/density/nyx/', f'nyx_{name}.bin'), 'wb') as f:
                f.write(struct.pack('<%df' % len(data), *data))
            with open(os.path.join('./data/density/nyx/', f'nyx_{name}_low_level{level}.bin'), 'wb') as f:
                f.write(struct.pack('<%df' % len(data_low), *data_low))
        
        # result_all_std = np.std(result_all, axis=0).reshape(-1)
        # with open(os.path.join('./data/density/nyx/gt', 'std_0.08.bin'), 'wb') as f:
        #     f.write(struct.pack('<%df' % len(result_all_std), *result_all_std))
    
        #     if np.min(data) < min_:
        #         min_ = np.min(data)
        #     if np.max(data) > max_:
        #         max_ = np.max(data)
        # print(min_, max_) # 8.649637, 13.584945
        # exit()

def get_mean_ensemble_low_res():
    directory = './data/density/nyx/'
    filenames = os.listdir(directory)
    filenames.sort()
    # print(filenames)
    for level in range(1, 5):
        filenames_level = [filename for filename in filenames if f'level{level}' in filename]
        print(level, filenames_level, len(filenames_level))
        # get mean of each level
        result_size = np.array([256, 256, 256]) // (np.power(2, level)) 
        result_level = np.zeros(np.insert(result_size, 0, len(filenames_level)))
        for i, filename in enumerate(filenames_level):
            data, _ = read_data_bin(os.path.join(directory, filename), shape=result_size)
            print(filename, data.shape)
            result_level[i] = data.reshape(result_size)
        # take mean
        result_level_mean = np.mean(result_level, axis=0)
        print(result_level_mean.shape)
        result_level_mean = result_level_mean.reshape(-1)
        with open(os.path.join('./data/density/nyx/low_res', f'nyx_low_level{level}.bin'), 'wb') as f:
            f.write(struct.pack('<%df' % len(result_level_mean), *result_level_mean))
        


if __name__ == '__main__':
    # corr_px_pz()
    # particle_uncertainty_analysis()
    # particle_to_scalar()
    # test_lerp()
    # compute_ssim()
    test_seam()
    # find_high_res_nyx_ensemble()
    # get_mean_ensemble_low_res()


