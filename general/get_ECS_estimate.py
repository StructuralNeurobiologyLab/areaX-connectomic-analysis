#get estimate on ECS in dataset
#estimate ECS based on number of voxels that are not segmentation
#do for n random cubes with size m

import sys
from multiprocessing import set_start_method
import knossos_utils as kds
import numpy as np
import os as os
import numba
import pandas as pd
from skimage.measure import regionprops
from scipy.ndimage import distance_transform_edt
import scipy.ndimage.filters as filters
from syconn.handler.config import initialize_logging
from syconn.mp.mp_utils import start_multiprocess_imap
from multiprocessing import Pool
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from syconn import global_params
from analysis_params import Analysis_Params

#from vesicle_seg vesicle coord extration
class DummyFile:
    def write(self, x):
        pass

def load_chunk_data(kd_path, size, offset, mag, cube_shape = None, load_seg = True):
    '''
    Loads data from a chunk of the knossos dataset with supressing its print statements
    :param kd_path: path to config file
    :param size: chunk size
    :param offset: offset
    :param mag: magnification
    :param cube_shape: cube_shape (either [256, 256, 256] or [128, 128, 128]
    :param load_seg: if True loads segmentation layer, if not loads raw layer (binary class often in raw)
    :return:
    '''
    kd_dataset = kds.KnossosDataset()
    kd_dataset.initialize_from_conf(kd_path)
    if cube_shape is not None:
        kd_dataset._cube_shape = cube_shape
    sys.stdout = DummyFile()
    if load_seg:
        chunk_data = kd_dataset.load_seg(size= size, offset = offset, mag = mag)
    else:
        chunk_data = kd_dataset.load_raw(size=size, offset=offset, mag=mag)
    sys.stdout = sys.__stdout__
    return chunk_data

def estimate_ecs_per_cube(cube_input):
    '''
    Get per cube estimate of ECS by getting all coordinates that are not segmentation.
    cube_input: path to segemtnation, cube_size, offset
    return: fraction of ECS in cube
    '''
    seg_path, cube_size, offset = cube_input

    chunk = load_chunk_data(kd_path=seg_path, size=([cube_size, cube_size, cube_size]),
                    offset=offset, mag=1, cube_shape=None, load_seg=True)
    voxel_number = chunk.size
    no_seg = chunk[chunk == 0]
    fraction_ecs = no_seg.shape[0] / voxel_number
    return fraction_ecs



if __name__ == '__main__':

    version = 'v6'
    bio_params = Analysis_Params(version=version)
    global_params.wd = bio_params.working_dir()
    n_coords = 1000
    cube_size = 512
    bins = 50

    f_name = f'cajal/scratch/users/arother/bio_analysis_results/general/' \
                 f'250516_{version}_j0251_ECS_estimate_n{n_coords}_cs{cube_size}_bins{bins}'
    if not os.path.exists(f_name):
        os.mkdir(f_name)

    log = initialize_logging('ECS_estimate_log', log_dir=f_name)
    log.info(f'Start ECS estimation with {n_coords} and cube size of {cube_size} voxels in each direction.')
    np.random.seed(42)

    kd_ssv_path = '/cajal/nvmescratch/projects/from_ssdscratch/songbird/j0251/segmentation/j0251_72_seg_20210127_agglo2/j0251_realigned.k.toml'

    log.info(f'Segmentation will be loaded from {kd_ssv_path}')

    kd_dataset = kds.KnossosDataset()
    kd_dataset.initialize_from_conf(kd_ssv_path)

    log.info(f'Step 1/3: Generate random coordinates with at least {cube_size} voxel distance to dataset')

    ecs_estimate_df = pd.DataFrame(columns = ['coord x', 'coord y', 'coord z', 'ECS estimate'], index = range(n_coords))
    dataset_boundaries = kd_dataset.boundary
    #also remove 15 µm off each side to be sure not in area on border where no data is anymore
    dataset_border_gap = 15000 / kd_dataset.scale
    pot_x_coords = np.arange(dataset_border_gap[0], dataset_boundaries[0] - cube_size - dataset_border_gap[0])
    pot_y_coords = np.arange(dataset_border_gap[0], dataset_boundaries[1] - cube_size - dataset_border_gap[1])
    pot_z_coords = np.arange(dataset_border_gap[0], dataset_boundaries[2] - cube_size - dataset_border_gap[2])
    rndm_x = np.random.choice(pot_x_coords, n_coords)
    rndm_y = np.random.choice(pot_y_coords, n_coords)
    rndm_z = np.random.choice(pot_z_coords, n_coords)
    ecs_estimate_df['coord x'] = rndm_x
    ecs_estimate_df['coord y'] = rndm_y
    ecs_estimate_df['coord z'] = rndm_z
    rndm_offsets = np.stack([rndm_x, rndm_y, rndm_z]).transpose()

    log.info(f'Step 2/3: Get ECS estimate for {n_coords} different random offsets')
    # eventually overlay with myelin segmentation but only available in mag = 4 or mag = 2?
    cube_inputs = [[kd_ssv_path, cube_size, offset] for offset in rndm_offsets]
    ecs_fractions = start_multiprocess_imap(estimate_ecs_per_cube, cube_inputs)
    ecs_fractions = np.array(ecs_fractions)

    ecs_estimate_df['ECS estimate'] = ecs_fractions
    ecs_estimate_df.to_csv(f'{f_name}/ecs_estimates.csv')
    # raise ValueError

    log.info('Step 3/3: Plot results')
    median = np.median(ecs_fractions)
    mean = np.mean(ecs_fractions)
    std = np.std(ecs_fractions)

    log.info(f' Median ECS fraction = {median:.4f}, mean = {mean:.4f}, std = {std:.4f} ')

    fontsize = 20
    log.info(f'Step 3/3: Plot distribution')
    sns.histplot(data=ecs_estimate_df, x='ECS estimate', fill=False,
                 kde=False, element='step', color = 'black', linewidth=3, bins=bins)
    plt.ylabel('number of cubes', fontsize=fontsize)
    plt.xlabel('fraction of ECS', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.title('Fraction of ecs')
    plt.savefig(f'{f_name}/ecs_fractions.png')
    plt.savefig(f'{f_name}/ecs_fractions.svg')
    plt.close()
    
    sns.histplot(data=ecs_estimate_df, x='ECS estimate', fill=False,
                 kde=False, element='step', color = 'black', linewidth=3, bins=bins, stat='percent')
    plt.ylabel('% of cubes', fontsize=fontsize)
    plt.xlabel('fraction of ECS', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.title('Fraction of ecs')
    plt.savefig(f'{f_name}/ecs_fractions_perc.png')
    plt.savefig(f'{f_name}/ecs_fractions_perc.svg')
    plt.close()
    
    log.info('Analysis finished')














    #get segmentation daatset



    #load segmentation of random cube in several threads


    #plot distribution over

