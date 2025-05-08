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

if __name__ == '__main__':

    version = 'v6'
    bio_params = Analysis_Params(version=version)
    global_params.wd = bio_params.working_dir()
    n_coords = 1000
    cube_size = 1024

    f_name = f'cajal/scratch/users/arother/bio_analysis_results/general/' \
                 f'250508_{version}_j0251_ECS_estimate_n{n_coords}_cs{cube_size}'
    if not os.path.exists(f_name):
        os.mkdir(f_name)

    log = initialize_logging('ECS_estimate_log', log_dir=f_name)
    log.info(f'Start ECS estimation with {n_coords} and cube size of {cube_size} voxels in each direction.')
    np.random.seed(42)

    kd_ssv_path = '/cajal/nvmescratch/projects/from_ssdscratch/songbird/j0251/segmentation/j0251_72_seg_20210127_agglo2/j0251_realigned.k.toml'

    log.info(f'Segmentation will be loaded from {kd_ssv_path}')

    kd_dataset = kds.KnossosDataset()
    kd_dataset.initialize_from_conf(kd_ssv_path)

    log.info(f'Step 1/X: Generate random coordinates with at least {cube_size} voxel distance to dataset')

    ecs_estimate_df = pd.DataFrame(columns = ['coord x', 'coord y', 'coord z', 'ECS estimate'], index = range(n_coords))
    dataset_boundaries = kd_dataset.boundary
    pot_x_coords = np.arange(cube_size, dataset_boundaries[0] - cube_size)
    pot_y_coords = np.arange(cube_size, dataset_boundaries[1] - cube_size)
    pot_z_coords = np.arange(cube_size, dataset_boundaries[2] - cube_size)
    rndm_x = np.random.choice(pot_x_coords, n_coords)
    rndm_y = np.random.choice(pot_y_coords, n_coords)
    rndm_z = np.random.choice(pot_z_coords, n_coords)
    ecs_estimate_df['coord x'] = rndm_x
    ecs_estimate_df['coord y'] = rndm_y
    ecs_estimate_df['coord z'] = rndm_z
    rndm_offsets = np.stack([rndm_x, rndm_y, rndm_z]).reshape(-1, 3)

    log.info(f'Step 2/3: Get ECS estimate for {n_coords} different random offsets')

    #To Do: multiprocess

    test_chunk = kd_dataset.load_seg(size = np.array([cube_size, cube_size, cube_size]), offset = rndm_offsets[0], mag = 1)
    voxel_number = test_chunk.size
    no_seg = test_chunk[test_chunk == 0]
    fraction_ecs = no_seg.shape[0] / voxel_number
    #eventually overlay with myelin segmentation but only available in mag = 4 or mag = 2?


    raise ValueError








    #get segmentation daatset



    #load segmentation of random cube in several threads


    #plot distribution over

