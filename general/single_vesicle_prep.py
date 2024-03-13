#save single vesicle information for each celltype differently
if __name__ == '__main__':
    import numpy as np
    from syconn import global_params
    from syconn.reps.super_segmentation import  SuperSegmentationDataset
    from syconn.handler.config import initialize_logging
    from tqdm import tqdm
    from analysis_params import Analysis_Params

    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    global_params.wd = analysis_params.working_dir()
    f_name = analysis_params.file_locations
    log = initialize_logging('sort single vesicles into celltypes',
                             log_dir=f_name + '/logs/')
    with_glia = True
    log.info(f'Sort single vesicles into celltypes v1 of single vesicles, with_glia = {with_glia}')

    ves_wd = f'{global_params.wd}/single_vesicles/multi_class_pred/'
    #ves_wd = '/cajal/nvmescratch/projects/data/songbird/j0251/j0251_organell_seg/231216_single_vesicles_slurm/'
    log.info(f'wd = {ves_wd}')
    log.info('Load single vesicle data')
    single_ves_ids = np.load(f'{ves_wd}/ids.npy')
    single_ves_coords = np.load(f'{ves_wd}/rep_coords.npy')
    ves_map2ssvids = np.load(f'{ves_wd}/mapping_ssv_ids.npy')
    ves_dist2matrix = np.load(f'{ves_wd}/dist2matrix.npy')


    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    ssd = SuperSegmentationDataset(working_dir=global_params.wd)
    celltypes = ssd.load_numpy_data(analysis_params.celltype_key())
    cellids = ssd.ssv_ids

    log.info('Iterate over celltypes to sort single vesicle data')
    for ct in tqdm(ct_dict.keys()):
        ct_ids = cellids[celltypes == ct]
        ct_ind = np.in1d(ves_map2ssvids, ct_ids)
        ct_ves_ids = single_ves_ids[ct_ind]
        ct_ves_map2ssvids = ves_map2ssvids[ct_ind]
        ct_ves_dist2matrix = ves_dist2matrix[ct_ind]
        ct_ves_coords = single_ves_coords[ct_ind]
        np.save(f'{f_name}/{ct_dict[ct]}_ids.npy', ct_ves_ids)
        np.save(f'{f_name}/{ct_dict[ct]}_rep_coords.npy', ct_ves_coords)
        np.save(f'{f_name}/{ct_dict[ct]}_mapping_ssv_ids.npy', ct_ves_map2ssvids)
        np.save(f'{f_name}/{ct_dict[ct]}_dist2matrix.npy', ct_ves_dist2matrix)

    log.info('Numpy arrays for single vesicles per celltype done')



