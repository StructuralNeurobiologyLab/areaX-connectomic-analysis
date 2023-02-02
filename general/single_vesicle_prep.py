#save single vesicle information for each celltype differently
if __name__ == '__main__':
    import numpy as np
    from syconn import global_params
    from syconn.reps.super_segmentation import  SuperSegmentationDataset
    from syconn.handler.config import initialize_logging
    from tqdm import tqdm

    f_name = "/cajal/nvmescratch/users/arother/j0251v4_prep"
    log = initialize_logging('sort single vesicles into celltypes',
                             log_dir=f_name + '/logs/')
    log.info('Load single vesicle data')
    ves_wd = 'cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811/single_vesicles'
    single_ves_ids = np.load(f'{ves_wd}/ids.npy')
    single_ves_coords = np.load(f'{ves_wd}/rep_coords.npy')
    ves_map2ssvids = np.load(f'{ves_wd}/mapping_ssv_ids.npy')
    ves_dist2matrix = np.load(f'{ves_wd}/dist2matrix.npy')

    global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   10: "NGF"}
    celltypes = ssd.load_numpy_data('celltype_cnn_e3')
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

    log.info('Caches for single vesicles per celltype done')



