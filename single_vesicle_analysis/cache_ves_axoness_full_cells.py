#cache axoness of single vesicles for full cells

if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    from cajal.nvmescratch.users.arother.bio_analysis.general.vesicle_helper import map_axoness2ves
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import numpy as np
    from syconn.mp.mp_utils import start_multiprocess_imap

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    f_name = "/cajal/nvmescratch/users/arother/j0251v5_prep"
    log = initialize_logging('axoness full celltypes',
                             log_dir=f_name + '/logs/')
    with_glia = False
    log.info(f'Cache axoness of celltypes that are not projecting axons (DA, LMAN, HVC), with_glia = {with_glia}')
    log.info('Cache only for full cells')
    version = 'v6'
    analysis_params = Analysis_Params(working_dir=global_params.wd, version='v6')
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    #only use celltypes that are not projecting axons
    ct_types = analysis_params.load_celltypes_full_cells(with_glia=with_glia)
    cache_name = analysis_params.file_locations

    log.info('Iterate over celltypes to map vesicles to axons')
    for ct in ct_types:
        log.info(f'Now processing celltype {ct_dict[ct]}')
        #get cellids for celltype
        ct_ids = analysis_params.load_full_cell_array(ct)
        # load caches prefiltered for celltype
        log.info('Load caches for celltype')
        ves_ids = np.load(f'{cache_name}/{ct_dict[ct]}_ids.npy')
        ves_coords = np.load(f'{cache_name}/{ct_dict[ct]}_rep_coords.npy')
        ves_map2ssvids = np.load(f'{cache_name}/{ct_dict[ct]}_mapping_ssv_ids.npy')
        ves_dist2matrix = np.load(f'{cache_name}/{ct_dict[ct]}_dist2matrix.npy')
        log.info('Prefilter according to full cells')
        ct_ind = np.in1d(ves_map2ssvids, ct_ids)
        ct_ves_ids = ves_ids[ct_ind]
        ct_ves_map2ssvids = ves_map2ssvids[ct_ind]
        ct_ves_coords = ves_coords[ct_ind]
        ct_ves_dist2matrix = ves_dist2matrix[ct_ind]
        log.info('Get axoness for each vesicle in cells')
        ves_input = [[cellid, ct_ves_ids, ct_ves_coords, ct_ves_map2ssvids] for cellid in ct_ids]
        if ct == 2 or ct >= 8:
            # for unknown reason gives error in this case that i should be between -2**31 < i < 2**31 if not only running on one cpu
            output = start_multiprocess_imap(map_axoness2ves, ves_input, nb_cpus=1)
        else:
            output = start_multiprocess_imap(map_axoness2ves, ves_input)
        output = np.array(output, dtype='object')
        ves_ids_reordered = np.concatenate(output[:, 0])
        ves_axoness_reordered = np.concatenate(output[:, 1])
        #output is now ordered according to cell
        #reorder according to original ves_ids
        #logic: if two arrays x and y have the same unique numbers y = y[y.argsort()][y.argsort().argsort()] = x[a.argsort()][y.argsort().argsort()]
        #since in this case y[y.argsort()] = x[x.argsort()]
        ct_ves_axoness = ves_axoness_reordered[ves_ids_reordered.argsort()][ct_ves_ids.argsort().argsort()]
        log.info('Save axoness cache and caches for full cells')
        np.save(f'{f_name}/{ct_dict[ct]}_axoness_coarse_fullcells.npy', ct_ves_axoness)
        np.save(f'{f_name}/{ct_dict[ct]}_ids_fullcells.npy', ct_ves_ids)
        np.save(f'{f_name}/{ct_dict[ct]}_rep_coords_fullcells.npy', ct_ves_coords)
        np.save(f'{f_name}/{ct_dict[ct]}_mapping_ssv_ids_fullcells.npy', ct_ves_map2ssvids)
        np.save(f'{f_name}/{ct_dict[ct]}_dist2matrix_fullcells.npy', ct_ves_dist2matrix)
        log.info(f'Caching for celltype {ct_dict[ct]} finished')

    log.info(f'Caching for {len(ct_types)} celltypes done.')