if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from analysis_morph_helper import map_axoness_cellid2org, map_cellid2org
    from analysis_params import Analysis_Params
    import numpy as np
    from syconn.mp.mp_utils import start_multiprocess_imap
    from syconn.reps.segmentation import SegmentationDataset

    #global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    global_params.wd = '/cajal/nvmescratch/projects/data/songbird/j0251/j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_20230822'
    version = 'v6'
    with_glia = False
    analysis_params = Analysis_Params(working_dir=global_params.wd, version='v6')
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    f_name = analysis_params.file_locations
    log = initialize_logging('axoness full celltypes',
                             log_dir=f_name + '/logs/')
    log.info(f'Cache axoness, cellid of celltypes that are not projecting axons (DA, LMAN, HVC), with_glia = {with_glia}')
    log.info('For projecting axons, only map cellids to organelles')
    log.info('Cache only for full cells')
    #only use celltypes that are not projecting axons
    num_cts = analysis_params.num_cts(with_glia=with_glia)
    cache_name = analysis_params.file_locations
    axon_cts = analysis_params.axon_cts()

    log.info('Iterate over celltypes to map mitos to compartments, celltypes and cellids')
    #load mitos
    sd_mi = SegmentationDataset('mi')
    mi_ids = sd_mi.ids
    mi_coords = sd_mi.load_numpy_data('rep_coord')
    mi_sizes = sd_mi.load_numpy_data('size')
    ct_types = np.arange(0, num_cts)[::-1]
    for ct in ct_types:
        log.info(f'Now processing celltype {ct_dict[ct]}')
        if ct in axon_cts:
            log.info('Get cellid for each mito in cells')
            cell_dict = analysis_params.load_cell_dict(ct)
            ct_ids = np.array(list(cell_dict.keys()))
            mito_input = [[cellid, mi_ids] for cellid in ct_ids]
            output = start_multiprocess_imap(map_cellid2org, mito_input)
            output = np.array(output, dtype='object')
            mito_ids_reordered = np.concatenate(output[:, 0])
            mito_cellids_reordered = np.concatenate(output[:, 1])
            #output is now ordered according to cell
            #reorder according to original mtio
            #logic: if two arrays x and y have the same unique numbers y = y[y.argsort()][y.argsort().argsort()] = x[a.argsort()][y.argsort().argsort()]
            #since in this case y[y.argsort()] = x[x.argsort()]
            ct_inds = np.in1d(mi_ids, mito_ids_reordered)
            ct_mi_ids = mi_ids[ct_inds]
            ct_mi_coords = mi_coords[ct_inds]
            ct_mi_sizes = mi_sizes[ct_inds]
            ct_mito_cellids = mito_cellids_reordered[mito_ids_reordered.argsort()][ct_mi_ids.argsort().argsort()]
            log.info('Save axoness cache and caches for projecting axons')
            np.save(f'{f_name}/{ct_dict[ct]}_mito_ids.npy', ct_mi_ids)
            np.save(f'{f_name}/{ct_dict[ct]}_mito_rep_coords.npy', ct_mi_coords)
            np.save(f'{f_name}/{ct_dict[ct]}_mito_sizes.npy', ct_mi_sizes)
            np.save(f'{f_name}/{ct_dict[ct]}_mito_mapping_ssv_ids.npy', ct_mito_cellids)
        else:
            log.info('Get cellid, axoness for each mito in cells')
            # get cellids for celltype
            ct_ids = analysis_params.load_full_cell_array(ct)
            mito_input = [[cellid, mi_ids, mi_coords] for cellid in ct_ids]
            output = start_multiprocess_imap(map_axoness_cellid2org, mito_input)
            output = np.array(output, dtype='object')
            mito_ids_reordered = np.concatenate(output[:, 0])
            mito_axoness_reordered = np.concatenate(output[:, 1])
            mito_cellids_reordered = np.concatenate(output[:, 2])
            #reorder according to original mtio
            ct_inds = np.in1d(mi_ids, mito_ids_reordered)
            ct_mi_ids = mi_ids[ct_inds]
            ct_mi_coords = mi_coords[ct_inds]
            ct_mi_sizes = mi_sizes[ct_inds]
            ct_mito_cellids = mito_cellids_reordered[mito_ids_reordered.argsort()][ct_mi_ids.argsort().argsort()]
            ct_mito_axoness = mito_axoness_reordered[mito_ids_reordered.argsort()][ct_mi_ids.argsort().argsort()]
            log.info('Save axoness cache and caches for full cells')
            np.save(f'{f_name}/{ct_dict[ct]}_mito_axoness_coarse_fullcells.npy', ct_mito_axoness)
            np.save(f'{f_name}/{ct_dict[ct]}_mito_ids_fullcells.npy', ct_mi_ids)
            np.save(f'{f_name}/{ct_dict[ct]}_mito_rep_coords_fullcells.npy', ct_mi_coords)
            np.save(f'{f_name}/{ct_dict[ct]}_mito_sizes_fullcells.npy', ct_mi_sizes)
            np.save(f'{f_name}/{ct_dict[ct]}_mito_mapping_ssv_ids_fullcells.npy', ct_mito_cellids)
        log.info(f'Caching for celltype {ct_dict[ct]} finished')

    log.info(f'Caching for {num_cts} celltypes done.')