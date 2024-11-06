if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from analysis_morph_helper import map_axoness_cellid2org, map_cellid2org
    from analysis_params import Analysis_Params
    import numpy as np
    from syconn.mp.mp_utils import start_multiprocess_imap
    from syconn.reps.segmentation import SegmentationDataset

    version = 'v6'
    cts = [12, 13, 14, 15, 17]
    analysis_params = Analysis_Params(version='v6')
    global_params.wd = analysis_params.working_dir()
    if np.any(np.array(cts) > 12):
        with_glia = True
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    #add OPC to cell dict
    if 17 in cts:
        ct_dict[17] = 'OPC'
    f_name = analysis_params.file_locations
    # organelles = 'mi', 'vc', 'er', 'golgi
    organell_keys = ['golgi']
    handpicked = True
    log = initialize_logging(f'241106_{organell_keys}_map2cells',
                             log_dir=f_name + '/logs/')
    log.info(f'Cache cellid of celltypes that are full cells, with_glia = {with_glia}')
    ct_str_lst = [ct_dict[ct] for ct in cts]
    log.info(f'Celltypes to be processed are {ct_str_lst}')
    if handpicked:
        log.info('Will only be processed for handpicked cells')

    #log.info('For projecting axons, only map cellids to organelles')
    #only use celltypes that are not projecting axons
    cache_name = analysis_params.file_locations

    log.info('Iterate over celltypes to map organelles to compartments, celltypes and cellids')
    for ok in organell_keys:
        log.info(f'Load {ok} params')
        # load organelle data
        sd_org = SegmentationDataset(ok)
        org_ids = sd_org.ids
        org_coords = sd_org.load_numpy_data('rep_coord')
        org_sizes = sd_org.load_numpy_data('size')
        org_mesh_area = sd_org.load_numpy_data('mesh_area')
        log.info(f'Now mapping {ok} to cellids')
        for ct in cts:
            log.info(f'Now processing celltype {ct_dict[ct]}')
            if handpicked:
                if ct > 16:
                    ct_ids = analysis_params.load_handpicked_ids(ct, ct_dict = ct_dict)
                else:
                    ct_ids = analysis_params.load_handpicked_ids(ct)
            else:
                cell_dict = analysis_params.load_cell_dict(ct)
                ct_ids = np.array(list(cell_dict.keys()))
            log.info(f'{len(ct_ids)} will be processed')
            org_input = [[cellid, org_ids, ok] for cellid in ct_ids]
            output = start_multiprocess_imap(map_cellid2org, org_input)
            output = np.array(output, dtype='object')
            org_ids_reordered = np.concatenate(output[:, 0])
            org_cellids_reordered = np.concatenate(output[:, 1])
            # output is now ordered according to cell
            # reorder according to original mtio
            # logic: if two arrays x and y have the same unique numbers y = y[y.argsort()][y.argsort().argsort()] = x[a.argsort()][y.argsort().argsort()]
            # since in this case y[y.argsort()] = x[x.argsort()]
            ct_inds = np.in1d(org_ids, org_ids_reordered)
            ct_org_ids = org_ids[ct_inds]
            ct_org_coords = org_coords[ct_inds]
            ct_org_sizes = org_sizes[ct_inds]
            ct_org_mesh_areas = org_mesh_area[ct_inds]
            ct_org_cellids = org_cellids_reordered[org_ids_reordered.argsort()][ct_org_ids.argsort().argsort()]
            log.info('Save axoness cache and caches for projecting axons')
            np.save(f'{f_name}/{ct_dict[ct]}_{ok}_ids.npy', ct_org_ids)
            np.save(f'{f_name}/{ct_dict[ct]}_{ok}_rep_coords.npy', ct_org_coords)
            np.save(f'{f_name}/{ct_dict[ct]}_{ok}_sizes.npy', ct_org_sizes)
            np.save(f'{f_name}/{ct_dict[ct]}_{ok}_mapping_ssv_ids.npy', ct_org_cellids)
            np.save(f'{f_name}/{ct_dict[ct]}_{ok}_mesh_areas', ct_org_mesh_areas)

    log.info('Analysis finished.')