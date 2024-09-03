#map organell properties to compartments of full cells
#previously based on map_mit2axoness_full_cells
if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from analysis_morph_helper import map_axoness_cellid2org, map_cellid2org
    from analysis_params import Analysis_Params
    import numpy as np
    from syconn.mp.mp_utils import start_multiprocess_imap
    from syconn.reps.segmentation import SegmentationDataset

    version = 'v6'
    with_glia = False
    analysis_params = Analysis_Params(version='v6')
    global_params.wd = analysis_params.working_dir()
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    f_name = analysis_params.file_locations
    # organelles = 'mi', 'vc', 'er', 'golgi
    organell_keys = ['golgi']
    full_cell_only = True
    log = initialize_logging(f'{organell_keys}_full_cells',
                             log_dir=f_name + '/logs/')
    log.info(f'Cache axoness, cellid of celltypes that are full cells, with_glia = {with_glia}')
    #log.info('For projecting axons, only map cellids to organelles')
    #only use celltypes that are not projecting axons
    num_cts = analysis_params.num_cts(with_glia=with_glia)
    cache_name = analysis_params.file_locations
    axon_cts = analysis_params.axon_cts()

    log.info('Iterate over celltypes to map mitos to compartments, celltypes and cellids')
    ct_types = np.arange(0, num_cts)
    for ok in organell_keys:
        log.info(f'Load {ok} params')
        # load organelle data
        sd_org = SegmentationDataset(ok)
        org_ids = sd_org.ids
        org_coords = sd_org.load_numpy_data('rep_coord')
        org_sizes = sd_org.load_numpy_data('size')
        org_mesh_area = sd_org.load_numpy_data('mesh_area')
        log.info(f'Now mapping {ok} to axoness')
        for ct in ct_types:
            log.info(f'Now processing celltype {ct_dict[ct]}')
            if ct in axon_cts:
                if full_cell_only:
                    continue
                else:
                    log.info(f'Get cellid for each {ok} in cells')
                    cell_dict = analysis_params.load_cell_dict(ct)
                    ct_ids = np.array(list(cell_dict.keys()))
                    org_input = [[cellid, org_ids, ok] for cellid in ct_ids]
                    output = start_multiprocess_imap(map_cellid2org, org_input)
                    output = np.array(output, dtype='object')
                    org_ids_reordered = np.concatenate(output[:, 0])
                    org_cellids_reordered = np.concatenate(output[:, 1])
                    #output is now ordered according to cell
                    #reorder according to original mtio
                    #logic: if two arrays x and y have the same unique numbers y = y[y.argsort()][y.argsort().argsort()] = x[a.argsort()][y.argsort().argsort()]
                    #since in this case y[y.argsort()] = x[x.argsort()]
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
            else:
                log.info(f'Get cellid, axoness for each {ok} in cells')
                # get cellids for celltype
                ct_ids = analysis_params.load_full_cell_array(ct)
                org_input = [[cellid, org_ids, org_coords, ok] for cellid in ct_ids]
                output = start_multiprocess_imap(map_axoness_cellid2org, org_input)
                output = np.array(output, dtype='object')
                org_ids_reordered = np.concatenate(output[:, 0])
                org_axoness_reordered = np.concatenate(output[:, 1])
                org_cellids_reordered = np.concatenate(output[:, 2])
                #reorder according to original mtio
                ct_inds = np.in1d(org_ids, org_ids_reordered)
                ct_org_ids = org_ids[ct_inds]
                ct_org_coords = org_coords[ct_inds]
                ct_org_sizes = org_sizes[ct_inds]
                ct_org_mesh_areas = org_mesh_area[ct_inds]
                ct_org_cellids = org_cellids_reordered[org_ids_reordered.argsort()][ct_org_ids.argsort().argsort()]
                ct_org_axoness = org_axoness_reordered[org_ids_reordered.argsort()][ct_org_ids.argsort().argsort()]
                log.info('Save axoness cache and caches for full cells')
                np.save(f'{f_name}/{ct_dict[ct]}_{ok}_axoness_coarse_fullcells.npy', ct_org_axoness)
                np.save(f'{f_name}/{ct_dict[ct]}_{ok}_ids_fullcells.npy', ct_org_ids)
                np.save(f'{f_name}/{ct_dict[ct]}_{ok}_rep_coords_fullcells.npy', ct_org_coords)
                np.save(f'{f_name}/{ct_dict[ct]}_{ok}_sizes_fullcells.npy', ct_org_sizes)
                np.save(f'{f_name}/{ct_dict[ct]}_{ok}_mapping_ssv_ids_fullcells.npy', ct_org_cellids)
                np.save(f'{f_name}/{ct_dict[ct]}_{ok}_mesh_areas_fullcells.npy', ct_org_mesh_areas)
            log.info(f'Caching for celltype {ct_dict[ct]} finished')

    log.info(f'Caching for {num_cts} celltypes done.')