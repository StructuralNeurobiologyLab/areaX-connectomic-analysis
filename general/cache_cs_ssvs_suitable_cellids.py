#cache cs_ssv_0 for cellids used with 200 µm
if __name__ == '__main__':
    from analysis_morph_helper import check_comp_lengths_ct
    from analysis_conn_helper import get_ct_information_npy, get_contact_size_axoness_per_cell
    from analysis_params import Analysis_Params
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    import os as os
    import numpy as np
    from tqdm import tqdm
    from syconn.mp.mp_utils import start_multiprocess_imap
    from collections import ChainMap

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    analysis_params = Analysis_Params(working_dir=global_params.wd, version='v5')
    ct_dict = analysis_params.ct_dict(with_glia=False)
    celltype_key = analysis_params.celltype_key()
    min_comp_len_ax = 50
    min_comp_len_cells = 200
    exclude_known_mergers = True
    f_name = "cajal/scratch/users/arother/bio_analysis_results/general/230831_cache_cs_ssv_mcl_%i_ax%i" % (
    min_comp_len_cells, min_comp_len_ax)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    save_svg = True
    log = initialize_logging('cache current cs_ssv for full cells', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i for full cells, min_comp_len = %i for axons, mergers excluded = %s" % (
        min_comp_len_cells, min_comp_len_ax, exclude_known_mergers))

    #this script should iterate over all celltypes
    axon_cts = [1, 3, 4]
    log.info("Step 1/3: Load cell dicts and get suitable cellids")
    if exclude_known_mergers:
        known_mergers = analysis_params.load_known_mergers()
    celltypes = np.array([ct_dict[ct] for ct in ct_dict])
    num_cts = len(celltypes)
    all_suitable_ids = []
    full_cell_suitable_ids = []
    axon_ct_suitable_ids = []
    for ct in tqdm(range(num_cts)):
        ct_str = ct_dict[ct]
        if ct in axon_cts:
            cell_dict = analysis_params.load_cell_dict(ct)
            #get ids with min compartment length
            cellids = np.array(list(cell_dict.keys()))
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_ax, axon_only=True,
                              max_path_len=None)
            axon_ct_suitable_ids.append(cellids_checked)
        else:
            cell_dict = analysis_params.load_cell_dict(ct)
            cellids = np.array(list(cell_dict.keys()))
            if exclude_known_mergers:
                merger_inds = np.in1d(cellids, known_mergers) == False
                cellids = cellids[merger_inds]
            if ct == 2:
                misclassified_asto_ids = analysis_params.load_potential_astros()
                astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
                cellids = cellids[astro_inds]
            cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cells,
                                                axon_only=False,
                                                max_path_len=None)
            #new wd has celltype_cnn_e3 for same celltypes as agglo2 and celltype_pts_e3 for other celltypes
            full_cell_suitable_ids.append(cellids_checked)
        all_suitable_ids.append(cellids_checked)
    
    all_suitable_ids = np.concatenate(all_suitable_ids)
    axon_ct_suitable_ids = np.concatenate(axon_ct_suitable_ids)
    full_cell_suitable_ids = np.concatenate(full_cell_suitable_ids)
    assert(np.all(np.in1d(full_cell_suitable_ids, axon_ct_suitable_ids)) == False)
    assert(np.all(np.in1d(all_suitable_ids, np.hstack([axon_ct_suitable_ids, full_cell_suitable_ids]))))
    log.info(f'{len(all_suitable_ids)} cells/ axons found that fulfill criteria')
    
    log.info('Step 2/2: Filter and cache cs_ssvs')
    '''
    sd_cs_ssv = SegmentationDataset('cs_ssv')
    cs_ssv_ids = sd_cs_ssv.ids
    cs_ssv_mesh_areas = sd_cs_ssv.load_numpy_data('mesh_area')
    cs_ssv_mesh_bb = sd_cs_ssv.load_numpy_data('mesh_bb')
    cs_ssv_coords = sd_cs_ssv.load_numpy_data('rep_coord')
    cs_ssv_partners = sd_cs_ssv.load_numpy_data('neuron_partners')
    log.info(f' In total there are {len(cs_ssv_ids)} cs_ssvs')
    '''
    old_cs_ssv_loading = f'{global_params.wd}/old_cs_ssv/'
    cs_ssv_ids = np.load(f'{old_cs_ssv_loading}/ids.npy')
    cs_ssv_mesh_areas = np.load(f'{old_cs_ssv_loading}/mesh_areas.npy')
    cs_ssv_mesh_bb = np.load(f'{old_cs_ssv_loading}/mesh_bbs.npy')
    cs_ssv_coords = np.load(f'{old_cs_ssv_loading}/rep_coords.npy')
    cs_ssv_partners = np.load(f'{old_cs_ssv_loading}/neuron_partnerss.npy')
    #filter for only ones in suitable ids
    suit_ct_inds = np.all(np.in1d(cs_ssv_partners, all_suitable_ids).reshape(len(cs_ssv_partners), 2), axis=1)
    cs_ssv_ids = cs_ssv_ids[suit_ct_inds]
    cs_ssv_mesh_areas = cs_ssv_mesh_areas[suit_ct_inds]
    cs_ssv_mesh_bb = cs_ssv_mesh_bb[suit_ct_inds]
    cs_ssv_coords = cs_ssv_coords[suit_ct_inds]
    cs_ssv_partners = cs_ssv_partners[suit_ct_inds]
    assert(np.all(np.in1d(np.unique(cs_ssv_partners), all_suitable_ids)))
    log.info('Filtering for suitable cellids done')
    log.info('Will now create a numpy array with information about celltypes')
    #get celltype information
    ssd = SuperSegmentationDataset(working_dir=global_params.wd)
    full_cellids = ssd.ssv_ids
    full_celltypes = ssd.load_numpy_data('celltype_pts_e3')
    partner_celltypes = get_ct_information_npy(ssv_partners=cs_ssv_partners, cellids_array_full=full_cellids,
                                               celltype_array_full=full_celltypes, desired_ssv_ids=all_suitable_ids)
    log.info('Created celltype array')
    log.info('Will start getting information to map axoness values to each cs')
    #get axoness information  for full cells and save it
    input = [[cellid, cs_ssv_ids, cs_ssv_coords, cs_ssv_partners] for cellid in full_cell_suitable_ids]
    comp_output = start_multiprocess_imap(get_contact_size_axoness_per_cell, input)
    log.info('Per cell axoness information processed via multiprocessing, will now start writing it '
             'in one numpy array.')
    comp_output = np.array(comp_output, dtype=object)
    comp_output_dict = dict(ChainMap(*comp_output))
    cs_ssv_axoness = np.zeros((len(cs_ssv_ids), 2)) - 1
    compartments = {0: 'dendrite cs ids', 1:'axon cs ids', 2:'soma cs ids'}
    #first fill in ids for all axon fragments
    axon_inds = np.in1d(cs_ssv_partners, axon_ct_suitable_ids).reshape(len(cs_ssv_partners), 2)
    axon_ind_inds = np.where(axon_inds == True)
    cs_ssv_axoness[axon_ind_inds] = 1
    for cellid in tqdm(full_cell_suitable_ids):
        for comp in compartments.keys():
            comp_cell_cs_ssv_ids = comp_output_dict[cellid][compartments[comp]]
            comp_cell_inds = np.in1d(cs_ssv_ids, comp_cell_cs_ssv_ids)
            comp_cell_ssv_partners = cs_ssv_partners[comp_cell_inds]
            partner_inds = np.where(comp_cell_ssv_partners == cellid)
            cs_ssv_axoness[comp_cell_inds, partner_inds[1]] = comp
    unique_cs_ssv_axoness = np.unique(cs_ssv_axoness)
    assert(len(unique_cs_ssv_axoness) == 3)
    assert(-1 not in unique_cs_ssv_axoness)
    log.info('Axoness values mapping done')
    log.info(f'There are {len(cs_ssv_ids)} cs_ssv between suitable cellids')
    np.save(f'{analysis_params.file_locations}/cs_ssv_ids_filtered.npy', cs_ssv_ids)
    np.save(f'{analysis_params.file_locations}/cs_ssv_mesh_areas_filtered.npy', cs_ssv_mesh_areas)
    np.save(f'{analysis_params.file_locations}/cs_ssv_mesh_bbs_filtered.npy', cs_ssv_mesh_bb)
    np.save(f'{analysis_params.file_locations}/cs_ssv_coords_filtered.npy', cs_ssv_coords)
    np.save(f'{analysis_params.file_locations}/cs_ssv_neuron_partners_filtered.npy', cs_ssv_partners)
    np.save(f'{analysis_params.file_locations}/cs_ssv_celltypes_filtered.npy', partner_celltypes)
    np.save(f'{analysis_params.file_locations}/cs_ssv_axoness_filtered.npy', cs_ssv_axoness)

    log.info(f'Caching finished, saved at {analysis_params.file_locations}')