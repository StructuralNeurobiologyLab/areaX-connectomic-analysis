#cache cs_ssv_0 for cellids used with 200 µm
if __name__ == '__main__':
    from analysis_morph_helper import check_comp_lengths_ct
    from analysis_params import Analysis_Params
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    import os as os
    import numpy as np
    from tqdm import tqdm

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    analysis_params = Analysis_Params(working_dir=global_params.wd, version='v5')
    ct_dict = analysis_params.ct_dict(with_glia=False)
    celltype_key = analysis_params.celltype_key()
    min_comp_len_ax = 50
    min_comp_len_cells = 200
    exclude_known_mergers = True
    f_name = "cajal/scratch/users/arother/bio_analysis_results/general/230817_cache_cs_ssv_mcl_%i_ax%i" % (
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
            all_suitable_ids.append(cellids_checked)
    
    all_suitable_ids = np.concatenate(all_suitable_ids)
    log.info(f'{len(all_suitable_ids)} cells/ axons found that fulfill criteria')
    
    log.info('Step 2/2: Filter and cache cs_ssvs')
    sd_cs_ssv = SegmentationDataset('cs_ssv')
    cs_ssv_ids = sd_cs_ssv.ids
    cs_ssv_mesh_areas = sd_cs_ssv.load_numpy_data('mesh_area')
    cs_ssv_mesh_bb = sd_cs_ssv.load_numpy_data('mesh_bb')
    cs_ssv_coords = sd_cs_ssv.load_numpy_data('rep_coord')
    cs_ssv_sizes = sd_cs_ssv.load_numpy_data('size')
    cs_ssv_partners = sd_cs_ssv.load_numpy_data('neuron_partners')
    log.info(f' In total there are {len(cs_ssv_ids)} cs_ssvs')
    #filter for only ones in suitable ids
    suit_ct_inds = np.all(np.in1d(cs_ssv_partners, all_suitable_ids).reshape(len(cs_ssv_partners), 2), axis=1)
    cs_ssv_ids = cs_ssv_ids[suit_ct_inds]
    cs_ssv_mesh_areas = cs_ssv_mesh_areas[suit_ct_inds]
    cs_ssv_mesh_bb = cs_ssv_mesh_bb[suit_ct_inds]
    cs_ssv_coords = cs_ssv_coords[suit_ct_inds]
    cs_ssv_sizes = cs_ssv_sizes[suit_ct_inds]
    cs_ssv_partners = cs_ssv_partners[suit_ct_inds]
    assert(np.all(np.in1d(np.unique(cs_ssv_partners), all_suitable_ids)))
    log.info(f'There are {len(cs_ssv_ids)} cs_ssv between suitable cellids')
    np.save(f'{analysis_params.file_locations}/cs_ssv_ids_filtered.npy', cs_ssv_ids)
    np.save(f'{analysis_params.file_locations}/cs_ssv_mesh_areas_filtered.npy', cs_ssv_mesh_areas)
    np.save(f'{analysis_params.file_locations}/cs_ssv_mesh_bbs_filtered.npy', cs_ssv_mesh_bb)
    np.save(f'{analysis_params.file_locations}/cs_ssv_coords_filtered.npy', cs_ssv_coords)
    np.save(f'{analysis_params.file_locations}/cs_ssv_sizes_filtered.npy', cs_ssv_sizes)
    np.save(f'{analysis_params.file_locations}/cs_ssv_neuron_partners_filtered.npy', cs_ssv_partners)

    log.info(f'Caching finished, saved at {analysis_params.file_locations}')