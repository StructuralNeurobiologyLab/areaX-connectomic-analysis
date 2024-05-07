if __name__ == '__main__':
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    from syconn.reps.segmentation import SegmentationDataset
    import numpy as np
    import time
    import os
    from syconn.handler.basics import load_pkl2obj, write_obj2pkl
    from analysis_params import Analysis_Params
    from analysis_prep_func import find_full_cells, synapse_amount_percell, get_axon_length_area_perct
    from syconn.handler.config import initialize_logging
    import pandas as pd

    #V3
    #global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"
    #V4
    #global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
    #v5
    #global_params.wd = '/cajal/nvmescratch/projects/data/songbird/j0251/j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_20230822'


    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    global_params.wd = analysis_params.working_dir()

    f_name = analysis_params.file_locations
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    syn_proba = 0.6
    min_syn_size = 0.1
    with_glia = False
    log = initialize_logging('analysis prep', log_dir=f_name + '/logs/')
    log.info(f'Data based on the working directory {global_params.wd} will be cached')
    log.info('Compared to v5 (agglo2) this involves; new celltype trainings (gt now involves INT1-3 instead of FS, NGF; also migrating neurons but old skeletons and compartments')
    log.info('syn_prob = %.2f, min syn size = %.2f, with_glia = %s' % (syn_proba, min_syn_size, with_glia))
    log.info("Step 0: Loading synapse data on all cells")
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.wd)

    # celltypes: j0256: STN = 0, DA = 1, MSN = 2, LMAN = 3, HVC = 4, TAN = 5, GPe = 6, GPi = 7,
    #                      FS=8, LTS=9, NGF=10, ASTRO = 11, OLIGO = 12, MICRO = 13, FRAG = 14
    #celltypes: 'v6': {0:'DA', 1:'LMAN', 2: 'HVC', 3:'MSN', 4:'STN', 5:'TAN', 6:'GPe', 7:'GPi', 8: 'LTS',
    #                      9:'INT1', 10:'INT2', 11:'INT3', 12:'ASTRO', 13:'OLIGO', 14:'MICRO', 15:'MIGR', 16:'FRAG'}
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    ax_list = analysis_params.axon_cts()
    ct_list = list(ct_dict.keys())
    ct_str = analysis_params.ct_str(with_glia=with_glia)
    celltype_key = analysis_params.celltype_key()
    log.info(f'Celltype key {celltype_key} is used with version {analysis_params._version}')
    syn_prob = sd_synssv.load_numpy_data("syn_prob")
    m = syn_prob > syn_proba
    m_cts = sd_synssv.load_numpy_data("partner_celltypes")[m]
    m_ssv_partners = sd_synssv.load_numpy_data("neuron_partners")[m]
    m_axs = sd_synssv.load_numpy_data("partner_axoness")[m]
    m_sizes = sd_synssv.load_numpy_data("mesh_area")[m] / 2
    size_inds = m_sizes > min_syn_size
    m_cts = m_cts[size_inds]
    m_axs = m_axs[size_inds]
    m_ssv_partners = m_ssv_partners[size_inds]
    m_sizes = m_sizes[size_inds]
    m_axs[m_axs == 3] = 1
    m_axs[m_axs == 4] = 1
    #create pd.Dataframe with information on how many cells with which length are there
    lengths_to_test = [100, 200, 500, 1000] #µm
    comps_to_test = [[f'axon length >= {i} µm', f'dendrite length >= {i} µm', f'axon and dendrite length >= {i} µm'] for i in lengths_to_test]
    columns = np.concatenate(comps_to_test)
    columns = np.hstack(['total', 'full cells', columns])
    cell_number_info = pd.DataFrame(columns = columns, index = ct_str)
    time_stamps = [time.time()]
    step_idents = ["finished preparations"]

    for ix, ct in enumerate(ct_list):
        if ct in ax_list:
            continue
        log.info(f'Step %.1i/%.1i find full cells of celltype {ct_dict[ct]}' % (ix+1,len(ct_list)))
        log.info("Get amount and sum of synapses")
        axon_syns, den_syns, soma_syns = synapse_amount_percell(celltype = ct, syn_cts = m_cts, syn_sizes = m_sizes, syn_ssv_partners = m_ssv_partners,
                                                                syn_axs = m_axs, axo_denso = True, all_comps = True)
        time_stamps = [time.time()]
        step_idents = ["per cell synapse data for celltype %s prepared" % ct_dict[ct]]
        log.info("Find full cells")
        cell_array, cell_dict = find_full_cells(ssd, celltype=ct, key=celltype_key)
        log.info(f'{len(cell_array)} full cells for celltype {ct_dict[ct]} found')
        time_stamps = [time.time()]
        step_idents = ["full cells (axon, dendrite, soma present) for celltype %s found" % ct_dict[ct]]
        log.info("Make per cell dictionary")
        for cellid in list(cell_dict.keys()):
            try:
                cell_dict[cellid]["axon synapse amount"] = axon_syns[cellid]["amount"]
                cell_dict[cellid]["axon summed synapse size"] = axon_syns[cellid]["summed size"]
            except KeyError:
                cell_dict[cellid]["axon synapse amount"] = 0
                cell_dict[cellid]["axon summed synapse size"] = 0
            try:
                cell_dict[cellid]["dendrite synapse amount"] = den_syns[cellid]["amount"]
                cell_dict[cellid]["dendrite summed synapse size"] = den_syns[cellid]["summed size"]
            except KeyError:
                cell_dict[cellid]["dendrite synapse amount"] = 0
                cell_dict[cellid]["dendrite summed synapse size"] = 0
            try:
                cell_dict[cellid]["soma synapse amount"] = soma_syns[cellid]["amount"]
                cell_dict[cellid]["soma summed synapse size"] = soma_syns[cellid]["summed size"]
            except KeyError:
                cell_dict[cellid]["soma synapse amount"] = 0
                cell_dict[cellid]["soma summed synapse size"] = 0
        dict_path = (f"{f_name}/full_{ct_dict[ct]}_dict.pkl")
        arr_path = (f"{f_name}/full_{ct_dict[ct]}_arr.pkl")
        cell_dict = dict(cell_dict)
        write_obj2pkl(dict_path, cell_dict)
        write_obj2pkl(arr_path, cell_array)
        step_idents = ["full cell dictionaries for celltype %s prepared" % ct_dict[ct]]
        log.info("full cell dictionaries for celltype %s prepared" % ct_dict[ct])
        log.info('Create statistics about cell number depending on axon or dendrite length')
        cell_number_info.loc[ct_dict[ct], 'total'] = len(ssd.ssv_ids[ssd.load_numpy_data(celltype_key) == ct])
        cell_number_info.loc[ct_dict[ct], 'full cells'] = len(cell_array)
        axon_lengths = np.array([cell_dict[ci]['axon length'] for ci in cell_array])
        dendrite_lengths = np.array([cell_dict[ci]['dendrite length'] for ci in cell_array])
        for test_length in lengths_to_test:
            cellids_axon_length_suitable = cell_array[axon_lengths >= test_length]
            cellids_dendrite_length_suitable = cell_array[dendrite_lengths >= test_length]
            cellids_both_suitable = cellids_axon_length_suitable[np.in1d(cellids_axon_length_suitable, cellids_dendrite_length_suitable)]
            cell_number_info.loc[ct_dict[ct], f'axon length >= {test_length} µm'] = len(cellids_axon_length_suitable)
            cell_number_info.loc[ct_dict[ct], f'dendrite length >= {test_length} µm'] = len(cellids_dendrite_length_suitable)
            cell_number_info.loc[ct_dict[ct], f'axon and dendrite length >= {test_length} µm'] = len(cellids_both_suitable)
        time_stamps = [time.time()]
        step_idents = ["cell number depending on axon/dendrite lengths %s prepared" % ct_dict[ct]]
        log.info("cell number depending on axon/dendrite lengths %s prepared" % ct_dict[ct])
    '''
    for ia, axct in enumerate(ax_list):
        log.info('Step %.1i/%.1i find synapse amount of celltype %.3s' % (ia + 1, len(ax_list), ct_dict[axct]))
        cell_ids = ssd.ssv_ids[ssd.load_numpy_data(celltype_key) == axct]
        axon_syns = synapse_amount_percell(celltype = axct, syn_cts = m_cts, syn_sizes = m_sizes, syn_ssv_partners = m_ssv_partners,
                                                                syn_axs = m_axs, axo_denso = True, all_comps = False)
        time_stamps = [time.time()]
        step_idents = ["per cell synapse data for celltype %s prepared" % ct_dict[axct]]
        log.info("Get axon length and surface area")
        axon_dict = get_axon_length_area_perct(ssd, celltype = axct, key=celltype_key)
        for axonid in list(axon_dict.keys()):
            try:
                axon_dict[axonid]["axon synapse amount"] = axon_syns[axonid]["amount"]
                axon_dict[axonid]["axon summed synapse size"] = axon_syns[axonid]["summed size"]
            except KeyError:
                axon_dict[axonid]["axon synapse amount"] = 0
                axon_dict[axonid]["axon summed synapse size"] = 0
        syn_path = (f"{f_name}/ax_{ct_dict[axct]}_dict.pkl")
        axon_dict = dict(axon_dict)
        write_obj2pkl(syn_path, axon_dict)
        time_stamps = [time.time()]
        step_idents = ["axon dictionaries for celltype %s prepared" % ct_dict[axct]]
        log.info("axon dictionaries for celltype %s prepared" % ct_dict[axct])
        log.info('Create statistics about axon number depending on length')
        axon_ids = np.array(list(axon_dict.keys()))
        cell_number_info.loc[ct_dict[axct], 'total'] = len(axon_ids)
        axon_lengths = np.array([axon_dict[ci]['axon length'] for ci in axon_ids])
        for test_length in lengths_to_test:
            cellids_axon_length_suitable = axon_ids[axon_lengths >= test_length]
            cell_number_info.loc[ct_dict[axct], f'axon length >= {test_length} µm'] = len(cellids_axon_length_suitable)
        time_stamps = [time.time()]
        step_idents = ["axon number depending on axon/dendrite lengths %s prepared" % ct_dict[axct]]
        log.info("axon number depending on axon/dendrite lengths %s prepared" % ct_dict[axct])
    '''
    cell_number_info.to_csv(f'{f_name}/cell_numbers.csv')
    time_stamps = [time.time()]
    log.info('Cell number infos saved, analysis finished')




