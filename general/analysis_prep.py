if __name__ == '__main__':
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    from syconn.reps.segmentation import SegmentationDataset
    import numpy as np
    import time
    import os
    from syconn.handler.basics import load_pkl2obj, write_obj2pkl

    from analysis_prep_func import find_full_cells, synapse_amount_percell, get_axon_length_area_perct
    from syconn.handler.config import initialize_logging
    from multiprocessing import Process
    from wholebrain.scratch.arother.bio_analysis.general.analysis_morph_helper import get_compartment_length, \
        get_compartment_mesh_area
    from tqdm import tqdm

    #V3
    #global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"
    #V4
    global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    # celltypes: j0256: STN = 0, DA = 1, MSN = 2, LMAN = 3, HVC = 4, TAN = 5, GPe = 6, GPi = 7,
    #                      FS=8, LTS=9, NGF=10

    start = time.time()
    time_stamps = [time.time()]
    step_idents = ['t-0']
    f_name = "/wholebrain/scratch/arother/j0251v4_prep"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    syn_proba = 0.8
    min_syn_size = 0.1
    log = initialize_logging('analysis prep, syn_prob = %.2f, min syn size = %.2f' % (syn_proba, min_syn_size), log_dir=f_name + '/logs/')
    log.info("Step 0: Loading synapse data on all cells")
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    #ct_list = [0]
    #ax_list = [3, 4]
    #ct_list = [2,5, 6, 7, 0, 8, 9, 10]
    ax_list = [3, 4, 1]
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9:"LTS", 10:"NGF"}
    #ct_list = [6, 7, 0, 5, 8, 9, 10, 2]
    #ct_list = [2]
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
    time_stamps = [time.time()]
    step_idents = ["finished preparations"]
    '''
    for ix, ct in enumerate(ct_list):
        log.info('Step %.1i/%.1i find full cells of celltype %.3s' % (ix+1,len(ct_list), ct_dict[ct]))
        log.info("Get amount and sum of synapses")
        axon_syns, den_syns, soma_syns = synapse_amount_percell(celltype = ct, syn_cts = m_cts, syn_sizes = m_sizes, syn_ssv_partners = m_ssv_partners,
                                                                syn_axs = m_axs, axo_denso = True, all_comps = True)
        time_stamps = [time.time()]
        step_idents = ["per cell synapse data for celltype %s prepared" % ct_dict[ct]]
        log.info("Find full cells")
        cell_array, cell_dict = find_full_cells(ssd, celltype=ct)
        time_stamps = [time.time()]
        step_idents = ["full cells for celltype %s found" % ct_dict[ct]]
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
        dict_path = ("%s/full_%.3s_dict.pkl" % (f_name, ct_dict[ct]))
        arr_path = ("%s/full_%.3s_arr.pkl" % (f_name, ct_dict[ct]))
        #syn_dict = synapse_amount_percell(ct, sd_synssv, syn_proba=0.6, cellids=cell_array)
        #syn_path = ("%s/full_%.3s_synam.pkl" % (f_name,ct_dict[ct]))
        cell_dict = dict(cell_dict)
        write_obj2pkl(dict_path, cell_dict)
        write_obj2pkl(arr_path, cell_array)
        #write_obj2pkl(syn_path, syn_dict)
        time_stamps = [time.time()]
        step_idents = ["full cell dictionaries for celltype %s prepared" % ct_dict[ct]]
        log.info("full cell dictionaries for celltype %s prepared" % ct_dict[ct])


    for ia, axct in enumerate(ax_list):
        log.info('Step %.1i/%.1i find synapse amount of celltype %.3s' % (ia + 1, len(ax_list), ct_dict[axct]))
        cell_ids = ssd.ssv_ids[ssd.load_numpy_data("celltype_cnn_e3") == axct]
        axon_syns = synapse_amount_percell(celltype = axct, syn_cts = m_cts, syn_sizes = m_sizes, syn_ssv_partners = m_ssv_partners,
                                                                syn_axs = m_axs, axo_denso = True, all_comps = False)
        time_stamps = [time.time()]
        step_idents = ["per cell synapse data for celltype %s prepared" % ct_dict[axct]]
        log.info("Get axon length and surface area")
        axon_dict = get_axon_length_area_perct(ssd, celltype = axct)
        for axonid in list(axon_dict.keys()):
            try:
                axon_dict[axonid]["axon synapse amount"] = axon_syns[axonid]["amount"]
                axon_dict[axonid]["axon summed synapse size"] = axon_syns[axonid]["summed size"]
            except KeyError:
                axon_dict[axonid]["axon synapse amount"] = 0
                axon_dict[axonid]["axon summed synapse size"] = 0
        syn_path = ("%s/ax_%.3s_dict.pkl" % (f_name, ct_dict[axct]))
        axon_dict = dict(axon_dict)
        write_obj2pkl(syn_path, axon_dict)
        time_stamps = [time.time()]
        step_idents = ["axon dictionaries for celltype %s prepared" % ct_dict[axct]]
        log.info("axon dictionaries for celltype %s prepared" % ct_dict[axct])
    '''
    #prepare synapse amount and sum per cell but only from cells and axon fragments with certain lengths
    mcl = 200
    log.info("get all cells with minimum compartment length = %i" % mcl)
    mcl_cellids = []
    mcl_cellids_perct = {}
    cell_dicts = {}
    ct_list = list(ct_dict.keys())
    for i in ct_dict.keys():
        log.info("get full cells from %s" % ct_dict[i])
        if i in ax_list:
            cell_dicts[i] = load_pkl2obj("%s/ax_%.3s_dict.pkl" % (f_name, ct_dict[i]))
        else:
            cell_dicts[i] = load_pkl2obj("%s/full_%.3s_dict.pkl" % (f_name, ct_dict[i]))
        mcl_cells = np.zeros(len(cell_dicts[i].keys()))
        for ic, cellid in enumerate(tqdm(cell_dicts[i].keys())):
            if cell_dicts[i][cellid]["axon length"] < mcl:
                continue
            if i not in ax_list:
                if cell_dicts[i][cellid]["dendrite length"] < mcl:
                    continue
            mcl_cells[ic] = cellid
        mcl_cells = mcl_cells[mcl_cells > 0].astype(int)
        mcl_cellids_perct[i] = mcl_cells
        mcl_cellids.append(mcl_cells)
        time_stamps = [time.time()]
        step_idents = ["full cells with mcl %i for celltype %s prepared" % (mcl, ct_dict[i])]
        log.info("full cells with mcl %i for celltype %s prepared" % (mcl, ct_dict[i]))

    mcl_cellids = np.hstack(np.concatenate(np.array([mcl_cellids])))
    write_obj2pkl("%s/ct_dict_mcl_%i.pkl" % (f_name, mcl), mcl_cellids_perct)
    write_obj2pkl("%s/cellids_mcl_%i.pkl" % (f_name, mcl), mcl_cellids)

    time_stamps = [time.time()]
    step_idents = ["full cells with mcl %i for all celltypes prepared" % mcl]
    log.info("full cells with mcl %i for all celltypes prepared" % mcl)

    raise ValueError

    #save per cell synapse amount and summed synapse size
    log.info("get per cell synapse amount and summed synapse size only from cells with mcl = %i" % mcl)
    log.info("prepare synapse caches to exclude all cells without mcl")
    mcl_cell_inds = np.any(np.in1d(m_ssv_partners, mcl_cellids).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[mcl_cell_inds]
    m_ssv_partners = m_ssv_partners[mcl_cell_inds]
    m_axs = m_axs[mcl_cell_inds]
    m_sizes = m_sizes[mcl_cell_inds]
    for ic, ct in enumerate(ct_dict.keys()):
        log.info('Step %.1i/%.1i find full cells of celltype %.3s' % (ic + 1, len(ct_list), ct_dict[ct]))
        log.info("Step %.1i/%.1i: Get amount and sum of synapses per cell/axon of celltype %s" % (ic + 1, len(ct_list), ct_dict[ct]))
        if ct in ax_list:
            axon_syns = synapse_amount_percell(celltype=ct, syn_cts=m_cts, syn_sizes=m_sizes,
                                               syn_ssv_partners=m_ssv_partners,
                                               syn_axs=m_axs, axo_denso=True, all_comps=False)
            time_stamps = [time.time()]
            step_idents = ["per cell synapse data for celltype %s prepared" % ct_dict[ct]]
            for axonid in list(cell_dicts[ct].keys()):
                try:
                    cell_dicts[ct][axonid]["axon synapse amount %i" % mcl] = axon_syns[axonid]["amount"]
                    cell_dicts[ct][axonid]["axon summed synapse size %i" % mcl] = axon_syns[axonid]["summed size"]
                except KeyError:
                    cell_dicts[ct][axonid]["axon synapse amount %i" % mcl] = 0
                    cell_dicts[ct][axonid]["axon summed synapse size %i" % mcl] = 0
            syn_path = ("%s/ax_%.3s_dict.pkl" % (f_name, ct_dict[ct]))
            axon_dict = dict(cell_dicts[ct])
            write_obj2pkl(syn_path, cell_dicts[ct])
        else:
            axon_syns, den_syns, soma_syns = synapse_amount_percell(celltype=ct, syn_cts=m_cts, syn_sizes=m_sizes,
                                                                    syn_ssv_partners=m_ssv_partners,
                                                                    syn_axs=m_axs, axo_denso=True, all_comps=True)
            time_stamps = [time.time()]
            step_idents = ["per cell synapse data for celltype %s prepared" % ct_dict[ct]]
            log.info("Add to per cell dictionary")
            for cellid in list(cell_dicts[ct].keys()):
                try:
                    cell_dicts[ct][cellid]["axon synapse amount %i" % mcl] = axon_syns[cellid]["amount"]
                    cell_dicts[ct][cellid]["axon summed synapse size %i" % mcl ] = axon_syns[cellid]["summed size"]
                except KeyError:
                    cell_dicts[ct][cellid]["axon synapse amount %i" % mcl] = 0
                    cell_dicts[ct][cellid]["axon summed synapse size %i" % mcl] = 0
                try:
                    cell_dicts[ct][cellid]["dendrite synapse amount %i" % mcl] = den_syns[cellid]["amount"]
                    cell_dicts[ct][cellid]["dendrite summed synapse size %i" % mcl] = den_syns[cellid]["summed size"]
                except KeyError:
                    cell_dicts[ct][cellid]["dendrite synapse amount %i" % mcl] = 0
                    cell_dicts[ct][cellid]["dendrite summed synapse size %i" % mcl] = 0
                try:
                    cell_dicts[ct][cellid]["soma synapse amount %i" % mcl] = soma_syns[cellid]["amount"]
                    cell_dicts[ct][cellid]["soma summed synapse size %i" % mcl] = soma_syns[cellid]["summed size"]
                except KeyError:
                    cell_dicts[ct][cellid]["soma synapse amount %i" % mcl] = 0
                    cell_dicts[ct][cellid]["soma summed synapse size %i" % mcl] = 0
            dict_path = ("%s/full_%.3s_dict.pkl" % (f_name, ct_dict[ct]))
            cell_dict = dict(cell_dicts[ct])
            write_obj2pkl(dict_path, cell_dict)

        time_stamps = [time.time()]
        step_idents = ["cell dictionaries for celltype %s completed" % ct_dict[ct]]
        log.info("cell dictionaries for celltype %s completed" % ct_dict[ct])



