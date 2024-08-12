#get 3 synapses from each bin and each neuronal cell type (outgoing; axo-dendritic, 0.1 µm, only neurons
#with suitable length)
#probability bins: 0-0.2, 0.2 - 0.4, 0.4 - 0.6, 0.6 - 0.8, > 0.8
#do once for all synapses and once for only those with min syn size and specific cell types

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from sklearn.utils import shuffle

    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    global_params.wd = analysis_params.working_dir()
    with_glia = True
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    neuron_ct_dict = analysis_params.ct_dict(with_glia = False)
    min_comp_len_cell = 200
    min_comp_len_ax = 50
    min_syn_size = 0.1
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw', 'STNGPINTv6', 'AxTePkBrv6', 'TePkBrNGF', 'TeBKv6MSNyw'
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/240726_j0251{version}_rfc_syn_eval_mcl_%i_ax%i_ms_%f" % (
                 min_comp_len_cell, min_comp_len_ax, min_syn_size)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('rfc_syn_eval_prep_log', log_dir=f_name)
    log.info(f'min comp len cells = {min_comp_len_cell} µm, min comp len ax = {min_comp_len_ax} µm,'
             f'min syn size = {min_syn_size} µm²')
    celltypes = np.array([ct_dict[ct] for ct in neuron_ct_dict])
    num_cts = len(celltypes)
    ct_types = np.arange(0, num_cts)
    syn_prob_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    syn_prob_bin_labels = syn_prob_bins[1:]
    num_samples = 3
    total_num_samples = num_samples * num_cts * len(syn_prob_bin_labels)
    log.info(f'number samples per bin and celltype = {num_samples}, syn prob bins are {syn_prob_bins},'
             f'in total this will result in {total_num_samples} synapses per category')
    log.info('Categories will be once all synapses where one is axon and once only between neuronal cells with the min comp len, '
             'axo-dendritc or axo-axonic and min syn size')

    log.info('Step 1/3: Prepare evaluation table of all synapses without filtering, where one is axon')
    sd_synssv = SegmentationDataset('syn_ssv')
    syn_prob = sd_synssv.load_numpy_data("syn_prob")
    m_ids = sd_synssv.ids
    m_axs = sd_synssv.load_numpy_data("partner_axoness")
    m_axs[m_axs == 4] = 1
    m_axs[m_axs == 3] = 1
    m_cts = sd_synssv.load_numpy_data("partner_celltypes")
    m_ssv_partners = sd_synssv.load_numpy_data("neuron_partners")
    m_sizes = sd_synssv.load_numpy_data("mesh_area") / 2
    m_rep_coord = sd_synssv.load_numpy_data("rep_coord")
    #get rid of synapses without axon (otherwise can't control for outgoing cell type)
    ax_inds = np.any(np.in1d(m_axs, 1).reshape(len(m_axs), 2), axis = 1)
    syn_prob = syn_prob[ax_inds]
    m_axs = m_axs[ax_inds]
    m_sizes = m_sizes[ax_inds]
    m_ids = m_ids[ax_inds]
    m_ssv_partners = m_ssv_partners[ax_inds]
    m_rep_coord = m_rep_coord[ax_inds]
    m_cts = m_cts[ax_inds]
    np.random.seed(42)
    log.info(f'There are {len(m_ids)} synapses in total where one partner is an axon.')

    columns = ['coord x', 'coord y', 'coord z', 'syn id',
               'celltype 1', 'celltype 2', 'syn size',
               'syn prob', 'syn prob bin', 'category', 'cellid 1', 'cellid 2']
    rndm_syn_table = pd.DataFrame(columns = columns, index = range(total_num_samples * 2))
    rndm_syn_table.loc[0: total_num_samples - 1, 'category'] = 'all syns'
    rndm_syn_table.loc[total_num_samples: 2 * total_num_samples - 1, 'category'] = 'filtered syns'
    #get syn prob into the categories wanted
    syn_prob_cats = np.array(pd.cut(syn_prob, syn_prob_bins, right = False, labels = syn_prob_bin_labels))
    rndm_syn_ids = []
    for ct in tqdm(ct_types):
        #get only synapses where ct is presynaptic
        ct_inds = np.any(np.in1d(m_cts, ct).reshape(len(m_cts), 2), axis = 1)
        ct_syn_ids = m_ids[ct_inds]
        ct_syn_prob_cats = syn_prob_cats[ct_inds]
        ct_cts = m_cts[ct_inds]
        ct_axs = m_axs[ct_inds]
        testct = np.in1d(ct_cts, ct).reshape(len(ct_cts), 2)
        testax = np.in1d(ct_axs, 1).reshape(len(ct_cts), 2)
        pre_ct_inds = np.any(testct == testax, axis=1)
        ct_syn_ids = ct_syn_ids[pre_ct_inds]
        ct_syn_prob_cats = ct_syn_prob_cats[pre_ct_inds]
        for prob_bin in syn_prob_bin_labels:
            #get only those of this bin
            prob_ct_syn_ids = ct_syn_ids[ct_syn_prob_cats == prob_bin]
            #select three random ones
            rndm_ids = np.random.choice(prob_ct_syn_ids, 3, replace=False)
            rndm_syn_ids.append(rndm_ids)

    rndm_syn_ids = np.concatenate(rndm_syn_ids)
    rndm_inds = np.in1d(m_ids, rndm_syn_ids)
    rndm_coords = m_rep_coord[rndm_inds]
    rndm_syn_ids_arr = m_ids[rndm_inds]
    rndm_sizes = m_sizes[rndm_inds]
    rndm_axs = m_axs[rndm_inds]
    rndm_cts = m_cts[rndm_inds]
    rndm_ct_str_1 = [ct_dict[ct] for ct in rndm_cts[:, 0]]
    rndm_ct_str_2 = [ct_dict[ct] for ct in rndm_cts[:, 1]]
    rndm_cell_ids = m_ssv_partners[rndm_inds]
    rndm_syn_prob = syn_prob[rndm_inds]
    rndm_syn_prob_cat = syn_prob_cats[rndm_inds]
    rndm_syn_table.loc[0: total_num_samples - 1, 'syn id'] = rndm_syn_ids_arr
    rndm_syn_table.loc[0: total_num_samples - 1, 'syn size'] = rndm_sizes
    rndm_syn_table.loc[0: total_num_samples - 1, 'coord x'] = rndm_coords[:, 0]
    rndm_syn_table.loc[0: total_num_samples - 1, 'coord y'] = rndm_coords[:, 1]
    rndm_syn_table.loc[0: total_num_samples - 1, 'coord z'] = rndm_coords[:, 2]
    rndm_syn_table.loc[0: total_num_samples - 1, 'celltype 1'] = rndm_ct_str_1
    rndm_syn_table.loc[0: total_num_samples - 1, 'celltype 2'] = rndm_ct_str_2
    rndm_syn_table.loc[0: total_num_samples - 1, 'cellid 1'] = rndm_cell_ids[:, 0]
    rndm_syn_table.loc[0: total_num_samples - 1, 'cellid 2'] = rndm_cell_ids[:, 1]
    rndm_syn_table.loc[0: total_num_samples - 1, 'syn prob'] = rndm_syn_prob
    rndm_syn_table.loc[0: total_num_samples - 1, 'syn prob bin'] = rndm_syn_prob_cat

    log.info('Step 2/3: Iterate over each celltypes check min length')
    suitable_ids_dict = {}
    all_suitable_ids = []
    all_cell_dict = {}
    all_celltypes = []
    all_celltypes_num = []
    axon_cts = analysis_params.axon_cts()
    for ct in ct_types:
        # only get cells with min_comp_len, MSN with max_comp_len or axons with min ax_len
        ct_str = ct_dict[ct]
        cell_dict = analysis_params.load_cell_dict(ct)
        all_cell_dict[ct] = cell_dict
        cellids = np.array(list(cell_dict.keys()))
        if ct in axon_cts:
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_ax,
                                            axon_only=True, max_path_len=None)
        else:
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cell,
                                            axon_only=False, max_path_len=None)
        cellids = np.sort(cellids)
        suitable_ids_dict[ct] = cellids
        all_suitable_ids.append(cellids)
        all_celltypes.append([ct_dict[ct] for i in cellids])
        all_celltypes_num.append([[ct] for i in cellids])
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict[ct]))

    all_suitable_ids = np.concatenate(all_suitable_ids)
    all_celltypes = np.concatenate(all_celltypes)

    log.info('Step 3/3: Filter synapses and select random ids')
    #filter synapses for suitable ids
    suit_ct_inds = np.any(np.in1d(m_ssv_partners, all_suitable_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[suit_ct_inds]
    m_ids = m_ids[suit_ct_inds]
    m_ssv_partners = m_ssv_partners[suit_ct_inds]
    m_sizes = m_sizes[suit_ct_inds]
    m_axs = m_axs[suit_ct_inds]
    m_rep_coord = m_rep_coord[suit_ct_inds]
    syn_prob = syn_prob[suit_ct_inds]
    syn_prob_cats = syn_prob_cats[suit_ct_inds]
    #filter for min synapse size
    size_inds = m_sizes > min_syn_size
    m_cts = m_cts[size_inds]
    m_ids = m_ids[size_inds]
    m_ssv_partners = m_ssv_partners[size_inds]
    m_sizes = m_sizes[size_inds]
    m_axs = m_axs[size_inds]
    m_rep_coord = m_rep_coord[size_inds]
    syn_prob = syn_prob[size_inds]
    syn_prob_cats = syn_prob_cats[size_inds]
    #filter for axo-dendritic
    #only check if no axo-axonic ones in there, that axon must be there already from above
    den_so = np.array([0, 2])
    den_so_inds = np.any(np.in1d(m_axs, den_so).reshape(len(m_axs), 2), axis=1)
    m_cts = m_cts[den_so_inds]
    m_ids = m_ids[den_so_inds]
    m_axs = m_axs[den_so_inds]
    m_ssv_partners = m_ssv_partners[den_so_inds]
    m_sizes = m_sizes[den_so_inds]
    m_rep_coord = m_rep_coord[den_so_inds]
    syn_prob = syn_prob[den_so_inds]
    syn_prob_cats = syn_prob_cats[den_so_inds]
    log.info(f'There are now {len(m_ids)} axo-dendritic or axo-somatic synapses with a minimal syn size of {min_syn_size} µm,'
             f'that come from {len(all_suitable_ids)} cell ids.')
    rndm_syn_ids = []
    #select radom ids
    for ct in tqdm(ct_types):
        #get only synapses where ct is presynaptic
        ct_inds = np.any(np.in1d(m_cts, ct).reshape(len(m_cts), 2), axis = 1)
        ct_syn_ids = m_ids[ct_inds]
        ct_syn_prob_cats = syn_prob_cats[ct_inds]
        ct_cts = m_cts[ct_inds]
        ct_axs = m_axs[ct_inds]
        testct = np.in1d(ct_cts, ct).reshape(len(ct_cts), 2)
        testax = np.in1d(ct_axs, 1).reshape(len(ct_cts), 2)
        pre_ct_inds = np.any(testct == testax, axis=1)
        ct_syn_ids = ct_syn_ids[pre_ct_inds]
        ct_syn_prob_cats = ct_syn_prob_cats[pre_ct_inds]
        for prob_bin in syn_prob_bin_labels:
            #get only those of this bin
            prob_ct_syn_ids = ct_syn_ids[ct_syn_prob_cats == prob_bin]
            #select three random ones
            if len(prob_ct_syn_ids) > 3:
                rndm_ids = np.random.choice(prob_ct_syn_ids, 3, replace=False)
            else:
                rndm_ids = prob_ct_syn_ids
                log.info(f'for cell type {ct_dict[ct]} in category {prob_bin} only {len(prob_ct_syn_ids)} exist, so'
                         f'these were used for evaluation')
            rndm_syn_ids.append(rndm_ids)

    #if not enough cells with all the categories select additional random cells from that category
    rndm_syn_ids = np.concatenate(rndm_syn_ids)
    rndm_inds = np.in1d(m_ids, rndm_syn_ids)
    rndm_syn_prob_cat = syn_prob_cats[rndm_inds]
    if len(rndm_syn_ids) < total_num_samples:
        expected_number_bins = total_num_samples / len(syn_prob_bin_labels)
        count_bins, bins = np.histogram(rndm_syn_prob_cat, bins = np.hstack([syn_prob_bin_labels, 1]))
        missing_syn_prob_cats = np.array(syn_prob_bin_labels)[np.where(count_bins < expected_number_bins)[0]]
        for mi, ms_prob_cat in enumerate(missing_syn_prob_cats):
            prob_syn_inds = syn_prob_cats == ms_prob_cat
            prob_syn_ids = m_ids[prob_syn_inds]
            #remove all syn_ids already in rndm_syn_ids
            chosen_inds = np.in1d(prob_syn_ids, rndm_syn_ids) == False
            prob_syn_ids = prob_syn_ids[chosen_inds]
            num_missing = int(expected_number_bins - count_bins[mi])
            if len(prob_syn_ids) > num_missing:
                rndm_ids = np.random.choice(prob_ct_syn_ids, num_missing, replace=False)
            else:
                rndm_ids = prob_syn_ids
                log.info(f'for category {ms_prob_cat} only {len(prob_syn_ids)} exist that are not in the selected synapses already, so '
                         f'all of these were added. This results in this category only having {len(prob_syn_ids) + count_bins[mi]} synapses.')
            rndm_syn_ids = np.hstack([rndm_syn_ids, rndm_ids])
            log.info(f'{len(rndm_ids)} ids in {ms_prob_cat} randomly appended')
        rndm_inds = np.in1d(m_ids, rndm_syn_ids)
        rndm_syn_prob_cat = syn_prob_cats[rndm_inds]

    if len(rndm_syn_ids) < total_num_samples:
        num_all_samples = total_num_samples + len(rndm_syn_ids)
    else:
        num_all_samples = total_num_samples * 2
    rndm_coords = m_rep_coord[rndm_inds]
    rndm_syn_ids_arr = m_ids[rndm_inds]
    rndm_sizes = m_sizes[rndm_inds]
    rndm_axs = m_axs[rndm_inds]
    rndm_cts = m_cts[rndm_inds]
    rndm_ct_str_1 = [ct_dict[ct] for ct in rndm_cts[:, 0]]
    rndm_ct_str_2 = [ct_dict[ct] for ct in rndm_cts[:, 1]]
    rndm_cell_ids = m_ssv_partners[rndm_inds]
    rndm_syn_prob = syn_prob[rndm_inds]
    rndm_syn_table.loc[total_num_samples: num_all_samples - 1, 'syn id'] = rndm_syn_ids_arr
    rndm_syn_table.loc[total_num_samples: num_all_samples - 1, 'syn size'] = rndm_sizes
    rndm_syn_table.loc[total_num_samples: num_all_samples - 1, 'coord x'] = rndm_coords[:, 0]
    rndm_syn_table.loc[total_num_samples: num_all_samples - 1, 'coord y'] = rndm_coords[:, 1]
    rndm_syn_table.loc[total_num_samples: num_all_samples - 1, 'coord z'] = rndm_coords[:, 2]
    rndm_syn_table.loc[total_num_samples: num_all_samples - 1, 'celltype 1'] = rndm_ct_str_1
    rndm_syn_table.loc[total_num_samples: num_all_samples - 1, 'celltype 2'] = rndm_ct_str_2
    rndm_syn_table.loc[total_num_samples: num_all_samples - 1, 'cellid 1'] = rndm_cell_ids[:, 0]
    rndm_syn_table.loc[total_num_samples: num_all_samples - 1, 'cellid 2'] = rndm_cell_ids[:, 1]
    rndm_syn_table.loc[total_num_samples: num_all_samples - 1, 'syn prob'] = rndm_syn_prob
    rndm_syn_table.loc[total_num_samples: num_all_samples - 1, 'syn prob bin'] = rndm_syn_prob_cat

    rndm_syn_table = rndm_syn_table.dropna(ignore_index=True)
    rndm_syn_table = shuffle(rndm_syn_table)
    rndm_syn_table = rndm_syn_table.reset_index(drop=True)
    rndm_syn_table.to_csv(f'{f_name}/random_syn_coords_rfc_eval.csv')

    log.info('Analysis finished.')


