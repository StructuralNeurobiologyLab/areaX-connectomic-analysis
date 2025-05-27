#similar to recurrent_conn_between2cts but only for one celltype

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors, CompColors
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_general, filter_synapse_caches_for_ct,get_ct_syn_number_sumsize, get_percell_number_sumsize
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    import os as os
    import pandas as pd
    import numpy as np
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    from tqdm import tqdm
    from scipy.stats import ranksums, kruskal
    import seaborn as sns
    import matplotlib.pyplot as plt

    version = 'v6'
    bio_params = Analysis_Params(version = version)
    ct_dict = bio_params.ct_dict(with_glia=False)
    global_params.wd = bio_params.working_dir()
    #min_comp_len = bio_params.min_comp_length()
    min_comp_len_cell = 200
    syn_prob = 0.6
    min_syn_size = 0.1
    exclude_known_mergers = True
    #color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw', 'STNGP'}
    color_key = 'STNGPINTv6'
    ct1 = 7
    ct1_str = ct_dict[ct1]
    fontsize = 20
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/250520_j0251{version}_{ct1_str}_recurr_conn_mcl_%i_synprob_%.2f_%s_fs%i" % (
    min_comp_len_cell, syn_prob, color_key, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('recurr_conn_log', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i for full cells, syn_prob = %.1f, min_syn_size = %.1f, known mergers excluded = %s" % (
        min_comp_len_cell, syn_prob, min_syn_size, exclude_known_mergers))
    log.info(f'Cell-specific connectivity within {ct1_str}')
    log.info('Goal is to see if input comes from same cells, output goes to')
    cls = CelltypeColors(ct_dict=ct_dict)
    ct_palette = cls.ct_palette(key=color_key)

    log.info('Step 1/6: Get suitable cellids')
    known_mergers = bio_params.load_known_mergers()
    #misclassified_asto_ids = bio_params.load_potential_astros()
    cell_dict = bio_params.load_cell_dict(ct1)
    # get ids with min compartment length
    cellids = np.array(list(cell_dict.keys()))
    if exclude_known_mergers:
        merger_inds = np.in1d(cellids, known_mergers) == False
        cellids = cellids[merger_inds]
        #astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
        #cellids = cellids[astro_inds]
    ct1_cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cell,
                                                axon_only=False,
                                                max_path_len=None)

    log.info('Step 2/6: Filter synapses for celltypes')
    # prefilter synapses for synapse prob thresh and min syn size
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob = filter_synapse_caches_general(
        sd_synssv,
        syn_prob_thresh=syn_prob,
        min_syn_size=min_syn_size)
    #make sure synapses only between suitable ids
    suit_ct_inds = np.all(np.in1d(m_ssv_partners, ct1_cellids).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[suit_ct_inds]
    m_ids = m_ids[suit_ct_inds]
    m_ssv_partners = m_ssv_partners[suit_ct_inds]
    m_sizes = m_sizes[suit_ct_inds]
    m_axs = m_axs[suit_ct_inds]
    m_spiness = m_spiness[suit_ct_inds]
    m_rep_coord = m_rep_coord[suit_ct_inds]
    syn_prob = syn_prob[suit_ct_inds]
    #make sure only axo-dendritic
    synapse_cache = [m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob]
    log.info(f'Step 3/6: Get number and sumsize per cell for {ct1_str} synapses')
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(pre_cts=[ct1],
                                                                                                        post_cts=None,
                                                                                                        syn_prob_thresh=None,
                                                                                                        min_syn_size=None,
                                                                                                        axo_den_so=True,
                                                                                                        synapses_caches=synapse_cache)
    log.info(f'Total synaptic strength within {ct1_str} is {np.sum(m_sizes):.2f} µm² from {len(m_sizes)} synapses')
    #get per cell syn numbers pre_synaptic
    sort_inds = np.where(m_axs == 1)
    pre_ssvs = m_ssv_partners[sort_inds]
    pre_syn_numbers, pre_syn_sizes, pre_ssvs_unique = get_percell_number_sumsize(ssvs = pre_ssvs, syn_sizes = m_sizes)

    log.info(
        f'{len(pre_ssvs_unique)} {ct1_str} project within this cell type. These are {100 * len(pre_ssvs_unique) / len(ct1_cellids):.2f}'
        f' percent of {ct1_str} cells')
    log.info(
        f'The median number of synapses are {np.median(pre_syn_numbers)}, sum size {np.median(pre_syn_sizes):.2f} per cell')
    write_obj2pkl(f'{f_name}/{ct1_str}_pre_ids.pkl', pre_ssvs_unique)
    #iterate over each ct1 cell and get number of ct1 cells it connects to with the corresponding summed synapse size
    pre_ids_dict = {}
    pre_sumsizes_dict = {}
    pre_syn_numbers_dict = {}
    pre_cell_number = np.zeros(len(pre_ssvs_unique))
    for ii, ct1_cellid in enumerate(pre_ssvs_unique):
        #get only synapses for this cellid
        ind = np.where(pre_ssvs == ct1_cellid)[0]
        id_partners = m_ssv_partners[ind]
        id_sizes = m_sizes[ind]
        ind = np.where(id_partners != ct1_cellid)
        post_partners = id_partners[ind]
        id_post_numbers, id_post_sizes, id_post_partners = get_percell_number_sumsize(post_partners, id_sizes)
        pre_cell_number[ii] = len(id_post_partners)
        pre_ids_dict[ct1_cellid] = id_post_partners
        pre_syn_numbers_dict[ct1_cellid] = id_post_numbers
        pre_sumsizes_dict[ct1_cellid] = id_post_sizes
    log.info(
        f'A median {ct1_str} projects to {np.median(pre_cell_number)} cells, with {np.median(pre_syn_numbers / pre_cell_number):.2f}'
        f' synapses and {np.median(pre_syn_sizes / pre_cell_number):.2f} synaptic area in µm²')

    write_obj2pkl(f'{f_name}/{ct1_str}_pre_ids.pkl', pre_ids_dict)
    write_obj2pkl(f'{f_name}/{ct1_str}_pre_syn_numbers.pkl', pre_syn_numbers_dict)
    write_obj2pkl(f'{f_name}/{ct1_str}_pre_syn_sumsizes.pkl', pre_sumsizes_dict)

    #plot numbers for presynaptic side
    sns.histplot(data=pre_cell_number, color=ct_palette[ct1_str], common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, stat='percent')
    plt.ylabel(f'% of {ct1_str} cells', fontsize=fontsize)
    plt.xlabel(f'number of postsynaptic partners', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(f'{ct1_str} presynaptic connectivity')
    plt.savefig(f'{f_name}/{ct1_str}_pre_num_partners_hist_perc.png')
    plt.savefig(f'{f_name}/{ct1_str}_pre_num_partners_hist_perc.svg')
    plt.close()
    sns.histplot(data=pre_cell_number, color=ct_palette[ct1_str], common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True)
    plt.ylabel(f'number of {ct1_str} cells', fontsize=fontsize)
    plt.xlabel(f'number of postsynaptic partners', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(f'{ct1_str} presynaptic connectivity')
    plt.savefig(f'{f_name}/{ct1_str}_pre_num_partners_hist.png')
    plt.savefig(f'{f_name}/{ct1_str}_pre_num_partners_hist.svg')
    plt.close()
    #get postsynaptic side
    sort_inds = np.where(m_axs != 1)
    post_ssvs = m_ssv_partners[sort_inds]
    post_syn_numbers, post_syn_sizes, post_ssvs_unique = get_percell_number_sumsize(ssvs=post_ssvs, syn_sizes=m_sizes)

    log.info(
        f'{len(post_ssvs_unique)} {ct1_str} receive input within this cell type. These are {100 * len(post_ssvs_unique) / len(ct1_cellids):.2f}'
        f' percent of {ct1_str} cells')
    log.info(
        f'The median number of synapses are {np.median(post_syn_numbers)}, sum size {np.median(post_syn_sizes):.2f} per cell')
    write_obj2pkl(f'{f_name}/{ct1_str}_post_ids.pkl', post_ssvs_unique)
    # iterate over each ct1 cell and get number of ct1 cells it connects to with the corresponding summed synapse size
    post_ids_dict = {}
    post_sumsizes_dict = {}
    post_syn_numbers_dict = {}
    post_cell_number = np.zeros(len(post_ssvs_unique))
    for ii, ct1_cellid in enumerate(post_ssvs_unique):
        # get only synapses for this cellid
        ind = np.where(post_ssvs == ct1_cellid)[0]
        id_partners = m_ssv_partners[ind]
        id_sizes = m_sizes[ind]
        ind = np.where(id_partners != ct1_cellid)
        pre_partners = id_partners[ind]
        id_pre_numbers, id_pre_sizes, id_pre_partners = get_percell_number_sumsize(pre_partners, id_sizes)
        post_cell_number[ii] = len(id_pre_partners)
        post_ids_dict[ct1_cellid] = id_pre_partners
        post_syn_numbers_dict[ct1_cellid] = id_pre_numbers
        post_sumsizes_dict[ct1_cellid] = id_pre_sizes
    log.info(
        f'A median {ct1_str} receives input from {np.median(post_cell_number)} cells, with {np.median(post_syn_numbers / post_cell_number):.2f}'
        f' synapses and {np.median(post_syn_sizes / post_cell_number):.2f} synaptic area in µm²')

    write_obj2pkl(f'{f_name}/{ct1_str}_post_ids.pkl', post_ids_dict)
    write_obj2pkl(f'{f_name}/{ct1_str}_post_syn_numbers.pkl', post_syn_numbers_dict)
    write_obj2pkl(f'{f_name}/{ct1_str}_post_syn_sumsizes.pkl', post_sumsizes_dict)

    # plot numbers for postsynaptic side
    sns.histplot(data=post_cell_number, color=ct_palette[ct1_str], common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, stat='percent')
    plt.ylabel(f'% of {ct1_str} cells', fontsize=fontsize)
    plt.xlabel(f'number of presynaptic partners', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(f'{ct1_str} postsynaptic connectivity')
    plt.savefig(f'{f_name}/{ct1_str}_post_num_partners_hist_perc.png')
    plt.savefig(f'{f_name}/{ct1_str}_post_num_partners_hist_perc.svg')
    plt.close()
    sns.histplot(data=pre_cell_number, color=ct_palette[ct1_str], common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True)
    plt.ylabel(f'number of {ct1_str} cells', fontsize=fontsize)
    plt.xlabel(f'number of presynaptic partners', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(f'{ct1_str} postsynaptic connectivity')
    plt.savefig(f'{f_name}/{ct1_str}_post_num_partners_hist.png')
    plt.savefig(f'{f_name}/{ct1_str}_post_num_partners_hist.svg')
    plt.close()

    log.info('Step 4/5: Calculate overlap between incoming and outgoing cells from same celltype')
    #compare pre and post dictionaries
    #only do for cells that get and receive input
    ct1_in_out_ids = pre_ssvs_unique[np.in1d(pre_ssvs_unique, post_ssvs_unique)]
    log.info(f'{len(ct1_in_out_ids)} {ct1_str} cells project to and get input from {ct1_str}')
    log.info(f'This is {100*len(ct1_in_out_ids)/len(pre_ssvs_unique):.2f} percent of projecting cells and '
             f'{100*len(ct1_in_out_ids)/len(post_ssvs_unique):.2f} percent of receiving cells.')

    #for each cell get percent overlap in number of cells, syn number, summed size
    #save in dataframe
    columns = ['cellid', 'celltype', 'number partner cells in', 'number partner cells out', 'fraction cell overlap in', 'fraction syn number overlap in', 'fraction syn sum size overlap in',
               'fraction cell overlap out', 'fraction syn number overlap out', 'fraction syn sum size overlap out', 'binary specificity']
    ct1_overlap_df = pd.DataFrame(columns=columns, index = range(len(ct1_in_out_ids)))
    ct1_overlap_df['cellid'] = ct1_in_out_ids
    ct1_overlap_df['celltype'] = ct1_str
    #iterate over each cell to calculate overlap
    #also check if most output goes to most input
    for ii, ct1_id in enumerate(tqdm(ct1_in_out_ids)):
        ids_to = pre_ids_dict[ct1_id]
        syn_nums_to = pre_syn_numbers_dict[ct1_id]
        sum_syns_to = pre_sumsizes_dict[ct1_id]
        ids_from = post_ids_dict[ct1_id]
        syn_nums_from = post_syn_numbers_dict[ct1_id]
        sum_syns_from = post_sumsizes_dict[ct1_id]
        ct1_overlap_df.loc[ii, 'number partner cells in'] = len(ids_from)
        ct1_overlap_df.loc[ii, 'number partner cells out'] = len(ids_to)
        mask_both_out = np.in1d(ids_to, ids_from)
        mask_both_in = np.in1d(ids_from, ids_to)
        ids_both = ids_to[mask_both_out]
        ct1_overlap_df.loc[ii, 'fraction cell overlap in'] = len(ids_both) / len(ids_from)
        ct1_overlap_df.loc[ii, 'fraction cell overlap out'] = len(ids_both) / len(ids_to)
        syn_nums_both_out = syn_nums_to[mask_both_out]
        syn_nums_both_in = syn_nums_from[mask_both_in]
        ct1_overlap_df.loc[ii, 'fraction syn number overlap in'] = np.sum(syn_nums_both_in) / np.sum(syn_nums_from)
        ct1_overlap_df.loc[ii, 'fraction syn number overlap out'] = np.sum(syn_nums_both_out) / np.sum(syn_nums_to)
        sum_sizes_both_out = sum_syns_to[mask_both_out]
        sum_sizes_both_in = sum_syns_from[mask_both_in]
        ct1_overlap_df.loc[ii, 'fraction syn sum size overlap in'] = np.sum(sum_sizes_both_in) / np.sum(sum_syns_from)
        ct1_overlap_df.loc[ii, 'fraction syn sum size overlap out'] = np.sum(sum_sizes_both_out) / np.sum(sum_syns_to)
        #check here if highest input is also highest output
        highest_id_incoming = ids_from[np.argmax(sum_syns_from)]
        highest_id_outgoing = ids_to[np.argmax(sum_syns_to)]
        if highest_id_incoming == highest_id_outgoing:
            ct1_overlap_df.loc[ii, 'binary specificity'] = 1
        else:
            ct1_overlap_df.loc[ii, 'binary specificity'] = 0
    ct1_overlap_df.to_csv(f'{f_name}/{ct1_str}_overlap_df.csv')

    log.info('Step 5/5: Get overview params and plot results')
    param_list = list(ct1_overlap_df.keys())[2:]
    param_list = np.hstack([param_list, 'number cells both'])
    overview_df = pd.DataFrame(columns = [f'median {ct1_str}', f'mean {ct1_str}', f'std {ct1_str}', f'total {ct1_str}'], index = param_list)
    for param in param_list:
        if 'both' in param:
            overview_df.loc[param, f'total {ct1_str}'] = len(ct1_overlap_df)
        elif 'binary' in param:
            bin_spec_cells_ct1 = ct1_overlap_df[ct1_overlap_df['binary specificity'] == 1]
            frac_bin_spec_ct1 = len(bin_spec_cells_ct1) / len(ct1_overlap_df)
            overview_df.loc[param, f'fraction of cells binary specific {ct1_str}'] = frac_bin_spec_ct1
        else:
            overview_df.loc[param, f'median {ct1_str}'] = ct1_overlap_df[param].median()
            overview_df.loc[param, f'mean {ct1_str}'] = ct1_overlap_df[param].mean()
            overview_df.loc[param, f'std {ct1_str}'] = ct1_overlap_df[param].std()
            #plot parameters as histogramm
            sns.histplot(data=ct1_overlap_df, x = param, color=ct_palette[ct1_str], common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True, stat='percent')
            plt.ylabel(f'% of {ct1_str} cells', fontsize=fontsize)
            plt.xlabel(param, fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.title(f'{ct1_str} {param}')
            plt.savefig(f'{f_name}/{ct1_str}_{param}_hist_perc.png')
            plt.savefig(f'{f_name}/{ct1_str}_{param}_hist_perc.svg')
            plt.close()
            sns.histplot(data=ct1_overlap_df, x=param, color=ct_palette[ct1_str], common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True)
            plt.ylabel(f'number of {ct1_str} cells', fontsize=fontsize)
            plt.xlabel(param, fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.title(f'{ct1_str} {param}')
            plt.savefig(f'{f_name}/{ct1_str}_{param}_hist.png')
            plt.savefig(f'{f_name}/{ct1_str}_{param}_hist.svg')
            plt.close()

    overview_df.to_csv(f'{f_name}/recurr_conn_overview_{ct1_str}.csv')

    log.info('Analysis done.')