#get information about msn synapses to GPe and GPi
#more details related to how many GP cells are targeted, how many monosynaptic connections etc.

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper \
        import filter_synapse_caches_for_ct, get_multi_syn_info_per_cell, get_percell_number_sumsize, filter_synapse_caches_general, get_ct_syn_number_sumsize
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    import os as os
    import pandas as pd
    import numpy as np
    from syconn.mp.mp_utils import start_multiprocess_imap
    from syconn.handler.basics import write_obj2pkl
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import ranksums, kruskal, spearmanr
    from itertools import combinations
    from collections import ChainMap
    from tqdm import tqdm

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"

    version = 'v5'
    analysis_params = Analysis_Params(working_dir=global_params.wd, version=version)
    ct_dict = analysis_params.ct_dict(with_glia=False)
    min_comp_len = 200
    syn_prob_thresh = 0.6
    min_syn_size = 0.1
    #celltype that gives input or output
    conn_ct = None
    #celltypes that are compared
    ct2 = 6
    ct3 = 7
    color_key = 'STNGP'
    fontsize_jointplot = 12
    kde = True
    if conn_ct == None:
        f_name = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/231115_j0251v5_%s_%s_syn_multisyn_mcl_%i_synprob_%.2f_kde%i" % (
            ct_dict[ct2], ct_dict[ct3], min_comp_len, syn_prob_thresh, kde)
    else:
        f_name = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/231115_j0251v5_%s_%s_%s_syn_multisyn_mcl_%i_synprob_%.2f_kde%i" % (
            ct_dict[conn_ct], ct_dict[ct2], ct_dict[ct3], min_comp_len, syn_prob_thresh, kde)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('Multisynaptic connectivity details', log_dir=f_name + '/logs/')
    ct_colors = CelltypeColors()
    ct_palette = ct_colors.ct_palette(key = color_key)
    log.info("Analysis of multisynapses starts")
    
    log.info('Step 1/4: Load and check all cells')
    known_mergers = analysis_params.load_known_mergers()
    misclassified_astro_ids = analysis_params.load_potential_astros()
    if conn_ct is not None:
        cts = [conn_ct, ct2, ct3]
    else:
        cts = [ct2, ct3]
    suitable_ids_dict = {}
    all_suitable_ids = []
    for ct in cts:
        cell_info_dict = analysis_params.load_cell_dict(celltype=ct)
        ct_ids = np.array(list(cell_info_dict.keys()))
        merger_inds = np.in1d(ct_ids, known_mergers) == False
        ct_ids = ct_ids[merger_inds]
        astro_inds = np.in1d(ct_ids, misclassified_astro_ids) == False
        ct_ids = ct_ids[astro_inds]
        ct_ids = check_comp_lengths_ct(cellids=ct_ids, fullcelldict=cell_info_dict, min_comp_len=min_comp_len,
                                        axon_only=False,
                                        max_path_len=None)
        #need to be sorted to link to ones from synapses
        ct_ids = np.sort(ct_ids)
        suitable_ids_dict[ct] = ct_ids
        all_suitable_ids.append(ct_ids)
        log.info(f'{len(ct_ids)} suitable cells for celltype {ct_dict[ct]}')

    all_suitable_ids = np.hstack(all_suitable_ids)
    
    log.info('Step 2/4: Get synapses between celltypes')
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.wd)
    m_cts, m_axs, m_ssv_partners, m_sizes, m_rep_coord, syn_prob = filter_synapse_caches_general(sd_synssv, syn_prob_thresh = syn_prob_thresh, min_syn_size = min_syn_size)
    synapse_caches = [m_cts, m_axs, m_ssv_partners, m_sizes, m_rep_coord, syn_prob]
    if conn_ct is None:
        #synapses between celltypes
        m_cts, m_axs, m_ssv_partners, m_sizes, m_rep_coord = filter_synapse_caches_for_ct(
            pre_cts=[ct2, ct3],
            post_cts= None,
            syn_prob_thresh=None,
            min_syn_size=None,
            axo_den_so=True,
            synapses_caches=synapse_caches)
        #synapses within each celltype, this filter only makes sure ct2 is there, not that it is only ct2 to ct2
        ct2_m_cts, ct2_m_axs, ct2_m_ssv_partners, ct2_m_sizes, ct2_m_rep_coord = filter_synapse_caches_for_ct(
            pre_cts=[ct2],
            post_cts=None,
            syn_prob_thresh=None,
            min_syn_size=None,
            axo_den_so=True,
            synapses_caches=synapse_caches)
        ct3_m_cts, ct3_m_axs, ct3_m_ssv_partners, ct3_m_sizes, ct3_m_rep_coord = filter_synapse_caches_for_ct(
            pre_cts=[ct3],
            post_cts=None,
            syn_prob_thresh=None,
            min_syn_size=None,
            axo_den_so=True,
            synapses_caches=synapse_caches)

    else:
        #compare outgoing synapses
        m_cts, m_axs, m_ssv_partners, m_sizes, m_rep_coord = filter_synapse_caches_for_ct(
            pre_cts=[conn_ct],
            post_cts=[ct2, ct3],
            syn_prob_thresh=None,
            min_syn_size=None,
            axo_den_so=True,
            synapses_caches=synapse_caches)
        #compare incoming synapses
        in_m_cts, in_m_axs, in_m_ssv_partners, in_m_sizes, in_m_rep_coord = filter_synapse_caches_for_ct(
            pre_cts=[ct2, ct3],
            post_cts=[conn_ct],
            syn_prob_thresh=None,
            min_syn_size=None,
            axo_den_so=True,
            synapses_caches=synapse_caches)

    suit_ct_inds = np.all(np.in1d(m_ssv_partners, all_suitable_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[suit_ct_inds]
    m_ssv_partners = m_ssv_partners[suit_ct_inds]
    m_sizes = m_sizes[suit_ct_inds]
    m_axs = m_axs[suit_ct_inds]
    m_rep_coord = m_rep_coord[suit_ct_inds]
    if conn_ct is None:
        # make result df for celltype per postsynaptic celltype
        log.info('Step 3/4: Get information to other celltype and plot per cell information')
        log.info('Get per cell information to other celltype')
        #celltype indicates postsynaptic celltype
        percell_columns = ['cellid', f'syn number', f'sum syn size', 'mean syn size', 'celltype']
        percell_result_df = pd.DataFrame(columns=percell_columns, index = range(len(all_suitable_ids)))
        percell_result_df['cellid'] = all_suitable_ids
        percell_result_df.loc[0: len(suitable_ids_dict[ct2]) -1, 'celltype'] = ct_dict[ct2]
        percell_result_df.loc[len(suitable_ids_dict[ct2]): len(all_suitable_ids) - 1, 'celltype'] = ct_dict[ct3]
        #sort into cells going to ct2 and cells going to ct3
        denso_inds = np.in1d(m_axs, [0, 2]).reshape(len(m_axs), 2)
        ct2_inds = np.in1d(m_cts, ct2).reshape(len(m_axs), 2)
        ct3_inds = np.in1d(m_cts, ct3).reshape(len(m_axs), 2)
        ct2_den_inds = np.any(ct2_inds == denso_inds, axis=1)
        ct3_den_inds = np.any(ct3_inds == denso_inds, axis=1)
        # get number synapses and sum size per cells to other celltype
        ct2_ssv_partners = m_ssv_partners[ct2_den_inds]
        ct2_cts = m_cts[ct2_den_inds]
        ct2_ssv_ids = ct2_ssv_partners[np.where(ct2_cts == ct2)]
        ct2_den_syn_sizes = m_sizes[ct2_den_inds]
        #get pre and post ids of all synapses
        post_ct2_ids = ct2_ssv_ids
        pre_inds = np.where(ct2_cts == ct3)
        pre_ct2_ids = ct2_ssv_partners[pre_inds]
        ct2_syn_numbers, ct2_syn_sizes, unique_ct2_cellids = get_percell_number_sumsize(ct2_ssv_ids, ct2_den_syn_sizes)
        log.info(f'{len(unique_ct2_cellids)} out of {len(suitable_ids_dict[ct2])} {ct_dict[ct2]} get synapses from {ct_dict[ct3]}')
        sort_inds = np.argsort(unique_ct2_cellids)
        sorted_unique_ct2_cellids = unique_ct2_cellids[sort_inds]
        sorted_ct2_syn_numbers = ct2_syn_numbers[sort_inds]
        sorted_ct2_syn_sizes = ct2_syn_sizes[sort_inds]
        id_inds = np.in1d(percell_result_df['cellid'], sorted_unique_ct2_cellids)
        percell_result_df.loc[id_inds, 'syn number'] = ct2_syn_numbers
        percell_result_df.loc[id_inds, 'sum syn size'] = ct2_syn_sizes
        percell_result_df.loc[id_inds, 'mean syn size'] = ct2_syn_sizes / ct2_syn_numbers
        ct3_ssv_partners = m_ssv_partners[ct3_den_inds]
        ct3_cts = m_cts[ct3_den_inds]
        ct3_ssv_ids = ct3_ssv_partners[np.where(ct3_cts == ct3)]
        ct3_den_syn_sizes = m_sizes[ct3_den_inds]
        # get pre and post ids of all synapses
        post_ct3_ids = ct3_ssv_ids
        pre_inds = np.where(ct3_cts == ct2)
        pre_ct3_ids = ct3_ssv_partners[pre_inds]
        ct3_syn_numbers, ct3_syn_sizes, unique_ct3_cellids = get_percell_number_sumsize(ct3_ssv_ids, ct3_den_syn_sizes)
        log.info(
            f'{len(unique_ct3_cellids)} out of {len(suitable_ids_dict[ct3])} {ct_dict[ct3]} get synapses from {ct_dict[ct2]}')
        sort_inds = np.argsort(unique_ct3_cellids)
        sorted_unique_ct3_cellids = unique_ct3_cellids[sort_inds]
        sorted_ct3_syn_numbers = ct3_syn_numbers[sort_inds]
        sorted_ct3_syn_sizes = ct3_syn_sizes[sort_inds]
        id_inds = np.in1d(percell_result_df['cellid'], sorted_unique_ct3_cellids)
        percell_result_df.loc[id_inds, 'syn number'] = ct3_syn_numbers
        percell_result_df.loc[id_inds, 'sum syn size'] = ct3_syn_sizes
        percell_result_df.loc[id_inds, 'mean syn size'] = ct3_syn_sizes / ct3_syn_numbers
        percell_result_df = percell_result_df.dropna()
        percell_result_df = percell_result_df.reset_index(drop = True)
        percell_result_df.to_csv(f'{f_name}/{ct_dict[ct2]}_{ct_dict[ct3]}_percell_results.csv')
        log.info('Make per cell statistics of synapses between celltypes and plots')
        ranksum_columns =  [f'to {ct_dict[ct2]} vs to {ct_dict[ct3]} inter', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]} intra',
                            f'{ct_dict[ct2]} inter vs intra', f'{ct_dict[ct3]} inter vs intra']
        ranksum_results = pd.DataFrame(columns = ranksum_columns)
        for key in percell_result_df.keys():
            if 'cellid' in key or 'celltype' in key:
                continue
            key_groups = [group[key].values for name, group in
                percell_result_df.groupby('celltype')]
            stats, p_value = ranksums(key_groups[0], key_groups[1])
            ranksum_results.loc[f'{key} per post-cell stats', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]} inter'] = stats
            ranksum_results.loc[f'{key} per post-cell p-value', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]} inter'] = p_value
            if 'size' in key:
                xlabel = f'{key} [µm²]'
            else:
                xlabel = key
            sns.histplot(x=key, data=percell_result_df, hue='celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True)
            plt.ylabel('number of cells')
            plt.xlabel(xlabel)
            plt.title(f'{key} between post-syn cells of {ct_dict[ct2]} and {ct_dict[ct3]}')
            plt.savefig(f'{f_name}/percell_{key}_inter_hist.png')
            plt.savefig(f'{f_name}/percell_{key}_inter_hist.svg')
            plt.close()
            sns.histplot(x=key, data=percell_result_df, hue='celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True, stat='percent')
            plt.ylabel('% of cells')
            plt.xlabel(xlabel)
            plt.title(f'{key} between post-syn cells of {ct_dict[ct2]} and {ct_dict[ct3]}')
            plt.savefig(f'{f_name}/percell_{key}_inter_hist_perc.png')
            plt.savefig(f'{f_name}/percell_{key}_inter_hist_perc.svg')
            plt.close()
            sns.histplot(x=key, data=percell_result_df, hue='celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True, log_scale=True)
            plt.ylabel('number of cells')
            plt.xlabel(xlabel)
            plt.title(f'{key} between post-syn cells of {ct_dict[ct2]} and {ct_dict[ct3]}')
            plt.savefig(f'{f_name}/percell_{key}_inter_hist_log.png')
            plt.savefig(f'{f_name}/percell_{key}_inter_hist_log.svg')
            plt.close()
            sns.histplot(x=key, data=percell_result_df, hue='celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True, stat='percent', log_scale=True)
            plt.ylabel('% of cells')
            plt.xlabel(xlabel)
            plt.title(f'{key} between post-syn cells of {ct_dict[ct2]} and {ct_dict[ct3]}')
            plt.savefig(f'{f_name}/percell_{key}_inter_hist_perc_log.png')
            plt.savefig(f'{f_name}/percell_{key}_inter_hist_perc_log.svg')
            plt.close()
            if len(percell_result_df) <= 500:
                sns.stripplot(x='celltype', y=key, data=percell_result_df, color = 'black',
                              alpha=0.4,
                              dodge=True, size=3)
                sns.boxplot(x = 'celltype', y = key, data = percell_result_df, palette=ct_palette)
                plt.ylabel(xlabel)
                plt.title(f'{key} between post-syn cells of {ct_dict[ct2]} and {ct_dict[ct3]}')
                plt.savefig(f'{f_name}/percell_{key}_inter_box.png')
                plt.savefig(f'{f_name}/percell_{key}_inter_box.svg')
                plt.close()
        log.info('Get syn sizes of all cells between celltypes and plot them')
        len_ct2 = len(ct2_den_syn_sizes)
        number_syns = len_ct2 + len(ct3_den_syn_sizes)
        syn_sizes_df = pd.DataFrame(columns=['syn sizes', 'celltype', 'cellid pre', 'cellid post'],
                                    index=range(number_syns))
        syn_sizes_df.loc[0:len_ct2 - 1, 'syn sizes'] = ct2_den_syn_sizes
        syn_sizes_df.loc[0:len_ct2 - 1, 'celltype'] = ct_dict[ct2]
        syn_sizes_df.loc[0:len_ct2 - 1, 'cellid pre'] = pre_ct2_ids
        syn_sizes_df.loc[0:len_ct2 - 1, 'cellid post'] = post_ct2_ids
        syn_sizes_df.loc[len_ct2: number_syns - 1, 'syn sizes'] = ct3_den_syn_sizes
        syn_sizes_df.loc[len_ct2: number_syns - 1, 'celltype'] = ct_dict[ct3]
        syn_sizes_df.loc[len_ct2: number_syns - 1, 'cellid pre'] = pre_ct3_ids
        syn_sizes_df.loc[len_ct2: number_syns - 1, 'cellid post'] = post_ct3_ids
        syn_sizes_df.to_csv(f'{f_name}/all_syn_sizes_between_{ct_dict[ct2]}_{ct_dict[ct3]}.csv')
        stats, p_value = ranksums(ct2_den_syn_sizes, ct3_den_syn_sizes)
        ranksum_results.loc['all syn sizes stats', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]} inter'] = stats
        ranksum_results.loc['all syn sizes p-value', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]} inter'] = p_value
        ranksum_results.to_csv(f'{f_name}/ranksums_results_{ct_dict[ct2]}_{ct_dict[ct3]}.csv')
        #plot all syn sizes values
        sns.histplot(x='syn sizes', data=syn_sizes_df, hue='celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True)
        plt.ylabel('number of synapses')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Synapse sizes within {ct_dict[ct2]} and {ct_dict[ct3]}')
        plt.savefig(f'{f_name}/all_synsizes_inter_hist.png')
        plt.savefig(f'{f_name}/all_synsizes_inter_hist.svg')
        plt.close()
        sns.histplot(x='syn sizes', data=syn_sizes_df, hue='celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, log_scale=True)
        plt.ylabel('number of synapses')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Synapse sizes within {ct_dict[ct2]} and {ct_dict[ct3]}')
        plt.savefig(f'{f_name}/all_synsizes_inter_hist_log.png')
        plt.savefig(f'{f_name}/all_synsizes_inter_hist_log.svg')
        plt.title(f'Synapse sizes within {ct_dict[ct2]} and {ct_dict[ct3]}')
        plt.close()
        sns.histplot(x='syn sizes', data=syn_sizes_df, hue='celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent')
        plt.ylabel('% of synapses')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Synapse sizes within {ct_dict[ct2]} and {ct_dict[ct3]}')
        plt.savefig(f'{f_name}/all_synsizes_inter_hist_perc.png')
        plt.savefig(f'{f_name}/all_synsizes_inter_hist_perc.svg')
        plt.close()
        sns.histplot(x='syn sizes', data=syn_sizes_df, hue='celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, log_scale=True, stat='percent')
        plt.ylabel('% of synapses')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Synapse sizes within {ct_dict[ct2]} and {ct_dict[ct3]}')
        plt.savefig(f'{f_name}/all_synsizes_inter_hist_log_perc.png')
        plt.savefig(f'{f_name}/all_synsizes_inter_hist_log_perc.svg')
        plt.close()
        log.info('Get information about multi-syn connections between celltypes')
        # get number of partner cells and size information for all of them
        # use syn_sizes df for information
        all_unique_pre_ids = np.unique(syn_sizes_df['cellid pre'])
        num_pre_ids = len(all_unique_pre_ids)
        all_unique_post_ids = np.unique(syn_sizes_df['cellid post'])
        num_post_ids = len(all_unique_post_ids)
        multi_conn_df = pd.DataFrame(
            columns=['number of synapses', 'sum syn area', 'celltype', 'cellid pre', 'cellid post'],
            index=range(num_pre_ids * num_post_ids))

        for i, cell_id in enumerate(tqdm(all_unique_post_ids)):
            cell_syn_sizes_df = syn_sizes_df[syn_sizes_df['cellid post'] == cell_id]
            # for ct2
            ct2_cell_syn_sizes_df = cell_syn_sizes_df[cell_syn_sizes_df['celltype'] == ct_dict[ct2]]
            ct2_syn_numbers, ct2_sum_sizes, unique_ct2_ids = get_percell_number_sumsize(
                ct2_cell_syn_sizes_df['cellid post'], ct2_cell_syn_sizes_df['syn sizes'])
            len_multi_syns_ct2 = len(ct2_syn_numbers)
            start_ct2 = i * num_post_ids
            end_ct2 = start_ct2 + len_multi_syns_ct2 - 1
            multi_conn_df.loc[start_ct2: end_ct2, 'number of synapses'] = ct2_syn_numbers
            multi_conn_df.loc[start_ct2:end_ct2, 'sum syn area'] = ct2_sum_sizes
            multi_conn_df.loc[start_ct2: end_ct2, 'celltype'] = ct_dict[ct2]
            multi_conn_df.loc[start_ct2: end_ct2, 'cellid pre'] = cell_id
            multi_conn_df.loc[start_ct2: end_ct2, 'cellid post'] = unique_ct2_ids
            ct3_cell_syn_sizes_df = cell_syn_sizes_df[cell_syn_sizes_df['celltype'] == ct_dict[ct3]]
            ct3_syn_numbers, ct3_sum_sizes, unique_ct3_ids = get_percell_number_sumsize(
                ct3_cell_syn_sizes_df['cellid post'], ct3_cell_syn_sizes_df['syn sizes'])
            len_multi_syns_ct3 = len(ct3_syn_numbers)
            start_ct3 = i * num_post_ids + len_multi_syns_ct2
            end_ct3 = start_ct3 + len_multi_syns_ct3 - 1
            multi_conn_df.loc[start_ct3: end_ct3, 'number of synapses'] = ct3_syn_numbers
            multi_conn_df.loc[start_ct3:end_ct3, 'sum syn area'] = ct3_sum_sizes
            multi_conn_df.loc[start_ct3: end_ct3, 'celltype'] = ct_dict[ct3]
            multi_conn_df.loc[start_ct3: end_ct3, 'cellid pre'] = cell_id
            multi_conn_df.loc[start_ct3: end_ct3, 'cellid post'] = unique_ct3_ids

        multi_conn_df = multi_conn_df.dropna()
        multi_conn_df = multi_conn_df.reset_index(drop=True)
        multi_conn_df.to_csv(f'{f_name}/multi_syns_between_{ct_dict[ct2]}_{ct_dict[ct3]}.csv')
        # make statistics and run results
        multi_ct2_numbers = multi_conn_df['number of synapses'][multi_conn_df['celltype'] == ct_dict[ct2]]
        multi_ct3_numbers = multi_conn_df['number of synapses'][multi_conn_df['celltype'] == ct_dict[ct3]]
        stats, p_value = ranksums(multi_ct2_numbers, multi_ct3_numbers)
        ranksum_results.loc['multi syn numbers stats', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]} inter'] = stats
        ranksum_results.loc['multi syn numbers p-value', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]} inter'] = p_value
        multi_ct2_sizes = multi_conn_df['sum syn area'][multi_conn_df['celltype'] == ct_dict[ct2]]
        multi_ct3_sizes = multi_conn_df['sum syn area'][multi_conn_df['celltype'] == ct_dict[ct3]]
        stats, p_value = ranksums(multi_ct2_sizes, multi_ct3_sizes)
        ranksum_results.loc['multi syn sum sizes stats', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]} inter'] = stats
        ranksum_results.loc['multi syn sum sizess p-value', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]} inter'] = p_value
        ranksum_results.to_csv(f'{f_name}/ranksums_results_{ct_dict[ct2]}_{ct_dict[ct3]}.csv')
        # plot results
        # plot sum sizes for pairwise cells as histogram
        sns.histplot(x='sum syn area', data=multi_conn_df, hue='celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True)
        plt.ylabel('number of cell-pairs')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Sum of synapse sizes of multi-syn connections between {ct_dict[ct2]}, {ct_dict[ct3]}')
        plt.savefig(f'{f_name}/multi_sum_sizes_inter_hist.png')
        plt.savefig(f'{f_name}/multi_sum_sizes_inter_hist.svg')
        plt.close()
        sns.histplot(x='sum syn area', data=multi_conn_df, hue='celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent')
        plt.ylabel('% of cell-pairs')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Sum of synapse sizes of multi-syn connections between {ct_dict[ct2]}, {ct_dict[ct3]}')
        plt.savefig(f'{f_name}/multi_sum_sizes_inter_hist_perc.png')
        plt.savefig(f'{f_name}/multi_sum_sizes_inter_hist_perc.svg')
        plt.close()
        sns.histplot(x='sum syn area', data=multi_conn_df, hue='celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, log_scale=True)
        plt.ylabel('number of cell-pairs')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Sum of synapse sizes of multi-syn connections between {ct_dict[ct2]}, {ct_dict[ct3]}')
        plt.savefig(f'{f_name}/multi_sum_sizes_inter_hist_log.png')
        plt.savefig(f'{f_name}/multi_sum_sizes_inter_hist_log.svg')
        plt.close()
        sns.histplot(x='sum syn area', data=multi_conn_df, hue='celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent', log_scale=True)
        plt.ylabel('% of cell-pairs')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Sum of synapse sizes of multi-syn connections between {ct_dict[ct2]}, {ct_dict[ct3]}')
        plt.savefig(f'{f_name}/multi_sum_sizes_inter_hist_log_perc.png')
        plt.savefig(f'{f_name}/multi_sum_sizes_inter_hist_log_perc.svg')
        plt.close()
        # plot again as barplot
        # plot number of synapses again as barplot
        # make bins for each number, seperated by ct2 and ct3
        ct2_multi_df = multi_conn_df[multi_conn_df['celltype'] == ct_dict[ct2]]
        ct2_inds, ct2_bins = pd.factorize(np.sort(ct2_multi_df['number of synapses']))
        ct2_counts = np.bincount(ct2_inds)
        ct3_multi_df = multi_conn_df[multi_conn_df['celltype'] == ct_dict[ct3]]
        ct3_inds, ct3_bins = pd.factorize(np.sort(ct3_multi_df['number of synapses']))
        ct3_counts = np.bincount(ct3_inds)
        num_ct2_bins = len(ct2_bins)
        len_bins = num_ct2_bins + len(ct3_bins)
        hist_df = pd.DataFrame(columns=['count of cell-pairs', 'percent of cell-pairs', 'bins', 'to celltype'],
                               index=range(len_bins))
        hist_df.loc[0: num_ct2_bins - 1, 'count of cell-pairs'] = ct2_counts
        hist_df.loc[0: num_ct2_bins - 1, 'percent of cell-pairs'] = 100 * ct2_counts / np.sum(ct2_counts)
        hist_df.loc[0: num_ct2_bins - 1, 'bins'] = ct2_bins
        hist_df.loc[0: num_ct2_bins - 1, 'celltype'] = ct_dict[ct2]
        hist_df.loc[num_ct2_bins: len_bins - 1, 'count of cell-pairs'] = ct3_counts
        hist_df.loc[num_ct2_bins: len_bins - 1, 'percent of cell-pairs'] = 100 * ct3_counts / np.sum(ct3_counts)
        hist_df.loc[num_ct2_bins: len_bins - 1, 'bins'] = ct3_bins
        hist_df.loc[num_ct2_bins: len_bins - 1, 'celltype'] = ct_dict[ct3]
        sns.barplot(data=hist_df, x='bins', y='count of cell-pairs', hue='celltype', palette=ct_palette)
        plt.xlabel('number of synapses')
        plt.savefig(f'{f_name}/multi_bar_syn_num_inter.svg')
        plt.savefig(f'{f_name}/multi_bar_syn_num_inter.png')
        plt.close()
        sns.barplot(data=hist_df, x='bins', y='percent of cell-pairs', hue='celltype', palette=ct_palette)
        plt.xlabel('number of synapses')
        plt.savefig(f'{f_name}/multi_bar_syn_num_inter_perc.svg')
        plt.savefig(f'{f_name}/multi_bar_syn_num_inter_perc.png')
        plt.close()
        log.info('Step 4/4: Get information to same celltype and plot per cell information')
        #get number synapses and sum size per cells to same celltype
        #make sure only suitable cells are used in all synapses
        percell_intra_result_df = pd.DataFrame(columns=percell_columns, index=range(len(all_suitable_ids)))
        percell_intra_result_df['cellid'] = all_suitable_ids
        percell_intra_result_df.loc[0: len(suitable_ids_dict[ct2]) - 1, 'celltype'] = ct_dict[ct2]
        percell_intra_result_df.loc[len(suitable_ids_dict[ct2]): len(all_suitable_ids) - 1, 'celltype'] = ct_dict[ct3]
        #prepare dataframe for all syns
        max_syn_number = len(ct2_m_sizes) + len(ct3_m_sizes)
        syn_sizes_intra_df = pd.DataFrame(columns=['syn sizes', 'celltype', 'cellid pre', 'cellid post'],
                                    index=range(max_syn_number))
        suit_ct_inds = np.all(np.in1d(ct2_m_ssv_partners, suitable_ids_dict[ct2]).reshape(len(ct2_m_ssv_partners), 2), axis=1)
        ct2_m_cts = ct2_m_cts[suit_ct_inds]
        ct2_m_axs = ct2_m_axs[suit_ct_inds]
        ct2_m_sizes = ct2_m_sizes[suit_ct_inds]
        ct2_m_ssv_partners = ct2_m_ssv_partners[suit_ct_inds]
        max_number_syns = len(ct2_m_sizes) + len(ct3_m_sizes)
        if len(ct2_m_cts) > 0:
            assert(len(np.unique(ct2_m_cts)) == 1 and np.unique(ct2_m_cts)[0] == ct2)
            #get percell information
            log.info(f'Get percell information for synapses within {ct_dict[ct2]}')
            denso_inds = np.in1d(ct2_m_axs, [0, 2]).reshape(len(ct2_m_axs), 2)
            # get number synapses and sum size per cells to same celltype
            ct2_ssv_ids = ct2_m_ssv_partners[denso_inds]
            #get pre and post cellids for multi_syn_analysis
            pre_inds = np.where(ct2_m_axs == 1)
            pre_ct2_ids = ct2_m_ssv_partners[pre_inds]
            post_inds = np.where(ct2_m_axs != 1)
            post_ct2_ids = ct2_m_ssv_partners[post_inds]
            ct2_syn_numbers, ct2_syn_sizes, unique_ct2_cellids = get_percell_number_sumsize(ct2_ssv_ids,
                                                                                            ct2_m_sizes)
            log.info(
                f'{len(unique_ct2_cellids)} out of {len(suitable_ids_dict[ct2])} {ct_dict[ct2]} get synapses from {ct_dict[ct2]}')
            sort_inds = np.argsort(unique_ct2_cellids)
            sorted_unique_ct2_cellids = unique_ct2_cellids[sort_inds]
            sorted_ct2_syn_numbers = ct2_syn_numbers[sort_inds]
            sorted_ct2_syn_sizes = ct2_syn_sizes[sort_inds]
            id_inds = np.in1d(percell_intra_result_df['cellid'], sorted_unique_ct2_cellids)
            percell_intra_result_df.loc[id_inds, 'syn number'] = ct2_syn_numbers
            percell_intra_result_df.loc[id_inds, 'sum syn size'] = ct2_syn_sizes
            percell_intra_result_df.loc[id_inds, 'mean syn size'] = ct2_syn_sizes / ct2_syn_numbers
            #fill in all synapse size
            len_ct2 = len(ct2_m_sizes)
            syn_sizes_intra_df.loc[0:len_ct2 - 1, 'syn sizes'] = ct2_m_sizes
            syn_sizes_intra_df.loc[0:len_ct2 - 1, 'celltype'] = ct_dict[ct2]
            syn_sizes_intra_df.loc[0:len_ct2 - 1, 'cellid pre'] = pre_ct2_ids
            syn_sizes_intra_df.loc[0:len_ct2 - 1, 'cellid post'] = post_ct2_ids
        else:
            log.info(f'There are no synapses between cells of {ct_dict[ct2]} which follow the filter criteria')
        suit_ct_inds = np.all(np.in1d(ct3_m_ssv_partners, suitable_ids_dict[ct3]).reshape(len(ct3_m_ssv_partners), 2),
                              axis=1)
        ct3_m_cts = ct3_m_cts[suit_ct_inds]
        ct3_m_axs = ct3_m_axs[suit_ct_inds]
        ct3_m_sizes = ct3_m_sizes[suit_ct_inds]
        ct3_m_ssv_partners = ct3_m_ssv_partners[suit_ct_inds]
        if len(ct3_m_cts) > 0:
            assert (len(np.unique(ct3_m_cts)) == 1 and np.unique(ct3_m_cts)[0] == ct3)
            denso_inds = np.in1d(ct3_m_axs, [0, 2]).reshape(len(ct3_m_axs), 2)
            # get number synapses and sum size per cells to same celltype
            ct3_ssv_ids = ct3_m_ssv_partners[denso_inds]
            # get pre and post cellids for multi_syn_analysis
            pre_inds = np.where(ct3_m_axs == 1)
            pre_ct3_ids = ct3_m_ssv_partners[pre_inds]
            post_inds = np.where(ct3_m_axs != 1)
            post_ct3_ids = ct3_m_ssv_partners[post_inds]
            ct3_syn_numbers, ct3_syn_sizes, unique_ct3_cellids = get_percell_number_sumsize(ct3_ssv_ids,
                                                                                            ct3_m_sizes)
            log.info(
                f'{len(unique_ct3_cellids)} out of {len(suitable_ids_dict[ct3])} {ct_dict[ct3]} get synapses from {ct_dict[ct3]}')
            sort_inds = np.argsort(unique_ct3_cellids)
            sorted_unique_ct3_cellids = unique_ct3_cellids[sort_inds]
            sorted_ct3_syn_numbers = ct3_syn_numbers[sort_inds]
            sorted_ct3_syn_sizes = ct3_syn_sizes[sort_inds]
            id_inds = np.in1d(percell_intra_result_df['cellid'], sorted_unique_ct3_cellids)
            percell_intra_result_df.loc[id_inds, 'syn number'] = ct3_syn_numbers
            percell_intra_result_df.loc[id_inds, 'sum syn size'] = ct3_syn_sizes
            percell_intra_result_df.loc[id_inds, 'mean syn size'] = ct3_syn_sizes / ct3_syn_numbers
            # fill in all synapse size

            len_ct3 = len(ct3_m_sizes)
            syn_sizes_intra_df.loc[len_ct2:len_ct2 + len_ct3 - 1, 'syn sizes'] = ct3_m_sizes
            syn_sizes_intra_df.loc[len_ct2:len_ct2 + len_ct3 - 1, 'celltype'] = ct_dict[ct3]
            syn_sizes_intra_df.loc[len_ct2:len_ct2 + len_ct3 - 1, 'cellid pre'] = pre_ct3_ids
            syn_sizes_intra_df.loc[len_ct2:len_ct2 + len_ct3 - 1, 'cellid post'] = post_ct3_ids
        else:
            log.info(f'There are no synapses between cells of {ct_dict[ct3]} which follow the filter criteria')
        percell_intra_result_df = percell_intra_result_df.dropna()
        percell_intra_result_df = percell_intra_result_df.dropna()
        percell_intra_result_df = percell_intra_result_df.reset_index(drop=True)
        percell_intra_result_df.to_csv(f'{f_name}/{ct_dict[ct2]}_{ct_dict[ct3]}_percell_intra_results.csv')
        log.info('Get statistics and plot results of intra celltype synapses')
        syn_sizes_intra_df = syn_sizes_intra_df.dropna()
        syn_sizes_intra_df = syn_sizes_intra_df.reset_index(drop = True)
        syn_sizes_intra_df.to_csv(f'{f_name}/all_syn_sizes_intra_{ct_dict[ct2]}_{ct_dict[ct3]}.csv')
        if len_ct2 > 0 or len_ct3 > 0:
            for key in percell_intra_result_df.keys():
                if 'cellid' in key or 'celltype' in key:
                    continue
                if len_ct2 > 0:
                    ct2_percell_results = percell_result_df[key][percell_result_df['celltype'] == ct_dict[ct2]]
                    ct2_intra_results = percell_intra_result_df[key][percell_intra_result_df['celltype'] == ct_dict[ct2]]
                    stats, p_value = ranksums(ct2_percell_results, ct2_intra_results)
                    ranksum_results.loc[
                        f'{key} per post-cell stats', f'{ct_dict[ct2]} inter vs intra'] = stats
                    ranksum_results.loc[
                        f'{key} per post-cell p-value', f'{ct_dict[ct2]} inter vs intra'] = p_value
                if len_ct3 > 0:
                    ct3_percell_results = percell_result_df[key][percell_result_df['celltype'] == ct_dict[ct3]]
                    ct3_intra_results = percell_intra_result_df[key][
                        percell_intra_result_df['celltype'] == ct_dict[ct3]]
                    stats, p_value = ranksums(ct3_percell_results, ct3_intra_results)
                    ranksum_results.loc[
                        f'{key} per post-cell stats', f'{ct_dict[ct3]} inter vs intra'] = stats
                    ranksum_results.loc[
                        f'{key} per post-cell p-value', f'{ct_dict[ct3]} inter vs intra'] = p_value
                if len_ct2 > 0 and len_ct3 > 0:
                    key_groups = [group[key].values for name, group in
                                  percell_intra_result_df.groupby('celltype')]
                    stats, p_value = ranksums(key_groups[0], key_groups[1])
                    ranksum_results.loc[
                        f'{key} per post-cell stats', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]} intra'] = stats
                    ranksum_results.loc[
                        f'{key} per post-cell p-value', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]} intra'] = p_value
                if 'size' in key:
                    xlabel = f'{key} [µm²]'
                else:
                    xlabel = key
                sns.histplot(x=key, data=percell_intra_result_df, hue='celltype', palette=ct_palette, common_norm=False,
                             fill=False, element="step", linewidth=3, legend=True)
                plt.ylabel('number of cells')
                plt.xlabel(xlabel)
                plt.title(f'{key} within cells of {ct_dict[ct2]} and {ct_dict[ct3]}')
                plt.savefig(f'{f_name}/percell_{key}_intra_hist.png')
                plt.savefig(f'{f_name}/percell_{key}_intra_hist.svg')
                plt.close()
                sns.histplot(x=key, data=percell_intra_result_df, hue='celltype', palette=ct_palette, common_norm=False,
                             fill=False, element="step", linewidth=3, legend=True, stat='percent')
                plt.ylabel('% of cells')
                plt.xlabel(xlabel)
                plt.title(f'{key} within cells of {ct_dict[ct2]} and {ct_dict[ct3]}')
                plt.savefig(f'{f_name}/percell_{key}_intra_hist_perc.png')
                plt.savefig(f'{f_name}/percell_{key}_intra_hist_perc.svg')
                plt.close()
                sns.histplot(x=key, data=percell_intra_result_df, hue='celltype', palette=ct_palette, common_norm=False,
                             fill=False, element="step", linewidth=3, legend=True, log_scale=True)
                plt.ylabel('number of cells')
                plt.xlabel(xlabel)
                plt.title(f'{key} within cells of {ct_dict[ct2]} and {ct_dict[ct3]}')
                plt.savefig(f'{f_name}/percell_{key}_intra_hist_log.png')
                plt.savefig(f'{f_name}/percell_{key}_intra_hist_log.svg')
                plt.close()
                sns.histplot(x=key, data=percell_intra_result_df, hue='celltype', palette=ct_palette, common_norm=False,
                             fill=False, element="step", linewidth=3, legend=True, stat='percent', log_scale=True)
                plt.ylabel('% of cells')
                plt.xlabel(xlabel)
                plt.title(f'{key} within cells of {ct_dict[ct2]} and {ct_dict[ct3]}')
                plt.savefig(f'{f_name}/percell_{key}_intra_hist_perc_log.png')
                plt.savefig(f'{f_name}/percell_{key}_intra_hist_perc_log.svg')
                plt.close()
                if len(percell_intra_result_df) <= 500:
                    sns.stripplot(x='celltype', y=key, data=percell_intra_result_df, color = 'black',
                                  alpha=0.4,
                                  dodge=True, size=3)
                    sns.boxplot(x='celltype', y = key, data=percell_intra_result_df, palette=ct_palette)
                    plt.ylabel(xlabel)
                    plt.title(f'{key} within cells of {ct_dict[ct2]} and {ct_dict[ct3]}')
                    plt.savefig(f'{f_name}/percell_{key}_intra_box.png')
                    plt.savefig(f'{f_name}/percell_{key}_intra_box.svg')
                    plt.close()
            if len(ct2_m_sizes) > 0:
                stats, p_value = ranksums(ct2_den_syn_sizes, ct2_m_sizes)
                ranksum_results.loc['all syn sizes stats', f'{ct_dict[ct2]} inter vs intra'] = stats
                ranksum_results.loc['all syn sizes p-value', f'{ct_dict[ct2]} inter vs intra'] = p_value
            if len(ct3_m_sizes) > 0:
                stats, p_value = ranksums(ct3_den_syn_sizes, ct3_m_sizes)
                ranksum_results.loc['all syn sizes stats', f'{ct_dict[ct3]} inter vs intra'] = stats
                ranksum_results.loc['all syn sizes p-value', f'{ct_dict[ct3]} inter vs intra'] = p_value
            if len(ct2_m_sizes) > 0 and len(ct3_m_sizes) > 0:
                stats, p_value = ranksums(ct2_den_syn_sizes, ct3_den_syn_sizes)
                ranksum_results.loc['all syn sizes stats', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]} intra'] = stats
                ranksum_results.loc['all syn sizes p-value', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]} intra'] = p_value
            ranksum_results.to_csv(f'{f_name}/ranksums_results_{ct_dict[ct2]}_{ct_dict[ct3]}.csv')
            # plot all syn sizes values
            sns.histplot(x='syn sizes', data=syn_sizes_intra_df, hue='celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True)
            plt.ylabel('number of synapses')
            plt.xlabel('synaptic mesh area [µm²]')
            plt.title(f'Synapse sizes within {ct_dict[ct2]} and {ct_dict[ct3]}')
            plt.savefig(f'{f_name}/all_synsizes_intra_hist.png')
            plt.savefig(f'{f_name}/all_synsizes_intra_hist.svg')
            plt.close()
            sns.histplot(x='syn sizes', data=syn_sizes_intra_df, hue='celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True, log_scale=True)
            plt.ylabel('number of synapses')
            plt.xlabel('synaptic mesh area [µm²]')
            plt.title(f'Synapse sizes within {ct_dict[ct2]} and {ct_dict[ct3]}')
            plt.savefig(f'{f_name}/all_synsizes_intra_hist_log.png')
            plt.savefig(f'{f_name}/all_synsizes_intra_hist_log.svg')
            plt.close()
            sns.histplot(x='syn sizes', data=syn_sizes_intra_df, hue='celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True, stat='percent')
            plt.ylabel('% of synapses')
            plt.xlabel('synaptic mesh area [µm²]')
            plt.title(f'Synapse sizes within {ct_dict[ct2]} and {ct_dict[ct3]}')
            plt.savefig(f'{f_name}/all_synsizes_intra_hist_perc.png')
            plt.savefig(f'{f_name}/all_synsizes_intra_hist_perc.svg')
            plt.close()
            sns.histplot(x='syn sizes', data=syn_sizes_intra_df, hue='celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True, log_scale=True, stat='percent')
            plt.ylabel('% of synapses')
            plt.xlabel('synaptic mesh area [µm²]')
            plt.title(f'Synapse sizes within {ct_dict[ct2]} and {ct_dict[ct3]}')
            plt.savefig(f'{f_name}/all_synsizes_intra_hist_log_perc.png')
            plt.savefig(f'{f_name}/all_synsizes_intra_hist_log_perc.svg')
            plt.close()
            log.info('Get number of multisynapses within celltypes')
            # use syn_sizes df for information
            all_unique_pre_ids = np.unique(syn_sizes_intra_df['cellid pre'])
            num_pre_ids = len(all_unique_pre_ids)
            all_unique_post_ids = np.unique(syn_sizes_intra_df['cellid post'])
            num_post_ids = len(all_unique_post_ids)
            multi_conn_df_intra = pd.DataFrame(
                columns=['number of synapses', 'sum syn area', 'celltype', 'cellid pre', 'cellid post'],
                index=range(num_pre_ids * num_post_ids))
            for i, cell_id in enumerate(tqdm(all_unique_post_ids)):
                cell_syn_sizes_df = syn_sizes_intra_df[syn_sizes_intra_df['cellid post'] == cell_id]
                # for ct2
                if len_ct2 > 0:
                    ct2_cell_syn_sizes_df = cell_syn_sizes_df[cell_syn_sizes_df['celltype'] == ct_dict[ct2]]
                    ct2_syn_numbers, ct2_sum_sizes, unique_ct2_ids = get_percell_number_sumsize(
                        ct2_cell_syn_sizes_df['cellid post'], ct2_cell_syn_sizes_df['syn sizes'])
                    len_multi_syns_ct2 = len(ct2_syn_numbers)
                    start_ct2 = i * num_post_ids
                    end_ct2 = start_ct2 + len_multi_syns_ct2 - 1
                    multi_conn_df_intra.loc[start_ct2: end_ct2, 'number of synapses'] = ct2_syn_numbers
                    multi_conn_df_intra.loc[start_ct2:end_ct2, 'sum syn area'] = ct2_sum_sizes
                    multi_conn_df_intra.loc[start_ct2: end_ct2, 'celltype'] = ct_dict[ct2]
                    multi_conn_df_intra.loc[start_ct2: end_ct2, 'cellid pre'] = cell_id
                    multi_conn_df_intra.loc[start_ct2: end_ct2, 'cellid post'] = unique_ct2_ids
                if len_ct3 > 0:
                    ct3_cell_syn_sizes_df = cell_syn_sizes_df[cell_syn_sizes_df['celltype'] == ct_dict[ct3]]
                    ct3_syn_numbers, ct3_sum_sizes, unique_ct3_ids = get_percell_number_sumsize(
                        ct3_cell_syn_sizes_df['cellid post'], ct3_cell_syn_sizes_df['syn sizes'])
                    len_multi_syns_ct3 = len(ct3_syn_numbers)
                    start_ct3 = i * num_post_ids + len_multi_syns_ct2
                    end_ct3 = start_ct3 + len_multi_syns_ct3 - 1
                    multi_conn_df_intra.loc[start_ct3: end_ct3, 'number of synapses'] = ct3_syn_numbers
                    multi_conn_df_intra.loc[start_ct3:end_ct3, 'sum syn area'] = ct3_sum_sizes
                    multi_conn_df_intra.loc[start_ct3: end_ct3, 'celltype'] = ct_dict[ct3]
                    multi_conn_df_intra.loc[start_ct3: end_ct3, 'cellid pre'] = cell_id
                    multi_conn_df_intra.loc[start_ct3: end_ct3, 'cellid post'] = unique_ct3_ids

            multi_conn_df_intra = multi_conn_df_intra.dropna()
            multi_conn_df_intra = multi_conn_df_intra.reset_index(drop=True)
            multi_conn_df_intra.to_csv(f'{f_name}/multi_syns_intra_{ct_dict[ct2]}_{ct_dict[ct3]}.csv')
            # make statistics and run results
            if len_ct2 > 0:
                #compare ct2 inter with intra
                ct2_multi_df = multi_conn_df[multi_conn_df['celltype'] == ct_dict[ct2]]
                ct2_multi_df_intra = multi_conn_df_intra[multi_conn_df_intra['celltype'] == ct_dict[ct2]]
                stats, p_vlaue = ranksums(ct2_multi_df['number of synapses'], ct2_multi_df_intra['number of synapses'])
                ranksum_results.loc['multi syn numbers stats', f'{ct_dict[ct2]} inter vs intra'] = stats
                ranksum_results.loc['multi syn numbers p-value', f'{ct_dict[ct2]} inter vs intra'] = p_value
                stats, p_vlaue = ranksums(ct2_multi_df['sum syn area'], ct2_multi_df_intra['sum syn area'])
                ranksum_results.loc['multi syn sum sizes stats', f'{ct_dict[ct2]} inter vs intra'] = stats
                ranksum_results.loc['multi syn sum sizess p-value', f'{ct_dict[ct2]} inter vs intra'] = p_value
            if len_ct3 > 0:
                #compare ct3 inter with intra
                ct3_multi_df = multi_conn_df[multi_conn_df['celltype'] == ct_dict[ct3]]
                ct3_multi_df_intra = multi_conn_df_intra[multi_conn_df_intra['celltype'] == ct_dict[ct3]]
                stats, p_vlaue = ranksums(ct3_multi_df['number of synapses'], ct3_multi_df_intra['number of synapses'])
                ranksum_results.loc['multi syn numbers stats', f'{ct_dict[ct3]} inter vs intra'] = stats
                ranksum_results.loc['multi syn numbers p-value', f'{ct_dict[ct3]} inter vs intra'] = p_value
                stats, p_vlaue = ranksums(ct3_multi_df['sum syn area'], ct3_multi_df_intra['sum syn area'])
                ranksum_results.loc['multi syn sum sizes stats', f'{ct_dict[ct3]} inter vs intra'] = stats
                ranksum_results.loc['multi syn sum sizess p-value', f'{ct_dict[ct3]} inter vs intra'] = p_value
            if len_ct2 > 0 and len_ct3 > 0:
                multi_ct2_numbers = multi_conn_df_intra['number of synapses'][multi_conn_df_intra['celltype'] == ct_dict[ct2]]
                multi_ct3_numbers = multi_conn_df_intra['number of synapses'][multi_conn_df_intra['celltype'] == ct_dict[ct3]]
                stats, p_value = ranksums(multi_ct2_numbers, multi_ct3_numbers)
                ranksum_results.loc['multi syn numbers stats', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]} intra'] = stats
                ranksum_results.loc['multi syn numbers p-value', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]} intra'] = p_value
                multi_ct2_sizes = multi_conn_df_intra['sum syn area'][multi_conn_df_intra['celltype'] == ct_dict[ct2]]
                multi_ct3_sizes = multi_conn_df_intra['sum syn area'][multi_conn_df_intra['celltype'] == ct_dict[ct3]]
                stats, p_value = ranksums(multi_ct2_sizes, multi_ct3_sizes)
            ranksum_results.loc['multi syn sum sizes stats', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]} intra'] = stats
            ranksum_results.loc['multi syn sum sizess p-value',f'to {ct_dict[ct2]} vs to {ct_dict[ct3]} intra'] = p_value
            ranksum_results.to_csv(f'{f_name}/ranksums_results_{ct_dict[ct2]}_{ct_dict[ct3]}.csv')
            # plot results
            # plot sum sizes for pairwise cells as histogram
            sns.histplot(x='sum syn area', data=multi_conn_df_intra, hue='celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True)
            plt.ylabel('number of cell-pairs')
            plt.xlabel('synaptic mesh area [µm²]')
            plt.title(f'Sum of synapse sizes in multisyn connections within celltypes')
            plt.savefig(f'{f_name}/multi_sum_sizes_intra_hist.png')
            plt.savefig(f'{f_name}/multi_sum_sizes_intra_hist.svg')
            plt.close()
            sns.histplot(x='sum syn area', data=multi_conn_df_intra, hue='celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True, stat='percent')
            plt.ylabel('% of cell-pairs')
            plt.xlabel('synaptic mesh area [µm²]')
            plt.title(f'Sum of synapse sizes in multisyn connections within celltypes')
            plt.savefig(f'{f_name}/multi_sum_sizes_intra_hist_perc.png')
            plt.savefig(f'{f_name}/multi_sum_sizes_intra_hist_perc.svg')
            plt.close()
            sns.histplot(x='sum syn area', data=multi_conn_df_intra, hue='celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True, log_scale=True)
            plt.ylabel('number of cell-pairs')
            plt.xlabel('synaptic mesh area [µm²]')
            plt.title(f'Sum of synapse sizes in multisyn connections within celltypes')
            plt.savefig(f'{f_name}/multi_sum_sizes_intra_hist_log.png')
            plt.savefig(f'{f_name}/multi_sum_sizes_intra_hist_log.svg')
            plt.close()
            sns.histplot(x='sum syn area', data=multi_conn_df_intra, hue='celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True, stat='percent', log_scale=True)
            plt.ylabel('% of cell-pairs')
            plt.xlabel('synaptic mesh area [µm²]')
            plt.title(f'Sum of synapse sizes in multisyn connections within celltypes')
            plt.savefig(f'{f_name}/multi_sum_sizes_intra_hist_log_perc.png')
            plt.savefig(f'{f_name}/multi_sum_sizes_intra_hist_log_perc.svg')
            plt.close()
            # plot again as barplot
            # plot number of synapses again as barplot
            # make bins for each number, seperated by ct2 and ct3
            if len_ct2 > 0 and len_ct3 > 0:
                ct2_multi_df = multi_conn_df_intra[multi_conn_df_intra['celltype'] == ct_dict[ct2]]
                ct2_inds, ct2_bins = pd.factorize(np.sort(ct2_multi_df['number of synapses']))
                ct2_counts = np.bincount(ct2_inds)
                ct3_multi_df = multi_conn_df_intra[multi_conn_df_intra['celltype'] == ct_dict[ct3]]
                ct3_inds, ct3_bins = pd.factorize(np.sort(ct3_multi_df['number of synapses']))
                ct3_counts = np.bincount(ct3_inds)
                num_ct2_bins = len(ct2_bins)
                len_bins = num_ct2_bins + len(ct3_bins)
                hist_df = pd.DataFrame(columns=['count of cell-pairs', 'percent of cell-pairs', 'bins', 'celltype'],
                                       index=range(len_bins))
                hist_df.loc[0: num_ct2_bins - 1, 'count of cell-pairs'] = ct2_counts
                hist_df.loc[0: num_ct2_bins - 1, 'percent of cell-pairs'] = 100 * ct2_counts / np.sum(ct2_counts)
                hist_df.loc[0: num_ct2_bins - 1, 'bins'] = ct2_bins
                hist_df.loc[0: num_ct2_bins - 1, 'celltype'] = ct_dict[ct2]
                hist_df.loc[num_ct2_bins: len_bins - 1, 'count of cell-pairs'] = ct3_counts
                hist_df.loc[num_ct2_bins: len_bins - 1, 'percent of cell-pairs'] = 100 * ct3_counts / np.sum(ct3_counts)
                hist_df.loc[num_ct2_bins: len_bins - 1, 'bins'] = ct3_bins
                hist_df.loc[num_ct2_bins: len_bins - 1, 'celltype'] = ct_dict[ct3]
                sns.barplot(data=hist_df, x='bins', y='count of cell-pairs', hue='celltype', palette=ct_palette)
                plt.xlabel('number of synapses')
                plt.savefig(f'{f_name}/multi_bar_syn_num_intra.svg')
                plt.savefig(f'{f_name}/multi_bar_syn_num_intra.png')
                plt.close()
                sns.barplot(data=hist_df, x='bins', y='percent of cell-pairs', hue='celltype', palette=ct_palette)
                plt.xlabel('number of synapses')
                plt.savefig(f'{f_name}/multi_bar_syn_num_intra_perc.svg')
                plt.savefig(f'{f_name}/multi_bar_syn_num_intra_perc.png')
                plt.close()
                # plot also only up to numbers of 10, 15, 20
                for num in [10, 15, 20]:
                    hist_df10 = hist_df[hist_df['bins'] <= num]
                    multi_conn_df11 = multi_conn_df_intra[multi_conn_df_intra['number of synapses'] > num]
                    hist_df11_ct2 = multi_conn_df11[multi_conn_df11['celltype'] == ct_dict[ct2]]
                    hist_df11_ct3 = multi_conn_df11[multi_conn_df11['celltype'] == ct_dict[ct3]]
                    log.info(
                        f'{len(multi_conn_df11)} cell-pairs out of {len(multi_conn_df_intra)} make multisynaptic connections with {num + 1} synapses or more, \n'
                        f'{len(hist_df11_ct2)} within {ct_dict[ct2]}, {len(hist_df11_ct3)} within {ct_dict[ct3]}')
                    sns.barplot(data=hist_df10, x='bins', y='count of cell-pairs', hue='celltype', palette=ct_palette)
                    plt.xlabel('number of synapses')
                    plt.savefig(f'{f_name}/cutoff{num}_multi_bar_syn_num_intra.svg')
                    plt.savefig(f'{f_name}/cutoff{num}_multi_bar_syn_num_intra.png')
                    plt.close()
                    sns.barplot(data=hist_df10, x='bins', y='percent of cell-pairs', hue='celltype', palette=ct_palette)
                    plt.xlabel('number of synapses')
                    plt.savefig(f'{f_name}/cutoff{num}_multi_bar_syn_num_intra_perc.svg')
                    plt.savefig(f'{f_name}/cutoff{num}_multi_bar_syn_num_intra_perc.png')
                    plt.close()
    else:
        log.info(f'Step 3/4: Get information outgoing from {ct_dict[conn_ct]}')
        log.info('Get per cell information to other celltype')
        # celltype indicates postsynaptic celltype
        percell_columns = ['cellid', 'celltype', f'syn number', f'sum syn size',
                           f'mean syn size', 'to celltype']
        num_conn_ct_ids = len(suitable_ids_dict[conn_ct])
        percell_result_df = pd.DataFrame(columns=percell_columns, index=range(num_conn_ct_ids*2))
        percell_result_df['cellid'] = np.hstack([suitable_ids_dict[conn_ct], suitable_ids_dict[conn_ct]])
        percell_result_df['celltype'] = ct_dict[conn_ct]
        # get number synapses and sum size per cells to other celltype
        #get information per cell about synapses to ct2
        #from function 'msn_spine_density_gp_ratio'
        log.info(f'Get per cell information to {ct_dict[ct2]}')
        ct2_inds = np.where(m_cts == ct2)
        ct2_ssv_partners = m_ssv_partners[ct2_inds[0]]
        ct2_sizes = m_sizes[ct2_inds[0]]
        ct2_cts = m_cts[ct2_inds[0]]
        #get cellids of all conncts
        conn_inds = np.where(ct2_cts == conn_ct)
        pre_ct2_ids = ct2_ssv_partners[conn_inds]
        #get cellids of ct2
        post_ct2_ids = m_ssv_partners[ct2_inds]
        ct2_syn_numbers, ct2_sum_sizes, unique_ssv_ids = get_ct_syn_number_sumsize(syn_sizes=ct2_sizes,
                                                                                   syn_ssv_partners=ct2_ssv_partners,
                                                                                   syn_cts=ct2_cts, ct=conn_ct)
        sort_inds_ct2 = np.argsort(unique_ssv_ids)
        unique_ssv_ids_sorted = unique_ssv_ids[sort_inds_ct2]
        ct2_syn_numbers_sorted = ct2_syn_numbers[sort_inds_ct2]
        ct2_sum_sizes_sorted = ct2_sum_sizes[sort_inds_ct2]
        sort_inds_ct2 = np.in1d(percell_result_df['cellid'], unique_ssv_ids_sorted)
        sort_inds_ct2[-num_conn_ct_ids:] = False
        percell_result_df.loc[sort_inds_ct2, 'syn number'] = ct2_syn_numbers_sorted
        percell_result_df.loc[sort_inds_ct2, 'sum syn size'] = ct2_sum_sizes_sorted
        percell_result_df.loc[sort_inds_ct2, 'mean syn size'] = ct2_sum_sizes_sorted / ct2_syn_numbers_sorted
        percell_result_df.loc[sort_inds_ct2, 'to celltype'] = ct_dict[ct2]
        log.info(
            f'{len(unique_ssv_ids_sorted)} out of {num_conn_ct_ids} {ct_dict[conn_ct]} make synapses to {ct_dict[ct2]}')
        log.info(f'Get per cell information to {ct_dict[ct3]}')
        ct3_inds = np.where(m_cts == ct3)
        ct3_ssv_partners = m_ssv_partners[ct3_inds[0]]
        ct3_sizes = m_sizes[ct3_inds[0]]
        ct3_cts = m_cts[ct3_inds[0]]
        # get cellids of all conncts
        conn_inds = np.where(ct3_cts == conn_ct)
        pre_ct3_ids = ct3_ssv_partners[conn_inds]
        # get cellids of ct2
        post_ct3_ids = m_ssv_partners[ct3_inds]
        ct3_syn_numbers, ct3_sum_sizes, unique_ssv_ids = get_ct_syn_number_sumsize(syn_sizes=ct3_sizes,
                                                                                   syn_ssv_partners=ct3_ssv_partners,
                                                                                   syn_cts=ct3_cts, ct=conn_ct)
        sort_inds_ct3 = np.argsort(unique_ssv_ids)
        unique_ssv_ids_sorted = unique_ssv_ids[sort_inds_ct3]
        ct3_syn_numbers_sorted = ct3_syn_numbers[sort_inds_ct3]
        ct3_sum_sizes_sorted = ct3_sum_sizes[sort_inds_ct3]
        sort_inds_ct3 = np.in1d(percell_result_df['cellid'], unique_ssv_ids_sorted)
        sort_inds_ct3[num_conn_ct_ids:] = False
        percell_result_df.loc[sort_inds_ct3, 'syn number'] = ct3_syn_numbers_sorted
        percell_result_df.loc[sort_inds_ct3, 'sum syn size'] = ct3_sum_sizes_sorted
        percell_result_df.loc[sort_inds_ct3, 'mean syn size'] = ct3_sum_sizes_sorted / ct3_syn_numbers_sorted
        percell_result_df.loc[sort_inds_ct3, 'to celltype'] = ct_dict[ct3]
        percell_result_df = percell_result_df.dropna()
        percell_result_df = percell_result_df.reset_index(drop = True)
        percell_result_df.to_csv(f'{f_name}/percell_{ct_dict[conn_ct]}_outgoing_{ct_dict[ct2]}_{ct_dict[ct3]}.csv')
        log.info(
            f'{len(unique_ssv_ids_sorted)} out of {num_conn_ct_ids} {ct_dict[conn_ct]} make synapses to {ct_dict[ct3]}')
        log.info('Plot and get stats')
        ranksum_columns = [f'to {ct_dict[ct2]} vs to {ct_dict[ct3]}',
                           f'from {ct_dict[ct2]} vs from {ct_dict[ct3]}']
        ranksum_results = pd.DataFrame(columns=ranksum_columns)
        for key in percell_result_df.keys():
            if 'cellid' in key or 'celltype' in key:
                continue
            key_groups = [group[key].values for name, group in
                          percell_result_df.groupby('to celltype')]
            stats, p_value = ranksums(key_groups[0], key_groups[1])
            ranksum_results.loc[f'{key} per cell stats', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]}'] = stats
            ranksum_results.loc[
                f'{key} per cell p-value', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]}'] = p_value
            if 'size' in key:
                xlabel = f'{key} [µm²]'
            else:
                xlabel = key
            sns.histplot(x=key, data=percell_result_df, hue='to celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True)
            plt.ylabel('number of cells')
            plt.xlabel(xlabel)
            plt.title(f'{key} synapses outgoing from {ct_dict[conn_ct]} cells')
            plt.savefig(f'{f_name}/percell_{key}_{ct_dict[conn_ct]}_outgoing_hist.png')
            plt.savefig(f'{f_name}/percell_{key}_{ct_dict[conn_ct]}_outgoing_hist.svg')
            plt.close()
            sns.histplot(x=key, data=percell_result_df, hue='to celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True, stat='percent')
            plt.ylabel('% of cells')
            plt.xlabel(xlabel)
            plt.title(f'{key} synapses outgoing from {ct_dict[conn_ct]} cells')
            plt.savefig(f'{f_name}/percell_{key}_{ct_dict[conn_ct]}_outgoing_hist_perc.png')
            plt.savefig(f'{f_name}/percell_{key}_{ct_dict[conn_ct]}_outgoing_hist_perc.svg')
            plt.close()
            sns.histplot(x=key, data=percell_result_df, hue='to celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True, log_scale=True)
            plt.ylabel('number of cells')
            plt.xlabel(xlabel)
            plt.title(f'{key} synapses outgoing from {ct_dict[conn_ct]} cells')
            plt.savefig(f'{f_name}/percell_{key}_{ct_dict[conn_ct]}_outgoing_hist_log.png')
            plt.savefig(f'{f_name}/percell_{key}_{ct_dict[conn_ct]}_outgoing_hist_log.svg')
            plt.close()
            sns.histplot(x=key, data=percell_result_df, hue='to celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True, stat='percent', log_scale=True)
            plt.ylabel('% of cells')
            plt.xlabel(xlabel)
            plt.title(f'{key} synapses outgoing from {ct_dict[conn_ct]} cells')
            plt.savefig(f'{f_name}/percell_{key}_{ct_dict[conn_ct]}_outgoing_hist_perc_log.png')
            plt.savefig(f'{f_name}/percell_{key}_{ct_dict[conn_ct]}_outgoing_hist_perc_log.svg')
            plt.close()
            if len(percell_result_df) <= 500:
                sns.stripplot(x='to celltype', y=key, data=percell_result_df, color='black',
                              alpha=0.4,
                              dodge=True, size=3)
                sns.boxplot(x='to celltype', y=key, data=percell_result_df, palette=ct_palette)
                plt.ylabel(xlabel)
                plt.title(f'{key} synapses outgoing from {ct_dict[conn_ct]} cells')
                plt.savefig(f'{f_name}/percell_{key}_{ct_dict[conn_ct]}_outgoing_box.png')
                plt.savefig(f'{f_name}/percell_{key}_{ct_dict[conn_ct]}_outgoing_box.svg')
                plt.close()
        log.info('Get information about all synapses, make statistics and plot')
        len_syns = len(ct2_sizes) + len(ct3_sizes)
        syn_sizes_df = pd.DataFrame(columns=['syn sizes', 'to celltype', 'cellid pre', 'cellid post'],
                                    index=range(len_syns))
        syn_sizes_df.loc[0: len(ct2_sizes) - 1, 'syn sizes'] = ct2_sizes
        syn_sizes_df.loc[0: len(ct2_sizes) - 1, 'to celltype'] = ct_dict[ct2]
        syn_sizes_df.loc[0: len(ct2_sizes) - 1, 'cellid pre'] = pre_ct2_ids
        syn_sizes_df.loc[0: len(ct2_sizes) - 1, 'cellid post'] = post_ct2_ids
        syn_sizes_df.loc[len(ct2_sizes): len_syns - 1, 'syn sizes'] = ct3_sizes
        syn_sizes_df.loc[len(ct2_sizes): len_syns - 1, 'to celltype'] = ct_dict[ct3]
        syn_sizes_df.loc[len(ct2_sizes): len_syns - 1, 'cellid pre'] = pre_ct3_ids
        syn_sizes_df.loc[len(ct2_sizes): len_syns - 1, 'cellid post'] = post_ct3_ids
        syn_sizes_df.to_csv(f'{f_name}/all_syn_sizes_{ct_dict[conn_ct]}_outgoing_{ct_dict[ct2]}_{ct_dict[ct3]}.csv')
        stats, p_value = ranksums(ct2_sizes, ct3_sizes)
        ranksum_results.loc['all syn sizes stats', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]}'] = stats
        ranksum_results.loc['all syn sizes p-value', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]}'] = p_value
        ranksum_results.to_csv(f'{f_name}/ranksums_results_{ct_dict[conn_ct]}_{ct_dict[ct3]}_{ct_dict[ct3]}.csv')
        # plot all syn sizes values
        sns.histplot(x='syn sizes', data=syn_sizes_df, hue='to celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True)
        plt.ylabel('number of synapses')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Synapse sizes outgoing from {ct_dict[conn_ct]} cells')
        plt.savefig(f'{f_name}/all_synsizes_outgoing_hist.png')
        plt.savefig(f'{f_name}/all_synsizes_outgoing_hist.svg')
        plt.close()
        sns.histplot(x='syn sizes', data=syn_sizes_df, hue='to celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, log_scale=True)
        plt.ylabel('number of synapses')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Synapse sizes outgoing from {ct_dict[conn_ct]} cells')
        plt.savefig(f'{f_name}/all_synsizes_outgoing_hist_log.png')
        plt.savefig(f'{f_name}/all_synsizes_outgoing_hist_log.svg')
        plt.close()
        sns.histplot(x='syn sizes', data=syn_sizes_df, hue='to celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent')
        plt.ylabel('% of synapses')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Synapse sizes outgoing from {ct_dict[conn_ct]} cells')
        plt.savefig(f'{f_name}/all_synsizes_outgoing_hist_perc.png')
        plt.savefig(f'{f_name}/all_synsizes_outgoing_hist_perc.svg')
        plt.close()
        sns.histplot(x='syn sizes', data=syn_sizes_df, hue='to celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, log_scale=True, stat='percent')
        plt.ylabel('% of synapses')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Synapse sizes outgoing from {ct_dict[conn_ct]} cells')
        plt.savefig(f'{f_name}/all_synsizes_outgoing_hist_log_perc.png')
        plt.savefig(f'{f_name}/all_synsizes_outgoing_hist_log_perc.svg')
        plt.close()
        log.info('Get information about multi-syn connections')
        # get number of ct2 partner cells and size information for all of them
        #use syn_sizes df for information
        all_unique_pre_ids = np.unique(syn_sizes_df['cellid pre'])
        num_pre_ids = len(all_unique_pre_ids)
        all_unique_post_ids = np.unique(syn_sizes_df['cellid post'])
        num_post_ids = len(all_unique_post_ids)
        multi_conn_df = pd.DataFrame(
            columns=['number of synapses', 'sum syn area', 'to celltype', 'cellid pre', 'cellid post'],
            index=range(num_pre_ids * num_post_ids))

        for i, cell_id in enumerate(tqdm(all_unique_pre_ids)):
            cell_syn_sizes_df = syn_sizes_df[syn_sizes_df['cellid pre'] == cell_id]
            # for ct2
            ct2_cell_syn_sizes_df = cell_syn_sizes_df[cell_syn_sizes_df['to celltype'] == ct_dict[ct2]]
            ct2_syn_numbers, ct2_sum_sizes, unique_ct2_ids = get_percell_number_sumsize(ct2_cell_syn_sizes_df['cellid post'], ct2_cell_syn_sizes_df['syn sizes'])
            len_multi_syns_ct2 = len(ct2_syn_numbers)
            start_ct2 = i * num_post_ids
            end_ct2 = start_ct2 + len_multi_syns_ct2 - 1
            multi_conn_df.loc[start_ct2: end_ct2, 'number of synapses'] = ct2_syn_numbers
            multi_conn_df.loc[start_ct2:end_ct2, 'sum syn area'] = ct2_sum_sizes
            multi_conn_df.loc[start_ct2: end_ct2, 'to celltype'] = ct_dict[ct2]
            multi_conn_df.loc[start_ct2: end_ct2, 'cellid pre'] = cell_id
            multi_conn_df.loc[start_ct2: end_ct2, 'cellid post'] = unique_ct2_ids
            ct3_cell_syn_sizes_df = cell_syn_sizes_df[cell_syn_sizes_df['to celltype'] == ct_dict[ct3]]
            ct3_syn_numbers, ct3_sum_sizes, unique_ct3_ids = get_percell_number_sumsize(
                ct3_cell_syn_sizes_df['cellid post'], ct3_cell_syn_sizes_df['syn sizes'])
            len_multi_syns_ct3 = len(ct3_syn_numbers)
            start_ct3 = i * num_post_ids + len_multi_syns_ct2
            end_ct3 = start_ct3 + len_multi_syns_ct3 - 1
            multi_conn_df.loc[start_ct3: end_ct3, 'number of synapses'] = ct3_syn_numbers
            multi_conn_df.loc[start_ct3:end_ct3, 'sum syn area'] = ct3_sum_sizes
            multi_conn_df.loc[start_ct3: end_ct3, 'to celltype'] = ct_dict[ct3]
            multi_conn_df.loc[start_ct3: end_ct3, 'cellid pre'] = cell_id
            multi_conn_df.loc[start_ct3: end_ct3, 'cellid post'] = unique_ct3_ids

        multi_conn_df = multi_conn_df.dropna()
        multi_conn_df = multi_conn_df.reset_index(drop = True)
        multi_conn_df.to_csv(f'{f_name}/multi_syns_{ct_dict[conn_ct]}_outgoing.csv')
        #make statistics and run results
        multi_ct2_numbers = multi_conn_df['number of synapses'][multi_conn_df['to celltype'] == ct_dict[ct2]]
        multi_ct3_numbers = multi_conn_df['number of synapses'][multi_conn_df['to celltype'] == ct_dict[ct3]]
        stats, p_value = ranksums(multi_ct2_numbers, multi_ct3_numbers)
        ranksum_results.loc['multi syn numbers stats', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]}'] = stats
        ranksum_results.loc['multi syn numbers p-value', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]}'] = p_value
        multi_ct2_sizes = multi_conn_df['sum syn area'][multi_conn_df['to celltype'] == ct_dict[ct2]]
        multi_ct3_sizes = multi_conn_df['sum syn area'][multi_conn_df['to celltype'] == ct_dict[ct3]]
        stats, p_value = ranksums(multi_ct2_sizes, multi_ct3_sizes)
        ranksum_results.loc['multi syn sum sizes stats', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]}'] = stats
        ranksum_results.loc['multi syn sum sizess p-value', f'to {ct_dict[ct2]} vs to {ct_dict[ct3]}'] = p_value
        ranksum_results.to_csv(f'{f_name}/ranksums_results_{ct_dict[conn_ct]}_{ct_dict[ct3]}_{ct_dict[ct3]}.csv')
        #plot results
        #plot sum sizes for pairwise cells as histogram
        sns.histplot(x='sum syn area', data=multi_conn_df, hue='to celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True)
        plt.ylabel('number of cell-pairs')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Sum of synapse sizes outgoing from {ct_dict[conn_ct]} multisyn connections')
        plt.savefig(f'{f_name}/multi_sum_sizes_outgoing_hist.png')
        plt.savefig(f'{f_name}/multi_sum_sizes_outgoing_hist.svg')
        plt.close()
        sns.histplot(x='sum syn area', data=multi_conn_df, hue='to celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat = 'percent')
        plt.ylabel('% of cell-pairs')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Sum of synapse sizes outgoing from {ct_dict[conn_ct]} multisyn connections')
        plt.savefig(f'{f_name}/multi_sum_sizes_outgoing_hist_perc.png')
        plt.savefig(f'{f_name}/multi_sum_sizes_outgoing_hist_perc.svg')
        plt.close()
        sns.histplot(x='sum syn area', data=multi_conn_df, hue='to celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, log_scale = True)
        plt.ylabel('number of cell-pairs')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Sum of synapse sizes outgoing from {ct_dict[conn_ct]} multisyn connections')
        plt.savefig(f'{f_name}/multi_sum_sizes_outgoing_hist_log.png')
        plt.savefig(f'{f_name}/multi_sum_sizes_outgoing_hist_log.svg')
        plt.close()
        sns.histplot(x='sum syn area', data=multi_conn_df, hue='to celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat = 'percent', log_scale=True)
        plt.ylabel('% of cell-pairs')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Sum of synapse sizes outgoing from {ct_dict[conn_ct]} multisyn connections')
        plt.savefig(f'{f_name}/multi_sum_sizes_outgoing_hist_log_perc.png')
        plt.savefig(f'{f_name}/multi_sum_sizes_outgoing_hist_log_perc.svg')
        plt.close()
        #plot again as barplot
        # plot number of synapses again as barplot
        # make bins for each number, seperated by ct2 and ct3
        ct2_multi_df = multi_conn_df[multi_conn_df['to celltype'] == ct_dict[ct2]]
        ct2_inds, ct2_bins = pd.factorize(np.sort(ct2_multi_df['number of synapses']))
        ct2_counts = np.bincount(ct2_inds)
        ct3_multi_df = multi_conn_df[multi_conn_df['to celltype'] == ct_dict[ct3]]
        ct3_inds, ct3_bins = pd.factorize(np.sort(ct3_multi_df['number of synapses']))
        ct3_counts = np.bincount(ct3_inds)
        num_ct2_bins = len(ct2_bins)
        len_bins = num_ct2_bins + len(ct3_bins)
        hist_df = pd.DataFrame(columns=['count of cell-pairs', 'percent of cell-pairs', 'bins', 'to celltype'],
                               index=range(len_bins))
        hist_df.loc[0: num_ct2_bins - 1, 'count of cell-pairs'] = ct2_counts
        hist_df.loc[0: num_ct2_bins - 1, 'percent of cell-pairs'] = 100 * ct2_counts / np.sum(ct2_counts)
        hist_df.loc[0: num_ct2_bins - 1, 'bins'] = ct2_bins
        hist_df.loc[0: num_ct2_bins - 1, 'to celltype'] = ct_dict[ct2]
        hist_df.loc[num_ct2_bins: len_bins - 1, 'count of cell-pairs'] = ct3_counts
        hist_df.loc[num_ct2_bins: len_bins - 1, 'percent of cell-pairs'] = 100 * ct3_counts / np.sum(ct3_counts)
        hist_df.loc[num_ct2_bins: len_bins - 1, 'bins'] = ct3_bins
        hist_df.loc[num_ct2_bins: len_bins - 1, 'to celltype'] = ct_dict[ct3]
        sns.barplot(data=hist_df, x='bins', y='count of cell-pairs', hue='to celltype', palette=ct_palette)
        plt.xlabel('number of synapses')
        plt.savefig(f'{f_name}/multi_bar_syn_num_outgoing.svg')
        plt.savefig(f'{f_name}/multi_bar_syn_num_outgoing.png')
        plt.close()
        sns.barplot(data=hist_df, x='bins', y='percent of cell-pairs', hue='to celltype', palette=ct_palette)
        plt.xlabel('number of synapses')
        plt.savefig(f'{f_name}/multi_bar_syn_num_outgoing_perc.svg')
        plt.savefig(f'{f_name}/multi_bar_syn_num_outgoing_perc.png')
        plt.close()
        #plot also only up to numbers of 10, 15, 20
        for num in [10, 15, 20]:
            hist_df10 = hist_df[hist_df['bins'] <= num]
            multi_conn_df11 = multi_conn_df[multi_conn_df['number of synapses'] > num]
            hist_df11_ct2 = multi_conn_df11[multi_conn_df11['to celltype'] == ct_dict[ct2]]
            hist_df11_ct3 = multi_conn_df11[multi_conn_df11['to celltype'] == ct_dict[ct3]]
            log.info(f'{len(multi_conn_df11)} cell-pairs out of {len(multi_conn_df)} make multisynaptic connections with {num + 1} synapses or more, \n'
                     f'{len(hist_df11_ct2)} to {ct_dict[ct2]}, {len(hist_df11_ct3)} to {ct_dict[ct3]}')
            sns.barplot(data=hist_df10, x='bins', y='count of cell-pairs', hue='to celltype', palette=ct_palette)
            plt.xlabel('number of synapses')
            plt.savefig(f'{f_name}/cutoff{num}_multi_bar_syn_num_outgoing.svg')
            plt.savefig(f'{f_name}/cutoff{num}_multi_bar_syn_num_outgoing.png')
            plt.close()
            sns.barplot(data=hist_df10, x='bins', y='percent of cell-pairs', hue='to celltype', palette=ct_palette)
            plt.xlabel('number of synapses')
            plt.savefig(f'{f_name}/cutoff{num}_multi_bar_syn_num_outgoing_perc.svg')
            plt.savefig(f'{f_name}/cutoff{num}_multi_bar_syn_num_outgoing_perc.png')
            plt.close()
        log.info(f'Step 4/4: Get information incoming to {ct_dict[conn_ct]}')
        suit_ct_inds = np.all(np.in1d(in_m_ssv_partners, all_suitable_ids).reshape(len(in_m_ssv_partners), 2), axis=1)
        in_m_cts = in_m_cts[suit_ct_inds]
        in_m_ssv_partners = in_m_ssv_partners[suit_ct_inds]
        in_m_sizes = in_m_sizes[suit_ct_inds]
        in_m_axs = in_m_axs[suit_ct_inds]
        in_m_rep_coord = in_m_rep_coord[suit_ct_inds]
        percell_columns = ['cellid', 'celltype', f'syn number', f'sum syn size',
                           f'mean syn size', 'from celltype']
        percell_result_df = pd.DataFrame(columns=percell_columns, index=range(num_conn_ct_ids * 2))
        percell_result_df['cellid'] = np.hstack([suitable_ids_dict[conn_ct], suitable_ids_dict[conn_ct]])
        percell_result_df['celltype'] = ct_dict[conn_ct]
        # get number synapses and sum size per cells to other celltype
        # get information per cell about synapses to ct2
        # from function 'msn_spine_density_gp_ratio'
        log.info(f'Get per cell information from {ct_dict[ct2]}')
        ct2_inds = np.where(in_m_cts == ct2)
        ct2_ssv_partners = in_m_ssv_partners[ct2_inds[0]]
        ct2_sizes = in_m_sizes[ct2_inds[0]]
        ct2_cts = in_m_cts[ct2_inds[0]]
        # get cellids of all conncts
        conn_inds = np.where(ct2_cts == conn_ct)
        post_ct2_ids = ct2_ssv_partners[conn_inds]
        # get cellids of ct2
        pre_ct2_ids = m_ssv_partners[ct2_inds]
        ct2_syn_numbers, ct2_sum_sizes, unique_ssv_ids = get_ct_syn_number_sumsize(syn_sizes=ct2_sizes,
                                                                                   syn_ssv_partners=ct2_ssv_partners,
                                                                                   syn_cts=ct2_cts, ct=conn_ct)
        sort_inds_ct2 = np.argsort(unique_ssv_ids)
        unique_ssv_ids_sorted = unique_ssv_ids[sort_inds_ct2]
        ct2_syn_numbers_sorted = ct2_syn_numbers[sort_inds_ct2]
        ct2_sum_sizes_sorted = ct2_sum_sizes[sort_inds_ct2]
        sort_inds_ct2 = np.in1d(percell_result_df['cellid'], unique_ssv_ids_sorted)
        sort_inds_ct2[-num_conn_ct_ids:] = False
        percell_result_df.loc[sort_inds_ct2, 'syn number'] = ct2_syn_numbers_sorted
        percell_result_df.loc[sort_inds_ct2, 'sum syn size'] = ct2_sum_sizes_sorted
        percell_result_df.loc[sort_inds_ct2, 'mean syn size'] = ct2_sum_sizes_sorted / ct2_syn_numbers_sorted
        percell_result_df.loc[sort_inds_ct2, 'from celltype'] = ct_dict[ct2]
        log.info(
            f'{len(unique_ssv_ids_sorted)} out of {num_conn_ct_ids} {ct_dict[conn_ct]} get synapses from {ct_dict[ct2]}')
        log.info(f'Get per cell information from {ct_dict[ct3]}')
        ct3_inds = np.where(m_cts == ct3)
        ct3_ssv_partners = m_ssv_partners[ct3_inds[0]]
        ct3_sizes = m_sizes[ct3_inds[0]]
        ct3_cts = m_cts[ct3_inds[0]]
        # get cellids of all conncts
        conn_inds = np.where(ct3_cts == conn_ct)
        post_ct3_ids = ct3_ssv_partners[conn_inds]
        # get cellids of ct2
        pre_ct3_ids = m_ssv_partners[ct3_inds]
        ct3_syn_numbers, ct3_sum_sizes, unique_ssv_ids = get_ct_syn_number_sumsize(syn_sizes=ct3_sizes,
                                                                                   syn_ssv_partners=ct3_ssv_partners,
                                                                                   syn_cts=ct3_cts, ct=conn_ct)
        sort_inds_ct3 = np.argsort(unique_ssv_ids)
        unique_ssv_ids_sorted = unique_ssv_ids[sort_inds_ct3]
        ct3_syn_numbers_sorted = ct3_syn_numbers[sort_inds_ct3]
        ct3_sum_sizes_sorted = ct3_sum_sizes[sort_inds_ct3]
        sort_inds_ct3 = np.in1d(percell_result_df['cellid'], unique_ssv_ids_sorted)
        sort_inds_ct3[num_conn_ct_ids:] = False
        percell_result_df.loc[sort_inds_ct3, 'syn number'] = ct3_syn_numbers_sorted
        percell_result_df.loc[sort_inds_ct3, 'sum syn size'] = ct3_sum_sizes_sorted
        percell_result_df.loc[sort_inds_ct3, 'mean syn size'] = ct3_sum_sizes_sorted / ct3_syn_numbers_sorted
        percell_result_df.loc[sort_inds_ct3, 'from celltype'] = ct_dict[ct3]
        percell_result_df = percell_result_df.dropna()
        percell_result_df = percell_result_df.reset_index(drop=True)
        percell_result_df.to_csv(f'{f_name}/percell_{ct_dict[conn_ct]}_incoming_{ct_dict[ct2]}_{ct_dict[ct3]}.csv')
        log.info(
            f'{len(unique_ssv_ids_sorted)} out of {num_conn_ct_ids} {ct_dict[conn_ct]} get synapses from {ct_dict[ct3]}')
        log.info('Plot and get stats')
        for key in percell_result_df.keys():
            if 'cellid' in key or 'celltype' in key:
                continue
            key_groups = [group[key].values for name, group in
                          percell_result_df.groupby('from celltype')]
            stats, p_value = ranksums(key_groups[0], key_groups[1])
            ranksum_results.loc[f'{key} per cell stats', f'from {ct_dict[ct2]} vs from {ct_dict[ct3]}'] = stats
            ranksum_results.loc[
                f'{key} per cell p-value', f'from {ct_dict[ct2]} vs from {ct_dict[ct3]}'] = p_value
            if 'size' in key:
                xlabel = f'{key} [µm²]'
            else:
                xlabel = key
            sns.histplot(x=key, data=percell_result_df, hue='from celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True)
            plt.ylabel('number of cells')
            plt.xlabel(xlabel)
            plt.title(f'{key} synapses incoming to {ct_dict[conn_ct]} cells')
            plt.savefig(f'{f_name}/percell_{key}_{ct_dict[conn_ct]}_incoming_hist.png')
            plt.savefig(f'{f_name}/percell_{key}_{ct_dict[conn_ct]}_incoming_hist.svg')
            plt.close()
            sns.histplot(x=key, data=percell_result_df, hue='from celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True, stat='percent')
            plt.ylabel('% of cells')
            plt.xlabel(xlabel)
            plt.title(f'{key} synapses incoming to {ct_dict[conn_ct]} cells')
            plt.savefig(f'{f_name}/percell_{key}_{ct_dict[conn_ct]}_incoming_hist_perc.png')
            plt.savefig(f'{f_name}/percell_{key}_{ct_dict[conn_ct]}_incoming_hist_perc.svg')
            plt.close()
            sns.histplot(x=key, data=percell_result_df, hue='from celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True, log_scale=True)
            plt.ylabel('number of cells')
            plt.xlabel(xlabel)
            plt.title(f'{key} synapses incoming to {ct_dict[conn_ct]} cells')
            plt.savefig(f'{f_name}/percell_{key}_{ct_dict[conn_ct]}_incoming_hist_log.png')
            plt.savefig(f'{f_name}/percell_{key}_{ct_dict[conn_ct]}_incoming_hist_log.svg')
            plt.close()
            sns.histplot(x=key, data=percell_result_df, hue='from celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True, stat='percent', log_scale=True)
            plt.ylabel('% of cells')
            plt.xlabel(xlabel)
            plt.title(f'{key} synapses incoming to {ct_dict[conn_ct]} cells')
            plt.savefig(f'{f_name}/percell_{key}_{ct_dict[conn_ct]}_incoming_hist_perc_log.png')
            plt.savefig(f'{f_name}/percell_{key}_{ct_dict[conn_ct]}_incoming_hist_perc_log.svg')
            plt.close()
            if len(percell_result_df) <= 500:
                sns.stripplot(x='from celltype', y=key, data=percell_result_df, color='black',
                              alpha=0.4,
                              dodge=True, size=3)
                sns.boxplot(x='from celltype', y=key, data=percell_result_df, palette=ct_palette)
                plt.ylabel(xlabel)
                plt.title(f'{key} synapses incoming to {ct_dict[conn_ct]} cells')
                plt.savefig(f'{f_name}/percell_{key}_{ct_dict[conn_ct]}_incoming_box.png')
                plt.savefig(f'{f_name}/percell_{key}_{ct_dict[conn_ct]}_incoming_box.svg')
                plt.close()
        log.info('Get information about all synapses, make statistics and plot')
        len_syns = len(ct2_sizes) + len(ct3_sizes)
        syn_sizes_df = pd.DataFrame(columns=['syn sizes', 'from celltype', 'cellid pre', 'cellid post'],
                                    index=range(len_syns))
        syn_sizes_df.loc[0: len(ct2_sizes) - 1, 'syn sizes'] = ct2_sizes
        syn_sizes_df.loc[0: len(ct2_sizes) - 1, 'from celltype'] = ct_dict[ct2]
        syn_sizes_df.loc[0: len(ct2_sizes) - 1, 'cellid pre'] = pre_ct2_ids
        syn_sizes_df.loc[0: len(ct2_sizes) - 1, 'cellid post'] = post_ct2_ids
        syn_sizes_df.loc[len(ct2_sizes): len_syns - 1, 'syn sizes'] = ct3_sizes
        syn_sizes_df.loc[len(ct2_sizes): len_syns - 1, 'from celltype'] = ct_dict[ct3]
        syn_sizes_df.loc[len(ct2_sizes): len_syns - 1, 'cellid pre'] = pre_ct3_ids
        syn_sizes_df.loc[len(ct2_sizes): len_syns - 1, 'cellid post'] = post_ct3_ids
        syn_sizes_df.to_csv(f'{f_name}/all_syn_sizes_{ct_dict[conn_ct]}_incoming_{ct_dict[ct2]}_{ct_dict[ct3]}.csv')
        stats, p_value = ranksums(ct2_sizes, ct3_sizes)
        ranksum_results.loc['all syn sizes stats', f'from {ct_dict[ct2]} vs from {ct_dict[ct3]}'] = stats
        ranksum_results.loc['all syn sizes p-value', f'from {ct_dict[ct2]} vs from {ct_dict[ct3]}'] = p_value
        ranksum_results.to_csv(f'{f_name}/ranksums_results_{ct_dict[conn_ct]}_{ct_dict[ct3]}_{ct_dict[ct3]}.csv')
        # plot all syn sizes values
        sns.histplot(x='syn sizes', data=syn_sizes_df, hue='from celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True)
        plt.ylabel('number of synapses')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Synapse sizes incoming to {ct_dict[conn_ct]} cells')
        plt.savefig(f'{f_name}/all_synsizes_incoming_hist.png')
        plt.savefig(f'{f_name}/all_synsizes_incoming_hist.svg')
        plt.close()
        sns.histplot(x='syn sizes', data=syn_sizes_df, hue='from celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, log_scale=True)
        plt.ylabel('number of synapses')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Synapse sizes incoming from {ct_dict[conn_ct]} cells')
        plt.savefig(f'{f_name}/all_synsizes_incoming_hist_log.png')
        plt.savefig(f'{f_name}/all_synsizes_incoming_hist_log.svg')
        plt.close()
        sns.histplot(x='syn sizes', data=syn_sizes_df, hue='from celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent')
        plt.ylabel('% of synapses')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Synapse sizes incoming to {ct_dict[conn_ct]} cells')
        plt.savefig(f'{f_name}/all_synsizes_incoming_hist_perc.png')
        plt.savefig(f'{f_name}/all_synsizes_incoming_hist_perc.svg')
        plt.close()
        sns.histplot(x='syn sizes', data=syn_sizes_df, hue='from celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, log_scale=True, stat='percent')
        plt.ylabel('% of synapses')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Synapse sizes incoming to {ct_dict[conn_ct]} cells')
        plt.savefig(f'{f_name}/all_synsizes_incoming_hist_log_perc.png')
        plt.savefig(f'{f_name}/all_synsizes_incoming_hist_log_perc.svg')
        plt.close()
        log.info('Get information about multi-syn connections')
        # get number of ct2 partner cells and size information for all of them
        # use syn_sizes df for information
        all_unique_pre_ids = np.unique(syn_sizes_df['cellid pre'])
        num_pre_ids = len(all_unique_pre_ids)
        all_unique_post_ids = np.unique(syn_sizes_df['cellid post'])
        num_post_ids = len(all_unique_post_ids)
        multi_conn_df = pd.DataFrame(
            columns=['number of synapses', 'sum syn area', 'from celltype', 'cellid pre', 'cellid post'],
            index=range(num_pre_ids * num_post_ids))

        for i, cell_id in enumerate(tqdm(all_unique_post_ids)):
            cell_syn_sizes_df = syn_sizes_df[syn_sizes_df['cellid post'] == cell_id]
            # for ct2
            ct2_cell_syn_sizes_df = cell_syn_sizes_df[cell_syn_sizes_df['from celltype'] == ct_dict[ct2]]
            ct2_syn_numbers, ct2_sum_sizes, unique_ct2_ids = get_percell_number_sumsize(
                ct2_cell_syn_sizes_df['cellid pre'], ct2_cell_syn_sizes_df['syn sizes'])
            len_multi_syns_ct2 = len(ct2_syn_numbers)
            start_ct2 = i * num_pre_ids
            end_ct2 = start_ct2 + len_multi_syns_ct2 - 1
            multi_conn_df.loc[start_ct2: end_ct2, 'number of synapses'] = ct2_syn_numbers
            multi_conn_df.loc[start_ct2:end_ct2, 'sum syn area'] = ct2_sum_sizes
            multi_conn_df.loc[start_ct2: end_ct2, 'from celltype'] = ct_dict[ct2]
            multi_conn_df.loc[start_ct2: end_ct2, 'cellid pre'] = unique_ct2_ids
            multi_conn_df.loc[start_ct2: end_ct2, 'cellid post'] = cell_id
            ct3_cell_syn_sizes_df = cell_syn_sizes_df[cell_syn_sizes_df['from celltype'] == ct_dict[ct3]]
            ct3_syn_numbers, ct3_sum_sizes, unique_ct3_ids = get_percell_number_sumsize(
                ct3_cell_syn_sizes_df['cellid pre'], ct3_cell_syn_sizes_df['syn sizes'])
            len_multi_syns_ct3 = len(ct3_syn_numbers)
            start_ct3 = i * num_pre_ids + len_multi_syns_ct2
            end_ct3 = start_ct3 + len_multi_syns_ct3 - 1
            multi_conn_df.loc[start_ct3: end_ct3, 'number of synapses'] = ct3_syn_numbers
            multi_conn_df.loc[start_ct3:end_ct3, 'sum syn area'] = ct3_sum_sizes
            multi_conn_df.loc[start_ct3: end_ct3, 'from celltype'] = ct_dict[ct3]
            multi_conn_df.loc[start_ct3: end_ct3, 'cellid pre'] = unique_ct3_ids
            multi_conn_df.loc[start_ct3: end_ct3, 'cellid post'] = cell_id

        multi_conn_df = multi_conn_df.dropna()
        multi_conn_df = multi_conn_df.reset_index(drop=True)
        multi_conn_df.to_csv(f'{f_name}/multi_syns_{ct_dict[conn_ct]}_incoming.csv')
        # make statistics and run results
        multi_ct2_numbers = multi_conn_df['number of synapses'][multi_conn_df['from celltype'] == ct_dict[ct2]]
        multi_ct3_numbers = multi_conn_df['number of synapses'][multi_conn_df['from celltype'] == ct_dict[ct3]]
        stats, p_value = ranksums(multi_ct2_numbers, multi_ct3_numbers)
        ranksum_results.loc['multi syn numbers stats', f'from {ct_dict[ct2]} vs from {ct_dict[ct3]}'] = stats
        ranksum_results.loc['multi syn numbers p-value', f'from {ct_dict[ct2]} vs from {ct_dict[ct3]}'] = p_value
        multi_ct2_sizes = multi_conn_df['sum syn area'][multi_conn_df['from celltype'] == ct_dict[ct2]]
        multi_ct3_sizes = multi_conn_df['sum syn area'][multi_conn_df['from celltype'] == ct_dict[ct3]]
        stats, p_value = ranksums(multi_ct2_sizes, multi_ct3_sizes)
        ranksum_results.loc['multi syn sum sizes stats', f'from {ct_dict[ct2]} vs from {ct_dict[ct3]}'] = stats
        ranksum_results.loc['multi syn sum sizess p-value', f'from {ct_dict[ct2]} vs from {ct_dict[ct3]}'] = p_value
        ranksum_results.to_csv(f'{f_name}/ranksums_results_{ct_dict[conn_ct]}_{ct_dict[ct3]}_{ct_dict[ct3]}.csv')
        # plot results
        # plot sum sizes for pairwise cells as histogram
        sns.histplot(x='sum syn area', data=multi_conn_df, hue='from celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True)
        plt.ylabel('number of cell-pairs')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Sum of synapse sizes incoming to {ct_dict[conn_ct]} multisyn connections')
        plt.savefig(f'{f_name}/multi_sum_sizes_incoming_hist.png')
        plt.savefig(f'{f_name}/multi_sum_sizes_incoming_hist.svg')
        plt.close()
        sns.histplot(x='sum syn area', data=multi_conn_df, hue='from celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent')
        plt.ylabel('% of cell-pairs')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Sum of synapse sizes outgoing from {ct_dict[conn_ct]} multisyn connections')
        plt.savefig(f'{f_name}/multi_sum_sizes_incoming_hist_perc.png')
        plt.savefig(f'{f_name}/multi_sum_sizes_incoming_hist_perc.svg')
        plt.close()
        sns.histplot(x='sum syn area', data=multi_conn_df, hue='from celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, log_scale=True)
        plt.ylabel('number of cell-pairs')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Sum of synapse sizes incoming from {ct_dict[conn_ct]} multisyn connections')
        plt.savefig(f'{f_name}/multi_sum_sizes_incoming_hist_log.png')
        plt.savefig(f'{f_name}/multi_sum_sizes_incoming_hist_log.svg')
        plt.close()
        sns.histplot(x='sum syn area', data=multi_conn_df, hue='from celltype', palette=ct_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent', log_scale=True)
        plt.ylabel('% of cell-pairs')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(f'Sum of synapse sizes incoming to {ct_dict[conn_ct]} multisyn connections')
        plt.savefig(f'{f_name}/multi_sum_sizes_incoming_hist_log_perc.png')
        plt.savefig(f'{f_name}/multi_sum_sizes_incoming_hist_log_perc.svg')
        plt.close()
        # plot again as barplot
        # plot number of synapses again as barplot
        # make bins for each number, seperated by ct2 and ct3
        ct2_multi_df = multi_conn_df[multi_conn_df['from celltype'] == ct_dict[ct2]]
        ct2_inds, ct2_bins = pd.factorize(np.sort(ct2_multi_df['number of synapses']))
        ct2_counts = np.bincount(ct2_inds)
        ct3_multi_df = multi_conn_df[multi_conn_df['from celltype'] == ct_dict[ct3]]
        ct3_inds, ct3_bins = pd.factorize(np.sort(ct3_multi_df['number of synapses']))
        ct3_counts = np.bincount(ct3_inds)
        num_ct2_bins = len(ct2_bins)
        len_bins = num_ct2_bins + len(ct3_bins)
        hist_df = pd.DataFrame(columns=['count of cell-pairs', 'percent of cell-pairs', 'bins', 'from celltype'],
                               index=range(len_bins))
        hist_df.loc[0: num_ct2_bins - 1, 'count of cell-pairs'] = ct2_counts
        hist_df.loc[0: num_ct2_bins - 1, 'percent of cell-pairs'] = 100 * ct2_counts / np.sum(ct2_counts)
        hist_df.loc[0: num_ct2_bins - 1, 'bins'] = ct2_bins
        hist_df.loc[0: num_ct2_bins - 1, 'from celltype'] = ct_dict[ct2]
        hist_df.loc[num_ct2_bins: len_bins - 1, 'count of cell-pairs'] = ct3_counts
        hist_df.loc[num_ct2_bins: len_bins - 1, 'percent of cell-pairs'] = 100 * ct3_counts / np.sum(ct3_counts)
        hist_df.loc[num_ct2_bins: len_bins - 1, 'bins'] = ct3_bins
        hist_df.loc[num_ct2_bins: len_bins - 1, 'from celltype'] = ct_dict[ct3]
        sns.barplot(data=hist_df, x='bins', y='count of cell-pairs', hue='from celltype', palette=ct_palette)
        plt.xlabel('number of synapses')
        plt.savefig(f'{f_name}/multi_bar_syn_num_incoming.svg')
        plt.savefig(f'{f_name}/multi_bar_syn_num_incoming.png')
        plt.close()
        sns.barplot(data=hist_df, x='bins', y='percent of cell-pairs', hue='from celltype', palette=ct_palette)
        plt.xlabel('number of synapses')
        plt.savefig(f'{f_name}/multi_bar_syn_num_incoming_perc.svg')
        plt.savefig(f'{f_name}/multi_bar_syn_num_incoming_perc.png')
        plt.close()
        # plot also only up to numbers of 10, 15, 20
        for num in [10, 15, 20]:
            hist_df10 = hist_df[hist_df['bins'] <= num]
            multi_conn_df11 = multi_conn_df[multi_conn_df['number of synapses'] > num]
            hist_df11_ct2 = multi_conn_df11[multi_conn_df11['from celltype'] == ct_dict[ct2]]
            hist_df11_ct3 = multi_conn_df11[multi_conn_df11['from celltype'] == ct_dict[ct3]]
            log.info(
                f'{len(multi_conn_df11)} cell-pairs out of {len(multi_conn_df)} make incoming multisynaptic connections with {num + 1} synapses or more, \n'
                f'{len(hist_df11_ct2)} to {ct_dict[ct2]}, {len(hist_df11_ct3)} to {ct_dict[ct3]}')
            sns.barplot(data=hist_df10, x='bins', y='count of cell-pairs', hue='from celltype', palette=ct_palette)
            plt.xlabel('number of synapses')
            plt.savefig(f'{f_name}/cutoff{num}_multi_bar_syn_num_incoming.svg')
            plt.savefig(f'{f_name}/cutoff{num}_multi_bar_syn_num_incoming.png')
            plt.close()
            sns.barplot(data=hist_df10, x='bins', y='percent of cell-pairs', hue='from celltype', palette=ct_palette)
            plt.xlabel('number of synapses')
            plt.savefig(f'{f_name}/cutoff{num}_multi_bar_syn_num_incoming_perc.svg')
            plt.savefig(f'{f_name}/cutoff{num}_multi_bar_syn_num_incoming_perc.png')
            plt.close()

    log.info('Analysis finished')