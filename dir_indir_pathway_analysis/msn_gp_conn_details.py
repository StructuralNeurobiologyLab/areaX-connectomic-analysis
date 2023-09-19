#get information about msn synapses to GPe and GPi
#more details related to how many GP cells are targeted, how many monosynaptic connections etc.

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct, get_multi_syn_info_per_cell
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    from syconn.reps.segmentation import SegmentationDataset
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
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

    ssd = SuperSegmentationDataset(working_dir=global_params.wd)
    version = 'v5'
    analysis_params = Analysis_Params(working_dir=global_params.wd, version=version)
    ct_dict = analysis_params.ct_dict(with_glia=False)
    min_comp_len = 200
    syn_prob = 0.6
    min_syn_size = 0.1
    msn_ct = 2
    gpe_ct = 6
    gpi_ct = 7
    fontsize_jointplot = 12
    kde = True
    f_name = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/230919_j0251v5_MSN_GP_syn_multisyn_mcl_%i_synprob_%.2f_kde%i" % (
        min_comp_len, syn_prob, kde)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('MSN GP connectivity details', log_dir=f_name + '/logs/')
    log.info("Analysis of MSN connectivity to GP starts")
    
    log.info('Step 1/5: Load and check all cells')
    known_mergers = analysis_params.load_known_mergers()
    MSN_dict = analysis_params.load_cell_dict(celltype=msn_ct)
    MSN_ids = np.array(list(MSN_dict.keys()))
    merger_inds = np.in1d(MSN_ids, known_mergers) == False
    MSN_ids = MSN_ids[merger_inds]
    misclassified_asto_ids = analysis_params.load_potential_astros()
    astro_inds = np.in1d(MSN_ids, misclassified_asto_ids) == False
    MSN_ids = MSN_ids[astro_inds]
    MSN_ids = check_comp_lengths_ct(cellids=MSN_ids, fullcelldict=MSN_dict, min_comp_len=min_comp_len,
                                    axon_only=False,
                                    max_path_len=None)
    GPe_dict = analysis_params.load_cell_dict(gpe_ct)
    GPe_ids = np.array(list(GPe_dict.keys()))
    merger_inds = np.in1d(GPe_ids, known_mergers) == False
    GPe_ids = GPe_ids[merger_inds]
    GPe_ids = check_comp_lengths_ct(cellids=GPe_ids, fullcelldict=GPe_dict, min_comp_len=min_comp_len,
                                    axon_only=False,
                                    max_path_len=None)

    GPi_dict = analysis_params.load_cell_dict(gpi_ct)
    GPi_ids = np.array(list(GPi_dict.keys()))
    merger_inds = np.in1d(GPi_ids, known_mergers) == False
    GPi_ids = GPi_ids[merger_inds]
    GPi_ids = check_comp_lengths_ct(cellids=GPi_ids, fullcelldict=GPi_dict, min_comp_len=min_comp_len,
                                    axon_only=False,
                                    max_path_len=None)
    # load information about MSN groups
    f_name_saving1 = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/230831_j0251v5_MSN_GPratio_spine_density_mcl_%i_synprob_%.2f_kde%i_replot" % (
        min_comp_len, syn_prob, kde)
    log.info(f'Use morph parameters from {f_name_saving1}')
    msn_result_df = pd.read_csv(f'{f_name_saving1}/msn_spine_density_GPratio.csv', index_col=0)
    for key in msn_result_df.keys():
        if 'cellid' in key or 'celltype' in key:
            continue
        if 'GP' in key:
            continue
        msn_result_df = msn_result_df.drop(key, axis=1)
    msn_ids_table = msn_result_df['cellid']
    msn_id_check = np.in1d(msn_ids_table, MSN_ids)
    if np.any(msn_id_check == False):
        msn_result_df = msn_result_df[msn_id_check]
    log.info(f'{len(MSN_ids)} MSN cells, {len(GPe_ids)} GPes and {len(GPi_ids)} GPis suitable for analysis')
    
    log.info('Step 2/5: Get synapses from MSN to GPe and GPi')
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.wd)
    m_cts, m_axs, m_ssv_partners, m_sizes, m_rep_coord = filter_synapse_caches_for_ct(
        sd_synssv=sd_synssv,
        pre_cts=[msn_ct],
        post_cts=[gpe_ct, gpi_ct],
        syn_prob_thresh=syn_prob,
        min_syn_size=min_syn_size,
        axo_den_so=True,
        synapses_caches=None)
    all_suitable_ids = np.hstack([MSN_ids, GPe_ids, GPi_ids])
    suit_ct_inds = np.all(np.in1d(m_ssv_partners, all_suitable_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[suit_ct_inds]
    m_ssv_partners = m_ssv_partners[suit_ct_inds]
    m_sizes = m_sizes[suit_ct_inds]
    m_axs = m_axs[suit_ct_inds]
    m_rep_coord = m_rep_coord[suit_ct_inds]
    log.info('Get GPe parameter information per MSN cell')
    # get number of GP partner cells per cell and information about syn size for each of them
    #get msn cells connected to gpe
    msn_gpe_ids = msn_result_df['cellid'][msn_result_df['syn number to GPe'] > 0]
    gpe_input = [[cellid, m_ssv_partners, m_sizes, m_rep_coord, m_cts, m_axs, gpe_ct] for cellid in msn_gpe_ids]
    gpe_output = start_multiprocess_imap(get_multi_syn_info_per_cell, gpe_input)
    gpe_output = np.array(gpe_output, dtype=object)
    number_target_gpe = gpe_output[:, 0]
    perc_single_conn_gpe = gpe_output[:, 1]
    perc_single_syns_gpe = gpe_output[:, 2]
    perc_single_syn_area_gpe = gpe_output[:, 3]
    gpe_output_dict = dict(ChainMap(*gpe_output[:, 4]))
    write_obj2pkl(f'{f_name}/msn_gpe_indiv_conns_dict.pkl', gpe_output_dict)

    log.info('Get GPi parameter information per MSN cell')
    msn_gpi_ids = msn_result_df['cellid'][msn_result_df['syn number to GPi'] > 0]
    gpi_input = [[cellid, m_ssv_partners, m_sizes, m_rep_coord, m_cts, m_axs, gpi_ct] for cellid in msn_gpi_ids]
    gpi_output = start_multiprocess_imap(get_multi_syn_info_per_cell, gpi_input)
    gpi_output = np.array(gpi_output, dtype=object)
    number_target_gpi = gpi_output[:, 0]
    perc_single_conn_gpi = gpi_output[:, 1]
    perc_single_syns_gpi = gpi_output[:, 2]
    perc_single_syn_area_gpi = gpi_output[:, 3]
    gpi_output_dict = dict(ChainMap(*gpi_output[:, 4]))
    write_obj2pkl(f'{f_name}/msn_gpi_indiv_conns_dict.pkl', gpi_output_dict)

    log.info('Step 3/5: Sort and plot per cell information')
    # add per cell information to dictionary
    msn_gpe_inds = np.in1d(msn_result_df['cellid'], msn_gpe_ids)
    msn_result_df.loc[msn_gpe_inds, 'number of GPe cells'] = number_target_gpe
    msn_result_df.loc[msn_gpe_inds, 'percent GPe monosynaptic'] = perc_single_conn_gpe
    msn_result_df.loc[msn_gpe_inds, 'percent syns monosynaptic GPe'] = perc_single_syns_gpe
    msn_result_df.loc[msn_gpe_inds, 'percent syn area monosynaptic GPe'] = perc_single_syn_area_gpe
    # add per cell information to dictionary
    msn_gpi_inds = np.in1d(msn_result_df['cellid'], msn_gpi_ids)
    msn_result_df.loc[msn_gpi_inds, 'number of GPi cells'] = number_target_gpi
    msn_result_df.loc[msn_gpi_inds, 'percent GPi monosynaptic'] = perc_single_conn_gpi
    msn_result_df.loc[msn_gpi_inds, 'percent syns monosynaptic GPi'] = perc_single_syns_gpi
    msn_result_df.loc[msn_gpi_inds, 'percent syn area monosynaptic GPi'] = perc_single_syn_area_gpi
    gp_nonzero = np.any([msn_result_df['syn number to GPi'] > 0, msn_result_df['syn number to GPe'] > 0], axis = 0)
    msn_result_df.loc[gp_nonzero, 'number of GP cells'] = msn_result_df.loc[gp_nonzero, 'number of GPe cells'].add(
        msn_result_df.loc[gp_nonzero, 'number of GPi cells'], fill_value=0)

    key_list = list(msn_result_df.keys())[2:]
    type_dict = {key: float for key in key_list}
    msn_result_df = msn_result_df.astype(type_dict)
    msn_result_df.to_csv(f'{f_name}/msn_per_cell_GP_info.csv')
    #make statistics and plot results
    #also plot for different MSN groups
    msn_colors = ["#EAAE34", "black", "#707070", '#2F86A8']
    msn_groups_str = np.unique(msn_result_df['celltype'])
    msn_palette = {ct: msn_colors[i] for i, ct in enumerate(msn_groups_str)}
    for key in msn_result_df.keys():
        if 'cellid' in key or 'celltype' in key or 'GP ratio' in key:
            continue
        sns.histplot(x=key, data=msn_result_df, color='black', common_norm=False,
                     fill=False, element="step", linewidth=3)
        plt.ylabel('count of cells')
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_hist.png')
        plt.close()
        sns.histplot(x=key, data=msn_result_df, color='black', common_norm=False,
                     fill=False, element="step", linewidth=3, stat='percent')
        plt.ylabel('percent of cells')
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_hist_perc.png')
        plt.close()
        sns.histplot(x=key, data=msn_result_df, color='black', common_norm=False,
                     fill=False, element="step", linewidth=3)
        plt.ylabel('count of cells')
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_hist.png')
        plt.close()
        sns.histplot(x=key, data=msn_result_df, color='black', common_norm=False,
                     fill=False, element="step", linewidth=3, stat='percent')
        plt.ylabel('percent of cells')
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_hist_perc.png')
        plt.close()
        sns.boxplot(data=msn_result_df, x='celltype', y=key, palette=msn_palette)
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_overview_box.png')
        plt.savefig(f'{f_name}/{key}_overview_box.svg')
        plt.close()
        sns.stripplot(x='celltype', y=key, data=msn_result_df, color='black', alpha=0.2,
                      dodge=True, size=2)
        sns.violinplot(x='celltype', y=key, data=msn_result_df, inner="box",
                       palette=msn_palette)
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_overview_violin.png')
        plt.savefig(f'{f_name}/{key}_overview_violin.svg')
        plt.close()
        sns.histplot(x=key, data=msn_result_df, hue='celltype', palette=msn_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True)
        plt.ylabel('number of cells')
        plt.savefig(f'{f_name}/{key}_celltype_hist.png')
        plt.savefig(f'{f_name}/{key}_celltype_hist.svg')
        plt.close()
        sns.histplot(x=key, data=msn_result_df, hue='celltype', palette=msn_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent')
        plt.ylabel('% of cells')
        plt.savefig(f'{f_name}/{key}_celltype_hist_perc.png')
        plt.savefig(f'{f_name}/{key}_celltype_hist_perc.svg')
        plt.close()

    log.info('Step 4/5: Sort information per connection of MSN and GP')
    #create DataFrame for all connections between MSN and GP to see how many multisynaptic
    gpe_conn_syn_numbers = []
    gpe_conn_syn_sizes = []
    gpe_conn_msn_ids = []
    gpe_multisyn_pair_size_diff = []
    gpe_multisyn_pair_dist_msn = []
    gpe_multisyn_pair_dist_gp = []
    gpe_multisyn_pair_syn_number = []
    gpe_multisyn_pair_axo = []
    gpe_multisyn_pair_size_frac = []
    log.info('Get information from connections to GPe')
    for cellid in tqdm(gpe_output_dict.keys()):
        cell_info = gpe_output_dict[cellid]
        gpe_conn_syn_numbers.append(cell_info['connected cell syn numbers'])
        gpe_conn_syn_sizes.append(cell_info['connected cell sum syn sizes'])
        gpe_conn_msn_ids.append(np.zeros(len(cell_info['connected cell syn numbers'])) + cellid)
        try:
            gpe_multisyn_pair_size_diff.append(cell_info['multi conn pairwise size difference'])
            gpe_multisyn_pair_dist_msn.append(cell_info['multi conn pairwise dist cell'])
            gpe_multisyn_pair_dist_gp.append(cell_info['multi conn pairwise dist target cell'])
            gpe_multisyn_pair_syn_number.append(cell_info['multi conn pairwise number syns'])
            gpe_multisyn_pair_axo.append(cell_info['multi conn pairwise comp'])
            gpe_multisyn_pair_size_frac.append(cell_info['multi conn pairwise size diff frac'])
        except KeyError:
            continue
    gpe_conn_syn_numbers = np.concatenate(gpe_conn_syn_numbers)
    gpe_conn_syn_sizes = np.concatenate(gpe_conn_syn_sizes)
    gpe_conn_msn_ids = np.concatenate(gpe_conn_msn_ids).astype(int)
    gpe_multisyn_pair_syn_number = np.concatenate(gpe_multisyn_pair_syn_number)
    gpe_multisyn_pair_size_diff = np.concatenate(gpe_multisyn_pair_size_diff)
    gpe_multisyn_pair_dist_msn = np.concatenate(gpe_multisyn_pair_dist_msn)
    gpe_multisyn_pair_dist_gp = np.concatenate(gpe_multisyn_pair_dist_gp)
    gpe_multisyn_pair_axo = np.concatenate(gpe_multisyn_pair_axo)
    gpe_multisyn_pair_size_frac = np.concatenate(gpe_multisyn_pair_size_frac)
    log.info('Get information from connections to GPi')
    gpi_conn_syn_numbers = []
    gpi_conn_syn_sizes = []
    gpi_conn_msn_ids = []
    gpi_multisyn_pair_size_diff = []
    gpi_multisyn_pair_dist_msn = []
    gpi_multisyn_pair_dist_gp = []
    gpi_multisyn_pair_syn_number = []
    gpi_multisyn_pair_axo = []
    gpi_multisyn_pair_size_frac = []
    for cell_id in tqdm(gpi_output_dict.keys()):
        cell_info = gpi_output_dict[cell_id]
        gpi_conn_syn_numbers.append(cell_info['connected cell syn numbers'])
        gpi_conn_syn_sizes.append(cell_info['connected cell sum syn sizes'])
        gpi_conn_msn_ids.append(np.zeros(len(cell_info['connected cell syn numbers'])) + cell_id)
        try:
            gpi_multisyn_pair_size_diff.append(cell_info['multi conn pairwise size difference'])
            gpi_multisyn_pair_dist_msn.append(cell_info['multi conn pairwise dist cell'])
            gpi_multisyn_pair_dist_gp.append(cell_info['multi conn pairwise dist target cell'])
            gpi_multisyn_pair_syn_number.append(cell_info['multi conn pairwise number syns'])
            gpi_multisyn_pair_axo.append(cell_info['multi conn pairwise comp'])
            gpi_multisyn_pair_size_frac.append(cell_info['multi conn pairwise size diff frac'])
        except KeyError:
            continue
    gpi_conn_syn_numbers = np.concatenate(gpi_conn_syn_numbers)
    gpi_conn_syn_sizes = np.concatenate(gpi_conn_syn_sizes)
    gpi_conn_msn_ids = np.concatenate(gpi_conn_msn_ids).astype(int)
    gpi_multisyn_pair_syn_number = np.concatenate(gpi_multisyn_pair_syn_number)
    gpi_multisyn_pair_size_diff = np.concatenate(gpi_multisyn_pair_size_diff)
    gpi_multisyn_pair_dist_msn = np.concatenate(gpi_multisyn_pair_dist_msn)
    gpi_multisyn_pair_dist_gp = np.concatenate(gpi_multisyn_pair_dist_gp)
    gpi_multisyn_pair_axo = np.concatenate(gpi_multisyn_pair_axo)
    gpi_multisyn_pair_size_frac = np.concatenate(gpi_multisyn_pair_size_frac)
    log.info('Sort information into df for MSN-GP pairs')
    len_gpe_conn = len(gpe_conn_syn_numbers)
    len_gp_conn = len_gpe_conn + len(gpi_conn_syn_numbers)
    gp_conn_df = pd.DataFrame(columns = ['number of synapses', 'sum syn area', 'to celltype', 'cellid', 'MSN group'], index = range(len_gp_conn))
    gp_conn_df.loc[0: len_gpe_conn - 1, 'number of synapses'] = gpe_conn_syn_numbers
    gp_conn_df.loc[0: len_gpe_conn - 1, 'sum syn area'] = gpe_conn_syn_sizes
    gp_conn_df.loc[0: len_gpe_conn - 1, 'to celltype'] = 'to GPe'
    gp_conn_df.loc[0: len_gpe_conn - 1, 'cellid'] = gpe_conn_msn_ids
    gp_conn_df.loc[len_gpe_conn: len_gp_conn - 1, 'number of synapses'] = gpi_conn_syn_numbers
    gp_conn_df.loc[len_gpe_conn: len_gp_conn - 1, 'sum syn area'] = gpi_conn_syn_sizes
    gp_conn_df.loc[len_gpe_conn: len_gp_conn - 1, 'to celltype'] = 'to GPi'
    gp_conn_df.loc[len_gpe_conn: len_gp_conn - 1, 'cellid'] = gpi_conn_msn_ids
    #information about msn groups
    for msn_str in msn_groups_str:
        inds = np.in1d(gp_conn_df['cellid'], msn_result_df['cellid'][msn_result_df['celltype'] == msn_str])
        gp_conn_df.loc[inds, 'MSN group'] = msn_str
    gp_conn_df = gp_conn_df.astype({'number of synapses':int, 'sum syn area': float})
    gp_conn_df.to_csv(f'{f_name}/multi_syn_info_permsn_gp_conn.csv')
    # make df per multisynaptic connections
    log.info('Sort information for pairs of synapses in multisynaptic connections')
    len_gpe_multi = len(gpe_multisyn_pair_syn_number)
    gp_multi_len = len_gpe_multi + len(gpi_multisyn_pair_syn_number)
    multisyn_columns = ['size difference', 'distance on MSN', 'distance on GP', 'to celltype',
                        'number of syns per connection']
    multi_syn_df = pd.DataFrame(columns=multisyn_columns, index = range(gp_multi_len))
    multi_syn_df.loc[0: len_gpe_multi - 1, 'size difference'] = gpe_multisyn_pair_size_diff
    multi_syn_df.loc[0: len_gpe_multi - 1, 'distance on MSN'] = gpe_multisyn_pair_dist_msn
    multi_syn_df.loc[0: len_gpe_multi - 1, 'distance on GP'] = gpe_multisyn_pair_dist_gp
    multi_syn_df.loc[0: len_gpe_multi - 1, 'fraction size difference'] = gpe_multisyn_pair_size_frac
    multi_syn_df.loc[0: len_gpe_multi - 1, 'to celltype'] = 'to GPe'
    multi_syn_df.loc[0: len_gpe_multi - 1, 'compartment 1'] = gpe_multisyn_pair_axo[:, 0]
    multi_syn_df.loc[0: len_gpe_multi - 1, 'compartment 2'] = gpe_multisyn_pair_axo[:, 1]
    multi_syn_df.loc[0: len_gpe_multi - 1, 'number of syns per connection'] = gpe_multisyn_pair_syn_number
    multi_syn_df.loc[len_gpe_multi: gp_multi_len - 1, 'size difference'] = gpi_multisyn_pair_size_diff
    multi_syn_df.loc[len_gpe_multi: gp_multi_len - 1, 'distance on MSN'] = gpi_multisyn_pair_dist_msn
    multi_syn_df.loc[len_gpe_multi: gp_multi_len - 1, 'distance on GP'] = gpi_multisyn_pair_dist_gp
    multi_syn_df.loc[len_gpe_multi: gp_multi_len - 1, 'fraction size difference'] = gpi_multisyn_pair_size_frac
    multi_syn_df.loc[len_gpe_multi: gp_multi_len - 1, 'to celltype'] = 'to GPi'
    multi_syn_df.loc[len_gpe_multi: gp_multi_len - 1, 'number of syns per connection'] = gpi_multisyn_pair_syn_number
    multi_syn_df.loc[len_gpe_multi: gp_multi_len - 1, 'compartment 1'] = gpi_multisyn_pair_axo[:, 0]
    multi_syn_df.loc[len_gpe_multi: gp_multi_len - 1, 'compartment 2'] = gpi_multisyn_pair_axo[:, 1]
    axo_sums = np.hstack([np.sum(gpe_multisyn_pair_axo, axis = 1), np.sum(gpi_multisyn_pair_axo, axis = 1)])
    diff_inds = axo_sums == 2
    multi_syn_df.loc[diff_inds, 'compartment'] = 'soma and dendrite'
    multi_syn_df.loc[diff_inds == False, 'compartment'] = 'same compartment'
    multi_syn_df = multi_syn_df.astype({'size difference': float, 'distance on MSN':float,
                                          'distance on GP': float, 'number of syns per connection': int})
    multi_syn_df['distance on MSN'] = multi_syn_df['distance on MSN'].replace({0: np.nan})
    multi_syn_df['distance on GP'] = multi_syn_df['distance on GP'].replace({0: np.nan})
    multi_syn_df.to_csv(f'{f_name}/multi_syn_connection_pairs.csv')

    log.info('Step 5/5: Make plots per MSN and GP connection and per synapse pair in multisynapse connections')
    #make statistics and plot results
    gp_palette = {'to GPe': '#592A87', 'to GPi': '#2AC644'}
    gpe_conn_df = gp_conn_df[gp_conn_df['to celltype'] == 'to GPe']
    gpi_conn_df = gp_conn_df[gp_conn_df['to celltype'] == 'to GPi']
    for key in gp_conn_df.keys():
        if 'celltype' in key or 'cellid' in key or 'MSN group' in key:
            continue
        if 'sum syn' in key:
            xhist = f'{key} [µm²]'
        else:
            xhist = key
        sns.histplot(x=key, data=gp_conn_df, color='black', common_norm=False,
                     fill=False, element="step", linewidth=3)
        plt.ylabel('count of cell-pairs')
        plt.xlabel(xhist)
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_hist_conns.png')
        plt.close()
        sns.histplot(x=key, data=gp_conn_df, color='black', common_norm=False,
                     fill=False, element="step", linewidth=3, log_scale=True)
        plt.ylabel('count of cell-pairs')
        plt.xlabel(xhist)
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_hist_log_conns.png')
        plt.close()
        sns.histplot(x=key, data=gp_conn_df, color='black', common_norm=False,
                     fill=False, element="step", linewidth=3, stat='percent')
        plt.ylabel('number of cell-pairs')
        plt.xlabel(xhist)
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_hist_perc_conns.png')
        plt.close()
        sns.histplot(x=key, data=gp_conn_df, hue='to celltype', palette=gp_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True)
        plt.ylabel('number of cell-pairs')
        plt.savefig(f'{f_name}/{key}_gptype_hist.png')
        plt.savefig(f'{f_name}/{key}_gptype_hist.svg')
        plt.close()
        sns.histplot(x=key, data=gp_conn_df, hue='to celltype', palette=gp_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent')
        plt.ylabel('% of cell-pairs')
        plt.savefig(f'{f_name}/{key}_gptype_hist_perc.png')
        plt.savefig(f'{f_name}/{key}_gptype_hist_perc.svg')
        plt.close()
        sns.histplot(x=key, data=gp_conn_df, hue='to celltype', palette=gp_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent', log_scale=True)
        plt.ylabel('% of cell-pairs')
        plt.savefig(f'{f_name}/{key}_gptype_hist_perc_log.png')
        plt.savefig(f'{f_name}/{key}_gptype_hist_per_log.svg')
        plt.close()
        sns.histplot(x=key, data=gpe_conn_df, hue='MSN group', palette=msn_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent')
        plt.ylabel('% of cell-pairs')
        plt.title(f'{key} to GPe')
        plt.savefig(f'{f_name}/{key}_gpe_msn_groups_hist_perc.png')
        plt.savefig(f'{f_name}/{key}_gpe_msn_groups_hist_perc.svg')
        plt.close()
        sns.histplot(x=key, data=gpe_conn_df, hue='MSN group', palette=msn_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent', log_scale=True)
        plt.ylabel('% of cell-pairs')
        plt.title(f'{key} to GPe')
        plt.savefig(f'{f_name}/{key}_gpe_msn_group_hist_perc_log.png')
        plt.savefig(f'{f_name}/{key}_gpe_msn_group_hist_per_log.svg')
        plt.close()
        sns.histplot(x=key, data=gpi_conn_df, hue='MSN group', palette=msn_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent')
        plt.ylabel('% of cell-pairs')
        plt.title(f'{key} to GPi')
        plt.savefig(f'{f_name}/{key}_gpi_msn_group_hist_perc.png')
        plt.savefig(f'{f_name}/{key}_gpi_msn_group_perc.svg')
        plt.close()
        sns.histplot(x=key, data=gpi_conn_df, hue='MSN group', palette=msn_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent', log_scale=True)
        plt.ylabel('% of cell-pairs')
        plt.title(f'{key} to GPi')
        plt.savefig(f'{f_name}/{key}_gpi_msn_group_hist_perc_log.png')
        plt.savefig(f'{f_name}/{key}_gpi_msn_group_per_log.svg')
        plt.close()

    #plot number of synapses again as barplot
    #make bins for each number, seperated by GPe and GPi
    gpe_inds, gpe_bins = pd.factorize(np.sort(gpe_conn_df['number of synapses']))
    gpe_counts = np.bincount(gpe_inds)
    gpi_inds, gpi_bins = pd.factorize(np.sort(gpi_conn_df['number of synapses']))
    gpi_counts = np.bincount(gpi_inds)
    len_bins = len(gpe_bins) + len(gpi_bins)
    hist_df = pd.DataFrame(columns = ['count of cell-pairs', 'percent of cell-pairs', 'bins', 'to celltype'], index = range(len_bins))
    hist_df.loc[0: len(gpe_bins) - 1, 'count of cell-pairs'] = gpe_counts
    hist_df.loc[0: len(gpe_bins) - 1, 'percent of cell-pairs'] = 100 * gpe_counts / np.sum(gpe_counts)
    hist_df.loc[0: len(gpe_bins) - 1, 'bins'] = gpe_bins
    hist_df.loc[0: len(gpe_bins) - 1, 'to celltype'] = 'GPe'
    hist_df.loc[len(gpe_bins): len_bins - 1, 'count of cell-pairs'] = gpi_counts
    hist_df.loc[len(gpe_bins): len_bins - 1, 'percent of cell-pairs'] = 100 * gpi_counts / np.sum(gpi_counts)
    hist_df.loc[len(gpe_bins): len_bins - 1, 'bins'] = gpe_bins
    hist_df.loc[len(gpe_bins): len_bins - 1, 'to celltype'] = 'GPi'
    sns.barplot(data = hist_df, x = 'bins', y = 'count of cell-pairs', hue = 'to celltype', palette=gp_palette)
    plt.xlabel('number of synapses')
    plt.savefig(f'{f_name}/bar_syn_number_gptype_hist.svg')
    plt.savefig(f'{f_name}/bar_syn_number_gptype_hist.png')
    plt.close()
    sns.barplot(data=hist_df, x='bins', y='percent of cell-pairs', hue='to celltype', palette=gp_palette)
    plt.xlabel('number of synapses')
    plt.savefig(f'{f_name}/bar_syn_number_gptype_hist_perc.svg')
    plt.savefig(f'{f_name}/bar_syn_number_gptype_hist_perc.png')
    plt.close()
    hist_df.to_csv(f'{f_name}/GP_num_syn_cellpairs_hist.csv')
    # then again seperate for GPe and MSN groups, GPi and MSN groups
    gpe_only_df = gpe_conn_df[gpe_conn_df['MSN group'] == 'MSN only GPe']
    gpe_only_inds, gpe_only_bins = pd.factorize(np.sort(gpe_only_df['number of synapses']))
    gpe_msnonlygpe_counts = np.bincount(gpe_only_inds)
    gpe_both_df = gpe_conn_df[gpe_conn_df['MSN group'] == 'MSN both GPs']
    gpe_both_inds, gpe_both_bins = pd.factorize(np.sort(gpe_both_df['number of synapses']))
    gpe_both_counts = np.bincount(gpe_both_inds)
    len_bins = len(gpe_only_bins) + len(gpe_both_bins)
    hist_df = pd.DataFrame(columns=['count of cell-pairs', 'percent of cell-pairs', 'bins', 'MSN group'],
                           index=range(len_bins))
    hist_df.loc[0: len(gpe_only_bins) - 1, 'count of cell-pairs'] = gpe_msnonlygpe_counts
    hist_df.loc[0: len(gpe_only_bins) - 1, 'percent of cell-pairs'] = 100 * gpe_msnonlygpe_counts / np.sum(gpe_msnonlygpe_counts)
    hist_df.loc[0: len(gpe_only_bins) - 1, 'bins'] = gpe_only_bins
    hist_df.loc[0: len(gpe_only_bins) - 1, 'MSN group'] = 'MSN only GPe'
    hist_df.loc[len(gpe_only_bins): len_bins - 1, 'count of cell-pairs'] = gpe_both_counts
    hist_df.loc[len(gpe_only_bins): len_bins - 1, 'percent of cell-pairs'] = 100 * gpe_both_counts / np.sum(gpe_both_counts)
    hist_df.loc[len(gpe_only_bins): len_bins - 1, 'bins'] = gpe_both_bins
    hist_df.loc[len(gpe_only_bins): len_bins - 1, 'MSN group'] = 'MSN both GPs'
    sns.barplot(data=hist_df, x='bins', y='count of cell-pairs', hue='MSN group', palette=msn_palette)
    plt.xlabel('number of synapses')
    plt.savefig(f'{f_name}/bar_syn_number_gpe_msns_hist.svg')
    plt.savefig(f'{f_name}/bar_syn_number_gpe_msns_hist.png')
    plt.close()
    sns.barplot(data=hist_df, x='bins', y='percent of cell-pairs', hue='MSN group', palette=msn_palette)
    plt.xlabel('number of synapses')
    plt.savefig(f'{f_name}/bar_syn_number_gpe_msns_hist_perc.svg')
    plt.savefig(f'{f_name}/bar_syn_number_gpe_msns_hist_perc.png')
    plt.close()
    hist_df.to_csv(f'{f_name}/GPe_num_syn_cellpairs_msn_groups_hist.csv')
    gpi_only_df = gpi_conn_df[gpi_conn_df['MSN group'] == 'MSN only GPi']
    gpi_only_inds, gpi_only_bins = pd.factorize(np.sort(gpi_only_df['number of synapses']))
    gpi_msnonlygpi_counts = np.bincount(gpi_only_inds)
    gpi_both_df = gpi_conn_df[gpi_conn_df['MSN group'] == 'MSN both GPs']
    gpi_both_inds, gpi_both_bins = pd.factorize(np.sort(gpi_both_df['number of synapses']))
    gpi_both_counts = np.bincount(gpi_both_inds)
    len_bins = len(gpi_only_bins) + len(gpi_both_bins)
    hist_df = pd.DataFrame(columns=['count of cell-pairs', 'percent of cell-pairs', 'bins', 'MSN group'],
                           index=range(len_bins))
    hist_df.loc[0: len(gpi_only_bins) - 1, 'count of cell-pairs'] = gpi_msnonlygpi_counts
    hist_df.loc[0: len(gpi_only_bins) - 1, 'percent of cell-pairs'] = 100 * gpi_msnonlygpi_counts / np.sum(
        gpi_msnonlygpi_counts)
    hist_df.loc[0: len(gpi_only_bins) - 1, 'bins'] = gpi_only_bins
    hist_df.loc[0: len(gpi_only_bins) - 1, 'MSN group'] = 'MSN only GPi'
    hist_df.loc[len(gpi_only_bins): len_bins - 1, 'count of cell-pairs'] = gpi_both_counts
    hist_df.loc[len(gpi_only_bins): len_bins - 1,
    'percent of cell-pairs'] = 100 * gpi_both_counts / np.sum(gpi_both_counts)
    hist_df.loc[len(gpi_only_bins): len_bins - 1, 'bins'] = gpi_both_bins
    hist_df.loc[len(gpi_only_bins): len_bins - 1, 'MSN group'] = 'MSN both GPs'
    sns.barplot(data=hist_df, x='bins', y='count of cell-pairs', hue='MSN group', palette=msn_palette)
    plt.xlabel('number of synapses')
    plt.savefig(f'{f_name}/bar_syn_number_gpi_msns_hist.svg')
    plt.savefig(f'{f_name}/bar_syn_number_gpi_msns_hist.png')
    plt.close()
    sns.barplot(data=hist_df, x='bins', y='percent of cell-pairs', hue='MSN group', palette=msn_palette)
    plt.xlabel('number of synapses')
    plt.savefig(f'{f_name}/bar_syn_number_gpi_msns_hist_perc.svg')
    plt.savefig(f'{f_name}/bar_syn_number_gpi_msns_hist_perc.png')
    plt.close()
    hist_df.to_csv(f'{f_name}/GPi_num_syn_cellpairs_msn_groups_hist.csv')


    #make plots for multisynaptic connections dependend on celltype and compartments
    size_key = ['size difference', 'fraction size difference']
    for key in multi_syn_df.keys():
        if 'compartment' in key or 'celltype' in key:
            continue
        if 'size' in key and not 'fraction' in key:
            xhist = f'{key} [µm²]'
        elif 'distance' in key:
            xhist = f'{key} [µm]'
        else:
            xhist = key
        #plot each parameter individually
        sns.histplot(x=key, data=multi_syn_df, color='black', common_norm=False,
                     fill=False, element="step", linewidth=3)
        plt.ylabel('count of synapse pairs')
        plt.xlabel(xhist)
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_hist_multi.png')
        plt.close()
        sns.histplot(x=key, data=multi_syn_df, color='black', common_norm=False,
                         fill=False, element="step", linewidth=3, log_scale=True)
        plt.ylabel('synapse pairs')
        plt.xlabel(xhist)
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_hist_log_multi.png')
        plt.close()
        sns.histplot(x=key, data=multi_syn_df, color='black', common_norm=False,
                     fill=False, element="step", linewidth=3, stat='percent')
        plt.ylabel('synapse pairs')
        plt.xlabel(xhist)
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_hist_perc_multi.png')
        plt.close()
        #plot each parameter with differences for MSN-GPe and MSN-GPi synapses
        sns.histplot(x=key, data=multi_syn_df, hue='to celltype', palette=gp_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True)
        plt.ylabel('synapse pairs')
        plt.savefig(f'{f_name}/{key}_gptype_hist_multi.png')
        plt.savefig(f'{f_name}/{key}_gptype_hist_multi.svg')
        plt.close()
        sns.histplot(x=key, data=multi_syn_df, hue='to celltype', palette=gp_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent')
        plt.ylabel('% of synapse pairs')
        plt.savefig(f'{f_name}/{key}_gptype_hist_perc_multi.png')
        plt.savefig(f'{f_name}/{key}_gptype_hist_perc_multi.svg')
        plt.close()
        sns.histplot(x=key, data=multi_syn_df, hue='to celltype', palette=gp_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent', log_scale=True)
        plt.ylabel('% of synapse pairs')
        plt.savefig(f'{f_name}/{key}_gptype_hist_perc_log_multi.png')
        plt.savefig(f'{f_name}/{key}_gptype_hist_perc_log_multi.svg')
        plt.close()
        #plot each parameter for differences in target cell and compartment differences
        sns.boxplot(data=multi_syn_df, x='to celltype', y=key, hue = 'compartment')
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_gptype_comps_box.png')
        plt.savefig(f'{f_name}/{key}_gptype_comps_box.svg')
        plt.close()
        sns.stripplot(x='to celltype', y=key, data=multi_syn_df, hue = 'compartment', palette='dark:black', alpha=0.2,
                      dodge=True, size=2)
        sns.violinplot(x='to celltype', y=key, data=multi_syn_df, inner="box", hue = 'compartment')
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_gptype_comps_violin.png')
        plt.savefig(f'{f_name}/{key}_gptype_comps_violin.svg')
        plt.close()
        if 'size' in key:
            continue
        for skey in size_key:
            if 'fraction' in skey:
                ylabel = skey
            else:
                ylabel = f'{skey} [µm²]'
            g = sns.JointGrid(data=multi_syn_df, x=key, y=skey)
            g.plot_joint(sns.kdeplot, color="#EAAE34")
            g.plot_joint(sns.scatterplot, color='black', alpha=0.3)
            g.plot_marginals(sns.histplot, fill=False,
                             kde=False, bins='auto', color='black')
            g.ax_joint.set_xticks(g.ax_joint.get_xticks())
            g.ax_joint.set_yticks(g.ax_joint.get_yticks())
            g.ax_joint.set_xticklabels(["%.2f" % i for i in g.ax_joint.get_xticks()], fontsize=fontsize_jointplot)
            g.ax_joint.set_yticklabels(["%.2f" % i for i in g.ax_joint.get_yticks()], fontsize=fontsize_jointplot)
            g.ax_joint.set_xlabel(xhist)
            g.ax_joint.set_ylabel(ylabel)
            plt.savefig(f'{f_name}/{key}_{skey}_multisyn.png')
            plt.savefig(f'{f_name}/{key}_{skey}_multisyn.svg')
            plt.close()
            g = sns.JointGrid(data=multi_syn_df, x=key, y=skey, hue='to celltype')
            g.plot_joint(sns.scatterplot, palette = gp_palette, alpha=0.3)
            g.plot_marginals(sns.histplot, fill=False,
                             kde=False, bins='auto', palette = gp_palette)
            g.ax_joint.set_xticks(g.ax_joint.get_xticks())
            g.ax_joint.set_yticks(g.ax_joint.get_yticks())
            g.ax_joint.set_xticklabels(["%.2f" % i for i in g.ax_joint.get_xticks()], fontsize=fontsize_jointplot)
            g.ax_joint.set_yticklabels(["%.2f" % i for i in g.ax_joint.get_yticks()], fontsize=fontsize_jointplot)
            g.ax_joint.set_xlabel(xhist)
            g.ax_joint.set_ylabel(ylabel)
            plt.savefig(f'{f_name}/{key}_{skey}_multisyn_gptype.png')
            plt.savefig(f'{f_name}/{key}_{skey}_multisyn_gptype.svg')
            plt.close()

    log.info('Analysis finished')