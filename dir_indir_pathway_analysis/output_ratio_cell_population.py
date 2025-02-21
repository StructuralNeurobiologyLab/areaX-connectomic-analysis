#same as input_ratio_cell_population but for output

if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    import os as os
    import pandas as pd
    import numpy as np
    from syconn.reps.segmentation import SegmentationDataset
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct, \
        get_ct_syn_number_sumsize
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_1samp
    import seaborn as sns
    from tqdm import tqdm

    #global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"

    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    ct_dict = analysis_params.ct_dict(with_glia=False)
    global_params.wd = analysis_params.working_dir()

    min_comp_len_cell = 200
    min_comp_len_ax = 50
    syn_prob = 0.6
    min_syn_size = 0.1
    #this celltype can not be out of axon ct
    celltype_population = 9
    output_ct_1 = 7
    output_ct_2 = 6
    fontsize = 20
    # spiness values: 0 = spine neck, 1 = spine head, 2 = dendritic shaft, 3 = other
    spiness = None
    spiness_dict = analysis_params._spiness_dict
    color_key = 'STNGPINTv6'
    axon_cts = analysis_params.axon_cts()
    if output_ct_1 in axon_cts or output_ct_2 in axon_cts:
        raise ValueError(f'Celltype receiving inputs can not be from projecting axons. Current celltypes are {output_ct_1}, {output_ct_2}')
    ctp_str = ct_dict[celltype_population]
    ct_o1_str = ct_dict[output_ct_1]
    ct_o2_str = ct_dict[output_ct_2]
    if spiness is None:
        f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/250221_j0251{version}_{ctp_str}_{ct_o1_str}_{ct_o2_str}_output_ratio_fs{fontsize}"
    else:
        f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/250221_j0251{version}_{ctp_str}_{ct_o1_str}_{ct_o2_str}_output_ratio_fs{fontsize}_sp_{spiness_dict[spiness]}"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'{ctp_str}_ioutput_ratio_logs', log_dir=f_name)
    log.info(f' min comp len cell = {min_comp_len_cell},min comp len ax = {min_comp_len_ax}, cell type for population = {ctp_str}, input celltypes are: {ct_o1_str}, {ct_o2_str}.')
    log.info(f'synapse probability = {syn_prob}, min syn size = {min_syn_size} µm²')
    if spiness is not None:
        log.info(f'Only synapses onto {spiness_dict[spiness]} are considered.')
    ct_colors = CelltypeColors(ct_dict=ct_dict)
    ct_palette = ct_colors.ct_palette(key=color_key)
    ct_color = ct_palette[ctp_str]

    log.info('Step 1/4: Get suitable cell ids')
    known_mergers = analysis_params.load_known_mergers()
    ctp_dict = analysis_params.load_cell_dict(celltype=celltype_population)
    ctp_ids = np.array(list(ctp_dict.keys()))
    merger_inds = np.in1d(ctp_ids, known_mergers) == False
    ctp_ids = ctp_ids[merger_inds]
    if celltype_population in axon_cts:
        ctp_ids = check_comp_lengths_ct(cellids=ctp_ids, fullcelldict=ctp_dict, min_comp_len=min_comp_len_ax,
                                        axon_only=True,
                                        max_path_len=None)
        log.info(f'{len(ctp_ids)} suitable axons of cell type {ctp_str}.')
    else:
        ctp_ids = check_comp_lengths_ct(cellids=ctp_ids, fullcelldict=ctp_dict, min_comp_len=min_comp_len_cell,
                                        axon_only=False,
                                        max_path_len=None)
        log.info(f'{len(ctp_ids)} suitable cells of cell type {ctp_str}.')
    ctp_ids = np.sort(ctp_ids)

    out1_dict = analysis_params.load_cell_dict(output_ct_1)
    out1_ids = np.array(list(out1_dict.keys()))
    merger_inds = np.in1d(out1_ids, known_mergers) == False
    out1_ids = out1_ids[merger_inds]

    out1_ids = check_comp_lengths_ct(cellids=out1_ids, fullcelldict=out1_dict, min_comp_len=min_comp_len_cell,
                                    axon_only=False,
                                    max_path_len=None)
    log.info(f'{len(out1_ids)} suitable cells of cell type {ct_o1_str}.')

    out2_dict = analysis_params.load_cell_dict(output_ct_2)
    out2_ids = np.array(list(out2_dict.keys()))
    merger_inds = np.in1d(out2_ids, known_mergers) == False
    out2_ids = out2_ids[merger_inds]
    out2_ids = check_comp_lengths_ct(cellids=out2_ids, fullcelldict=out2_dict, min_comp_len=min_comp_len_cell,
                                    axon_only=False,
                                    max_path_len=None)
    log.info(f'{len(out2_ids)} suitable cells of cell type {ct_o2_str}.')

    log.info('Step 2/4: Filter synapses')
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.wd)
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(
        sd_synssv=sd_synssv,
        pre_cts=[celltype_population],
        post_cts=[output_ct_1, output_ct_2],
        syn_prob_thresh=syn_prob,
        min_syn_size=min_syn_size,
        axo_den_so=True,
        synapses_caches=None)
    all_suitable_ids = np.hstack([ctp_ids, out1_ids, out2_ids])
    suit_ct_inds = np.all(np.in1d(m_ssv_partners, all_suitable_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[suit_ct_inds]
    m_ssv_partners = m_ssv_partners[suit_ct_inds]
    m_sizes = m_sizes[suit_ct_inds]
    m_axs = m_axs[suit_ct_inds]
    m_rep_coord = m_rep_coord[suit_ct_inds]
    m_spiness = m_spiness[suit_ct_inds]
    if spiness is not None:
        testct = np.in1d(m_cts, [output_ct_1, output_ct_2]).reshape(len(m_cts), 2)
        testspiness = np.in1d(m_spiness, spiness).reshape(len(m_cts), 2)
        spine_inds = np.all(testct == testspiness, axis=1)
        m_sizes = m_sizes[spine_inds]
        m_cts = m_cts[spine_inds]
        m_ssv_partners = m_ssv_partners[spine_inds]
    log.info(f'{len(m_sizes)} synapses found from {ctp_str} to {ct_o1_str} and {ct_o2_str}. The summed synaptic area = {np.sum(m_sizes)} µm²; mean per {ctp_str} cell: {np.sum(m_sizes)/ len(ctp_ids)}')

    log.info(f'Step 3/4: Calculate ratio of inputs per {ctp_str} cell')
    ctp_columns = ['cellid', f'syn number {ct_o1_str}', f'syn area {ct_o1_str}', f'syn number {ct_o2_str}',
                   f'syn area {ct_o2_str}', 'ratio syn numbers', 'ratio syn area']
    num_ctp_ids = len(ctp_ids)
    ctp_result_per_cell = pd.DataFrame(columns = ctp_columns, index=range(num_ctp_ids))
    ctp_result_per_cell['cellid'] = ctp_ids

    #first for input 1, code adapted from syn_conn_details
    ct1_inds = np.where(m_cts == output_ct_1)
    ct1_ssv_partners = m_ssv_partners[ct1_inds[0]]
    ct1_sizes = m_sizes[ct1_inds[0]]
    ct1_cts = m_cts[ct1_inds[0]]
    # get cellids of ct1
    ct1_syn_numbers, ct1_sum_sizes, unique_ssv_ids = get_ct_syn_number_sumsize(syn_sizes=ct1_sizes,
                                                                               syn_ssv_partners=ct1_ssv_partners,
                                                                               syn_cts=ct1_cts, ct=celltype_population)
    sort_inds_ct1 = np.argsort(unique_ssv_ids)
    unique_ssv_ids_sorted = unique_ssv_ids[sort_inds_ct1]
    ct1_syn_numbers_sorted = ct1_syn_numbers[sort_inds_ct1]
    ct1_sum_sizes_sorted = ct1_sum_sizes[sort_inds_ct1]
    sort_inds_ct1 = np.in1d(ctp_result_per_cell['cellid'], unique_ssv_ids_sorted)
    ctp_result_per_cell.loc[sort_inds_ct1, f'syn number {ct_o1_str}'] = ct1_syn_numbers_sorted
    ctp_result_per_cell.loc[sort_inds_ct1, f'syn area {ct_o1_str}'] = ct1_sum_sizes_sorted
    log.info(
        f'{len(unique_ssv_ids_sorted)} out of {num_ctp_ids} {ctp_str} gmake synapses to {ct_o1_str}, the summed synapse area = {np.sum(ct1_sum_sizes_sorted):.2f} µm²')
    #for input2
    ct2_inds = np.where(m_cts == output_ct_2)
    ct2_ssv_partners = m_ssv_partners[ct2_inds[0]]
    ct2_sizes = m_sizes[ct2_inds[0]]
    ct2_cts = m_cts[ct2_inds[0]]
    # get cellids of ct1
    ct2_syn_numbers, ct2_sum_sizes, unique_ssv_ids = get_ct_syn_number_sumsize(syn_sizes=ct2_sizes,
                                                                               syn_ssv_partners=ct2_ssv_partners,
                                                                               syn_cts=ct2_cts, ct=celltype_population)
    sort_inds_ct2 = np.argsort(unique_ssv_ids)
    unique_ssv_ids_sorted = unique_ssv_ids[sort_inds_ct2]
    ct2_syn_numbers_sorted = ct2_syn_numbers[sort_inds_ct2]
    ct2_sum_sizes_sorted = ct2_sum_sizes[sort_inds_ct2]
    sort_inds_ct2 = np.in1d(ctp_result_per_cell['cellid'], unique_ssv_ids_sorted)
    ctp_result_per_cell.loc[sort_inds_ct2, f'syn number {ct_o2_str}'] = ct2_syn_numbers_sorted
    ctp_result_per_cell.loc[sort_inds_ct2, f'syn area {ct_o2_str}'] = ct2_sum_sizes_sorted
    log.info(
        f'{len(unique_ssv_ids_sorted)} out of {num_ctp_ids} {ctp_str} make synapses to {ct_o2_str}, the summed synapse area = {np.sum(ct2_sum_sizes_sorted):.2f} µm²')
    #not get syn number and syn area ratio
    #fill nan values with 0
    ctp_result_per_cell = ctp_result_per_cell.fillna(0)
    ctp_result_per_cell['ratio syn numbers'] = np.array(ctp_result_per_cell[f'syn number {ct_o1_str}']) / (ctp_result_per_cell[f'syn number {ct_o1_str}'] + ctp_result_per_cell[f'syn number {ct_o2_str}'])
    ctp_result_per_cell['ratio syn area'] = np.array(ctp_result_per_cell[f'syn area {ct_o1_str}']) / (
                ctp_result_per_cell[f'syn area {ct_o1_str}'] + ctp_result_per_cell[f'syn area {ct_o2_str}'])
    ctp_result_per_cell.to_csv(f'{f_name}/{ctp_str}_{ct_o1_str}_{ct_o2_str}_result_per_cell.csv')
    log.info('Step 4/4: Plot results and get overview params')
    #get median, mean and std of all parameters
    ov_columns = ['median', 'mean', 'std']
    overview_df = pd.DataFrame(columns= ov_columns, index = ctp_columns[1:])
    for key in ctp_columns:
        if 'cellid' in key:
            continue
        overview_df.loc[key, 'median'] = ctp_result_per_cell[key].median()
        overview_df.loc[key, 'mean'] = ctp_result_per_cell[key].mean()
        overview_df.loc[key, 'std'] = ctp_result_per_cell[key].std()

    overview_df.to_csv(f'{f_name}/overview_parameters.csv')
    #plot as histogram
    for param in ctp_columns:
        if not 'ratio' in param:
            continue
        if 'number' in param:
            xlabel = f'{ct_o1_str}/ ({ct_o1_str} + {ct_o2_str}) syn number'
        elif 'area' in param:
            xlabel = f'{ct_o1_str}/ ({ct_o1_str} + {ct_o2_str}) syn area'
        else:
            raise ValueError(f'unknown ratio parameter: {param}')
        sns.histplot(x=param, data=ctp_result_per_cell, color=ct_color,
                     common_norm=False, fill=False, element="step", linewidth=3)
        plt.ylabel('number of cells', fontsize=fontsize)
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.xlim(0, 1)
        plt.title(f'{param} for {ctp_str} cells')
        plt.savefig(f'{f_name}/{ctp_str}_2_{ct_o1_str}_{ct_o2_str}_{param}_hist.png')
        plt.savefig(f'{f_name}/{ctp_str}_2_{ct_o1_str}_{ct_o2_str}_{param}_hist.svg')
        plt.close()
        sns.histplot(x=param, data=ctp_result_per_cell, color=ct_color,
                     common_norm=False, fill=False, element="step", linewidth=3, stat='percent')
        plt.ylabel('% of cells', fontsize=fontsize)
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.xlim(0, 1)
        plt.title(f'{param} for {ctp_str} cells')
        plt.savefig(f'{f_name}/{ctp_str}_2_{ct_o1_str}_{ct_o2_str}_{param}_hist_perc.png')
        plt.savefig(f'{f_name}/{ctp_str}_2_{ct_o1_str}_{ct_o2_str}_{param}_hist_perc.svg')
        plt.close()

    log.info('Analysis finished')