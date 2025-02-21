#get ratio of inputs for one cell population to see differences
#example: MSN: calculate fraction of LMAN are out of HVC/LMAN input

if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    import os as os
    import pandas as pd
    import numpy as np
    from syconn.reps.segmentation import SegmentationDataset
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct, \
        get_multi_syn_info_per_cell
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
    celltype_population = 3
    input_ct_1 = 1
    input_ct_2 = 2
    fontsize = 20

    axon_cts = analysis_params.axon_cts()
    if celltype_population in axon_cts:
        raise ValueError('Celltype receiving inputs can not be from projecting axons.')
    ctp_str = ct_dict[celltype_population]
    ct_i1_str = ct_dict[input_ct_1]
    ct_i2_str = ct_dict[input_ct_2]
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/250221_j0251{version}_{ctp_str}_{ct_i1_str}_{ct_i2_str}_GP_input_ratio_fs{fontsize}
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'{ctp_str}_input_ratio_logs', log_dir=f_name)
    log.info(f' min comp len cell = {min_comp_len_cell},min comp len ax = {min_comp_len_ax}, cell type for population = {ctp_str}, input celltypes are: {ct_i1_str}, {ct_i2_str}.')
    log.info(f'synapse probability = {syn_prob}, min syn size = {min_syn_size} µm²')

    log.info('Step 1/4: Get suitable cell ids')
    known_mergers = analysis_params.load_known_mergers()
    ctp_dict = analysis_params.load_cell_dict(celltype=celltype_population)
    ctp_ids = np.array(list(ctp_dict.keys()))
    merger_inds = np.in1d(ctp_ids, known_mergers) == False
    ctp_ids = ctp_ids[merger_inds]
    ctp_ids = check_comp_lengths_ct(cellids=ctp_ids, fullcelldict=ctp_dict, min_comp_len=min_comp_len_cell,
                                    axon_only=False,
                                    max_path_len=None)
    log.info(f'{len(ctp_ids)} suitable cells of cell type {ctp_str}.')

    in1_dict = analysis_params.load_cell_dict(input_ct_1)
    in1_ids = np.array(list(in1_dict.keys()))
    merger_inds = np.in1d(in1_ids, known_mergers) == False
    in1_ids = in1_ids[merger_inds]
    if input_ct_1 in axon_cts:
        in1_ids = check_comp_lengths_ct(cellids=in1_ids, fullcelldict=in1_dict, min_comp_len=min_comp_len_ax,
                                        axon_only=True,
                                        max_path_len=None)
        log.info(f'{len(in1_ids)} suitable axons of cell type {ct_i1_str}.')
    else:
        in1_ids = check_comp_lengths_ct(cellids=in1_ids, fullcelldict=in1_dict, min_comp_len=min_comp_len_cell,
                                        axon_only=False,
                                        max_path_len=None)
        log.info(f'{len(in1_ids)} suitable cells of cell type {ct_i1_str}.')

    in2_dict = analysis_params.load_cell_dict(input_ct_2)
    in2_ids = np.array(list(in2_dict.keys()))
    merger_inds = np.in1d(in2_ids, known_mergers) == False
    in2_ids = in2_ids[merger_inds]
    if input_ct_2 in axon_cts:
        in2_ids = check_comp_lengths_ct(cellids=in2_ids, fullcelldict=in2_dict, min_comp_len=min_comp_len_ax,
                                        axon_only=True,
                                        max_path_len=None)
        log.info(f'{len(in2_ids)} suitable axons of cell type {ct_i2_str}.')
    else:
        in2_ids = check_comp_lengths_ct(cellids=in2_ids, fullcelldict=in2_dict, min_comp_len=min_comp_len_cell,
                                        axon_only=False,
                                        max_path_len=None)
        log.info(f'{len(in2_ids)} suitable cells of cell type {ct_i2_str}.')

    log.info('Step 2/4: Filter synapses')
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.wd)
    m_cts, m_axs, m_ssv_partners, m_sizes, m_rep_coord = filter_synapse_caches_for_ct(
        sd_synssv=sd_synssv,
        pre_cts=[celltype_population],
        post_cts=[input_ct_1, input_ct_2],
        syn_prob_thresh=syn_prob,
        min_syn_size=min_syn_size,
        axo_den_so=True,
        synapses_caches=None)
    all_suitable_ids = np.hstack([ctp_ids, in1_ids, in2_ids])
    suit_ct_inds = np.all(np.in1d(m_ssv_partners, all_suitable_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[suit_ct_inds]
    m_ssv_partners = m_ssv_partners[suit_ct_inds]
    m_sizes = m_sizes[suit_ct_inds]
    m_axs = m_axs[suit_ct_inds]
    m_rep_coord = m_rep_coord[suit_ct_inds]
    log.info(f'{len(m_sizes)} synapses found from {ct_i1_str} and {ct_i2_str} to {ctp_str}. The summed synaptic area = {np.sum(m_sizes)} µm²; mean per {ctp_str} cell: {np.sum(m_sizes)/ len(ctp_ids)}')

    log.info(f'Step 3/4: Calculate ratio of inputs per {ctp_str} cell')
    ctp_columns = ['cellid', f'syn number {ct_i1_str}', f'syn area {ct_i1_str}', f'syn number {ct_i2_str}',
                   f'syn area {ct_i1_str}', 'ratio syn numbers', 'ratio syn area']
    ctp_result_per_cell = pd.DataFrame(columns = ctp_columns, index=range(len(ctp_ids)))

    #first for input 1, code adapted from syn_conn_details
    ct1_inds = np.where(m_cts == input_ct_1)
    ct1_ssv_partners = m_ssv_partners[ct1_inds[0]]
    ct1_sizes = m_sizes[ct1_inds[0]]
    ct1_cts = m_cts[ct1_inds[0]]
    # get cellids of all ctp_ids
    conn_inds = np.where(ct1_cts == celltype_population)
    pre_ct1_ids = ct2_ssv_partners[conn_inds]
    # get cellids of ct2
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



    log.info('Step 4/4: Plot results')


    log.info('Analysis finished')