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
    f_name = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/230831_j0251v5_MSN_GP_syn_multisyn_mcl_%i_synprob_%.2f_kde%i_replot" % (
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
    f_name_saving1 = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/230830_j0251v5_MSN_GPratio_spine_density_mcl_%i_synprob_%.2f_kde%i_replot" % (
        min_comp_len, syn_prob, kde)
    log.info(f'Use morph parameters from {f_name_saving1}')
    msn_result_df = pd.read_csv(f'{f_name_saving1}/msn_spine_density_GPratio.csv', index_col=0)
    for key in msn_result_df.keys():
        if not 'GP' in key or 'cellid' in key:
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
    log.info('Get GPe parameter information per MSN cell')
    # get number of GP partner cells per cell and information about syn size for each of them
    #get msn cells connected to gpe
    msn_gpe_ids = msn_result_df['cellid'][msn_result_df['syn number to GPe'] > 0]
    gpe_input = [[cellid, m_ssv_partners, m_sizes, m_rep_coord, m_cts, gpe_ct] for cellid in msn_gpe_ids]
    gpe_output = start_multiprocess_imap(get_multi_syn_info_per_cell, input)
    gpe_output_dict = dict(ChainMap(*gpe_output))
    write_obj2pkl(f'{f_name}/msn_gpe_indiv_conns_dict.pkl', gpe_output_dict)
    #add per cell information to dictionary
    log.info('Get GPi parameter information per MSN cell')
    msn_gpi_ids = msn_result_df['cellid'][msn_result_df['syn number to GPi'] > 0]
    gpi_input = [[cellid, m_ssv_partners, m_sizes, m_rep_coord, m_cts, gpi_ct] for cellid in msn_gpi_ids]
    gpi_output = start_multiprocess_imap(get_multi_syn_info_per_cell, input)
    gpi_output_dict = dict(ChainMap(*gpe_output))
    write_obj2pkl(f'{f_name}/msn_gpi_indiv_conns_dict.pkl', gpi_output_dict)
    # add per cell information to dictionary
    #make result df only for multi_synaptic connections

    #make per cell dictionary 
    #% of GP cells targeted are monosynaptic
    #% 2, 3 etc. -> histogram
    #distance and size between individual synapses (both from MSN side but also from GPe/ GPi side
    
    log.info('Step 3/5: Plot information about MSN to GP synapses in general')
    
    
    log.info('Step 4/5: Calculate statistics between MSN cells')

    log.info('Step 5/5: Plot information about MSN to GP synaspes dependend on MSN-GP groups')
    
    log.info('Analysis finished')