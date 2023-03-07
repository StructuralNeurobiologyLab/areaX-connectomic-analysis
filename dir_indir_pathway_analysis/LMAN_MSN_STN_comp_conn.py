#Alexandra Rother
#script to see if LMAN axons going to MSN also synapse to STN
#is there a compartment preference of LMAN -> MSN shaft and STN spine head?

#check which of them also make STN synapses
#how many synapses, percentage of compartment

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct
    from cajal.nvmescratch.users.arother.bio_analysis.dir_indir_pathway_analysis.connectivity_between2cts import get_compartment_specific_connectivity
    from cajal.nvmescratch.users.arother.bio_analysis.general.result_helper import ResultsForPlotting
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    import os as os
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    import numpy as np
    from tqdm import tqdm
    import scipy

    global_params.wd = "cajal/nvmescrastch/projects/from_ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    analysis_params = Analysis_Params(global_params.wd)
    ct_dict = analysis_params.ct_dict()
    min_comp_len = 200
    handpicked_LMAN = True
    # samples per ct
    syn_prob = 0.8
    min_syn_size = 0.1
    msn_ct = 2
    lman_ct = 3
    stn_ct = 0
    color_key = 'axoness_avg10000_comp_maj'
    if handpicked_LMAN:
        f_name = "cajal/scratch/users/arother/bio_analysis_results/general/230228_j0251v4_ct_LMAN_MSN_STN_mcl_%i_k%s_sp_%.1f_ms_%.1f_LMANhp" % (
            min_comp_len, color_key, syn_prob, min_syn_size)
    else:
        f_name = "cajal/scratch/users/arother/bio_analysis_results/general/230228_j0251v4_ct_LMAN_MSN_STN_mcl_%i_k%s_sp_%.1f_ms_%.1f" % (
            min_comp_len, color_key, syn_prob, min_syn_size)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('Connectivity from LMAN to MSN and STN',
                             log_dir=f_name + '/logs/')
    log.info(
        f"min_comp_len = %i, key to color to = %s, handpicked LMAN = %s, syn_prob = %.1f, min_syn_size = %.1f µm²" % (
            min_comp_len, color_key, handpicked_LMAN, syn_prob, min_syn_size))
    time_stamps = [time.time()]
    step_idents = ['t-0']
    cts_for_analysis = [lman_ct, msn_ct, stn_ct]
    cts_str = [ct_dict[i] for i in cts_for_analysis]

    log.info("Step 1/8: load suitable LMAN, MSN and STN, filter for min_comp_lenn")
    known_mergers = analysis_params.load_known_mergers()
    pot_astros = analysis_params.load_potential_astros()
    cache_name = "/cajal/nvmescratch/users/arother/j0251v4_prep"
    cellids_dict = dict()
    for ct in tqdm(cts_for_analysis):
        # only get cells with min_comp_len, MSN with max_comp_len or axons with min ax_len
        ct_str = ct_dict[ct]
        if ct == lman_ct:
            if handpicked_LMAN == True:
                cellids = load_pkl2obj(f"{cache_name}/LMAN_handpicked_arr.pkl")
            else:
                cell_dict = analysis_params.load_cell_dict(ct)
                cellids = np.array(list(cell_dict.keys()))
                merger_inds = np.in1d(cellids, known_mergers) == False
                cellids = cellids[merger_inds]
                cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                axon_only=True, max_path_len=None)
            cellids_dict[ct] = cellids
        else:
            cell_dict = analysis_params.load_cell_dict(ct)
            cellids = load_pkl2obj(
                f"{cache_name}/full_%.3s_arr.pkl" % ct_dict[ct])
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            if ct == 2:
                misclassified_asto_ids = load_pkl2obj(f'{cache_name}/pot_astro_ids.pkl')
                astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
                cellids = cellids[astro_inds]
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                            axon_only=False, max_path_len=None)
            cellids_dict[ct] = cellids
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict[ct]))

    time_stamps = [time.time()]
    step_idents = ["load and filter cells"]

    # get all LMAN ids that synapse to MSN
    # save information on how many synapses, percentage of going to shaft
    #similar to function from LMAN_number_per_MSN

    log.info("Step 2/8: filter synapses from suitable LMAN and MSN")
    compartments = ['soma', 'spine neck', 'spine head', 'dendritic shaft']
    num_comps = len(compartments)
    #prefilter synapse caches from LMAN onto MSN synapses
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(sd_synssv, pre_cts = [lman_ct],
                                                                                                        post_cts = [msn_ct], syn_prob_thresh = syn_prob,
                                                                                                        min_syn_size = min_syn_size, axo_den_so = True)
    #filter out synapses that are not from LMAN or MSN ids
    msnids_inds = np.any(np.in1d(m_ssv_partners, cellids_dict[msn_ct]).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[msnids_inds]
    m_ids = m_ids[msnids_inds]
    m_ssv_partners = m_ssv_partners[msnids_inds]
    m_sizes  = m_sizes[msnids_inds]
    m_axs = m_axs[msnids_inds]
    m_spiness = m_spiness[msnids_inds]
    lmanids_inds = np.any(np.in1d(m_ssv_partners, cellids_dict[lman_ct]).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[lmanids_inds]
    m_ids = m_ids[lmanids_inds]
    m_ssv_partners = m_ssv_partners[lmanids_inds]
    m_sizes = m_sizes[lmanids_inds]
    m_axs = m_axs[lmanids_inds]
    m_spiness = m_spiness[lmanids_inds]

    time_stamps = [time.time()]
    step_idents = ["filtered synapses"]

    log.info("Step 3/8: Get comp information from each LMAN to MSN")
    #make per LMAN dictionary
    #parameters: number of synapses per MSN, sum of syn size, number of syns per comp, sum size per comp,
    #percentage of sum size per comp


    perlman_params, syn_params = get_compartment_specific_connectivity(ct_post=msn_ct,
                                                                       cellids_post=cellids_dict[msn_ct],
                                                                       sd_synssv=sd_synssv,
                                                                       syn_prob=syn_prob,
                                                                       min_syn_size=min_syn_size,
                                                                       ct_pre=lman_ct, cellids_pre=cellids_dict[lman_ct],
                                                                       sort_per_postsyn_ct = False)

    # syn_numbers_ct, sum_sizes_ct, syn_number_perc_ct, sum_sizes_perc_ct, ids_ct = percell_params
    lman_ids2msn = np.unique(perlman_params[-1])
    num_lman2msn_ids = len(lman_ids2msn)
    num_lman_cellids = len(cellids_dict[lman_ct])
    log.info(f'{num_lman2msn_ids} of {num_lman_cellids} LMAN cells '
             f'({100 * num_lman2msn_ids / num_lman_cellids}%) make synapses to MSN cells.')

    percell_params_str = ['number of synapses', 'summed synapse size', 'percentage of synapses',
               'percentage of synapse sizes']
    columns = np.hstack([percell_params_str, 'compartment of postynaptic cells', 'postsynaptic ct'])
    lman_df = pd.DataFrame(columns=columns, index=range(num_lman_cellids*num_comps*2))

    for i_comp, compartment in enumerate(compartments):
        # fill in numbers that summarise all synapses per compartment per celltype
        lman_df.loc[i_comp * num_lman_cellids: (i_comp + 1) * num_lman2msn_ids - 1, 'compartment of postynaptic cells'] = compartment
        lman_df.loc[i_comp * num_lman_cellids: (i_comp + 1) * num_lman2msn_ids - 1,
        'postsynaptic ct'] = ct_dict[msn_ct]
        for iy in range(len(percell_params_str)):
            lman_df.loc[i_comp * num_lman_cellids: (i_comp + 1) * num_lman2msn_ids - 1, percell_params_str[iy]] = perlman_params[iy][compartment]

    lman_df.to_csv(f'{f_name}/lman_msn_comp_dict')
    median_perc_shaft_syns = np.median(lman_df['percentage of synapse sizes'][lman_df['compartment of postynaptic cells'] == 'dendritic shaft'])
    median_perc_sh_syns = np.median(
        lman_df['percentage of synapse sizes'][lman_df['compartment of postynaptic cells'] == 'spine head'])
    log.info(f'Median percentage of shaft synapse sizes to MSN: {median_perc_shaft_syns}')
    log.info(f'Median percentage of spine head synapse sizes to MSM: {median_perc_sh_syns}')

    time_stamps = [time.time()]
    step_idents = ["got params for LMAN to MSN"]

    log.info("Step 4/8: Get comp information from each LMAN to STN")
    # make per LMAN dictionary
    # parameters: number of synapses per MSN, sum of syn size, number of syns per comp, sum size per comp,
    # percentage of sum size per comp

    perlman_params, syn_params = get_compartment_specific_connectivity(ct_post=stn_ct,
                                                                       cellids_post=cellids_dict[stn_ct],
                                                                       sd_synssv=sd_synssv,
                                                                       syn_prob=syn_prob,
                                                                       min_syn_size=min_syn_size,
                                                                       ct_pre=lman_ct,
                                                                       cellids_pre=cellids_dict[lman_ct],
                                                                       sort_per_postsyn_ct=False)

    # syn_numbers_ct, sum_sizes_ct, syn_number_perc_ct, sum_sizes_perc_ct, ids_ct = percell_params
    lman_ids2stn = np.unique(perlman_params[-1])
    num_lman2stn_ids = len(lman_ids2msn)
    log.info(f'{num_lman2stn_ids} of {num_lman_cellids} LMAN cells '
             f'({100 * num_lman2stn_ids / num_lman_cellids}%) make synapses to STN cells.')

    for i_comp, compartment in enumerate(compartments):
        # fill in numbers that summarise all synapses per compartment per celltype
        lman_df.loc[i_comp * num_lman_cellids: (i_comp + 1) * num_lman2msn_ids - 1,
        'compartment of postynaptic cells'] = compartment
        lman_df.loc[i_comp * num_lman_cellids: (i_comp + 1) * num_lman2msn_ids - 1,
        'postsynaptic ct'] = ct_dict[stn_ct]
        for iy in range(len(percell_params_str)):
            lman_df.loc[i_comp * num_lman_cellids: (i_comp + 1) * num_lman2msn_ids - 1, percell_params_str[iy]] = \
            perlman_params[iy][compartment]

    lman_df.to_csv(f'{f_name}/lman_stn_comp_dict')
    median_perc_shaft_syns = np.median(
        lman_df['percentage of synapse sizes'][lman_df['compartment of postynaptic cells'] == 'dendritic shaft'])
    median_perc_sh_syns = np.median(
        lman_df['percentage of synapse sizes'][lman_df['compartment of postynaptic cells'] == 'spine head'])
    log.info(f'Median percentage of shaft synapse sizes to STN: {median_perc_shaft_syns}')
    log.info(f'Median percentage of spine head synapse sizes to STN: {median_perc_sh_syns}')

    time_stamps = [time.time()]
    step_idents = ["got params for LMAN to STN"]

    log.info("Step 4/8: Get comp information from each LMAN to STN")

    time_stamps = [time.time()]
    step_idents = ["got params for LMAN to MSN"]

    raise ValueError


