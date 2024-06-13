#get cell-specific connectivity of recurrently connecte cellypes
#get cellids each cell get input from and projects to
#also other way around
#see if input and output of each cell matches
#calculate percentage of overlap based on cellid number
#calculate percentage of overlap based on synapse size

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors, CompColors
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_general, filter_synapse_caches_for_ct
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    import os as os
    import pandas as pd
    import numpy as np
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
    ct1 = 4
    ct2 = 6
    ct1_str = ct_dict[ct1]
    ct2_str = ct_dict[ct2]
    fontsize = 20
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/240613_j0251{version}_{ct1_str}_{ct2_str}_recurr_conn_mcl_%i_synprob_%.2f_%s_fs%i" % (
    min_comp_len_cell, syn_prob, color_key, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('recurr_conn_log', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i for full cells, syn_prob = %.1f, min_syn_size = %.1f, known mergers excluded = %s" % (
        min_comp_len_cell, syn_prob, min_syn_size, exclude_known_mergers))
    log.info(f'Cell-specific connectivity between {ct1_str} and {ct2_str} will be analysed')
    log.info('Goal is to see if input comes from same cells, output goes to')

    log.info('Step 1/X: Get suitable cellids')
    known_mergers = bio_params.load_known_mergers()
    misclassified_asto_ids = bio_params.load_potential_astros()
    cts = [ct1, ct2]
    suitable_ids_dict = {}
    all_suitable_ids = []
    for ct in cts:
        ct_str = ct_dict[ct]
        cell_dict = bio_params.load_cell_dict(ct)
        # get ids with min compartment length
        cellids = np.array(list(cell_dict.keys()))
        if exclude_known_mergers:
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
            cellids = cellids[astro_inds]
        cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cell,
                                                    axon_only=False,
                                                    max_path_len=None)
        suitable_ids_dict[ct] = cellids_checked
        all_suitable_ids.append(cellids_checked)

    all_suitable_ids = np.concatenate(all_suitable_ids)

    log.info('Step 2/3: Filter synapses for celltypes')
    # prefilter synapses for synapse prob thresh and min syn size
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob = filter_synapse_caches_general(
        sd_synssv,
        syn_prob_thresh=syn_prob,
        min_syn_size=min_syn_size)
    #make sure synapses only between suitable ids
    suit_ct_inds = np.any(np.in1d(m_ssv_partners, all_suitable_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[suit_ct_inds]
    m_ids = m_ids[suit_ct_inds]
    m_ssv_partners = m_ssv_partners[suit_ct_inds]
    m_sizes = m_sizes[suit_ct_inds]
    m_axs = m_axs[suit_ct_inds]
    synapse_cache = [m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob]
    #get synapses from ct1 -> ct2
    ct1_2_ct2_cts, ct1_2_ct2_ids, ct1_2_ct2_axs, ct1_2_ct2_ssv_partners, ct1_2_ct2_sizes, ct1_2_ct2_spiness, ct1_2_ct2_rep_coord = filter_synapse_caches_for_ct(pre_cts=[ct1],
                                                                                                        post_cts=[ct2],
                                                                                                        syn_prob_thresh=None,
                                                                                                        min_syn_size=None,
                                                                                                        axo_den_so=True,
                                                                                                        synapses_caches=synapse_cache)

    #get synapses from ct2 -> ct1
    ct2_2_ct1_cts, ct2_2_ct1_ids, ct2_2_ct1_axs, ct2_2_ct1_ssv_partners, ct2_2_ct1_sizes, ct2_2_ct1_spiness, ct2_2_ct1_rep_coord = filter_synapse_caches_for_ct(
        pre_cts=[ct2],
        post_cts=[ct1],
        syn_prob_thresh=None,
        min_syn_size=None,
        axo_den_so=True,
        synapses_caches=synapse_cache)


    #create dictionary for each cells which cells it projects to, receives input from
    #also calculate summed synapse size per cell pair (maybe also number)
    #create all this once for ct1 outgoing, ct2 incoming, ct1 incoming, ct2 outgoing

    #get number each cell projects to, recevies inputs from
    #calculate overlap between the two in terms of cellids
    #calculate overlap between the two in terms of syn number
    #plot for both celltypes

    #get statistics, overview params