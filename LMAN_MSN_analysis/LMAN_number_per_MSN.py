
if __name__ == '__main__':
    from wholebrain.scratch.arother.bio_analysis.dir_indir_pathway_analysis.subpopulations_per_connectivity import sort_by_connectivity
    from wholebrain.scratch.arother.bio_analysis.dir_indir_pathway_analysis.connectivity_between2cts import synapses_between2cts, compare_connectivity, synapses_ax2ct, compare_connectivity_multiple
    from wholebrain.scratch.arother.bio_analysis.dir_indir_pathway_analysis.compartment_volume_celltype import \
        axon_den_arborization_ct, compare_compartment_volume_ct_multiple
    from wholebrain.scratch.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from wholebrain.scratch.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    from syconn.reps.segmentation import SegmentationDataset
    from wholebrain.scratch.arother.bio_analysis.general.result_helper import plot_nx_graph
    import os as os
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    from tqdm import tqdm
    import numpy as np


    global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    start = time.time()
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    min_comp_len = 200
    max_MSN_path_len = 7500
    syn_prob = 0.8
    min_syn_size = 0.1
    f_name = "wholebrain/scratch/arother/bio_analysis_results/LMAN_MSN_analysis/220728_j0251v4_LMAN_MSN_GP_est_mcl_%i_synprob_%.2f" % (
    min_comp_len, syn_prob)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('LMAN MSN connectivity estimate', log_dir=f_name + '/logs/')
    log.info("min_comp_len = %i, max_MSN_path_len = %i, syn_prob = %i, min_syn_size = %i" % (min_comp_len, max_MSN_path_len, syn_prob, min_syn_size))
    time_stamps = [time.time()]
    step_idents = ['t-0']

    # 1st part of the analysis: get estimate on how many "complete" LMAN branches
    # project to one MSN and how many MSN one LMAN projects to

    log.info("Step 1/X load suitable LMAN and MSN, filter for min_comp_len and max_path_len")
    # load full MSN and filter for min_comp_len, also filter out if total_comp_len > 7500 mm (likely glia merger)
    LMAN_dict = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/ax_LMA_dict.pkl")
    LMAN_ids = load_pkl2obj("/wholebrain/scratch/arother/j0251v4_prep/LMAN_handpicked_arr.pkl")
    MSN_ids  = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_MSN_arr.pkl")
    MSN_dict = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_MSN_dict.pkl")
    MSN_ids = check_comp_lengths_ct(cellids = MSN_ids, fullcelldict = MSN_dict, min_comp_len = min_comp_len, axon_only = False, max_path_len = max_MSN_path_len)

    time_stamps = [time.time()]
    step_idents = ["load and filter cells"]

    log.info("Step 2/X: filter synapses from suitable LMAN and MSN")
    #prefilter synapse caches from LMAN onto MSN synapses
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(sd_synssv, pre_cts = [3],
                                                                                                        post_cts = [2], syn_prob_thresh = syn_prob,
                                                                                                        min_syn_size = min_syn_size, axo_den_so = True)
    #filter out synapses that are not from LMAN or MSN ids
    msnids_inds = np.any(np.in1d(m_ssv_partners, MSN_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[msnids_inds]
    m_ids = m_ids[msnids_inds]
    m_axs = m_axs[msnids_inds]
    m_ssv_partners = m_ssv_partners[msnids_inds]
    m_sizes  = m_sizes[msnids_inds]
    m_spiness = m_spiness[msnids_inds]
    m_rep_coord = m_rep_coord[msnids_inds]
    lmanids_inds = np.any(np.in1d(m_ssv_partners, LMAN_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[lmanids_inds]
    m_ids = m_ids[lmanids_inds]
    m_axs = m_axs[lmanids_inds]
    m_ssv_partners = m_ssv_partners[lmanids_inds]
    m_sizes = m_sizes[lmanids_inds]
    m_spiness = m_spiness[lmanids_inds]
    m_rep_coord = m_rep_coord[lmanids_inds]

    time_stamps = [time.time()]
    step_idents = ["filtered synapses"]

    log.info("Step 3/X: Create per cell dictionary for LMAN and MSN")
    #make per MSN dictionary of LMAN ids they are getting input from, number of LMAN
    #number of synapses, in total and per axon, sum of mesh area
    #create similar dictionary but with LMAN ids as key
    LMAN_proj_dict = {id: {"MSN ids": [], "number of synapses": 0, "number MSN cells": 0, "sum size synapses": 0,
                           "number of synapses per MSN": 0, "sum size synapses per MSN": 0}}
    MSN_rec_dict = {id: {"LMAN ids": [], "number of synapses": 0, "number LMAN cells": 0, "sum size synapses": 0,
                           "number of synapses per LMAN": 0, "sum size synapses per MSN": 0}}

    time_stamps = [time.time()]
    step_idents = ["created per cell dictionaries for LMAN and MSN"]

    log.info("Step 4/X: Plot LMAN to MSN results")
    #plot results and save dictionary

    time_stamps = [time.time()]
    step_idents = ["plotted results for LMAN and MSN"]

    #2nd part of analysis: see how MSN from different LMANs project to GPi

    log.info("Step 5/X: load and filter GPi cells for min_comp_len")
    GPi_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_GPi_arr.pkl")
    GPi_dict = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_GPi_dict.pkl")
    GPi_ids = check_comp_lengths_ct(cellids=GPi_ids, fullcelldict=GPi_dict, min_comp_len=min_comp_len, axon_only=False,
                                    max_path_len=None)
    time_stamps = [time.time()]
    step_idents = ["filtered GPi cells"]

    log.info("Step 6/X: Filter MSN to GPi synapses")
    #prefilter synapses from MSN -> GP
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(sd_synssv,
                                                                                                        pre_cts=[2],
                                                                                                        post_cts=[7],
                                                                                                        syn_prob_thresh=syn_prob,
                                                                                                        min_syn_size=min_syn_size,
                                                                                                        axo_den_so=True)
    #filter synapses again to only include selected MSN, GPi ids (only use MSN ids that are keys in the dictionary above)
    MSN_ids = list(x_dict.keys())
    msnids_inds = np.any(np.in1d(m_ssv_partners, MSN_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[msnids_inds]
    m_ids = m_ids[msnids_inds]
    m_axs = m_axs[msnids_inds]
    m_ssv_partners = m_ssv_partners[msnids_inds]
    m_sizes = m_sizes[msnids_inds]
    m_spiness = m_spiness[msnids_inds]
    m_rep_coord = m_rep_coord[msnids_inds]
    gpiids_inds = np.any(np.in1d(m_ssv_partners, GPi_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[gpiids_inds]
    m_ids = m_ids[gpiids_inds]
    m_axs = m_axs[gpiids_inds]
    m_ssv_partners = m_ssv_partners[gpiids_inds]
    m_sizes = m_sizes[gpiids_inds]
    m_spiness = m_spiness[gpiids_inds]
    m_rep_coord = m_rep_coord[gpiids_inds]

    time_stamps = [time.time()]
    step_idents = ["filtered synapses"]

    log.info("Step 7/X: Create per cell dictionary for GPi, add GPi info to existing MSN, LMAN dicts")
    #create dictionary with GPi as keys to see how many MSNs they get input from and how many LMANs
    #also add different GPis that MSN project to MSN dictionary
    #add GPis that are projected to via MSN to LMAN dictionary
    GPi_rec_dict = {id: {"MSN ids": [], "number of synapses": 0, "number MSN cells": 0, "sum size synapses": 0,
                         "number of synapses per MSN": 0, "sum size synapses per MSN": 0}}
    #plot results