
if __name__ == '__main__':
    from wholebrain.scratch.arother.bio_analysis.dir_indir_pathway_analysis.subpopulations_per_connectivity import sort_by_connectivity
    from wholebrain.scratch.arother.bio_analysis.dir_indir_pathway_analysis.connectivity_between2cts import synapses_between2cts, compare_connectivity, synapses_ax2ct, compare_connectivity_multiple
    from wholebrain.scratch.arother.bio_analysis.dir_indir_pathway_analysis.compartment_volume_celltype import \
        axon_den_arborization_ct, compare_compartment_volume_ct_multiple
    from wholebrain.scratch.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    from syconn.reps.segmentation import SegmentationDataset
    from wholebrain.scratch.arother.bio_analysis.general.result_helper import plot_nx_graph
    import os as os
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj


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

    log.info("Steo 1/X load suitable LMAN and MSN, filter for min_comp_len and max_path_len")
    # load full MSN and filter for min_comp_len, also filter out if total_comp_len > 7500 mm (likely glia merger)
    LMAN_dict = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/ax_LMA_dict.pkl")
    LMAN_ids = load_pkl2obj("/wholebrain/scratch/arother/j0251v4_prep/LMAN_handpicked_arr.pkl")
    MSN_ids  = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_MSN_arr.pkl")
    MSN_dict = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_MSN_dict.pkl")
    MSN_ids = che



    #prefilter synapse caches from LMAN onto MSN synapses
    #filter out synapses that are not from LMAN or MSN ids

    #make per MSN dictionary of LMAN ids they are getting input from, number of LMAN
    #number of synapses, in total and per axon, sum of mesh area
    #create similar dictionary but with LMAN ids as key
    #plot results and save dictionary

    #second part of analysis: see how MSNs from different LMANs project to GPi
    #load full GPis
    #filter for min_comp_len

    #prefilter synapses from MSN -> GP
    #filter synapses again to only include selected MSN, GPi ids (only use MSN ids that are keys in the dictionary above)

    #create dictionary with GPi as keys to see how many MSNs they get input from and how many LMANs
    #also add different GPis that MSN project to MSN dictionary
    #add GPis that are projected to via MSN to LMAN dictionary
    #plot results