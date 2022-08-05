
if __name__ == '__main__':
    from wholebrain.scratch.arother.bio_analysis.general.analysis_morph_helper import remove_myelinated_part_axon
    from wholebrain.scratch.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct
    from wholebrain.scratch.arother.bio_analysis.general.result_helper import ResultsForPlotting
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


    global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
    start = time.time()
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    overlap_threshold = 0.8
    non_overlap_threshold = 0.2
    kdtree_radius = 20 #µm
    f_name = "wholebrain/scratch/arother/bio_analysis_results/LMAN_MSN_analysis/220805_j0251v4_LMAN_overlap_analysis_ot_%.1f_not_%.1f_kdtr_%i" % (
    overlap_threshold, non_overlap_threshold, kdtree_radius)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('LMAN overlap analysis', log_dir=f_name + '/logs/')
    log.info("overlap threshold = %.2f, non-overlap threshold = %.2f, kdtree radius = %i" % (overlap_threshold, non_overlap_threshold, kdtree_radius))
    time_stamps = [time.time()]
    step_idents = ['t-0']

    #anaylsis to see if overlapping LMAN axons target same MSN cells
    #also if there are differences between overlapping and non-overlapping lman

    #1st part of analysis: divide LMAN axons in overlapping and non-overlapping
    log.info("Step 1/X: Divide LMAN in overlapping and non-overlapping")
    LMAN_dict = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/ax_LMA_dict.pkl")
    LMAN_ids = load_pkl2obj("/wholebrain/scratch/arother/j0251v4_prep/LMAN_handpicked_arr.pkl")
    #iterate over lmans to remove myelinated parts of axon
    LMAN_skel_dict = {}
    #TO DO: add mutliprocessing once it works
    for lman_id in LMAN_ids:
        branched_node_positions = remove_myelinated_part_axon(lman_id)

#1: divide LMAN axons into overlapping and non-overlapping
#get rid of myelinated part: remove nodes with attr myelin
#only take connected component with most nodes
#compute kdtree with overlapp between all skeleton nodes of two axons (from both sides)
#radius of 10, 20, 50 µm
#determine overlap based on this
#if overlap over threshold (e.g. 80% from both sides) considered as overlap
#if under certain threshold (e.g. 20 %) considered further away
#do this analysis with varying thresholds (put parameter in log file and name)

#2: compare how many MSN ids are shared between overlapping LMANs
#and non- overlapping LMANs
#use dictionary from LMAN_number_per_MSN to lookup MSN partners
#also look at soma distance of MSN
#also check if there is differences going to the GPi
#same via MSN targeted GPi between overlapping LMANs, difference to ones far apart?
#distance of GPi soma?
