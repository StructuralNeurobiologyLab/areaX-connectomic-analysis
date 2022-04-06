if __name__ == '__main__':
    from wholebrain.scratch.arother.bio_analysis.dir_indir_pathway_analysis.subpopulations_per_connectivity import sort_by_connectivity
    from wholebrain.scratch.arother.bio_analysis.dir_indir_pathway_analysis.connectivity_between2cts import synapses_between2cts, compare_connectivity
    from wholebrain.scratch.arother.bio_analysis.dir_indir_pathway_analysis.spiness_sorting import saving_spiness_percentiles
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    from syconn.reps.segmentation import SegmentationDataset
    from wholebrain.scratch.arother.bio_analysis.general.result_helper import plot_nx_graph
    import os as os
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj

    global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    start = time.time()
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    cl = 200
    syn_prob = 0.8
    min_syn_size = 0.1
    f_name = "wholebrain/scratch/arother/bio_analysis_results/dir_indir_pathway_analysis/220404_j0251v4_MSN_connGP_comparison_mcl_%i_synprob_%.2f" % (cl, syn_prob)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('MSN percentile comparison connectivity', log_dir=f_name + '/logs/')
    log.info("MSN percentile comparison starts")
    time_stamps = [time.time()]
    step_idents = ['t-0']
    f_name_saving = "/wholebrain/scratch/arother/j0251v4_prep/"

    GPe_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_GPe_arr.pkl")
    GPi_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_GPi_arr.pkl")
    MSN_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_MSN_arr.pkl")

    log.info("Step 1/X: sort MSN based on connectivity to GPe and GPi")
    sort_by_connectivity(sd_synssv, ct1 = 2, ct2 = 6, ct3 = 7, cellids1 = MSN_ids, cellids2 = GPe_ids, cellids3 = GPi_ids,
                         f_name = f_name, f_name_saving = f_name_saving, min_comp_len = cl, syn_prob_thresh = syn_prob, min_syn_size = min_syn_size)
