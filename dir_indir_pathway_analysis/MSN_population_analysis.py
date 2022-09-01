if __name__ == '__main__':
    from wholebrain.scratch.arother.bio_analysis.dir_indir_pathway_analysis.compartment_volume_celltype import axon_den_arborization_ct, compare_compartment_volume_ct
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

    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    start = time.time()
    f_name = "wholebrain/scratch/arother/bio_analysis_results/dir_indir_pathway_analysis/211203_j0251v3_MSN_compartment"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('MSN compartment volume', log_dir=f_name + '/logs/')
    log.info("MSN percentile comparison starts")
    time_stamps = [time.time()]
    step_idents = ['t-0']
    #ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   #10: "NGF"}
    cl = 200

    log.info("Step 1/1: MSN compartment parameter")
    # create MSN spiness percentiles with different comp_lengths
    result_MSN_filename_p1 = axon_den_arborization_ct(ssd, celltype=2, filename=f_name, full_cells=True, handpicked=False, min_comp_len = cl)

    time_stamps = [time.time()]
    step_idents = ["MSN compartment parameters calculated"]