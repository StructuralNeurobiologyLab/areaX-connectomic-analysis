# get mitos, axon median radius, myelinasation of GPe/i and plot against each other

if __name__ == '__main__':
    from wholebrain.scratch.arother.bio_analysis.dir_indir_pathway_analysis.compartment_volume_celltype import axon_den_arborization_ct, compare_compartment_volume_ct
    from wholebrain.scratch.arother.bio_analysis.dir_indir_pathway_analysis.connectivity_between2cts import synapses_between2cts, compare_connectivity
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    from syconn.reps.segmentation import SegmentationDataset
    import os as os
    import pandas as pd
    import numpy as np
    from wholebrain.scratch.arother.bio_analysis.general.result_helper import plot_nx_graph
    from wholebrain.scratch.arother.bio_analysis.general.analysis_helper import get_myelin_fraction, get_compartment_radii, get_organell_volume_density
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    from tqdm import tqdm

    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    start = time.time()
    f_name = "u/arother/bio_analysis_results/dir_indir_pathway_analysis/211126_j0251v3_GPe_i_comparison"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('GPe, GPi comparison connectivity', log_dir=f_name + '/logs/')
    log.info("GPe/i comparison starts")
    time_stamps = [time.time()]
    step_idents = ['t-0']
    #ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   #10: "NGF"}
    comp_length = 200

    GPe_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v3_prep/handpicked_GPe_c%i.pkl" % comp_length)
    GPe_axon_length_dict = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v3_prep/full_GPe_axondict.pkl")
    GPe_dendrite_length_dict = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v3_prep/full_GPe_dendritedict.pkl")
    GPi_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v3_prep/handpicked_GPi_c%i.pkl" % comp_length)
    GPi_axon_length_dict = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v3_prep/full_GPi_axondict.pkl")
    GPi_dendrite_length_dict = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v3_prep/full_GPi_dendritedict.pkl")


    axon_median_radius_gpe = np.zeros(len(GPe_ids))
    axon_mito_volume_density_gpe = np.zeros(len(GPe_ids))
    axon_myelin_gpe = np.zeros(len(GPe_ids))
    axon_median_radius_gpi = np.zeros(len(GPi_ids))
    axon_mito_volume_density_gpi = np.zeros(len(GPi_ids))
    axon_myelin_gpi = np.zeros(len(GPi_ids))

    log.info("Step 1/X: Get information from GPe")
    for i, cell in enumerate(tqdm(ssd.get_super_segmentation_object(GPe_ids))):
        cell.load_skeleton()
        abs_myelin_cell, rel_myelin_cell = get_myelin_fraction(cell, min_comp_len = comp_length)
        if abs_myelin_cell == 0:
            continue
        axon_inds = np.nonzero(cell.skeleton["axoness_avg10000"] == 1)[0]
        axon_radii_cell = get_compartment_radii(cell, comp_inds = axon_inds)
        ax_median_radius_cell = np.median(axon_radii_cell)
        dendrite_inds = np.nonzero(cell.skeleton["axoness_avg10000"] == 0)[0]
        den_radii_cell = get_compartment_radii(cell, comp_inds=dendrite_inds)
        den_median_radius_cell = np.median(den_radii_cell)

