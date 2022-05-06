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
    from scipy.stats import ranksums
    from wholebrain.scratch.arother.bio_analysis.general.analysis_morph_helper import get_myelin_fraction, get_compartment_radii, get_organell_volume_density
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    from wholebrain.scratch.arother.bio_analysis.general.result_helper import  ComparingResultsForPLotting
    import itertools
    import seaborn as sns
    import matplotlib.pyplot as plt
    from multiprocessing import pool
    from functools import partial

    from tqdm import tqdm

    global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    start = time.time()
    comp_length = 200
    f_name = "wholebrain/scratch/arother/bio_analysis_results/dir_indir_pathway_analysis/220506_j0251v4_GPe_i_hpv3_myelin_mito_radius_%i_newcolors" % comp_length
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('GPe, GPi comparison connectivity', log_dir=f_name + '/logs/')
    log.info("GPe/i comparison starts")
    time_stamps = [time.time()]
    step_idents = ['t-0']
    #ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   #10: "NGF"}

    '''
    GPe_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v3_prep/handpicked_GPe_arr.pkl")
    GPe_axon_length_dict = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v3_prep/full_GPe_axondict.pkl")
    GPe_dendrite_length_dict = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v3_prep/full_GPe_dendritedict.pkl")
    GPi_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v3_prep/handpicked_GPi_arr.pkl")
    GPi_axon_length_dict = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v3_prep/full_GPi_axondict.pkl")
    GPi_dendrite_length_dict = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v3_prep/full_GPi_dendritedict.pkl")
    '''

    GPi_full_cell_dict = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_GPi_dict.pkl")
    GPe_full_cell_dict = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_GPe_dict.pkl")

    GPe_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_GPe_arr.pkl")
    GPi_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_GPi_arr.pkl")

    # cellids handpicked from v3, via mapping dict and included in V4_full_cells of celltype
    #GPe_ids = load_pkl2obj(
    #    "/wholebrain/scratch/arother/j0251v4_prep/full_GPe_arr_hp_v3.pkl")
    #GPi_ids = load_pkl2obj(
    #    "/wholebrain/scratch/arother/j0251v4_prep/full_GPi_arr_hp_v3.pkl")


    axon_median_radius_gpe = np.zeros(len(GPe_ids))
    axon_mito_volume_density_gpe = np.zeros(len(GPe_ids))
    axon_myelin_gpe = np.zeros(len(GPe_ids))
    axon_median_radius_gpi = np.zeros(len(GPi_ids))
    axon_mito_volume_density_gpi = np.zeros(len(GPi_ids))
    axon_myelin_gpi = np.zeros(len(GPi_ids))
    sd_mitossv = SegmentationDataset("mi", working_dir=global_params.config.working_dir)
    cached_mito_ids = sd_mitossv.ids
    cached_mito_mesh_bb = sd_mitossv.load_numpy_data("mesh_bb")
    cached_mito_rep_coords = sd_mitossv.load_numpy_data("rep_coord")
    cached_mito_volumes = sd_mitossv.load_numpy_data("size")


    log.info("Step 1/3: Get information from GPe")
    for i, cellid in enumerate(tqdm(GPe_ids)):
        cell = SuperSegmentationObject(cellid)
        cell.load_skeleton()
        myelin_results= get_myelin_fraction(cellid = cellid, cell = cell, min_comp_len = comp_length, load_skeleton = False)
        abs_myelin_cell = myelin_results[0]
        rel_myelin_cell = myelin_results[1]
        if abs_myelin_cell == 0:
            continue
        #cell.load_skeleton()
        mito_results = get_organell_volume_density(cellid = cellid, cell = cell, cached_so_ids = cached_mito_ids,
                                    cached_so_rep_coord = cached_mito_rep_coords, cached_so_volume = cached_mito_volumes,
                                    full_cell_dict = GPe_full_cell_dict, k=3, min_comp_len=100, skeleton_loaded = True)
        axo_mito_density_cell = mito_results[0]
        den_mito_density_cell = mito_results[1]
        axo_mito_volume_density_cell = mito_results[2]
        den_mito_volume_density_cell = mito_results[3]
        if den_mito_density_cell == 0:
            continue
        axon_inds = np.nonzero(cell.skeleton["axoness_avg10000"] == 1)[0]
        axon_radii_cell = get_compartment_radii(cell, comp_inds = axon_inds)
        ax_median_radius_cell = np.median(axon_radii_cell)
        dendrite_inds = np.nonzero(cell.skeleton["axoness_avg10000"] == 0)[0]
        den_radii_cell = get_compartment_radii(cell, comp_inds=dendrite_inds)
        den_median_radius_cell = np.median(den_radii_cell)
        axon_median_radius_gpe[i] = ax_median_radius_cell
        axon_mito_volume_density_gpe[i] = axo_mito_volume_density_cell
        axon_myelin_gpe[i] = rel_myelin_cell

    gpe_nonzero = axon_median_radius_gpe > 0
    GPe_params = {"axon median radius": axon_median_radius_gpe[gpe_nonzero], "axon mitochondria volume density": axon_mito_volume_density_gpe[gpe_nonzero],
                  "axon myelin fraction": axon_myelin_gpe[gpe_nonzero], "cellids": GPe_ids[gpe_nonzero]}
    GPe_param_df = pd.DataFrame(GPe_params)
    GPe_param_df.to_csv("%s/GPe_params.csv" % f_name)
    write_obj2pkl("%s/GPe_dict.pkl" % f_name, GPe_params)

    time_stamps = [time.time()]
    step_idents = ["GPe axon parameters collected"]

    log.info("Step 2/3: Get information from GPi")
    for i, cellid in enumerate(tqdm(GPi_ids)):
        cell = SuperSegmentationObject(cellid)
        cell.load_skeleton()
        myelin_results = get_myelin_fraction(cellid=cellid, cell=cell, min_comp_len=comp_length, load_skeleton=False)
        abs_myelin_cell = myelin_results[0]
        rel_myelin_cell = myelin_results[1]
        if abs_myelin_cell == 0:
            continue
        # cell.load_skeleton()
        mito_results = get_organell_volume_density(cellid=cellid, cell=cell, cached_so_ids=cached_mito_ids,
                                                   cached_so_rep_coord=cached_mito_rep_coords,
                                                   cached_so_volume=cached_mito_volumes,
                                                   full_cell_dict=GPi_full_cell_dict, k=3, min_comp_len=100,
                                                   skeleton_loaded=True)
        axo_mito_density_cell = mito_results[0]
        den_mito_density_cell = mito_results[1]
        axo_mito_volume_density_cell = mito_results[2]
        den_mito_volume_density_cell = mito_results[3]
        if den_mito_density_cell == 0:
            continue
        axon_inds = np.nonzero(cell.skeleton["axoness_avg10000"] == 1)[0]
        axon_radii_cell = get_compartment_radii(cell, comp_inds=axon_inds)
        ax_median_radius_cell = np.median(axon_radii_cell)
        dendrite_inds = np.nonzero(cell.skeleton["axoness_avg10000"] == 0)[0]
        den_radii_cell = get_compartment_radii(cell, comp_inds=dendrite_inds)
        den_median_radius_cell = np.median(den_radii_cell)
        axon_median_radius_gpi[i] = ax_median_radius_cell
        axon_mito_volume_density_gpi[i] = axo_mito_volume_density_cell
        axon_myelin_gpi[i] = rel_myelin_cell

    gpi_nonzero = axon_median_radius_gpi > 0
    GPi_params = {"axon median radius": axon_median_radius_gpi[gpi_nonzero],
                  "axon mitochondria volume density": axon_mito_volume_density_gpi[gpi_nonzero],
                  "axon myelin fraction": axon_myelin_gpi[gpi_nonzero], "cellids": GPi_ids[gpi_nonzero]}
    GPi_param_df = pd.DataFrame(GPi_params)
    GPi_param_df.to_csv("%s/GPi_params.csv" % f_name)
    write_obj2pkl("%s/GPi_dict.pkl" % f_name, GPi_params)

    time_stamps = [time.time()]
    step_idents = ["GPi axon parameters collected"]

    log.info("Step 3/3 compare GPe and GPi")
    key_list = list(GPe_params.keys())[:-1]
    results_comparison = ComparingResultsForPLotting(celltype1 = "GPe", celltype2 = "GPi", filename = f_name, dictionary1 = GPe_params, dictionary2 = GPi_params, color1 = "#592A87", color2 = "#2AC644")
    ranksum_results = pd.DataFrame(columns=key_list, index=["stats", "p value"])
    GPe_len = len(GPe_params["cellids"])
    GPi_len = len(GPi_params["cellids"])
    sum_length = GPe_len + GPi_len
    all_param_df = pd.DataFrame(columns=np.hstack([key_list, "celltype"]), index=range(sum_length))
    all_param_df.loc[0: GPe_len- 1, "celltype"] = "GPe"
    all_param_df.loc[GPe_len: sum_length - 1, "celltype"] = "GPi"
    for key in key_list:
        if "cellids" in key:
            continue
        results_for_plotting = results_comparison.result_df_per_param(key)
        stats, p_value = ranksums(GPe_params[key], GPi_params[key])
        ranksum_results.loc["stats", key] = stats
        ranksum_results.loc["p value", key] = p_value
        if "mito" in key:
            subcell = "mitochondria"
        elif "myelin" in key:
            subcell = "myelin"
        else:
            subcell = "axon"
        results_comparison.plot_violin(key, results_for_plotting, subcell=subcell, stripplot=True)
        results_comparison.plot_box(key, results_for_plotting, subcell=subcell, stripplot=False)
        results_comparison.plot_hist_comparison(key, subcell=subcell, bins=10, norm_hist=False)
        results_comparison.plot_hist_comparison(key, subcell=subcell, bins=10, norm_hist=True)
        all_param_df.loc[0: GPe_len - 1, key] = GPe_params[key]
        all_param_df.loc[GPe_len: sum_length - 1, key] = GPi_params[key]

    ranksum_results.to_csv("%s/ranksum_results.csv" % f_name)
    all_param_df.to_csv("%s/GPe_GPi_params.csv" % f_name)

    combinations = list(itertools.combinations(range(len(key_list)), 2))
    #sns.set(font_scale=1.5)
    for comb in combinations:
        x = key_list[comb[0]]
        y = key_list[comb[1]]
        raise ValueError
        g = sns.JointGrid(data= all_param_df, x = x, y = y, hue = "celltype", palette = results_comparison.color_palette)
        g.plot_joint(sns.scatterplot)
        g.plot_marginals(sns.histplot,  fill = True, alpha = 0.3,
                         kde=False, bins=10, palette = results_comparison.color_palette)
        plt.legend()
        g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), fontsize = 20)
        g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), fontsize=20)
        if "radius" in x:
            plt.xlabel("%s in µm" % x)
        elif "volume density" in x:
            plt.xlabel("%s in µm³/µm" % x)

        if "radius" in y:
            plt.ylabel("%s in µm" % y)
        elif "volume density" in y:
            plt.ylabel("%s in µm³/µm" % y)

        plt.savefig("%s/%s_%s_joinplot.svg" % (f_name, x, y))
        plt.close()









