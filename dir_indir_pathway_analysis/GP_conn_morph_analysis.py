
if __name__ == '__main__':
    from wholebrain.scratch.arother.bio_analysis.dir_indir_pathway_analysis.subpopulations_per_connectivity import sort_by_connectivity, get_ct_via_inputfraction
    from wholebrain.scratch.arother.bio_analysis.dir_indir_pathway_analysis.connectivity_between2cts import synapses_between2cts, compare_connectivity, synapses_ax2ct, compare_connectivity_multiple
    from wholebrain.scratch.arother.bio_analysis.dir_indir_pathway_analysis.compartment_volume_celltype import \
        axon_den_arborization_ct, compare_compartment_volume_ct_multiple, compare_compartment_volume_ct
    from wholebrain.scratch.arother.bio_analysis.dir_indir_pathway_analysis.spiness_sorting import saving_spiness_percentiles
    from wholebrain.scratch.arother.bio_analysis.general.analysis_morph_helper import get_organell_volume_density, get_compartment_radii, get_myelin_fraction
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    from syconn.reps.segmentation import SegmentationDataset
    from wholebrain.scratch.arother.bio_analysis.general.result_helper import plot_nx_graph
    import os as os
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    import numpy as np
    from multiprocessing import pool
    from functools import partial
    from tqdm import tqdm
    import seaborn as sns
    import itertools
    import matplotlib.pyplot as plt

    global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    start = time.time()
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    cl = 200
    syn_prob = 0.8
    min_syn_size = 0.1
    sumsize_threshold = 0.25
    mito_gp_threshold = 0.025
    f_name = "wholebrain/scratch/arother/bio_analysis_results/dir_indir_pathway_analysis/220505_j0251v4_GP_conn_morph_comparison_mcl_%i_synprob_%.2f__sumt_%f_mitot_%f" % (cl, syn_prob, sumsize_threshold, mito_gp_threshold)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('GP identificationa and comparison connectivity', log_dir=f_name + '/logs/')
    log.info("GP indentification starts")
    time_stamps = [time.time()]
    step_idents = ['t-0']

    #this analysis aims at identifying GPs based on input ratio from MSNs,
    #seperating GPe and GPi based on morphology: mitochondria volume density, myelin fraction, axon median radius

    log.info("Step 1/9: Plot synaptic input from MSN to all other full celltypes to identify GPs")
    #use full_cell_dict to identify overall synapse amount and area per cell with compartment length threshold
    #for each celltype (not MSNs): get fraction of MSN input in synapse amount and summed synapse size
    MSN_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_MSN_arr.pkl")
    non_MSN_fullcts = [0, 5, 6, 7, 8, 9, 10]
    non_MSN_cellids_cts = np.array([load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_%.3s_arr.pkl" % ct_dict[i]) for i in non_MSN_fullcts])
    non_MSN_celldicts = np.array([load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_%.3s_dict.pkl" % ct_dict[i]) for i in non_MSN_fullcts])

    log.info("Step 1a/9: Get GP cellids and MSN inputs to full cells")
    GP_ids, msn_input_results_dict = get_ct_via_inputfraction(sd_synssv, pre_ct = 2, post_cts = non_MSN_fullcts, pre_cellids = MSN_ids, post_cellids = non_MSN_cellids_cts,
                                                              filename = f_name, celltype_threshold = sumsize_threshold, pre_label = None, post_labels = None,
                                                              min_comp_len = cl, min_syn_size = min_syn_size, syn_prob_thresh = syn_prob, compare2mcl = True)
    log.info("Step 1b/9: plot results in 2D vs organelle density")
    sd_mitossv = SegmentationDataset("mi", working_dir=global_params.config.working_dir)
    cached_mito_ids = sd_mitossv.ids
    cached_mito_mesh_bb = sd_mitossv.load_numpy_data("mesh_bb")
    cached_mito_rep_coords = sd_mitossv.load_numpy_data("rep_coord")
    cached_mito_volumes = sd_mitossv.load_numpy_data("size")
    amount_postcts = len(non_MSN_fullcts)
    p = pool.Pool()
    mito_results = []
    non_MSN_cellids = msn_input_results_dict["cellids"]
    for mi in range(len(non_MSN_fullcts)):
        cellids = non_MSN_cellids_cts[mi]
        cellids = cellids[np.in1d(cellids, non_MSN_cellids)]
        full_cell_dict =non_MSN_celldicts[mi]
        mito_results_cell = p.map(partial(get_organell_volume_density, cached_so_ids = cached_mito_ids,
                                                               cached_so_rep_coord = cached_mito_rep_coords,
                                          cached_so_volume = cached_mito_volumes, full_cell_dict = full_cell_dict,
                                          skeleton_loaded = False, k = 3, min_comp_len = cl), tqdm(cellids))
        mito_results.append(mito_results_cell)
    mito_results = np.concatenate(mito_results)
    msn_input_results_dict["axon mitochondria volume density"] = mito_results[:, 2]
    msn_input_results_dict["dendrite mitochondria volume density"] = mito_results[:, 3]
    gp_inds = np.in1d(msn_input_results_dict["cellids"], GP_ids)
    new_gp_ids = msn_input_results_dict["cellids"][gp_inds]
    new_ax_mitos = msn_input_results_dict["axon mitochondria volume density"][gp_inds]
    new_pred_cts = msn_input_results_dict["predicted celltype"][gp_inds]
    mito_inds = new_ax_mitos >= mito_gp_threshold
    new_gp_ids = new_gp_ids[mito_inds]
    new_ax_mitos = new_ax_mitos[mito_inds]
    new_pred_cts = new_pred_cts[mito_inds]
    write_obj2pkl("%s/synsize_mito_gp_ids.pkl" % f_name, new_gp_ids)

    key_list = list(msn_input_results_dict.keys())
    key_list.remove("cellids")
    key_list.remove("predicted celltype")
    results_df = pd.DataFrame(msn_input_results_dict)
    combinations = list(itertools.combinations(range(len(key_list)), 2))
    palette = {ct_dict[0]: "#707070", ct_dict[5]:"#707070", ct_dict[6]:"#592A87", ct_dict[7]:"#2AC644", ct_dict[8]:"#707070", ct_dict[9]: "#707070", ct_dict[10]: "#707070"}
    for comb in combinations:
        x = key_list[comb[0]]
        y = key_list[comb[1]]
        g = sns.JointGrid(data=results_df, x=x, y=y)
        g.plot_joint(sns.scatterplot)
        g.plot_marginals(sns.histplot, fill=True, alpha=0.3,
                         kde=False, bins=10)
        g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), fontsize=20)
        g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), fontsize=20)
        if "synapse" in x:
            plt.xlabel("%s" % x)
        elif "volume density" in x:
            plt.xlabel("%s in µm³/µm" % x)
        if "synapse" in x:
            plt.ylabel("%s" % x)
        elif "volume density" in y:
            plt.ylabel("%s in µm³/µm" % y)

        plt.savefig("%s/%s_%s_joinplot.svg" % (f_name, x, y))
        plt.close()
        g = sns.JointGrid(data=results_df, x=x, y=y, hue = "predicted celltype", palette = palette)
        g.plot_joint(sns.scatterplot)
        g.plot_marginals(sns.histplot, fill=True, alpha=0.3,
                         kde=False, bins=10, palette = palette)
        plt.legend()
        g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), fontsize=20)
        g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), fontsize=20)
        if "synapse" in x:
            plt.xlabel("%s" % x)
        elif "volume density" in x:
            plt.xlabel("%s in µm³/µm" % x)
        if "synapse" in x:
            plt.ylabel("%s" % x)
        elif "volume density" in y:
            plt.ylabel("%s in µm³/µm" % y)

        plt.savefig("%s/%s_%s_joinplot_overlay.svg" % (f_name, x, y))
        plt.close()
    #plot
    time_stamps = [time.time()]
    step_idents = ["GP identification based on MSN input finished, sum size threshold = %f, mito threshold = %f" % (sumsize_threshold, mito_gp_threshold)]

    log.info("Step 2/9: Seperate GPe and GPi based on mitochondrial volume density, axon median radius and axon mylein fraction")
    #get mylein fraction
    myelin_results = p.map(partial(get_myelin_fraction, min_comp_len = cl, load_skeleton = True), tqdm(new_gp_ids))
    myelin_results = np.array(myelin_results)
    abs_myelin = myelin_results[:, 0]
    rel_myelin = myelin_results[:, 1]
    axon_median_radius = np.zeros(len(new_gp_ids))
    for ig, gpid in enumerate(new_gp_ids):
        gp = SuperSegmentationObject(gpid)
        gp.load_skeleton()
        axon_inds = np.nonzero(gp.skeleton["axoness_avg10000"] == 1)[0]
        axon_radii_cell = get_compartment_radii(gp.id, cell = gp, comp_inds=axon_inds, load_skeleton=False)
        ax_median_radius_cell = np.median(axon_radii_cell)
        axon_median_radius[ig] = ax_median_radius_cell

    gp_results_dict = {"axon median radius": axon_median_radius,
     "axon mitochondria volume density": new_ax_mitos,
     "axon myelin fraction": rel_myelin, "cellids": new_gp_ids, "predicted celltype": new_pred_cts}
    GP_results_df = pd.DataFrame(gp_results_dict)
    GP_results_df.to_csv("%s/GP_myelin_mito_results.csv" % f_name)
    write_obj2pkl("%s/GP_mylein_mito_dict.pkl" % f_name, gp_results_dict)
    key_list = list(gp_results_dict.keys())
    key_list.remove("cellids")
    key_list.remove("predicted celltype")
    combinations = list(itertools.combinations(range(len(key_list)), 2))
    for comb in combinations:
        x = key_list[comb[0]]
        y = key_list[comb[1]]
        g = sns.JointGrid(data=GP_results_df, x=x, y=y)
        g.plot_joint(sns.scatterplot)
        g.plot_marginals(sns.histplot, fill=True, alpha=0.3,
                         kde=False, bins=10)
        plt.legend()
        g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), fontsize=20)
        g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), fontsize=20)
        if "radius" in x:
            plt.xlabel("%s in µm" % x)
        elif "volume density" in x:
            plt.xlabel("%s in µm³/µm" % x)

        if "radius" in y:
            plt.ylabel("%s in µm" % y)
        elif "volume density" in y:
            plt.ylabel("%s in µm³/µm" % y)

        plt.savefig("%s/%s_%s_joinplot_gp.svg" % (f_name, x, y))
        plt.close()
        g = sns.JointGrid(data=GP_results_df, x=x, y=y, hue="predicted celltype", palette=palette)
        g.plot_joint(sns.scatterplot)
        g.plot_marginals(sns.histplot, fill=True, alpha=0.3,
                         kde=False, bins=10, palette=palette)
        plt.legend()
        g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), fontsize=20)
        g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), fontsize=20)
        if "radius" in x:
            plt.xlabel("%s in µm" % x)
        elif "volume density" in x:
            plt.xlabel("%s in µm³/µm" % x)

        if "radius" in y:
            plt.ylabel("%s in µm" % y)
        elif "volume density" in y:
            plt.ylabel("%s in µm³/µm" % y)

        plt.savefig("%s/%s_%s_joinplot_overlay_gp.svg" % (f_name, x, y))
        plt.close()
    threshold_myelin = 0.05
    gpe_myelin_inds = gp_results_dict["axon myelin fraction"] < threshold_myelin
    gpi_myelin_inds = gp_results_dict["axon myelin fraction"] >= threshold_myelin
    GPe_ids = gp_results_dict["cellids"][gpe_myelin_inds]
    GPi_ids = gp_results_dict["cellids"][gpi_myelin_inds]
    write_obj2pkl("%s/gpe_ids.pkl" % f_name, GPe_ids)
    write_obj2pkl("%s/gpe_ids.pkl" % f_name, GPi_ids)
    write_obj2pkl("wholebrain/scratch/arother//j0251v4_prep/conn_morph_gpe_ids.pkl", GPe_ids)
    write_obj2pkl("wholebrain/scratch/arother/j0251v4_prep/conn_morph_gpi_ids.pkl", GPi_ids)
    time_stamps = [time.time()]
    step_idents = ["GP identification based on MSN input finished, threshold myelin fraction = %f" % (threshold_myelin)]

    #from here on similar to GP_i_comparison_connectivity

    log.info("Step 3/9: GPe/i compartment comparison")
    # calculate parameters such as axon/dendrite length, volume, tortuosity and compare within celltypes
    result_GPe_filename = axon_den_arborization_ct(ssd, celltype=6, filename=f_name, cellids=GPe_ids, full_cells=True,
                                                   min_comp_len=cl)
    result_GPi_filename = axon_den_arborization_ct(ssd, celltype=7, filename=f_name, cellids=GPi_ids, full_cells=True,
                                                   min_comp_len=cl)
    compare_compartment_volume_ct(celltype1=6, celltype2=7, filename=f_name, filename1=result_GPe_filename,
                                  filename2=result_GPi_filename, percentile=None, min_comp_len=cl)

    time_stamps = [time.time()]
    step_idents = ["compartment comparison finished"]

    log.info("Step 4/9: GPe and GPi connectivity")
    # see how GPe and GPi are connected
    GPe_GPi_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=6, celltype2=7, filename=f_name,
                                                              cellids1=GPe_ids, cellids2=GPi_ids, full_cells=True,
                                                              min_comp_len=cl, syn_prob_thresh=syn_prob)
    GPe_i_sum_synapses = compare_connectivity(comp_ct1=6, comp_ct2=7, filename=f_name,
                                              foldername_ct1=GPe_GPi_connectivity_resultsfolder,
                                              foldername_ct2=GPe_GPi_connectivity_resultsfolder,
                                              min_comp_len=cl)

    time_stamps = [time.time()]
    step_idents = ["connctivity among GPe/i finished"]

    MSN_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_MSN_arr.pkl")

    log.info("Step 5/9: GPe/i - MSN connectivity")
    # see how GPe and GPi are connected to STN
    GPe_MSN_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=6, celltype2=2, filename=f_name,
                                                              full_cells=True, cellids1=GPe_ids, cellids2=MSN_ids,
                                                              min_comp_len=cl, syn_prob_thresh=syn_prob)
    GPi_MSN_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=7, celltype2=2, filename=f_name,
                                                              full_cells=True, cellids1=GPi_ids, cellids2=MSN_ids,
                                                              min_comp_len=cl, syn_prob_thresh=syn_prob)
    GPe_i_MSN_sum_synapses = compare_connectivity(comp_ct1=6, comp_ct2=7, connected_ct=2, filename=f_name,
                                                  foldername_ct1=GPe_MSN_connectivity_resultsfolder,
                                                  foldername_ct2=GPi_MSN_connectivity_resultsfolder,
                                                  min_comp_len=cl)

    time_stamps = [time.time()]
    step_idents = ["connctivity GPe/i - MSN finished"]

    STN_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_STN_arr.pkl")

    log.info("Step 6/9: GPe/i - STN connectivity")
    # see how GPe and GPi are connected to STN
    GPe_STN_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=6, celltype2=0, filename=f_name,
                                                              full_cells=True, cellids1=GPe_ids, cellids2=STN_ids,
                                                              min_comp_len=cl, syn_prob_thresh=syn_prob)
    GPi_STN_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=7, celltype2=0, filename=f_name,
                                                              full_cells=True, cellids1=GPi_ids, cellids2=STN_ids,
                                                              min_comp_len=cl, syn_prob_thresh=syn_prob)
    GPe_i_STN_sum_synapses = compare_connectivity(comp_ct1=6, comp_ct2=7, connected_ct=0, filename=f_name,
                                                  foldername_ct1=GPe_STN_connectivity_resultsfolder,
                                                  foldername_ct2=GPi_STN_connectivity_resultsfolder,
                                                  min_comp_len=cl)

    time_stamps = [time.time()]
    step_idents = ["connctivity GPe/i - STN finished"]

    FS_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_FS_arr.pkl")

    log.info("Step 7/9: GPe/i - FS connectivity")
    # see how GPe and GPi are connected to FS
    GPe_FS_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=6, celltype2=8, filename=f_name,
                                                             full_cells=True, cellids1=GPe_ids, cellids2=FS_ids,
                                                             syn_prob_thresh=syn_prob, min_comp_len = cl)
    GPi_FS_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=7, celltype2=8, filename=f_name,
                                                             full_cells=True, cellids1=GPi_ids, cellids2=FS_ids,
                                                             syn_prob_thresh=syn_prob)
    GPe_i_FS_sum_synapses = compare_connectivity(comp_ct1=6, comp_ct2=7, connected_ct=8, filename=f_name,
                                                 foldername_ct1=GPe_FS_connectivity_resultsfolder,
                                                 foldername_ct2=GPi_FS_connectivity_resultsfolder, min_comp_len = cl)

    time_stamps = [time.time()]
    step_idents = ["connctivity GPe/i - FS finished"]

    TAN_ids = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/full_TAN_arr.pkl")

    log.info("Step 8/9: GPe/i - TAN connectivity")
    # see how GPe and GPi are connected to TAN
    GPe_TAN_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=6, celltype2=5, filename=f_name,
                                                              full_cells=True, cellids1=GPe_ids, cellids2=TAN_ids,
                                                              syn_prob_thresh=syn_prob, min_comp_len = cl)
    GPi_TAN_connectivity_resultsfolder = synapses_between2cts(sd_synssv, celltype1=7, celltype2=5, filename=f_name,
                                                              full_cells=True, cellids1=GPi_ids, cellids2=TAN_ids,
                                                              syn_prob_thresh=syn_prob, min_comp_len = cl)
    GPe_i_TAN_sum_synapses = compare_connectivity(comp_ct1=6, comp_ct2=7, connected_ct=5, filename=f_name,
                                                  foldername_ct1=GPe_TAN_connectivity_resultsfolder,
                                                  foldername_ct2=GPi_TAN_connectivity_resultsfolder, min_comp_len = cl)

    time_stamps = [time.time()]
    step_idents = ["connctivity GPe/i - TAN finished"]

    log.info("Step 9/9 Overview Graph")
    # make connectivity overview graph with networkx
    # first put all dictionaries together
    sum_synapse_dict = {**GPe_i_sum_synapses, **GPe_i_STN_sum_synapses, **GPe_i_FS_sum_synapses,
                        **GPe_i_TAN_sum_synapses}
    write_obj2pkl("%s/ct_sum_synapses.pkl" % f_name, sum_synapse_dict)
    # plot
    plot_nx_graph(sum_synapse_dict, filename=("%s/summed_synapses_nx_overview_mcl%i.png" % (f_name, cl)),
                  title="sum of synapses between celltypes")

    msn_summed_synapse_pd = pd.DataFrame(sum_synapse_dict, index=[0])
    msn_summed_synapse_pd.to_csv("%s/ct_summed_synapses.csv" % f_name)

    log.info("GPe/i compartment and connectivity analysis finished")
    step_idents = ["GPe/i compartment and connectivity analysis finished"]
