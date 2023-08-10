# get mitos, axon median radius, myelinasation of GPe/i and plot against each other

if __name__ == '__main__':
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    from syconn.reps.segmentation import SegmentationDataset
    import os as os
    import pandas as pd
    import numpy as np
    from scipy.stats import ranksums
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct, get_per_cell_mito_myelin_info, get_cell_soma_radius
    from syconn.handler.basics import write_obj2pkl
    from cajal.nvmescratch.users.arother.bio_analysis.general.result_helper import  ComparingResultsForPLotting
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import itertools
    import seaborn as sns
    import matplotlib.pyplot as plt
    from syconn.mp.mp_utils import start_multiprocess_imap

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    start = time.time()
    bio_params = Analysis_Params(working_dir = global_params.wd, version = 'v5')
    ct_dict = bio_params.ct_dict()
    min_comp_len = 200
    syn_prob = bio_params.syn_prob_thresh()
    min_syn_size = bio_params.min_syn_size()
    fontsize_jointplot = 10
    use_skel = False  # if true would use skeleton labels for getting soma; vertex labels more exact, also probably faster
    f_name = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/230810_j0251v5_GPe_i_myelin_mito_radius_mcl%i_newcolors_fs%i" % (min_comp_len, fontsize_jointplot)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('GPe, GPi comparison connectivity', log_dir=f_name + '/logs/')
    if use_skel:
        log.info('use skeleton node predictions to get soma mesh coordinates')
    else:
        log.info('use vertex label dict predictions to get soma vertices')
    log.info("GPe/i comparison starts")
    time_stamps = [time.time()]
    step_idents = ['t-0']
    #ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   #10: "NGF"}
    known_mergers = bio_params.load_known_mergers()
    GPi_full_cell_dict = bio_params.load_cell_dict(7)
    GPi_ids = np.array(list(GPi_full_cell_dict.keys()))
    merger_inds = np.in1d(GPi_ids, known_mergers) == False
    GPi_ids = GPi_ids[merger_inds]
    GPi_ids = check_comp_lengths_ct(cellids=GPi_ids, fullcelldict=GPi_full_cell_dict, min_comp_len=min_comp_len,
                                            axon_only=False,
                                            max_path_len=None)
    GPe_full_cell_dict = bio_params.load_cell_dict(6)
    GPe_ids = np.array(list(GPe_full_cell_dict.keys()))
    merger_inds = np.in1d(GPe_ids, known_mergers) == False
    GPe_ids = GPe_ids[merger_inds]
    GPe_ids = check_comp_lengths_ct(cellids=GPe_ids, fullcelldict=GPe_full_cell_dict, min_comp_len=min_comp_len,
                                    axon_only=False,
                                    max_path_len=None)

    axon_median_radius_gpi = np.zeros(len(GPi_ids))
    axon_mito_volume_density_gpi = np.zeros(len(GPi_ids))
    axon_myelin_gpi = np.zeros(len(GPi_ids))
    sd_mitossv = SegmentationDataset("mi", working_dir=global_params.config.working_dir)
    cached_mito_ids = sd_mitossv.ids
    cached_mito_mesh_bb = sd_mitossv.load_numpy_data("mesh_bb")
    cached_mito_rep_coords = sd_mitossv.load_numpy_data("rep_coord")
    cached_mito_volumes = sd_mitossv.load_numpy_data("size")


    log.info("Step 1/3: Get information from GPe")
    gpe_input = [[gpe_id, min_comp_len, cached_mito_ids, cached_mito_rep_coords, cached_mito_volumes, GPe_full_cell_dict] for gpe_id in GPe_ids]
    gpe_output = start_multiprocess_imap(get_per_cell_mito_myelin_info, gpe_input)
    gpe_output = np.array(gpe_output)
    #[ax_median_radius_cell, axo_mito_volume_density_cell, rel_myelin_cell]
    axon_median_radius_gpe = gpe_output[:, 0]
    axon_mito_volume_density_gpe = gpe_output[:, 1]
    axon_myelin_gpe = gpe_output[:, 2]

    gpe_nonzero = axon_median_radius_gpe > 0
    # get soma diameter from all GPe
    gpe_soma_results = start_multiprocess_imap(get_cell_soma_radius, GPe_ids)
    gpe_soma_results = np.array(gpe_soma_results, dtype='object')
    gpe_diameters = gpe_soma_results[:, 1].astype(float) * 2
    GPe_params = {"axon median radius": axon_median_radius_gpe[gpe_nonzero], "axon mitochondria volume density": axon_mito_volume_density_gpe[gpe_nonzero],
                  "axon myelin fraction": axon_myelin_gpe[gpe_nonzero], 'soma diameter': gpe_diameters[gpe_nonzero], "cellids": GPe_ids[gpe_nonzero]}
    GPe_param_df = pd.DataFrame(GPe_params)
    GPe_param_df.to_csv("%s/GPe_params.csv" % f_name)
    write_obj2pkl("%s/GPe_dict.pkl" % f_name, GPe_params)

    time_stamps = [time.time()]
    step_idents = ["GPe axon parameters collected"]

    log.info("Step 2/3: Get information from GPi")
    gpi_input = [
        [gpi_id, min_comp_len, cached_mito_ids, cached_mito_rep_coords, cached_mito_volumes, GPi_full_cell_dict] for
        gpi_id in GPi_ids]
    gpi_output = start_multiprocess_imap(get_per_cell_mito_myelin_info, gpi_input)
    gpi_output = np.array(gpi_output)
    # [ax_median_radius_cell, axo_mito_volume_density_cell, rel_myelin_cell]
    axon_median_radius_gpi = gpi_output[:, 0]
    axon_mito_volume_density_gpi = gpi_output[:, 1]
    axon_myelin_gpi = gpi_output[:, 2]

    gpi_nonzero = axon_median_radius_gpi > 0
    # get soma diameter from all GPi
    gpi_soma_results = start_multiprocess_imap(get_cell_soma_radius, GPi_ids)
    gpi_soma_results = np.array(gpi_soma_results, dtype='object')
    gpi_diameters = gpi_soma_results[:, 1].astype(float) * 2
    GPi_params = {"axon median radius": axon_median_radius_gpi[gpi_nonzero],
                  "axon mitochondria volume density": axon_mito_volume_density_gpi[gpi_nonzero],
                  "axon myelin fraction": axon_myelin_gpi[gpi_nonzero],
                  'soma diameter': gpi_diameters[gpi_nonzero], "cellids": GPi_ids[gpi_nonzero]}
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
        elif 'soma' in key:
            subcell = 'soma'
        else:
            subcell = "axon"
        results_comparison.plot_violin(key, results_for_plotting, subcell=subcell, stripplot=True)
        results_comparison.plot_box(key, results_for_plotting, subcell=subcell, stripplot=False)
        results_comparison.plot_hist_comparison(key, results_for_plotting, subcell=subcell, bins=10, norm_hist=False)
        results_comparison.plot_hist_comparison(key, results_for_plotting, subcell=subcell, bins=10, norm_hist=True)
        all_param_df.loc[0: GPe_len - 1, key] = GPe_params[key]
        all_param_df.loc[GPe_len: sum_length - 1, key] = GPi_params[key]

    ranksum_results.to_csv("%s/ranksum_results.csv" % f_name)
    all_param_df.to_csv("%s/GPe_GPi_params.csv" % f_name)

    combinations = list(itertools.combinations(range(len(key_list)), 2))
    #sns.set(font_scale=1.5)
    for comb in combinations:
        x = key_list[comb[0]]
        y = key_list[comb[1]]
        g = sns.JointGrid(data= all_param_df, x = x, y = y, hue = "celltype", palette = results_comparison.color_palette)
        g.plot_joint(sns.scatterplot)
        g.plot_marginals(sns.histplot,  fill = True, alpha = 0.3,
                         kde=False, bins=10, palette = results_comparison.color_palette)
        g.ax_joint.set_xticks(g.ax_joint.get_xticks())
        g.ax_joint.set_yticks(g.ax_joint.get_yticks())
        if g.ax_joint.get_xticks()[0] < 0:
            g.ax_marg_x.set_xlim(0)
        if g.ax_joint.get_yticks()[0] < 0:
            g.ax_marg_y.set_ylim(0)
        g.ax_joint.set_xticklabels(["%.2f" % i for i in g.ax_joint.get_xticks()], fontsize = fontsize_jointplot)
        g.ax_joint.set_yticklabels(["%.2f" % i for i in g.ax_joint.get_yticks()], fontsize= fontsize_jointplot)
        if "radius" in x or 'diameter' in x:
            g.ax_joint.set_xlabel("%s [µm]" % x)
        elif "volume density" in x:
            g.ax_joint.set_xlabel("%s [µm³/µm]" % x)

        if "radius" in y or 'diameter' in y:
            g.ax_joint.set_ylabel("%s [µm]" % y)
        elif "volume density" in y:
            g.ax_joint.set_xlabel("%s [µm³/µm]" % y)

        plt.savefig("%s/%s_%s_joinplot.svg" % (f_name, x, y))
        plt.savefig("%s/%s_%s_joinplot.png" % (f_name, x, y))
        plt.close()









