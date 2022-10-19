#get an estimate about the average synapse density per celltype per axon

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.result_helper import ComparingMultipleForPLotting
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    import os as os
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    import numpy as np
    from tqdm import tqdm

    global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    start = time.time()
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    min_comp_len = 200
    max_MSN_path_len = 7500
    min_syn_size = 0.1
    syn_prob = 0.8
    cls = CelltypeColors()
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw'}
    color_key = 'BlYw'
    f_name = "cajal/nvmescratch/users/arother/bio_analysis_results/general/221014_j0251v4_avg_syn_den_sb_%.2f_mcl_%i_%s" % (
        syn_prob, min_comp_len, color_key)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('get average synapse density of axon per celltype', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i, max_MSN_path_len = %i, syn_prob = %.2f, min_syn_size = %.i, colors = %s" % (
            min_comp_len, max_MSN_path_len, syn_prob, min_syn_size, color_key))
    time_stamps = [time.time()]
    step_idents = ['t-0']
    known_mergers = load_pkl2obj("cajal/nvmescratch/users/arother/j0251v4_prep/merger_arr.pkl")
    log.info("Iterate over celltypes to get eachs estimate of synapse density")
    cts = list(ct_dict.keys())
    ax_ct = [1, 3, 4]
    res_ct_dict = {i: {} for i in ct_dict}
    rows = ["mean synapse density [1/µm]", "median synapse density [1/µm]", "mean synapse size density [µm²/µm]", "median synapse size density [µm²/µm]", "mean distance between synapses [µm]", "median distance between synapses [µm]"]
    mean_result_df = pd.DataFrame(columns=cts, index= rows)
    for i, ct in enumerate(tqdm(cts)):
        log.info("Start getting random samples from celltype %s, %i/%i" % (ct_dict[ct], i, len(cts)))
        # only get cells with min_comp_len, MSN with max_comp_len or axons with min ax_len
        if ct in ax_ct:
            cell_dict = load_pkl2obj(
                "/wholebrain/scratch/arother/j0251v4_prep/ax_%.3s_dict.pkl" % (ct_dict[ct]))
            cellids = np.array(list(cell_dict.keys()))
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                            axon_only=True, max_path_len=None)
        else:
            cell_dict = load_pkl2obj(
                "/wholebrain/scratch/arother/j0251v4_prep/full_%.3s_dict.pkl" % (ct_dict[ct]))
            cellids = load_pkl2obj(
                "/wholebrain/scratch/arother/j0251v4_prep/full_%.3s_arr.pkl" % ct_dict[ct])
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            if ct == 2:
                misclassified_asto_ids = load_pkl2obj('cajal/nvmescratch/users/arother/j0251v4_prep/pot_astro_ids.pkl')
                astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
                cellids = cellids[astro_inds]
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                axon_only=False, max_path_len=None)
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict[ct]))
        log.info("Get axon pathlength per cell %s" % ct_dict[ct])
        axon_pathlength = np.zeros(len(cellids))
        for i, cellid in enumerate(cellids):
            axon_pathlength[i] = cell_dict[cellid]["axon length"]
        log.info("Get number of synapses per cell %s" % ct_dict[ct])
        #filter synapse caches for synapses with only synapse sof celltypes
        m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(sd_synssv,
                                                                                                            pre_cts=[
                                                                                                                ct],
                                                                                                            syn_prob_thresh=syn_prob,
                                                                                                            min_syn_size=min_syn_size,
                                                                                                            axo_den_so=True)
        # only those from cellids
        ct_inds = np.any(np.in1d(m_ssv_partners, cellids).reshape(len(m_ssv_partners), 2), axis=1)
        m_cts = m_cts[ct_inds]
        m_ids = m_ids[ct_inds]
        m_ssv_partners = m_ssv_partners[ct_inds]
        m_sizes = m_sizes[ct_inds]
        m_axs = m_axs[ct_inds]
        #filter synapses to get ones where cellids is axon
        testct = np.in1d(m_ssv_partners, cellids).reshape(len(m_ssv_partners), 2)
        testax = np.in1d(m_axs, 1).reshape(len(m_cts), 2)
        pre_ct_inds = np.any(testct == testax, axis=1)
        m_cts = m_cts[pre_ct_inds]
        m_ids = m_ids[pre_ct_inds]
        m_axs = m_axs[pre_ct_inds]
        m_ssv_partners = m_ssv_partners[pre_ct_inds]
        m_sizes = m_sizes[pre_ct_inds]
        #get number, sum size per cell
        cell_inds = np.where(m_axs == 1)
        ct_ssvsids = m_ssv_partners[cell_inds]
        ct_ssv_inds, unique_ct_ssvs = pd.factorize(ct_ssvsids)
        ct_syn_sumsizes = np.bincount(ct_ssv_inds, m_sizes)
        ct_syn_number = np.bincount(ct_ssv_inds)
        #re-order cellids, unique_ct_ssvs
        sorted_cellids_inds = np.argsort(cellids)
        sorted_cellids = cellids[sorted_cellids_inds]
        sorted_axon_pathlength = axon_pathlength[sorted_cellids_inds]
        sorted_syn_cellids_inds = np.argsort(unique_ct_ssvs)
        sorted_unique_ct_ids = unique_ct_ssvs[sorted_syn_cellids_inds]
        sorted_syn_sumsizes = ct_syn_sumsizes[sorted_syn_cellids_inds]
        sorted_syn_numbers = ct_syn_number[sorted_syn_cellids_inds]
        if len(sorted_cellids) != len(sorted_unique_ct_ids):
            syn_inds = np.in1d(sorted_cellids, sorted_unique_ct_ids)
            sorted_cellids = sorted_cellids[syn_inds]
            sorted_axon_pathlength = sorted_axon_pathlength[syn_inds]
        #get synapse density
        syn_size_density = sorted_syn_sumsizes / sorted_axon_pathlength
        syn_num_density = sorted_syn_numbers / sorted_axon_pathlength
        dist_between_syns = sorted_axon_pathlength / sorted_syn_numbers
        res_ct_dict[ct]["synapse size density"] = syn_size_density
        res_ct_dict[ct]["synapse density"] = syn_num_density
        res_ct_dict[ct]["cellids"] = sorted_cellids
        res_ct_dict[ct]["mean distance between synapses"] = dist_between_syns
        rows = ["mean synapse density [1/µm]", "median synapse density [1/µm]", "mean synapse size density [µm²/µm]",
                "median synapse size density [µm²/µm]", "mean distance between synapses [µm]",
                "median distance between synapses [µm]"]
        mean_result_df.loc["mean synapse density [1/µm]", ct] = np.mean(syn_num_density)
        mean_result_df.loc["median synapse density [1/µm]", ct] = np.median(syn_num_density)
        mean_result_df.loc["mean synapse size density [µm²/µm]", ct] = np.mean(syn_size_density)
        mean_result_df.loc["median synapse size density [µm²/µm]", ct] = np.median(syn_size_density)
        mean_result_df.loc["mean distance between synapses [µm]", ct] = np.mean(dist_between_syns)
        mean_result_df.loc["median distance between synapses [µm]", ct] = np.median(dist_between_syns)
        log.info("Calculated average syn density of %s" % ct_dict[ct])
        step_idents = ["%s" % ct_dict[ct]]

    mean_result_df.to_csv("%s/mean_result_parameters.csv" % f_name)
    write_obj2pkl("%s/syn_results_dict.pkl"  % f_name, res_ct_dict)

    log.info("Plot results")
    #make violinplot for values of all cells
    ct_dict_list = [res_ct_dict[ct] for ct in res_ct_dict]
    ct_colours = cls.colors[color_key]
    multiple_ct_for_plotting = ComparingMultipleForPLotting(ct_list = cts, dictionary_list = ct_dict_list, colour_list = ct_colours, filename = f_name)
    for key in res_ct_dict[ct].keys():
        result_df = multiple_ct_for_plotting.result_df_per_param(key)
        multiple_ct_for_plotting.plot_violin(key = key, result_df = result_df, subcell = "synapse", x=None, stripplot = True, outgoing = False)
        multiple_ct_for_plotting.plot_box(key=key, result_df=result_df, subcell="synapse", x=None, stripplot=True,
                                             outgoing=False)


    log.info("Analysis finished")