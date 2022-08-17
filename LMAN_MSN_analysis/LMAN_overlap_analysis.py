
if __name__ == '__main__':
    from wholebrain.scratch.arother.bio_analysis.general.analysis_morph_helper import remove_myelinated_part_axon, compute_overlap_skeleton
    from wholebrain.scratch.arother.bio_analysis.general.result_helper import ComparingResultsForPLotting

    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    import os as os
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    import numpy as np
    from tqdm import tqdm
    from multiprocessing import pool
    from functools import partial
    from scipy.stats import ranksums
    import matplotlib.pyplot as plt
    import seaborn as sns


    global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
    start = time.time()
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    overlap_threshold = 0.75
    non_overlap_threshold = 0.001
    kdtree_radius = 50 #µm
    f_name = "wholebrain/scratch/arother/bio_analysis_results/LMAN_MSN_analysis/220808_j0251v4_LMAN_overlap_analysis_ot_%.2f_not_%.3f_kdtr_%i" % (
    overlap_threshold, non_overlap_threshold, kdtree_radius)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('LMAN overlap analysis', log_dir=f_name + '/logs/')
    log.info("overlap threshold = %.2f, non-overlap threshold = %.3f, kdtree radius = %i" % (overlap_threshold, non_overlap_threshold, kdtree_radius))
    time_stamps = [time.time()]
    step_idents = ['t-0']

    #anaylsis to see if overlapping LMAN axons target same MSN cells
    #also if there are differences between overlapping and non-overlapping lman

    #1st part of analysis: divide LMAN axons in overlapping and non-overlapping
    log.info("Step 1/3: Divide LMAN in overlapping and non-overlapping")
    LMAN_dict = load_pkl2obj(
        "/wholebrain/scratch/arother/j0251v4_prep/ax_LMA_dict.pkl")
    LMAN_ids = load_pkl2obj("/wholebrain/scratch/arother/j0251v4_prep/LMAN_handpicked_arr.pkl")
    #iterate over lmans to remove myelinated parts of axon
    p = pool.Pool()
    node_positions = p.map(remove_myelinated_part_axon, tqdm(LMAN_ids))
    #get overlapp between all LMAN axons, from both sides
    #calculate overlap of one axon with all others with kdtree of nodes
    overlap_cells = p.map(partial(compute_overlap_skeleton, sso_ids = LMAN_ids, all_node_positions = node_positions, kdtree_radius = kdtree_radius), tqdm(LMAN_ids))
    overlap_cells = np.array(overlap_cells)
    #get pairs of overlapping cells
    overlap_candidates = []
    non_overlap_candidates = []
    for i, lman_id in enumerate(LMAN_ids):
        overlap = overlap_cells[i]
        overlap_inds = np.nonzero(overlap > overlap_threshold)
        non_overlap_inds = np.nonzero(overlap < non_overlap_threshold)
        overlap_candidates.append(overlap_inds)
        non_overlap_candidates.append(non_overlap_inds)
    overlap_pairs = []
    non_overlap_pairs = []
    for i, lman_id in enumerate(LMAN_ids):
        overlap_can = overlap_candidates[i][0]
        overlap_can = overlap_can[overlap_can !=  i]
        non_overlap_can = non_overlap_candidates[i][0]
        if len(overlap_can) > 0:
            for ov_id in overlap_can:
                overlap_cand_2 = overlap_candidates[ov_id][0]
                if i in overlap_cand_2:
                    if ov_id < i:
                        continue
                    overlap_id = LMAN_ids[ov_id]
                    overlap_pairs.append([lman_id, overlap_id])
        if len(non_overlap_can) > 0:
            for nov_id in non_overlap_can:
                noverlap_cand_2 = non_overlap_candidates[nov_id][0]
                if i in noverlap_cand_2:
                    if nov_id < i:
                        continue
                    noverlap_id = LMAN_ids[nov_id]
                    non_overlap_pairs.append([lman_id, noverlap_id])

    log.info("%i overlap pairs were found" % len(overlap_pairs))
    log.info("%i non-overlap pairs were found" % len(non_overlap_pairs))
    time_stamps = [time.time()]
    step_idents = ['t-0']

    #2nd part of analysis: compare number of same MSNs between overlapping LMAN
    #also do this for non-overlapping LMAN
    log.info("Step 2/3: Compare MSN of overlapping and non-overlapping MSN")
    syn_prob = 0.8
    mcl = 200
    #load connectivity dicts
    LMAN_proj_dict = load_pkl2obj("wholebrain/scratch/arother/bio_analysis_results/LMAN_MSN_analysis/220809_j0251v4_LMAN_MSN_GP_est_mcl_%i_synprob_%.2f/lman_dict_percell.pkl" % (
    mcl, syn_prob))
    log.info("LMAN proj dict was loaded with min comp len = %i, syn_prob = %.2f" % (mcl, syn_prob))
    #comapare overlapping LMAN
    shared_msn_perc_overlap = np.zeros(len(overlap_pairs))
    shared_msn_number_overlap = np.zeros(len(overlap_pairs))
    shared_gpi_perc_overlap = np.zeros(len(overlap_pairs))
    shared_gpi_number_overlap = np.zeros(len(overlap_pairs))
    for li, lman_pair in enumerate(tqdm(overlap_pairs)):
        lman_1 = lman_pair[0]
        lman_2 = lman_pair[1]
        lman_1_msn_ids = LMAN_proj_dict[lman_1]["MSN ids"]
        lman_2_msn_ids = LMAN_proj_dict[lman_2]["MSN ids"]
        shared_msn = lman_1_msn_ids[np.in1d(lman_1_msn_ids, lman_2_msn_ids)]
        perc_shared = len(shared_msn)/ (len(lman_1_msn_ids) + len(lman_2_msn_ids)/ 2)
        shared_msn_perc_overlap[li] = perc_shared
        shared_msn_number_overlap[li] = len(shared_msn)
        lman_1_gpi_ids = LMAN_proj_dict[lman_1]["indirect GPi ids"]
        lman_2_gpi_ids = LMAN_proj_dict[lman_2]["indirect GPi ids"]
        shared_gpi = lman_1_gpi_ids[np.in1d(lman_1_gpi_ids, lman_2_gpi_ids)]
        perc_gpi_shared = len(shared_gpi) / (len(lman_1_gpi_ids) + len(lman_2_gpi_ids) / 2)
        shared_gpi_perc_overlap[li] = perc_gpi_shared
        shared_gpi_number_overlap[li] = len(shared_gpi)

    #compare non-overlapping MSN, GPi
    shared_msn_perc_non = np.zeros(len(non_overlap_pairs))
    shared_msn_number_non = np.zeros(len(non_overlap_pairs))
    shared_gpi_perc_non = np.zeros(len(non_overlap_pairs))
    shared_gpi_number_non = np.zeros(len(non_overlap_pairs))
    for li, lman_pair in enumerate(tqdm(non_overlap_pairs)):
        lman_1 = lman_pair[0]
        lman_2 = lman_pair[1]
        lman_1_msn_ids = LMAN_proj_dict[lman_1]["MSN ids"]
        lman_2_msn_ids = LMAN_proj_dict[lman_2]["MSN ids"]
        shared_msn = lman_1_msn_ids[np.in1d(lman_1_msn_ids, lman_2_msn_ids)]
        perc_shared = len(shared_msn) / (len(lman_1_msn_ids) + len(lman_2_msn_ids) / 2)
        shared_msn_perc_non[li] = perc_shared
        shared_msn_number_non[li] = len(shared_msn)
        lman_1_gpi_ids = LMAN_proj_dict[lman_1]["indirect GPi ids"]
        lman_2_gpi_ids = LMAN_proj_dict[lman_2]["indirect GPi ids"]
        shared_gpi = lman_1_gpi_ids[np.in1d(lman_1_gpi_ids, lman_2_gpi_ids)]
        perc_gpi_shared = len(shared_gpi) / (len(lman_1_gpi_ids) + len(lman_2_gpi_ids) / 2)
        shared_gpi_perc_non[li] = perc_gpi_shared
        shared_gpi_number_non[li] = len(shared_gpi)

    overlap_dict = {"overlap pair": overlap_pairs, "percentage of same MSNs": shared_msn_perc_overlap, "number of same MSNs": shared_msn_number_overlap,
                    "percentage of same indirectly targeted GPis": shared_gpi_perc_overlap, "number of same indirectly targeted GPi": shared_gpi_number_overlap}
    overlap_pd = pd.DataFrame(overlap_dict)
    write_obj2pkl("%s/overlap_dict.pkl" % f_name, overlap_dict)
    overlap_pd.to_csv("%s/overlap_dict.csv" % f_name)

    non_overlap_dict = {"non-overlap pair": non_overlap_pairs, "percentage of same MSNs": shared_msn_perc_non,
                    "number of same MSNs": shared_msn_number_non,
                    "percentage of same indirectly targeted GPis": shared_gpi_perc_non, "number of same indirectly targeted GPi": shared_gpi_number_non}
    non_overlap_pd = pd.DataFrame(non_overlap_dict)
    write_obj2pkl("%s/non_overlap_dict.pkl" % f_name, non_overlap_dict)
    non_overlap_pd.to_csv("%s/non_overlap_dict.csv" % f_name)

    log.info("Average percentage of MSN shared between overlapping LMAN = %.2f" % np.mean(shared_msn_perc_overlap))
    log.info("Average number of MSN shared between overlapping LMAN = %.2f" % np.mean(shared_msn_number_overlap))
    log.info("Average percentage of MSN shared between non-overlapping LMAN = %.2f" % np.mean(shared_msn_perc_non))
    log.info("Average number of MSN shared between non-overlapping LMAN = %.2f" % np.mean(shared_msn_number_non))
    log.info("Median percentage of MSN shared between overlapping LMAN = %.2f" % np.median(shared_msn_perc_overlap))
    log.info("Median number of MSN shared between overlapping LMAN = %.2f" % np.median(shared_msn_number_overlap))
    log.info("Median percentage of MSN shared between non-overlapping LMAN = %.2f" % np.median(shared_msn_perc_non))
    log.info("Median number of MSN shared between non-overlapping LMAN = %.2f" % np.median(shared_msn_number_non))
    log.info("Median percentage of GPi shared between overlapping LMAN = %.2f" % np.median(shared_gpi_perc_overlap))
    log.info("Median number of GPi shared between overlapping LMAN = %.2f" % np.median(shared_gpi_number_overlap))
    log.info("Median percentage of GPi shared between non-overlapping LMAN = %.2f" % np.median(shared_gpi_perc_non))
    log.info("Median number of GPi shared between non-overlapping LMAN = %.2f" % np.median(shared_gpi_number_non))
    time_stamps = [time.time()]
    step_idents = ['t-0']

    #To DO
    #also look at soma distance of MSN
    #same via MSN targeted GPi between overlapping LMANs, difference to ones far apart?
    #distance of GPi soma

    log.info("Step 3/3: compute statistics and plot results")
    keys = list(overlap_dict.keys())
    results_for_plotting = ComparingResultsForPLotting(celltype1 = "overlapping pairs", celltype2 = "non-overlapping pairs", dictionary1 = overlap_dict,
                                                       dictionary2 = non_overlap_dict, color1 = '#60A6A6', color2 = '#051A26', filename = f_name)
    ranksum_results = pd.DataFrame(columns=keys[1:], index = ["stats", "p value"])
    for key in keys:
        if "pair" in key:
            continue
        stats, p_value = ranksums(overlap_dict[key], non_overlap_dict[key])
        ranksum_results.loc["stats", key] = stats
        ranksum_results.loc["p value", key] = p_value
        sns.distplot(overlap_pd[key],
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1},
                     kde=False, norm_hist=False, bins=10, label = "overlapping pairs", color='#60A6A6')
        sns.distplot(non_overlap_pd[key],
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1},
                     kde=False, norm_hist=False, bins=10, label = "non-overlapping pairs", color = '#051A26')
        plt.legend()
        plt.xlabel(key)
        plt.ylabel("count of cell pairs")
        plt.savefig("%s/%s_hist_comparison.svg" % (f_name, key))
        plt.savefig("%s/%s_hist_comparison.png" % (f_name, key))
        plt.close()
        sns.distplot(overlap_pd[key],
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1},
                     kde=False, norm_hist=True, bins=10, label="overlapping pairs", color='#60A6A6')
        sns.distplot(non_overlap_pd[key],
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1},
                     kde=False, norm_hist=True, bins=10, label="non-overlapping pairs", color='#051A26')
        plt.legend()
        plt.xlabel(key)
        plt.ylabel("count of cell pairs")
        plt.savefig("%s/%s_hist_comparison_norm.svg" % (f_name, key))
        plt.savefig("%s/%s_hist_comparison_norm.png" % (f_name, key))
        plt.close()
        result_df = results_for_plotting.result_df_per_param(key)
        results_for_plotting.plot_violin(result_df=result_df, key = key, subcell = "cell pairs")

    ranksum_results.to_csv("%s/ranksum_results.csv" % f_name)

    log.info("Comparison of overlapping and non-overlapping LMAN done")


