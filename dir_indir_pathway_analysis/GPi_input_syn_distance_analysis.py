#write script that shows the distance of synaptic inputs to the soma for GPi
#script gets distance of different synaptic inputs to GPi soma and plots differences between different celltypes

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct, get_number_sum_size_synapses
    from cajal.nvmescratch.users.arother.bio_analysis.general.result_helper import ConnMatrix
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from cajal.nvmescratch.users.arother.bio_analysis.dir_indir_pathway_analysis.synapse_input_distance import get_syn_distances
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    import os as os
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    import numpy as np
    from tqdm import tqdm
    from scipy.stats import ranksums
    import scipy
    import seaborn as sns
    import matplotlib.pyplot as plt

    global_params.wd = "ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    start = time.time()
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    min_comp_len = 200
    syn_prob = 0.8
    min_syn_size = 0.1
    msn_ct = 2
    lman_ct = 3
    gpi_ct = 7
    exclude_known_mergers = True
    #color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw'}
    color_key = 'TePkBr'
    f_name = "cajal/nvmescratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/221020_j0251v4_GPi_syn_distances_mcl_%i_synprob_%.2f_%s" % (
    min_comp_len, syn_prob, color_key)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('Analysis of distance to soma for GPi and different synaptic inputs', log_dir=f_name + '/logs/')
    cts_for_loading = [0, 2, 3, 6, 7, 8]
    cts_str_analysis = [ct_dict[ct] for ct in cts_for_loading]
    num_cts = len(cts_for_loading)
    dist2ct = 7
    dist2ct_str = ct_dict[dist2ct]
    log.info(
        "min_comp_len = %i, syn_prob = %.1f, min_syn_size = %.1f, known mergers excluded = %s, colors = %s" % (
        min_comp_len, syn_prob, min_syn_size, exclude_known_mergers, color_key))
    log.info(f'Distance of synapses for celltypes {cts_str_analysis} will be compared to {dist2ct_str}')
    time_stamps = [time.time()]
    step_idents = ['t-0']

    log.info("Step 1/3: Load celltypes and check suitability")

    axon_cts = [1, 3, 4]
    if exclude_known_mergers:
        known_mergers = load_pkl2obj("/cajal/nvmescratch/users/arother/j0251v4_prep/merger_arr.pkl")
        misclassified_asto_ids = load_pkl2obj('cajal/nvmescratch/users/arother/j0251v4_prep/pot_astro_ids.pkl')
    suitable_ids_dict = {}
    for ct in tqdm(cts_for_loading):
        ct_str = ct_dict[ct]
        if ct in axon_cts:
            cell_dict = load_pkl2obj(
            "/cajal/nvmescratch/users/arother/j0251v4_prep/ax_%.3s_dict.pkl" % ct_str)
            #get ids with min compartment length
            cellids = np.array(list(cell_dict.keys()))
            if exclude_known_mergers:
                merger_inds = np.in1d(cellids, known_mergers) == False
                cellids = cellids[merger_inds]
            cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len, axon_only=True,
                              max_path_len=None)
            suitable_ids_dict[ct] = cellids
        else:
            cell_dict = load_pkl2obj(
                "/cajal/nvmescratch/users/arother/j0251v4_prep/full_%.3s_dict.pkl" % ct_str)
            cellids = np.array(list(cell_dict.keys()))
            if exclude_known_mergers:
                merger_inds = np.in1d(cellids, known_mergers) == False
                cellids = cellids[merger_inds]
                if ct == 2:
                    astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
                    cellids = cellids[astro_inds]
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                axon_only=False,
                                                max_path_len=None)
            suitable_ids_dict[ct] = cellids

    number_ids = [len(suitable_ids_dict[ct]) for ct in cts_for_loading]
    log.info(f"Suitable ids from celltypes {cts_str_analysis} were selected: {number_ids}")
    time_stamps = [time.time()]
    step_idents = ['loading cells']

    log.info("Step 2/3: Get synapse distance to soma from different celltypes to %s" % dist2ct_str)
    median_dist_df = pd.DataFrame(columns = cts_str_analysis, index=range(len(suitable_ids_dict[dist2ct])))
    min_dist_df = pd.DataFrame(columns=cts_str_analysis, index=range(len(suitable_ids_dict[dist2ct])))
    max_dist_df = pd.DataFrame(columns=cts_str_analysis, index=range(len(suitable_ids_dict[dist2ct])))
    distances_dict = {}
    for ct in tqdm(cts_for_loading):
        ct_str = ct_dict[ct]
        #get median, min, max synapse distance to soma per cell
        #function uses multiprocessing
        if ct == dist2ct:
            post_ids, median_distances_per_ids, min_distances_per_ids, max_distances_per_ids, syn_numbers, syn_ssv_sizes = get_syn_distances(ct_post = dist2ct, cellids_post = suitable_ids_dict[dist2ct],
                                                                         sd_synssv = sd_synssv, syn_prob=syn_prob,
                                                                         min_syn_size=min_syn_size, ct_pre=None,
                                                                         cellids_pre=None)
        else:
            post_ids, median_distances_per_ids, min_distances_per_ids, max_distances_per_ids, syn_numbers, syn_ssv_sizes = get_syn_distances(ct_post=dist2ct,
                                                                         cellids_post=suitable_ids_dict[dist2ct],
                                                                         sd_synssv=sd_synssv, syn_prob=syn_prob,
                                                                         min_syn_size=min_syn_size, ct_pre=ct,
                                                                         cellids_pre=suitable_ids_dict[ct])
        distances_dict[(ct_str, dist2ct_str)] = {'ids': post_ids, 'median synapse distance to soma': median_distances_per_ids,
                                                 'min synapse distance to soma': min_distances_per_ids,
                                                 'max synapse distance to soma': max_distances_per_ids,
                                                 'synapse number': syn_numbers, 'syn ssv sizes': syn_ssv_sizes}
        median_dist_df.loc[0:len(post_ids) - 1, ct_str] = median_distances_per_ids
        min_dist_df.loc[0:len(post_ids) - 1, ct_str] = min_distances_per_ids
        max_dist_df.loc[0:len(post_ids) - 1, ct_str] = max_distances_per_ids

    write_obj2pkl('%s/distances_result_dict.pkl' % f_name, distances_dict)
    median_dist_df.to_csv('%s/median_syn_distance2soma.csv' % f_name)
    max_dist_df.to_csv('%s/median_syn_distance2soma.csv' % f_name)
    min_dist_df.to_csv('%s/median_syn_distance2soma.csv' % f_name)
    time_stamps = [time.time()]
    step_idents = ['get synapse distances to soma']

    log.info("Step 3/3 Plot results and calculate statistics")
    str_params = ['median synapse distance to soma', 'min synapse distance to soma', 'max synapse distance to soma']
    param_dfs = [median_dist_df, min_dist_df, max_dist_df]
    cls = CelltypeColors()
    ct_palette = cls.ct_palette(color_key, num=False)
    ranksum_results = pd.DataFrame()
    for i, param in enumerate(tqdm(param_dfs)):
        #use ranksum test (non-parametric) to calculate results
        for c1 in cts_for_loading:
            for c2 in cts_for_loading:
                if c2 >= c1:
                    continue
                stats, p_value = ranksums(param[c1], param[c2])
                ranksum_results.loc["stats " + str_params[i] + ' of ' + dist2ct_str, ct_dict[c1] + " vs " + ct_dict[c2]] = stats
                ranksum_results.loc["p value " + str_params[i] + ' of ' + dist2ct_str, ct_dict[c1] + " vs " + ct_dict[c2]] = p_value
        #make violinplot, boxplot, histplot
        ylabel = 'distance in µm'
        sns.stripplot(data=param, color="black", alpha=0.2,
                      dodge=True, size=2)
        sns.violinplot(data=param, inner="box",
                       palette=ct_palette)
        plt.title(str_params[i] + ' of ' + dist2ct_str)
        plt.ylabel(ylabel)
        plt.savefig('%s/%s_syn_dst2soma_violin.png' % (f_name, str_params[i].split()[0]))
        plt.close()
        sns.boxplot(data=param, palette=ct_palette)
        plt.title(str_params[i] + ' of ' + dist2ct_str)
        plt.ylabel(ylabel)
        plt.savefig('%s/%s_syn_dst2soma_box.png' % (f_name, str_params[i].split()[0]))
        plt.close()
        sns.histplot(data=param, palette=ct_palette, legend= True)
        plt.title(str_params[i] + ' of ' + dist2ct_str)
        plt.xlabel(ylabel)
        plt.ylabel('count of cells')
        plt.savefig('%s/%s_syn_dst2soma_dist.png' % (f_name, str_params[i].split()[0]))
        plt.close()

    log.info('Distance to synapse analysis done')
    time_stamps = time.time()
    step_idents = ['Plotting finished']

