#get information about compartment specific connectivity between different cells
#similar to CT_input_syn_distance_analysis
if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    form cajal.nvmescratch.users.arother.bio_analysis.dir_indir_pathway_analysis.connecivity_between2cts import get_compartment_specific_connectivity
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from cajal.nvmescratch.users.arother.bio_analysis.dir_indir_pathway_analysis.synapse_input_distance import get_syn_distances
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import analysis_params
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

    bio_params = analysis_params
    ct_dict = bio_params.ct_dict()
    min_comp_len = bio_params.min_comp_length()
    syn_prob = bio_params.syn_prob_thresh()
    min_syn_size = bio_params.min_syn_size()
    exclude_known_mergers = True
    #color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw'}
    color_key = 'TePkBr'
    post_ct = 6
    post_ct_str = ct_dict[post_ct]
    f_name = "cajal/nvmescratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/221109_j0251v4_%s_input_comps_mcl_%i_synprob_%.2f_%s" % (
    post_ct_str, min_comp_len, syn_prob, color_key)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('Analysis of synaptic inputs to compartments of %s' % post_ct_str, log_dir=f_name + '/logs/')
    cts_for_loading = [0, 2, 3, 6, 7, 8]
    cts_str_analysis = [ct_dict[ct] for ct in cts_for_loading]
    num_cts = len(cts_for_loading)
    log.info(
        "min_comp_len = %i, syn_prob = %.1f, min_syn_size = %.1f, known mergers excluded = %s, colors = %s" % (
        min_comp_len, syn_prob, min_syn_size, exclude_known_mergers, color_key))
    log.info(f'Distance of synapses for celltypes {cts_str_analysis} will be compared to {post_ct_str}')
    time_stamps = [time.time()]
    step_idents = ['t-0']

    log.info("Step 1/3: Load celltypes and check suitability")
    axon_cts = bio_params.axon_cts()
    cls = CelltypeColors()
    ct_palette = cls.ct_palette(color_key, num=False)
    if exclude_known_mergers:
        known_mergers = bio_params.load_known_mergers()
        misclassified_asto_ids = bio_params.load_potential_astros()
    suitable_ids_dict = {}
    for ct in tqdm(cts_for_loading):
        ct_str = ct_dict[ct]
        if ct in axon_cts:
            cell_dict = bio_params.load_cell_dict(ct)
            #get ids with min compartment length
            cellids = np.array(list(cell_dict.keys()))
            if exclude_known_mergers:
                merger_inds = np.in1d(cellids, known_mergers) == False
                cellids = cellids[merger_inds]
            cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len, axon_only=True,
                              max_path_len=None)
            suitable_ids_dict[ct] = cellids
        else:
            cell_dict = bio_params.load_cell_dict(ct)
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
    distances_df = pd.DataFrame(columns=cts_str_analysis, index=range(len(suitable_ids_dict[dist2ct])*5000))
    distances_dict = {}
    for ct in tqdm(cts_for_loading):
        ct_str = ct_dict[ct]
        #get median, min, max synapse distance to soma per cell
        #function uses multiprocessing
        if ct == dist2ct:
            post_ids, median_distances_per_ids, min_distances_per_ids, max_distances_per_ids, distances_per_cell, syn_numbers, syn_ssv_sizes = get_syn_distances(ct_post = dist2ct, cellids_post = suitable_ids_dict[dist2ct],
                                                                         sd_synssv = sd_synssv, syn_prob=syn_prob,
                                                                         min_syn_size=min_syn_size, ct_pre=None,
                                                                         cellids_pre=None, dendrite_only = only_dendrite)
        else:
            post_ids, median_distances_per_ids, min_distances_per_ids, max_distances_per_ids, distances_per_cell, syn_numbers, syn_ssv_sizes = get_syn_distances(ct_post=dist2ct,
                                                                         cellids_post=suitable_ids_dict[dist2ct],
                                                                         sd_synssv=sd_synssv, syn_prob=syn_prob,
                                                                         min_syn_size=min_syn_size, ct_pre=ct,
                                                                         cellids_pre=suitable_ids_dict[ct], dendrite_only = only_dendrite)
        distances_dict[(ct_str, dist2ct_str)] = {'ids': post_ids, 'median synapse distance to soma': median_distances_per_ids,
                                                 'min synapse distance to soma': min_distances_per_ids,
                                                 'max synapse distance to soma': max_distances_per_ids,
                                                 'synapse number': syn_numbers, 'syn ssv sizes': syn_ssv_sizes}
        median_dist_df.loc[0:len(post_ids) - 1, ct_str] = median_distances_per_ids
        min_dist_df.loc[0:len(post_ids) - 1, ct_str] = min_distances_per_ids
        max_dist_df.loc[0:len(post_ids) - 1, ct_str] = max_distances_per_ids
        distances_df.loc[0:len(distances_per_cell) - 1, ct_str] = distances_per_cell
        f_name_ct = f'{f_name}/{ct_str}'
        if not os.path.exists(f_name_ct):
            os.mkdir(f_name_ct)
        xlabel = "distance in µm"
        ylabel = "count of cells"
        ct_color = ct_palette[ct_str]
        sns.histplot(data=median_distances_per_ids, color=ct_color, legend=True, fill=True, element="step", bins = 15)
        plt.title('Median distance to soma' + ' of ' + dist2ct_str)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig('%s/median_syn_dst2soma_dist_%s.png' % (f_name_ct, ct_str))
        plt.close()
        sns.histplot(data=min_distances_per_ids, color=ct_color, legend=True, fill=True, element="step", bins = 15)
        plt.title('Min distance to soma' + ' of ' + dist2ct_str)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig('%s/min_syn_dst2soma_dist_%s.png' % (f_name_ct, ct_str))
        plt.close()
        sns.histplot(data=max_distances_per_ids, color=ct_color, legend=True, fill=True, element="step", bins = 15)
        plt.title('Max distance to soma' + ' of ' + dist2ct_str)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig('%s/max_syn_dst2soma_dist_%s.png' % (f_name_ct, ct_str))
        plt.close()
        sns.histplot(data=distances_per_cell, color=ct_color, legend=True, fill=True, element="step", bins=30)
        plt.title('Distance to soma of all synapses' + ' of ' + dist2ct_str)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig('%s/all_syn_dst2soma_dist_%s.png' % (f_name_ct, ct_str))
        plt.close()



    write_obj2pkl('%s/distances_result_dict.pkl' % f_name, distances_dict)
    median_dist_df.to_csv('%s/median_syn_distance2soma.csv' % f_name)
    max_dist_df.to_csv('%s/max_syn_distance2soma.csv' % f_name)
    min_dist_df.to_csv('%s/min_syn_distance2soma.csv' % f_name)
    distances_df.to_csv('%s/all_syn_distances.csv' % f_name)
    time_stamps = [time.time()]
    step_idents = ['get synapse distances to soma']

    log.info("Step 3/3 Plot results and calculate statistics")
    str_params = ['median synapse distance to soma', 'min synapse distance to soma', 'max synapse distance to soma', 'synapse distances to soma']
    param_dfs = [median_dist_df, min_dist_df, max_dist_df, distances_df]
    ranksum_results = pd.DataFrame()
    for i, param in enumerate(tqdm(param_dfs)):
        #use ranksum test (non-parametric) to calculate results
        for c1 in cts_for_loading:
            for c2 in cts_for_loading:
                if c1 >= c2:
                    continue
                p_c1 = np.array(param[ct_dict[c1]]).astype(float)
                p_c2 = np.array(param[ct_dict[c2]]).astype(float)
                stats, p_value = ranksums(p_c1, p_c2, nan_policy = 'omit')
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
        sns.histplot(data=param, palette=ct_palette, legend= True, fill=True, element="step")
        plt.title(str_params[i] + ' of ' + dist2ct_str)
        plt.xlabel(ylabel)
        plt.ylabel('count of cells')
        plt.savefig('%s/%s_syn_dst2soma_dist.png' % (f_name, str_params[i].split()[0]))
        plt.close()

    ranksum_results.to_csv("%s/ranksum_results.csv" % f_name)

    log.info('Distance to synapse analysis done')
    time_stamps = time.time()
    step_idents = ['Plotting finished']