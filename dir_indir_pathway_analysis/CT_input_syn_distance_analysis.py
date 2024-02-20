#write script that shows the distance of synaptic inputs to the soma for GPi
#script gets distance of different synaptic inputs to GPi soma and plots differences between different celltypes

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_general
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from cajal.nvmescratch.users.arother.bio_analysis.dir_indir_pathway_analysis.synapse_input_distance import get_syn_distances
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    import os as os
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl
    import numpy as np
    from tqdm import tqdm
    from scipy.stats import ranksums, kruskal
    import seaborn as sns
    import matplotlib.pyplot as plt

    version = 'v6'
    bio_params = Analysis_Params(version=version)
    ct_dict = bio_params.ct_dict()
    global_params.wd = bio_params.working_dir()
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    start = time.time()
    min_comp_len_cell = 200
    min_comp_len_ax = 50
    syn_prob = bio_params.syn_prob_thresh()
    min_syn_size = bio_params.min_syn_size()
    exclude_known_mergers = True
    #color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw', 'STNGP'}
    color_key = 'STNGPINTv6'
    only_dendrite = False
    dist2ct = 6
    dist2ct_str = ct_dict[dist2ct]
    save_svg = True
    fontsize = 20
    if only_dendrite:
        f_name = f'cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/240220_j0251{version}_{dist2ct_str}_syn_distances_' \
                 f'mcl_{min_comp_len_cell}_ax{min_comp_len_ax}_synprob_{syn_prob}_{color_key}_f{fontsize}_denonly'
    else:
        f_name = f'cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/240220_j0251{version}_{dist2ct_str}_syn_distances_' \
                 f'mcl_{min_comp_len_cell}_ax{min_comp_len_ax}_synprob_{syn_prob}_{color_key}_f{fontsize}_withsoma'
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('Analysis of distance to soma for GPi and different synaptic inputs', log_dir=f_name + '/logs/')
    cts_for_loading = [ 1, 2, 3, 4, 6, 7, 9]
    cts_str_analysis = [ct_dict[ct] for ct in cts_for_loading]
    num_cts = len(cts_for_loading)
    log.info(
        "min_comp_len = %i for full cells, min_comp_len = %i for axons, syn_prob = %.1f, min_syn_size = %.1f, known mergers excluded = %s, colors = %s, only from dendrite = %s" % (
        min_comp_len_cell, min_comp_len_ax, syn_prob, min_syn_size, exclude_known_mergers, color_key, only_dendrite))
    log.info(f'Distance of synapses for celltypes {cts_str_analysis} will be compared to {dist2ct_str}')

    log.info("Step 1/3: Load celltypes and check suitability")

    axon_cts = bio_params.axon_cts()
    cls = CelltypeColors(ct_dict = ct_dict)
    ct_palette = cls.ct_palette(color_key, num=False)
    all_suitable_ids = []
    if exclude_known_mergers:
        known_mergers = bio_params.load_known_mergers()
    suitable_ids_dict = {}
    for ct in tqdm(cts_for_loading):
        ct_str = ct_dict[ct]
        cell_dict = bio_params.load_cell_dict(ct)
        # get ids with min compartment length
        cellids = np.array(list(cell_dict.keys()))
        if exclude_known_mergers:
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            if ct == 2:
                misclassified_asto_ids = bio_params.load_potential_astros()
                astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
                cellids = cellids[astro_inds]
        if ct in axon_cts:
            cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_ax,
                                                    axon_only=True,
                                                    max_path_len=None)
        else:
            cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cell,
                                                    axon_only=False,
                                                    max_path_len=None)
        suitable_ids_dict[ct] = cellids_checked
        all_suitable_ids.append(cellids_checked)

    number_ids = [len(suitable_ids_dict[ct]) for ct in cts_for_loading]
    all_suitable_ids = np.concatenate(all_suitable_ids)
    log.info(f"Suitable ids from celltypes {cts_str_analysis} were selected: {number_ids}")
    time_stamps = [time.time()]
    step_idents = ['loading cells']

    log.info("Step 2/3: Get synapse distance to soma from different celltypes to %s" % dist2ct_str)
    median_dist_df = pd.DataFrame(columns = cts_str_analysis, index=range(len(suitable_ids_dict[dist2ct])))
    min_dist_df = pd.DataFrame(columns=cts_str_analysis, index=range(len(suitable_ids_dict[dist2ct])))
    max_dist_df = pd.DataFrame(columns=cts_str_analysis, index=range(len(suitable_ids_dict[dist2ct])))
    distances_df = pd.DataFrame(columns=cts_str_analysis, index=range(len(suitable_ids_dict[dist2ct])*5000))
    distances_dict = {}
    #prefilter synapses
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob = filter_synapse_caches_general(
        sd_synssv,
        syn_prob_thresh=syn_prob,
        min_syn_size=min_syn_size)
    # prefilter so that all synapses are between suitable ids
    suit_ids_ind = np.all(np.in1d(m_ssv_partners, all_suitable_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_ssv_partners = m_ssv_partners[suit_ids_ind]
    m_ids = m_ids[suit_ids_ind]
    m_sizes = m_sizes[suit_ids_ind]
    m_axs = m_axs[suit_ids_ind]
    m_rep_coord = m_rep_coord[suit_ids_ind]
    m_spiness = m_spiness[suit_ids_ind]
    m_cts = m_cts[suit_ids_ind]
    syn_prob = syn_prob[suit_ids_ind]
    synapse_cache = [m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob]
    for ct in tqdm(cts_for_loading):
        ct_str = ct_dict[ct]
        #get median, min, max synapse distance to soma per cell
        #function uses multiprocessing
        if ct == dist2ct:
            post_ids, median_distances_per_ids, min_distances_per_ids, max_distances_per_ids, distances_per_cell, syn_numbers, syn_ssv_sizes = get_syn_distances(
                ct_post=dist2ct, cellids_post=suitable_ids_dict[dist2ct],
                sd_synssv=None, syn_prob=syn_prob,
                min_syn_size=min_syn_size, ct_pre=None,
                cellids_pre=None, dendrite_only=only_dendrite, prefiltered_syn_params = synapse_cache)
        else:
            post_ids, median_distances_per_ids, min_distances_per_ids, max_distances_per_ids, distances_per_cell, syn_numbers, syn_ssv_sizes = get_syn_distances(
                ct_post=dist2ct, sd_synssv = None,
                cellids_post=suitable_ids_dict[dist2ct],
                syn_prob=syn_prob,
                min_syn_size=min_syn_size, ct_pre=ct,
                cellids_pre=suitable_ids_dict[ct], dendrite_only=only_dendrite, prefiltered_syn_params = synapse_cache)
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
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.savefig('%s/median_syn_dst2soma_dist_%s.png' % (f_name_ct, ct_str))
        if save_svg:
            plt.savefig('%s/median_syn_dst2soma_dist_%s.svg' % (f_name_ct, ct_str))
        plt.close()
        sns.histplot(data=min_distances_per_ids, color=ct_color, legend=True, fill=True, element="step", bins = 15)
        plt.title('Min distance to soma' + ' of ' + dist2ct_str)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.savefig('%s/min_syn_dst2soma_dist_%s.png' % (f_name_ct, ct_str))
        if save_svg:
            plt.savefig('%s/min_syn_dst2soma_dist_%s.svg' % (f_name_ct, ct_str))
        plt.close()
        sns.histplot(data=max_distances_per_ids, color=ct_color, legend=True, fill=True, element="step", bins = 15)
        plt.title('Max distance to soma' + ' of ' + dist2ct_str)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.savefig('%s/max_syn_dst2soma_dist_%s.png' % (f_name_ct, ct_str))
        if save_svg:
            plt.savefig('%s/max_syn_dst2soma_dist_%s.svg' % (f_name_ct, ct_str))
        plt.close()
        sns.histplot(data=distances_per_cell, color=ct_color, legend=True, fill=True, element="step", bins=30)
        plt.title('Distance to soma of all synapses' + ' of ' + dist2ct_str)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.savefig('%s/all_syn_dst2soma_dist_%s.png' % (f_name_ct, ct_str))
        if save_svg:
            plt.savefig('%s/all_syn_dst2soma_dist_%s.svg' % (f_name_ct, ct_str))
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
    kruskal_res_df = pd.DataFrame(columns = ['stats', 'p-value'])
    #also make DataFrame with overview values: mean, median, std
    summary_df = pd.DataFrame(columns = cts_str_analysis)
    for i, param in enumerate(tqdm(param_dfs)):
        param_groups = [param[i].dropna() for i in param.columns]
        kruskal_res = kruskal(*param_groups, nan_policy='omit')
        kruskal_res_df.loc[str_params[i], 'stats'] = kruskal_res[0]
        kruskal_res_df.loc[str_params[i], 'p-value'] = kruskal_res[1]
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
        #add median, mean, std to summary statistics
        summary_df.loc[str_params[i] + ' median'] = param.median()
        summary_df.loc[str_params[i] + ' mean'] = param.mean()
        summary_df.loc[str_params[i] + ' std'] = param.std()
        #make violinplot, boxplot, histplot
        ylabel = 'distance in µm'
        sns.stripplot(data=param, palette='dark:black', alpha=0.2,
                      dodge=True, size=2)
        sns.violinplot(data=param, inner="box",
                       palette=ct_palette)
        plt.title(str_params[i] + ' of ' + dist2ct_str)
        plt.ylabel(ylabel)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.savefig('%s/%s_syn_dst2soma_violin.png' % (f_name, str_params[i].split()[0]))
        if save_svg:
            plt.savefig('%s/%s_syn_dst2soma_violin.svg' % (f_name, str_params[i].split()[0]))
        plt.close()
        sns.boxplot(data=param, palette=ct_palette)
        plt.title(str_params[i] + ' of ' + dist2ct_str)
        plt.ylabel(ylabel)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.savefig('%s/%s_syn_dst2soma_box.png' % (f_name, str_params[i].split()[0]))
        if save_svg:
            plt.savefig('%s/%s_syn_dst2soma_box.svg' % (f_name, str_params[i].split()[0]))
        plt.close()
        sns.histplot(data=param, palette=ct_palette, legend= True, fill=True, element="step")
        plt.title(str_params[i] + ' of ' + dist2ct_str)
        plt.xlabel(ylabel)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ylabel('count of cells')
        plt.savefig('%s/%s_syn_dst2soma_dist.png' % (f_name, str_params[i].split()[0]))
        if save_svg:
            plt.savefig('%s/%s_syn_dst2soma_dist.svg' % (f_name, str_params[i].split()[0]))
        plt.close()

    ranksum_results.to_csv(f'{f_name}/ranksum_results.csv')
    kruskal_res_df.to_csv(f'{f_name}/kruskal_results.csv')
    summary_df.to_csv(f'{f_name}/summary_df.csv')

    log.info('Distance to synapse analysis done')

