#analysis to see which fraction of vesicles close to membrane is close to the synapse

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct,  filter_synapse_caches_general
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from cajal.nvmescratch.users.arother.bio_analysis.general.vesicle_helper import get_synapse_proximity_vesicle_percell
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    from syconn.reps.segmentation import SegmentationDataset
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from syconn.mp.mp_utils import start_multiprocess_imap
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import kruskal, ranksums
    from itertools import combinations

    #global_params.wd = '/cajal/nvmescratch/projects/data/songbird/j0251/j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_20230822'
    #ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
     #          10: "NGF"}
    version = 'v6'
    analysis_params = Analysis_Params(version = version)
    global_params.wd = analysis_params.working_dir()
    ct_dict = analysis_params.ct_dict(with_glia=False)
    num_cts = analysis_params.num_cts(with_glia=False)
    axon_cts = analysis_params.axon_cts()
    min_comp_len_cell = 200
    min_comp_len_ax = 200
    dist_threshold = 10 #nm
    min_syn_size = 0.1
    syn_prob_thresh = 0.6
    syn_dist_threshold = 500 #nm
    nonsyn_dist_threshold = 1000 #nm
    cls = CelltypeColors(ct_dict = ct_dict)
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw'}
    color_key = 'TePkBrNGF'
    fontsize = 20
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/single_vesicle_analysis/241029_j0251{version}_ct_syn_fraction_closemembrane_mcl_%i_ax%i_dt_%i_st_%i_%i_%s" % (
        min_comp_len_cell, min_comp_len_ax, dist_threshold, syn_dist_threshold, nonsyn_dist_threshold, color_key)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('get fraction of vesicles close to membrane which are not close to synapse', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i for full cells, min_comp_len = %i for axons, min_syn_size = %.1f, syn_prob_thresh = %.1f, distance threshold to membrane = %s nm, "
        "distance threshold to synapse = %i nm, distance threshold for not at synapse = %i nm, colors = %s" % (
            min_comp_len_cell, min_comp_len_ax, min_syn_size, syn_prob_thresh, dist_threshold, syn_dist_threshold, nonsyn_dist_threshold, color_key))
    known_mergers = analysis_params.load_known_mergers()
    log.info("Step 1/6: synapse segmentation dataset")

    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    cache_name = analysis_params.file_locations

    log.info('Step 1/4: Iterate over each celltypes check min length')
    ct_types = np.arange(0, num_cts)
    suitable_ids_dict = {}
    all_suitable_ids = []
    all_cell_dict = {}
    all_suitable_cts = []
    #misclassified_asto_ids = analysis_params.load_potential_astros()
    for ct in ct_types:
        # only get cells with min_comp_len, MSN with max_comp_len or axons with min ax_len
        ct_str = ct_dict[ct]
        cell_dict = analysis_params.load_cell_dict(ct)
        all_cell_dict[ct] = cell_dict
        cellids = np.array(list(cell_dict.keys()))
        merger_inds = np.in1d(cellids, known_mergers) == False
        cellids = cellids[merger_inds]
        if ct in axon_cts:
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_ax,
                                            axon_only=True, max_path_len=None)
        else:
            #astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
            #cellids = cellids[astro_inds]
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cell,
                                            axon_only=False, max_path_len=None)
        cellids = np.sort(cellids)
        suitable_ids_dict[ct] = cellids
        all_suitable_ids.append(cellids)
        all_suitable_cts.append([ct_dict[ct] for i in cellids])
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict[ct]))

    all_suitable_ids = np.concatenate(all_suitable_ids)
    all_suitable_cts =  np.concatenate(all_suitable_cts)

    log.info('Step 2/6: Prepare dataframes for results')
    ct_str_list = analysis_params.ct_str(with_glia=False)
    ct_palette = cls.ct_palette(color_key, num=False)
    pc_columns = ['cellid', 'celltype', 'fraction of non-synaptic membrane-close vesicles',
                  'density of non-synaptic membrane-close vesicles', 'density of synaptic membrane-close vesicles']
    vesicle_df = pd.DataFrame(columns=pc_columns, index=range(len(all_suitable_ids)))
    vesicle_df['cellid'] = all_suitable_ids
    vesicle_df['celltype'] = all_suitable_cts

    log.info('Step 3/6: Get information for vesicles close to membrane and synapse for all celltypes')
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob = filter_synapse_caches_general(
        sd_synssv,
        syn_prob_thresh=syn_prob_thresh,
        min_syn_size=min_syn_size)
    synapse_cache = [m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob]
    for ct in tqdm(range(num_cts)):
        # only get cells with min_comp_len, MSN with max_comp_len or axons with min ax_len
        ct_str = ct_dict[ct]
        log.info(f'Now processing cells from cell type {ct_str}')
        cellids = suitable_ids_dict[ct]
        cell_dict = all_cell_dict[ct]
        log.info('Prefilter synapses for celltype')
        #filter synapses to only have specific celltype
        m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(
            pre_cts=[ct],
            post_cts=None,
            syn_prob_thresh=None,
            min_syn_size=None,
            axo_den_so=True,
            synapses_caches=synapse_cache)
        #filter so that only filtered cellids are included and are all presynaptic
        ct_inds = np.in1d(m_ssv_partners, cellids).reshape(len(m_ssv_partners), 2)
        comp_inds = np.in1d(m_axs, 1).reshape(len(m_ssv_partners), 2)
        filtered_inds = np.all(ct_inds == comp_inds, axis=1)
        syn_coords = m_rep_coord[filtered_inds]
        syn_axs = m_axs[filtered_inds]
        syn_ssv_partners = m_ssv_partners[filtered_inds]
        log.info('Prefilter vesicles for celltype')
        #load caches prefiltered for celltype (if celltype with full cells -> load only those)
        if ct in axon_cts:
            ct_ves_ids = np.load(f'{cache_name}/{ct_dict[ct]}_ids.npy')
            ct_ves_coords = np.load(f'{cache_name}/{ct_dict[ct]}_rep_coords.npy')
            ct_ves_map2ssvids = np.load(f'{cache_name}/{ct_dict[ct]}_mapping_ssv_ids.npy')
            ct_ves_dist2matrix = np.load(f'{cache_name}/{ct_dict[ct]}_dist2matrix.npy')
        else:
            ct_ves_ids = np.load(f'{cache_name}/{ct_dict[ct]}_ids_fullcells.npy')
            ct_ves_coords = np.load(f'{cache_name}/{ct_dict[ct]}_rep_coords_fullcells.npy')
            ct_ves_map2ssvids = np.load(f'{cache_name}/{ct_dict[ct]}_mapping_ssv_ids_fullcells.npy')
            ct_ves_dist2matrix = np.load(f'{cache_name}/{ct_dict[ct]}_dist2matrix_fullcells.npy')
            ct_ves_axoness = np.load(f'{cache_name}/{ct_dict[ct]}_axoness_coarse_fullcells.npy')
        #filter for selected cellids
        ct_ind = np.in1d(ct_ves_map2ssvids, cellids)
        ct_ves_ids = ct_ves_ids[ct_ind]
        ct_ves_map2ssvids = ct_ves_map2ssvids[ct_ind]
        ct_ves_dist2matrix = ct_ves_dist2matrix[ct_ind]
        ct_ves_coords = ct_ves_coords[ct_ind]
        if ct not in axon_cts:
            ct_ves_axoness = ct_ves_axoness[ct_ind]
            #make sure for full cells vesicles are only in axon
            ax_ind = np.in1d(ct_ves_axoness, 1)
            ct_ves_ids = ct_ves_ids[ax_ind]
            ct_ves_map2ssvids = ct_ves_map2ssvids[ax_ind]
            ct_ves_dist2matrix = ct_ves_dist2matrix[ax_ind]
            ct_ves_coords = ct_ves_coords[ax_ind]
        assert len(np.unique(ct_ves_map2ssvids)) <= len(cellids)
        # get axon_pathlength for corrensponding cellids
        axon_pathlengths = np.zeros(len(cellids))
        for c, cellid in enumerate(tqdm(cellids)):
            axon_pathlengths[c] = cell_dict[cellid]['axon length']
        log.info('Iterate over cells to get vesicles associated to axon, vesicle info for synapses')
        #prepare inputs for multiprocessing
        cell_inputs = [
            [cellids[i], ct_ves_coords, ct_ves_map2ssvids, ct_ves_dist2matrix, dist_threshold, syn_coords, syn_axs,
             syn_ssv_partners, syn_dist_threshold, nonsyn_dist_threshold, axon_pathlengths[i]] for i in range(len(cellids))]
        outputs = start_multiprocess_imap(get_synapse_proximity_vesicle_percell, cell_inputs)
        outputs = np.array(outputs)
        fraction_non_syn_mem_vesicles = outputs[:, 0]
        density_non_syn_mem_vesicles = outputs[:, 1]
        density_syn_mem_vesicles = outputs[:, 2]
        cts_inds = np.in1d(vesicle_df['cellid'], cellids)
        vesicle_df.loc[cts_inds, 'fraction of non-synaptic membrane-close vesicles'] = fraction_non_syn_mem_vesicles
        vesicle_df.loc[cts_inds,  'density of non-synaptic membrane-close vesicles'] = density_non_syn_mem_vesicles
        vesicle_df.loc[cts_inds,  'density of synaptic membrane-close vesicles'] = density_syn_mem_vesicles

    #filter out cells that don't have any close-membrane vesicles
    num_cells_before = len(vesicle_df)
    num_cts_before = vesicle_df.groupby('celltype').size()
    log.info(f'In total {num_cells_before} cells were processed, in the following celltypes: {num_cts_before}')
    vesicle_df = vesicle_df.dropna()
    vesicle_df = vesicle_df.reset_index(drop = True)
    num_cells_after = len(vesicle_df)
    ct_groups = vesicle_df.groupby('celltype')
    num_cts_after = ct_groups.size()
    log.info(f'Number of cells with membrane close vesicles is {num_cells_after}, in the following celltypes: {num_cts_after}')
    log.info(f'{num_cells_before - num_cells_after} cells were removed')
    param_list = pc_columns[2:]
    vesicle_df = vesicle_df.astype({param:float for param in param_list})
    vesicle_df.to_csv(f'{f_name}/vesicle_close_membrane_densities_results.csv')

    log.info('Step 5/6: Get overview params and calculate statistics')
    unique_cts = np.unique(vesicle_df['celltype'])
    overview_columns = [[f'{param} median', f'{param} mean', f'{param} std'] for param in param_list]
    overview_columns = np.concatenate(overview_columns)
    overview_df = pd.DataFrame(columns=overview_columns, index=unique_cts)
    group_comps = list(combinations(unique_cts, 2))
    ranksum_columns = [f'{gc[0]} vs {gc[1]}' for gc in group_comps]
    ranksum_group_df = pd.DataFrame(columns=ranksum_columns)
    for param in param_list:
        overview_df.loc[unique_cts, f'{param} median'] = ct_groups[param].median()
        overview_df.loc[unique_cts, f'{param} mean'] = ct_groups[param].mean()
        overview_df.loc[unique_cts, f'{param} std'] = ct_groups[param].std()
        # calculate kruskal wallis for differences between the two groups
        param_groups = [group[param].values for name, group in
                        vesicle_df.groupby('celltype')]
        kruskal_res = kruskal(*param_groups, nan_policy='omit')
        log.info(
            f'Kruskal results for {param} are: stats = {kruskal_res[0]:.2f}, p-value = {kruskal_res[1]:.2f}')
        # get ranksum results if significant
        if kruskal_res[1] < 0.05:
            for group in group_comps:
                ranksum_res = ranksums(ct_groups.get_group(group[0])[param],
                                       ct_groups.get_group(group[1])[param])
                ranksum_group_df.loc[f'{param} stats', f'{group[0]} vs {group[1]}'] = ranksum_res[0]
                ranksum_group_df.loc[f'{param} p-value', f'{group[0]} vs {group[1]}'] = ranksum_res[1]

    overview_df.to_csv(f'{f_name}/overview_df.csv')
    ranksum_group_df.to_csv(f'{f_name}/ranksum_results.csv')

    log.info('Step 6/6: Plot results')
    for key in vesicle_df.columns:
        if 'cellid' in key or 'celltype' in key:
            continue
        if 'fraction' in key:
            ylabel = key
        else:
            ylabel = f'{key} [1/µm]'
        sns.boxplot(data = vesicle_df, x = 'celltype', y = key, palette=ct_palette, order=ct_str_list)
        plt.ylabel(key, fontsize = fontsize)
        plt.xlabel('celltype', fontsize = fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.title(f'{key} ({syn_dist_threshold} nm)')
        plt.savefig(f'{f_name}/{key}_{dist_threshold}nm_{syn_dist_threshold}nm.svg')
        plt.savefig(f'{f_name}/{key}_{dist_threshold}nm_{syn_dist_threshold}nm.png')
        plt.close()

    #plot overview
    median_plotting_df = pd.DataFrame(columns=['celltype', 'vesicle density', 'location'], index=range(num_cts * 2))
    median_plotting_df.loc[0:num_cts - 1, 'vesicle density'] = np.array(overview_df['density of non-synaptic membrane-close vesicles median'])
    median_plotting_df.loc[0:num_cts - 1, 'location'] = 'non-synaptic'
    median_plotting_df.loc[0:num_cts - 1, 'celltype'] = unique_cts
    median_plotting_df.loc[num_cts:2*num_cts - 1, 'vesicle density'] = np.array(overview_df[
        'density of synaptic membrane-close vesicles median'])
    median_plotting_df.loc[num_cts:2*num_cts - 1, 'location'] = 'synaptic'
    median_plotting_df.loc[num_cts:2 * num_cts - 1, 'celltype'] = unique_cts
    median_plotting_df.to_csv(f'{f_name}/vesicle_densities_medians.csv')
    palette = {'non-synaptic': 'black', 'synaptic': '#00BFB2' }
    sns.pointplot(x = 'celltype', y = 'vesicle density', data = median_plotting_df, hue='location', palette=palette, join=False, order = ct_str_list)
    plt.ylabel('median vesicle density [1/µm]', fontsize=fontsize)
    plt.xlabel('celltype', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.title('Median density of vesicles close to membrane')
    plt.savefig(f'{f_name}/mem_close_comb_median_point.svg')
    plt.savefig(f'{f_name}/mem_close_comb_median_point.png')
    plt.close()

    log.info(f'Analysis for vesicles closer to {dist_threshold}nm, split into synaptic '
             f'and non-synaptic in all celltypes done')