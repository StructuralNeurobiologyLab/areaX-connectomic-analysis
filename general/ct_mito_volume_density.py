#mito volume density axon per celltype
if __name__ == '__main__':
    from analysis_morph_helper import check_comp_lengths_ct
    from analysis_colors import CelltypeColors
    from analysis_morph_helper import get_organell_volume_density_comps
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from analysis_params import Analysis_Params
    import os as os
    from syconn.reps.segmentation import SegmentationDataset
    import pandas as pd
    import numpy as np
    from syconn.mp.mp_utils import start_multiprocess_imap
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import ranksums, kruskal, spearmanr
    from itertools import combinations
    from sklearn.linear_model import LinearRegression

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    start = time.time()
    #ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
     #          10: "NGF"}
    version = 'v6'
    with_glia = False
    min_comp_len_cell = 200
    min_comp_len_ax = 50
    mito_k = 3
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw'}
    color_key = 'TePkBrNGF'
    full_cell_only = True
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/general/231220_j0251{version}_ct_mito_vol_density_mcl_%i_ax%i_k%i_%s_fconly" % (
        min_comp_len_cell, min_comp_len_ax, mito_k, color_key)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('get volume density mito per celltype', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i for full cells, min_comp_len = %i for axons, mito k = %i, colors = %s" % (
            min_comp_len_cell, min_comp_len_ax, mito_k, color_key))
    analysis_params = Analysis_Params(working_dir=global_params.wd, version=version)
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    known_mergers = analysis_params.load_known_mergers()
    misclassified_asto_ids = analysis_params.load_potential_astros()
    axon_cts = analysis_params.axon_cts()
    num_cts = analysis_params.num_cts(with_glia=with_glia)
    ct_str_list = analysis_params.ct_str(with_glia=with_glia)
    cls = CelltypeColors(ct_dict= ct_dict)
    ct_palette = cls.ct_palette(key = color_key)
    if with_glia:
        glia_cts = analysis_params._glia_cts
    sd_mi = SegmentationDataset('mi', working_dir=global_params.wd)
    mi_ids = sd_mi.ids
    mito_coords = sd_mi.load_numpy_data("rep_coord")
    mito_volumes = sd_mi.load_numpy_data("size")
    firing_rate_dict = {'DA': 15, 'MSN': 1.58, 'LMAN': 34.9, 'HVC': 1, 'TAN': 65.1, 'GPe': 135, 'GPi': 258, 'FS': 19.1, 'LTS': 35.8}

    log.info('Step 1/4: Iterate over each celltypeto check min length')
    suitable_ids_dict = {}
    all_suitable_ids = []
    all_cell_dict = {}
    for ct in range(num_cts):
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
            astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
            cellids = cellids[astro_inds]
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cell,
                                                axon_only=False, max_path_len=None)
        cellids = np.sort(cellids)
        suitable_ids_dict[ct] = cellids
        all_suitable_ids.append(cellids)
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict[ct]))

    all_suitable_ids = np.concatenate(all_suitable_ids)

    log.info('Step 2/4: Get mito volume density per cell')
    #ellid, cached_so_ids, cached_so_rep_coord, cached_so_volume, full_cell_dict, k, min_comp_len = input
    # generate pd Dataframe as overview per celltype
    ov_columns = ['celltype', 'mean firing rate singing', 'mean total mito volume density',
                  'std total mito volume density',
                  'mean axon mito volume density', 'std axon mito volume density']
    overview_df = pd.DataFrame(columns=ov_columns, index=range(num_cts))
    # generate df for each cell
    pc_columns = ['cellid', 'mean firing rate singing', 'total mito volume density', 'axon mito volume density']
    percell_mito_df = pd.DataFrame(columns=pc_columns, index=range(len(all_suitable_ids)))
    percell_mito_df['cellid'] = all_suitable_ids
    for ct in range(num_cts):
        ct_str = ct_dict[ct]
        try:
            firing_value = firing_rate_dict[ct_str]
        except KeyError:
            firing_value = np.nan
        if ct in axon_cts:
            input = [[cellid, mi_ids, mito_coords, mito_volumes, all_cell_dict[ct], mito_k, min_comp_len_ax, True] for cellid in suitable_ids_dict[ct]]
        else:
            input = [[cellid, mi_ids, mito_coords, mito_volumes, all_cell_dict[ct], mito_k, min_comp_len_cell, False] for cellid in suitable_ids_dict[ct]]
        output = start_multiprocess_imap(get_organell_volume_density_comps, input)
        output = np.array(output, dtype='object')
        axon_so_density = np.concatenate(output[:, 0])
        axon_volume_density = np.concatenate(output[:, 1])
        #for axons this is same as axon volume density
        full_volume_density = np.concatenate(output[:, 4])
        if ct not in axon_cts:
            dendrite_so_density = np.concatenate(output[:, 2])
            dendrite_volume_density = np.concatenate(output[:, 3])
        mean_axo_den = np.mean(axon_volume_density)
        std_axo_den = np.std(axon_volume_density)
        mean_total_den = np.mean(full_volume_density)
        std_total_den = np.std(full_volume_density)
        overview_df['celltype'] = ct
        overview_df['mean firing rate singing'] = firing_value
        overview_df['mean total mito volume density'] = mean_total_den
        overview_df['std total mito volume density'] = std_total_den
        overview_df['mean axon mito volume density'] = mean_axo_den
        overview_df['std axon mito volume density'] = std_axo_den
        #for percell df
        ct_inds = np.in1d(percell_mito_df['cellid'], suitable_ids_dict[ct])
        percell_mito_df[ct_inds, 'celltype'] = ct_str
        percell_mito_df[ct_inds, 'mean firing rate singing'] = firing_value
        percell_mito_df[ct_inds, 'total mito volume density'] = full_volume_density
        percell_mito_df[ct_inds, 'axon mito volume density'] = axon_volume_density

    overview_df.to_csv(f'{f_name}/overview_df_mito_den.csv')
    percell_mito_df.to_csv(f'{f_name}/percell_df_mito_den.csv')

    log.info('Step 3/4: Calculate statistics and plot results')
    group_comps = list(combinations(range(num_cts), 2))
    ranksum_columns = [f'{ct_str_list[gc[0]]} vs {ct_str_list[gc[1]]}' for gc in group_comps]
    ranksum_group_df = pd.DataFrame(columns=ranksum_columns)

    for key in percell_mito_df.keys():
        if 'mito' in key:
            key_groups = [group[key].values for name, group in
                                percell_mito_df.groupby('celltype')]
            kruskal_res = kruskal(*key_groups, nan_policy='omit')
            log.info(f'Kruskal Wallis test result for {key}: {kruskal_res}')
            spearman_res = spearmanr(percell_mito_df[key], percell_mito_df['mean firing rate singing'], nan_policy='omit')
            log.info(f'Spearman correlation test result for {key}: {spearman_res}')
            #ranksum results
            for group in group_comps:
                ranksum_res = ranksums(key_groups[group[0]], key_groups[group[1]])
                ranksum_group_df.loc[f'{key} stats', f'{ct_str_list[group[0]]} vs {ct_str_list[group[1]]}'] = ranksum_res[0]
                ranksum_group_df.loc[f'{key} p-value',f'{ct_str_list[group[0]]} vs {ct_str_list[group[1]]}'] = ranksum_res[1]
            #plot with increasing median as boxplot and violinplot
            median_order = key_groups.median().index
            sns.boxplot(data=percell_mito_df, x='celltype', y=key, palette=ct_palette, order=median_order)
            plt.title(key)
            plt.savefig(f'{f_name}/{key}_box.png')
            plt.savefig(f'{f_name}/{key}_box.svg')
            plt.ylabel(f'{key} [µm³/µm]')
            plt.close()
            sns.stripplot(data=percell_mito_df, x='celltype', y=key, palette=ct_palette, color='black', alpha=0.2,
                          dodge=True, size=2, order=median_order)
            sns.violinplot(data=percell_mito_df, x='celltype', y=key, palette=ct_palette, inner="box", order=median_order)
            plt.title(key)
            plt.ylabel(f'{key} [µm³/µm]')
            plt.savefig(f'{f_name}/{key}_violin.png')
            plt.savefig(f'{f_name}/{key}_violin.svg')
            plt.close()

    ranksum_group_df.to_csv(f'{f_name}/ranksum_results.csv')

    log.info('Analysis finished')

    log.info('Step 4/4: Plot mean firing rates vs mito density')
    #plot once with and once without unknown literature values
    known_values_only_ov = overview_df.dropna()
    known_values_only_percell = percell_mito_df.dropna()
    fs_dict = {'celltype': 'FS', 'mean firing rate singing': firing_rate_dict['FS']}
    overview_df = overview_df.append(fs_dict, ignore_index=True)
    for key in overview_df.keys():
        if 'mean' in key:
            sns.pointplot(data= known_values_only_ov, x = key, y = 'mean firing rate singing', color = 'black')
            plt.xlabel(f'{key} [µm³/µm]')
            plt.ylabel('mean firing rate singing [Hz]')
            plt.savefig(f'{f_name}/{key}_firing_rate_known_only.png')
            plt.savefig(f'{f_name}/{key}_firing_rate_known_only.svg')
            plt.close()
            #lin reg code adopted from ChatGPT
            reg_model = LinearRegression()
            reg_model.fit(known_values_only_percell[key], known_values_only_percell['mean firing rate singing'])
            #get coeff and intercept
            coefficient = reg_model.coef_
            intercept = reg_model.intercept_
            log.info(f'Regression coefficient for {key} and mean firing rate: {coefficient}, intercept: {intercept}')
            #get prediction for unkown numbers
            for ct in range(num_cts):
                ct_str = ct_dict[ct]
                if ct_str in firing_rate_dict.keys():
                    continue
                key_ct_ind = np.where(overview_df['celltype'] == ct_str)[0]
                key_ct_value = overview_df[key][key_ct_ind]
                firing_pred = reg_model.predict(key_ct_value)
                overview_df.loc[key_ct_ind, key] = firing_pred
            sns.scatterplot(data=overview_df, x=key, y='mean firing rate singing', color='black')
            plt.xlabel(f'{key} [µm³/µm]')
            plt.ylabel('mean firing rate singing [Hz]')
            plt.savefig(f'{f_name}/{key}_firing_rate_pred.png')
            plt.savefig(f'{f_name}/{key}_firing_rate_pred.svg')
            plt.close()
            #also predict 'FS' mito density value
            fs_mito_pred = (firing_rate_dict['FS'] - intercept) / coefficient
            fs_ind = np.where(overview_df['celltype'] == 'FS')[0]
            overview_df.loc[fs_ind, key] = fs_mito_pred
            sns.scatterplot(data=overview_df, x=key, y='mean firing rate singing', color='black')
            plt.xlabel(f'{key} [µm³/µm]')
            plt.ylabel('mean firing rate singing [Hz]')
            plt.savefig(f'{f_name}/{key}_firing_rate_pred_withFS.png')
            plt.savefig(f'{f_name}/{key}_firing_rate_pred_withFS.svg')
            plt.close()

        log.info('Analysis done')















