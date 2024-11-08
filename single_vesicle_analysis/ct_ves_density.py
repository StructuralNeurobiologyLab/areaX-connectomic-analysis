#get vesicle density for all celltypes
#similar code to ct mito volume density

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from cajal.nvmescratch.users.arother.bio_analysis.general.vesicle_helper import get_ves_density_presaved, get_ves_comp_density_presaved
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    import pandas as pd
    import numpy as np
    from syconn.mp.mp_utils import start_multiprocess_imap
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import ranksums, kruskal, spearmanr
    from itertools import combinations
    # from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm

    #ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
     #          10: "NGF"}
    version = 'v6'
    with_glia = False
    min_comp_len_cell = 200
    min_comp_len_ax = 200
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw'}
    color_key = 'TePkBrNGF'
    full_cells_only = False
    analysis_params = Analysis_Params(version = version)
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    global_params.wd = analysis_params.working_dir()
    fontsize = 20
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/single_vesicle_analysis/241108_j0251{version}_ct_vesicle_density_mcl_%i_ax%i_%s_fs%i_newmerger" % (
        min_comp_len_cell, min_comp_len_ax, color_key, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('get single vesicle density per celltype', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i for full cells, min_comp_len = %i for axons, colors = %s" % (
            min_comp_len_cell, min_comp_len_ax, color_key))
    log.info('use mean of vesicle density for regression fit')
    if full_cells_only:
        log.info('Full cells only')
    known_mergers = analysis_params.load_known_mergers()
    #misclassified_asto_ids = analysis_params.load_potential_astros()
    axon_cts = analysis_params.axon_cts()
    num_cts = analysis_params.num_cts(with_glia=with_glia)
    ct_str_list = analysis_params.ct_str(with_glia=with_glia)
    cls = CelltypeColors(ct_dict= ct_dict)
    ct_palette = cls.ct_palette(key = color_key)
    np_presaved_loc = analysis_params.file_locations
    if with_glia:
        glia_cts = analysis_params._glia_cts
    if full_cells_only:
        ct_types = analysis_params.load_celltypes_full_cells()
        #ct_types = ct_types[1:]
    else:
        ct_types = np.arange(0, num_cts)
    firing_rate_dict = {'DA': 15, 'MSN': 1.58, 'LMAN': 34.9, 'HVC': 1, 'TAN': 65.1, 'GPe': 135, 'GPi': 258, 'FS': 19.1, 'LTS': 35.8}

    log.info('Step 1/4: Iterate over each celltype to check min length')
    suitable_ids_dict = {}
    all_suitable_ids = []
    all_cell_dict = {}
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
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict[ct]))

    all_suitable_ids = np.concatenate(all_suitable_ids)

    log.info('Step 2/4: Get vesicle density per cell of each axon')
    #ellid, cached_so_ids, cached_so_rep_coord, cached_so_volume, full_cell_dict, k, min_comp_len = input
    # generate pd Dataframe as overview per celltype

    # generate df for each cell
    pc_columns = ['cellid', 'mean firing rate singing','vesicle density']
    percell_ves_df = pd.DataFrame(columns=pc_columns, index=range(len(all_suitable_ids)))
    percell_ves_df['cellid'] = all_suitable_ids
    for i,ct in enumerate(ct_types):
        ct_str = ct_dict[ct]
        log.info(f'process {ct_str}')
        try:
            firing_value = firing_rate_dict[ct_str]
        except KeyError:
            firing_value = np.nan
        if ct in axon_cts:
            ct_ves_ids = np.load(f'{np_presaved_loc}/{ct_dict[ct]}_ids.npy')
            ct_ves_map2ssvids = np.load(f'{np_presaved_loc}/{ct_dict[ct]}_mapping_ssv_ids.npy')
        else:
            ct_ves_ids = np.load(f'{np_presaved_loc}/{ct_dict[ct]}_ids_fullcells.npy')
            ct_ves_map2ssvids = np.load(f'{np_presaved_loc}/{ct_dict[ct]}_mapping_ssv_ids_fullcells.npy')
            ct_ves_axoness = np.load(f'{np_presaved_loc}/{ct_dict[ct]}_axoness_coarse_fullcells.npy')
        # filter for selected cellids
        ct_ind = np.in1d(ct_ves_map2ssvids, suitable_ids_dict[ct])
        ct_ves_ids = ct_ves_ids[ct_ind]
        ct_ves_map2ssvids = ct_ves_map2ssvids[ct_ind]
        #get percell vesicle density
        if ct in axon_cts:
            input = [[cellid, ct_ves_map2ssvids, all_cell_dict[ct][cellid], True] for cellid in suitable_ids_dict[ct]]
            output = start_multiprocess_imap(get_ves_density_presaved, input)
        else:
            ct_ves_axoness = ct_ves_axoness[ct_ind]
            input = [[cellid, ct_ves_map2ssvids, ct_ves_axoness, all_cell_dict[ct][cellid]] for cellid in suitable_ids_dict[ct]]
            output = start_multiprocess_imap(get_ves_comp_density_presaved, input)
        axon_ves_density = np.array(output)
        #for percell df
        ct_inds = np.in1d(percell_ves_df['cellid'], suitable_ids_dict[ct])
        percell_ves_df.loc[ct_inds, 'celltype'] = ct_str
        percell_ves_df.loc[ct_inds, 'mean firing rate singing'] = firing_value
        percell_ves_df.loc[ct_inds, 'vesicle density'] = axon_ves_density

    # create overview df for summary params
    # save mean, median and std for all parameters per ct
    ct_str = np.unique(percell_ves_df['celltype'])
    ct_groups = percell_ves_df.groupby('celltype')
    overview_df = pd.DataFrame(index=ct_str)
    overview_df['celltype'] = ct_str
    overview_df['numbers'] = ct_groups.size()
    param_list = ['mean firing rate singing', f'vesicle density']
    for key in param_list:
        if 'firing rate' in key:
            overview_df[key] = ct_groups[key].mean()
        else:
            overview_df[f'{key} mean'] = ct_groups[key].mean()
            overview_df[f'{key} std'] = ct_groups[key].std()
            overview_df[f'{key} median'] = ct_groups[key].median()

    overview_df = overview_df.astype(
        {'mean firing rate singing': float, 'vesicle density mean': float,
         'vesicle density median': float})
    overview_df.to_csv(f'{f_name}/overview_df_ves_den.csv')
    percell_ves_df = percell_ves_df.astype(
        {'mean firing rate singing': float, 'vesicle density': float})
    percell_ves_df.to_csv(f'{f_name}/percell_df_ves_den.csv')

    log.info('Step 3/4: Calculate statistics and plot results')
    group_comps = list(combinations(ct_str, 2))
    ranksum_columns = [f'{gc[0]} vs {gc[1]}' for gc in group_comps]
    ranksum_group_df = pd.DataFrame(columns=ranksum_columns)
    known_values_only_percell = percell_ves_df.dropna()
    if full_cells_only:
        axon_str = [ct_dict[ct] for ct in axon_cts]
        ct_str_list_plotting = ct_str_list[np.in1d(ct_str_list, axon_str) == False]
    else:
        ct_str_list_plotting = ct_str_list

    key_groups = [group['vesicle density'].values for name, group in
                        percell_ves_df.groupby('celltype')]
    #medians = [np.median(kg) for kg in key_groups]
    #median_order = np.unique(percell_ves_df['celltype'])[np.argsort(medians)]
    #ct_colors = cls.colors[color_key]
    #ct_palette = {median_order[i]: ct_colors[i] for i in range(len(median_order))}
    kruskal_res = kruskal(*key_groups, nan_policy='omit')
    log.info(f'Kruskal Wallis test result for vesicle density: {kruskal_res}')
    #ranksum results
    for group in group_comps:
        ranksum_res = ranksums(ct_groups.get_group(group[0])['vesicle density'], ct_groups.get_group(group[1])['vesicle density'])
        ranksum_group_df.loc[f'vesicle density stats', f'{group[0]} vs {group[1]}'] = ranksum_res[0]
        ranksum_group_df.loc[f'vesicle density p-value', f'{group[0]} vs {group[1]}'] = ranksum_res[1]
    #plot with increasing median as boxplot and violinplot
    sns.boxplot(data=percell_ves_df, x='celltype', y='vesicle density', palette=ct_palette, order=ct_str_list)
    plt.title('vesicle density')
    plt.ylabel(f'vesicle density [1/µm]', fontsize = fontsize)
    plt.xlabel('celltype', fontsize = fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/vesicle_density_box.png')
    plt.savefig(f'{f_name}/vesicle_density_box.svg')
    plt.close()
    if full_cells_only:
        sns.stripplot(data=percell_ves_df, x='celltype', y='vesicle density', color='black', alpha=0.2,
                  dodge=True, size=2, order=ct_str_list)
    sns.violinplot(data=percell_ves_df, x='celltype', y='vesicle density', palette=ct_palette, inner="box", order=ct_str_list)
    plt.title('vesicle density')
    plt.ylabel(f'vesicle density [1/µm]', fontsize=fontsize)
    plt.xlabel('celltype', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/vesicle_density_violin.png')
    plt.savefig(f'{f_name}/vesicle_density_violin.svg')
    plt.close()

    ranksum_group_df.to_csv(f'{f_name}/ranksum_results.csv')

    log.info('Analysis finished')

    log.info('Step 4/4: Plot mean firing rates vs vesicle density')
    #plot once with and once without unknown literature values
    known_values_only_ov = overview_df.dropna()
    known_cts_only = np.unique(known_values_only_ov['celltype'])
    fs_dict = {'celltype': 'FS', 'mean firing rate singing': firing_rate_dict['FS']}
    fs_df = pd.DataFrame(fs_dict, index=['FS'])
    overview_df = pd.concat([overview_df, fs_df])
    ov_palette = {known_cts_only[i]: '#232121' for i in range(len(known_cts_only))}
    ov_palette['FS'] = '#232121'
    for ct in overview_df['celltype']:
        if ct not in known_cts_only and ct != 'FS':
            ov_palette[ct] = '#15AEAB'
    for key in overview_df.keys():
        if ('mean' in key or 'median' in key) and 'ves' in key:
            sns.scatterplot(data= known_values_only_ov, x = key, y = 'mean firing rate singing', hue = 'celltype', palette=ov_palette, legend=False)
            for x, y, t in zip(overview_df[key], overview_df['mean firing rate singing'], overview_df['celltype']):
                plt.text(x = x, y = y + 10, s = t)
            plt.xlabel(f'{key} [1/µm]', fontsize = fontsize)
            plt.ylabel('mean firing rate singing [Hz]', fontsize = fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.savefig(f'{f_name}/{key}_firing_rate_known_only.png')
            plt.savefig(f'{f_name}/{key}_firing_rate_known_only.svg')
            plt.close()
            # calculate spearmannr only for known celltypes
            spearman_res = spearmanr(known_values_only_ov[key], known_values_only_ov['mean firing rate singing'], nan_policy='omit')
            spearman_cts = np.unique(known_values_only_ov['celltype'])
            log.info(f'Spearman correlation test result for vesicle density: {spearman_res}, for these celltypes {spearman_cts}')
            # percell_key = key.split(' ')[1:]
            # percell_key = ' '.join(percell_key)
            # lin reg code adopted from ChatGPT
            X_with_intercept = sm.add_constant(known_values_only_ov[key])
            reg_model = sm.OLS(known_values_only_ov['mean firing rate singing'], X_with_intercept)
            # reg_model.fit(np.array(known_values_only_ov[key]).reshape(-1, 1), known_values_only_ov['mean firing rate singing'])
            # get coeff and intercept
            results = reg_model.fit()
            # coefficient = reg_model.coef_
            # intercept = reg_model.intercept_
            coefficient = results.params[key]
            intercept = results.params['const']
            log.info(f'Regression coefficient for {key} and mean firing rate: {coefficient}, intercept: {intercept}')
            log.info(f'Summary:  {results.summary()}')
            # for plotted line
            start_xseq = np.min(known_values_only_ov[key])
            end_xseq = np.max(known_values_only_ov[key])
            num_xseq = len(known_values_only_ov)
            xseq = np.linspace(start_xseq, end_xseq, num=num_xseq)
            # plot scatterplot again with fitted line
            plt.plot(xseq, intercept + coefficient * xseq, color='#707070', lw=1.5, linestyle='dashed')
            sns.scatterplot(data=known_values_only_ov, x=key, y='mean firing rate singing', hue='celltype',
                            palette=ov_palette, legend=False)
            for x, y, t in zip(overview_df[key], overview_df['mean firing rate singing'], overview_df['celltype']):
                plt.text(x=x, y=y + 10, s=t)
            plt.xlabel(f'{key} [1/µm]', fontsize=fontsize)
            plt.ylabel('mean firing rate singing [Hz]', fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.savefig(f'{f_name}/{key}_firing_rate_known_only_fit.png')
            plt.savefig(f'{f_name}/{key}_firing_rate_known_only_fit.svg')
            plt.close()
            #get prediction for unkown numbers
            for ct in range(num_cts):
                ct_str = ct_dict[ct]
                if ct_str in firing_rate_dict.keys():
                    continue
                key_ct_value = overview_df.loc[ct_str, key]
                firing_pred = coefficient * key_ct_value + intercept
                overview_df.loc[ct_str, 'mean firing rate singing'] = firing_pred
            sns.scatterplot(data=overview_df, x=key, y='mean firing rate singing', hue = 'celltype', palette=ov_palette, legend=False)
            for x, y, t in zip(overview_df[key], overview_df['mean firing rate singing'], overview_df['celltype']):
                plt.text(x = x, y = y + 10, s = t)
            plt.xlabel(f'{key} [1/µm]', fontsize = fontsize)
            plt.ylabel('mean firing rate singing [Hz]', fontsize = fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.savefig(f'{f_name}/{key}_firing_rate_pred.png')
            plt.savefig(f'{f_name}/{key}_firing_rate_pred.svg')
            plt.close()
            # for plotted line
            start_xseq = np.min(overview_df[key])
            end_xseq = np.max(overview_df[key])
            num_xseq = len(overview_df)
            xseq = np.linspace(start_xseq, end_xseq, num=num_xseq)
            # plot scatterplot again with fitted line
            plt.plot(xseq, intercept + coefficient * xseq, color='#707070', lw=1.5, linestyle='dashed')
            sns.scatterplot(data=overview_df, x=key, y='mean firing rate singing', hue='celltype', palette=ov_palette,
                            legend=False)
            for x, y, t in zip(overview_df[key], overview_df['mean firing rate singing'], overview_df['celltype']):
                plt.text(x=x, y=y + 10, s=t)
            plt.xlabel(f'{key} [1/µm]', fontsize=fontsize)
            plt.ylabel('mean firing rate singing [Hz]', fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.savefig(f'{f_name}/{key}_firing_rate_pred_fit.png')
            plt.savefig(f'{f_name}/{key}_firing_rate_pred_fit.svg')
            plt.close()
            #also predict 'FS' mito density value
            fs_ves_pred = (firing_rate_dict['FS'] - intercept) / coefficient
            overview_df.loc['FS', key] = fs_ves_pred
            sns.scatterplot(data=overview_df, x=key, y='mean firing rate singing', hue = 'celltype', palette=ov_palette, legend=False)
            for x, y, t in zip(overview_df[key], overview_df['mean firing rate singing'], overview_df['celltype']):
                plt.text(x = x, y = y + 10, s = t)
            plt.xlabel(f'{key} [1/µm]', fontsize = fontsize)
            plt.ylabel('mean firing rate singing [Hz]', fontsize = fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.savefig(f'{f_name}/{key}_firing_rate_pred_withFS.png')
            plt.savefig(f'{f_name}/{key}_firing_rate_pred_withFS.svg')
            plt.close()
            # for plotted line
            start_xseq = np.min(overview_df[key])
            end_xseq = np.max(overview_df[key])
            num_xseq = len(overview_df)
            xseq = np.linspace(start_xseq, end_xseq, num=num_xseq)
            # plot scatterplot again with fitted line
            plt.plot(xseq, intercept + coefficient * xseq, color='#707070', lw=1.5, linestyle='dashed')
            sns.scatterplot(data=overview_df, x=key, y='mean firing rate singing', hue='celltype', palette=ov_palette,
                            legend=False)
            for x, y, t in zip(overview_df[key], overview_df['mean firing rate singing'], overview_df['celltype']):
                plt.text(x=x, y=y + 10, s=t)
            plt.xlabel(f'{key} [1/µm]', fontsize=fontsize)
            plt.ylabel('mean firing rate singing [Hz]', fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.savefig(f'{f_name}/{key}_firing_rate_pred_withFS_fit.png')
            plt.savefig(f'{f_name}/{key}_firing_rate_pred_withFS_fit.svg')
            plt.close()
            overview_df.to_csv(f'{f_name}/overview_df_with_preds_{key}.csv')

    log.info('Analysis done')

