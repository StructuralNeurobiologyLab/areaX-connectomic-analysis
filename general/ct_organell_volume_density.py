#based on ct_mito_violume_density
#write function for every organell which has a mesh and is a segmentation dataset
#ct_mito_volume_density is obsolete then

if __name__ == '__main__':
    from analysis_morph_helper import check_comp_lengths_ct
    from analysis_colors import CelltypeColors
    from analysis_morph_helper import get_percell_organell_volume_density, get_organelle_comp_density_presaved
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
    #from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm

    #ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
     #          10: "NGF"}
    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    global_params.wd = analysis_params.working_dir()
    with_glia = False
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    full_cells_only = True
    min_comp_len_cell = 200
    min_comp_len_ax = 200
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw'}
    color_key = 'TePkBrNGF'
    fontsize = 20
    #organelles = 'mi', 'vc', 'er', 'golgi
    organelle_key = 'golgi'
    comp_dict = {0:'dendrite', 1:'axon', 2:'soma'}
    compartment = 2
    comp_str = comp_dict[compartment]
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/general/240904_j0251{version}_ct_{organelle_key}_{comp_str}_vol_density_mcl_%i_ax%i_%s_fs%i" % (
        min_comp_len_cell, min_comp_len_ax, color_key, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'{organelle_key}_vol_density_ct_log', log_dir=f_name + '/logs/')
    log.info(f'get volume density {organelle_key} per celltype')
    log.info(
        "min_comp_len = %i for full cells, min_comp_len = %i for axons, colors = %s" % (
            min_comp_len_cell, min_comp_len_ax, color_key))
    log.info(f'use mean of {organelle_key} volume density for regression fit')
    if full_cells_only:
        log.info('Plot for full cells only')
    if compartment == 2:
        log.info('Soma selected as compartment. Density will be calculated as sum of organelle volume per soma volume. The soma volume is calculated from the soma radius.'
                 'Total volume density will be all organelles per total pathlength.')
    else:
        log.info(f'{comp_str} selected as compartment. Density will be calculated as sum of organelle volume per compartment skeleton length. '
                 f'Total volume density will be all organelles per total pathlength.')
    known_mergers = analysis_params.load_known_mergers()
    misclassified_asto_ids = analysis_params.load_potential_astros()
    axon_cts = analysis_params.axon_cts()
    num_cts = analysis_params.num_cts(with_glia=with_glia)
    np_presaved_loc = analysis_params.file_locations
    if full_cells_only:
        ct_types = analysis_params.load_celltypes_full_cells()
        #ct_types = ct_types[1:]
    else:
        ct_types = np.arange(0, num_cts)
        sd_orgssv = SegmentationDataset(organelle_key, working_dir=global_params.config.working_dir)
        cached_org_ids = sd_orgssv.ids
        cached_org_volumes = sd_orgssv.load_numpy_data("size")
    ct_str_list = analysis_params.ct_str(with_glia=with_glia)
    cls = CelltypeColors(ct_dict= ct_dict)
    #ct_palette = cls.ct_palette(key = color_key)
    if with_glia:
        glia_cts = analysis_params._glia_cts
    firing_rate_dict = {'DA': 15, 'MSN': 1.58, 'LMAN': 34.9, 'HVC': 1, 'TAN': 65.1, 'GPe': 135, 'GPi': 258, 'FS': 19.1, 'LTS': 35.8}

    log.info('Step 1/4: Iterate over each celltypes check min length')
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
            astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
            cellids = cellids[astro_inds]
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cell,
                                                axon_only=False, max_path_len=None)
        cellids = np.sort(cellids)
        suitable_ids_dict[ct] = cellids
        all_suitable_ids.append(cellids)
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict[ct]))

    all_suitable_ids = np.concatenate(all_suitable_ids)

    log.info(f'Step 2/4: Get {organelle_key} volume density per cell')
    #ellid, cached_so_ids, cached_so_rep_coord, cached_so_volume, full_cell_dict, k, min_comp_len = input
    # generate df for each cell
    pc_columns = ['cellid', 'mean firing rate singing', f'total {organelle_key} volume density', f'{comp_str} {organelle_key} volume density']
    percell_org_df = pd.DataFrame(columns=pc_columns, index=range(len(all_suitable_ids)))
    percell_org_df['cellid'] = all_suitable_ids
    for i, ct in enumerate(ct_types):
        ct_str = ct_dict[ct]
        log.info(f'process {ct_str}')
        try:
            firing_value = firing_rate_dict[ct_str]
        except KeyError:
            firing_value = np.nan
        if ct in axon_cts:
            if compartment != 1:
                continue
            org_input = [[cellid, cached_org_ids, cached_org_volumes, all_cell_dict[ct][cellid], True, organelle_key] for cellid in suitable_ids_dict[ct]]
            org_output = start_multiprocess_imap(get_percell_organell_volume_density, org_input)
            org_output = np.array(org_output)
            comp_volume_density = org_output
            full_volume_density = comp_volume_density
        else:
            #To DO: write function to cache all organelles for full cells with organelle_key
            ct_org_ids = np.load(f'{np_presaved_loc}/{ct_dict[ct]}_{organelle_key}_ids_fullcells.npy')
            ct_org_map2ssvids = np.load(f'{np_presaved_loc}/{ct_dict[ct]}_{organelle_key}_mapping_ssv_ids_fullcells.npy')
            ct_org_axoness = np.load(f'{np_presaved_loc}/{ct_dict[ct]}_{organelle_key}_axoness_coarse_fullcells.npy')
            ct_org_sizes = np.load(f'{np_presaved_loc}/{ct_dict[ct]}_{organelle_key}_sizes_fullcells.npy')
            #filter for suitable cellids
            ct_ind = np.in1d(ct_org_map2ssvids, suitable_ids_dict[ct])
            ct_org_ids = ct_org_ids[ct_ind]
            ct_org_map2ssvids = ct_org_map2ssvids[ct_ind]
            ct_org_sizes = ct_org_sizes[ct_ind]
            ct_org_axoness = ct_org_axoness[ct_ind]
            org_input = [[cellid, ct_org_map2ssvids, ct_org_sizes, ct_org_axoness, all_cell_dict[ct][cellid], compartment] for cellid in suitable_ids_dict[ct]]
            org_output = start_multiprocess_imap(get_organelle_comp_density_presaved, org_input)
            org_output = np.array(org_output, dtype = float)
            comp_volume_density = org_output[:, 0]
            full_volume_density = org_output[:, 1]
        #for percell df
        ct_inds = np.in1d(percell_org_df['cellid'], suitable_ids_dict[ct])
        percell_org_df.loc[ct_inds, 'celltype'] = ct_str
        percell_org_df.loc[ct_inds, 'mean firing rate singing'] = firing_value
        percell_org_df.loc[ct_inds, f'total {organelle_key} volume density'] = full_volume_density
        percell_org_df.loc[ct_inds, f'{comp_str} {organelle_key} volume density'] = comp_volume_density

    #create overview df for summary params
    # save mean, median and std for all parameters per ct
    ct_str = np.unique(percell_org_df['celltype'])
    ct_groups = percell_org_df.groupby('celltype')
    overview_df = pd.DataFrame(index=ct_str)
    overview_df['celltype'] = ct_str
    overview_df['numbers'] = ct_groups.size()
    param_list = ['mean firing rate singing', f'total {organelle_key} volume density',
                  f'{comp_str} {organelle_key} volume density']
    for key in param_list:
        if 'firing rate' in key:
            overview_df[key] = ct_groups[key].mean()
        else:
            overview_df[f'{key} mean'] = ct_groups[key].mean()
            overview_df[f'{key} std'] = ct_groups[key].std()
            overview_df[f'{key} median'] = ct_groups[key].median()

    overview_df = overview_df.astype(
        {'mean firing rate singing': float, f'{comp_str} {organelle_key} volume density mean': float,
         f'total {organelle_key} volume density mean': float, f'{comp_str} {organelle_key} volume density median': float,
         f'total {organelle_key} volume density median': float})
    overview_df.to_csv(f'{f_name}/overview_df_{organelle_key}_den.csv')
    percell_org_df = percell_org_df.astype(
        {'mean firing rate singing': float, f'{comp_str} {organelle_key} volume density': float,
         f'total {organelle_key} volume density': float})
    percell_org_df.to_csv(f'{f_name}/percell_df_{organelle_key}_den.csv')

    log.info('Step 3/4: Calculate statistics and plot results')
    group_comps = list(combinations(ct_str, 2))
    ranksum_columns = [f'{gc[0]} vs {gc[1]}' for gc in group_comps]
    ranksum_group_df = pd.DataFrame(columns=ranksum_columns)
    known_values_only_percell = percell_org_df.dropna()

    for key in percell_org_df.keys():
        if organelle_key in key:
            key_groups = [group[key].values for name, group in
                                percell_org_df.groupby('celltype')]
            medians = [np.median(kg) for kg in key_groups]
            median_order = np.unique(percell_org_df['celltype'])[np.argsort(medians)]
            ct_colors = cls.colors[color_key]
            ct_palette = {median_order[i]: ct_colors[i] for i in range(len(median_order))}
            kruskal_res = kruskal(*key_groups, nan_policy='omit')
            log.info(f'Kruskal Wallis test result for {key}: {kruskal_res}')
            #ranksum results
            for group in group_comps:
                ranksum_res = ranksums(ct_groups.get_group(group[0])[key], ct_groups.get_group(group[1])[key])
                ranksum_group_df.loc[f'{key} stats', f'{group[0]} vs {group[1]}'] = ranksum_res[0]
                ranksum_group_df.loc[f'{key} p-value',f'{group[0]} vs {group[1]}'] = ranksum_res[1]
            if 'soma' in key:
                ylabel = f'{key} [µm³/µm³]'
            else:
                ylabel = f'{key} [µm³/µm]'
            #plot with increasing median as boxplot and violinplot
            sns.boxplot(data=percell_org_df, x='celltype', y=key, palette=ct_palette, order=median_order)
            plt.title(key)
            plt.ylabel(ylabel, fontsize = fontsize)
            plt.xlabel('celltype', fontsize = fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.savefig(f'{f_name}/{key}_box.png')
            plt.savefig(f'{f_name}/{key}_box.svg')
            plt.close()
            if full_cells_only:
                sns.stripplot(data=percell_org_df, x='celltype', y=key, color='black', alpha=0.2,
                              dodge=True, size=2, order=median_order)
            sns.violinplot(data=percell_org_df, x='celltype', y=key, palette=ct_palette, inner="box", order=median_order)
            plt.title(key)
            plt.ylabel(ylabel, fontsize=fontsize)
            plt.xlabel('celltype', fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.savefig(f'{f_name}/{key}_violin.png')
            plt.savefig(f'{f_name}/{key}_violin.svg')
            plt.close()

    ranksum_group_df.to_csv(f'{f_name}/ranksum_results.csv')

    log.info(f'Step 4/4: Plot mean firing rates vs {organelle_key} density')
    #plot once with and once without unknown literature values
    known_values_only_ov = overview_df.dropna()
    known_cts_only = np.unique(known_values_only_ov['celltype'])
    fs_dict = {'celltype': 'FS', 'mean firing rate singing': firing_rate_dict['FS']}
    fs_df = pd.DataFrame(fs_dict, index = ['FS'])
    overview_df = pd.concat([overview_df, fs_df])
    ov_palette = {known_cts_only[i]: '#232121' for i in range(len(known_cts_only))}
    ov_palette['FS'] = '#232121'
    for ct in overview_df['celltype']:
        if ct not in known_cts_only and ct != 'FS':
            ov_palette[ct] = '#15AEAB'


    for key in overview_df.keys():
        if ('mean' in key or 'median' in key) and organelle_key in key:
            if 'soma' in key:
                xlabel = f'{key} [µm³/µm³]'
            else:
                xlabel = f'{key} [µm³/µm]'
            sns.scatterplot(data= known_values_only_ov, x = key, y = 'mean firing rate singing', hue = 'celltype', palette=ov_palette, legend=False)
            for x, y, t in zip(known_values_only_ov[key], known_values_only_ov['mean firing rate singing'], known_values_only_ov['celltype']):
                plt.text(x = x, y = y + 10, s = t)
            plt.xlabel(xlabel, fontsize = fontsize)
            plt.ylabel('mean firing rate singing [Hz]', fontsize = fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.savefig(f'{f_name}/{key}_firing_rate_known_only.png')
            plt.savefig(f'{f_name}/{key}_firing_rate_known_only.svg')
            plt.close()
            #calculate spearman nr for mean/median of value
            spearman_res = spearmanr(known_values_only_ov[key],
                                     known_values_only_ov['mean firing rate singing'], nan_policy='omit')
            spearman_cts = np.unique(known_values_only_ov['celltype'])
            log.info(f'Spearman correlation test result for {key}: {spearman_res}, for these celltypes {spearman_cts}')
            #percell_key = key.split(' ')[1:]
            #percell_key = ' '.join(percell_key)
            #lin reg code adopted from ChatGPT
            X_with_intercept = sm.add_constant(known_values_only_ov[key])
            reg_model = sm.OLS(known_values_only_ov['mean firing rate singing'], X_with_intercept)
            #reg_model.fit(np.array(known_values_only_ov[key]).reshape(-1, 1), known_values_only_ov['mean firing rate singing'])
            #get coeff and intercept
            results = reg_model.fit()
            #coefficient = reg_model.coef_
            #intercept = reg_model.intercept_
            coefficient = results.params[key]
            intercept = results.params['const']
            log.info(f'Regression coefficient for {key} and mean firing rate: {coefficient}, intercept: {intercept}')
            log.info(f'Summary:  {results.summary()}')
            #for plotted line
            start_xseq = np.min(known_values_only_ov[key])
            end_xseq = np.max(known_values_only_ov[key])
            num_xseq = len(known_values_only_ov)
            xseq = np.linspace(start_xseq, end_xseq, num=num_xseq)
            #plot scatterplot again with fitted line
            plt.plot(xseq, intercept + coefficient * xseq, color = '#707070', lw = 1.5, linestyle = 'dashed')
            sns.scatterplot(data=known_values_only_ov, x=key, y='mean firing rate singing', hue = 'celltype', palette=ov_palette, legend=False)
            for x, y, t in zip(known_values_only_ov[key], known_values_only_ov['mean firing rate singing'],
                               known_values_only_ov['celltype']):
                plt.text(x=x, y=y + 10, s=t)
            plt.xlabel(xlabel, fontsize=fontsize)
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
                firing_pred = coefficient*key_ct_value + intercept
                overview_df.loc[ct_str, 'mean firing rate singing'] = firing_pred
            sns.scatterplot(data=overview_df, x=key, y='mean firing rate singing', hue = 'celltype', palette=ov_palette, legend=False)
            for x, y, t in zip(overview_df[key], overview_df['mean firing rate singing'], overview_df['celltype']):
                plt.text(x = x, y = y + 10, s = t)
            plt.xlabel(xlabel, fontsize = fontsize)
            plt.ylabel('mean firing rate singing [Hz]', fontsize = fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.savefig(f'{f_name}/{key}_firing_rate_pred.png')
            plt.savefig(f'{f_name}/{key}_firing_rate_pred.svg')
            plt.close()
            #plot with line
            # for plotted line
            start_xseq = np.min(overview_df[key])
            end_xseq = np.max(overview_df[key])
            num_xseq = len(overview_df)
            xseq = np.linspace(start_xseq, end_xseq, num=num_xseq)
            plt.plot(xseq, intercept + coefficient * xseq, color='#707070', lw=1.5, linestyle='dashed')
            sns.scatterplot(data=overview_df, x=key, y='mean firing rate singing', hue = 'celltype', palette=ov_palette, legend=False)
            for x, y, t in zip(overview_df[key], overview_df['mean firing rate singing'], overview_df['celltype']):
                plt.text(x = x, y = y + 10, s = t)
            plt.xlabel(xlabel, fontsize=fontsize)
            plt.ylabel('mean firing rate singing [Hz]', fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.savefig(f'{f_name}/{key}_firing_rate_pred_fit.png')
            plt.savefig(f'{f_name}/{key}_firing_rate_pred_fit.svg')
            plt.close()
            #also predict 'FS' org density value
            fs_org_pred = (firing_rate_dict['FS'] - intercept) / coefficient
            overview_df.loc['FS', key] = fs_org_pred
            sns.scatterplot(data=overview_df, x=key, y='mean firing rate singing', hue = 'celltype', palette=ov_palette, legend=False)
            for x, y, t in zip(overview_df[key], overview_df['mean firing rate singing'], overview_df['celltype']):
                plt.text(x = x, y = y + 10, s = t)
            plt.xlabel(xlabel, fontsize = fontsize)
            plt.ylabel('mean firing rate singing [Hz]', fontsize = fontsize)
            plt.savefig(f'{f_name}/{key}_firing_rate_pred_withFS.png')
            plt.savefig(f'{f_name}/{key}_firing_rate_pred_withFS.svg')
            plt.close()
            start_xseq = np.min(overview_df[key])
            end_xseq = np.max(overview_df[key])
            num_xseq = len(overview_df)
            xseq = np.linspace(start_xseq, end_xseq, num=num_xseq)
            plt.plot(xseq, intercept + coefficient * xseq, color='#707070', lw=1.5, linestyle='dashed')
            sns.scatterplot(data=overview_df, x=key, y='mean firing rate singing', hue='celltype', palette=ov_palette, legend=False)
            for x, y, t in zip(overview_df[key], overview_df['mean firing rate singing'], overview_df['celltype']):
                plt.text(x=x, y=y + 10, s=t)
            plt.xlabel(x, fontsize=fontsize)
            plt.ylabel('mean firing rate singing [Hz]', fontsize=fontsize)
            plt.savefig(f'{f_name}/{key}_firing_rate_pred_withFS_fit.png')
            plt.savefig(f'{f_name}/{key}_firing_rate_pred_withFS_fit.svg')
            plt.close()
            overview_df.to_csv(f'{f_name}/overview_df_with_preds_{key}.csv')

    log.info('Analysis done')