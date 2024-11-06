#based on organell volume density but compare only full cells and to cell volume
if __name__ == '__main__':
    from analysis_morph_helper import check_comp_lengths_ct
    from analysis_colors import CelltypeColors
    from analysis_morph_helper import get_org_density_volume_presaved
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
    ct_dict = analysis_params.ct_dict(with_glia=True)
    min_comp_len_cell = 200
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw'}
    color_key = 'GliaOPC'
    cls = CelltypeColors(ct_dict=ct_dict)
    ct_palette = cls.ct_palette(key=color_key)
    fontsize = 20
    #organelles = 'mi', 'vc', 'er', 'golgi
    organelle_key = 'mi'
    cts = [12, 13, 14, 17, 15, 3, 7]
    handpicked_glia = True
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/general/241025_j0251{version}_ct_{organelle_key}_vol_density_full_mcl_%i_%s_fs%i_nm" % (
        min_comp_len_cell, color_key, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'{organelle_key}_vol_density_ct_log', log_dir=f_name + '/logs/')
    log.info(f'get volume density {organelle_key} per celltype')
    log.info(
        "min_comp_len = %i for full cells, min_comp_len = %i for axons, colors = %s" % (
            min_comp_len_cell, color_key))
    log.info(f'use mean of {organelle_key} volume density for regression fit')
    if handpicked_glia:
        log.info('Manually selected glia cells will be used for glia cells in analysis')
        if 17 in cts:
            log.info('manually selected OPC are included in analysis')
            ct_dict[17] = 'OPC'
    else:
        if 17 in cts:
            raise ValueError('OPC can only be part of analysis when glia are manually selected.')
    known_mergers = analysis_params.load_known_mergers()
    axon_cts = analysis_params.axon_cts()
    cts_str = [ct_dict[ct] for ct in cts]

    glia_cts = analysis_params._glia_cts
    if np.any(np.in1d(cts, axon_cts)):
        raise ValueError('Analysis currently not enabled for projecting axons, only full cells')

    log.info('Step 1/4: Iterate over each celltypes check min length')
    suitable_ids_dict = {}
    all_suitable_ids = []
    all_cell_dict = {}
    all_suitable_cts = []
    for ct in cts:
        # only get cells with min_comp_len, MSN with max_comp_len or axons with min ax_len
        ct_str = ct_dict[ct]
        if handpicked_glia and ct in glia_cts:
            cellids = analysis_params.load_handpicked_ids(ct, ct_dict=ct_dict)
        else:
            cell_dict = analysis_params.load_cell_dict(ct)
            all_cell_dict[ct] = cell_dict
            cellids = np.array(list(cell_dict.keys()))
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            if ct in axon_cts:
                cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cell,
                                                axon_only=True, max_path_len=None)
            else:
                #astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
                #cellids = cellids[astro_inds]
                cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cell,
                                                    axon_only=False, max_path_len=None)
        cellids = np.sort(cellids)
        suitable_ids_dict[ct] = cellids
        all_suitable_ids.append(cellids)
        all_suitable_cts.append([[ct] for i in cellids])
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict[ct]))

    all_suitable_ids = np.concatenate(all_suitable_ids)
    all_suitable_cts = np.concatenate(all_suitable_cts)

    log.info(f'Step 2/4: Get {organelle_key} volume density per cell')
    #ellid, cached_so_ids, cached_so_rep_coord, cached_so_volume, full_cell_dict, k, min_comp_len = input
    # generate df for each cell
    pc_columns = ['cellid', 'celltype', f'total {organelle_key} volume density']
    percell_org_df = pd.DataFrame(columns=pc_columns, index=range(len(all_suitable_ids)))
    percell_org_df['cellid'] = all_suitable_ids
    percell_org_df['celltype'] = all_suitable_cts
    if organelle_key == 'er':
        sd_er = SegmentationDataset('er')
        org_ids = sd_er.ids
        org_sizes = sd_er.load_numpy_data('size')
    for i, ct in enumerate(cts):
        ct_str = ct_dict[ct]
        log.info(f'process {ct_str}')
        if organelle_key == 'er':
            ct_ind = np.in1d(org_ids, suitable_ids_dict[ct])
            #for er, the mapped cellid is the er id
            ct_org_map2ssvids = org_ids[ct_ind]
            ct_org_sizes = org_sizes[ct_ind]
        else:
            if ct in glia_cts:
                ct_org_ids = np.load(f'{analysis_params.file_locations}/{ct_dict[ct]}_{organelle_key}_ids.npy')
                ct_org_map2ssvids = np.load(
                    f'{analysis_params.file_locations}/{ct_dict[ct]}_{organelle_key}_mapping_ssv_ids.npy')
                ct_org_sizes = np.load(
                    f'{analysis_params.file_locations}/{ct_dict[ct]}_{organelle_key}_sizes.npy')
            else:
                ct_org_ids = np.load(f'{analysis_params.file_locations}/{ct_dict[ct]}_{organelle_key}_ids_fullcells.npy')
                ct_org_map2ssvids = np.load(f'{analysis_params.file_locations}/{ct_dict[ct]}_{organelle_key}_mapping_ssv_ids_fullcells.npy')
                ct_org_sizes = np.load(f'{analysis_params.file_locations}/{ct_dict[ct]}_{organelle_key}_sizes_fullcells.npy')
            #filter for suitable cellids
            ct_ind = np.in1d(ct_org_map2ssvids, suitable_ids_dict[ct])
            ct_org_ids = ct_org_ids[ct_ind]
            ct_org_map2ssvids = ct_org_map2ssvids[ct_ind]
            ct_org_sizes = ct_org_sizes[ct_ind]
        org_input = [[cellid, ct_org_map2ssvids, ct_org_sizes, organelle_key] for cellid in suitable_ids_dict[ct]]
        org_output = start_multiprocess_imap(get_org_density_volume_presaved, org_input)
        full_volume_density = np.array(org_output, dtype = float)
        #for percell df
        ct_inds = np.in1d(percell_org_df['cellid'], suitable_ids_dict[ct])
        percell_org_df.loc[ct_inds, f'total {organelle_key} volume density'] = full_volume_density

    percell_org_df = percell_org_df.astype({f'total {organelle_key} volume density': float})
    percell_org_df.to_csv(f'{f_name}/percell_{organelle_key}_volume_density')

    #create overview df for summary params
    # save mean, median and std for all parameters per ct
    ct_str = np.unique(percell_org_df['celltype'])
    ct_groups = percell_org_df.groupby('celltype')
    overview_df = pd.DataFrame(index=ct_str)
    overview_df['celltype'] = ct_str
    overview_df['numbers'] = ct_groups.size()
    param = f'total {organelle_key} volume density'
    overview_df[f'{param} mean'] = ct_groups[param].mean()
    overview_df[f'{param} std'] = ct_groups[param].std()
    overview_df[f'{param} median'] = ct_groups[param].median()
    overview_df.to_csv(f'{f_name}/overview_df_{organelle_key}_den.csv')


    percell_org_df.to_csv(f'{f_name}/percell_df_{organelle_key}_den.csv')

    log.info('Step 3/4: Calculate statistics and plot results')
    group_comps = list(combinations(ct_str, 2))
    ranksum_columns = [f'{gc[0]} vs {gc[1]}' for gc in group_comps]
    ranksum_group_df = pd.DataFrame(columns=ranksum_columns)

    param_groups = [group[param].values for name, group in
                  percell_org_df.groupby('celltype')]
    kruskal_res = kruskal(*param_groups, nan_policy='omit')
    log.info(f'Kruskal Wallis test result for {param_groups}: {kruskal_res}')
    if kruskal_res[0] < 0.05:
        # ranksum results
        for group in group_comps:
            ranksum_res = ranksums(ct_groups.get_group(group[0])[param], ct_groups.get_group(group[1])[param])
            ranksum_group_df.loc[f'{param} stats', f'{group[0]} vs {group[1]}'] = ranksum_res[0]
            ranksum_group_df.loc[f'{param} p-value', f'{group[0]} vs {group[1]}'] = ranksum_res[1]
        ranksum_group_df.to_csv(f'{f_name}/ranksum_results.csv')

    #plot results
    ylabel = f'{param} [µm³/µm³]'
    sns.boxplot(data=percell_org_df, x='celltype', y=param, palette=ct_palette, order=ct_str)
    plt.title(param)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xlabel('celltype', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/{organelle_key}_vol_density_box.png')
    plt.savefig(f'{f_name}/{organelle_key}_vol_density_box.svg')
    plt.close()
    sns.stripplot(data=percell_org_df, x='celltype', y=param, color='black', alpha=0.2,
                  dodge=True, size=2, order=ct_str)
    sns.violinplot(data=percell_org_df, x='celltype', y=param, palette=ct_palette, inner="box", order=ct_str)
    plt.title(param)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xlabel('celltype', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/{organelle_key}_vol_density_violin.png')
    plt.savefig(f'{f_name}/{organelle_key}_vol_density_violin.svg')
    plt.close()


    log.info('Analysis done')