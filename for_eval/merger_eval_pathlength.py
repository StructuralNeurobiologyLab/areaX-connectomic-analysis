#get pathlength of all cells used and calculate number of mergers / pathlength

if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from syconn.handler.basics import write_obj2pkl

    version = 'v6'
    analysis_params = Analysis_Params(version = version)
    ct_dict = analysis_params.ct_dict(with_glia = False)
    fontsize = 20
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/241212_j0251{version}_merger_pathlength_eval"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'merger_eval_pathlength_log', log_dir=f_name)
    eval_path1 = 'cajal/scratch/users/arother/bio_analysis_results/general/' \
                '240723_j0251v6_all_cellids_for_exclusion/' \
                       '241212_Random MSN IDS_RM_check_merger_numbers.csv'
    eval_path2 = 'cajal/scratch/users/arother/bio_analysis_results/general/' \
                '240723_j0251v6_all_cellids_for_exclusion/' \
                       '241212_all_full_cell_ids_no_msn_manuall_checks_final_num_mergers.csv'
    eval_path3 = 'cajal/scratch/users/arother/bio_analysis_results/general/' \
                '240723_j0251v6_all_cellids_for_exclusion/' \
                       '241212_rndm_INT_ids_merger _rm_ar_num_mergers.csv'

    log.info(f'evaluation results loaded from {eval_path1}, {eval_path2}, {eval_path3}')
    eval_df1 = pd.read_csv(eval_path1)
    eval_df2 = pd.read_csv(eval_path2)
    eval_df3 = pd.read_csv(eval_path3)
    eval_df = pd.concat([eval_df1, eval_df2, eval_df3], ignore_index=True)
    unique_cts = np.unique(eval_df['celltype'])
    num_cells = len(eval_df)
    log.info(f'In total {num_cells} cells were inspected from the following celltypes {unique_cts}')
    #get total pathlength for each cell
    remap_dict_ct = {ct_dict[ct]: ct for ct in ct_dict.keys()}
    for ct_str in unique_cts:
        cell_dict = analysis_params.load_cell_dict(remap_dict_ct[ct_str])
        cellids_series = eval_df['cellid'][eval_df['celltype'] == ct_str]
        cellids_ct = np.array(cellids_series)
        inds_ct = cellids_series.index
        for ind, cellid in zip(inds_ct, cellids_ct):
            total_pathlength_cell = cell_dict[cellid]['complete pathlength']
            eval_df.loc[ind, 'pathlength'] = total_pathlength_cell

    total_num_mergers = np.sum(eval_df['number of mergers'])
    #in mm
    total_pathlength = np.sum(eval_df['pathlength']) / 1000
    mergers_per_pathlength_total = total_num_mergers / total_pathlength
    log.info(f'In total there are {total_num_mergers} mergers, and all cells add up to a pathlength of {total_pathlength:.2f} mm,'
             f'which means there are {mergers_per_pathlength_total:.4f} mergers/ mm pathlength.')

    log.info('Get overview params per celltype')

    ov_columns = ['celltype', 'number total', 'number excluded', 'fraction excluded', 'number merger', 'fraction merger',
                  'number incomplete', 'fraction incomplete', 'number merger with neuron', 'fraction merger with neuron',
                  'number merger with glia', 'fraction merger with glia', 'number of mergers', 'summed pathlength mm', 'mergers per mm']
    overview_df = pd.DataFrame(columns=ov_columns)
    overview_df['celltype'] = unique_cts
    ct_groups = eval_df.groupby('celltype')
    overview_df['number total'] = np.array(ct_groups.size())
    overview_df['number of mergers'] = np.array(ct_groups['number of mergers'].sum())
    overview_df['summed pathlength mm'] = np.array(ct_groups['pathlength'].sum()/1000)
    overview_df['mergers per mm'] = overview_df['number of mergers'] / overview_df['summed pathlength mm']
    #get number of excluded ones
    eval_excluded = eval_df[eval_df['include?'] == 'n']
    exclude_reasons = np.unique(eval_excluded['exclude because'])
    excl_groups = eval_excluded.groupby('exclude because')

    log.info(f'{len(eval_excluded)} cells ({100*len(eval_excluded)/len(eval_df):.2f} %) were excluded in total due to {exclude_reasons}, with this '
             f'number of cells: {excl_groups.size()} ({100 * excl_groups.size()/len(eval_df)} %)')
    ct_groups_excluded = eval_excluded.groupby('celltype')
    unique_eval_cts = np.unique(eval_excluded['celltype'])
    if len(unique_eval_cts) < len(unique_cts):
        excl_inds = np.in1d(unique_cts, unique_eval_cts)
    else:
        excl_inds = range(len(unique_cts))
    overview_df.loc[excl_inds, 'number excluded'] = np.array(ct_groups_excluded.size())
    overview_df.loc[excl_inds,'fraction excluded'] = overview_df['number excluded'] / overview_df['number total']
    #get cell numbers that are merger
    merger_df = eval_excluded[eval_excluded['merger?'] == 'y']
    ct_merge_groups = merger_df.groupby('celltype')
    unique_merge_cts = np.unique(merger_df['celltype'])
    if len(unique_merge_cts) < len(unique_cts):
        excl_inds = np.in1d(unique_cts, unique_merge_cts)
    else:
        excl_inds = range(len(unique_cts))
    overview_df.loc[excl_inds, 'number merger'] = np.array(ct_merge_groups.size())
    overview_df.loc[excl_inds, 'fraction merger'] = overview_df['number merger'] / overview_df['number total']
    #get number of incomplete cells
    incomp_df = eval_excluded[eval_excluded['exclude because'] == 'not all compartments']
    ct_in_groups = incomp_df.groupby('celltype')
    unique_in_cts = np.unique(incomp_df['celltype'])
    if len(unique_in_cts) < len(unique_cts):
        excl_inds = np.in1d(unique_cts, unique_in_cts)
    else:
        excl_inds = range(len(unique_cts))
    overview_df.loc[excl_inds, 'number incomplete'] = np.array(ct_in_groups.size())
    overview_df.loc[excl_inds, 'fraction incomplete'] = overview_df['number incomplete'] / overview_df['number total']
    #get number of merger with cells and glia separately
    merger_glia_df = merger_df[merger_df['merger with'] == 'glia']
    ct_glia_groups = merger_glia_df.groupby('celltype')
    unique_glia_cts = np.unique(merger_glia_df['celltype'])
    if len(unique_glia_cts) < len(unique_cts):
        excl_inds = np.in1d(unique_cts, unique_glia_cts)
    else:
        excl_inds = range(len(unique_cts))
    overview_df.loc[excl_inds, 'number merger with glia'] = np.array(ct_glia_groups.size())
    overview_df.loc[excl_inds, 'fraction merger with glia'] = overview_df['number merger with glia'] / overview_df['number total']
    merger_cell_df = merger_df[merger_df['merger with'] != 'glia']
    ct_mc_groups = merger_cell_df.groupby('celltype')
    unique_mc_cts = np.unique(merger_cell_df['celltype'])
    if len(unique_in_cts) < len(unique_cts):
        excl_inds = np.in1d(unique_cts, unique_mc_cts)
    else:
        excl_inds = range(len(unique_cts))
    overview_df.loc[excl_inds, 'number merger with neuron'] = np.array(ct_mc_groups.size())
    overview_df.loc[excl_inds, 'fraction merger with neuron'] = overview_df['number merger with neuron'] / overview_df['number total']
    overview_df.to_csv(f'{f_name}/overview_df.csv')

    #get merger categories
    merger_cats = np.unique(merger_df['merger with'])
    merger_cat_df = pd.DataFrame(columns=['number', 'fraction mergers', 'fraction all', 'merger with'], index= range(len(merger_cats)))
    merger_cat_df['merger with'] = merger_cats
    merger_groups = merger_df.groupby('merger with')
    merger_cat_df['number'] = np.array(merger_groups.size())
    merger_cat_df['fraction mergers'] = np.array(merger_groups.size()) / len(merger_df)
    merger_cat_df['fraction all'] = np.array(merger_groups.size()) / num_cells
    merger_cat_df.to_csv(f'{f_name}/merger_categories.csv')

    log.info('Plot results as barplot')
    #plot result as barplot
    sns.barplot(data = overview_df, y = 'number excluded', x = 'celltype')
    plt.ylabel('number excluded cells', fontsize = fontsize)
    plt.xlabel('celltype', fontsize = fontsize)
    plt.title('All eval cells')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/number_excl.png')
    plt.savefig(f'{f_name}/number_excl.svg')
    plt.close()
    sns.barplot(data=overview_df, y='number excluded', x='celltype')
    plt.ylabel('fraction excluded cells', fontsize=fontsize)
    plt.xlabel('celltype', fontsize=fontsize)
    plt.title('All eval cells')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/frac_excl.png')
    plt.savefig(f'{f_name}/frac_excl.svg')
    plt.close()
    sns.barplot(data=merger_cat_df, x='merger with', y='number')
    plt.ylabel('number of cells', fontsize=fontsize)
    plt.xlabel('merger category', fontsize=fontsize)
    plt.title('merger category')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/number_mergers_cats.png')
    plt.savefig(f'{f_name}/number_mergers_cats.svg')
    plt.close()
    sns.barplot(data=merger_cat_df, x='merger with', y='fraction mergers')
    plt.ylabel('fraction of cells', fontsize=fontsize)
    plt.xlabel('merger category', fontsize=fontsize)
    plt.title('merger category fractions')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/frac_mergers_cats.png')
    plt.savefig(f'{f_name}/frac_mergers_cats.svg')
    plt.close()
    log.info('Analysis finished')