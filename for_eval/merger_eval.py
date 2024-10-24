#get number and plot mergers

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
    cts_check = [4, 5, 6, 7, 8]
    cts_str_check = [ct_dict[ct] for ct in cts_check]
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/241024_j0251{version}_manual_nomsn_merger_eval"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'merger_eval_log', log_dir=f_name)
    log.info(f'Check mergers for celltypes: {cts_str_check}')

    #eval_path = 'cajal/scratch/users/arother/bio_analysis_results/general/' \
    #            '240723_j0251v6_all_cellids_for_exclusion/' \
     #                   '240826_Random MSN IDS_RM_check.csv'
    eval_path = 'cajal/scratch/users/arother/bio_analysis_results/general/' \
                '240723_j0251v6_all_cellids_for_exclusion/' \
                       '241024_all_full_cell_ids_no_msn_manuall_checks_final.csv'
    log.info(f'evaluation results loaded from {eval_path}')
    eval_df = pd.read_csv(eval_path)
    #only get celltyoes wanted
    eval_df = eval_df[np.in1d(eval_df['celltype'], cts_str_check)]
    unique_cts = np.unique(eval_df['celltype'])
    num_cells = len(eval_df)
    log.info(f'In total {num_cells} cells of these celltypes')
    ct_groups = eval_df.groupby('celltype')
    ov_columns = ['celltype', 'number total', 'number excluded', 'fraction excluded', 'number merger', 'fraction merger',
                  'number incomplete', 'fraction incomplete', 'number merger with neuron', 'fraction merger with neuron',
                  'number merger with glia', 'fraction merger with glia']
    overview_df = pd.DataFrame(columns=ov_columns)
    overview_df['celltype'] = unique_cts
    overview_df['number total'] = np.array(ct_groups.size())
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
    #save excluded and included ids
    exclude_ids = np.array(eval_excluded['cellid'])
    write_obj2pkl(analysis_params._merger_file_location, exclude_ids)
    include_ids = np.array(eval_df[eval_df['include?'] == 'y'])
    write_obj2pkl(f'{analysis_params.file_locations}/include_manual_checked_ids.pkl', include_ids)
    assert(len(include_ids) + len(exclude_ids) == len(eval_df))
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