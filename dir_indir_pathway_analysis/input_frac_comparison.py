#compare fraction of input in different cell types
if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.handler.basics import load_pkl2obj
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    import os as os
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import ranksums, kruskal
    from itertools import  combinations

    #global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"

    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    ct_dict = analysis_params.ct_dict(with_glia=False)
    global_params.wd = analysis_params.working_dir()
    #celltypes that are compared
    comp_cts = [9, 10, 11]
    color_key = 'RdTeINTv6'
    fontsize = 20
    #select which incoming an outgoing celltypes should be plottet extra as well
    input_ct = 7
    input_ct_str = ct_dict[input_ct]
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/240915_j0251{version}_input_comp_{input_ct_str}_INT_f{fontsize}"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('Input_conn_comp', log_dir=f_name)
    ct_colors = CelltypeColors(ct_dict = ct_dict)
    ct_palette = ct_colors.ct_palette(key=color_key)
    comp_ct_str = [ct_dict[ct] for ct in comp_cts]


    conn_filename = 'cajal/scratch/users/arother/bio_analysis_results/general/240411_j0251v6_cts_percentages_mcl_200_ax50_synprob_0.60_TePkBrNGF_annot_bw_fs_20'
    log.info(f'Step 1/4: Load connectivity data from {conn_filename}')
    conn_dict = load_pkl2obj(f'{conn_filename}/synapse_dict_per_ct.pkl')

    log.info(f'Step 2/4 generate dataframe for celltypes {comp_ct_str}')
    #use similar code as in connectivity_fraction_per_ct
    key_list = ['outgoing synapse sum size percentage', 'outgoing synapse sum size',
                'incoming synapse sum size percentage', 'incoming synapse sum size' ]
    celltypes = np.array([ct_dict[ct] for ct in ct_dict])
    num_cts = len(celltypes)
    axon_cts = analysis_params.axon_cts()
    non_ax_celltypes = celltypes[np.in1d(np.arange(0, num_cts), axon_cts) == False]
    number_cells_comp = [len(conn_dict[ct]['in cellids']) for ct in comp_cts]
    log.info(f'the follwing cell types will be compared {comp_ct_str} with these number of cells: {number_cells_comp}')
    number_cells_total = np.sum(number_cells_comp)

    if input_ct in axon_cts:
        columns = ['celltype', 'incoming synapse sum size percentage']

    else:
        columns = ['celltype', 'incoming synapse sum size percentage', 'outgoing synapse sum size percentage']
        number_cells_comp_out = [len(conn_dict[ct]['out cellids']) for ct in comp_cts]
        log.info(f'For outgoing cells there is this number of cellids: {number_cells_comp_out}')
    result_df_comp = pd.DataFrame(columns = columns, index = range(number_cells_total))

    for column in columns:
        if 'celltype' in column:
            continue
        start = 0
        for i, ct in enumerate(comp_cts):
            len_ids = len(conn_dict[ct][f'{column} of {input_ct_str}'])
            result_df_comp.loc[start: start + len_ids - 1, 'celltype'] = comp_ct_str[i]
            result_df_comp.loc[start: start + len_ids - 1, column] = conn_dict[ct][f'{column} of {input_ct_str}']
            start += number_cells_comp[i]
        result_df_comp = result_df_comp.astype({column: float})

    result_df_comp.to_csv(f'{f_name}/{input_ct_str}_conn_comp.csv')


    log.info('Step 3/4: Plot results')
    for column in columns:
        if 'celltype' in column:
            continue
        sns.boxplot(data=result_df_comp, x='celltype', y=column, palette=ct_palette)
        plt.title(f'{column} connectivity with {input_ct_str}')
        plt.ylim(0, 100)
        plt.ylabel(f'fraction synapse area with {input_ct_str}')
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.savefig(f'{f_name}/{input_ct_str}_{column}_comp_conn.png')
        plt.savefig(f'{f_name}/{input_ct_str}_{column}_comp_conn.svg')
        plt.close()

    log.info('Step 4/4: Get statistics')
    ct_groups = result_df_comp.groupby('celltype')
    group_comps = list(combinations(comp_ct_str, 2))
    ranksum_columns = [f'{gc[0]} vs {gc[1]}' for gc in group_comps]
    ranksum_group_df = pd.DataFrame(columns=ranksum_columns)
    for column in columns:
        if 'celltype' in column:
            continue
        if len(comp_cts) > 2:
            key_groups = [group[column].values for name, group in
                          result_df_comp.groupby('celltype')]
            kruskal_res = kruskal(*key_groups, nan_policy='omit')
            log.info(f'Kruskal Wallis test result for {column}: {kruskal_res}')
        for group in group_comps:
            ranksum_res = ranksums(ct_groups.get_group(group[0])[column], ct_groups.get_group(group[1])[column])
            ranksum_group_df.loc[f'{column} stats', f'{group[0]} vs {group[1]}'] = ranksum_res[0]
            ranksum_group_df.loc[f'{column} p-value', f'{group[0]} vs {group[1]}'] = ranksum_res[1]

    ranksum_group_df.to_csv(f'{f_name}/ranksum_results.csv')

    log.info('Analysis done')
