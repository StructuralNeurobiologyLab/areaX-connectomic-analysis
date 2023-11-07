#compare manually measured soma diameters to calculated ones

if __name__ == '__main__':
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from syconn.handler.config import initialize_logging
    import os as os
    import pandas as pd
    from syconn.mp.mp_utils import start_multiprocess_imap
    from scipy.stats import kruskal, ranksums
    from itertools import combinations
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params

    f_name = "cajal/scratch/users/arother/bio_analysis_results/for_eval/231107_j0251v5_soma_diameter_comp"
    bio_params = Analysis_Params(working_dir=global_params.wd, version='v5')
    ct_dict = bio_params.ct_dict()
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('Compare manual vs calculated soma diameters', log_dir=f_name + '/logs/')
    log.info('Cells randomly selected (min_comp_len = 200) and measured with several different methods; one of them RM manually measured')
    comp_folder = "cajal/scratch/users/arother/bio_analysis_results/for_eval/230809_rndm_cts_soma_diameter_comparison/"
    log.info(f'Files loaded from {comp_folder}, diameters in µm')

    log.info('Step 1/3: Load result and create new DataFrame')
    comp_table_path = f'{comp_folder}/231107_rndm_soma_diamters_comparison.csv'
    soma_diameter_comp_df = pd.read_csv(comp_table_path)
    num_cells = len(soma_diameter_comp_df)
    methods = soma_diameter_comp_df.columns[3:]
    num_methods = len(methods)
    diameter_results_df = pd.DataFrame(columns=['cellid', 'celltype', 'diameter', 'method'], index = range(num_cells * num_methods))
    for i, met in enumerate(methods):
        diameter_results_df.loc[i * num_cells: (i + 1) * num_cells - 1, 'cellid'] = np.array(soma_diameter_comp_df['cellid'])
        diameter_results_df.loc[i * num_cells: (i + 1) * num_cells - 1, 'celltype'] = np.array(soma_diameter_comp_df['celltype'])
        diameter_results_df.loc[i * num_cells: (i + 1) * num_cells - 1, 'diameter'] = np.array(soma_diameter_comp_df[met])
        diameter_results_df.loc[i * num_cells: (i + 1) * num_cells - 1, 'method'] = met

    diameter_results_df.to_csv(f'{f_name}/diameter_results_df.csv')

    log.info('Step 2/3: Calculate statistics')
    #calculate overview params such as difference from manual
    deviation_df_µm = pd.DataFrame(columns = methods[1:], index = range(num_cells))
    deviation_df_perc = pd.DataFrame(columns = methods[1:], index = range(num_cells))
    for i, met in enumerate(methods):
        if i == 0:
            continue
        deviation_df_µm[met] = soma_diameter_comp_df[met]  - soma_diameter_comp_df[methods[0]]
        deviation_df_perc[met] = 100 * (soma_diameter_comp_df[met]  - soma_diameter_comp_df[methods[0]]) / soma_diameter_comp_df[methods[0]]
        diameter_results_df.loc[i * num_cells: (i + 1) * num_cells - 1, 'diff µm'] = np.array(deviation_df_µm[met])
        diameter_results_df.loc[i * num_cells: (i + 1) * num_cells - 1, 'diff perc'] = np.array(deviation_df_perc[met])

    deviation_df_µm.to_csv(f'{f_name}/diff2manual_µm.csv')
    deviation_df_perc.to_csv(f'{f_name}/diff2manual_perc.csv')
    diameter_results_df.to_csv(f'{f_name}/diameter_results_df.csv')
    diff_summary_df = pd.DataFrame(columns=methods[1:], index = ['mean diff µm', 'median diff µm', 'median diff perc', 'mean diff perc'])
    for met in deviation_df_µm.columns:
        diff_summary_df.loc['mean diff µm', met] = deviation_df_µm[met].mean()
        diff_summary_df.loc['median diff µm', met] = deviation_df_µm[met].median()
        diff_summary_df.loc['mean diff perc', met] = deviation_df_perc[met].mean()
        diff_summary_df.loc['median diff perc', met] = deviation_df_perc[met].median()
    diff_summary_df.to_csv(f'{f_name}/diff_summary.csv')
    #kruskal-wallis test on different methods
    #if significant ranksum test between pairs of methods
    method_groups = [group['diameter'].values for name, group in
                      diameter_results_df.groupby('method')]
    kruskal_res = kruskal(*method_groups, nan_policy='omit')
    log.info(f'Kruskal results on different methods: stats = {kruskal_res[0]}, p-value = {kruskal_res[1]}')
    if kruskal_res[1] < 0.05:
        group_comps = list(combinations(range(len(methods)), 2))
        ranksum_rows = [f'{methods[gc[0]]} vs {methods[gc[1]]}' for gc in group_comps]
        ranksum_res_df = pd.DataFrame(columns=['stats', 'p-value'], index=ranksum_rows)
        for gc in group_comps:
            ranksum_res = ranksums(method_groups[gc[0]], method_groups[gc[1]])
            ranksum_res_df.loc[f'stats', f'{methods[gc[0]]} vs {methods[gc[1]]}'] = ranksum_res[0]
            ranksum_res_df.loc[f'p-value', f'{methods[gc[0]]} vs {methods[gc[1]]}'] = ranksum_res[1]
        ranksum_res_df.to_csv(f'{f_name}/ranksum_results.csv')

    log.info('Step 3/3: Plot results')
    #make once plot with different methods only
    sns.boxplot(data = diameter_results_df, y = 'diameter', x = 'method')
    plt.ylabel('diameter [µm]')
    plt.title('Comparison of diameter values for different measurement methods')
    plt.savefig(f'{f_name}/method_comp_box.png')
    plt.savefig(f'{f_name}/method_comp_box.svg')
    plt.close()
    #plot differences
    diff_res_df = diameter_results_df[diameter_results_df['method'] != methods[0]]
    sns.boxplot(data=diff_res_df, y='diff µm', x='method')
    plt.ylabel('difference to manual [µm]')
    plt.title('Difference to manual measurement')
    plt.savefig(f'{f_name}/method_diff_µm_box.png')
    plt.savefig(f'{f_name}/method_diff_µm_box.svg')
    plt.close()
    sns.boxplot(data=diff_res_df, y='diff perc', x='method')
    plt.ylabel('difference to manual [%]')
    plt.title('Difference to manual measurement')
    plt.savefig(f'{f_name}/method_diff_perc_box.png')
    plt.savefig(f'{f_name}/method_diff_perc_box.svg')
    plt.close()

    #make plot with different methods and different celltypes
    sns.swarmplot(data=diff_res_df, y='diameter', x='celltype', hue = 'method')
    plt.ylabel('diameter [µm]')
    plt.title('Comparison of diamter values for different measurement methods')
    plt.savefig(f'{f_name}/method_comp_cts.png')
    plt.savefig(f'{f_name}/method_comp_cts.svg')
    plt.close()
    sns.swarmplot(data=diff_res_df, y='diff µm', x='celltype', hue = 'method')
    plt.ylabel('difference to manual [µm]')
    plt.title('Difference to manual measurement')
    plt.savefig(f'{f_name}/method_diff_µm_cts.png')
    plt.savefig(f'{f_name}/method_diff_µm_cts.svg')
    plt.close()
    sns.swarmplot(data=diff_res_df, y='diff perc', x='celltype', hue = 'method')
    plt.ylabel('difference to manual [%]')
    plt.title('Difference to manual measurement')
    plt.savefig(f'{f_name}/method_diff_perc_cts_point.png')
    plt.savefig(f'{f_name}/method_diff_perc_cts_point.svg')
    plt.close()

    log.info('Analysis done')



