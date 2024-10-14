#evaluate vesicle data
#plot results

if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    global_params.wd = analysis_params.working_dir()
    with_glia = False
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    fontsize = 20
    organelle = 'er'
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/241014_j0251{version}_manual_{organelle}_eval"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'{organelle}_eval_log', log_dir=f_name)

    eval_path = 'cajal/scratch/users/arother/bio_analysis_results/for_eval/241014_j0251v6_ct_random_er_eval_n15_v7gt/' \
                'random_er_evaluation_RM_annot.csv'
    log.info(f'Load manual evaluation results from {eval_path}')
    eval_df = pd.read_csv(eval_path)
    ct_palette = {'er': '#54A9DA', 'golgi':'#DE9B1E', 'mi': '#EE1BE0', 'syn':'#29FC15'}
    # put unsure/yes and unsure/no into yes/no groups, only single class had unsure label
    eval_df.loc[eval_df[f'{organelle}?'] == 'y', f'{organelle}?'] = 'True'
    eval_df.loc[eval_df[f'{organelle}?'] == 'n', f'{organelle}?'] = 'False'
    eval_df.to_csv(f'{f_name}/eval_df.csv')
    #get overview df with numbers of different classes
    log.info('Get overview over true and false organelles')
    ov_columns = ['number total', 'fraction true', 'fraction false']
    overview_df = pd.DataFrame(columns=ov_columns)
    overview_df['number total'] = len(eval_df)
    eval_true_df = eval_df[eval_df[f'{organelle}?'] == 'True']
    eval_true_df = eval_true_df.reset_index(drop=True)
    eval_false_df = eval_df[eval_df[f'{organelle}?'] == 'False']
    eval_false_df = eval_false_df.reset_index(drop = True)
    overview_df['fraction true'] = len(eval_true_df) / overview_df['number total']
    overview_df['fraction false'] = len(eval_false_df) / overview_df['number total']
    #one plot with true and false vesicles as overview
    sns.barplot(data = eval_df, x = f'{organelle}?', color=ct_palette[organelle])
    plt.xlabel(f'{organelle}?', fontsize = fontsize)
    plt.ylabel('percent of coordinates', fontsize = fontsize)
    plt.title(f'overview {organelle}')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/pred_overview_{organelle}_perc.png')
    plt.savefig(f'{f_name}/pred_overview_{organelle}_perc.svg')
    plt.close()
    log.info('Plot categories of false labels')
    eval_false_df = eval_false_df.astype({'other structure': str})
    for false_cat in np.unique(eval_false_df['other structure']):
        false_cat_df = eval_false_df[eval_false_df['other structure'] == false_cat]
        overview_df.loc[f'false {false_cat} number'] = len(false_cat_df)
        overview_df.loc[f'false {false_cat} fraction'] = len(false_cat_df) / len(eval_false_df)

    overview_df.to_csv(f'{f_name}/overview_df.csv')
    #plot categories of false labels
    sns.barplot(data=eval_false_df, x='other structure', stat='percent', color=ct_palette[organelle])
    plt.xlabel('reason false label', fontsize=fontsize)
    plt.ylabel('percent of coords', fontsize=fontsize)
    plt.title('overview false labels')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/false_categories_{organelle}_perc.png')
    plt.savefig(f'{f_name}/false_categories_{organelle}_perc.svg')
    plt.close()
    eval_false_df.to_csv(f'{f_name}/eval_false_df.csv')
    log.info('get more information about true synapses')
    #plot true percentage depending on cell type
    unique_cts = np.unique(eval_df['celltype'])
    num_cts = len(unique_cts)
    ct_overview_df = pd.DataFrame(columns = ['celltype', f'number of {organelle} total', f'fraction true {organelle}'], index = range(num_cts))
    ct_overview_df.loc[0 : num_cts - 1, 'celltype'] = unique_cts
    all_ct_groups = eval_df.groupby('celltype')
    ct_overview_df.loc[0 : num_cts - 1, f'number of {organelle} total'] = np.array(all_ct_groups.size())
    true_groups = eval_true_df.groupby('celltype')
    ct_overview_df.loc[0: num_cts - 1, f'fraction true {organelle}'] = 100  * np.array(true_groups.size()) / np.array(all_ct_groups.size())
    ct_overview_df.to_csv(f'{f_name}/ct_overview_df.csv')
    sns.barplot(data = ct_overview_df, x = 'celltype', y = f'fraction true {organelle}', color=ct_palette[organelle])
    plt.xlabel('celltype', fontsize=fontsize)
    plt.ylabel('percent of vesicles', fontsize=fontsize)
    plt.title('fraction true vesicles all ct')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/true_perc_celltype_{organelle}.png')
    plt.savefig(f'{f_name}/true_perc_celltype_{organelle}.svg')
    plt.close()

    log.info('Analysis done.')