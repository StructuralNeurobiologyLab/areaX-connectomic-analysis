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
    organelle1 = 'er'
    organelle2 = 'golgi'
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/241107_j0251{version}_manual_{organelle1}_{organelle2}_eval"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'{organelle1}_{organelle2}_eval_log', log_dir=f_name)

    eval_path = 'cajal/scratch/users/arother/bio_analysis_results/for_eval/241014_j0251v6_ct_random_er_eval_n15_v7gt/' \
                'random_er_evaluation_RM_annot.csv'
    eval_path2 = 'cajal/scratch/users/arother/bio_analysis_results/for_eval/241014_j0251v6_ct_random_golgi_eval_n15_v7gt/' \
                '241107_Random_golgi_evaluation.csv'
    log.info(f'Load manual evaluation results from {eval_path}')
    eval_df = pd.read_csv(eval_path)
    eval_df['organelle'] = organelle1
    eval_df['true structure'] = eval_df[f'{organelle1}?']
    eval_df2 = pd.read_csv(eval_path2)
    eval_df2['organelle'] = organelle2
    eval_df2['true structure'] = eval_df2[f'{organelle2}?']
    eval_df = pd.concat([eval_df, eval_df2])

    ct_palette = {'er': '#54A9DA', 'golgi':'#DE9B1E', 'mi': '#EE1BE0', 'syn':'#29FC15'}
    # put unsure/yes and unsure/no into yes/no groups, only single class had unsure label
    eval_df.loc[eval_df['true structure'] == 'y', 'true structure'] = 'True'
    eval_df.loc[eval_df['true structure'] == 'n', 'true structure'] = 'False'
    eval_df.to_csv(f'{f_name}/eval_df.csv')
    #get overview df with numbers of different classes
    log.info('Get overview over true and false organelles')
    ov_columns = ['number total', 'fraction true', 'fraction false']
    unique_organelles = np.unique(eval_df['organelle'])
    org_groups = eval_df.groupby('organelle')
    overview_df = pd.DataFrame(columns=ov_columns, index = unique_organelles)
    overview_df['number total'] = np.array(org_groups.size())
    eval_true_df = eval_df[eval_df['true structure'] == 'True']
    eval_true_df = eval_true_df.reset_index(drop=True)
    true_groups = eval_true_df.groupby('organelle')
    eval_false_df = eval_df[eval_df['true structure'] == 'False']
    eval_false_df = eval_false_df.reset_index(drop = True)
    false_groups = eval_false_df.groupby('organelle')
    overview_df['fraction true'] = np.array(true_groups.size()) / overview_df['number total']
    overview_df['fraction false'] = np.array(false_groups.size()) / overview_df['number total']
    #one plot with true and false vesicles as overview
    sns.histplot(data=eval_df, x=f'true structure', stat='percent', hue='organelle',
                 palette=ct_palette, common_norm=False, multiple='dodge', shrink=0.8,
                 ec=None, alpha=1)
    plt.xlabel('true structure', fontsize = fontsize)
    plt.ylabel('percent of coordinates', fontsize = fontsize)
    plt.title(f'overview {organelle1}, {organelle2}')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/pred_overview_{organelle1}_{organelle2}_perc.png')
    plt.savefig(f'{f_name}/pred_overview_{organelle1}_{organelle2}_perc.svg')
    plt.close()
    log.info('Plot categories of false labels')
    eval_false_df = eval_false_df.astype({'other structure': str})
    for false_cat in np.unique(eval_false_df['other structure']):
        false_cat_df = eval_false_df[eval_false_df['other structure'] == false_cat]
        unique_orgs_cat = np.unique(false_cat_df['organelle'])
        if len(unique_orgs_cat) > 1:
            false_cat_groups = false_cat_df.groupby('organelle')
            overview_df[f'false {false_cat} number'] = np.array(false_cat_groups.size())
            overview_df[f'false {false_cat} fraction'] = np.array(false_cat_groups.size()) / np.array(false_groups.size())
        else:
            overview_df.loc[unique_orgs_cat[0], f'false {false_cat} number'] = len(false_cat_df)
            overview_df.loc[unique_orgs_cat[0], f'false {false_cat} fraction'] = len(false_cat_df) / len(false_groups.get_group(unique_orgs_cat[0]))

    overview_df.to_csv(f'{f_name}/overview_df.csv')
    #plot categories of false labels
    sns.histplot(data=eval_false_df, x='other structure', stat='percent', hue = 'organelle',
                 palette=ct_palette, common_norm=False, multiple='dodge', shrink=0.8,
                 ec=None, alpha=1)
    plt.xlabel('reason false label', fontsize=fontsize)
    plt.ylabel('percent of coords', fontsize=fontsize)
    plt.title('overview false labels')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/false_categories_{organelle1}_{organelle2}_perc.png')
    plt.savefig(f'{f_name}/false_categories_{organelle1}_{organelle2}_perc.svg')
    plt.close()
    eval_false_df.to_csv(f'{f_name}/eval_false_df.csv')
    log.info('get more information about true synapses')
    #plot true percentage depending on cell type
    org1_eval_df = eval_df[eval_df['organelle'] == organelle1]
    org2_eval_df = eval_df[eval_df['organelle'] == organelle2]
    unique_cts1 = np.unique(org1_eval_df['celltype'])
    num_cts1 = len(unique_cts1)
    unique_cts2 = np.unique(org2_eval_df['celltype'])
    num_cts2 = len(unique_cts2)
    total_num_cts = num_cts1 + num_cts2

    ct_overview_df = pd.DataFrame(columns = ['celltype', f'number of organelle total', f'fraction true organelle', 'organelle'], index = range(total_num_cts))
    ct_overview_df.loc[0 : num_cts1 - 1, 'celltype'] = unique_cts1
    ct_overview_df.loc[0: num_cts1 -1, 'organelle'] = organelle1
    org1_all_ct_groups = org1_eval_df.groupby('celltype')

    ct_overview_df.loc[0 : num_cts1 - 1, f'number of organelle total'] = np.array(org1_all_ct_groups.size())
    true_org1_df = org1_eval_df[org1_eval_df['true structure'] == 'True']
    org1_true_groups = true_org1_df.groupby('celltype')
    ct_overview_df.loc[0: num_cts1 - 1, f'fraction true organelle'] = 100  * np.array(org1_true_groups.size()) / np.array(org1_all_ct_groups.size())
    ct_overview_df.loc[num_cts1: total_num_cts - 1, 'celltype'] = unique_cts2
    ct_overview_df.loc[num_cts1: total_num_cts - 1, 'organelle'] = organelle2
    org2_all_ct_groups = org2_eval_df.groupby('celltype')
    ct_overview_df.loc[num_cts1: total_num_cts - 1, f'number of organelle total'] = np.array(org2_all_ct_groups.size())
    true_org2_df = org2_eval_df[org2_eval_df['true structure'] == 'True']
    org2_true_groups = true_org2_df.groupby('celltype')
    ct_overview_df.loc[num_cts1: total_num_cts - 1, f'fraction true organelle'] = 100 * np.array(org2_true_groups.size()) / np.array(
        org2_all_ct_groups.size())
    ct_overview_df.to_csv(f'{f_name}/ct_overview_df.csv')
    sns.barplot(data = ct_overview_df, x = 'celltype', y = f'fraction true organelle', hue = 'organelle', palette=ct_palette)
    plt.xlabel('celltype', fontsize=fontsize)
    plt.ylabel('percent of coords', fontsize=fontsize)
    plt.title(f'fraction true {organelle1}, {organelle2} all ct')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/true_perc_celltype_{organelle1}_{organelle2}.png')
    plt.savefig(f'{f_name}/true_perc_celltype_{organelle1}_{organelle2}.svg')
    plt.close()

    #also plot if true golgi in soma
    golgi_eval_df = eval_true_df[eval_true_df['organelle'] == organelle2]
    golgi_eval_df.loc[golgi_eval_df['in soma?'] == 'y', 'in soma?'] = 'True'
    golgi_eval_df.loc[golgi_eval_df['in soma?'] == 'n', 'in soma?'] = 'False'
    num_true_golgi_soma = len(golgi_eval_df[golgi_eval_df['in soma?'] == 'True'])
    overview_df.loc['golgi', 'number true in soma'] = num_true_golgi_soma
    overview_df.loc['golgi', 'fraction true in soma'] = num_true_golgi_soma / len(golgi_eval_df)
    overview_df.to_csv(f'{f_name}/overview_df.csv')
    sns.histplot(data=golgi_eval_df, x='in soma?', stat='percent',
                 color=ct_palette[organelle2], common_norm=False, multiple='dodge', shrink=0.8,
                 ec=None, alpha=1)
    plt.xlabel('in soma', fontsize=fontsize)
    plt.ylabel('percent of coords', fontsize=fontsize)
    plt.title('true golgi in soma')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/true_soma_{organelle2}_perc.png')
    plt.savefig(f'{f_name}/true_soma_{organelle1}_{organelle2}_perc.svg')
    plt.close()

    log.info('Analysis done.')