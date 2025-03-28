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
    plotwidth = 0.5
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/250327_j0251{version}_manual_org_eval"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'org_eval_log', log_dir=f_name)

    eval_path = 'cajal/scratch/users/arother/bio_analysis_results/for_eval/241210_ves_er_golgi_gt7_rndm_coords_merged/' \
                '250305_random_organelles_RM_for_analysis.csv'
    log.info(f'Load manual evaluation results from {eval_path}')
    eval_df = pd.read_csv(eval_path)
    org_palette = {'er': '#54A9DA', 'golgi': '#DE9B1E', 'mi': '#EE1BE0', 'syn': '#29FC15', 'single vesicle': 'black'}

    #adjust labels to predicted labels
    eval_df['manual organelle'] = eval_df['manual organelle'].str.strip().str.lower()
    eval_df.loc[eval_df['manual organelle'] == 'sv', 'manual organelle'] = 'single vesicle'
    eval_df.loc[eval_df['manual organelle'] == 'unknown', 'manual organelle'] = 'other'
    eval_df.loc[eval_df['manual organelle'] == 'unkown', 'manual organelle'] = 'other'
    eval_df.loc[eval_df['manual organelle'] == 'ncl.', 'manual organelle'] = 'nucleus'
    # allow only compartments: axon, dendrite, soma, glia process
    comps = ['axon', 'dendrite', 'soma', 'glia process']
    eval_df['compartment'] = eval_df['compartment'].str.strip().str.lower()
    eval_df.loc[eval_df['compartment'] == 'axon collateral', 'compartment'] = 'axon'
    eval_df.loc[eval_df['compartment'] == 'glia cell', 'compartment'] = 'glia process'
    assert (len(np.unique(eval_df['compartment'])) == len(comps))
    eval_df.to_csv(f'{f_name}/eval_df.csv')

    #for each organelle, get fraction that was predicted true
    log.info('Get overview over true and false organelles')
    pred_org_labelles = np.unique(eval_df['organelle'])
    org_groups = eval_df.groupby('organelle')
    ov_columns = ['number total', 'number true', 'fraction true']
    overview_df = pd.DataFrame(columns=ov_columns, index=pred_org_labelles)
    overview_df['number total'] = np.array(org_groups.size())
    for po in pred_org_labelles:
        org_group = org_groups.get_group(po)
        true_org_number = len(org_group[org_group['manual organelle'] == po])
        overview_df.loc[po, 'number true'] = true_org_number
        overview_df.loc[po, 'fraction true'] = 100 * true_org_number / overview_df.loc[po, 'number total']

    sns.barplot(data = overview_df, y = 'fraction true', x = overview_df.index, palette= org_palette, width=plotwidth)
    plt.xlabel('organelle', fontsize=fontsize)
    plt.ylabel('% true', fontsize=fontsize)
    plt.title(f'overview of organelles')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/pred_overview_perc.png')
    plt.savefig(f'{f_name}/pred_overview_perc.svg')
    plt.close()

    # get dataframe only with correct predictions
    eval_df_true = eval_df[eval_df['organelle'] == eval_df['manual organelle']]
    log.info(f'{100 * len(eval_df_true)/ len(eval_df):.2f} % of all coordinates are predicted correct.')
    eval_df_false = eval_df[eval_df['organelle'] != eval_df['manual organelle']]
    log.info(f'{100 * len(eval_df_false) / len(eval_df):.2f} % of all coordinates are predicted wrong.')

    log.info('Plot categories of false labels')
    false_groups = eval_df_false.groupby('organelle')
    false_groups_size = false_groups.size()
    for false_cat in np.unique(eval_df_false['manual organelle']):
        false_cat_df = eval_df_false[eval_df_false['manual organelle'] == false_cat]
        unique_orgs_cat = np.unique(false_cat_df['organelle'])
        false_cat_groups = false_cat_df.groupby('organelle')
        false_cat_groups_size = false_cat_groups.size()
        overview_df.loc[false_cat_groups_size.index, f'false {false_cat} number'] = false_cat_groups_size.values
        overview_df.loc[false_cat_groups_size.index, f'false {false_cat} fraction'] = false_cat_groups_size.values / np.array(false_groups_size[false_cat_groups_size.index])

    overview_df.to_csv(f'{f_name}/org_overview.csv')

    sns.histplot(data=eval_df_false, x='manual organelle', stat='percent', hue='organelle',
                 palette=org_palette, common_norm=False, multiple='dodge', shrink=0.8,
                 ec=None, alpha=1, hue_order=pred_org_labelles, binwidth= plotwidth)
    plt.xlabel('reason false label', fontsize=fontsize)
    plt.ylabel('percent of coords', fontsize=fontsize)
    plt.title('overview false labels')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/false_categories_perc.png')
    plt.savefig(f'{f_name}/false_categories_perc.svg')
    plt.close()
    eval_df_false.to_csv(f'{f_name}/eval_false_df.csv')

    #get compartments with fraction of true
    log.info('Get true fraction for each compartment')
    comp_groups = eval_df.groupby('compartment')
    comp_numbers_total = comp_groups.size()
    log.info(f'The dataframe has the following numbers of each compartment: {comp_numbers_total} and the following fractions: {comp_numbers_total/ comp_numbers_total.sum()}')
    num_orgs = len(pred_org_labelles)
    num_comps = len(comps)
    comp_overview_df = pd.DataFrame(columns = ['organelle', 'compartment', 'number total', 'fraction comp', 'number true', 'fraction true'], index = range(num_comps * num_orgs))
    true_org_groups = eval_df_true.groupby('organelle')
    for ip, po in enumerate(pred_org_labelles):
        org_group = org_groups.get_group(po)
        comp_overview_df.loc[ip * num_comps: (ip + 1) * num_comps - 1, 'organelle'] = po
        comp_groups = org_group.groupby('compartment')
        comp_group_sizes = comp_groups.size()
        len_cg = len(comp_group_sizes.index)
        comp_overview_df.loc[ip * num_comps: ip * num_comps + len_cg - 1, 'compartment'] = comp_group_sizes.index
        comp_overview_df.loc[ip * num_comps: ip * num_comps + len_cg - 1, 'number total'] = comp_group_sizes.values
        comp_overview_df.loc[ip * num_comps: ip * num_comps + len_cg - 1, 'fraction comp'] = 100 * comp_group_sizes.values / comp_group_sizes.sum()

        org_group_true = true_org_groups.get_group(po)
        comp_groups_true = org_group_true.groupby('compartment')
        comp_group_sizes_true = comp_groups_true.size()
        #make sure that all compartments actually have true values; need to write code differently if this is ever not the case
        assert(len(comp_group_sizes_true.index) == len_cg)
        comp_overview_df.loc[ip * num_comps: ip * num_comps + len_cg - 1, 'number true'] = comp_group_sizes_true.values
        comp_overview_df.loc[ip * num_comps: ip * num_comps + len_cg - 1, 'fraction true'] = 100 * comp_group_sizes_true.values/ comp_group_sizes.values

    comp_overview_df.to_csv(f'{f_name}/comp_overview_df.csv')
    sns.barplot(data=comp_overview_df, y='fraction true', x='compartment', hue = 'organelle', palette=org_palette, width=plotwidth)
    plt.xlabel('compartment', fontsize=fontsize)
    plt.ylabel('% true', fontsize=fontsize)
    plt.title(f'overview of organelles')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/pred_comp_perc.png')
    plt.savefig(f'{f_name}/pred_comp_perc.svg')
    plt.close()
    sns.barplot(data=comp_overview_df, y='number total', x='compartment', hue='organelle', palette=org_palette,
                width=plotwidth)
    plt.xlabel('compartment', fontsize=fontsize)
    plt.ylabel('number of coordinates', fontsize=fontsize)
    plt.title(f'overview of organelles')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/pred_comp_total_nums.png')
    plt.savefig(f'{f_name}/pred_comp_total_nums.svg')
    plt.close()

    sns.barplot(data=comp_overview_df, y='fraction comp', x='compartment', hue='organelle', palette=org_palette,
                width=plotwidth)
    plt.xlabel('compartment', fontsize=fontsize)
    plt.ylabel('% of coordinates', fontsize=fontsize)
    plt.title(f'overview of organelles')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/pred_comp_total_perc.png')
    plt.savefig(f'{f_name}/pred_comp_total_perc.svg')
    plt.close()

    # get golgi false label and true fraction depending on golgi stack oder vesicle
    log.info('Get fraction true for golgi stacks vs golgi vesicles')
    golgi_group = org_groups.get_group('golgi')
    golgi_true_group = true_org_groups.get_group('golgi')
    golgi_overview_df = pd.DataFrame(columns = ['golgi compartment', 'number total', 'fraction total', 'number true', 'fraction true'], index = range(2))
    golgi_overview_df['golgi compartment'] = ['stack', 'vesicle']
    gg_comp_groups = golgi_group.groupby('specification')
    gg_comp_groups_true = golgi_true_group.groupby('specification')
    gg_comp_groups_size = gg_comp_groups.size()
    gg_comp_groups_true_size = gg_comp_groups_true.size()
    raise ValueError
    #currently getting number of predicted one which are stack and sv
    #maybe also get number of ones that were manually found at which fraction of them predicted were stack and sv?

    #also get false labels with golgi divided into stacks and sv


    log.info('Get fraction true for each cell type in each organelle')


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