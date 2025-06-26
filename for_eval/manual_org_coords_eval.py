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
    golgi_true_group = true_org_groups.get_group('golgi')
    golgi_overview_df = pd.DataFrame(columns = ['golgi compartment', 'number total manual', 'fraction total manual', 'number true', 'fraction true'], index = range(2))
    golgi_overview_df['golgi compartment'] = ['stack', 'vesicle']
    gg_comp_groups_true = golgi_true_group.groupby('specification')
    gg_comp_groups_true_size = gg_comp_groups_true.size()
    #get number of ones that were manually found at which fraction of them predicted were stack and sv?
    manual_org_groups = eval_df.groupby('manual organelle')
    man_golgi_group = manual_org_groups.get_group('golgi')
    man_golgi_group_comps = man_golgi_group.groupby('specification')
    gg_man_comps_size = man_golgi_group_comps.size()
    golgi_overview_df['number total manual'] = gg_man_comps_size.values
    golgi_overview_df['fraction total manual'] = 100 * gg_man_comps_size.values / gg_man_comps_size.sum()
    golgi_overview_df['number true'] = gg_comp_groups_true_size.values
    golgi_overview_df['fraction true'] = 100 * gg_comp_groups_true_size.values / gg_man_comps_size.values
    golgi_overview_df.to_csv(f'{f_name}/golgi_comps_ov.csv')
    #plot
    sns.barplot(data=golgi_overview_df, y='fraction total manual', x='golgi compartment', color = org_palette['golgi'],
                width=plotwidth)
    plt.xlabel('golgi compartment', fontsize=fontsize)
    plt.ylabel('% of coordinates', fontsize=fontsize)
    plt.title(f'manually identified golgi compartments')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/golgi_comps_manual.png')
    plt.savefig(f'{f_name}/golgi_comps_manual.svg')
    plt.close()
    sns.barplot(data=golgi_overview_df, y='fraction true', x='golgi compartment', color=org_palette['golgi'],
                width=plotwidth)
    plt.xlabel('golgi compartment', fontsize=fontsize)
    plt.ylabel('% of coordinates', fontsize=fontsize)
    plt.title(f'manually identified golgi compartments that were predicted as golgi')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/golgi_comps_manual_true.png')
    plt.savefig(f'{f_name}/golgi_comps_manual_true.svg')
    plt.close()
    #also get false labels with golgi divided into stacks and sv
    false_manual = eval_df[eval_df['manual organelle'] != eval_df['organelle']]
    false_group_manual = false_manual.groupby('manual organelle')
    golgi_false_manual = false_group_manual.get_group('golgi')
    golgi_comp_palette = {'stack': org_palette['golgi'], 'sv': 'gray'}
    golgi_false_manual.to_csv(f'{f_name}/golgi_manual_comps_false_cats.csv')

    sns.histplot(data=golgi_false_manual, x='organelle', stat='percent', hue='specification',
                 palette=golgi_comp_palette, common_norm=False, multiple='dodge', shrink=0.8,
                 ec=None, alpha=1, binwidth=plotwidth)
    plt.xlabel('prediction', fontsize=fontsize)
    plt.ylabel('% of coords', fontsize=fontsize)
    plt.title('prediction of manually identified golgi comps')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/false_cgolgi_comps_manual_perc.png')
    plt.savefig(f'{f_name}/false_cgolgi_comps_manual_perc.svg')
    plt.close()
    sns.histplot(data=golgi_false_manual, x='organelle', hue='specification',
                 palette=golgi_comp_palette, common_norm=False, multiple='dodge', shrink=0.8,
                 ec=None, alpha=1, binwidth=plotwidth)
    plt.xlabel('prediction', fontsize=fontsize)
    plt.ylabel('% of coords', fontsize=fontsize)
    plt.title('prediction of manually identified golgi comps')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/false_cgolgi_comps_manual.png')
    plt.savefig(f'{f_name}/false_cgolgi_comps_manual.svg')
    plt.close()

    log.info('Get fraction true for each cell type in each organelle')
    unique_cts = np.unique(eval_df['celltype'])
    total_num_cts = len(unique_cts)
    ct_overview_df = pd.DataFrame(
        columns=['celltype', 'number total', 'number true', 'fraction true', 'organelle'],
        index=range(total_num_cts* len(pred_org_labelles)))
    for ip, po in enumerate(pred_org_labelles):
        org_group = org_groups.get_group(po)
        org_group_true = true_org_groups.get_group(po)
        ct_overview_df.loc[ip*total_num_cts: (ip + 1)*total_num_cts - 1, 'organelle'] = po
        org_group_ctg = org_group.groupby('celltype')
        ct_sizes = org_group_ctg.size()
        org_group_true_ctg = org_group_true.groupby('celltype')
        true_ct_sizes = org_group_true_ctg.size()
        org_cts = np.unique(org_group['celltype'])
        org_true_cts = np.unique(org_group_true['celltype'])
        len_org_cts = len(org_cts)
        assert(len_org_cts == len(org_true_cts))
        #change code if one celltype doesn't have nay true organelles of one group
        ct_overview_df.loc[ip * total_num_cts: ip * total_num_cts + len_org_cts - 1, 'celltype'] = org_cts
        ct_overview_df.loc[ip * total_num_cts: ip * total_num_cts + len_org_cts - 1, 'number total'] = ct_sizes.values
        ct_overview_df.loc[ip * total_num_cts: ip * total_num_cts + len_org_cts - 1, 'number true'] = true_ct_sizes.values
        ct_overview_df.loc[ip * total_num_cts: ip * total_num_cts + len_org_cts - 1, 'fraction true'] = 100 * true_ct_sizes.values / ct_sizes.values

    ct_overview_df.to_csv(f'{f_name}/ct_overview_df')
    sns.barplot(data=ct_overview_df, y='number true', hue='organelle', x = 'celltype', palette=org_palette,
                width=plotwidth)
    plt.xlabel('cell type', fontsize=fontsize)
    plt.ylabel('number of coordinates', fontsize=fontsize)
    plt.title(f'true organelles per cell type')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/ct_true_org_num.png')
    plt.savefig(f'{f_name}/ct_true_org_num.svg')
    plt.close()
    sns.barplot(data=ct_overview_df, y='fraction true', x ='celltype', hue = 'organelle', palette=org_palette,
                width=plotwidth)
    plt.xlabel('cell type', fontsize=fontsize)
    plt.ylabel('% of coordinates', fontsize=fontsize)
    plt.title(f'true organelles per cell type')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/ct_true_org_perc.png')
    plt.savefig(f'{f_name}/ct_true_org_perc.svg')
    plt.close()

    log.info('Analysis done.')
