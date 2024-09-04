#evaluate vesicle data
#plot results

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct, get_cell_close_surface_area
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    from cajal.nvmescratch.users.arother.bio_analysis.general.vesicle_helper import get_non_synaptic_vesicle_coords
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from syconn.mp.mp_utils import start_multiprocess_imap
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import kruskal, ranksums
    from itertools import combinations

    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    global_params.wd = analysis_params.working_dir()
    with_glia = False
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    fontsize = 20
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/240723_j0251{version}_manual_ves_eval"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'single_ves_eval_log', log_dir=f_name)

    eval_path = 'cajal/scratch/arother/bio_analysis_results/for_eval/240723_j0251v6_manual_ves_eval/' \
                '240904_random_single_vesicles_evaluation_RM_with_dist_results.csv'
    log.info(f'Load manual evaluation results from {eval_path}')
    eval_df = pd.read_csv(eval_path)
    ct_palette = {'single class': '#E8AA47', 'multi class': '#3287A8'}
    # put unsure/yes and unsure/no into yes/no groups, only single class had unsure label
    eval_df.loc[eval_df['single vesicle?'] == 'u/y', 'single vesicle?'] = 'y'
    eval_df.loc[eval_df['single vesicle?'] == 'u/n', 'single vesicle?'] = 'n'
    eval_df.loc[eval_df['single vesicle?'] == 'y', 'single vesicle?'] = 'True'
    eval_df.loc[eval_df['single vesicle?'] == 'n', 'single vesicle?'] = 'False'
    eval_df.to_csv(f'{f_name}/eval_df.csv')
    pred_class_groups = eval_df.groupby('prediction')
    #get overview df with numbers of different classes
    log.info('Get overview over true and false vesicles')
    ov_columns = ['number total', 'fraction true', 'fraction false']
    overview_df = pd.DataFrame(columns=ov_columns, index=np.unique(eval_df['prediction']))
    overview_df['number total'] = np.array(pred_class_groups.size())
    eval_true_df = eval_df[eval_df['single vesicle?'] == 'True']
    eval_true_df = eval_true_df.reset_index(drop=True)
    eval_false_df = eval_df[eval_df['single vesicle?'] == 'False']
    eval_false_df = eval_false_df.reset_index(drop = True)
    true_groups = eval_true_df.groupby('prediction')
    false_groups = eval_false_df.groupby('prediction')
    overview_df['fraction true'] = np.array(true_groups.size()) / overview_df['number total']
    overview_df['fraction false'] = np.array(false_groups.size()) / overview_df['number total']
    #one plot with true and false vesicles as overview
    sns.histplot(data = eval_df, x = 'single vesicle?', stat='percent', hue = 'prediction',
                 palette=ct_palette, common_norm= False, multiple= 'dodge', shrink=0.8,
                 ec = None, alpha = 1, )
    plt.xlabel('single vesicle?', fontsize = fontsize)
    plt.ylabel('percent of vesicles', fontsize = fontsize)
    plt.title('overview single vesicles')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/pred_overview_perc.png')
    plt.savefig(f'{f_name}/pred_overview_perc.svg')
    plt.close()
    log.info('Plot categories of false labels')
    #if no other structure detected, label as no clear vesicle
    eval_false_df.loc[:,'other structure'] = eval_false_df['other structure'].fillna('no clear vesicle')
    for false_cat in np.unique(eval_false_df['other structure']):
        false_cat_df = eval_false_df[eval_false_df['other structure'] == false_cat]
        if len(np.unique(false_cat_df['prediction'])) == 1:
            group = np.unique(false_cat_df['prediction'])[0]
            overview_df.loc[group, f'false {false_cat} number'] = len(false_cat_df)
            overview_df.loc[group, f'false {false_cat} fraction'] = \
                len(false_cat_df) / false_groups.get_group(group).size
        else:
            cat_groups = false_cat_df.groupby('prediction')
            overview_df[ f'false {false_cat} number'] = np.array(cat_groups.size())
            overview_df[f'false {false_cat} fraction'] = np.array(cat_groups.size()) / (false_groups.size())
    #plot categories of false labels
    sns.histplot(data=eval_false_df, x='other structure', stat='percent', hue='prediction',
                 palette=ct_palette, common_norm=False, multiple='dodge', shrink=0.8,
                 ec=None, alpha=1)
    plt.xlabel('reason false label', fontsize=fontsize)
    plt.ylabel('percent of vesicles', fontsize=fontsize)
    plt.title('overview false vesicles')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/false_categories_perc.png')
    plt.savefig(f'{f_name}/false_categories_perc.svg')
    plt.close()
    eval_false_df.to_csv(f'{f_name}/eval_false_df.csv')
    log.info('get more information about true synapses')
    #plot percentage in axon
    eval_true_df.loc[eval_true_df['in axon?'] == 'y', 'in axon?'] = 'True'
    eval_true_df.loc[eval_true_df['in axon?'] == 'n', 'in axon?'] = 'False'
    sns.histplot(data=eval_true_df, x='in axon?', stat='percent', hue='prediction',
                 palette=ct_palette, common_norm=False, multiple='dodge', shrink=0.8,
                 ec=None, alpha=1)
    plt.xlabel('in axon?', fontsize=fontsize)
    plt.ylabel('percent of vesicles', fontsize=fontsize)
    plt.title('single vesicles in axon')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/true_in_axon_perc.png')
    plt.savefig(f'{f_name}/true_in_axon_perc.svg')
    plt.close()
    #plot true percentage depending on cell type
    unique_cts = np.unique(eval_df['celltype'])
    num_cts = len(unique_cts)
    ct_overview_df = pd.DataFrame(columns = ['celltype', 'number of vesicles total', 'fraction true vesicles', 'prediction'], index = range(num_cts * 2))
    ct_overview_df.loc[0 : num_cts - 1, 'celltype'] = unique_cts
    ct_overview_df.loc[num_cts: num_cts * 2 - 1, 'celltype'] = unique_cts
    ct_overview_df.loc[0: num_cts - 1, 'prediction'] = 'multi class'
    ct_overview_df.loc[num_cts: num_cts * 2 - 1, 'prediction'] = 'single class'
    all_ct_groups_mc = eval_df[eval_df['prediction'] == 'multi class'].groupby('celltype')
    all_ct_groups_sc = eval_df[eval_df['prediction'] == 'single class'].groupby('celltype')
    ct_overview_df.loc[0 : num_cts - 1, 'number of vesicles total'] = np.array(all_ct_groups_mc.size())
    ct_overview_df.loc[num_cts: num_cts * 2 - 1, 'number of vesicles total'] = np.array(all_ct_groups_sc.size())
    true_groups_mc = eval_true_df[eval_true_df['prediction'] == 'multi class'].groupby('celltype')
    true_groups_sc = eval_true_df[eval_true_df['prediction'] == 'single class'].groupby('celltype')
    ct_overview_df.loc[0: num_cts - 1, 'fraction true vesicles'] = 100  * np.array(true_groups_mc.size()) / np.array(all_ct_groups_mc.size())
    ct_overview_df.loc[num_cts: num_cts * 2 - 1, 'fraction true vesicles'] = 100 * np.array(true_groups_sc.size()) / np.array(all_ct_groups_sc.size())
    ct_overview_df.to_csv(f'{f_name}/ct_overview_df.csv')
    sns.barplot(data = ct_overview_df, x = 'celltype', y = 'fraction true vesicles', hue = 'prediction', palette=ct_palette)
    plt.xlabel('celltype', fontsize=fontsize)
    plt.ylabel('percent of vesicles', fontsize=fontsize)
    plt.title('fraction true vesicles all ct')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/true_perc_celltype.png')
    plt.savefig(f'{f_name}/true_perc_celltype.svg')
    plt.close()
    #plot percentage of vesicles close to axon membrane depending on distance measurement
    eval_true_df.loc[eval_true_df['close to membrane?'] == 'y', 'close to membrane?'] = 'True'
    eval_true_df.loc[eval_true_df['close to membrane?'] == 'n', 'close to membrane?'] = 'False'
    eval_true_df['close to cell membrane?'] = np.array(eval_true_df['close to membrane?'])
    eval_true_df.loc[eval_true_df['close to membrane?'] == 'True', 'close to cell membrane?'] = 'True'
    eval_true_df.loc[eval_true_df['close to membrane?'] == 'False', 'close to cell membrane?'] = 'False'
    #rewrite table so that every entry that is close to membrane but not to axon, synapse or cell membrane
    #is set to False; also all entries with axon/mito are set to axon
    for i, label in enumerate(eval_true_df['kind of membrane?']):
        if type(label) != str:
            continue
        if 'axon' in label or 'synapse' in label:
            continue
        else:
            eval_true_df.loc[i, 'close to cell membrane?'] = 'False'
    #get fraction of true close-membrane cells depending on distance
    distance_thresholds = [5, 10, 15, np.inf]
    dist_columns = ['number vesicles', 'fraction close membrane vesicles', 'distance', 'prediction']
    distance_df = pd.DataFrame(columns = dist_columns,index = range(len(distance_thresholds) * 2))
    for di,dist in enumerate(distance_thresholds):
        if dist == np.inf:
            distance_df.loc[di*2: di*2 + 1, 'distance'] = '>15'
        else:
            distance_df.loc[di * 2: di * 2 + 1, 'distance'] = dist
        distance_df.loc[di*2: di*2 + 1, 'prediction'] = np.unique(eval_true_df['prediction'])
        #get number and fraction of true groups for each distance bin
        dist_df = eval_true_df[eval_true_df['dist 2 membrane'] <= dist]
        if di > 0:
            dist_df = dist_df[dist_df['dist 2 membrane'] > distance_thresholds[di - 1]]
        dist_df_groups = dist_df.groupby('prediction')
        distance_df.loc[di*2: di*2 + 1, 'number vesicles'] = np.array(dist_df_groups.size())
        close_mem_dist_df = dist_df[dist_df['close to cell membrane?'] == 'True']
        close_dist_df_groups = close_mem_dist_df.groupby('prediction')
        fraction_close_vesicles = np.array(close_dist_df_groups.size()) / np.array(dist_df_groups.size())
        distance_df.loc[di*2: di*2 + 1, 'fraction close membrane vesicles'] = fraction_close_vesicles
    distance_df.to_csv(f'{f_name}/distance_close_membrane.csv')
    sns.barplot(data=distance_df, x='distance', y= 'number vesicles', hue='prediction',
                 palette=ct_palette)
    plt.xlabel('distance [nm]', fontsize=fontsize)
    plt.ylabel('number vesicles', fontsize=fontsize)
    plt.title('close membrane vesicles')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/num_ves_dist_close_membrane.png')
    plt.savefig(f'{f_name}/num_ves_dist_close_membrane.svg')
    plt.close()
    sns.barplot(data=distance_df, x='distance', y='fraction close membrane vesicles', hue='prediction',
                palette=ct_palette)
    plt.xlabel('distance [nm]', fontsize=fontsize)
    plt.ylabel('fraction close membrane vesicles', fontsize=fontsize)
    plt.title('close membrane vesicles')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/frac_ves_dist_close_membrane.png')
    plt.savefig(f'{f_name}/frac_ves_dist_close_membrane.svg')
    plt.close()

    #get fraction of cells at synapse depending on distance
    eval_true_df.loc[eval_true_df['at synapse?'] == 'y', 'at synapse?'] = 'True'
    eval_true_df.loc[eval_true_df['at synapse?'] == 'n', 'at synapse?'] = 'False'
    syn_distance_thresholds = np.arange(100, 1100, 100)
    syn_distance_thresholds = np.hstack([syn_distance_thresholds, np.inf])
    syn_dist_columns = ['number vesicles', 'fraction close synapse vesicles', 'distance', 'prediction']
    syn_distance_df = pd.DataFrame(columns=dist_columns, index=range(len(syn_distance_thresholds) * 2))
    for di, dist in enumerate(syn_distance_thresholds):
        if dist == np.inf:
            syn_distance_df.loc[di * 2: di * 2 + 1, 'distance'] = '>1000'
        else:
            syn_distance_df.loc[di * 2: di * 2 + 1, 'distance'] = dist
        syn_distance_df.loc[di * 2: di * 2 + 1, 'prediction'] = np.unique(eval_true_df['prediction'])
        # get number and fraction of true groups for each distance bin
        dist_df = eval_true_df[eval_true_df['dist 2 synapse'] <= dist]
        if di > 0:
            dist_df = dist_df[dist_df['dist 2 synapse'] > syn_distance_thresholds[di - 1]]
        dist_df_groups = dist_df.groupby('prediction')
        syn_distance_df.loc[di * 2: di * 2 + 1, 'number vesicles'] = np.array(dist_df_groups.size())
        close_syn_dist_df = dist_df[dist_df['at synapse?'] == 'True']
        if len(close_syn_dist_df) == 0:
            syn_distance_df.loc[di * 2: di * 2 + 1, 'fraction close synapse vesicles'] = 0
        else:
            close_dist_df_groups = close_syn_dist_df.groupby('prediction')
            fraction_close_vesicles = np.array(close_dist_df_groups.size()) / np.array(dist_df_groups.size())
            syn_distance_df.loc[di * 2: di * 2 + 1, 'fraction close synapse vesicles'] = fraction_close_vesicles
    syn_distance_df.to_csv(f'{f_name}/distance_close_synapse.csv')
    sns.barplot(data=syn_distance_df, x='distance', y='number vesicles', hue='prediction',
                palette=ct_palette)
    plt.xlabel('distance [nm]', fontsize=fontsize)
    plt.ylabel('number vesicles', fontsize=fontsize)
    plt.title('close membrane vesicles')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/num_ves_dist_close_synapse.png')
    plt.savefig(f'{f_name}/num_ves_dist_close_synapse.svg')
    plt.close()
    sns.barplot(data=syn_distance_df, x='distance', y='fraction close synapse vesicles', hue='prediction',
                palette=ct_palette)
    plt.xlabel('distance [nm]', fontsize=fontsize)
    plt.ylabel('fraction close synapse vesicles', fontsize=fontsize)
    plt.title('close synapse vesicles')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/frac_ves_dist_close_synapse.png')
    plt.savefig(f'{f_name}/frac_ves_dist_close_synapse.svg')
    plt.close()

    eval_true_df.to_csv(f'{f_name}/eval_true_df.csv')
    overview_df.to_csv(f'{f_name}/overview_df.csv')
    raise ValueError

    #plot also if in axon or not

    #for those that are single vesicles
    #plot number of close membrane vesicles
    #plot number of true and close membrane vesicles depending on distance from membrane calculated

    #plot if really synaptic or not
