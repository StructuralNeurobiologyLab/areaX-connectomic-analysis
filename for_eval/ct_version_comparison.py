#check where new HVC_ids come from

if __name__ == '__main__':
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os as os
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from scipy.stats import kruskal, ranksums
    from itertools import combinations
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params

    #v6_wd = '/cajal/nvmescratch/projects/data/songbird/j0251/j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_20230822'
    #v5_wd = '/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811'

    version1 = 'v6'
    version2 = 'v4'
    analysis_params1 = Analysis_Params(version = version1)
    v1_wd = analysis_params1.working_dir()
    v1_ct_dict = analysis_params1.ct_dict(with_glia = True)
    ct_key1 = analysis_params1.celltype_key()
    ct_certainty1 = analysis_params1.celltype_certainty_key()
    analysis_params2 = Analysis_Params(version=version2)
    v2_wd = analysis_params2.working_dir()
    v2_ct_dict = analysis_params2.ct_dict(with_glia=True)
    ct_key2 = analysis_params2.celltype_key()
    ct_certainty2 = analysis_params2.celltype_certainty_key()


    color_key = 'AxRdYwBev5'
    ct_str = 'LMAN'
    ct_num_1 = 1
    ct_num_2 = 3
    fontsize = 20
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/250213_j0251{version1}_vs_{version2}_{ct_str}_cellids_{color_key}_f{fontsize}"
    if not os.path.exists(f_name):
        os.mkdir(f_name)

    ct_colors = CelltypeColors(ct_dict = v2_ct_dict)
    ct_palette = ct_colors.ct_palette(key = color_key)

    #load all cellids and celltypes
    ssd_v1 = SuperSegmentationDataset(working_dir=v1_wd)
    ssd_v2 = SuperSegmentationDataset(working_dir=v2_wd)
    v1_cellids = ssd_v1.ssv_ids
    v2_cellids = ssd_v2.ssv_ids
    v1_celltypes = ssd_v1.load_numpy_data(ct_key1)
    v2_celltypes = ssd_v2.load_numpy_data(ct_key2)
    summary_pd = pd.DataFrame(columns = [version1, version2])
    #get certainties of celltypes
    v1_certainty = ssd_v1.load_numpy_data(ct_certainty1)
    v2_certainty = ssd_v2.load_numpy_data(ct_certainty2)

    #get celltype ids from both versions
    v1_ct_inds = v1_celltypes == ct_num_1
    v2_ct_inds = v2_celltypes == ct_num_2
    v1_ct_ids = v1_cellids[v1_ct_inds]
    v2_ct_ids = v2_cellids[v2_ct_inds]
    v1_ct_certainty = v1_certainty[v1_ct_inds]
    v2_ct_certainty = v2_certainty[v2_ct_inds]
    summary_pd.loc[f'total number {ct_str} ids', version1] = len(v1_ct_ids)
    summary_pd.loc[f'total number {ct_str} ids', version2] = len(v2_ct_ids)
    #check which ids from v5 are also in v6
    same_inds = np.in1d(v1_ct_ids, v2_ct_ids)
    same_ct_ids = v1_ct_ids[same_inds]
    num_same_ct = len(same_ct_ids)
    summary_pd.loc[f'number same {ct_str} ids', version1] = num_same_ct
    summary_pd.loc[f'number same {ct_str} ids', version2] = num_same_ct
    summary_pd.loc[f'percent same {ct_str} ids', version1] = 100 * num_same_ct/ len(v1_ct_ids)
    summary_pd.loc[f'percent same {ct_str} ids', version2] = 100 * num_same_ct / len(v2_ct_ids)
    summary_pd.loc['mean ct certainty', version1] = np.mean(v1_ct_certainty)
    summary_pd.loc['mean ct certainty', version2] = np.mean(v2_ct_certainty)
    summary_pd.to_csv(f'{f_name}/summary_df.csv')
    #get celltypes of all cellids that are now hvc
    v2_v1_inds = np.in1d(v2_cellids, v1_ct_ids)
    ct_celltypes_v2_all = v2_celltypes[v2_v1_inds]
    ct_certainty_v2_all = v2_certainty[v2_v1_inds]
    ct_celltypes_v2_all_str = [v2_ct_dict[i] for i in ct_celltypes_v2_all]
    v1_ids_df = pd.DataFrame(columns = ['cellid', f'celltype {version1}', f'certainty {version1}', f'celltype {version2}', f'certainty {version2}'], index = range(len(v1_ct_ids)))
    v1_ids_df['cellid'] = v1_ct_ids
    v1_ids_df[f'celltype {version2}'] = ct_celltypes_v2_all_str
    v1_ids_df[f'certainty {version2}'] = ct_certainty_v2_all
    v1_ids_df[f'celltype {version1}'] = ct_str
    v1_ids_df[f'certainty {version1}'] = v1_ct_certainty
    v1_ids_df.to_csv(f'{f_name}/{ct_str}_ids_{version2}_{version1}_cts.csv')
    #get statistics for v5 certainty
    #do kurskal wallis test for all celltypes
    stats_df = pd.DataFrame(columns = ['stats', 'p-value'])
    certainty_ct_groups = [group[f'certainty {version2}'].values for name, group in
                        v1_ids_df.groupby(f'celltype {version2}')]
    kruskal_v2_res = kruskal(*certainty_ct_groups, nan_policy='omit')
    stats_df.loc[f'kruskal {version2} cts', 'stats'] = kruskal_v2_res[0]
    stats_df.loc[f'kruskal {version2} cts', 'p-value'] = kruskal_v2_res[1]
    #compare certainty of ct in v5 vs v6
    ranksum_ct_res = ranksums(v1_ct_certainty, v2_ct_certainty)
    stats_df.loc[f'ranksum {ct_str} {version1} vs {version2}', 'stats'] = ranksum_ct_res[0]
    stats_df.loc[f'ranksum {ct_str} {version1} vs {version2}', 'p-value'] = ranksum_ct_res[1]
    #get ranksum results of pairwise comparison if kruskal < 0.05 for all cts in v5
    v2_unique_cts = np.unique(ct_celltypes_v2_all_str)
    num_unique_v2_cts = len(v2_unique_cts)
    if kruskal_v2_res[1] < 0.05:
        group_comps = list(combinations(range(num_unique_v2_cts), 2))
        for gc in group_comps:
            ranksum_res = ranksums(certainty_ct_groups[gc[0]], certainty_ct_groups[gc[1]])
            stats_df.loc[f'ranksum {v2_unique_cts[gc[0]]} vs {v2_unique_cts[gc[1]]}', 'stats'] = ranksum_res[0]
            stats_df.loc[f'ranksum {v2_unique_cts[gc[0]]} vs {v2_unique_cts[gc[1]]}', 'p-value'] = ranksum_res[1]
    stats_df.to_csv(f'{f_name}/stats_{ct_str}.csv')
    #plot according to celltypes
    v2_ct_groups = v1_ids_df.groupby(f'celltype {version2}')
    per_ct_numbers_v2 = pd.DataFrame(columns = [f'celltype {version2}', 'number of cells', 'percent of cells'])
    per_ct_numbers_v2[f'celltype {version2}'] = v2_ct_groups.groups.keys()
    ct_numbers = np.array(v2_ct_groups.size())
    per_ct_numbers_v2['number of cells'] = ct_numbers
    per_ct_numbers_v2['percent of cells'] = 100 * ct_numbers / np.sum(ct_numbers)
    per_ct_numbers_v2.to_csv(f'{f_name}/ct_{version2}_{ct_str}_numbers.csv')
    plot_order_cts = list(ct_palette.keys())
    sns.barplot(data = per_ct_numbers_v2, x=f'celltype {version2}', y = 'number of cells', palette=ct_palette, order=plot_order_cts)
    plt.ylabel('number of cells', fontsize = fontsize)
    plt.xlabel(f'celltype {version2}', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/number_{ct_str}_{version1}_{version2}_cts.png')
    plt.savefig(f'{f_name}/number_{ct_str}_{version1}_{version2}_cts.svg')
    plt.close()
    sns.barplot(data = per_ct_numbers_v2, x=f'celltype {version2}', y = 'percent of cells', palette=ct_palette, order=plot_order_cts)
    plt.ylabel('percent of cells', fontsize=fontsize)
    plt.xlabel(f'celltype {version2}', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/perc_{ct_str}_{version1}_{version2}_cts.png')
    plt.savefig(f'{f_name}/perc_{ct_str}_{version1}_{version2}_cts.svg')
    plt.close()
    #if seaborn 0.13.0 in environment
    #sns.countplot(data=v6_hvc_ids_df, x='celltype v5', palette=ct_palette, stat='percent')
    #plt.ylabel('% of cells')
    #plt.savefig(f'{f_name}/perc_{ct_str}_v6_v5_cts.png')
    #plt.close()
    #plot number per celltype
    # plot according to certainty in v5
    sns.boxplot(data=v1_ids_df, x=f'celltype {version2}', y=f'certainty {version2}', order=plot_order_cts, palette=ct_palette)
    plt.ylabel(f'certainty {version2}', fontsize=fontsize)
    plt.xlabel(f'celltype {version2}', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/certainty_{ct_str}_{version2}_cts.png')
    plt.savefig(f'{f_name}/certainty_{ct_str}_{version2}_cts.svg')
    plt.close()
    sns.boxplot(data=v1_ids_df, x=f'celltype {version2}', y=f'certainty {version1}', order=plot_order_cts, palette=ct_palette)
    plt.ylabel(f'certainty {version1}', fontsize=fontsize)
    plt.xlabel(f'celltype {version2}', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/certainty_{ct_str}_{version1}_cts.png')
    plt.savefig(f'{f_name}/certainty_{ct_str}_{version1}_cts.svg')
    plt.close()
