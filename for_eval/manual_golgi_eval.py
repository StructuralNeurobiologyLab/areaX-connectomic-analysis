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
    from syconn.reps.super_segmentation import SuperSegmentationObject

    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    global_params.wd = analysis_params.working_dir()
    with_glia = True
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    fontsize = 20
    plotwidth = 0.5
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/250401_j0251{version}_manual_golgi_eval"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'golgi_eval_log', log_dir=f_name)

    eval_path = 'cajal/scratch/users/arother/bio_analysis_results/for_eval/240903_j0251v6_rndm_cellids_golgi_eval_mcl_200_samples_3/' \
                '240926_rnd_cellids_for_annot_golgi_RM.csv'
    eval_other_struc_path = 'cajal/scratch/users/arother/bio_analysis_results/for_eval/240903_j0251v6_rndm_cellids_golgi_eval_mcl_200_samples_3/' \
                '20241014_golgi_annot_other_structure.csv'
    log.info(f'Load manual evaluation results from {eval_path} and {eval_other_struc_path}')
    eval_df = pd.read_csv(eval_path)
    eval_other_struc_df = pd.read_csv(eval_other_struc_path)
    ct_palette = {'er': '#54A9DA', 'golgi':'#DE9B1E', 'mi': '#EE1BE0', 'syn':'#29FC15'}
    #get celltypes from cellids
    for i, cellid in enumerate(np.array(eval_df['cellid'])):
        cell = SuperSegmentationObject(cellid)
        cell.load_attr_dict()
        ct_num = cell.attr_dict['celltype_pts_e3']
        celltype = ct_dict[ct_num]
        eval_df.loc[i, 'celltype'] = celltype
    #make overview df with results
    unique_cts = np.unique(eval_df['celltype'])
    num_cts = len(unique_cts)
    ov_columns = ['celltype', 'number', 'percentage total', 'percentage false', 'category']
    cats = ['mapped total', 'mapped true', 'mapped false', 'true forgotten', 'mapped false soma', 'mapped false outside soma']
    cat_dict = {'mapped total': 'number separate mapped', 'mapped true':'number true golgi mapped',
                'mapped false': 'number falsely classified mapped', 'true forgotten':'number forgotten classified golgi',
                'mapped false soma': 'number falsely classified golgi in soma', 'mapped false outside soma':'number falsely classified golgi'}
    overview_df = pd.DataFrame(columns=ov_columns, index= range((num_cts + 1) * len(cats)))
    ct_cats = np.hstack(['total', unique_cts])
    ct_groups = eval_df.groupby('celltype')
    for i, ct in enumerate(ct_cats):
        overview_df.loc[i * len(cats): (i + 1) * len(cats) - 1, 'celltype'] = ct
        if not 'total' in ct:
            ct_group = ct_groups.get_group(ct)
        for ci, cat in enumerate(cats):
            overview_df.loc[i * len(cats) + ci, 'category'] = cat
            if 'total' in ct:
                overview_df.loc[i * len(cats) + ci, 'number'] = np.sum(eval_df[cat_dict[cat]])
            else:
                overview_df.loc[i * len(cats) + ci, 'number'] = np.sum(ct_group[cat_dict[cat]])

    #get percentage
    ov_cat_groups = overview_df.groupby('category')
    for cat in cats:
        cat_inds = overview_df['category'] == cat
        overview_df.loc[cat_inds, 'percentage total'] =  100 * np.array(ov_cat_groups.get_group(cat)['number']) / np.array(ov_cat_groups.get_group('mapped total')['number'])
        if 'soma' in cat:
            overview_df.loc[cat_inds, 'percentage false'] = 100 * np.array(ov_cat_groups.get_group(cat)['number']) / \
                                                            np.array(ov_cat_groups.get_group('mapped false')['number'])

    overview_df.to_csv(f'{f_name}/overview_df.csv')
    #get plot with total numbers
    full_cats = ['mapped true', 'mapped false', 'true forgotten']
    total_ov_df = overview_df[overview_df['celltype'] == 'total']
    total_ov_df_all = total_ov_df[np.in1d(total_ov_df['category'], full_cats)]
    sns.barplot(data = total_ov_df_all, x='category', y = 'percentage total', color=ct_palette['golgi'], width=plotwidth)
    plt.xlabel(f'category', fontsize=fontsize)
    plt.ylabel('percent of golgi stacks', fontsize=fontsize)
    plt.title(f'overview of separated golgi stacks')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/overview_golgi_perc.png')
    plt.savefig(f'{f_name}/overview_golgi_perc.svg')
    plt.close()
    sns.barplot(data=total_ov_df_all, x='category', y='number', color=ct_palette['golgi'], width=plotwidth)
    plt.xlabel(f'category', fontsize=fontsize)
    plt.ylabel('number of golgi stacks', fontsize=fontsize)
    plt.title(f'overview of separated golgi stacks')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/overview_golgi_num.png')
    plt.savefig(f'{f_name}/overview_golgi_num.svg')
    plt.close()

    #get plot with total number falsely mapped
    false_cats = ['mapped false soma', 'mapped false outside soma']
    total_ov_df_false = total_ov_df[np.in1d(total_ov_df['category'], false_cats)]
    soma_ind = np.where(total_ov_df_false['category'] == 'mapped false soma')[0][0]
    total_ov_df_false = total_ov_df_false.reset_index(drop = True)
    total_ov_df_false.loc[soma_ind, 'compartment'] = 'soma'
    non_soma_ind = np.where(total_ov_df_false['category'] == 'mapped false outside soma')[0][0]
    total_ov_df_false.loc[non_soma_ind, 'compartment'] = 'not soma'
    sns.barplot(data=total_ov_df_false, x='compartment', y='percentage false', color=ct_palette['golgi'], width=plotwidth)
    plt.xlabel(f'category', fontsize=fontsize)
    plt.ylabel('percent of golgi stacks', fontsize=fontsize)
    plt.title(f'overview of falsely mapped golgi stacks')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/overview_golgi_false_comps_perc.png')
    plt.savefig(f'{f_name}/overview_golgi_false_comps_perc.svg')
    plt.close()
    sns.barplot(data=total_ov_df_false, x='compartment', y='number', color=ct_palette['golgi'], width=plotwidth)
    plt.xlabel(f'category', fontsize=fontsize)
    plt.ylabel('number of golgi stacks', fontsize=fontsize)
    plt.title(f'overview of falesly mapped golgi stacks')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/overview_golgi_false_comps_num.png')
    plt.savefig(f'{f_name}/overview_golgi_false_comps_num.svg')
    plt.close()

    #plot with overview categories but with all celltypes
    cat_palette = {'mapped true': '#15AEAB', 'mapped false': '#232121', 'true forgotten': '#707070'}
    ct_ov_df = overview_df[overview_df['celltype'] != 'total']
    ct_ov_df_all = ct_ov_df[np.in1d(ct_ov_df['category'], full_cats)]
    sns.barplot(data=ct_ov_df_all, x='celltype', y='percentage total', hue = 'category', palette=cat_palette, width=plotwidth)
    plt.xlabel(f'category', fontsize=fontsize)
    plt.ylabel('percent of golgi stacks', fontsize=fontsize)
    plt.title(f'overview of separated golgi stacks in different celltypes')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/cts_golgi_perc.png')
    plt.savefig(f'{f_name}/cts_golgi_perc.svg')
    plt.close()
    sns.barplot(data=ct_ov_df_all, x='celltype', y='number', hue = 'category', palette=cat_palette, width=plotwidth)
    plt.xlabel(f'category', fontsize=fontsize)
    plt.ylabel('number of golgi stacks', fontsize=fontsize)
    plt.title(f'overview of separated golgi stacks in different celltypes')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/cts_golgi_num.png')
    plt.savefig(f'{f_name}/cts_golgi_num.svg')
    plt.close()

    #add plot with number of other category
    other_structures = np.unique(eval_other_struc_df['other structure'])
    eval_other_struc_ov_df = pd.DataFrame(columns= ['other structure', 'number', 'percentage'], index=range(len(other_structures)))
    struc_groups = eval_other_struc_df.groupby('other structure')
    sum_total = eval_other_struc_df['number occurrence'].sum()
    for i, struc in enumerate(other_structures):
        struc_df_number = struc_groups.get_group(struc)['number occurrence'].sum()
        eval_other_struc_ov_df.loc[i, 'other structure'] = struc
        eval_other_struc_ov_df.loc[i, 'number'] = struc_df_number
        eval_other_struc_ov_df.loc[i, 'percentage'] = 100 * struc_df_number / sum_total

    eval_other_struc_ov_df.to_csv(f'{f_name}/ov_other_structures.csv')
    sns.barplot(data=eval_other_struc_ov_df, x='other structure', y='percentage', color=ct_palette['golgi'], width=plotwidth)
    plt.xlabel(f'other structure', fontsize=fontsize)
    plt.ylabel('percent of golgi stacks', fontsize=fontsize)
    plt.title(f'overview of falsely mapped golgi stacks')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/overview_golgi_false_other_perc.png')
    plt.savefig(f'{f_name}/overview_golgi_false_other_perc.svg')
    plt.close()
    sns.barplot(data=eval_other_struc_ov_df, x='other structure', y='number', color=ct_palette['golgi'], width=plotwidth)
    plt.xlabel(f'other structure', fontsize=fontsize)
    plt.ylabel('number of golgi stacks', fontsize=fontsize)
    plt.title(f'overview of falesly mapped golgi stacks')
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/overview_golgi_false_other_num.png')
    plt.savefig(f'{f_name}/overview_golgi_false_other_num.svg')
    plt.close()

    log.info('Analysis done.')