#plot lengths of axonic fragments for HVC, LMAN, DA

if __name__ == '__main__':
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from syconn.handler.config import initialize_logging
    import os as os
    import pandas as pd
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import get_cell_length
    from syconn.mp.mp_utils import start_multiprocess_imap
    from scipy.stats import kruskal, ranksums
    from itertools import combinations

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    bio_params = Analysis_Params(working_dir=global_params.wd, version='v5')
    ct_dict = bio_params.ct_dict()
    f_name = "cajal/scratch/users/arother/bio_analysis_results/for_eval/231107_j0251v5_ax_fraglengths"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('Projecting axon lengths', log_dir=f_name + '/logs/')

    #get cellids of all HVC. LMAN and DA fragments
    log.info('Step 1/4: Get cellids from all projecting axons (HVC, LMAN, DA)')
    ax_cts = bio_params.axon_cts()
    cellids = ssd.ssv_ids
    celltypes = ssd.load_numpy_data('celltype_pts_e3')
    all_ax_ids = []
    all_ax_cts = []
    for ct in ax_cts:
        ct_inds = np.where(celltypes == ct)[0]
        ct_ids = cellids[ct_inds]
        log.info(f'{len(ct_ids)} axons for {ct_dict[ct]}')
        all_ax_ids.append(ct_ids)
        all_ax_cts.append([ct_dict[ct] for i in ct_ids])


    all_ax_ids = np.hstack(all_ax_ids)
    all_ax_cts = np.hstack(all_ax_cts)
    number_axons = len(all_ax_ids)
    axon_df = pd.DataFrame(columns = ['cellid', 'celltype', 'skeleton length'], index = range(number_axons))
    axon_df['cellid'] = all_ax_ids
    axon_df['celltype'] = all_ax_cts

    log.info('Step 2/4: Get total length from all cellids')
    all_lengths = start_multiprocess_imap(get_cell_length, all_ax_ids)
    axon_df['skeleton length'] = np.array(all_lengths)
    #remove fragments with length = 0
    zero_lengths_ind = np.where(np.array(all_lengths) == 0)[0]
    zero_lengths_df = axon_df.loc[zero_lengths_ind]
    zero_lengths_df.to_csv(f'{f_name}/zero_lengths.csv')
    nonzero_axon_df = axon_df.replace(0, np.nan)
    nonzero_axon_df = nonzero_axon_df.dropna()
    nonzero_axon_df = nonzero_axon_df.reset_index()
    assert(len(axon_df) == len(zero_lengths_df) + len(nonzero_axon_df))
    da_0 = len(zero_lengths_df[zero_lengths_df['celltype'] == 'DA'])
    hvc_0 = len(zero_lengths_df[zero_lengths_df['celltype'] == 'HVC'])
    lman_0 = len(zero_lengths_df[zero_lengths_df['celltype'] == 'LMAN'])
    log.info(f'{len(zero_lengths_df)} ids were removed because length was 0; {da_0} DA, {hvc_0} HVC, {lman_0} LMAN')
    axon_df = nonzero_axon_df
    axon_df.to_csv(f'{f_name}/proj_axon_lengths.csv')
    #get parameters for cell ids
    ax_groups_str = np.unique(axon_df['celltype'])
    param_df = pd.DataFrame(columns = ax_groups_str, index = ['total number', 'mean length', 'median length'])
    for ct in ax_cts:
        ct_lengths = axon_df[axon_df['celltype'] == ct_dict[ct]]
        ct_lengths.to_csv(f'{f_name}/lengths_{ct_dict[ct]}.csv')
        param_df.loc['total number', ct_dict[ct]] = len(ct_lengths)
        param_df.loc['mean length', ct_dict[ct]] = np.mean(ct_lengths)
        param_df.loc['median length', ct_dict[ct]] = np.median(ct_lengths)

    param_df.to_csv(f'{f_name}/summary_params.csv')

    log.info('Step 3/4: Calculate statistics')
    lengths_groups = [group['skeleton length'].values for name, group in
                    axon_df.groupby('celltype')]
    kruskal_res = kruskal(*lengths_groups, nan_policy='omit')
    log.info(f'Kruskal results: stats = {kruskal_res[0]}, p-value = {kruskal_res[1]}')
    if kruskal_res[1] < 0.05:
        group_comps = list(combinations(range(len(ax_groups_str)), 2))
        ranksum_columns = [f'{ax_groups_str[gc[0]]} vs {ax_groups_str[gc[1]]}' for gc in group_comps]
        ranksum_res_df = pd.DataFrame(columns=ranksum_columns, index = ['stats', 'p-value'])
        for gc in group_comps:
            ranksum_res = ranksums(lengths_groups[gc[0]], lengths_groups[gc[1]])
            ranksum_res_df.loc[f'stats', f'{ax_groups_str[gc[0]]} vs {ax_groups_str[gc[1]]}'] = ranksum_res[0]
            ranksum_res_df.loc[f'p-value', f'{ax_groups_str[gc[0]]} vs {ax_groups_str[gc[1]]}'] = ranksum_res[1]
        ranksum_res_df.to_csv(f'{f_name}/ranksum_results.csv')

    log.info('Step 4/4: Plot results')
    sns.histplot(data = axon_df, x = 'skeleton length', hue = 'celltype', fill=False,
                 kde=False, element='step')
    plt.ylabel('number of axons')
    plt.xlabel('skeleton length [µm]')
    plt.title('Lengths of axon fragments')
    plt.savefig(f'{f_name}/proj_axons_lengths.png')
    plt.savefig(f'{f_name}/proj_axons_lengths.svg')
    plt.close()
    sns.histplot(data=axon_df, x='skeleton length', hue='celltype', fill=False,
                 kde=False, element='step', stat='percent')
    plt.xlabel('skeleton length [µm]')
    plt.title('Lengths of axon fragments')
    plt.savefig(f'{f_name}/proj_axons_lengths_perc.png')
    plt.savefig(f'{f_name}/proj_axons_lengths_perc.svg')
    plt.close()
    sns.histplot(data=axon_df, x='skeleton length', hue='celltype', fill=False,
                 kde=False, element='step', log_scale=True)
    plt.ylabel('number of axons')
    plt.xlabel('skeleton length [µm]')
    plt.title('Lengths of axon fragments')
    plt.savefig(f'{f_name}/proj_axons_lengths_log.png')
    plt.savefig(f'{f_name}/proj_axons_lengths_log.svg')
    plt.close()
    sns.histplot(data=axon_df, x='skeleton length', hue='celltype', fill=False,
                 kde=False, element='step', log_scale=True, stat='percent')
    plt.xlabel('skeleton length [µm]')
    plt.title('Lengths of axon fragments')
    plt.savefig(f'{f_name}/proj_axons_lengths_log_perc.png')
    plt.savefig(f'{f_name}/proj_axons_lengths_log_perc.svg')
    plt.close()

    log.info('Analysis done')




