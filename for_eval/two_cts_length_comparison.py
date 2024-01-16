#similar to plot proj axon length comparison
#plot axon length comparison for any two celltypes

if __name__ == '__main__':
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    from syconn.reps.segmentation import SegmentationDataset
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from syconn.handler.config import initialize_logging
    import os as os
    import pandas as pd
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import get_cell_length, get_compartment_length_mp
    from syconn.mp.mp_utils import start_multiprocess_imap
    from scipy.stats import kruskal, ranksums
    from itertools import combinations

    #global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    global_params.wd = '/cajal/nvmescratch/projects/data/songbird/j0251/j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_20230822'

    version = 'v6'
    bio_params = Analysis_Params(working_dir=global_params.wd, version=version)
    ct_dict = bio_params.ct_dict()
    use_gt = False
    filter_syns = False
    ct1 = 4
    ct2 = 0
    ct1_str = ct_dict[ct1]
    ct2_str = ct_dict[ct2]
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/240115_j0251{version}_ax_lengths_comparison_{ct1_str}_{ct2_str}_nosyn"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('Comparison axon lengths', log_dir=f_name + '/logs/')
    if use_gt:
        gt_path = "cajal/nvmescratch/projects/data/songbird/j0251/groundtruth/celltypes/j0251_celltype_gt_v7_j0251_72_seg_20210127_agglo2_IDs.csv"
        #gt_path = 'cajal/nvmescratch/users/arother/202301_syconnv5_wd_tests/20231013_new_celltype_gt/231115_ar_j0251_celltype_gt_v7_j0251_72_seg_20210127_agglo2_IDs.csv'
        gt = pd.read_csv(gt_path,names=["cellids", "celltype"])
        log.info(f'Ground truth cells from {gt_path} used for analysis')
    else:
        ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
        cellids = ssd.ssv_ids
        celltypes = ssd.load_numpy_data('celltype_pts_e3')
        log.info(f'Axons used from {global_params.wd}')
    if filter_syns:
        log.info('Only axons used that have one axo-somatic or axo-dendritic outgoing synapse')
        log.info('No filter was applied to syn_prob or min_syn_size')

    #get cellids of all HVC. LMAN and DA fragments
    log.info(f'Step 1/4: Get cellids from celltypes to compare: {ct1_str}, {ct2_str}')
    comp_cts = [ct1, ct2]
    axon_cts = bio_params.axon_cts()

    all_cell_ids = []
    all_cell_cts = []
    ids_dict = {}
    for ct in comp_cts:
        if use_gt:
            ct_ids = gt['cellids'][gt['celltype'] == ct_dict[ct]]
        else:
            ct_inds = np.where(celltypes == ct)[0]
            ct_ids = cellids[ct_inds]
        log.info(f'{len(ct_ids)} cellids for {ct_dict[ct]}')
        all_cell_ids.append(ct_ids)
        ids_dict[ct] = ct_ids
        all_cell_cts.append([ct_dict[ct] for i in ct_ids])

    all_cell_ids = np.hstack(all_cell_ids)
    all_cell_cts = np.hstack(all_cell_cts)
    number_cellids = len(all_cell_ids)
    axon_df = pd.DataFrame(columns = ['cellid', 'celltype', 'axon skeleton length'], index = range(number_cellids))
    axon_df['cellid'] = all_cell_ids
    axon_df['celltype'] = all_cell_cts

    if filter_syns:
        log.info('Step 1b: Filter for fragments with synapses')
        sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.wd)
        syn_axs = sd_synssv.load_numpy_data('partner_axoness')
        syn_axs[syn_axs == 3] = 1
        syn_axs[syn_axs == 4] = 1
        syn_ssv_partners = sd_synssv.load_numpy_data('neuron_partners')
        syn_cts = sd_synssv.load_numpy_data('partner_celltypes')
        #filter for axo-dendritic or axo-somatic
        ax_inds = np.any(np.in1d(syn_axs, 1).reshape(len(syn_axs), 2), axis = 1)
        syn_axs = syn_axs[ax_inds]
        syn_cts = syn_cts[ax_inds]
        syn_ssv_partners = syn_ssv_partners[ax_inds]
        denso_inds = np.any(np.in1d(syn_axs, [0,2]).reshape(len(syn_axs), 2), axis = 1)
        syn_axs = syn_axs[denso_inds]
        syn_cts = syn_cts[denso_inds]
        syn_ssv_partners = syn_ssv_partners[denso_inds]
        # filter for ax_cts at axon
        ax_inds = np.in1d(syn_axs, 1).reshape(len(syn_axs), 2)
        ct_inds = np.in1d(syn_cts, comp_cts).reshape(len(syn_cts), 2)
        ct_ax_inds = np.all(ax_inds == ct_inds, axis = 1)
        syn_axs = syn_axs[ct_ax_inds]
        syn_cts = syn_cts[ct_ax_inds]
        syn_ssv_partners = syn_ssv_partners[ct_ax_inds]
        #get cellids on presynaptic side
        ax_inds = np.where(syn_axs == 1)
        pre_cellids = syn_ssv_partners[ax_inds]
        unique_pre_cellids = np.unique(pre_cellids)
        filtered_inds = np.in1d(all_cell_ids, unique_pre_cellids)
        axon_df = axon_df[filtered_inds]
        all_cell_ids = all_cell_ids[filtered_inds]
        axon_df = axon_df.reset_index(drop = True)
        ct_group_sizes = axon_df.groupby('celltype').size()
        for ct in comp_cts:
            filtered_ids = axon_df['cellid'][axon_df['celltype'] == ct_dict[ct]]
            ids_dict[ct] = filtered_ids
        log.info(f'After synapse filtering: total of {len(axon_df)} axons with the following sizes {ct_group_sizes}')

    log.info('Step 2/4: Get total length from all cellids')
    all_lengths = []
    for ct in comp_cts:
        log.info(f'Now processing length of {ct_dict[ct]}')
        if ct in axon_cts:
            ct_lengths = start_multiprocess_imap(get_cell_length, ids_dict[ct])
        else:
            full_ct_dict = bio_params.load_cell_dict(ct)
            comp_input = [[cellid, 1, None, full_ct_dict] for cellid in ids_dict[ct]]
            ct_lengths = start_multiprocess_imap(get_compartment_length_mp, comp_input)
        all_lengths.append(ct_lengths)
    all_lengths = np.hstack(all_lengths)
    axon_df['axon skeleton length'] = np.array(all_lengths)
    #remove fragments with length = 0
    zero_lengths_ind = np.where(np.array(all_lengths) == 0)[0]
    zero_lengths_df = axon_df.loc[zero_lengths_ind]
    zero_lengths_df.to_csv(f'{f_name}/zero_lengths.csv')
    nonzero_axon_df = axon_df.replace(0, np.nan)
    nonzero_axon_df = nonzero_axon_df.dropna()
    nonzero_axon_df = nonzero_axon_df.reset_index(drop = True)
    assert(len(axon_df) == len(zero_lengths_df) + len(nonzero_axon_df))
    ct1_0 = len(zero_lengths_df[zero_lengths_df['celltype'] == ct1_str])
    ct2_0 = len(zero_lengths_df[zero_lengths_df['celltype'] == ct2_str])
    log.info(f'{len(zero_lengths_df)} ids were removed because length was 0; {ct1_0} {ct1_str}, {ct2_0} {ct2_str}')
    axon_df = nonzero_axon_df
    axon_df.to_csv(f'{f_name}/ax_comp_{ct1_str}_{ct2_str}.csv')
    #get parameters for cell ids
    ax_groups_str = np.unique(axon_df['celltype'])
    param_df = pd.DataFrame(columns = ax_groups_str, index = ['total number', 'mean length', 'median length'])
    for ct in comp_cts:
        ct_lengths = axon_df[axon_df['celltype'] == ct_dict[ct]]
        ct_lengths.to_csv(f'{f_name}/lengths_{ct_dict[ct]}.csv')
        ct_lengths_values = ct_lengths['axon skeleton length']
        param_df.loc['total number', ct_dict[ct]] = len(ct_lengths)
        param_df.loc['mean length', ct_dict[ct]] = np.mean(np.array(ct_lengths_values))
        param_df.loc['median length', ct_dict[ct]] = np.median(np.array(ct_lengths_values))

    param_df.to_csv(f'{f_name}/summary_params_{ct1_str}_{ct2_str}.csv')

    log.info('Step 3/4: Calculate statistics')
    lengths_groups = [group['axon skeleton length'].values for name, group in
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
    sns.histplot(data = axon_df, x = 'axon skeleton length', hue = 'celltype', fill=False,
                 kde=False, element='step')
    plt.ylabel('number of cells')
    plt.xlabel('axon skeleton length [µm]')
    plt.title('Lengths of axons')
    plt.savefig(f'{f_name}/comp_ax_lengths_{ct1_str}_{ct2_str}.png')
    plt.savefig(f'{f_name}/comp_ax_lengths_{ct1_str}_ {ct2_str}.svg')
    plt.close()
    sns.histplot(data=axon_df, x='axon skeleton length', hue='celltype', fill=False,
                 kde=False, element='step', stat='percent')
    plt.xlabel('axon skeleton length [µm]')
    plt.title('Lengths of axons')
    plt.savefig(f'{f_name}/comp_ax_lengths_perc_{ct1_str}_{ct2_str}.png')
    plt.savefig(f'{f_name}/comp_ax_lengths_perc_{ct1_str}_{ct2_str}.svg')
    plt.close()
    sns.histplot(data=axon_df, x='axon skeleton length', hue='celltype', fill=False,
                 kde=False, element='step', log_scale=True)
    plt.ylabel('number of cells')
    plt.xlabel('axon skeleton length [µm]')
    plt.title('Lengths of axons')
    plt.savefig(f'{f_name}/comp_ax_lengths_log_{ct1_str}_{ct2_str}.png')
    plt.savefig(f'{f_name}/comp_ax_lengths_log_{ct1_str}_{ct2_str}.svg')
    plt.close()
    sns.histplot(data=axon_df, x='axon skeleton length', hue='celltype', fill=False,
                 kde=False, element='step', log_scale=True, stat='percent')
    plt.xlabel('axon skeleton length [µm]')
    plt.title('Lengths of axons')
    plt.savefig(f'{f_name}/comp_ax_lengths_log_perc_{ct1_str}_{ct2_str}.png')
    plt.savefig(f'{f_name}/comp_ax_lengths_log_perc_{ct1_str}_{ct2_str}.svg')
    plt.close()

    if filter_syns:
        log.info('Step 4b: Plot synapse density dependent on fragment length')
        #get number of synapses for each cellid
        ssv_inds, unique_ssv_ids = pd.factorize(pre_cellids)
        syn_numbers = np.bincount(ssv_inds)
        nonzero_inds = np.in1d(unique_ssv_ids, axon_df['cellid'])
        unique_ssv_ids = unique_ssv_ids[nonzero_inds]
        syn_numbers = syn_numbers[nonzero_inds]
        sort_inds = np.argsort(unique_ssv_ids)
        sorted_unique_ids = unique_ssv_ids[sort_inds]
        sorted_syn_numbers = syn_numbers[sort_inds]
        axon_df = axon_df.sort_values(by = 'cellid')
        # calculate synapse density
        axon_df['syn number'] = sorted_syn_numbers
        axon_df['synapse density'] = axon_df['syn number'] / axon_df['axon skeleton length']
        #split into bins of different length
        cats = [0.0, 10, 50, 100, 500, 1000, 5000, np.max(axon_df['axon skeleton length'])]
        length_cats = np.array(pd.cut(axon_df['axon skeleton length'], cats, right=False, labels=cats[:-1]))
        axon_df['length bins'] = length_cats
        axon_df.to_csv(f'{f_name}/comp_ax_lengths_{ct1_str}_{ct2_str}.csv')
        #make boxplot with hue
        sns.boxplot(data = axon_df, x = 'length bins', y= 'synapse density', hue = 'celltype')
        plt.ylabel('synapse density [1/µm]')
        plt.savefig(f'{f_name}/syn_density_length_bins_{ct1_str}_{ct2_str}.png')
        plt.savefig(f'{f_name}/syn_density_length_bins_{ct1_str}_{ct2_str}.svg')
        plt.close()
        #make boxplot again with only skeleton lengths larger than one
        axon_df_one = axon_df[axon_df['axon skeleton length'] > 1]
        sns.boxplot(data=axon_df_one, x='length bins', y='synapse density', hue='celltype')
        plt.ylabel('synapse density [1/µm]')
        plt.title('Synapse density for cellids with axon length of > 1 µm')
        plt.savefig(f'{f_name}/filtered_syn_density_length_bins_{ct1_str}_{ct2_str}.png')
        plt.savefig(f'{f_name}/filtered_syn_density_length_bins_{ct1_str}_{ct2_str}.svg')
        plt.close()
        axon_df_cut = axon_df[axon_df['synapse density'] <= 1]
        sns.boxplot(data=axon_df_cut, x='length bins', y='synapse density', hue='celltype')
        plt.ylabel('synapse density [1/µm]')
        plt.title('Synapse density up to 1 cellids with axon length > 1 µm')
        plt.savefig(f'{f_name}/cut_syn_density_length_bins_{ct1_str}_{ct2_str}.png')
        plt.savefig(f'{f_name}/cut_syn_density_length_bins_{ct1_str}_{ct2_str}.svg')
        plt.close()
        #make histogram with lengths bins to see number
        sns.histplot(data=axon_df, x='length bins', hue='celltype',fill=False,
                 kde=False, element='step')
        plt.savefig(f'{f_name}/length_bins_hist_{ct1_str}_{ct2_str}.png')
        plt.savefig(f'{f_name}/length_bins_hist_{ct1_str}_{ct2_str}.svg')
        plt.close()
        sns.histplot(data=axon_df, x='length bins', hue='celltype',fill=False,
                 kde=False, element='step')
        plt.savefig(f'{f_name}/length_bins_hist_perc_{ct1_str}_{ct2_str}.png')
        plt.savefig(f'{f_name}/length_bins_hist_perc_{ct1_str}_{ct2_str}.svg')
        plt.close()

    if use_gt:
        #make plot with different categories and how many cells are in there
        cats = [0.0, 50, 100, 500, 1000, np.max(axon_df['axon skeleton length'])]
        length_cats = np.array(pd.cut(axon_df['axon skeleton length'], cats, right=False, labels=cats[:-1]))
        axon_df['lengths bins'] = length_cats
        axon_df.to_csv(f'{f_name}/comp_ax_lengths_{ct1_str}_{ct2_str}.csv')
        length_sizes_df = pd.DataFrame(columns=[ct1_str, ct2_str], index = cats)
        for ax_ct in comp_cts:
            ax_axon_df = axon_df[axon_df['celltype'] == ct_dict[ax_ct]]
            ax_length_sizes = ax_axon_df.groupby('lengths bins').size()
            for cat in ax_length_sizes.keys():
                length_sizes_df.loc[cat, ct_dict[ax_ct]] = ax_length_sizes[cat]
        length_sizes_df.to_csv(f'{f_name}/length_bins_summary_{ct1_str}_{ct2_str}.csv')

    log.info('Analysis done')