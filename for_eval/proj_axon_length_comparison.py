#plot lengths of axonic fragments for HVC, LMAN, DA

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
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import get_cell_length_chunks
    from syconn.mp.mp_utils import start_multiprocess_imap
    from scipy.stats import kruskal, ranksums
    from itertools import combinations
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors

    #global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    #global_params.wd = '/cajal/nvmescratch/projects/data/songbird/j0251/j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_20230822'

    version = 'v6'
    bio_params = Analysis_Params(version=version)
    global_params.wd = bio_params.working_dir()
    ct_dict = bio_params.ct_dict()
    use_gt = False
    filter_syns = True
    color_key = 'AxRdYwBev6'
    cls = CelltypeColors(ct_dict=ct_dict)
    ct_palette = cls.ct_palette(color_key, num=False)
    fontsize = 20
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/250122_j0251{version}_ax_fraglengths_f{fontsize}_filter_syns"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('Projecting axon lengths', log_dir=f_name + '/logs/')
    if use_gt:
        #for gt: agglo2 uses v4 (11 neuron classes, no glia, no fragments), gt v5 (manually checked gt for mergers, add fragment class + some astros)
        #v5 wd uses v6: fragment class, astros, oligo + microglia, v6 wd: v7: INT1⁻3 instead of FS, NGF, + migrating neurons; add short axon fragments
        gt_path = "cajal/nvmescratch/projects/data/songbird/j0251/groundtruth/celltypes/j0251_celltype_gt_v7_j0251_72_seg_20210127_agglo2_IDs.csv"
        #gt_path = 'cajal/nvmescratch/users/arother/202301_syconnv5_wd_tests/20231013_new_celltype_gt/231115_ar_j0251_celltype_gt_v7_j0251_72_seg_20210127_agglo2_IDs.csv'
        gt = pd.read_csv(gt_path,names=["cellids", "celltype"])
        log.info(f'Ground truth cells from {gt_path} used for analysis')
    else:
        ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
        cellids = ssd.ssv_ids
        celltypes = ssd.load_numpy_data(bio_params.celltype_key())
        log.info(f'Axons used from {global_params.wd}')
    if filter_syns:
        log.info('Only axons used that have one axo-somatic or axo-dendritic outgoing synapse')
        log.info('No filter was applied to syn_prob or min_syn_size')

    #get cellids of all HVC. LMAN and DA fragments
    log.info('Step 1/4: Get cellids from all projecting axons (HVC, LMAN, DA)')
    ax_cts = bio_params.axon_cts()

    all_ax_ids = []
    all_ax_cts = []
    for ct in ax_cts:
        if use_gt:
            ct_ids = gt['cellids'][gt['celltype'] == ct_dict[ct]]
        else:
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
        ct_inds = np.in1d(syn_cts, ax_cts).reshape(len(syn_cts), 2)
        ct_ax_inds = np.all(ax_inds == ct_inds, axis = 1)
        syn_axs = syn_axs[ct_ax_inds]
        syn_cts = syn_cts[ct_ax_inds]
        syn_ssv_partners = syn_ssv_partners[ct_ax_inds]
        #get cellids on presynaptic side
        ax_inds = np.where(syn_axs == 1)
        pre_cellids = syn_ssv_partners[ax_inds]
        unique_pre_cellids = np.unique(pre_cellids)
        filtered_inds = np.in1d(all_ax_ids, unique_pre_cellids)
        axon_df = axon_df[filtered_inds]
        all_ax_ids = all_ax_ids[filtered_inds]
        axon_df = axon_df.reset_index(drop = True)
        ct_group_sizes = axon_df.groupby('celltype').size()
        log.info(f'After synapse filtering: total of {len(axon_df)} axons with the following sizes {ct_group_sizes}')

    log.info('Step 2/4: Get total length from all cellids')
    cellid_chunks = np.array_split(all_ax_ids, np.ceil(len(all_ax_ids) / 1000))
    all_lengths = start_multiprocess_imap(get_cell_length_chunks, cellid_chunks)
    all_lengths = np.concatenate(all_lengths)
    axon_df['skeleton length'] = np.array(all_lengths)
    #remove fragments with length = 0
    zero_lengths_ind = np.where(np.array(all_lengths) == 0)[0]
    zero_lengths_df = axon_df.loc[zero_lengths_ind]
    zero_lengths_df.to_csv(f'{f_name}/zero_lengths.csv')
    nonzero_axon_df = axon_df.replace(0, np.nan)
    nonzero_axon_df = nonzero_axon_df.dropna()
    nonzero_axon_df = nonzero_axon_df.reset_index(drop = True)
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
        ct_lengths_values = ct_lengths['skeleton length']
        param_df.loc['total number', ct_dict[ct]] = len(ct_lengths)
        param_df.loc['mean length', ct_dict[ct]] = np.mean(np.array(ct_lengths_values))
        param_df.loc['median length', ct_dict[ct]] = np.median(np.array(ct_lengths_values))

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
                 kde=False, element='step', palette=ct_palette, linewidth=3, bins = 100)
    plt.ylabel('number of axons', fontsize = fontsize)
    plt.xlabel('skeleton length [µm]', fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.xticks(fontsize = fontsize)
    plt.title('Lengths of axon fragments')
    plt.savefig(f'{f_name}/proj_axons_lengths.png')
    plt.savefig(f'{f_name}/proj_axons_lengths.svg')
    plt.close()
    sns.histplot(data=axon_df, x='skeleton length', hue='celltype', fill=False,
                 kde=False, element='step', stat='percent', palette=ct_palette, linewidth=3, common_norm=False)
    plt.xlabel('skeleton length [µm]', fontsize = fontsize)
    plt.title('Lengths of axon fragments', fontsize = fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/proj_axons_lengths_perc.png')
    plt.savefig(f'{f_name}/proj_axons_lengths_perc.svg')
    plt.close()
    sns.histplot(data=axon_df, x='skeleton length', hue='celltype', fill=False,
                 kde=False, element='step', log_scale=True, palette=ct_palette, linewidth=3)
    plt.ylabel('number of axons', fontsize = fontsize)
    plt.xlabel('skeleton length [µm]', fontsize = fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.title('Lengths of axon fragments')
    plt.savefig(f'{f_name}/proj_axons_lengths_log.png')
    plt.savefig(f'{f_name}/proj_axons_lengths_log.svg')
    plt.close()
    sns.histplot(data=axon_df, x='skeleton length', hue='celltype', fill=False,
                 kde=False, element='step', log_scale=True, stat='percent', palette=ct_palette, linewidth=3, common_norm=False)
    plt.xlabel('skeleton length [µm]', fontsize = fontsize)
    plt.ylabel('percent of axons', fontsize = fontsize)
    plt.title('Lengths of axon fragments', fontsize = fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/proj_axons_lengths_log_perc.png')
    plt.savefig(f'{f_name}/proj_axons_lengths_log_perc.svg')
    plt.close()
    sns.histplot(data=axon_df, x='skeleton length', hue='celltype', fill=False,
                 kde=False, element='step', log_scale=True, stat='percent', palette=ct_palette, linewidth=3,
                 common_norm=False, bins = 50)
    plt.xlabel('skeleton length [µm]', fontsize=fontsize)
    plt.ylabel('percent of axons', fontsize=fontsize)
    plt.title('Lengths of axon fragments', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/proj_axons_lengths_log_perc_b50.png')
    plt.savefig(f'{f_name}/proj_axons_lengths_log_perc_b50.svg')
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
        axon_df['synapse density'] = axon_df['syn number'] / axon_df['skeleton length']
        #split into bins of different length
        cats = [0.0, 10, 50, 100, 500, 1000, 5000, np.max(axon_df['skeleton length'])]
        length_cats = np.array(pd.cut(axon_df['skeleton length'], cats, right=False, labels=cats[:-1]))
        axon_df['length bins'] = length_cats
        axon_df.to_csv(f'{f_name}/proj_axon_lengths.csv')
        #make boxplot with hue
        sns.boxplot(data = axon_df, x = 'length bins', y= 'synapse density', hue = 'celltype', palette=ct_palette)
        plt.ylabel('synapse density [1/µm]', fontsize = fontsize)
        plt.xlabel('length bins [µm]', fontsize = fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(f'{f_name}/syn_density_length_bins.png')
        plt.savefig(f'{f_name}/syn_density_length_bins.svg')
        plt.close()
        sns.boxplot(data=axon_df, x='length bins', y='syn number', hue='celltype', palette=ct_palette)
        plt.ylabel('synapse number', fontsize=fontsize)
        plt.xlabel('length bins [µm]', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(f'{f_name}/syn_num_length_bins.png')
        plt.savefig(f'{f_name}/syn_num_length_bins.svg')
        plt.close()
        #make boxplot again with only skeleton lengths larger than one
        axon_df_one = axon_df[axon_df['skeleton length'] > 1]
        sns.boxplot(data=axon_df_one, x='length bins', y='synapse density', hue='celltype', palette=ct_palette)
        plt.ylabel('synapse density [1/µm]', fontsize=fontsize)
        plt.xlabel('length bins [µm]', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.title('Synapse density for fragment with length of > 1 µm')
        plt.savefig(f'{f_name}/filtered_syn_density_length_bins.png')
        plt.savefig(f'{f_name}/filtered_syn_density_length_bins.svg')
        plt.close()
        axon_df_cut = axon_df[axon_df['synapse density'] <= 1]
        sns.boxplot(data=axon_df_cut, x='length bins', y='synapse density', hue='celltype', palette=ct_palette)
        plt.ylabel('synapse density [1/µm]', fontsize=fontsize)
        plt.xlabel('length bins [µm]', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.title('Synapse density up to 1 with fragments > 1 µm')
        plt.savefig(f'{f_name}/cut_syn_density_length_bins.png')
        plt.savefig(f'{f_name}/cut_syn_density_length_bins.svg')
        plt.close()
        #make histogram with lengths bins to see number
        sns.histplot(data=axon_df, x='length bins', hue='celltype',fill=False,
                 kde=False, element='step', palette=ct_palette, linewidth=3)
        plt.ylabel('number of axons', fontsize=fontsize)
        plt.xlabel('length bins [µm]', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(f'{f_name}/length_bins_hist.png')
        plt.savefig(f'{f_name}/length_bins_hist.svg')
        plt.close()
        sns.histplot(data=axon_df, x='length bins', hue='celltype',fill=False,
                 kde=False, element='step', palette=ct_palette, stat='percent', linewidth=3)
        plt.ylabel('percent of axons', fontsize=fontsize)
        plt.xlabel('length bins [µm]', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(f'{f_name}/length_bins_hist_perc.png')
        plt.savefig(f'{f_name}/length_bins_hist_perc.svg')
        plt.close()
        #plot synapse number vs axon length
        sns.scatterplot(data = axon_df, x = 'skeleton length', y = 'syn number', hue = 'celltype', palette = ct_palette, alpha = 0.2)
        plt.ylabel('synapse number', fontsize=fontsize)
        plt.xlabel('skeleton length[µm]', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(f'{f_name}/syn_num_len_scatter.png')
        plt.savefig(f'{f_name}/syn_num_len_scatter.svg')
        plt.close()
        sns.kdeplot(data=axon_df, x='skeleton length', y='syn number', hue='celltype', palette=ct_palette, alpha=0.2)
        plt.ylabel('synapse number', fontsize=fontsize)
        plt.xlabel('skeleton length[µm]', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(f'{f_name}/syn_num_len_kde.png')
        plt.savefig(f'{f_name}/syn_num_len_kde.svg')
        plt.close()

    if use_gt:
        #make plot with different categories and how many cells are in there
        cats = [0.0, 50, 100, 500, 1000, np.max(axon_df['skeleton length'])]
        length_cats = np.array(pd.cut(axon_df['skeleton length'], cats, right=False, labels=cats[:-1]))
        axon_df['lengths bins'] = length_cats
        axon_df.to_csv(f'{f_name}/proj_axon_lengths.csv')
        length_sizes_df = pd.DataFrame(columns=['HVC', 'LMAN', 'DA'], index = cats)
        num_cats = len(cats) - 1
        length_sizes_bins_plotting = pd.DataFrame(columns=['celltype', 'length bins', 'number of axons'],
                                                  index=range(num_cats * len(ax_cts)))
        for a, ax_ct in enumerate(ax_cts):
            ax_axon_df = axon_df[axon_df['celltype'] == ct_dict[ax_ct]]
            ax_length_sizes = ax_axon_df.groupby('lengths bins').size()
            length_sizes_bins_plotting.loc[a * num_cats: (a + 1) * num_cats -1, 'celltype'] = ct_dict[ax_ct]
            length_sizes_bins_plotting.loc[a * num_cats: (a + 1) * num_cats - 1, 'length bins'] = cats[:-1]
            for cat in ax_length_sizes.keys():
                length_sizes_df.loc[cat, ct_dict[ax_ct]] = ax_length_sizes[cat]
            for ci, cat in enumerate(cats[:-1]):
                try:
                    length_sizes_bins_plotting.loc[a * num_cats + ci: (a + 1) * num_cats + ci - 1, 'number of axons'] = \
                        ax_length_sizes[cat]
                except KeyError:
                    length_sizes_bins_plotting.loc[a * num_cats + ci: (a + 1) * num_cats + ci - 1, 'number of axons'] = np.nan
        length_sizes_df.to_csv(f'{f_name}/length_bins_summary.csv')
        length_sizes_bins_plotting.to_csv((f'{f_name}/length_bins_plotting_summary.csv'))
        sns.barplot(data = length_sizes_bins_plotting, x = 'length bins', y = 'number of axons', hue = 'celltype', palette=ct_palette, errorbar=None)
        plt.xlabel('length bins [µm]', fontsize = fontsize)
        plt.ylabel('number of axons', fontsize = fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(f'{f_name}/length_bins_num_axs.png')
        plt.savefig(f'{f_name}/length_bins_num_axs.svg')
        plt.close()


    log.info('Analysis done')






