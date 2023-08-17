#plot cs_ssv sizes for all cells
#plot then for all astrocyte cells
#then again for cells with astrocyte ids used by delta

if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    import os as os
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from syconn.reps.connectivity_helper import cs_id_to_partner_ids_vec

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    f_name = "cajal/scratch/users/arother/bio_analysis_results/general/230817_plot_cs_size_dsitribution" 
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    subsampling_hist = [10000, 1000, 100]
    log = initialize_logging('plot cs_size distribution', log_dir=f_name + '/logs/')
    log.info(f'use different subsamplings to get distribution: {subsampling_hist}')
    log.info('Use DS astrocytes')

    log.info('Step 1/2: Load and plot histogram for cs_sizes')
    sd_cs = SegmentationDataset('cs')
    cs_ids = sd_cs.ids
    cs_sizes = sd_cs.load_numpy_data('size')
    log.info(f' In total there are {len(cs_ids)} cs_ids')
    #subsample data and create histogramm of sizes
    for subs in subsampling_hist:
        sizes_subs = cs_sizes[::subs]
        sns.histplot(sizes_subs, fill=False, element='step')
        plt.xlabel('number of voxel')
        plt.ylabel('count of cs')
        plt.savefig(f'{f_name}/{subs}_hist_cs_sizes.png')
        plt.close()
        sns.histplot(sizes_subs, fill=False, element='step', log_scale=10)
        plt.xlabel('number of voxel')
        plt.ylabel('count of cs')
        plt.savefig(f'{f_name}/{subs}_hist_cs_sizes_log.png')
        plt.close()
    #plot distribution for all larger 10 000
    larger_sizes = cs_sizes[cs_sizes > 10000]
    for subs in subsampling_hist:
        sizes_subs = larger_sizes[::subs]
        sns.histplot(sizes_subs, fill=False, element='step')
        plt.xlabel('number of voxel')
        plt.ylabel('count of cs')
        plt.savefig(f'{f_name}/{subs}_hist_cs_sizes_larger10000.png')
        plt.close()
        sns.histplot(sizes_subs, fill=False, element='step', log_scale=10)
        plt.xlabel('number of voxel')
        plt.ylabel('count of cs')
        plt.savefig(f'{f_name}/{subs}_hist_cs_sizes_larger10000_log.png')
        plt.close()

    prev_excluded_sizs = cs_sizes[cs_sizes > 10**6]
    percent_excluded_number = 100 * len(prev_excluded_sizs) / len(cs_sizes)
    percent_excluded_sizes = 100 * np.sum(prev_excluded_sizs)/ np.sum(cs_sizes)

    log.info(f' Sizes of larger than 10**6 voxel were previously excluded. This were {percent_excluded_number:.2f} % of cs ids'
             f' with {percent_excluded_sizes:.2f} % of the total cs_size.')

    log.info('Step 2/2: Plot cs of astrocytes')
    astro_ids = np.load(f'cajal/scratch/users/arother/ds_glia_ids/astro_ids.npy')
    ssv_partners = cs_id_to_partner_ids_vec(cs_ids)
    suit_ct_inds = np.any(np.in1d(ssv_partners, astro_ids).reshape(len(ssv_partners), 2), axis=1)
    astro_cs_sizes = cs_sizes[suit_ct_inds]
    sns.histplot(astro_cs_sizes, fill=False, element='step')
    plt.xlabel('number of voxel')
    plt.ylabel('count of cs')
    plt.savefig(f'{f_name}/astro_ds_hist_cs_sizes.png')
    plt.close()
    sns.histplot(astro_cs_sizes, fill=False, element='step', log_scale=10)
    plt.xlabel('number of voxel')
    plt.ylabel('count of cs')
    plt.savefig(f'{f_name}/astro_ds_hist_cs_sizes_log.png')
    plt.close()

    log.info('Plotting cs sizes finished')

    