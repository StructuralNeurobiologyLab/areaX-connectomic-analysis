#shuffle analysis for GPe/i adn MSN
#plot MSN GPe/i ratio and shuffle synapses
#similar to JK code

if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    import os as os
    import pandas as pd
    import numpy as np
    from syconn.reps.segmentation import SegmentationDataset
    import matplotlib.pyplot as plt
    from scipy.stats import ranksums, kruskal
    from itertools import combinations
    import seaborn as sns

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"

    min_comp_len = 200
    syn_prob = 0.6
    min_syn_size = 0.1
    msn_ct = 2
    gpe_ct = 6
    gpi_ct = 7
    n_bootstrap = 1000
    f_name = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/230913_j0251v5_MSN_GP_ratio_shuffle_boots%i" % n_bootstrap
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('MSN conn GP ratio shuffle', log_dir=f_name + '/logs/')
    log.info(f' min comp len = {min_comp_len}, number of bootstraps = {n_bootstrap}')
    log.info("Analysis of GP ratio vs random starts")

    # load information about MSN groups and GP ratio
    kde = True
    f_name_saving1 = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/230911_j0251v5_MSN_GPratio_spine_density_mcl_%i_synprob_%.2f_kde%i_replot" % (
        min_comp_len, syn_prob, kde)
    log.info(f'Use morph parameters from {f_name_saving1}')
    msn_result_df = pd.read_csv(f'{f_name_saving1}/msn_spine_density_GPratio.csv', index_col=0)

    #load information about GP cells
    fontsize_jointplot = 10
    use_median = True
    f_name_saving2 = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/230911_j0251v5_GPe_i_myelin_mito_radius_mcl%i_newcolors_fs%i_med%i" % \
             (min_comp_len, fontsize_jointplot, use_median)
    log.info(f'Use morph parameters from {f_name_saving2}')
    gp_morph_df = pd.read_csv(f'{f_name_saving2}/GPe_GPi_params.csv', index_col=0)

    # get the synapses between MSN and GP from syn_sizes_df Dataframe
    log.info(f'Use syn sizes info from {f_name_saving1}')
    syn_sizes_df = pd.read_csv(f'{f_name_saving1}/syn_sizes_toGP.csv', index_col=0)

    log.info('Step 1/5: Calculate probabilites to use for shuffling')
    shuffle_cats = np.array(['observed', 'random', 'random with syn ratio', 'random with syn area ratio', 'random GP cell number ratio',
                 'random GP cell volume ratio'])
    syn_sizes_df['observed'] = syn_sizes_df['to celltype']
    #make shuffling analysis also with median synapse size
    syn_sizes_df['med syn size'] = np.zeros(len(syn_sizes_df)) + np.median(syn_sizes_df['syn sizes'])
    #get different probabilites
    #probability depending on synapse number
    syn_number_GPe = msn_result_df['syn number to GPe']
    syn_number_GPi = msn_result_df['syn number to GPi']
    sum_syn_number_GPe = syn_number_GPe.sum()
    sum_syn_number_GPi = syn_number_GPi.sum()
    syn_number_GPtotal = sum_syn_number_GPe + sum_syn_number_GPi
    p_syn_GPi_number = sum_syn_number_GPi / syn_number_GPtotal
    #probability depending on total synaptic area
    sum_syn_area_GPe = msn_result_df['syn size to GPe'].sum()
    sum_syn_area_GPi = msn_result_df['syn size to GPi'].sum()
    total_syn_area_GP = sum_syn_area_GPe + sum_syn_area_GPi
    p_syn_GPi_area = sum_syn_area_GPi/ total_syn_area_GP
    # probability depending on cell number
    number_GPe = len(gp_morph_df[gp_morph_df['celltype'] == 'GPe'])
    number_GPi = len(gp_morph_df[gp_morph_df['celltype'] == 'GPi'])
    GP_number = number_GPi + number_GPe
    p_GPi_number = number_GPi / GP_number
    #probability depending on cell volume
    volume_GPe = gp_morph_df['cell volume'][gp_morph_df['celltype'] == 'GPe'].sum()
    volume_GPi = gp_morph_df['cell volume'][gp_morph_df['celltype'] == 'GPi'].sum()
    GP_total_volume = volume_GPi + volume_GPe
    p_GPi_volume = volume_GPi / GP_total_volume
    bool_choice = [True, False]
    len_sizes = len(syn_sizes_df)

    log.info('Step 2/5: Shuffle syn sizes with different probabilities')
    #random
    rndm_inds = np.random.choice(bool_choice, len_sizes)
    syn_sizes_df.loc[rndm_inds, 'random'] = 'GPi'
    syn_sizes_df.loc[rndm_inds == False, 'random'] = 'GPe'
    #respect to synapse number
    rndm_inds = np.random.choice(bool_choice, len_sizes, p = [p_syn_GPi_number, 1 - p_syn_GPi_number])
    syn_sizes_df.loc[rndm_inds, 'random with syn ratio'] = 'GPi'
    syn_sizes_df.loc[rndm_inds == False, 'random with syn ratio'] = 'GPe'
    #respect to synapse syn area
    rndm_inds = np.random.choice(bool_choice, len_sizes, p=[p_syn_GPi_area, 1 - p_syn_GPi_area])
    syn_sizes_df.loc[rndm_inds, 'random with syn area ratio'] = 'GPi'
    syn_sizes_df.loc[rndm_inds == False, 'random with syn area ratio'] = 'GPe'
    #according to cell number
    rndm_inds = np.random.choice(bool_choice, len_sizes, p=[p_GPi_number, 1 - p_GPi_number])
    syn_sizes_df.loc[rndm_inds, 'random GP cell number ratio'] = 'GPi'
    syn_sizes_df.loc[rndm_inds == False, 'random GP cell number ratio'] = 'GPe'
    # according to cell volume
    rndm_inds = np.random.choice(bool_choice, len_sizes, p=[p_GPi_volume, 1 - p_GPi_volume])
    syn_sizes_df.loc[rndm_inds, 'random GP cell volume ratio'] = 'GPi'
    syn_sizes_df.loc[rndm_inds == False, 'random GP cell volume ratio'] = 'GPe'

    log.info('Step 3/6: Plot size distributions for observed and shuffled data')
    gp_palette = {'GPe': '#592A87', 'GPi': '#2AC644'}
    for sc in shuffle_cats:
        sns.histplot(x='syn sizes', data=syn_sizes_df, hue=sc, palette=gp_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent')
        plt.ylabel('% of synapses')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.savefig(f'{f_name}/synsizes_to_GP_{sc}_hist_perc.png')
        plt.savefig(f'{f_name}/synsizes_to_GP_{sc}_hist_perc.svg')
        plt.title(sc)
        plt.close()
        sns.histplot(x='syn sizes', data=syn_sizes_df, hue=sc, palette=gp_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, log_scale=True, stat='percent')
        plt.ylabel('% of synapses')
        plt.xlabel('synaptic mesh area [µm²]')
        plt.title(sc)
        plt.savefig(f'{f_name}/synsizes_to_GP_{sc}_hist_log_perc.png')
        plt.savefig(f'{f_name}/synsizes_to_GP_{sc}_hist_log_perc.svg')
        plt.close()

    log.info('Step 4/5: Calculate per MSN GP syn area ratio')
    #calculate ratio for actual synapses but also for synapses of univariante size
    msn_ids = np.sort(np.unique(syn_sizes_df['cellid']))
    num_msn = len(msn_ids)
    shuffle_df = pd.DataFrame(columns=['cellid', 'GP ratio sum syn area', 'shuffle category'], index = range(num_msn * len(shuffle_cats)))

    for i, sc in enumerate(shuffle_cats):
        gpe_syn_info = syn_sizes_df[syn_sizes_df[sc] == 'GPe']
        gpi_syn_info = syn_sizes_df[syn_sizes_df[sc] == 'GPi']
        rem_columns = shuffle_cats[shuffle_cats != sc]
        gpi_syn_info = gpi_syn_info.drop(columns=rem_columns)
        gpe_syn_info = gpe_syn_info.drop(columns=rem_columns)
        gpe_msn_inds, unique_msngpe_ids = pd.factorize(gpe_syn_info['cellid'])
        gpe_syn_msn_sizes = np.bincount(gpe_msn_inds, gpe_syn_info['syn sizes'])
        gpe_uni_syn_sizes = np.bincount(gpe_msn_inds, gpe_syn_info['med syn size'])
        gpe_argsort = np.argsort(unique_msngpe_ids)
        sorted_msngpe_ids = unique_msngpe_ids[gpe_argsort]
        sorted_gpe_syn_sizes = gpe_syn_msn_sizes[gpe_argsort]
        sorted_gpe_uni_sizes = gpe_uni_syn_sizes[gpe_argsort]
        gpi_msn_inds, unique_msngpi_ids = pd.factorize(gpi_syn_info['cellid'])
        gpi_syn_msn_sizes = np.bincount(gpi_msn_inds, gpi_syn_info['syn sizes'])
        gpi_uni_syn_sizes = np.bincount(gpi_msn_inds, gpi_syn_info['med syn size'])
        gpi_argsort = np.argsort(unique_msngpi_ids)
        sorted_msngpi_ids = unique_msngpi_ids[gpi_argsort]
        sorted_gpi_syn_sizes = gpi_syn_msn_sizes[gpi_argsort]
        sorted_gpi_uni_sizes = gpi_uni_syn_sizes[gpi_argsort]
        gpe_syn_area = np.zeros(len(msn_ids))
        gpe_uni_area = np.zeros(len(msn_ids))
        gpe_msn_inds = np.in1d(msn_ids, sorted_msngpe_ids)
        gpe_syn_area[gpe_msn_inds] = sorted_gpe_syn_sizes
        gpe_uni_area[gpe_msn_inds] = sorted_gpe_uni_sizes
        gpi_syn_area = np.zeros(len(msn_ids))
        gpi_uni_area = np.zeros(len(msn_ids))
        gpi_msn_inds = np.in1d(msn_ids, sorted_msngpi_ids)
        gpi_syn_area[gpi_msn_inds] = sorted_gpi_syn_sizes
        gpi_uni_area[gpi_msn_inds] = sorted_gpi_uni_sizes
        GP_syn_area_ratio = gpi_syn_area / (gpe_syn_area + gpi_syn_area)
        GP_uni_area_ratio = gpi_uni_area/(gpe_uni_area + gpi_uni_area)
        shuffle_df.loc[i * num_msn: (i + 1) * num_msn - 1, 'cellid'] = msn_ids
        shuffle_df.loc[i * num_msn: (i + 1) * num_msn - 1, 'GP ratio sum syn area'] = GP_syn_area_ratio
        shuffle_df.loc[i * num_msn: (i + 1) * num_msn - 1, 'shuffle category'] = sc
        shuffle_df.loc[i * num_msn: (i + 1) * num_msn - 1, 'GP ratio uni syn area'] = GP_uni_area_ratio

    shuffle_df.to_csv(f'{f_name}/shuffle_results.csv')

    log.info('Step 5/5: Plot results')
    #plot results for different categories
    cat_palette = {'observed': "#EAAE34", 'random': '#D9D9D9', 'random with syn ratio':'#A6A6A6',
        'random with syn area ratio': '#595959', 'random GP cell number ratio': '#262626',
                 'random GP cell volume ratio': '#0D0D0D'}
    sns.histplot(x='GP ratio sum syn area', data=shuffle_df, hue='shuffle category', palette=cat_palette, common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, stat='percent')
    plt.ylabel('% of cells')
    plt.xlabel('GPi/(GPe + GPi) syn area')
    plt.title('GP area ratio with real synapse sizes')
    plt.savefig(f'{f_name}/GP_area_ratio_cats_hist_perc.png')
    plt.savefig(f'{f_name}/GP_area_ratio_cats_hist_perc.svg')
    plt.close()
    sns.histplot(x='GP ratio uni syn area', data=shuffle_df, hue='shuffle category', palette=cat_palette,
                 common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, stat='percent')
    plt.ylabel('% of cells')
    plt.xlabel('GPi/(GPe + GPi) syn area with median syn size')
    plt.title('GP area ratio with univariat synapse size')
    plt.savefig(f'{f_name}/GP_uni_ratio_cats_hist_perc.png')
    plt.savefig(f'{f_name}/GP_uni_ratio_cats_hist_perc.svg')
    plt.close()

    log.info('Analysis done')
















