#shuffle analysis for GPe/i adn MSN
#plot MSN GPe/i ratio and shuffle synapses
#similar to JK code

if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    import os as os
    import pandas as pd
    import numpy as np
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
    f_name = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/230911_j0251v5_MSN_GP_ratio_shuffle_boots%i" % n_bootstrap
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('MSN conn GP ratio shuffle', log_dir=f_name + '/logs/')
    log.info(f' min comp len = {min_comp_len}, number of bootstraps = {n_bootstrap}')
    log.info("Analysis of GP ratio vs random starts")

    # load information about MSN groups and GP ratio
    kde = True
    f_name_saving1 = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/230831_j0251v5_MSN_GPratio_spine_density_mcl_%i_synprob_%.2f_kde%i_replot" % (
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

    log.info('Step 1/6: Create new dataframe with real data and bootstrapping results')
    len_msn_df = len(msn_result_df)
    bt_cats = ['observed', 'random', 'random with syn ratio', 'random GPi/GPe cell number ratio',
                 'random GPi/GPe cell volume ratio']
    #get different probabilites
    #probability depending on synapse number
    syn_number_GPe = msn_result_df['syn number to GPe']
    syn_number_GPi = msn_result_df['syn number to GPi']
    sum_syn_number_GPe = syn_number_GPe.sum()
    sum_syn_number_GPi = syn_number_GPi.sum()
    syn_number_GPtotal = sum_syn_number_GPe + sum_syn_number_GPi
    p_syn_GPe_number = sum_syn_number_GPe / syn_number_GPtotal
    p_syn_GPi_number = sum_syn_number_GPi / syn_number_GPtotal
    #probability depending on total synaptic area

    # probability depending on cell number
    number_GPe = len(gp_morph_df[gp_morph_df['celltype'] == 'GPe'])
    number_GPi = len(gp_morph_df[gp_morph_df['celltype'] == 'GPi'])
    GP_number = number_GPi + number_GPe
    p_GPe_number = number_GPe / GP_number
    p_GPi_number = number_GPi / GP_number
    #probability depending on cell volume
    volume_GPe = gp_morph_df['cell volume'][gp_morph_df['celltype'] == 'GPe'].sum()
    volume_GPi = gp_morph_df['cell volume'][gp_morph_df['celltype'] == 'GPi'].sum()
    GP_total_volume = volume_GPi + volume_GPe
    p_GPe_volume = volume_GPe / GP_total_volume
    p_GPi_volume = volume_GPi / GP_total_volume




