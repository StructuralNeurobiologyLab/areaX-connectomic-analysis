#shuffle analysis for GPe/i adn MSN
#plot MSN GPe/i ratio and shuffle synapses
#update from GP_ratio_shuffle
#no toin coss but get synapses with cellid and if to GPe or GPi
#then shuffle existing synapses around (not if to GPe or GPi but which MSN id they belong to
#run for several iterations
#plot then with mean and confidence interval

if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    import os as os
    import pandas as pd
    import numpy as np
    from syconn.reps.segmentation import SegmentationDataset
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_1samp
    import seaborn as sns
    from tqdm import tqdm

    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    ct_dict = analysis_params.ct_dict(with_glia=False)
    global_params.wd = analysis_params.working_dir()

    min_comp_len = 200
    syn_prob = 0.6
    min_syn_size = 0.1
    msn_ct = 3
    gpe_ct = 6
    gpi_ct = 7
    n_it = 10
    hist_bins = 30
    fontsize = 20
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/250311_j0251{version}_MSN_GP_ratio_shuffle2_it{n_it}_fs{fontsize}_hb{hist_bins}"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('MSN conn GP ratio shuffle', log_dir=f_name + '/logs/')
    log.info(f' min comp len = {min_comp_len}, number of iterations = {n_it}')
    log.info("Analysis of GP ratio vs random starts")
    # load information about MSN groups and GP ratio
    kde = True
    f_name_saving1 = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/241025_j0251v6_MSN_3_GPe_GPiratio_spine_density_mcl_200_synprob_0.60_kde1_f20"
    log.info(f'Use morph parameters from {f_name_saving1}')
    msn_result_df = pd.read_csv(f'{f_name_saving1}/MSN_morph_GPe_GPiratio.csv', index_col=0)
    np.random.seed(42)

    # get the synapses between MSN and GP from syn_sizes_df Dataframe
    log.info(f'Use syn sizes info from {f_name_saving1}')
    syn_sizes_df = pd.read_csv(f'{f_name_saving1}/syn_sizes_toGPe_GPi.csv', index_col=0)


    log.info('Step 1/5: Calculate fraction of GPi synapses and synaptic area')
    #get ratio of GPe to GPi synapses in number and area
    syn_number_GPe = msn_result_df['syn number to GPe']
    syn_number_GPi = msn_result_df['syn number to GPi']
    sum_syn_number_GPe = syn_number_GPe.sum()
    sum_syn_number_GPi = syn_number_GPi.sum()
    syn_number_GPtotal = sum_syn_number_GPe + sum_syn_number_GPi
    fraction_syn_GPi_number = sum_syn_number_GPi / syn_number_GPtotal
    log.info(f'{syn_number_GPtotal} synapses go from MSN to either GPe or GPi. {fraction_syn_GPi_number:.2f} of those go to GPi.')
    #probability depending on total synaptic area
    sum_syn_area_GPe = msn_result_df['syn size to GPe'].sum()
    sum_syn_area_GPi = msn_result_df['syn size to GPi'].sum()
    total_syn_area_GP = sum_syn_area_GPe + sum_syn_area_GPi
    fraction_syn_GPi_area = sum_syn_area_GPi/ total_syn_area_GP
    log.info(
        f'{total_syn_area_GP} synaptic area goes from MSN to either GPe or GPi. {fraction_syn_GPi_area:.2f} of those go to GPi.')


    log.info('Step 2/3: Shuffle synapses between MSN cells')

    msn_ids = np.unique(syn_sizes_df['cellid'])
    log.info(f'{len(msn_ids)} project to either GPe or GPi')
    GP_ratio_per_cell_df = pd.DataFrame(index = msn_ids)
    GP_number_histogram_df = pd.DataFrame(index = range(hist_bins))
    GP_area_histogram_df = pd.DataFrame(index = range(hist_bins))
    #also get number that have 0.9 in syn area or higher for each it
    GP_area_num_sel = np.zeros(n_it + 1)
    for ni in tqdm(range(n_it + 1)):
        #shuffle msn cellids to get different allocation of cellids to synapses
        #for first iteration take observed values
        if ni > 0:
            syn_sizes_df[f'shuffle ids {ni}'] = np.random.permutation(syn_sizes_df['cellid'].values)
        #get GP ratio in numbers and synaptic area for each MSN cell
        gpe_info = syn_sizes_df[syn_sizes_df['to celltype'] == 'GPe']
        gpi_info = syn_sizes_df[syn_sizes_df['to celltype'] == 'GPi']
        if ni == 0:
            column_name = 'cellid'
            ratio_column_name = 'observed'
        else:
            column_name = f'shuffle ids {ni}'
            ratio_column_name = f'shuffle {ni}'
        gpe_cellid_groups = gpe_info.groupby(column_name)
        gpi_cellid_groups = gpi_info.groupby(column_name)
        #get number of synapses with gpe
        gpe_percell_number = gpe_cellid_groups.size()
        gpe_percell_area = gpe_cellid_groups['syn sizes'].sum()
        GP_ratio_per_cell_df.loc[gpe_percell_number.index, f'{ratio_column_name} GPe number'] = gpe_percell_number.values
        GP_ratio_per_cell_df.loc[
            gpe_percell_area.index, f'{ratio_column_name} GPe area'] = gpe_percell_area.values
        # get number of synapses with gpi
        gpi_percell_number = gpi_cellid_groups.size()
        gpi_percell_area = gpi_cellid_groups['syn sizes'].sum()
        GP_ratio_per_cell_df.loc[
            gpi_percell_number.index, f'{ratio_column_name} GPi number'] = gpi_percell_number.values
        GP_ratio_per_cell_df.loc[
            gpi_percell_area.index, f'{ratio_column_name} GPi area'] = gpi_percell_area.values
        #calculate GP ratio for numbers and areas
        GP_ratio_per_cell_df = GP_ratio_per_cell_df.fillna(0)
        GP_ratio_per_cell_df[f'{ratio_column_name} number ratio'] = np.array(GP_ratio_per_cell_df[f'{ratio_column_name} GPi number']) / \
                                                                    (np.array(GP_ratio_per_cell_df[f'{ratio_column_name} GPi number']) + np.array(GP_ratio_per_cell_df[f'{ratio_column_name} GPe number']))
        GP_ratio_per_cell_df[f'{ratio_column_name} area ratio'] = np.array(
            GP_ratio_per_cell_df[f'{ratio_column_name} GPi area']) / \
                                                                    (np.array(GP_ratio_per_cell_df[
                                                                                  f'{ratio_column_name} GPi area']) + np.array(
                                                                        GP_ratio_per_cell_df[
                                                                            f'{ratio_column_name} GPe area']))
        #get number of MSN cells with GPi area ratio of 0.9 or higher
        GPi_sel_num = len(GP_ratio_per_cell_df[GP_ratio_per_cell_df[f'{ratio_column_name} area ratio'] >= 0.9])
        GP_area_num_sel[ni] = GPi_sel_num
        #get histogram for GP ratio numbers
        counts_numbers, bin_edges = np.histogram(GP_ratio_per_cell_df[f'{ratio_column_name} number ratio'], bins=hist_bins)
        counts_areas, bin_edges = np.histogram(GP_ratio_per_cell_df[f'{ratio_column_name} area ratio'], bins=hist_bins)
        GP_number_histogram_df[ratio_column_name] = counts_numbers
        GP_area_histogram_df[ratio_column_name] = counts_areas

    syn_sizes_df.to_csv(f'{f_name}/syn_sizes_shuffling.csv')
    GP_ratio_per_cell_df.to_csv(f'{f_name}/GP_ratio_percell_shuffling.csv')
    GP_number_histogram_df.to_csv(f'{f_name}/GP_ratio_number_hist_counts.csv')
    GP_area_histogram_df.to_csv(f'{f_name}/GP_ratio_area_hist_counts.csv')
    #get GPi selectivity number for observed data
    log.info(f'{GP_area_num_sel[0]} MSN cells have a GP ratio of 0.9 or higher ({100 * GP_area_num_sel[0]/ len(msn_ids):.2f} %).')
    GP_area_num_sel = GP_area_num_sel[1:] #remove first element as it is observed value
    np.save(f'{f_name}/GPi_sel_msn_nums_shuffled.npy', GP_area_num_sel)


    log.info('Plot for shuffled mean, 90% CI for GP number and GP area ratio')
    mean_GPi_sel_num = np.mean(GP_area_num_sel)
    std_GPi_sel_num = np.std(GP_area_num_sel)
    log.info(f'With shuffling for {n_it} iterations, the number of MSN with GP area ratio >= 0.9 had a mean of {mean_GPi_sel_num} with std {std_GPi_sel_num}'
             f' ({100 * mean_GPi_sel_num/ len(msn_ids):.2f} +- {100*std_GPi_sel_num/len(msn_ids):.2f} %).')

    #plot number
    observed_data_number = GP_number_histogram_df['observed']
    shuffled_data_number = GP_number_histogram_df.drop(columns=['observed'])
    # Compute mean and confidence interval (95%) for shuffled data
    shuffle_mean_number = shuffled_data_number.mean(axis=1)
    shuffle_lower_number = shuffled_data_number.quantile(0.025, axis=1)  # 2.5th percentile
    shuffle_upper_number = shuffled_data_number.quantile(0.975, axis=1)  # 97.5th percentile
    plt.step(bin_edges[:-1], observed_data_number, color="#EAAE34", linewidth=3, where='mid', label='observed')
    plt.step(bin_edges[:-1], shuffle_mean_number, color='black', linewidth=3, where='mid', label = 'shuffle mean', alpha = 0.8)
    plt.fill_between(bin_edges[:-1], shuffle_lower_number, shuffle_upper_number, color = 'black', label = '95% CI shuffle', alpha = 0.3, step = 'mid')
    plt.ylabel('number of cells', fontsize=fontsize)
    plt.xlabel('GPi/(GPe + GPi) syn number', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.title(f'GP number ratio with {n_it} iterations')
    plt.savefig(f'{f_name}/GP_ratio_number_hist.png')
    plt.savefig(f'{f_name}/GP_ratio_number_hist.svg')
    plt.close()
    #get number percentages for all values
    total_count = np.sum(observed_data_number)
    obs_count_numbers = (observed_data_number/ total_count) * 100
    sh_mean_count_number = (shuffle_mean_number/ total_count) * 100
    sh_lower_count_number = (shuffle_lower_number / total_count) * 100
    sh_upper_count_number = (shuffle_upper_number / total_count) * 100
    plt.step(bin_edges[:-1], obs_count_numbers, color="#EAAE34", linewidth=3, where='mid', label='observed')
    plt.step(bin_edges[:-1], sh_mean_count_number, color='black', linewidth=3, where='mid', label='shuffle mean',
             alpha=0.8)
    plt.fill_between(bin_edges[:-1], sh_lower_count_number, sh_upper_count_number, color='black', label='95% CI shuffle',
                     alpha=0.3, step='mid')
    plt.ylabel('% of cells', fontsize=fontsize)
    plt.xlabel('GPi/(GPe + GPi) syn number', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.title(f'GP number ratio with {n_it} iterations')
    plt.savefig(f'{f_name}/GP_area_ratio_number_hist_perc.png')
    plt.savefig(f'{f_name}/GP_area_ratio_number_hist_perc.svg')
    plt.close()
    #plot for area
    observed_data_area = GP_area_histogram_df['observed']
    shuffled_data_area = GP_area_histogram_df.drop(columns=['observed'])
    # Compute mean and confidence interval (95%) for shuffled data
    shuffle_mean_area = shuffled_data_area.mean(axis=1)
    shuffle_lower_area = shuffled_data_area.quantile(0.025, axis=1)  # 2.5th percentile
    shuffle_upper_area = shuffled_data_area.quantile(0.975, axis=1)  # 97.5th percentile
    plt.step(bin_edges[:-1], observed_data_area, color="#EAAE34", linewidth=3, where='mid', label='observed')
    plt.step(bin_edges[:-1], shuffle_mean_area, color='black', linewidth=3, where='mid', label='shuffle mean',
             alpha=0.8)
    plt.fill_between(bin_edges[:-1], shuffle_lower_area, shuffle_upper_area, color='black', label='95% CI shuffle',
                     alpha=0.3, step='mid')
    plt.ylabel('number of cells', fontsize=fontsize)
    plt.xlabel('GPi/(GPe + GPi) syn area', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.title(f'GP area ratio with {n_it} iterations')
    plt.savefig(f'{f_name}/GP_ratio_area_hist.png')
    plt.savefig(f'{f_name}/GP_ratio_area_hist.svg')
    plt.close()
    # get number percentages for all values
    total_count = np.sum(observed_data_area)
    obs_count_area = (observed_data_area / total_count) * 100
    sh_mean_count_area = (shuffle_mean_area / total_count) * 100
    sh_lower_count_area = (shuffle_lower_area / total_count) * 100
    sh_upper_count_area = (shuffle_upper_area / total_count) * 100
    plt.step(bin_edges[:-1], obs_count_area, color="#EAAE34", linewidth=3, where='mid', label='observed')
    plt.step(bin_edges[:-1], sh_mean_count_area, color='black', linewidth=3, where='mid', label='shuffle mean',
             alpha=0.8)
    plt.fill_between(bin_edges[:-1], sh_lower_count_area, sh_upper_count_area, color='black', label='95% CI shuffle',
                     alpha=0.3, step='mid')
    plt.ylabel('% of cells', fontsize=fontsize)
    plt.xlabel('GPi/(GPe + GPi) syn area', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.title(f'GP number area with {n_it} iterations')
    plt.savefig(f'{f_name}/GP_area_ratio_area_hist_perc.png')
    plt.savefig(f'{f_name}/GP_area_ratio_area_hist_perc.svg')
    plt.close()
    log.info('Analysis done')