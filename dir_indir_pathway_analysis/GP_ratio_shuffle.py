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
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_1samp
    import seaborn as sns
    from tqdm import tqdm

    #global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"

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
    n_it = 100
    n_plot_it = 3
    fontsize = 20
    binary_syns = True
    if binary_syns:
        f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/241002_j0251v5_MSN_GP_ratio_shuffle_binary_it{n_it}_fs{fontsize}"
    else:
        f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/241002_j0251v5_MSN_GP_ratio_shuffle_it{n_it}_fs{fontsize}"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('MSN conn GP ratio shuffle', log_dir=f_name + '/logs/')
    log.info(f' min comp len = {min_comp_len}, number of iterations = {n_it}')
    log.info("Analysis of GP ratio vs random starts")
    if binary_syns:
        log.info('Synapse sizes are all set to 1')
    # load information about MSN groups and GP ratio
    kde = True
    f_name_saving1 = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/240229_j0251v6_%s_GPratio_spine_density_mcl_%i_synprob_%.2f_kde%i_f20" % (
        ct_dict[msn_ct], min_comp_len, syn_prob, kde)
    log.info(f'Use morph parameters from {f_name_saving1}')
    msn_result_df = pd.read_csv(f'{f_name_saving1}/MSN_morph_GPratio.csv', index_col=0)
    np.random.seed(42)

    #load information about GP cells
    fontsize_jointplot = 20
    use_median = True
    f_name_saving2 = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/240220_j0251v6_GPe_i_myelin_mito_radius_mcl%i_fs%i_med%i" % \
             (min_comp_len, fontsize_jointplot, use_median)
    log.info(f'Use morph parameters from {f_name_saving2}')
    gp_morph_df = pd.read_csv(f'{f_name_saving2}/GPe_GPi_params.csv', index_col=0)

    # get the synapses between MSN and GP from syn_sizes_df Dataframe
    log.info(f'Use syn sizes info from {f_name_saving1}')
    syn_sizes_df = pd.read_csv(f'{f_name_saving1}/syn_sizes_toGP.csv', index_col=0)
    if binary_syns:
        syn_sizes_df['syn sizes'] = 1

    log.info('Step 1/5: Calculate probabilites to use for shuffling')
    shuffle_cats = np.array(['observed', 'random', 'random with syn ratio', 'random with syn area ratio', 'random GP cell number ratio',
                 'random GP cell volume ratio'])
    shuffle_cats_its = []
    for i in range(n_plot_it):
        shuffle_cats_its.append(np.array([f'{sc} {i}' for sc in shuffle_cats]))
    shuffle_cats_its = np.concatenate(shuffle_cats_its)
    #shuffle_cats = np.array(['observed', 'random', 'random with syn ratio'])
    for i in range(n_plot_it):
        syn_sizes_df[f'observed {i}'] = syn_sizes_df['to celltype']
    #make shuffling analysis also with median synapse size
    syn_sizes_df['med syn size'] = np.zeros(len(syn_sizes_df)) + np.median(syn_sizes_df['syn sizes'])
    #get different probabilites
    #probability depending on synapse number
    prob_df = pd.DataFrame(columns = ['to GPe', 'to GPi'], index = ['syn number ratio', 'syn size ratio', 'cell number ratio', 'cell volume ratio'])
    syn_number_GPe = msn_result_df['syn number to GPe']
    syn_number_GPi = msn_result_df['syn number to GPi']
    sum_syn_number_GPe = syn_number_GPe.sum()
    sum_syn_number_GPi = syn_number_GPi.sum()
    syn_number_GPtotal = sum_syn_number_GPe + sum_syn_number_GPi
    p_syn_GPi_number = sum_syn_number_GPi / syn_number_GPtotal
    prob_df.loc['syn number ratio', 'to GPi'] = p_syn_GPi_number
    prob_df.loc['syn number ratio', 'to GPe'] = sum_syn_number_GPe/ syn_number_GPtotal
    #probability depending on total synaptic area
    sum_syn_area_GPe = msn_result_df['syn size to GPe'].sum()
    sum_syn_area_GPi = msn_result_df['syn size to GPi'].sum()
    total_syn_area_GP = sum_syn_area_GPe + sum_syn_area_GPi
    p_syn_GPi_area = sum_syn_area_GPi/ total_syn_area_GP
    prob_df.loc['syn size ratio', 'to GPi'] = p_syn_GPi_area
    prob_df.loc['syn size ratio', 'to GPe'] = sum_syn_area_GPe / total_syn_area_GP
    # probability depending on cell number
    number_GPe = len(gp_morph_df[gp_morph_df['celltype'] == 'GPe'])
    number_GPi = len(gp_morph_df[gp_morph_df['celltype'] == 'GPi'])
    GP_number = number_GPi + number_GPe
    p_GPi_number = number_GPi / GP_number
    prob_df.loc['cell number ratio', 'to GPi'] = p_GPi_number
    prob_df.loc['cell number ratio', 'to GPe'] = number_GPe / GP_number
    #probability depending on cell volume
    volume_GPe = gp_morph_df['cell volume'][gp_morph_df['celltype'] == 'GPe'].sum()
    volume_GPi = gp_morph_df['cell volume'][gp_morph_df['celltype'] == 'GPi'].sum()
    GP_total_volume = volume_GPi + volume_GPe
    p_GPi_volume = volume_GPi / GP_total_volume
    prob_df.loc['cell volume ratio', 'to GPi'] = p_GPi_volume
    prob_df.loc['cell volume ratio', 'to GPe'] = volume_GPe / GP_total_volume
    num_cats = len(shuffle_cats)
    prob_dict = {shuffle_cats[i + 2]: prob_df.loc[ind, 'to GPi'] for i, ind in enumerate(prob_df.index)}
    prob_dict['random'] = 0.5
    #raise ValueError
    prob_df.to_csv(f'{f_name}/prob_ratios_for_shuffling.csv')

    bool_choice = [True, False]
    len_sizes = len(syn_sizes_df)

    log.info('Step 2/5: Shuffle syn sizes with different probabilities')

    mean_df = pd.DataFrame(columns = ['mean', 'mean(abs(GP ratio - 0.5))', 'shuffle category'], index = range(n_it * num_cats))
    temp_df = pd.DataFrame(index = range(len(syn_sizes_df)))
    temp_df['syn sizes'] = syn_sizes_df['syn sizes']
    temp_df['cellid'] = syn_sizes_df['cellid']
    msn_ids = np.sort(np.unique(syn_sizes_df['cellid']))
    num_msn = len(msn_ids)
    shuffle_df = pd.DataFrame(columns=['cellid', 'GP ratio sum syn area', 'shuffle category'],
                              index=range(num_msn * num_cats * n_plot_it))

    log.info('Iterate and get GP ratio randomly with different probs')
    for ni in tqdm(range(n_it)):
        for i, sc in enumerate(shuffle_cats):
            #only get observed once
            if 'observed' in sc and ni > 0:
                continue
            #get probability for each category
            if not 'observed' in sc:
                sc_prob_gpi = prob_dict[sc]
                rndm_inds = np.random.choice(bool_choice, len_sizes, p = [sc_prob_gpi, 1 - sc_prob_gpi])
                temp_df.loc[rndm_inds, sc] = 'GPi'
                temp_df.loc[rndm_inds == False, sc] = 'GPe'
                if ni < n_plot_it:
                    syn_sizes_df.loc[rndm_inds, f'{sc} {ni}'] = 'GPi'
                    syn_sizes_df.loc[rndm_inds == False, f'{sc} {ni}'] = 'GPe'
            else:
                temp_df[sc] = syn_sizes_df[f'{sc} {ni}']
            #directly calculate ratio here
            gpe_syn_info = temp_df[temp_df[sc] == 'GPe']
            gpi_syn_info = temp_df[temp_df[sc] == 'GPi']
            gpe_msn_inds, unique_msngpe_ids = pd.factorize(gpe_syn_info['cellid'])
            gpe_syn_msn_sizes = np.bincount(gpe_msn_inds, gpe_syn_info['syn sizes'])
            gpe_argsort = np.argsort(unique_msngpe_ids)
            sorted_msngpe_ids = unique_msngpe_ids[gpe_argsort]
            sorted_gpe_syn_sizes = gpe_syn_msn_sizes[gpe_argsort]
            gpi_msn_inds, unique_msngpi_ids = pd.factorize(gpi_syn_info['cellid'])
            gpi_syn_msn_sizes = np.bincount(gpi_msn_inds, gpi_syn_info['syn sizes'])
            gpi_argsort = np.argsort(unique_msngpi_ids)
            sorted_msngpi_ids = unique_msngpi_ids[gpi_argsort]
            sorted_gpi_syn_sizes = gpi_syn_msn_sizes[gpi_argsort]
            gpe_syn_area = np.zeros(len(msn_ids))
            gpe_msn_inds = np.in1d(msn_ids, sorted_msngpe_ids)
            gpe_syn_area[gpe_msn_inds] = sorted_gpe_syn_sizes
            gpi_syn_area = np.zeros(len(msn_ids))
            gpi_uni_area = np.zeros(len(msn_ids))
            gpi_msn_inds = np.in1d(msn_ids, sorted_msngpi_ids)
            gpi_syn_area[gpi_msn_inds] = sorted_gpi_syn_sizes
            GP_syn_area_ratio = gpi_syn_area / (gpe_syn_area + gpi_syn_area)
            if ni < n_plot_it:
                shuffle_df.loc[(ni * n_plot_it + i) * num_msn: (ni * n_plot_it + i+ 1) * num_msn - 1, 'cellid'] = msn_ids
                shuffle_df.loc[(ni * n_plot_it + i) * num_msn: (ni * n_plot_it + i + 1) * num_msn - 1,
                'GP ratio sum syn area'] = GP_syn_area_ratio
                shuffle_df.loc[(ni * n_plot_it + i) * num_msn: (ni * n_plot_it + i + 1) * num_msn - 1,
                'shuffle category'] = sc
                shuffle_df.loc[(ni * n_plot_it + i) * num_msn: (ni * n_plot_it + i + 1) * num_msn - 1,
                'iteration'] = ni
            mean_df.loc[(ni * num_cats)+ i, 'shuffle category'] = sc
            mean_df.loc[(ni * num_cats) + i, 'mean'] = np.mean(GP_syn_area_ratio)
            mean_df.loc[(ni * num_cats) + i, 'mean(abs(GP ratio - 0.5))'] = np.mean(np.abs(GP_syn_area_ratio - 0.5))

    log.info('Iterations done')

    #remove nan values from emtpy observed rows which were calculated only once
    shuffle_df = shuffle_df.dropna()
    shuffle_df.to_csv(f'{f_name}/shuffle_results.csv')
    mean_df = mean_df.dropna()
    mean_df = mean_df.reset_index(drop= True)
    mean_df.to_csv(f'{f_name}/mean_df_{n_it}.csv')


    log.info('Step 3/5: Plot size distributions for observed and shuffled data')
    gp_palette = {'GPe': '#592A87', 'GPi': '#2AC644'}
    for sc in shuffle_cats:
        for i in range(n_plot_it):
            sns.histplot(x='syn sizes', data=syn_sizes_df, hue=f'{sc} {i}', palette=gp_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True, stat='percent')
            plt.ylabel('% of synapses', fontsize = fontsize)
            plt.xlabel('synaptic mesh area [µm²]', fontsize = fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.savefig(f'{f_name}/synsizes_to_GP_{sc}_{i}_hist_perc.png')
            plt.savefig(f'{f_name}/synsizes_to_GP_{sc}_{i}_hist_perc.svg')
            plt.title(sc)
            plt.close()
            sns.histplot(x='syn sizes', data=syn_sizes_df, hue=f'{sc} {i}', palette=gp_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True, log_scale=True, stat='percent')
            plt.ylabel('% of synapses', fontsize = fontsize)
            plt.xlabel('synaptic mesh area [µm²]', fontsize = fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.title(sc)
            plt.savefig(f'{f_name}/synsizes_to_GP_{sc}_{i}_hist_log_perc.png')
            plt.savefig(f'{f_name}/synsizes_to_GP_{sc}_{i}_hist_log_perc.svg')
            plt.close()

    log.info('Step 4/5: Plot results')
    #plot results for different categories

    cat_palette = {'observed': "#EAAE34", 'random': '#D9D9D9',
        'random with syn ratio': '#595959', 'random GP cell number ratio': '#262626',
                 'random GP cell volume ratio': '#0D0D0D', 'random with syn area ratio':'#A6A6A6'}
    it_palette = {1: '#A6A6A6', 2:'#595959', 3:'#0D0D0D'}
    '''
    cat_palette = {'observed': "#EAAE34", 'random': '#D9D9D9',
                   'random with syn ratio': '#0D0D0D'}
    '''
    # plot for each iteration all the categories
    for i in range(n_plot_it):
        it_shuffle_df = shuffle_df[shuffle_df['iteration'] == i]
        sns.histplot(x='GP ratio sum syn area', data=it_shuffle_df, hue='shuffle category', palette=cat_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent')
        plt.ylabel('% of cells', fontsize = fontsize)
        plt.xlabel('GPi/(GPe + GPi) syn area', fontsize = fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.title(f'GP area ratio with real synapse sizes it {i}')
        plt.savefig(f'{f_name}/GP_area_ratio_cats_{i}_hist_perc.png')
        plt.savefig(f'{f_name}/GP_area_ratio_cats_{i}_hist_perc.svg')
        plt.close()
        #plot only observed, syn_ratio and random
        rndm_inds = it_shuffle_df['shuffle category'] == 'random'
        obs_inds = it_shuffle_df['shuffle category'] == 'observed'
        syn_ratio_inds = it_shuffle_df['shuffle category'] == 'random with syn ratio'
        spec_inds = np.any([rndm_inds, obs_inds, syn_ratio_inds], axis = 0)
        it_shuffle_df_spec = it_shuffle_df[spec_inds]
        sns.histplot(x='GP ratio sum syn area', data=it_shuffle_df_spec, hue='shuffle category', palette=cat_palette,
                     common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent')
        plt.ylabel('% of cells', fontsize = fontsize)
        plt.xlabel('GPi/(GPe + GPi) syn area', fontsize = fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.title(f'GP area ratio with real synapse sizes it {i}')
        plt.savefig(f'{f_name}/GP_area_ratio_cats_{i}_synratio_hist_perc.png')
        plt.savefig(f'{f_name}/GP_area_ratio_cats_{i}_synratio_hist_perc.svg')
        plt.close()

    #plot for each category all the runs
    for sc in shuffle_cats:
        if sc == 'observed':
            continue
        sc_shuffle_df = shuffle_df[shuffle_df['shuffle category'] == sc]
        sns.histplot(x='GP ratio sum syn area', data=sc_shuffle_df, hue='iteration', palette=it_palette,
                     common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent')
        plt.ylabel('% of cells', fontsize = fontsize)
        plt.xlabel('GPi/(GPe + GPi) syn area', fontsize = fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.title(f'GP area ratio with real synapse sizes {sc}')
        plt.savefig(f'{f_name}/GP_area_ratio_{sc}_its_hist_perc.png')
        plt.savefig(f'{f_name}/GP_area_ratio_{sc}_its_hist_perc.svg')
        plt.close()

    #plot mean values and mean abs values
    observed_df = mean_df[mean_df['shuffle category'] == 'observed']
    non_obs_df = mean_df[mean_df['shuffle category'] != 'observed']
    for column in mean_df.columns:
        if 'shuffle' in column:
            continue
        #make one plot just random, syn_ratio and observed
        if 'abs' in column:
            xlabel = column
        else:
            xlabel = f'{column} GPi/(GPe + GPi) syn area'
        #TO DO: plot observed value as one line and rest as distributions
        sns.histplot(x=column, data=non_obs_df, hue='shuffle category', palette=cat_palette,
                     common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent', bins=100)
        plt.axvline(np.array(observed_df[column]), color = cat_palette['observed'])
        plt.ylabel('% of cells', fontsize=fontsize)
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.title(xlabel)
        plt.savefig(f'{f_name}/{column}_{n_it}_hist_perc.png')
        plt.savefig(f'{f_name}/{column}_{n_it}_hist_perc.svg')
        plt.close()
        # plot only observed, syn_ratio and random
        rndm_inds = mean_df['shuffle category'] == 'random'
        obs_inds = mean_df['shuffle category'] == 'observed'
        syn_ratio_inds = mean_df['shuffle category'] == 'random with syn ratio'
        spec_inds = np.any([rndm_inds, obs_inds, syn_ratio_inds], axis=0)
        it_shuffle_df_spec = mean_df[spec_inds]
        sns.histplot(x=column, data=non_obs_df, hue='shuffle category', palette=cat_palette,
                     common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent', bins = 100)
        plt.axvline(np.array(observed_df[column]), color = cat_palette['observed'])
        plt.ylabel('% of cells', fontsize=fontsize)
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.title(xlabel)
        plt.savefig(f'{f_name}/{column}_{n_it}_synratio_hist_perc.png')
        plt.savefig(f'{f_name}/{column}_{n_it}_synratio_hist_perc.svg')
        plt.close()

    log.info('Step 5/5: calculate statistics')
    raise ValueError
    #p-vaslue if observed mean can be part of distribution
    #scipy ttest1samp

    log.info('Analysis done')
















