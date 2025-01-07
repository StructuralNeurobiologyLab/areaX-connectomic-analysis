#compare different ves number modulatory syn size runs to find best sample size

if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import ranksums, skew, skewtest, kstest

    f_name = f"cajal/scratch/users/arother/bio_analysis_results/single_vesicle_analysis/250107_bootstrap_sample_size_comparison/"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'bootstrap_sample_size_log', log_dir=f_name)
    #step 1: load all p-values for various different iterations
    n_it = 1000
    non_syn_thresh = 3000
    mcl = 200
    bootstrap_n = [20, 50, 100, 200, 500, 1000, 2000, 5000, 7500, 10000]
    fontsize = 20
    log.info(f'Load different runs of vesicle number modulatory versus syn size with the following different sample size for bootstrapping: {bootstrap_n}')
    pre_post_pairs = [('HVC', 'MSN'), ('GPi', 'STN'), ('MSN', 'GPi')]
    #save data in dataframe
    log.info('Step 1/3: Load p-values from different analysis and get parameters for each connection pair and each sample size')
    boot_p_values = pd.DataFrame(columns = ['p value', 'connection', 'sample size'], index = range(n_it * len(bootstrap_n) * len(pre_post_pairs)))
    ov_foldername = f"cajal/scratch/users/arother/bio_analysis_results/single_vesicle_analysis/"
    for bi, bn in enumerate(bootstrap_n):
        start_bi = bi*n_it*len(pre_post_pairs)
        boot_p_values.loc[start_bi: (bi + 1) *n_it*len(pre_post_pairs) - 1, 'sample size'] = bn
        for ci, cn in enumerate(pre_post_pairs):
            filename = f'{ov_foldername}/2501067_j0251v6_TAN_ves_num_syn_size_modulatory_{mcl}_syn{non_syn_thresh}_r5_{cn[0]}_{cn[1]}_it{n_it}_bn{bn}/bootstrapped_{bn}_ranksum_values.csv'
            cn_bn_pd = pd.read_csv(filename, index_col=0)
            p_values = np.array(cn_bn_pd['p value'])
            boot_p_values.loc[ci*n_it + start_bi: (1 + ci)*n_it + start_bi - 1, 'connection'] = f'{cn[0]} {cn[1]}'
            boot_p_values.loc[ci * n_it + start_bi: (1 + ci) * n_it + start_bi - 1, 'p value'] = p_values

    boot_p_values.to_csv(f'{f_name}/bootstrap_samples_p_values_combined.csv')

    log.info('Step 2/3: Get median p-value, skewness, skewtest and ks test result for each connection and sample size')
    ov_columns = ['median p value', 'skewness', 'skewtest p value', 'ks test p value', 'fraction < 0.05', 'sample size', 'connection']
    boot_ov_df = pd.DataFrame(columns = ov_columns, index = range(len(bootstrap_n) * len(pre_post_pairs)))
    threshold = 0.05
    for ci, cn in enumerate(pre_post_pairs):
        cn_str = f'{cn[0]} {cn[1]}'
        ci_start = ci*len(bootstrap_n)
        ci_end = (ci + 1)* len(bootstrap_n) - 1
        boot_ov_df.loc[ci_start: ci_end, 'connection'] = cn_str
        cn_df = boot_p_values[boot_p_values['connection'] == cn_str]
        bn_groups = cn_df.groupby('sample size')
        boot_ov_df.loc[ci_start: ci_end, 'sample size'] = bootstrap_n
        boot_ov_df.loc[ci_start: ci_end, 'median p value'] = np.array(bn_groups['p value'].median())
        for bi, bn in enumerate(bootstrap_n):
            bn_group_p_values = np.array(bn_groups.get_group(bn)['p value'], dtype = 'float')
            #get skewness to see if left-skewed distribution
            skewness = skew(bn_group_p_values)
            boot_ov_df.loc[ci_start + bi, 'skewness'] = skewness
            #use skewtest to see if skewness signficantly different than normal distribution
            skew_stats, skew_p_value = skewtest(bn_group_p_values)
            boot_ov_df.loc[ci_start + bi, 'skewtest p value'] = skew_p_value
            #use ks test to see if signficanlty different from uniform distribution
            ks_stat, ks_p_value = kstest(bn_group_p_values, 'uniform')
            boot_ov_df.loc[ci_start + bi, 'ks test p value'] = skew_p_value
            #calculate fraction under 0.05
            frac_below_threshold = np.sum(bn_group_p_values< threshold) / len(bn_group_p_values)
            boot_ov_df.loc[ci_start + bi, 'fraction < 0.05'] = frac_below_threshold

    boot_ov_df.to_csv(f'{f_name}/bootstrap_sample_size_ov.csv')

    log.info('Step 3/3: Plot results')
    colors = ['#38C2BA', '#232121', '#912043']
    unique_conns = np.unique(boot_ov_df['connection'])
    conn_palette = {unique_conns[i]: colors[i] for i in range(len(unique_conns))}
    for param in boot_ov_df:
        if 'sample size' in param or 'connection' in param:
            continue
        sns.barplot(data = boot_ov_df, x = 'sample size', y = param, hue = 'connection', legend=True, palette=conn_palette)
        plt.xlabel('sample size', fontsize=fontsize)
        plt.ylabel(param, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title(f'param in different cell connections with different bootstrap sizes')
        if not 'skewness' in param:
            plt.axhline(0.05, color='black', linestyle='--', linewidth=1)
        plt.savefig(f'{f_name}/{param}_bootstrap_sample_sizes.png')
        plt.savefig(f'{f_name}/{param}_bootstrap_sample_sizes.svg')
        plt.close()

    log.info('Analysis finished.')