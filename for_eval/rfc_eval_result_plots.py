#write script for rfc eval results
#plot for all syns and filtered syns: fraction of true synapses based on each bin

if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    import os as os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    version = 'v6'
    fontsize = 20
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/240819_j0251{version}_manual_syn_rfc_eval"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'rfc_syn_eval_res_log', log_dir=f_name)

    eval_path = 'cajal/scratch/users/arother/bio_analysis_results/for_eval/' \
                '240726_j0251v6_rfc_syn_eval_mcl_200_ax50_ms_0.100000/' \
                        'random_syn_coords_evaluation_results_withprobs.csv'
    log.info(f'evaluation results laoded from {eval_path}')
    eval_df = pd.read_csv(eval_path)
    eval_df.loc[eval_df['final'] == 'y', 'final'] = 'True'
    eval_df.loc[eval_df['final'] == 'n', 'final'] = 'False'
    assert(len(np.unique(eval_df['final'])) == 2)
    log.info(f'In total {len(eval_df)} synapses')
    # get number of true synapses overall
    true_syn_df = eval_df[eval_df['final'] == 'True']
    true_syn_num = len(true_syn_df)
    log.info(f'{true_syn_num} are evaluated as true synapses.')

    #split into categories
    eval_all_syns_df = eval_df[eval_df['category'] == 'all syns']
    all_syns_true_df = eval_all_syns_df[eval_df['final'] == 'True']
    all_syns_true_num = len(all_syns_true_df)
    log.info(f'In cateogry with all synapses, {len(eval_all_syns_df)} syns were evaluated.')
    log.info(f'{all_syns_true_num} synapses of them were evaluated as true.')
    eval_filtered_syns_df = eval_df[eval_df['category'] == 'filtered syns']
    log.info(f'In cateogry with only filtered synapses, {len(eval_filtered_syns_df)} syns were evaluated.')
    filtered_syns_true_df = eval_filtered_syns_df[eval_df['final'] == 'True']
    filtered_syns_true_num = len(filtered_syns_true_df)
    log.info(f'{filtered_syns_true_num} synapses of them were evaluated as true.')

    #plot fraction of true synapses in each probability bin
    #do for all three dfs
    syn_prob_bins = np.unique(eval_df['syn prob bin'])
    overview_columns = ['syn prob bin', 'number syns combined', 'number all syns', 'number filtered syns',
                        'fraction true combined', 'fraction true all syns', 'fraction true filtered syns']
    overview_df = pd.DataFrame(columns = overview_columns, index = range(len(syn_prob_bins)))
    combined_prob_groups = eval_df.groupby('syn prob bin')
    all_syn_prob_groups = eval_all_syns_df.groupby('syn prob bin')
    filtered_syn_prob_groups = eval_filtered_syns_df.groupby('syn prob bin')
    overview_df['syn prob bin'] = syn_prob_bins
    overview_df['number syns combined'] = np.array(combined_prob_groups.size())
    overview_df['number all syns'] = np.array(all_syn_prob_groups.size())
    overview_df['number filtered syns'] = np.array(filtered_syn_prob_groups.size())
    true_prob_groups_combined = true_syn_df.groupby('syn prob bin')
    all_syns_true_prob_groups = all_syns_true_df.groupby('syn prob bin')
    filtered_syns_true_prob_groups = filtered_syns_true_df.groupby('syn prob bin')
    overview_df['fraction true combined'] = np.array(true_prob_groups_combined.size()) / overview_df['number syns combined']
    overview_df['fraction true all syns'] = np.array(all_syns_true_prob_groups.size()) / overview_df['number all syns']
    overview_df['fraction true filtered syns'] = np.array(filtered_syns_true_prob_groups.size()) / overview_df['number filtered syns']
    overview_df.to_csv(f'{f_name}/overview_probs_df.csv')

    log.info('Plot results as barplot')
    #plot result as barplot
    sns.barplot(data = overview_df, x = 'syn prob bin', y = 'fraction true combined')
    plt.ylabel('fraction of true synapses', fontsize = fontsize)
    plt.xlabel('synapse probability', fontsize = fontsize)
    plt.title('All eval synapses')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/fraction_true_combined.png')
    plt.savefig(f'{f_name}/fraction_true_combined.svg')
    plt.close()
    sns.barplot(data=overview_df, x='syn prob bin', y='fraction true all syns')
    plt.ylabel('fraction of true synapses', fontsize=fontsize)
    plt.xlabel('synapse probability', fontsize=fontsize)
    plt.title('All syns eval synapses')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/fraction_true_all_syns.png')
    plt.savefig(f'{f_name}/fraction_true_all_syns.svg')
    plt.close()
    sns.barplot(data=overview_df, x='syn prob bin', y='fraction true filtered syns')
    plt.ylabel('fraction of true synapses', fontsize=fontsize)
    plt.xlabel('synapse probability', fontsize=fontsize)
    plt.title('Filtered eval synapses')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/fraction_true_filtered.png')
    plt.savefig(f'{f_name}/fraction_true_filtered.svg')
    plt.close()
    log.info('Analysis finished')