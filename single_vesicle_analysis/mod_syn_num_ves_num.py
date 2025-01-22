#plot vesicle number in dependence of synapse size

if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params

    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    with_glia = False
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    celltype = 5
    ct_str = ct_dict[celltype]
    pre_cts = 2
    post_cts = 3
    pre_ct_str = ct_dict[pre_cts]
    post_ct_str = ct_dict[post_cts]
    fontsize = 20
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/single_vesicle_analysis/250122_mod_ves_num_syn_num_{ct_str}_{pre_ct_str}_{post_ct_str}/"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'mod_ves_num_syn_num_log', log_dir=f_name)
    file_name = f"cajal/scratch/users/arother/bio_analysis_results/single_vesicle_analysis/250122_{ct_str}_{pre_ct_str}_{post_ct_str}_1000_subsamples_syns.csv"
    log.info(f'Load results from different subsampled runs from {file_name}')
    sub_sampled_res = pd.read_csv(file_name)

    log.info('Plot mean vesicle number versus synapse number per quantile')
    sub_sampled_res['mean ves number'] = np.array(sub_sampled_res['ves number small']) + np.array(sub_sampled_res['ves number large']) / 2
    sub_sampled_res.to_csv(f'{f_name}/subsampled_syn_results.csv')
    sns.pointplot(data = sub_sampled_res, x = 'n syns quantile', y = 'mean ves number', color='black', linestyles='none')
    plt.xlabel('n syns quantile', fontsize=fontsize)
    plt.ylabel('mean ves number', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(f'Mean vesicle number vs synapse number')
    plt.savefig(f'{f_name}/ves_num_syn_num.png')
    plt.savefig(f'{f_name}/ves_num_syn_num.svg')
    plt.close()

    log.info('Plot done')

