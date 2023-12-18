#randomly select 10 cellids for manual checkin

if __name__ == '__main__':
    from syconn import global_params
    import numpy as np
    from syconn.handler.config import initialize_logging
    import os as os
    import pandas as pd
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    bio_params = Analysis_Params(working_dir=global_params.wd, version='v5')
    ct_dict = bio_params.ct_dict()
    ax_cts = bio_params.axon_cts()
    num_samples = 15
    np.random.seed(41)
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/231201_j0251v5_check_ax_forgt_{num_samples}"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('Get rnd ids for different ax lengths', log_dir=f_name + '/logs/')

    #load all axon fragments with skeleton lengths from results
    length_filename = 'cajal/scratch/users/arother/bio_analysis_results/for_eval/231108_j0251v5_ax_fraglengths/proj_axon_lengths.csv'
    log.info(f'Load data from {length_filename}')
    log.info('Only ids selected which have one axo-dendritic or axo-somatic synapse')
    axon_df = pd.read_csv(length_filename)
    #get rid of old indexes
    axon_df = axon_df[axon_df.columns[3:]]

    log.info(f'Get {num_samples} rndm ids per length category and per celltype')
    #add bins for different categories
    cats = [0.0, 50, 100, 500, 1000, np.max(axon_df['skeleton length'])]
    length_cats = np.array(pd.cut(axon_df['skeleton length'], cats, right=False, labels=cats[:-1]))
    axon_df['length bins'] = length_cats
    #randomly select 10 per celltype and category and put into new working dir
    random_ids = []
    for ax_ct in ax_cts:
        ax_axon_df = axon_df[axon_df['celltype'] == ct_dict[ax_ct]]
        ax_length_groups = [group['cellid'].values for name, group in
                    ax_axon_df.groupby('length bins')]
        for i in range(len(cats) - 1):
            rnd_ids = np.random.choice(ax_length_groups[i], num_samples)
            random_ids.append(rnd_ids)

    random_ids = np.concatenate(random_ids)
    axon_df = axon_df[np.in1d(axon_df['cellid'], random_ids)]
    axon_df.to_csv(f'{f_name}/rnd_ax_ids.csv')

    log.info('Random ids saved')

