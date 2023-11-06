#plot syn_prob values for different versions
import pandas as pd

if __name__ == '__main__':
    from syconn.reps.segmentation import SegmentationDataset
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from syconn.handler.config import initialize_logging
    import os as os

    f_name = 'cajal/scratch/users/arother/bio_analysis_results/for_eval/231106_syn_prob/'
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('syn prob comparison', log_dir=f_name + '/logs/')

    log.info('Intitialize different versions of the data to compare')
    #PS from 2022 paper
    agglo2_wd = '/cajal/nvmescratch/projects/from_ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2'
    #new synapses but gt for rfc still from old synapses
    old_wd = '/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811'
    #new synapses and new gt
    new_wd = '/cajal/nvmescratch/projects/data/songbird/j0251/j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_20230822'
    log.info(f'Comparing old syns, old syn gt from {agglo2_wd} \n'
             f' new syns but old syn gt from {old_wd} \n '
             f'new syns and new gt from {new_wd}')

    log.info('Load syn numpy data from different directories')

    agglo2_sd_syn_ssv = SegmentationDataset('syn_ssv', working_dir=agglo2_wd)
    old_sd_syn_ssv = SegmentationDataset('syn_ssv', working_dir=old_wd)
    new_sd_syn_ssv = SegmentationDataset('syn_ssv', working_dir=new_wd)

    agglo2_syn_probs = agglo2_sd_syn_ssv.load_numpy_data('syn_prob')
    agglo2_syn_axoness = agglo2_sd_syn_ssv.load_numpy_data('partner_axoness')
    old_syn_probs = old_sd_syn_ssv.load_numpy_data('syn_prob')
    old_syn_axoness = old_sd_syn_ssv.load_numpy_data('partner_axoness')
    old_syn_cts = old_sd_syn_ssv.load_numpy_data('partner_celltypes')
    new_syn_probs = new_sd_syn_ssv.load_numpy_data('syn_prob')
    new_syn_axoness = new_sd_syn_ssv.load_numpy_data('partner_axoness')
    new_syn_cts = new_sd_syn_ssv.load_numpy_data('partner_celltypes')

    log.info('Get mean and median syn_prob parameters')
    sp_columns = ['total number', 'median syn prob', 'mean syn prob']
    sp_index = ['agglo2 all', 'v5 old gt all', 'v5 new gt all', 'agglo2 axo-denso', 'v5 old gt axo-denso', 'v5 new gt axo-denso',
                'v5 old gt neurons', 'v5 new gt neurons']
    sp_params = pd.DataFrame(columns=sp_columns, index=sp_index)
    sp_params.loc['agglo2 all', 'total number'] = len(agglo2_syn_probs)
    sp_params.loc['agglo2 all', 'median syn prob'] = np.median(agglo2_syn_probs)
    sp_params.loc['agglo2 all', 'mean syn prob'] = np.mean(agglo2_syn_probs)
    sp_params.loc['v5 old gt all', 'total number'] = len(old_syn_probs)
    sp_params.loc['v5 old gt all', 'median syn prob'] = np.median(old_syn_probs)
    sp_params.loc['v5 old gt all', 'mean syn prob'] = np.mean(old_syn_probs)
    sp_params.loc['v5 new gt all', 'total number'] = len(new_syn_probs)
    sp_params.loc['v5 new gt all', 'median syn prob'] = np.median(new_syn_probs)
    sp_params.loc['v5 new gt all', 'mean syn prob'] = np.mean(new_syn_probs)
    sp_params.to_csv(f'{f_name}/syn_prob_params.csv')

    log.info('Plot complete syn prob distribution')
    total_number = len(agglo2_syn_probs) + len(old_syn_probs) + len(new_syn_probs)
    full_syn_prob_df = pd.DataFrame(columns = ['synapse probability', 'version'], index = range(total_number))
    full_syn_prob_df.loc[0: len(agglo2_syn_probs) - 1, 'synapse probability'] = agglo2_syn_probs
    full_syn_prob_df.loc[0: len(agglo2_syn_probs) - 1, 'version'] = 'agglo2'
    full_syn_prob_df.loc[len(agglo2_syn_probs): len(agglo2_syn_probs) + len(old_syn_probs) - 1, 'synapse probability'] = old_syn_probs
    full_syn_prob_df.loc[len(agglo2_syn_probs): len(agglo2_syn_probs) + len(old_syn_probs) - 1,
    'version'] = 'v5 old gt'
    full_syn_prob_df.loc[len(agglo2_syn_probs) + len(old_syn_probs): total_number - 1, 'synapse probability'] = new_syn_probs
    full_syn_prob_df.loc[len(agglo2_syn_probs) + len(old_syn_probs): total_number - 1,
    'version'] = 'v5 new gt'
    full_syn_prob_df.to_csv(f'{f_name}/full_syn_prob_values.csv')
    #make histogram
    sns.histplot(data = full_syn_prob_df, x = 'synapse probability', hue = 'version', fill= False,
                 kde= False, element='step')
    plt.title('Synapse probability of all synapses in different versions')
    plt.savefig(f'{f_name}/all_syn_prob.png')
    plt.savefig(f'{f_name}/all_syn_prob.svg')
    plt.close()
    #plot percent histogram
    sns.histplot(data=full_syn_prob_df, x='synapse probability', hue='version', fill=False,
                 kde=False, element='step', stat='percent')
    plt.title('Synapse probability of all synapses in different versions')
    plt.savefig(f'{f_name}/all_syn_prob_perc.png')
    plt.savefig(f'{f_name}/all_syn_prob_perc.svg')
    plt.close()

    log.info('Filter synapses for axo-dendritic  or axo-somatic synapses only')
    den_so = [0, 2]
    #agglo2
    agglo2_ax_inds = np.any(agglo2_syn_axoness == 1, axis=1)
    agglo2_syn_probs = agglo2_syn_probs[agglo2_ax_inds]
    agglo2_syn_axoness = agglo2_syn_axoness[agglo2_ax_inds]
    agglo2_denso_inds = np.any(np.in1d(agglo2_syn_axoness, den_so).reshape(len(agglo2_syn_axoness), 2), axis=1)
    agglo2_syn_probs = agglo2_syn_probs[agglo2_denso_inds]
    #v5 old gt
    old_ax_inds = np.any(old_syn_axoness == 1, axis=1)
    old_syn_probs = old_syn_probs[old_ax_inds]
    old_syn_axoness = old_syn_axoness[old_ax_inds]
    old_syn_cts = old_syn_cts[old_ax_inds]
    old_denso_inds = np.any(np.in1d(old_syn_axoness, den_so).reshape(len(old_syn_axoness), 2), axis=1)
    old_syn_probs = old_syn_probs[old_denso_inds]
    old_syn_cts = old_syn_cts[old_denso_inds]
    #v5 new gt
    new_ax_inds = np.any(new_syn_axoness == 1, axis=1)
    new_syn_probs = new_syn_probs[new_ax_inds]
    new_syn_axoness = new_syn_axoness[new_ax_inds]
    new_syn_cts = new_syn_cts[new_ax_inds]
    new_denso_inds = np.any(np.in1d(new_syn_axoness, den_so).reshape(len(new_syn_axoness), 2), axis=1)
    new_syn_probs = new_syn_probs[new_denso_inds]
    new_syn_cts = new_syn_cts[new_denso_inds]
    #since synapses in v5 are the same, the length should be the same
    assert(len(old_syn_probs) == len(new_syn_probs))

    log.info('Get params and plot histogram of filtered synapses')
    sp_params.loc['agglo2 axo-denso', 'total number'] = len(agglo2_syn_probs)
    sp_params.loc['agglo2 axo-denso', 'median syn prob'] = np.median(agglo2_syn_probs)
    sp_params.loc['agglo2 axo-denso', 'mean syn prob'] = np.mean(agglo2_syn_probs)
    sp_params.loc['v5 old gt axo-denso', 'total number'] = len(old_syn_probs)
    sp_params.loc['v5 old gt axo-denso', 'median syn prob'] = np.median(old_syn_probs)
    sp_params.loc['v5 old gt axo-denso', 'mean syn prob'] = np.mean(old_syn_probs)
    sp_params.loc['v5 new gt axo-denso', 'total number'] = len(new_syn_probs)
    sp_params.loc['v5 new gt axo-denso', 'median syn prob'] = np.median(new_syn_probs)
    sp_params.loc['v5 new gt axo-denso', 'mean syn prob'] = np.mean(new_syn_probs)
    sp_params.to_csv(f'{f_name}/syn_prob_params.csv')
    #plot similar as above
    total_number = len(agglo2_syn_probs) + len(old_syn_probs) + len(new_syn_probs)
    axdenso_syn_prob_df = pd.DataFrame(columns=['synapse probability', 'version'], index=range(total_number))
    axdenso_syn_prob_df.loc[0: len(agglo2_syn_probs) - 1, 'synapse probability'] = agglo2_syn_probs
    axdenso_syn_prob_df.loc[0: len(agglo2_syn_probs) - 1, 'version'] = 'agglo2'
    axdenso_syn_prob_df.loc[len(agglo2_syn_probs): len(agglo2_syn_probs) + len(old_syn_probs) - 1,
    'synapse probability'] = old_syn_probs
    axdenso_syn_prob_df.loc[len(agglo2_syn_probs): len(agglo2_syn_probs) + len(old_syn_probs) - 1,
    'version'] = 'v5 old gt'
    axdenso_syn_prob_df.loc[len(agglo2_syn_probs) + len(old_syn_probs): total_number - 1,
    'synapse probability'] = new_syn_probs
    axdenso_syn_prob_df.loc[len(agglo2_syn_probs) + len(old_syn_probs): total_number - 1,
    'version'] = 'v5 new gt'
    axdenso_syn_prob_df.to_csv(f'{f_name}/axdenso_syn_prob_values.csv')
    #make sure values were filtered
    assert (len(axdenso_syn_prob_df) < len(full_syn_prob_df))
    #histogram
    sns.histplot(data=axdenso_syn_prob_df, x='synapse probability', hue='version', fill=False,
                 kde=False, element='step')
    plt.title('Synapse probability of axo-dendritic and axo-somatic synapses in different versions')
    plt.savefig(f'{f_name}/axdenso_syn_prob.png')
    plt.savefig(f'{f_name}/axdenso_syn_prob.svg')
    plt.close()
    #normalised histogram
    sns.histplot(data=axdenso_syn_prob_df, x='synapse probability', hue='version', fill=False,
                 kde=False, element='step', stat='percent')
    plt.title('Synapse probability of axo-dendritic and axo-somatic synapses in different versions')
    plt.savefig(f'{f_name}/axdenso_syn_prob_perc.png')
    plt.savefig(f'{f_name}/axdenso_syn_prob_perc.svg')
    plt.close()

    log.info('Filter v5 old and new gt for synapses only between neurons')
    #agglo2 neurons did not have glia celltypes distinguished
    #neurons in v5: celltypes 0-10 (check for newer versions of data; might differ between wds)
    neuron_cts = np.arange(0, 11)
    old_ct_inds = np.all(np.in1d(old_syn_cts, neuron_cts).reshape(len(old_syn_cts), 2), axis=1)
    old_syn_probs = old_syn_probs[old_ct_inds]
    new_ct_inds = np.all(np.in1d(new_syn_cts, neuron_cts).reshape(len(new_syn_cts), 2), axis=1)
    new_syn_probs = new_syn_probs[new_ct_inds]

    log.info('Plot v5 only synapses between neurons')
    #get stats
    sp_params.loc['v5 old gt neurons', 'total number'] = len(old_syn_probs)
    sp_params.loc['v5 old gt neurons', 'median syn prob'] = np.median(old_syn_probs)
    sp_params.loc['v5 old gt neurons', 'mean syn prob'] = np.mean(old_syn_probs)
    sp_params.loc['v5 new gt neurons', 'total number'] = len(new_syn_probs)
    sp_params.loc['v5 new gt neurons', 'median syn prob'] = np.median(new_syn_probs)
    sp_params.loc['v5 new gt neurons', 'mean syn prob'] = np.mean(new_syn_probs)
    sp_params.to_csv(f'{f_name}/syn_prob_params.csv')
    # get df for histogramm
    total_number = len(old_syn_probs) + len(new_syn_probs)
    neuron_syn_prob_df = pd.DataFrame(columns=['synapse probability', 'version'], index=range(total_number))
    neuron_syn_prob_df.loc[0: len(old_syn_probs) - 1,'synapse probability'] = old_syn_probs
    neuron_syn_prob_df.loc[0: len(old_syn_probs) - 1,
    'version'] = 'v5 old gt'
    neuron_syn_prob_df.loc[len(old_syn_probs): total_number - 1,
    'synapse probability'] = new_syn_probs
    neuron_syn_prob_df.loc[len(old_syn_probs): total_number - 1,
    'version'] = 'v5 new gt'
    neuron_syn_prob_df.to_csv(f'{f_name}/neurons_syn_prob_values.csv')
    # make sure values were filtered
    assert (len(neuron_syn_prob_df) < len(axdenso_syn_prob_df))
    # histogram
    sns.histplot(data=neuron_syn_prob_df, x='synapse probability', hue='version', fill=False,
                 kde=False, element='step')
    plt.title('Synapse probability of axo-dendritic and axo-somatic synapses between neurons in different versions')
    plt.savefig(f'{f_name}/neuron_syn_prob.png')
    plt.savefig(f'{f_name}/neuron_syn_prob.svg')
    plt.close()
    # normalised histogram
    sns.histplot(data=neuron_syn_prob_df, x='synapse probability', hue='version', fill=False,
                 kde=False, element='step', stat='percent')
    plt.title('Synapse probability of axo-dendritic and axo-somatic synapses in different versions')
    plt.savefig(f'{f_name}/neuron_syn_prob_perc.png')
    plt.savefig(f'{f_name}/neuron_syn_prob_perc.svg')
    plt.close()

    log.info('Syn prob plots done')

