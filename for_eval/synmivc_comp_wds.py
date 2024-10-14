#compare two versions of working directories
#compare synapse density, mito density, vc density
#compare rfc distribution

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    version1 = 'v4'
    version2 = 'v6'
    analysis_params1 = Analysis_Params(version=version1)
    analysis_params2 = Analysis_Params(version=version2)
    global_params.wd = analysis_params1.working_dir()
    #this analysis assumes the different versions are done on the same dataset
    #so scaling and dataset size are the same
    with_glia = True
    ct_dict1 = analysis_params1.ct_dict(with_glia=with_glia)
    ct_dict2 = analysis_params2.ct_dict(with_glia=with_glia)
    neuron_ct_dict1 = analysis_params1.ct_dict(with_glia=False)
    neuron_ct_dict2 = analysis_params2.ct_dict(with_glia = False)
    min_comp_len_cell = 200
    min_comp_len_ax = 50
    min_syn_size = 0.1
    fontsize = 20
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw', 'STNGPINTv6', 'AxTePkBrv6', 'TePkBrNGF', 'TeBKv6MSNyw'
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/2401014_j0251{version1}vs_{version2}_wd_comp_%i_ax%i_ms_%f" % (
                 min_comp_len_cell, min_comp_len_ax, min_syn_size)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('wd_comp_log', log_dir=f_name)
    log.info(f'min comp len cells = {min_comp_len_cell} µm, min comp len ax = {min_comp_len_ax} µm,'
             f'min syn size = {min_syn_size} µm²')
    celltypes1 = np.array([ct_dict1[ct] for ct in neuron_ct_dict1])
    celltypes2 = np.array([ct_dict2[ct] for ct in neuron_ct_dict2])
    num_cts1 = len(celltypes1)
    num_cts2 = len(celltypes2)
    ver_palette = {version1: '#232121', version2: '#15AEAB'}
    versions = [version1, version2]
    num_versions = len(versions)
    organelles = ['syn_ssv', 'mi', 'vc']
    num_organells = len(organelles)
    log.info(f'Comparison will be done on these versions {versions} and these organelles {organelles}')
    wds = {version1: analysis_params1.working_dir(), version2: analysis_params2.working_dir()}

    log.info(' Step 1/5: Get syn, mito, vc number and volume density')
    columns = ['number', 'summed size', 'volume density', 'organelle', 'version']
    dataset_scaling = global_params.config['scaling']
    dataset_axis_sizes = np.array([27119, 27350, 15494])
    dataset_size =  np.prod(dataset_axis_sizes) * np.prod(dataset_scaling) * 10**(-9)
    log.info(f'The dataset measures {dataset_axis_sizes} in x,y,z with a scaling of {dataset_scaling}, this results in a volume of {dataset_size:.2f} µm³.')
    #also get value of dataset volume here
    density_df = pd.DataFrame(columns= columns, index = range(num_versions * num_organells))
    for vi, ver in enumerate(versions):
        wd = wds[ver]
        for oi, org in enumerate(organelles):
            sd = SegmentationDataset(org, working_dir=wd)
            #for synapses use mesh area [µm²]
            #for other organelles use size [voxels] and convert to µm³
            if org == 'syn_ssv':
                org_sizes = sd.load_numpy_data('mesh_area') / 2
            else:
                org_sizes = sd.load_numpy_data('size') * np.prod(dataset_scaling) * 10**(-9)
            density_df.loc[vi * num_organells + oi, 'version'] = ver
            density_df.loc[vi * num_organells + oi, 'organelle'] = org
            density_df.loc[vi * num_organells + oi, 'number'] = len(org_sizes)
            density_df.loc[vi * num_organells + oi, 'summed size'] = np.sum(org_sizes)
            density_df.loc[vi * num_organells + oi, 'volume density'] = np.sum(org_sizes) / dataset_size

    density_df.to_csv(f'{f_name}/synmivc_density_comp.csv')
    for col in density_df.columns:
        if 'organelle' in col or 'version' in col:
            continue
        sns.barplot(data=density_df, x='organelle', y=col, hue='version',
                    palette=ver_palette)
        plt.xlabel('organelle', fontsize=fontsize)
        plt.ylabel(col, fontsize=fontsize)
        plt.title(f'{col} comparison of different organelles')
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(f'{f_name}/{col}_{version1}_{version2}_orgs.png')
        plt.savefig(f'{f_name}/{col}_{version1}_{version2}_orgs.svg')
        plt.close()

    log.info('Step 2/5: Get syn prob distribution of all synapses')
    num_syns_total = density_df[density_df['organelle'] == 'syn_ssv']['number'].sum()
    syn_prob_df = pd.DataFrame(columns = ['syn prob', 'syn id', 'version', 'syn size'], index = range(num_syns_total))
    start = 0
    for ver in versions:
        wd = wds[ver]
        sd = SegmentationDataset('syn_ssv', working_dir=wd)
        syn_ids = sd.ids
        syn_prob = sd.load_numpy_data('syn_prob')
        syn_size = sd.load_numpy_data('mesh_area') / 2
        num_syns = len(syn_ids)
        syn_prob_df.loc[start: start + num_syns - 1, 'syn id'] = syn_ids
        syn_prob_df.loc[start: start + num_syns - 1, 'syn prob'] = syn_prob
        syn_prob_df.loc[start: start + num_syns - 1, 'version'] = ver
        syn_prob_df.loc[start: start + num_syns - 1, 'syn size'] = syn_size
        start += len(syn_ids)

    #syn_prob_df.to_csv(f'{f_name}/syn_prob_df_comp_{version1}_{version2}.csv')
    sns.histplot(data = syn_prob_df, x = 'syn prob', hue = 'version', palette= ver_palette, fill=False,
                 kde=False, element='step', linewidth=3)
    plt.xlabel('synapse probability', fontsize=fontsize)
    plt.ylabel('number of synapses', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/{version1}_{version2}_synprob_hist.png')
    plt.savefig(f'{f_name}/{version1}_{version2}_synprob_hist.svg')
    plt.close()
    sns.histplot(data=syn_prob_df, x='syn prob', hue='version', palette=ver_palette, stat='percent', fill=False,
                 kde=False, element='step', linewidth=3)
    plt.xlabel('synapse probability', fontsize=fontsize)
    plt.ylabel('% of synapses', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/{version1}_{version2}_synprob_hist_perc.png')
    plt.savefig(f'{f_name}/{version1}_{version2}_synprob_hist_perc.svg')
    plt.close()

    log.info('Step 3/5: Filter synapses for min_comp_len, min_syn_size, full neuronal cells and axo-dendritic synapses only')

    suitable_ids_dict = {}
    all_suitable_ids = []
    all_cell_dict = {}
    all_celltypes = []
    all_celltypes_num = []
    axon_cts = analysis_params2.axon_cts()
    log.info(f'Use {version2} to load suitable cells')
    for ct in range(num_cts2):
        # only get cells with min_comp_len, MSN with max_comp_len or axons with min ax_len
        ct_str = ct_dict2[ct]
        cell_dict = analysis_params2.load_cell_dict(ct)
        all_cell_dict[ct] = cell_dict
        cellids = np.array(list(cell_dict.keys()))
        if ct in axon_cts:
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_ax,
                                            axon_only=True, max_path_len=None)
        else:
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cell,
                                            axon_only=False, max_path_len=None)
        cellids = np.sort(cellids)
        suitable_ids_dict[ct] = cellids
        all_suitable_ids.append(cellids)
        all_celltypes.append([ct_dict2[ct] for i in cellids])
        all_celltypes_num.append([[ct] for i in cellids])
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict2[ct]))

    all_suitable_ids = np.concatenate(all_suitable_ids)
    all_celltypes = np.concatenate(all_celltypes)

    start_density_df = len(density_df)
    for i, ver in enumerate(versions):
        wd = wds[ver]
        sd = SegmentationDataset('syn_ssv', working_dir=wd)
        m_ids = sd.ids
        m_axs = sd.load_numpy_data("partner_axoness")
        m_axs[m_axs == 4] = 1
        m_axs[m_axs == 3] = 1
        m_ssv_partners = sd.load_numpy_data("neuron_partners")
        m_sizes = sd.load_numpy_data("mesh_area") / 2
        # get rid of synapses without axon
        ax_inds = np.any(np.in1d(m_axs, 1).reshape(len(m_axs), 2), axis=1)
        m_axs = m_axs[ax_inds]
        m_sizes = m_sizes[ax_inds]
        m_ids = m_ids[ax_inds]
        m_ssv_partners = m_ssv_partners[ax_inds]
        ##filter for axo-dendritic
        #only check if no axo-axonic ones in there, that axon must be there already from above
        den_so = np.array([0, 2])
        den_so_inds = np.any(np.in1d(m_axs, den_so).reshape(len(m_axs), 2), axis=1)
        m_ids = m_ids[den_so_inds]
        m_ssv_partners = m_ssv_partners[den_so_inds]
        m_sizes = m_sizes[den_so_inds]
        #filter for min syn size
        size_inds = m_sizes > min_syn_size
        m_ids = m_ids[size_inds]
        m_ssv_partners = m_ssv_partners[size_inds]
        m_sizes = m_sizes[size_inds]
        #filter for suitable cellids
        suit_ct_inds = np.any(np.in1d(m_ssv_partners, all_suitable_ids).reshape(len(m_ssv_partners), 2), axis=1)
        m_ids = m_ids[suit_ct_inds]
        m_sizes = m_sizes[suit_ct_inds]
        log.info(f'{len(m_ids)} synapses suitable from {version1}')
        #get number, surface area and density
        density_df.loc[start_density_df + i, 'organelle'] = 'filtered syns'
        density_df.loc[start_density_df + i, 'version'] = ver
        density_df.loc[start_density_df + i, 'number'] = len(syn_ids)
        density_df.loc[start_density_df + i, 'summed size'] = np.sum(m_sizes)
        density_df.loc[start_density_df + i, 'volume density'] = np.sum(m_sizes) / dataset_size
        #get syn prob of filtered synapses
        filtered_inds = np.in1d(syn_prob_df['syn id'], m_ids)
        ver_id = syn_prob_df['version'] == ver
        raise ValueError
        syn_prob_df = syn_prob_df[np.all(filtered_ids, ver_id, axis = 1)]

    density_df.to_csv(f'{f_name}/synmivc_density_comp.csv')

    log.info('Step 4/5: Get syn prob distribution of filtered synapses')
    sns.histplot(data=syn_prob_df, x='syn prob', hue='version', palette=ver_palette, fill=False,
                 kde=False, element='step', linewidth=3)
    plt.xlabel('synapse probability', fontsize=fontsize)
    plt.ylabel('number of synapses', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/{version1}_{version2}_synprob_hist_filtered.png')
    plt.savefig(f'{f_name}/{version1}_{version2}_synprob_hist_filtered.svg')
    plt.close()
    sns.histplot(data=syn_prob_df, x='syn prob', hue='version', palette=ver_palette, stat='percent', fill=False,
                 kde=False, element='step', linewidth=3)
    plt.xlabel('synapse probability', fontsize=fontsize)
    plt.ylabel('% of synapses', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/{version1}_{version2}_synprob_hist_filtered_perc.png')
    plt.savefig(f'{f_name}/{version1}_{version2}_synprob_hist_filtered_perc.svg')
    plt.close()

    log.info('Step 5/5: Plot syn density of filtered synapses')
    for col in density_df.columns:
        if 'organelle' in col or 'version' in col:
            continue
        sns.barplot(data=density_df, x='organelle', y=col, hue='version',
                    palette=ver_palette)
        plt.xlabel('organelle', fontsize=fontsize)
        plt.ylabel(col, fontsize=fontsize)
        plt.title(f'{col} comparison of different organelles')
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(f'{f_name}/{col}_{version1}_{version2}_orgs_filtered.png')
        plt.savefig(f'{f_name}/{col}_{version1}_{version2}_orgs_filtered.svg')
        plt.close()


    syn_prob = 0.6
    log.info(f'Also get syn density with the additional filter of syn_prob = {syn_prob}')
    syn_prob_df = syn_prob_df[syn_prob_df['syn prob'] > syn_prob]
    start_density_df = len(density_df)
    for i, ver in enumerate(versions):
        ver_syn_prob_df = syn_prob_df[syn_prob_df['version'] == ver]
        ver_summed_size = np.array(ver_syn_prob_df['syn size'].sum())
        density_df.loc[start_density_df + i, 'organelle'] = 'filtered syns with prob'
        density_df.loc[start_density_df + i, 'version'] = ver
        density_df.loc[start_density_df + i, 'number'] = len(ver_syn_prob_df)
        density_df.loc[start_density_df + i, 'summed size'] = ver_summed_size
        density_df.loc[start_density_df + i, 'volume density'] = ver_summed_size / dataset_size

    density_df.to_csv(f'{f_name}/synmivc_density_comp.csv')
    for col in density_df.columns:
        if 'organelle' in col or 'version' in col:
            continue
        sns.barplot(data=density_df, x='organelle', y=col, hue='version',
                    palette=ver_palette)
        plt.xlabel('organelle', fontsize=fontsize)
        plt.ylabel(col, fontsize=fontsize)
        plt.title(f'{col} comparison of different organelles')
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(f'{f_name}/{col}_{version1}_{version2}_orgs_filtered_prob.png')
        plt.savefig(f'{f_name}/{col}_{version1}_{version2}_orgs_filtered_prob.svg')
        plt.close()

    log.info('Analysis done')