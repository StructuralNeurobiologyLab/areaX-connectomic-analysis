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
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/for_eval/2401003_j0251{version1}vs_{version2}_wd_comp_%i_ax%i_ms_%f" % (
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

    log.info('Step 2:5: Get syn prob distribution of all synapses')


    log.info('Step 3/5: Filter synapses for min_comp_len, min_syn_size')

    log.info('Step 4/5: Get syn prob distribution of filtered synapses')

    log.info('Steo 5/5: Get syn density of filtered synapses')


    syn_prob = 0.6
    log.info(f'Also get syn density with the additional filter of syn_prob = {syn_prob}')

    log.info('Analysis done')