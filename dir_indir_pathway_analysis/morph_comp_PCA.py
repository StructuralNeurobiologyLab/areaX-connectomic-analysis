#morphological comparison of different celltypes + PCA

if __name__ == '__main__':
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    from syconn.reps.segmentation import SegmentationDataset
    import os as os
    import pandas as pd
    import numpy as np
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct, get_per_cell_mito_myelin_info, \
        get_spine_density, get_cell_soma_radius
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from cajal.nvmescratch.users.arother.bio_analysis.general.result_helper import plot_histogram_selection
    import seaborn as sns
    import matplotlib.pyplot as plt
    from syconn.mp.mp_utils import start_multiprocess_imap
    from scipy.stats import ranksums, kruskal
    from itertools import combinations
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    version = 'v6'
    bio_params = Analysis_Params(version = version)
    global_params.wd = bio_params.working_dir()
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)
    ct_dict = bio_params.ct_dict()
    min_comp_len = 200
    syn_prob = bio_params.syn_prob_thresh()
    min_syn_size = bio_params.min_syn_size()
    fontsize_jointplot = 20
    use_skel = False  # if true would use skeleton labels for getting soma; vertex labels more exact, also probably faster
    use_median = True  # if true use median of vertex coordinates to find centre
    cts = [9, 10, 11]
    cts_str = [ct_dict[ct] for ct in cts]
    color_key = 'RdTeINTv6'
    n_comps_PCA = 1
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/240422_j0251{version}_{cts_str}_morph_comp_radius_spiness_examplecells_mcl%i_fs%i_med%i_{color_key}_nc{n_comps_PCA}" % \
             (min_comp_len, fontsize_jointplot, use_median)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('Celltype seperation morphology', log_dir=f_name + '/logs/')
    log.info('Parameters to check: soma diameter, axon median radius, axon mitochondria density, axon myelin fraction, spine density')
    log.info(f'Celltypes to compare = {cts_str}')
    if use_skel:
        log.info('use skeleton node predictions to get soma mesh coordinates')
    else:
        log.info('use vertex label dict predictions to get soma vertices')
    if use_median:
        log.info('Median of coords used to get soma centre')
    else:
        log.info('Mean of coords used to get soma centre')
    log.info("Finding potential ngf subpopulations starts")
    #ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   #10: "NGF"}
    known_mergers = bio_params.load_known_mergers()
    suitable_ids_dict = dict()
    suitable_cell_dict = dict()
    all_suitable_ids = []
    all_celltypes = []
    cls = CelltypeColors(ct_dict=ct_dict)
    ct_palette = cls.ct_palette(color_key, num=False)
    for ct in cts:
        ct_str = ct_dict[ct]
        cell_dict = bio_params.load_cell_dict(ct)
        suitable_cell_dict[ct] = cell_dict
        cell_ids = np.array(list(cell_dict.keys()))
        merger_inds = np.in1d(cell_ids, known_mergers) == False
        cell_ids = cell_ids[merger_inds]
        cell_ids = check_comp_lengths_ct(cellids=cell_ids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                            axon_only=False,
                                            max_path_len=None)
        suitable_ids_dict[ct] = cell_ids
        all_suitable_ids.append(cell_ids)
        all_celltypes.append([ct_dict[ct] for i in cell_ids])

    log.info(f'{len(cell_ids)} {ct_str} are suitable for analysis')

    all_suitable_ids = np.concatenate(all_suitable_ids)
    all_celltypes = np.concatenate(all_celltypes)

    sd_mitossv = SegmentationDataset("mi", working_dir=global_params.config.working_dir)
    np_presaved_loc = bio_params.file_locations


    log.info("Step 1/4: Get morphological information from cellids")
    log.info('Get information about mitos, myelin and axon radius')
    columns = ['cellids', 'celltype', "axon median radius", "axon mitochondria volume density",
               'soma diameter', 'spine density', 'total mitochondria volume density']
    param_df = pd.DataFrame(columns = columns, index = range(len(all_suitable_ids)))
    param_df['cellids'] = all_suitable_ids
    param_df['celltype'] = all_celltypes

    for ct in cts:
        ct_str = ct_dict[ct]
        log.info(f'Analysis starts for {ct_str}')
        ct_org_ids = np.load(f'{np_presaved_loc}/{ct_dict[ct]}_mi_ids_fullcells.npy')
        ct_org_map2ssvids = np.load(f'{np_presaved_loc}/{ct_dict[ct]}_mi_mapping_ssv_ids_fullcells.npy')
        ct_org_axoness = np.load(f'{np_presaved_loc}/{ct_dict[ct]}_mi_axoness_coarse_fullcells.npy')
        ct_org_sizes = np.load(f'{np_presaved_loc}/{ct_dict[ct]}_mi_sizes_fullcells.npy')
        #cellid, min_comp_len, mi_ssv_ids, mi_sizes, mi_axoness, full_cell_dict = input
        morph_input = [[cell_id, min_comp_len, ct_org_map2ssvids, ct_org_sizes, ct_org_axoness, suitable_cell_dict[ct][cell_id]] for cell_id in suitable_ids_dict[ct]]
        morph_output = start_multiprocess_imap(get_per_cell_mito_myelin_info, morph_input)
        morph_output = np.array(morph_output)
        #[ax_median_radius_cell, axo_mito_volume_density_cell, rel_myelin_cell]
        axon_median_radius_ct = morph_output[:, 0]
        axon_mito_volume_density_ct = morph_output[:, 1]
        axon_myelin_ct= morph_output[:, 2]
        total_mito_volume_density_ct = morph_output[:, 4]
        ct_nonzero = axon_median_radius_ct > 0
        ids_nonzero = suitable_ids_dict[ct][ct_nonzero]
        # get soma diameter
        log.info('Get information about soma diameter')
        ct_soma_results = start_multiprocess_imap(get_cell_soma_radius, suitable_ids_dict[ct])
        ct_soma_results = np.array(ct_soma_results, dtype='object')
        ct_diameters = ct_soma_results[:, 1].astype(float) * 2
        #get spine density from all ngf cells
        log.info('Get cell spine density')
        morph_input = [[cell_id, min_comp_len, suitable_cell_dict[ct][cell_id]] for cell_id in suitable_ids_dict[ct]]
        spine_density = start_multiprocess_imap(get_spine_density, morph_input)
        spine_density = np.array(spine_density)[:, 0]
        nonzero_id_inds = np.in1d(param_df['cellids'], ids_nonzero)
        param_df.loc[nonzero_id_inds, 'axon median radius'] = axon_median_radius_ct[ct_nonzero]
        param_df.loc[nonzero_id_inds, 'axon mitochondria volume density'] = axon_mito_volume_density_ct[ct_nonzero]
        param_df.loc[nonzero_id_inds, 'soma diameter'] = ct_diameters[ct_nonzero]
        param_df.loc[nonzero_id_inds, 'spine density'] = spine_density[ct_nonzero]
        param_df.loc[nonzero_id_inds, 'total mitochondria volume density'] = total_mito_volume_density_ct[ct_nonzero]
        
    param_df.to_csv(f"{f_name}/{cts_str}_params.csv")

    log.info('Step 2/4: Plot results')
    example_cellids = [126798179, 1155532413, 15724767, 24397945, 1080627023]
    example_inds = np.in1d(param_df['cellids'], example_cellids)
    param_list = param_df.columns[2:]
    combs = list(combinations(range(len(param_list)), 2))
    for comb in combs:
        x = param_list[comb[0]]
        y = param_list[comb[1]]
        g = sns.JointGrid(data=param_df, x=x, y=y, hue="celltype", palette=ct_palette)
        g.plot_joint(sns.scatterplot)
        g.plot_marginals(sns.histplot, fill=True, alpha=0.3,
                         kde=False, bins=20, palette=ct_palette)
        g.ax_joint.set_xticks(g.ax_joint.get_xticks())
        g.ax_joint.set_yticks(g.ax_joint.get_yticks())
        if g.ax_joint.get_xticks()[0] < 0:
            g.ax_marg_x.set_xlim(0)
        if g.ax_joint.get_yticks()[0] < 0:
            g.ax_marg_y.set_ylim(0)
        g.ax_joint.set_xticklabels(["%.2f" % i for i in g.ax_joint.get_xticks()], fontsize=fontsize_jointplot)
        g.ax_joint.set_yticklabels(["%.2f" % i for i in g.ax_joint.get_yticks()], fontsize=fontsize_jointplot)
        if "radius" in x or 'diameter' in x:
            scatter_x = "%s [µm]" % x
        elif "volume density" in x:
            scatter_x = "%s [µm³/µm]" % x
        elif 'spine density' in x:
            scatter_x = '%s [1/µm]' % x
        else:
            scatter_x = x
        if "radius" in y or 'diameter' in y:
            scatter_y = "%s [µm]" % y
        elif "volume density" in y:
            scatter_y = "%s [µm³/µm]" % y
        elif 'spine density' in y:
            scatter_y = '%s [1/µm]' % y
        g.ax_joint.set_xlabel(scatter_x)
        g.ax_joint.set_ylabel(scatter_y)
        plt.savefig("%s/%s_%s_joinplot_comb.svg" % (f_name, x, y))
        plt.savefig("%s/%s_%s_joinplot_comb.png" % (f_name, x, y))
        plt.close()
        example_x = param_df[x][example_inds]
        example_y = param_df[y][example_inds]
        plt.scatter(param_df[x], param_df[y], color='gray')
        plt.scatter(example_x, example_y, color='red')
        plt.xlabel(scatter_x)
        plt.ylabel(scatter_y)
        plt.savefig(f'{f_name}/{x}_{y}_scatter_examplecells_ngffs.png')
        plt.close()

    log.info('Step 3/4 Calculate statistics')
    ct_str = np.unique(param_df['celltype'])
    ct_groups = param_df.groupby('celltype')
    #get overview stats
    summary_ct_df = pd.DataFrame(index=ct_str)
    summary_ct_df['numbers'] = ct_groups.size()
    for key in param_list:
        summary_ct_df[f'{key} mean'] = ct_groups[key].mean()
        summary_ct_df[f'{key} std'] = ct_groups[key].std()
        summary_ct_df[f'{key} median'] = ct_groups[key].median()
    summary_ct_df.to_csv(f'{f_name}/summary_params_ct.csv')
    # kruskal wallis test to get statistics over different celltypes
    kruskal_results_df = pd.DataFrame(columns=['stats', 'p-value'], index=param_list)
    # get kruskal for all syn sizes between groups
    # also get ranksum results between celltyes
    group_comps = list(combinations(ct_str, 2))
    ranksum_columns = [f'{gc[0]} vs {gc[1]}' for gc in group_comps]
    ranksum_df = pd.DataFrame(columns=ranksum_columns)
    for key in param_list:
        key_groups = [group[key].values for name, group in
                            param_df.groupby('celltype')]
        kruskal_res = kruskal(*key_groups, nan_policy='omit')
        kruskal_results_df.loc[key, 'stats'] = kruskal_res[0]
        kruskal_results_df.loc[key, 'p-value'] = kruskal_res[1]
        for group in group_comps:
            ranksum_res = ranksums(ct_groups.get_group(group[0])[key], ct_groups.get_group(group[1])[key])
            ranksum_df.loc[f'{key} stats', f'{group[0]} vs {group[1]}'] = ranksum_res[0]
            ranksum_df.loc[f'{key} p-value', f'{group[0]} vs {group[1]}'] = ranksum_res[1]

    kruskal_results_df.to_csv(f'{f_name}/kruskal_results.csv')
    ranksum_df.to_csv(f'{f_name}/ranksum_results.csv')

    #get overview params

    
    log.info(f'Step 4/4 PCA with {n_comps_PCA} components')
    pca_params = ['axon median radius', 'axon mitochondria volume density', 'spine density', 'soma diameter']
    unique_groups = np.unique(param_df['celltype'])
    log.info(f'apply PCA to the following features {pca_params} and these groups {unique_groups}')
    # code based on chatgpt, used in PCA_UMAP_morph_cts
    features = param_df[pca_params]
    labels = param_df['celltype']
    # standardize features for PCA
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features)
    pca = PCA(n_components=n_comps_PCA)
    principal_components = pca.fit_transform(features_standardized)
    # new dataframe with principal components and labels
    if n_comps_PCA == 1:
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1'])
        pca_df['celltype'] = labels
        pca_df.to_csv(f'{f_name}/pca_principal_component.csv')
        log.info('Step 3/3: Plot results')
        plot_histogram_selection(dataframe=pca_df, x_data='PC1',
                                 color_palette=ct_palette, label='pca_one_comp', count='cells', foldername=f_name,
                                 hue_data='celltype', title=f'Separation by first principal component',
                                 fontsize=fontsize_jointplot)
    elif n_comps_PCA == 2:
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        pca_df['celltype'] = labels
        pca_df.to_csv(f'{f_name}/pca_two_principal_components.csv')

        log.info('Step 3/3: Plot results')
        g = sns.JointGrid(data=pca_df, x='PC1', y='PC2', hue="celltype", palette=ct_palette)
        g.plot_joint(sns.scatterplot)
        g.plot_marginals(sns.histplot, fill=True, alpha=0.3,
                         kde=False, bins=10, palette=ct_palette)
        g.ax_joint.set_xticklabels(["%.2f" % i for i in g.ax_joint.get_xticks()], fontsize=fontsize_jointplot)
        g.ax_joint.set_yticklabels(["%.2f" % i for i in g.ax_joint.get_yticks()], fontsize=fontsize_jointplot)
        plt.savefig(f"{f_name}/pca_joinplot_celltypes.svg")
        plt.savefig(f"{f_name}/pca_joinplot_celltypes.png")
        plt.close()

    log.info(f'{cts_str} morphology analysis is done')