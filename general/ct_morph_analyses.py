#write analyses to check several parameters for full cells
#compare celltypes based on morphology

if __name__ == '__main__':
    from analysis_morph_helper import check_comp_lengths_ct
    from analysis_colors import CelltypeColors
    from analysis_morph_helper import check_comp_lengths_ct, get_spine_density, get_cell_soma_radius, get_myelin_fraction, get_median_comp_radii_cell
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from analysis_params import Analysis_Params
    import os as os
    from syconn.reps.super_segmentation import SuperSegmentationObject
    import pandas as pd
    import numpy as np
    from syconn.mp.mp_utils import start_multiprocess_imap
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import ranksums, kruskal
    from itertools import combinations
    from tqdm import tqdm
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import umap

    #ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
     #          10: "NGF"}
    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    global_params.wd = analysis_params.working_dir()
    with_glia = False
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    full_cells_only = False
    axon_only = True
    min_comp_len_cell = 200
    min_comp_len_ax = 200
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw', 'STNGPINTv6', 'AxTePkBrv6', 'TePkBrNGF', 'TeBKv6MSNyw'
    color_key = 'AxTePkBrv6'
    fontsize = 20
    n_comps_PCA = 2
    n_umap_runs = 5
    process_morph_parameters = False
    use_mito_density = True
    use_vc_density = False
    use_ves_density = True
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/general/240422_j0251{version}_ct_morph_analyses_mcl_%i_ax%i_%s_fs%i" \
             f"npca{n_comps_PCA}_umap{n_umap_runs}_axonly_axmives" % (
        min_comp_len_cell, min_comp_len_ax, color_key, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('ct_morph_analyses', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i for full cells, min_comp_len = %i for axons, colors = %s" % (
            min_comp_len_cell, min_comp_len_ax, color_key))
    log.info('use vertex label dict predictions to get soma vertices')
    log.info('Median of coords used to get soma centre')
    log.info(f' Process morphological parameters = {process_morph_parameters}, use mito volume density = {use_mito_density},'
             f' use vc volume density = {use_vc_density}, use ves density = {use_ves_density}')
    if full_cells_only:
        log.info('Plot for full cells only')
    known_mergers = analysis_params.load_known_mergers()
    misclassified_asto_ids = analysis_params.load_potential_astros()
    axon_cts = analysis_params.axon_cts()
    num_cts = analysis_params.num_cts(with_glia=with_glia)
    np_presaved_loc = analysis_params.file_locations
    if full_cells_only:
        ct_types = analysis_params.load_celltypes_full_cells()
        #ct_types = ct_types[1:]

    else:
        ct_types = np.arange(0, num_cts)
    full_ct_types = ct_types[np.in1d(ct_types, axon_cts) == False]
    ct_str_list = analysis_params.ct_str(with_glia=with_glia)
    cls = CelltypeColors(ct_dict= ct_dict)
    ct_palette = cls.ct_palette(key = color_key)
    if axon_only:
        ct_types = axon_cts

    log.info('Step 1/4: Iterate over each celltypes check min length')
    suitable_ids_dict = {}
    all_suitable_ids = []
    all_cell_dict = {}
    all_celltypes = []
    all_celltypes_num = []
    for ct in ct_types:
        # only get cells with min_comp_len, MSN with max_comp_len or axons with min ax_len
        ct_str = ct_dict[ct]
        cell_dict = analysis_params.load_cell_dict(ct)
        all_cell_dict[ct] = cell_dict
        cellids = np.array(list(cell_dict.keys()))
        merger_inds = np.in1d(cellids, known_mergers) == False
        cellids = cellids[merger_inds]
        if ct in axon_cts:
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_ax,
                                            axon_only=True, max_path_len=None)
        else:
            astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
            cellids = cellids[astro_inds]
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cell,
                                                axon_only=False, max_path_len=None)
        cellids = np.sort(cellids)
        suitable_ids_dict[ct] = cellids
        all_suitable_ids.append(cellids)
        all_celltypes.append([ct_dict[ct] for i in cellids])
        all_celltypes_num.append([[ct] for i in cellids])
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict[ct]))

    all_suitable_ids = np.concatenate(all_suitable_ids)
    all_celltypes = np.concatenate(all_celltypes)
    all_celltypes_num = np.concatenate(all_celltypes_num)
    sorted_suitable_ids = np.sort(all_suitable_ids)

    if axon_only:
        columns = ['cellid', 'celltype', 'axon length', 'axon median radius', 'axon myelin fraction', 'axon surface area']
    else:
        columns = ['cellid', 'celltype', 'soma diameter', 'spine density', 'axon length', 'dendrite length',
                   'axon median radius', 'dendrite median radius','soma surface area',
                   'axon surface area', 'dendrite surface area', 'axon myelin fraction', 'cell volume']
    morph_df = pd.DataFrame(columns=columns, index=range(len(all_suitable_ids)))
    morph_df['cellid'] = all_suitable_ids
    morph_df['celltype'] = all_celltypes
    param_list = columns[2:]

    if process_morph_parameters:
        log.info('Step 2/8: Get compartment surface area and length for cells, get compartment length, spine density, cell volume')
        #get only axon and not dendrite length as use dendrite length without spines for this value for
        #better comparison
        for ct in ct_types:
            ct_ids = suitable_ids_dict[ct]
            cell_dict = all_cell_dict[ct]
            if ct in axon_cts:
                for cellid in tqdm(ct_ids):
                    id_ind = np.where(morph_df['cellid'] == cellid)[0]
                    morph_df.loc[id_ind, 'axon length'] = cell_dict[cellid]['axon length']
                    morph_df.loc[id_ind, 'axon surface area'] = cell_dict[cellid]['axon mesh surface area']
            else:
                for cellid in tqdm(ct_ids):
                    id_ind = np.where(morph_df['cellid'] == cellid)[0]
                    morph_df.loc[id_ind, 'axon length'] = cell_dict[cellid]['axon length']
                    morph_df.loc[id_ind, 'axon surface area'] = cell_dict[cellid]['axon mesh surface area']
                    morph_df.loc[id_ind, 'dendrite surface area'] = cell_dict[cellid]['dendrite mesh surface area']
                    morph_df.loc[id_ind, 'soma surface area'] = cell_dict[cellid]['soma mesh surface area']
                    cell = SuperSegmentationObject(cellid)
                    cell_volume = np.abs(cell.size * np.prod(cell.scaling) * 10**(-9)) #in µm³
                    morph_df.loc[id_ind, 'cell volume'] = cell_volume
                ct_inds = morph_df['celltype'] == ct_dict[ct]
                cell_input = [[cell_id, min_comp_len_cell, cell_dict[cell_id]] for cell_id in ct_ids]
                spine_density_res = start_multiprocess_imap(get_spine_density, cell_input)
                spine_density_res = np.array(spine_density_res, dtype='object')
                spine_density = spine_density_res[:, 0]
                no_spine_den_length = spine_density_res[:, 1]
                morph_df.loc[ct_inds, 'spine density'] = spine_density
                morph_df.loc[ct_inds, 'dendrite length'] = no_spine_den_length

        log.info('Step 3/9: Get axon and dendrite median radius from cell, remove spines from dendrite for that')
        if not full_cells_only:
            axon_inds = np.in1d(all_celltypes_num, axon_cts)
            axon_ids = all_suitable_ids[axon_inds]
            #cellid, only_axon, no_spine
            cell_input = [[cell_id, True, True] for cell_id in axon_ids]
            ct_radii_res = start_multiprocess_imap(get_median_comp_radii_cell, cell_input)
            ct_radii_res = np.array(ct_radii_res, dtype='object')
            ct_ax_median_radius = ct_radii_res[:, 0]
            morph_df.loc[axon_inds, 'axon median radius'] = ct_ax_median_radius
            full_cell_inds = np.in1d(all_celltypes_num, full_ct_types)
            full_cell_ids = all_suitable_ids[full_cell_inds]
        else:
            if not axon_only:
                full_cell_inds = np.in1d(all_celltypes_num, full_ct_types)
                full_cell_ids = all_suitable_ids
        if not axon_only:
            cell_input = [[cell_id, False, True] for cell_id in full_cell_ids]
            ct_radii_res = start_multiprocess_imap(get_median_comp_radii_cell, cell_input)
            ct_radii_res = np.array(ct_radii_res, dtype='object')
            ct_ax_median_radius = ct_radii_res[:, 0]
            ct_den_median_radius = ct_radii_res[:, 1]
            morph_df.loc[full_cell_inds, 'axon median radius'] = ct_ax_median_radius
            morph_df.loc[full_cell_inds, 'dendrite median radius'] = ct_den_median_radius

            log.info('Step 4/9: Get soma diameter for cells')
            ct_soma_results = start_multiprocess_imap(get_cell_soma_radius, full_cell_ids)
            ct_soma_results = np.array(ct_soma_results, dtype='object')
            ct_diameters = ct_soma_results[:, 1].astype(float) * 2
            morph_df.loc[full_cell_inds, 'soma diameter'] = ct_diameters

        log.info('Step 5/9: Get myelin fraction from cells')
        if not full_cells_only:
            axon_input = [[cellid, min_comp_len_ax, True, True] for cellid in axon_ids]
            ax_myelin_results = start_multiprocess_imap(get_myelin_fraction, axon_input)
            ax_myelin_results = np.array(ax_myelin_results, dtype='object')
            ax_rel_myelin = ax_myelin_results[:, 1]
            morph_df.loc[axon_inds, 'axon myelin fraction'] = ax_rel_myelin
        if not axon_only:
            cell_input = [[cellid, min_comp_len_cell, True, False] for cellid in full_cell_ids]
            ct_myelin_results = start_multiprocess_imap(get_myelin_fraction, cell_input)
            ct_myelin_results = np.array(ct_myelin_results, dtype='object')
            rel_myelin = ct_myelin_results[:, 1]
            morph_df.loc[full_cell_inds, 'axon myelin fraction'] = rel_myelin

        morph_df = morph_df.astype({param: float for param in param_list})
        #make sure no NaN values are in table for PCA; fill NaN with 0
        morph_df = morph_df.fillna(0)
        morph_df.to_csv(f'{f_name}/ct_morph_df.csv')

    else:
        morph_path = 'cajal/scratch/users/arother/bio_analysis_results/general/' \
                     '2403410_j0251v6_ct_morph_analyses_mcl_200_ax200_TePkBrNGF_fs20npca2_umap5/ct_morph_df.csv'
        log.info(f'Step 2/9: Use morphological parameters from {morph_path}')
        loaded_morph_df = pd.read_csv(morph_path, index_col = 0)
        if len(all_suitable_ids) > len(loaded_morph_df):
            raise ValueError('Not all selected cellids are part of this table with morphological parameters, please load other one')
        else:
            loaded_morph_df_sorted = loaded_morph_df.sort_values('cellid')
            suit_inds = np.in1d(loaded_morph_df_sorted['cellid'], sorted_suitable_ids)
            morph_df = loaded_morph_df_sorted.loc[suit_inds]
        axon_ct_str = [ct_dict[ct] for ct in axon_cts]
        full_ct_str = [ct_dict[ct] for ct in full_ct_types]
        axon_inds = np.in1d(morph_df['celltype'], axon_ct_str)
        full_cell_inds = np.in1d(morph_df['celltype'], full_ct_str)

    if use_mito_density:
        #calculate these values with function ct_organell_volume_density
        mito_density_path = 'cajal/scratch/users/arother/bio_analysis_results/general/' \
                     '240326_j0251v6_ct_mi_vol_density_mcl_200_ax200_TePkBrNGF_fs20/percell_df_mi_den.csv'
        log.info(f'Axon mito volume density loaded from {mito_density_path}')
        mito_den_df = pd.read_csv(mito_density_path, index_col=0)
        if len(all_suitable_ids) > len(mito_den_df):
            raise ValueError('Not all selected cellids are part of this table with mitochondria volume densities, please load other one')
        else:
            mito_den_df_sorted = mito_den_df.sort_values('cellid')
            suit_inds = np.in1d(mito_den_df_sorted['cellid'], sorted_suitable_ids)
            mito_den_df_sorted = mito_den_df_sorted.loc[suit_inds]
            #morph_df = morph_df.join(mito_den_df_sorted['total mi volume density'])
            morph_df = morph_df.join(mito_den_df_sorted['axon mi volume density'])
            #param_list = np.hstack([param_list, 'total mi volume density', 'axon mi volume density'])
            param_list = np.hstack([param_list, 'axon mi volume density'])

    if use_vc_density:
        # calculate these values with function ct_organell_volume_density
        vc_density_path = 'cajal/scratch/users/arother/bio_analysis_results/general/' \
                     '240320_j0251v6_ct_vc_vol_density_mcl_200_ax200_TePkBrNGF_fs20/percell_df_vc_den.csv'
        log.info(f'Axon vc volume density loaded from {vc_density_path}')
        vc_den_df = pd.read_csv(vc_density_path, index_col=0)
        if len(all_suitable_ids) > len(vc_den_df):
            raise ValueError('Not all selected cellids are part of this table with vesicle cloud volume densities, please load other one')
        else:
            vc_den_df_sorted = vc_den_df.sort_values('cellid')
            suit_inds = np.in1d(vc_den_df_sorted['cellid'], sorted_suitable_ids)
            vc_den_df_sorted = vc_den_df_sorted.loc[suit_inds]
            morph_df = morph_df.join(vc_den_df_sorted['axon vc volume density'])
            param_list = np.hstack([param_list, 'axon vc volume density'])

    if use_ves_density:
        #calculate density with function single_vesicle_analysis/ct_ves_density
        ves_density_path = 'cajal/scratch/users/arother/bio_analysis_results/single_vesicle_analysis/' \
                     '240319_j0251v6_ct_vesicle_density_mcl_200_ax200_TePkBrNGF_fs20/percell_df_ves_den.csv'
        log.info(f'Axon vesicle density loaded from {ves_density_path}')
        ves_den_df = pd.read_csv(ves_density_path, index_col=0)
        if len(all_suitable_ids) > len(ves_den_df):
            raise ValueError('Not all selected cellids are part of this table with vesicle densities, please load other one')
        else:
            ves_den_df_sorted = ves_den_df.sort_values('cellid')
            suit_inds = np.in1d(ves_den_df_sorted['cellid'], sorted_suitable_ids)
            ves_den_df_sorted = ves_den_df_sorted.loc[suit_inds]
            morph_df = morph_df.join(ves_den_df_sorted['vesicle density'])
            param_list = np.hstack([param_list, 'vesicle density'])
    morph_df.to_csv(f'{f_name}/ct_morph_df.csv')

    log.info('Step 6/9: Calculate statistics and get overview params')
    #save mean, median and std for all parameters per ct
    ct_str = np.unique(morph_df['celltype'])
    ct_groups = morph_df.groupby('celltype')
    summary_ct_df = pd.DataFrame(index = ct_str)
    summary_ct_df['numbers'] = ct_groups.size()
    for key in param_list:
        summary_ct_df[f'{key} mean'] = ct_groups[key].mean()
        summary_ct_df[f'{key} std'] = ct_groups[key].std()
        summary_ct_df[f'{key} median'] = ct_groups[key].median()
    summary_ct_df.to_csv(f'{f_name}/summary_params_ct.csv')

    #do kruskal wallis and ranksum results as posthoc test
    kruskal_results_df = pd.DataFrame(columns=['stats', 'p-value'], index=param_list)
    # get kruskal for all syn sizes between groups
    # also get ranksum results between celltyes
    group_comps = list(combinations(ct_str, 2))
    ranksum_columns = [f'{gc[0]} vs {gc[1]}' for gc in group_comps]
    ranksum_df = pd.DataFrame(columns=ranksum_columns)
    for key in param_list:
        key_groups = [group[key].values for name, group in
                      morph_df.groupby('celltype')]
        kruskal_res = kruskal(*key_groups, nan_policy='omit')
        kruskal_results_df.loc[key, 'stats'] = kruskal_res[0]
        kruskal_results_df.loc[key, 'p-value'] = kruskal_res[1]
        for group in group_comps:
            ranksum_res = ranksums(ct_groups.get_group(group[0])[key], ct_groups.get_group(group[1])[key])
            ranksum_df.loc[f'{key} stats', f'{group[0]} vs {group[1]}'] = ranksum_res[0]
            ranksum_df.loc[f'{key} p-value', f'{group[0]} vs {group[1]}'] = ranksum_res[1]

    kruskal_results_df.to_csv(f'{f_name}/kruskal_results.csv')
    ranksum_df.to_csv(f'{f_name}/ranksum_results.csv')

    log.info('Step 7/9: Plot results as boxplot')
    for key in param_list:
        # plot with increasing median as boxplot and violinplot
        if 'length' in key or 'diameter' in key or 'radius' in key:
            ylabel = f'{key} [µm]'
        elif 'spine density' in key or 'vesicle density' in key:
            ylabel = f'{key} [1/µm]'
        elif 'area' in key:
            ylabel = f'{key} [µm²]'
        elif 'volume' in key:
            ylabel = f'{key} [µm³]'
        elif 'fraction' in key:
            ylabel = f'{key}'
        else:
            raise ValueError(f'No units were defined for this parameter: {key}')
        sns.boxplot(data=morph_df, x='celltype', y=key, palette=ct_palette)
        plt.title(key)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.xlabel('celltype', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(f'{f_name}/{key}_box.png')
        plt.savefig(f'{f_name}/{key}_box.svg')
        plt.close()
        if full_cells_only:
            sns.stripplot(data=morph_df, x='celltype', y=key, color='black', alpha=0.2,
                          dodge=True, size=2)
        sns.violinplot(data=morph_df, x='celltype', y=key, palette=ct_palette, inner="box")
        plt.title(key)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.xlabel('celltype', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(f'{f_name}/{key}_violin.png')
        plt.savefig(f'{f_name}/{key}_violin.svg')
        plt.close()

    #morph_df = morph_df[morph_df['celltype'] == 'MSN']

    log.info('Step 8/9: Get PCA of results')
    log.info(f'Number of components for PCA = {n_comps_PCA}')
    #get one PCA for proj axons
    #use same code as in morph_comp_PCA
    # standardize features for PCA (will also be used for UMAP)
    #this means each feature will be converted into a z_score (number of std from mean)
    # for comparability (source umap-learn.readthedocs.io(en/latest/basic_usage.html)
    scaler = StandardScaler()
    if not full_cells_only:
        axon_params = ['axon length', 'axon myelin fraction', 'axon surface area', 'axon median radius']
        if use_mito_density:
            axon_params = np.hstack([axon_params, 'axon mi volume density'])
        if use_vc_density:
            axon_params = np.hstack([axon_params, 'axon vc volume density'])
        if use_ves_density:
            axon_params = np.hstack([axon_params, 'vesicle density'])
        log.info(f'Params for PCA for projecting axon celltypes: {axon_params}')
        axon_morph_df = morph_df.loc[axon_inds]
        #only take non-NaN values with parameters needed for axon, fill with 0
        ax_features = axon_morph_df[axon_params]
        ax_labels = axon_morph_df['celltype']
        ax_features_standardized = scaler.fit_transform(ax_features)
        pca = PCA(n_components=n_comps_PCA)
        principal_components = pca.fit_transform(ax_features_standardized)
        # new dataframe with principal components and labels
        if n_comps_PCA == 1:
            pca_df = pd.DataFrame(data=principal_components, columns=['PC1'])
            pca_df['celltype'] = ax_labels
            pca_df.to_csv(f'{f_name}/pca_proj_axons_principal_component.csv')
            log.info('Step 3/3: Plot results')
            sns.histplot(x='PC1', data=pca_df, hue='celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True)
            plt.ylabel(f'number of axons', fontsize=fontsize)
            plt.xlabel('PC1', fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.title('PCA of axons')
            plt.savefig(f'{f_name}/pca_ax_hist.png')
            plt.savefig(f'{f_name}/pca_ax_hist.svg')
            plt.close()
            sns.histplot(x='PC1', data=pca_df, hue='celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True, stat='percent')
            plt.ylabel(f'% of axons', fontsize=fontsize)
            plt.xlabel('PC1', fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.title('PCA of axons')
            plt.savefig(f'{f_name}/pca_ax_hist_perc.png')
            plt.savefig(f'{f_name}/pca_ax_hist_perc.svg')
            plt.close()
        elif n_comps_PCA == 2:
            pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
            pca_df['celltype'] = ax_labels
            pca_df.to_csv(f'{f_name}/pca_proj_axons_two_principal_components.csv')

            log.info('Step 3/3: Plot results')
            g = sns.JointGrid(data=pca_df, x='PC1', y='PC2', hue="celltype", palette=ct_palette)
            g.plot_joint(sns.scatterplot)
            g.plot_marginals(sns.histplot, fill=True, alpha=0.3,
                             kde=False, bins=10, palette=ct_palette)
            g.ax_joint.set_xticklabels(["%.2f" % i for i in g.ax_joint.get_xticks()], fontsize=fontsize)
            g.ax_joint.set_yticklabels(["%.2f" % i for i in g.ax_joint.get_yticks()], fontsize=fontsize)
            plt.savefig(f"{f_name}/pca_joinplot_ax_celltypes.svg")
            plt.savefig(f"{f_name}/pca_joinplot_ax_celltypes.png")
            plt.close()
    if not axon_only:
        log.info(f'Params for PCA for full celltypes: {param_list}')
        if full_cells_only:
            full_cell_df = morph_df
        else:
            full_cell_df = morph_df.loc[full_cell_inds]
        fc_features = full_cell_df[param_list]
        fc_labels = full_cell_df['celltype']
        # standardize features for PCA
        fc_features_standardized = scaler.fit_transform(fc_features)
        pca = PCA(n_components=n_comps_PCA)
        principal_components = pca.fit_transform(fc_features_standardized)
        # new dataframe with principal components and labels
        if n_comps_PCA == 1:
            pca_df = pd.DataFrame(data=principal_components, columns=['PC1'])
            pca_df['celltype'] = fc_labels
            pca_df.to_csv(f'{f_name}/pca_full_cells_principal_component.csv')
            log.info('Step 3/3: Plot results')
            sns.histplot(x='PC1', data=pca_df, hue='celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True)
            plt.ylabel(f'number of cells', fontsize=fontsize)
            plt.xlabel('PC1', fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.title('PCA of cells')
            plt.savefig(f'{f_name}/pca_fc_hist.png')
            plt.savefig(f'{f_name}/pca_fc_hist.svg')
            plt.close()
            sns.histplot(x='PC1', data=pca_df, hue='celltype', palette=ct_palette, common_norm=False,
                         fill=False, element="step", linewidth=3, legend=True, stat='percent')
            plt.ylabel(f'% of cells', fontsize=fontsize)
            plt.xlabel('PC1', fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.title('PCA of cells')
            plt.savefig(f'{f_name}/pca_fc_hist_perc.png')
            plt.savefig(f'{f_name}/pca_fc_hist_perc.svg')
            plt.close()
        elif n_comps_PCA == 2:
            pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
            pca_df['celltype'] = fc_labels
            pca_df.to_csv(f'{f_name}/pca_full_cells_two_principal_components.csv')

            log.info('Step 3/3: Plot results')
            g = sns.JointGrid(data=pca_df, x='PC1', y='PC2', hue="celltype", palette=ct_palette)
            g.plot_joint(sns.scatterplot)
            g.plot_marginals(sns.histplot, fill=True, alpha=0.3,
                             kde=False, bins=10, palette=ct_palette)
            g.ax_joint.set_xticklabels(["%.2f" % i for i in g.ax_joint.get_xticks()], fontsize=fontsize)
            g.ax_joint.set_yticklabels(["%.2f" % i for i in g.ax_joint.get_yticks()], fontsize=fontsize)
            plt.savefig(f"{f_name}/pca_joinplot_fc_celltypes.svg")
            plt.savefig(f"{f_name}/pca_joinplot_fc_celltypes.png")
            plt.close()

    log.info('Step 9/9: Get umap of results')
    #seed umap analyses
    np.random.seed(42)
    #same as above make one umap on only axon types and one on full cells
    log.info(f'UMAP will be run {n_umap_runs} times')
    #code inspired by Chatgpt4
    if not axon_only:
        for i in range(n_umap_runs + 1):
            fc_reducer = umap.UMAP()
            fc_embedding = fc_reducer.fit_transform(fc_features_standardized)
            umap_df = pd.DataFrame(columns = ['cellid', 'celltype', 'UMAP 1', 'UMAP 2'])
            umap_df['cellid'] = full_cell_df['cellid']
            umap_df['celltype'] = full_cell_df['celltype']
            umap_df['UMAP 1'] = fc_embedding[:, 0]
            umap_df['UMAP 2'] = fc_embedding[:, 1]
            umap_df.to_csv(f'{f_name}/fc_umap_embeddings_{i}.csv')
            for label in set(fc_labels):
                plt.scatter(fc_embedding[fc_labels == label, 0], fc_embedding[fc_labels == label, 1], label=label, s=10, color = ct_palette[label])
            plt.title('UMAP Visualization of full cells')
            plt.xlabel('UMAP 1', fontsize=fontsize)
            plt.ylabel('UMAP 2', fontsize=fontsize)
            plt.legend()
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.savefig(f'{f_name}/fc_umap_{i}.png')
            plt.savefig(f'{f_name}/fc_umap_{i}.svg')
            plt.close()
    if not full_cells_only:
        for i in range(n_umap_runs + 1):
            ax_reducer = umap.UMAP()
            ax_embedding = ax_reducer.fit_transform(ax_features_standardized)
            umap_df = pd.DataFrame(columns=['cellid', 'celltype', 'UMAP 1', 'UMAP 2'])
            umap_df['cellid'] = axon_morph_df['cellid']
            umap_df['celltype'] = axon_morph_df['celltype']
            umap_df['UMAP 1'] = ax_embedding[:, 0]
            umap_df['UMAP 2'] = ax_embedding[:, 1]
            umap_df.to_csv(f'{f_name}/ax_umap_embeddings_{i}')
            for label in set(ax_labels):
                plt.scatter(ax_embedding[ax_labels == label, 0], ax_embedding[ax_labels == label, 1], label=label, s=10, color = ct_palette[label])
            plt.title('UMAP Visualization of projecting axons')
            plt.xlabel('UMAP 1', fontsize = fontsize)
            plt.ylabel('UMAP 2', fontsize = fontsize)
            plt.legend()
            plt.savefig(f'{f_name}/ax_umap_{i}.png')
            plt.savefig(f'{f_name}/ax_umap_{i}.svg')
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.close()
    log.info('Morphological analyses of celltypes done.')




