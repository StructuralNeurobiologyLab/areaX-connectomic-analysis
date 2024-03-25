#get surface area of dendrite and soma for full cells
#save in pandas dataframe

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import get_percell_organell_volume_density, get_organelle_comp_density_presaved
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    from syconn.reps.segmentation import SegmentationDataset
    import pandas as pd
    import numpy as np
    from syconn.mp.mp_utils import start_multiprocess_imap
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import ranksums, kruskal, spearmanr
    from itertools import combinations
    #from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm
    from tqdm import tqdm

    #ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
     #          10: "NGF"}
    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    global_params.wd = analysis_params.working_dir()
    with_glia = False
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    full_cells_only = False
    min_comp_len_cell = 200
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw'}
    color_key = 'TePkBrNGF'
    fontsize = 20
    #organelles = 'mi', 'vc', 'er', 'golgi
    organelle_key = 'mi'
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/gain_estimation/240325_j0251{version}_ct_den_so_surface_area_mcl_%i_%s_fs%i" % (
        min_comp_len_cell, color_key, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('denso_surface_area_ct_log', log_dir=f_name + '/logs/')
    log.info('get dendritic and somatic surface area per cell and celltype')
    log.info(
        "min_comp_len = %i for full cells, colors = %s" % (
            min_comp_len_cell, color_key))
    log.info(f'use mean of {organelle_key} volume density for regression fit')
    if full_cells_only:
        log.info('Plot for full cells only')
    known_mergers = analysis_params.load_known_mergers()
    misclassified_asto_ids = analysis_params.load_potential_astros()
    axon_cts = analysis_params.axon_cts()
    num_cts = analysis_params.num_cts(with_glia=with_glia)
    ct_types = analysis_params.load_celltypes_full_cells()
    ct_str_list = [ct_dict[ct] for ct in ct_types]
    cls = CelltypeColors(ct_dict= ct_dict)
    ct_palette = cls.ct_palette(key=color_key)
    #ct_palette = cls.ct_palette(key = color_key)
    if with_glia:
        glia_cts = analysis_params._glia_cts

    log.info('Step 1/4: Iterate over each celltypes check min length')
    suitable_ids_dict = {}
    all_suitable_ids = []
    all_cell_dict = {}
    all_celltypes = []
    all_ct_nums = []
    for ct in ct_types:
        # only get cells with min_comp_len, MSN with max_comp_len or axons with min ax_len
        ct_str = ct_dict[ct]
        cell_dict = analysis_params.load_cell_dict(ct)
        all_cell_dict[ct] = cell_dict
        cellids = np.array(list(cell_dict.keys()))
        merger_inds = np.in1d(cellids, known_mergers) == False
        cellids = cellids[merger_inds]
        astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
        cellids = cellids[astro_inds]
        cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cell,
                                            axon_only=False, max_path_len=None)
        cellids = np.sort(cellids)
        suitable_ids_dict[ct] = cellids
        all_suitable_ids.append(cellids)
        all_celltypes.append([ct_dict[ct] for i in cellids])
        all_ct_nums.append([ct for i in cellids])
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict[ct]))

    all_suitable_ids = np.concatenate(all_suitable_ids)
    all_celltypes = np.concatenate(all_celltypes)
    all_ct_nums = np.concatenate(all_ct_nums)
    number_cells = len(all_suitable_ids)
    params_df = pd.DataFrame(columns=['cellid', 'celltype', 'dendritic surface area', 'somatic surface area', 'denso surface area'], index=range(number_cells))
    params_df['cellid'] = all_suitable_ids
    params_df['celltype'] = all_celltypes

    log.info('Step 2/4: Get surface area for all cells')
    for i, cellid in enumerate(tqdm(all_suitable_ids)):
        celltype = all_ct_nums[i]
        cellid_dict = all_cell_dict[celltype][cellid]
        #meshes are calculated in µm²
        den_mesh_area = cellid_dict["dendrite mesh surface area"]
        soma_mesh_area = cellid_dict["soma mesh surface area"]
        params_df.loc[i, 'dendritic surface area'] = den_mesh_area
        params_df.loc[i, 'somatic surface area'] = soma_mesh_area
        params_df.loc[i, 'denso surface area'] = den_mesh_area + soma_mesh_area

    params_df = params_df.astype({'dendritic surface area': float, 'somatic surface area': float, 'denso surface area': float})
    params_df.to_csv(f'{f_name}/den_so_surface_area.csv')

    log.info('Step 3/4: get statistics and overview values per ct')
    ct_groups = [group['denso surface area'].values for name, group in
                      params_df.groupby('celltype')]
    kruskal_res = kruskal(*ct_groups, nan_policy='omit')
    log.info(f'Kruskal results: stats = {kruskal_res[0]}, p-value = {kruskal_res[1]}')
    if kruskal_res[1] < 0.05:
        group_comps = list(combinations(range(len(ct_str_list)), 2))
        ranksum_columns = [f'{ct_str_list[gc[0]]} vs {ct_str_list[gc[1]]}' for gc in group_comps]
        ranksum_res_df = pd.DataFrame(columns=ranksum_columns, index=['stats', 'p-value'])
        for gc in group_comps:
            ranksum_res = ranksums(ct_groups[gc[0]], ct_groups[gc[1]])
            ranksum_res_df.loc[f'stats', f'{ct_str_list[gc[0]]} vs {ct_str_list[gc[1]]}'] = ranksum_res[0]
            ranksum_res_df.loc[f'p-value', f'{ct_str_list[gc[0]]} vs {ct_str_list[gc[1]]}'] = ranksum_res[1]
        ranksum_res_df.to_csv(f'{f_name}/ranksum_results.csv')

    overview_df = pd.DataFrame(columns = ['number of cells', 'mean denso surface area', 'median denso surface area', 'std denso surface area'])
    ct_groups_params = params_df.groupby('celltype')
    overview_df['number of cells'] = ct_groups_params.size()
    overview_df['mean denso surface area'] = ct_groups_params['denso surface area'].mean()
    overview_df['median denso surface area'] = ct_groups_params['denso surface area'].median()
    overview_df['std denso surface area'] = ct_groups_params['denso surface area'].std()
    overview_df.to_csv(f'{f_name}/overview_params.csv')

    log.info('Step 4/4: Plot results')
    for key in list(params_df.keys()):
        if 'cellid' in key or 'celltype' in key:
            continue
        sns.boxplot(data=params_df, x='celltype', y=key, palette=ct_palette)
        plt.title(key)
        plt.ylabel(f'{key} [µm²]', fontsize=fontsize)
        plt.xlabel('celltype', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(f'{f_name}/{key}_box.png')
        plt.savefig(f'{f_name}/{key}_box.svg')
        plt.close()
        sns.stripplot(data=params_df, x='celltype', y=key, color='black', alpha=0.2,
                      dodge=True, size=2)
        sns.violinplot(data=params_df, x='celltype', y=key, palette=ct_palette, inner="box")
        plt.title(key)
        plt.ylabel(f'{key} [µm²]', fontsize=fontsize)
        plt.xlabel('celltype', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(f'{f_name}/{key}_violin.png')
        plt.savefig(f'{f_name}/{key}_violin.svg')
        plt.close()

    log.info('Analyses done')


