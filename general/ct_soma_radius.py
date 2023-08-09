#get radius of soma for all celltypes

if __name__ == '__main__':
    from analysis_morph_helper import check_comp_lengths_ct, get_cell_soma_radius
    from analysis_colors import CelltypeColors
    from analysis_params import Analysis_Params
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    import os as os
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    import seaborn as sns
    import matplotlib.pyplot as plt
    from syconn.mp.mp_utils import start_multiprocess_imap
    from itertools import combinations
    from scipy.stats import kruskal, ranksums

    global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    analysis_params = Analysis_Params(working_dir=global_params.wd, version='v5')
    ct_dict = analysis_params.ct_dict(with_glia=False)
    celltype_key = analysis_params.celltype_key()
    min_comp_len_cells = 200
    exclude_known_mergers = True
    cls = CelltypeColors()
    fontsize = 12
    save_svg = False
    use_skel = False #if true would use skeleton labels for getting soma; vertex labels more exact
    #color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw', 'STNGP'}
    color_key = 'STNGP'
    f_name = "cajal/scratch/users/arother/bio_analysis_results/general/230809_j0251v5_cts_soma_radius_mcl_%i_%s_fs%i" % (
    min_comp_len_cells,color_key, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    save_svg = True
    log = initialize_logging('soma radius calculation', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i for full cells, known mergers excluded = %s, colors = %s, fonsize = %i" % (
        min_comp_len_cells,exclude_known_mergers, color_key, fontsize))
    if use_skel:
        log.info('use skeleton node predictions to get soma mesh coordinates')
    else:
        log.info('use vertex label dict predictions to get soma vertices')
    time_stamps = [time.time()]
    step_idents = ['t-0']

    #filter cells for min_comp_len
    log.info("Step 1/4: Load cell dicts and get suitable cellids")
    if exclude_known_mergers:
        known_mergers = analysis_params.load_known_mergers()
    # To Do: also exlclude MSNs from list
    celltypes = analysis_params.load_celltypes_full_cells(with_glia=False)
    num_cts = len(celltypes)
    full_cell_dicts = {}
    suitable_ids_dict = {}
    all_suitable_ids = []
    all_suitable_ids_cts = []
    for ct in tqdm(celltypes):
        ct_str = ct_dict[ct]
        cell_dict = analysis_params.load_cell_dict(ct)
        cellids = np.array(list(cell_dict.keys()))
        if exclude_known_mergers:
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
        if ct == 2:
            misclassified_asto_ids = analysis_params.load_potential_astros()
            astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
            cellids = cellids[astro_inds]
        cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict,
                                                min_comp_len=min_comp_len_cells,
                                                axon_only=False,
                                                max_path_len=None)
        full_cell_dicts[ct] = cell_dict
        suitable_ids_dict[ct] = cellids_checked
        all_suitable_ids.append(cellids_checked)
        all_suitable_ids_cts.append([ct_str for i in range(len(cellids_checked))])

    all_suitable_ids = np.hstack(all_suitable_ids)
    all_suitable_ids_cts = np.hstack(all_suitable_ids_cts)

    log.info('Step 2/4: Get soma radius from all cellids')
    columns = ['cellid', 'celltype', 'soma radius', 'soma diameter', 'soma centre voxel x', 'soma centre voxel y', 'soma centre voxel z']
    soma_results_pd = pd.DataFrame(columns=columns, index = range(len(all_suitable_ids)))
    soma_results_pd['cellid'] = all_suitable_ids
    soma_results_pd['celltype'] = all_suitable_ids_cts
    output = start_multiprocess_imap(get_cell_soma_radius, all_suitable_ids)
    output = np.array(output, dtype='object')
    soma_centres = np.concatenate(output[:, 0]).reshape(len(output), 3)
    soma_centres_vox = soma_centres / [10, 10, 25]
    soma_radii = output[:, 1].astype(float)
    soma_diameters = soma_radii * 2
    soma_results_pd['soma centre voxel x'] = soma_centres_vox[:, 0].astype(int)
    soma_results_pd['soma centre voxel y'] = soma_centres_vox[:, 1].astype(int)
    soma_results_pd['soma centre voxel z'] = soma_centres_vox[:, 2].astype(int)
    soma_results_pd['soma radius'] = soma_radii
    soma_results_pd['soma diameter'] = soma_diameters
    soma_results_pd= soma_results_pd.round(2)
    len_before_drop = len(soma_results_pd)
    cellids_before_drop = soma_results_pd['cellid']
    soma_results_pd = soma_results_pd.dropna()
    len_after_drop = len(soma_results_pd)
    cellids_after_drop = soma_results_pd['cellid']
    cellids_no_soma = cellids_before_drop[np.in1d(cellids_before_drop, cellids_after_drop) == False]
    soma_results_pd.to_csv(f'{f_name}/soma_radius_results.csv')
    log.info(f'{len_before_drop - len_after_drop} cells had to be excluded for '
             f'missing soma mesh(id: {cellids_no_soma})')


    log.info('Step 3/4: Plot results')
    ct_colours = cls.colors[color_key]
    ct_palette = cls.ct_palette(color_key, num=False)
    #make box-and violinplot as overview
    ylabel = 'soma diameter [µm]'
    title = 'soma diameter of different celltypes'
    sns.stripplot(x = 'celltype', y = 'soma diameter', data=soma_results_pd, color='black', alpha=0.2,
                  dodge=True, size=2)
    sns.violinplot(x = 'celltype', y = 'soma diameter', data=soma_results_pd, inner="box",
                   palette=ct_palette)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/soma_diameter_celltypes_violin.png')
    if save_svg:
        plt.savefig(f'{f_name}/soma_diameter_celltypes_violin.svg')
    plt.close()
    sns.boxplot(x = 'celltype', y = 'soma diameter', data=soma_results_pd, palette=ct_palette)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(f'{f_name}/soma_diameter_celltypes_box.png')
    if save_svg:
        plt.savefig(f'{f_name}/soma_diameter_celltypes_box.svg')
    plt.close()
    #make individual histogram plots for each celltype
    xhist = 'soma diameter [µm]'
    for ct in celltypes:
        ct_str = ct_dict[ct]
        # make individual histogram plots for each celltype to see eventual bimodal distributions
        data = soma_results_pd[soma_results_pd['celltype'] == ct_str]
        sns.histplot(x = 'soma diameter', data = data, color=ct_palette[ct_str], common_norm=False,
                     fill=False,element="step", linewidth=3)
        plt.ylabel('count of cells')
        plt.xlabel(xhist)
        plt.title(f'Soma diameter for {ct_str}')
        plt.savefig(f'{f_name}/{ct_str}_soma_diameter_hist.png')
        plt.close()
        sns.histplot(x='soma diameter', data=data, color=ct_palette[ct_str], common_norm=True,
                     fill=False,element="step", linewidth=3
                     )
        plt.ylabel('fraction of cells')
        plt.xlabel(xhist)
        plt.title(f'Soma diameter for {ct_str}')
        plt.savefig(f'{f_name}/{ct_str}_soma_diameter_hist_norm.png')
        plt.close()

    log.info('Step 4/4: run statistical tests')
    celltype_diameter_groups = [group['soma diameter'].values for name, group in soma_results_pd.groupby('celltype')]
    celltype_groups = soma_results_pd.groupby('celltype')
    celltype_diameter_median = celltype_groups['soma diameter'].median()
    celltype_diameter_median.to_csv(f'{f_name}/median_diameters_celltype.csv')
    ind_celltypes_mapped = {ct: i for i, ct in enumerate(celltypes)}
    #run kruskal wallis test
    kruskal_results = kruskal(*celltype_diameter_groups)
    log.info(f'Results of kruskal-wallis-test: p-value = {kruskal_results[1]:.2f}, stats: {kruskal_results[0]:.2f}')
    #run ranksum test on each combination
    celltype_combs = combinations(celltypes, 2)
    comb_list = list(celltype_combs)
    comb_list_str = [f'{ct_dict[ct1]} vs {ct_dict[ct2]}' for (ct1, ct2) in comb_list]
    ranksum_results = pd.DataFrame(columns=comb_list_str, index = ['stats', 'p-value'])
    for comb in comb_list:
        ct1 = comb[0]
        ct2 = comb[1]
        ct1_data = celltype_diameter_groups[ind_celltypes_mapped[ct1]]
        ct2_data = celltype_diameter_groups[ind_celltypes_mapped[ct2]]
        stats, p_value = ranksums(ct1_data, ct2_data)
        ranksum_results.loc['stats', f'{ct_dict[ct1]} vs {ct_dict[ct2]}'] = stats
        ranksum_results.loc['p-value', f'{ct_dict[ct1]} vs {ct_dict[ct2]}'] = p_value

    ranksum_results.to_csv(f'{f_name}/diameter_ranksum_results.csv')

    log.info('soma diameter analysis for all celltypes done')




