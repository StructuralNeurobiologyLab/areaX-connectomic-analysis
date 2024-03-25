#load single vesicle data
#filter cells for completeness
#check distribution of dist2matrix

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors, CompColors
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from cajal.nvmescratch.users.arother.bio_analysis.general.vesicle_helper import get_ves_distance_per_cell, get_ves_distance_multiple_per_cell
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    import numpy as np
    from tqdm import tqdm
    from syconn.mp.mp_utils import start_multiprocess_imap
    import matplotlib.pyplot as plt
    import seaborn as sns

    #global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    #ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
    #           10: "NGF"}
    version = 'v6'
    analysis_params = Analysis_Params(version = version)
    global_params.wd = analysis_params.working_dir()
    ct_dict = analysis_params.ct_dict(with_glia=False)
    min_comp_len_cell = 200
    min_comp_len_ax = 200
    dist_threshold = [15, 10, 5] #nm
    #dist_threshold = 15
    cls = CelltypeColors(ct_dict = ct_dict)
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw'}
    color_key = 'TePkBrNGF'
    comp_color_key = 'TeYw'
    if type(dist_threshold) == list:
        f_name = f"cajal/scratch/users/arother/bio_analysis_results/single_vesicle_analysis/240320_j0251{version}_ct_dist2matrix_mcl_%i_ax%i_dt_%i_%i_%s_%s" % (
            min_comp_len_cell, min_comp_len_ax, dist_threshold[0], dist_threshold[1], color_key, comp_color_key)
    else:
        f_name = f"cajal/scratch/users/arother/bio_analysis_results/single_vesicle_analysis/240311_j0251{version}_ct_dist2matrix_mcl_%i_ax%i_dt_%i_%s_%s" % (
            min_comp_len_cell, min_comp_len_ax, dist_threshold, color_key, comp_color_key)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('get distribution of membrane distance for single vesicles', log_dir=f_name + '/logs/')
    log.info(
        "min_comp_len = %i for full cells, min_comp_len = %i for axons, colors = %s" % (
            min_comp_len_cell, min_comp_len_ax, color_key))
    step_idents = ['t-0']
    known_mergers = analysis_params.load_known_mergers()
    log.info("Step 1/3: Prepare empty result DataFrames")
    cache_name = analysis_params.file_locations

    ct_palette = cls.ct_palette(color_key, num=False)
    comp_cls = CompColors()
    comp_cls_list = comp_cls.colors[comp_color_key]
    cts = list(ct_dict.keys())
    ax_ct = analysis_params.axon_cts()
    num_cts = len(cts)
    cts_str = [ct_dict[i] for i in range(num_cts)]
    ves_density_all = pd.DataFrame(columns=cts_str, index=range(10500))
    combined_columns = ['vesicle density', 'celltype', 'distance threshold']
    if type(dist_threshold) == list:
        dist_str = ['all']
        for dt in dist_threshold:
            dist_str.append(str(dt) + ' nm')
    else:
        dist_str = ['all', str(dist_threshold) + ' nm']
    num_cat = len(dist_str)
    ves_density_close = {dt: pd.DataFrame(columns=cts_str, index=range(10500)) for dt in dist_str}
    combined_density_data = pd.DataFrame(columns=combined_columns,
                                         index=range(num_cts * num_cat * 5000))
    comb_median_data = pd.DataFrame(columns=combined_columns, index=range(num_cts * num_cat))
    if type(dist_threshold) == list and len(dist_threshold) > len(comp_cls_list):
        raise ValueError(
            f'Choose Color Palette with more colors. This has {len(comp_cls_list)} but you need {len(dist_threshold)}')
    else:
        dist_palette = {dist_str[i]: comp_cls_list[i] for i in range(num_cat)}
    log.info("Step 2/3: Iterate over celltypes to get suitable cellids, filter vesicles")
    prev_len_cellids = 0
    for ct in tqdm(range(num_cts)):
        # only get cells with min_comp_len, MSN with max_comp_len or axons with min ax_len
        ct_str = ct_dict[ct]
        if ct in ax_ct:
            cell_dict = analysis_params.load_cell_dict(ct)
            cellids = np.array(list(cell_dict.keys()))
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_ax,
                                            axon_only=True, max_path_len=None)
        else:
            cell_dict = analysis_params.load_cell_dict(ct)
            cellids = np.array(list(cell_dict.keys()))
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            if ct == 2:
                misclassified_asto_ids = analysis_params.load_potential_astros()
                astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
                cellids = cellids[astro_inds]
            cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_cell,
                                                axon_only=False, max_path_len=None)
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict[ct]))
        log.info('Prefilter vesicles for celltype')
        # load caches prefiltered for celltype
        if ct in ax_ct:
            ct_ves_ids = np.load(f'{cache_name}/{ct_dict[ct]}_ids.npy')
            ct_ves_coords = np.load(f'{cache_name}/{ct_dict[ct]}_rep_coords.npy')
            ct_ves_map2ssvids = np.load(f'{cache_name}/{ct_dict[ct]}_mapping_ssv_ids.npy')
            ct_ves_dist2matrix = np.load(f'{cache_name}/{ct_dict[ct]}_dist2matrix.npy')
        else:
            ct_ves_ids = np.load(f'{cache_name}/{ct_dict[ct]}_ids_fullcells.npy')
            ct_ves_coords = np.load(f'{cache_name}/{ct_dict[ct]}_rep_coords_fullcells.npy')
            ct_ves_map2ssvids = np.load(f'{cache_name}/{ct_dict[ct]}_mapping_ssv_ids_fullcells.npy')
            ct_ves_dist2matrix = np.load(f'{cache_name}/{ct_dict[ct]}_dist2matrix_fullcells.npy')
            ct_ves_axoness = np.load(f'{cache_name}/{ct_dict[ct]}_axoness_coarse_fullcells.npy')
        # filter for selected cellids
        ct_ind = np.in1d(ct_ves_map2ssvids, cellids)
        ct_ves_ids = ct_ves_ids[ct_ind]
        ct_ves_map2ssvids = ct_ves_map2ssvids[ct_ind]
        ct_ves_dist2matrix = ct_ves_dist2matrix[ct_ind]
        ct_ves_coords = ct_ves_coords[ct_ind]
        if ct not in ax_ct:
            ct_ves_axoness = ct_ves_axoness[ct_ind]
            # make sure for full cells vesicles are only in axon
            ax_ind = np.in1d(ct_ves_axoness, 1)
            ct_ves_ids = ct_ves_ids[ax_ind]
            ct_ves_map2ssvids = ct_ves_map2ssvids[ax_ind]
            ct_ves_dist2matrix = ct_ves_dist2matrix[ax_ind]
            ct_ves_coords = ct_ves_coords[ax_ind]
        assert len(np.unique(ct_ves_map2ssvids)) <= len(cellids)
        log.info('Iterate over cells to get vesicles associated to axon')
        #get axon_pathlength for corrensponding cellids
        axon_pathlengths = np.zeros(len(cellids))
        for c, cellid in enumerate(tqdm(cellids)):
            axon_pathlengths[c] = cell_dict[cellid]['axon length']
        #prepare inputs for multiprocessing
        cell_inputs = [[cellids[i], ct_ves_coords, ct_ves_map2ssvids, ct_ves_dist2matrix, dist_threshold, axon_pathlengths[i]] for i in range(len(cellids))]
        if type(dist_threshold) == list:
            outputs = start_multiprocess_imap(get_ves_distance_multiple_per_cell, cell_inputs)
        else:
            outputs = start_multiprocess_imap(get_ves_distance_per_cell, cell_inputs)
        outputs = np.array(outputs)
        ct_ves_density_dict = {dist_str[i]: outputs[:, i] for i in range(num_cat)}
        ves_density_all.loc[0:len(cellids) -1 , ct_str] = ct_ves_density_dict['all']
        comb_median_data.loc[ct * num_cat: ct * num_cat + num_cat - 1, 'celltype'] = ct_str
        combined_density_data.loc[prev_len_cellids: prev_len_cellids + num_cat * len(cellids) - 1,
        'celltype'] = ct_str
        for i, dt in enumerate(dist_str):
            ves_density_close[dt].loc[0:len(cellids) - 1, ct_str] = ct_ves_density_dict[dt]
            combined_density_data.loc[
            prev_len_cellids + i * len(cellids): prev_len_cellids + (i + 1) * len(cellids) - 1,
            'distance threshold'] = dt
            combined_density_data.loc[
            prev_len_cellids + i * len(cellids): prev_len_cellids + (i + 1) * len(cellids) - 1,
            'vesicle density'] = ct_ves_density_dict[dt]
            median_den = np.median(ct_ves_density_dict[dt])
            comb_median_data.loc[ct * num_cat + i, 'vesicle density'] = median_den
            comb_median_data.loc[ct * num_cat + i, 'distance threshold'] = dt
            if i == 0:
                log.info(f'{ct_str} cells have a median vesicle density of {median_den:.2f} 1/µm')
            else:
                log.info(f'{ct_str} cells have a median vesicle density of {median_den:.2f} 1/µm '
                     f'for vesicles closer than {dt} nm')
        prev_len_cellids += len(cellids) * num_cat

    log.info('Step 3/3: Plot results')
    ves_density_all.to_csv(f'{f_name}/ves_density_all.csv')
    comb_median_data.to_csv(f'{f_name}/ves_density_median_comb_data.csv')
    write_obj2pkl(f'{f_name}/comb_den_data.pkl', combined_density_data)
    sns.boxplot(ves_density_all, palette=ct_palette)
    plt.ylabel('vesicle density [1/µm]')
    plt.title('Number of vesicles per axon pathlength')
    plt.savefig(f'{f_name}/all_ves_box.svg')
    plt.close()
    sns.pointplot(x = 'celltype', y = 'vesicle density', data=comb_median_data, hue='distance threshold', palette=dist_palette, join=False)
    plt.ylabel('median vesicle density [1/µm]')
    plt.title('Median number of vesicles per axon pathlength with different thresholds in membrane distance')
    plt.savefig(f'{f_name}/all_ves_comb_median_point.svg')
    plt.close()
    sns.boxplot(x='celltype', y='vesicle density', data=combined_density_data, hue='distance threshold', palette=dist_palette)
    plt.ylabel('vesicle density [1/µm]')
    plt.title('Number of vesicles per axon pathlength with different thresholds in membrane distance')
    plt.savefig(f'{f_name}/all_ves_comb_box.svg')
    plt.close()
    for i, dt_str in enumerate(dist_str):
        if i == 0:
            continue
        dt = dist_threshold[i - 1]
        ves_density_close[dt_str].to_csv(f'{f_name}/ves_density_close_{dt}nm.csv')
        sns.boxplot(ves_density_close[dt_str], palette=ct_palette)
        plt.ylabel('vesicle density [1/µm]')
        plt.title(f'Number of vesicles closer than {dt} to membrane per axon pathlength')
        plt.savefig(f'{f_name}/close_ves_{dt}nm_box.svg')
        plt.close()

    log.info(f'Analysis for vesicles closer to {dist_threshold}nm in all celltypes done')




