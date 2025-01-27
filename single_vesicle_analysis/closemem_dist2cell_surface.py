#close membrane analysis distance to surface of cells
#use cell meshes to see what in certain radius of vesicle
#similar to closemem_dist2dataset_syns

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct, get_cell_close_surface_area
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    from cajal.nvmescratch.users.arother.bio_analysis.general.vesicle_helper import get_non_synaptic_vesicle_coords
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    import os as os
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from syconn.mp.mp_utils import start_multiprocess_imap
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import kruskal, ranksums
    from itertools import combinations

    version = 'v6'
    analysis_params = Analysis_Params(version = version)
    global_params.wd = analysis_params.working_dir()
    with_glia = False
    ct_dict = analysis_params.ct_dict(with_glia=with_glia)
    min_comp_len = 200
    full_cell = True
    dist_threshold = 10  # nm
    min_syn_size = 0.1
    syn_prob_thresh = 0.6
    nonsyn_dist_threshold = 5  # nm
    release_thresh = 1 #µm
    cls = CelltypeColors(ct_dict = ct_dict)
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw'}
    color_key = 'TePkBrNGF'
    celltype = 0
    ct_str = ct_dict[celltype]
    fontsize = 20
    suitable_ids_only = True
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/single_vesicle_analysis/250127_j0251{version}_{ct_str}_dist2cell_surface_mcl_%i_dt_%i_syn%i_r%i_%s" % (
        min_comp_len, dist_threshold, nonsyn_dist_threshold, release_thresh, color_key)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'close_mem_2_cell_surface_analysis_{ct_str}', log_dir=f_name + '/logs/')
    log.info("min_comp_len = %i, colors = %s" % (min_comp_len, color_key))
    log.info(f'min syn size = {min_syn_size} µm², syn prob thresh = {syn_prob_thresh}, threshold for close to membrane = {dist_threshold} nm, '
             f'threshold to putative release site = {release_thresh} µm, non syn dist threshold = {nonsyn_dist_threshold} nm')
    if full_cell:
        log.info('Only full cells will be processed')
    if with_glia:
        log.info('Putative glia synapses are not filtered out')
    else:
        log.info('Putative glia synapses are filtered out')
    if suitable_ids_only:
        id_path = 'cajal/scratch/users/arother/bio_analysis_results/general/' \
                     '240524_j0251v6_ct_morph_analyses_mcl_200_ax200_TeBKv6MSNyw_fs20npca2_umap5/ct_morph_df.csv'
        log.info(f'Only suitable ids will be used, loaded from {id_path}')
        loaded_morph_df = pd.read_csv(id_path, index_col=0)
        suitable_ids = np.array(loaded_morph_df['cellid'])
        suitable_cts = np.array(loaded_morph_df['celltype'])
        log.info(f'{len(suitable_ids)} cells were selected to be suitable for analysis')
    else:
        ssd = SuperSegmentationDataset(working_dir=global_params.wd)
        suitable_ids = ssd.ssv_ids
        suitable_cts = ssd.load_numpy_data('celltype_pts_e3')
        suitable_cts = [ct_dict[ct] for ct in suitable_cts]
    ct_str_list = analysis_params.ct_str(with_glia=with_glia)
    #remove suitable ids from ct that is looked at to avoid over-representation of own surface area
    suitable_ids = suitable_ids[suitable_cts != ct_str]
    suitable_cts = suitable_cts[suitable_cts != ct_str]

    known_mergers = analysis_params.load_known_mergers()
    #misclassified_asto_ids = analysis_params.load_potential_astros()
    cache_name = analysis_params.file_locations

    log.info('Step 1/4: Filter suitable cellids')
    cell_dict = analysis_params.load_cell_dict(celltype)
    # get ids with min compartment length
    cellids = np.array(list(cell_dict.keys()))
    merger_inds = np.in1d(cellids, known_mergers) == False
    cellids = cellids[merger_inds]
    #astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
    #cellids = cellids[astro_inds]
    axon_cts = analysis_params.axon_cts()
    if full_cell and celltype not in axon_cts:
        cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                axon_only=False,
                                                max_path_len=None)
    else:
        cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                        axon_only=True,
                                        max_path_len=None)
    cellids = np.sort(cellids)
    log.info(f'{len(cellids)} {ct_str} cells fulfull criteria')

    log.info('Step 2/4: Get coordinates of all close-membrane vesicles')
    # load caches prefiltered for celltype
    if celltype in axon_cts:
        ct_ves_ids = np.load(f'{cache_name}/{ct_str}_ids.npy')
        ct_ves_coords = np.load(f'{cache_name}/{ct_str}_rep_coords.npy')
        ct_ves_map2ssvids = np.load(f'{cache_name}/{ct_str}_mapping_ssv_ids.npy')
        ct_ves_dist2matrix = np.load(f'{cache_name}/{ct_str}_dist2matrix.npy')
    else:
        ct_ves_ids = np.load(f'{cache_name}/{ct_str}_ids_fullcells.npy')
        ct_ves_coords = np.load(f'{cache_name}/{ct_str}_rep_coords_fullcells.npy')
        ct_ves_map2ssvids = np.load(f'{cache_name}/{ct_str}_mapping_ssv_ids_fullcells.npy')
        ct_ves_dist2matrix = np.load(f'{cache_name}/{ct_str}_dist2matrix_fullcells.npy')
        ct_ves_axoness = np.load(f'{cache_name}/{ct_str}_axoness_coarse_fullcells.npy')
    # filter for selected cellids
    ct_ind = np.in1d(ct_ves_map2ssvids, cellids)
    ct_ves_ids = ct_ves_ids[ct_ind]
    ct_ves_map2ssvids = ct_ves_map2ssvids[ct_ind]
    ct_ves_dist2matrix = ct_ves_dist2matrix[ct_ind]
    ct_ves_coords = ct_ves_coords[ct_ind]
    if celltype not in axon_cts:
        ct_ves_axoness = ct_ves_axoness[ct_ind]
        # make sure for full cells vesicles are only in axon
        ax_ind = np.in1d(ct_ves_axoness, 1)
        ct_ves_ids = ct_ves_ids[ax_ind]
        ct_ves_map2ssvids = ct_ves_map2ssvids[ax_ind]
        ct_ves_dist2matrix = ct_ves_dist2matrix[ax_ind]
        ct_ves_coords = ct_ves_coords[ax_ind]
    assert len(np.unique(ct_ves_map2ssvids)) <= len(cellids)
    all_ves_number = len(ct_ves_ids)
    #get all mebrane close vesicles
    dist_inds = ct_ves_dist2matrix < dist_threshold
    ct_ves_ids = ct_ves_ids[dist_inds]
    ct_ves_map2ssvids = ct_ves_map2ssvids[dist_inds]
    ct_ves_coords = ct_ves_coords[dist_inds]
    log.info(f'In celltype {ct_str}, {len(ct_ves_ids)} out of {all_ves_number} vesicles are membrane close ({100*len(ct_ves_ids)/ all_ves_number:.2f} %)')
    cellids_close_mem = np.sort(np.unique(ct_ves_map2ssvids))
    log.info(f'{len(cellids_close_mem)} out of {len(cellids)} have close-membrane vesicles ({100*len(cellids_close_mem)/ len(cellids):.2f} %).')

    log.info('Step 3/4: Get non-synaptic vesicles')
    #get synapses outgoing from this celltype
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    ct_syn_cts, ct_syn_ids, ct_syn_axs, ct_syn_ssv_partners, ct_syn_sizes, ct_syn_spiness, ct_syn_rep_coord = filter_synapse_caches_for_ct(
        pre_cts=[celltype],
        post_cts=None,
        syn_prob_thresh=syn_prob_thresh,
        min_syn_size=min_syn_size,
        axo_den_so=True,
        synapses_caches=None, sd_synssv = sd_synssv)
    # filter so that only filtered cellids are included and are all presynaptic
    ct_inds = np.in1d(ct_syn_ssv_partners, cellids_close_mem).reshape(len(ct_syn_ssv_partners), 2)
    comp_inds = np.in1d(ct_syn_axs, 1).reshape(len(ct_syn_ssv_partners), 2)
    filtered_inds = np.all(ct_inds == comp_inds, axis=1)
    ct_syn_coords = ct_syn_rep_coord[filtered_inds]
    ct_syn_axs = ct_syn_axs[filtered_inds]
    ct_syn_ssv_partners = ct_syn_ssv_partners[filtered_inds]
    #get non-synaptic vesicles
    cell_input = [[cellid, ct_ves_ids, ct_ves_coords, ct_ves_map2ssvids, ct_syn_coords, nonsyn_dist_threshold] for
                  cellid in cellids_close_mem]
    cell_output = start_multiprocess_imap(get_non_synaptic_vesicle_coords, cell_input)
    cell_output = np.array(cell_output, dtype='object')
    # output still list of arrays
    non_syn_ves_ids = cell_output[:, 0]
    non_syn_ves_coords = cell_output[:, 1]
    # get all ves ids
    non_syn_ves_ids_con = np.concatenate(non_syn_ves_ids)
    non_syn_ves_coords_con = np.concatenate(non_syn_ves_coords)
    log.info(
        f'{len(non_syn_ves_ids_con)} vesicles are close-membrane non-synaptic ({100 * len(non_syn_ves_ids_con) / len(ct_ves_ids):.2f} %)')
    # make dataframe with all vesicles
    columns = ['cellid', 'ves coord x', 'ves coord y', 'ves coord z']
    non_syn_ves_coords_df = pd.DataFrame(columns=columns, index=non_syn_ves_ids_con)
    for i, cellid in enumerate(tqdm(cellids_close_mem)):
        cell_ves_ids = non_syn_ves_ids[i]
        cell_ves_coords = non_syn_ves_coords[i]
        non_syn_ves_coords_df.loc[cell_ves_ids, 'cellid'] = cellid
        non_syn_ves_coords_df.loc[cell_ves_ids, 'ves coord x'] = cell_ves_coords[:, 0]
        non_syn_ves_coords_df.loc[cell_ves_ids, 'ves coord y'] = cell_ves_coords[:, 1]
        non_syn_ves_coords_df.loc[cell_ves_ids, 'ves coord z'] = cell_ves_coords[:, 2]

    non_syn_ves_coords_df.to_csv(
        f'{f_name}/non_syn_{nonsyn_dist_threshold}_close_mem_{dist_threshold}_{ct_str}_ves_coords.csv')

    log.info('Step 3/4: Get coordinates of all suitable cells')
    #for all cells load their vertex coordinates
    cell_ves_coords_nm = non_syn_ves_coords_con * global_params.config['scaling']
    mesh_input = [[cellid, cell_ves_coords_nm, release_thresh] for cellid in suitable_ids]
    ct_mesh_output = start_multiprocess_imap(get_cell_close_surface_area, mesh_input)
    ct_mesh_output = np.array(ct_mesh_output, dtype=object)
    ct_ves_number = ct_mesh_output[:, 0]
    ct_summed_surface_area = ct_mesh_output[:, 1]
    ct_mesh_surface_areas = ct_mesh_output[:, 2]
    #put results in per cell df
    columns = ['cellid', 'celltype', 'number vesicles close', 'summed surface area close', 'surface mesh area',
               'ratio number vesicles', 'ratio summed surface area close', 'fraction surface mesh area', 'freq number vesicles',
               'freq surface area close']
    result_percell_df = pd.DataFrame(columns= columns, index = range(len(suitable_ids)))
    result_percell_df['cellid'] = suitable_ids
    result_percell_df['celltype'] = suitable_cts
    result_percell_df['surface mesh area'] = ct_mesh_surface_areas
    result_percell_df['number vesicles close'] = ct_ves_number
    result_percell_df['summed surface area close'] = ct_summed_surface_area
    #get ratio between number of vesicles and summed surface area per cell
    result_percell_df['ratio number vesicles'] = result_percell_df['number vesicles close'] / result_percell_df['surface mesh area']
    result_percell_df['ratio summed surface area close'] = result_percell_df['summed surface area close'] / result_percell_df['surface mesh area']
    #normalise values by multiplying with fraction of surface mesh area
    summed_surface_area = np.sum(result_percell_df['surface mesh area'])
    total_number_vesicles = np.sum(result_percell_df['number vesicles close'])
    total_surface_area_close = np.sum(result_percell_df['summed surface area close'])
    result_percell_df['fraction surface mesh area'] = result_percell_df['surface mesh area'] / summed_surface_area
    fraction_number_vesicles = result_percell_df['number vesicles close'] / total_number_vesicles
    fraction_close_surface_area = result_percell_df['summed surface area close'] / total_surface_area_close
    result_percell_df['freq number vesicles'] = fraction_number_vesicles / result_percell_df['fraction surface mesh area']
    result_percell_df['freq surface area close'] = fraction_close_surface_area / result_percell_df['fraction surface mesh area']
    params = columns[2:]
    result_percell_df= result_percell_df.astype({param: float for param in params})
    result_percell_df = result_percell_df.astype({'celltype': str})
    result_percell_df.to_csv(f'{f_name}/percell_results.csv')

    log.info('Step 4/4: Get overview params, calculate statistics and plot results')
    cls = CelltypeColors(ct_dict=ct_dict)
    ct_palette = cls.ct_palette(key=color_key)
    ct_groups = result_percell_df.groupby('celltype')
    unique_ct_str = np.unique(result_percell_df['celltype'])
    ct_str_list = np.array(ct_str_list)[np.in1d(ct_str_list, unique_ct_str)]
    overview_df = pd.DataFrame(index = range(len(unique_ct_str)))
    overview_df['celltype'] = unique_ct_str
    overview_df['numbers'] = np.array(ct_groups.size())
    kruskal_results_df = pd.DataFrame(columns = ['stats', 'p-value'], index = params)
    group_comps = list(combinations(unique_ct_str, 2))
    ranksum_columns = [f'{gc[0]} vs {gc[1]}' for gc in group_comps]
    ranksum_df = pd.DataFrame(columns=ranksum_columns)
    for param in params:
        #calculate summed fraction, freq and ratios new with summed areas per celltype
        if 'ratio' in param or 'freq' in param:
            if param == 'freq number vesicles':
                overview_df[f'{param} sum'] = (np.array(ct_groups['number vesicles close'].sum()) / total_number_vesicles) / np.array(ct_groups['fraction surface mesh area'].sum())
            elif param == 'freq surface area close':
                overview_df[f'{param} sum'] = (np.array(ct_groups['summed surface area close'].sum()) / total_surface_area_close) / np.array(ct_groups['fraction surface mesh area'].sum())
            elif param == 'ratio number vesicles':
                overview_df[f'{param} sum'] = np.array(ct_groups['number vesicles close'].sum()) / np.array(ct_groups['surface mesh area'].sum())
            elif param == 'ratio summed surface area close':
                overview_df[f'{param} sum'] = np.array(ct_groups['summed surface area close'].sum()) / np.array(ct_groups['surface mesh area'].sum())
            else:
                raise ValueError(f'unknown parameter value {param}')
        #get overview params
        else:
            overview_df[f'{param} sum'] = np.array(ct_groups[param].sum())
        overview_df[f'{param} mean'] = np.array(ct_groups[param].mean())
        overview_df[f'{param} std'] = np.array(ct_groups[param].std())
        overview_df[f'{param} median'] = np.array(ct_groups[param].median())
        #calculate kruskal wallis result
        key_groups = [group[param].values for name, group in
                      result_percell_df.groupby('celltype')]
        kruskal_res = kruskal(*key_groups, nan_policy='omit')
        kruskal_results_df.loc[param, 'stats'] = kruskal_res[0]
        kruskal_results_df.loc[param, 'p-value'] = kruskal_res[1]
        if kruskal_res[1] < 0.05:
            for group in group_comps:
                ranksum_res = ranksums(ct_groups.get_group(group[0])[param], ct_groups.get_group(group[1])[param])
                ranksum_df.loc[f'{param} stats', f'{group[0]} vs {group[1]}'] = ranksum_res[0]
                ranksum_df.loc[f'{param} p-value', f'{group[0]} vs {group[1]}'] = ranksum_res[1]
        #plot results
        if 'surface area' in param and (not 'ratio' in param or not 'fraction' in param):
            ylabel = f'{param} [µm²]'
        elif 'ratio' in param and 'number' in param:
            ylabel = f'{param} [1/µm²]'
        else:
            ylabel = param
        sns.boxplot(data=result_percell_df, x='celltype', y=param, palette=ct_palette, order=ct_str_list)
        plt.title(param)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.xlabel('celltype', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(f'{f_name}/{param}_box.png')
        plt.savefig(f'{f_name}/{param}_box.svg')
        plt.close()
        sns.violinplot(data=result_percell_df, x='celltype', y=param, palette=ct_palette, inner="box", order=ct_str_list)
        plt.title(param)
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.xlabel('celltype', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(f'{f_name}/{param}_violin.png')
        plt.savefig(f'{f_name}/{param}_violin.svg')
        plt.close()
        #also plot overview df params as barplot
        sns.barplot(data = overview_df, x = 'celltype', y = f'{param} sum', palette=ct_palette, order=ct_str_list)
        plt.title(f'{param} sum')
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.xlabel('celltype', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(f'{f_name}/{param}_sum_ct_bar.png')
        plt.savefig(f'{f_name}/{param}_sum_ct_bar.svg')
        plt.close()
        sns.barplot(data=overview_df, x='celltype', y=f'{param} median', palette=ct_palette, order=ct_str_list)
        plt.title(f'{param} median')
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.xlabel('celltype', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(f'{f_name}/{param}_median_ct_bar.png')
        plt.savefig(f'{f_name}/{param}_median_ct_bar.svg')
        plt.close()

    overview_df.to_csv(f'{f_name}/overview_df.csv')
    kruskal_results_df.to_csv(f'{f_name}/kruskal_results.csv')
    ranksum_df.to_csv(f'{f_name}/ranksum_results.csv')



    log.info('Analysis finsihed.')


