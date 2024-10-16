#script to plot spine density for all msns or other celltype
#plot spine density with msn subgroups overlayed or other celltype
#plot spine density as jointplot vs GPe/ GPi ratio in synapse number and synapse sum size
#GPe/GPi ratios = GPi/(GPi + GPe

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct, get_spine_density, get_cell_soma_radius, \
        get_dendrite_info_cell, check_cutoff_dendrites
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct, get_ct_syn_number_sumsize, filter_contact_sites_axoness
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    from syconn.reps.segmentation import SegmentationDataset
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    import os as os
    import pandas as pd
    import numpy as np
    from syconn.mp.mp_utils import start_multiprocess_imap
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import ranksums, kruskal, spearmanr
    from itertools import combinations
    from tqdm import tqdm

    #global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"


    version = 'v6'
    analysis_params = Analysis_Params(version=version)
    ct_dict = analysis_params.ct_dict(with_glia=False)
    global_params.wd = analysis_params.working_dir()
    ssd = SuperSegmentationDataset(working_dir=global_params.wd)
    min_comp_len = 200
    syn_prob = 0.6
    min_syn_size = 0.1
    conn_ct = 3
    ct1 = 6
    ct2 = 7
    ct1_str = ct_dict[ct1]
    ct2_str = ct_dict[ct2]
    fontsize_jointplot = 20
    kde = True
    check_dens= True
    color_key = 'STNGPINTv6'
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/241015_j0251{version}_%s_{conn_ct}_{ct1_str}_{ct2_str}ratio_spine_density_mcl_%i_synprob_%.2f_kde%i_f{fontsize_jointplot}_fullden" % (
    ct_dict[conn_ct], min_comp_len, syn_prob, kde)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'{ct_dict[conn_ct]} spine density vs {ct1_str}, {ct2_str} ratio', log_dir=f_name + '/logs/')
    log.info(f"Analysis of spine density vs {ct1_str}, {ct2_str} ratio starts")
    if kde:
        log.info('Centre of jointplot will be kdeplot')
    else:
        log.info('Centre of jointplot will be scatter')
    if check_dens:
        dist_thresh = 7000
        log.info('Check if dendrites are cutoff from MSN cells and if so, exclude these cells')
        log.info(f'{dist_thresh/1000} µm were selected as distance threshold to dataset borders')
    ct_colors = CelltypeColors(ct_dict=ct_dict)
    ct_palette = ct_colors.ct_palette(key=color_key)


    log.info(f'Step 1/7: Load and check all {ct_dict[conn_ct]} cells')
    known_mergers = analysis_params.load_known_mergers()
    misclassified_asto_ids = analysis_params.load_potential_astros()

    cell_dict = analysis_params.load_cell_dict(celltype=conn_ct)
    cell_ids = np.array(list(cell_dict.keys()))
    merger_inds = np.in1d(cell_ids, known_mergers) == False
    cell_ids = cell_ids[merger_inds]
    astro_inds = np.in1d(cell_ids, misclassified_asto_ids) == False
    cell_ids = cell_ids[astro_inds]
    cell_ids = check_comp_lengths_ct(cellids=cell_ids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                    axon_only=False,
                                    max_path_len=None)
    log.info(f'{len(cell_ids)} suitable for analysis')

    columns = ['cellid', 'celltype','spine density', 'dendritic length', 'number primary dendrites',
               'number branching points','ratio branching points vs primary dendrites',f'{ct1_str}, {ct2_str} ratio syn number',
               f'{ct1_str}, {ct2_str} ratio sum syn size', f'syn number to {ct1_str}', f'syn size to {ct1_str}', f'syn number to {ct2_str}', f'syn size to {ct2_str}',
               f'{ct1_str}, {ct2_str} ratio cs number', f'{ct1_str}, {ct2_str}  ratio sum cs size', f'cs number to {ct1_str}', f'cs size to {ct2_str}',
               f'cs number to {ct2_str}', f'cs size to {ct2_str}']

    cell_ids = np.sort(cell_ids)

    if check_dens:
        log.info('Step 1b/7: Check if dendrites are cutoff from cells and if so, exclude them')
        #at least 5 µm needed to reach to segmented part of dataset
        dataset_borders = np.array(global_params.config.entries['cube_of_interest_bb']) * [10, 10, 25]
        cell_input = [[cellid, dist_thresh, dataset_borders] for cellid in cell_ids]
        cell_output = start_multiprocess_imap(check_cutoff_dendrites, cell_input)
        cell_output = np.array(cell_output)
        cell_ids = cell_output[:, 0]
        non_cutoff_mask = cell_output[:, 1]
        cell_ids = cell_ids[non_cutoff_mask == 1]
        log.info(f'{len(cell_ids)} cells do not have cutoff_dendrites')

    conn_result_df = pd.DataFrame(columns=columns, index=range(len(cell_ids)))
    conn_result_df['cellid'] = cell_ids

    log.info(f'Step 2/7: Get spine density of all {ct_dict[conn_ct]} cells')
    cell_input = [[cell_id, min_comp_len, cell_dict[cell_id]] for cell_id in cell_ids]
    spine_density_res = start_multiprocess_imap(get_spine_density, cell_input)
    spine_density_res = np.array(spine_density_res, dtype='object')
    spine_density = spine_density_res[:, 0]
    conn_result_df['spine density'] = spine_density
    conn_result_df.to_csv(f'{f_name}/{ct_dict[conn_ct]}_spine_density_results.csv')

    log.info(f'Step 3/7: Get soma diameter for {ct_dict[conn_ct]} cells')
    cell_soma_results = start_multiprocess_imap(get_cell_soma_radius, cell_ids)
    cell_soma_results = np.array(cell_soma_results, dtype='object')
    cell_diameters = cell_soma_results[:, 1].astype(float) * 2
    conn_result_df['soma diameter'] = cell_diameters


    log.info('Step 4/7: Get number of primary dendrites and dendritic lengths')
    #can also add number of branching points but currently difficult with skeleton parameter
    #set min_comp_len to None to include all cells that were found suitable before
    input = [[cell_id, None] for cell_id in cell_ids]
    cell_dendrite_results = start_multiprocess_imap(get_dendrite_info_cell, input)
    cell_dendrite_results = np.array(cell_dendrite_results, dtype='object')
    dendrite_lengths = cell_dendrite_results[:, 0]
    number_primary_dendrites = cell_dendrite_results[:, 1]
    number_branching_points = cell_dendrite_results[:, 2]
    conn_result_df['dendritic length'] = dendrite_lengths/ 1000 #in mm
    conn_result_df['number primary dendrites'] = number_primary_dendrites
    conn_result_df['number branching points'] = number_branching_points
    conn_result_df['ratio branching points vs primary dendrites'] = number_branching_points/ number_primary_dendrites
    conn_result_df.to_csv(f'{f_name}/{ct_dict[conn_ct]}_morph_results.csv')
    '''
    f_name_saving1 = "cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/240229_j0251v6_%s_GPratio_spine_density_mcl_%i_synprob_%.2f_kde%i_f20" % (
        ct_dict[conn_ct], min_comp_len, syn_prob, kde)
    log.info(f'Use morph parameters from {f_name_saving1}')
    conn_result_df= pd.read_csv(f'{f_name_saving1}/{ct_dict[conn_ct]}_morph_results.csv', index_col = 0)
    for key in conn_result_df.keys():
        if 'cs' in key:
            conn_result_df= conn_result_df.drop(key, axis = 1)
    '''

    log.info(f'Step 4/7: Get syn ratio for {ct_dict[conn_ct]} cells')
    conn_result_df[f'syn number to {ct1_str}'] = 0
    conn_result_df[f'syn size to {ct1_str}'] = 0
    conn_result_df[f'mean syn size to {ct1_str}'] = 0
    conn_result_df[f'syn number to {ct2_str}'] = 0
    conn_result_df[f'syn size to {ct2_str}'] = 0
    conn_result_df[f'mean syn size to {ct2_str}'] = 0
    log.info(f'Get suitable cellids from {ct1_str} and {ct2_str}')
    ct1_dict = analysis_params.load_cell_dict(ct1)
    ct1_ids = np.array(list(ct1_dict.keys()))
    merger_inds = np.in1d(ct1_ids, known_mergers) == False
    ct1_ids = ct1_ids[merger_inds]
    astro_inds = np.in1d(ct1_ids, misclassified_asto_ids) == False
    ct1_ids = ct1_ids[astro_inds]
    ct1_ids = check_comp_lengths_ct(cellids=ct1_ids, fullcelldict=ct1_dict, min_comp_len=min_comp_len,
                                    axon_only=False,
                                    max_path_len=None)

    ct2_dict = analysis_params.load_cell_dict(ct2)
    ct2_ids = np.array(list(ct2_dict.keys()))
    merger_inds = np.in1d(ct2_ids, known_mergers) == False
    ct2_ids = ct2_ids[merger_inds]
    astro_inds = np.in1d(ct2_ids, misclassified_asto_ids) == False
    ct2_ids = ct2_ids[astro_inds]
    ct2_ids = check_comp_lengths_ct(cellids=ct2_ids, fullcelldict=ct2_dict, min_comp_len=min_comp_len,
                                    axon_only=False,
                                    max_path_len=None)
    log.info(f'{len(ct1_ids)} suitable for analysis and {len(ct2_ids)} suitable for analysis')

    log.info(f'Get information from synapses to {ct1_str} and {ct2_str}')
    #prefilter synapses for syn_prob, min_syn_size and celltype
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.wd)
    #use different version of function as long as m_ids not available
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(sd_synssv =sd_synssv,
                                                                                                            pre_cts=[conn_ct],
                                                                                                            post_cts=[ct1, ct2],
                                                                                                            syn_prob_thresh=syn_prob,
                                                                                                            min_syn_size=min_syn_size,
                                                                                                            axo_den_so=True,
                                                                                                            synapses_caches=None)
    #filter synapses to only include full cells
    all_suitable_ids = np.hstack([cell_ids, ct1_ids, ct2_ids])
    suit_ct_inds = np.all(np.in1d(m_ssv_partners, all_suitable_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[suit_ct_inds]
    m_ssv_partners = m_ssv_partners[suit_ct_inds]
    m_sizes = m_sizes[suit_ct_inds]
    m_axs = m_axs[suit_ct_inds]
    #get per cell information about synapses to ct1
    ct1_inds = np.where(m_cts == ct1)[0]
    ct1_ssv_partners = m_ssv_partners[ct1_inds]
    ct1_sizes = m_sizes[ct1_inds]
    ct1_cts = m_cts[ct1_inds]
    ct1_syn_numbers, ct1_sum_sizes, unique_conn_ct1 = get_ct_syn_number_sumsize(syn_sizes = ct1_sizes,
                                                                               syn_ssv_partners = ct1_ssv_partners,
                                                                               syn_cts= ct1_cts, ct = conn_ct)
    sort_inds_ct1 = np.argsort(unique_conn_ct1)
    unique_conn_ct1_sorted = unique_conn_ct1[sort_inds_ct1]
    ct1_syn_numbers_sorted = ct1_syn_numbers[sort_inds_ct1]
    ct1_sum_sizes_sorted = ct1_sum_sizes[sort_inds_ct1]
    ct1_df_inds = np.in1d(conn_result_df['cellid'], unique_conn_ct1_sorted)
    conn_result_df.loc[ct1_df_inds, f'syn number to {ct1_str}'] = ct1_syn_numbers_sorted
    conn_result_df.loc[ct1_df_inds, f'syn size to {ct1_str}'] = ct1_sum_sizes_sorted
    conn_result_df.loc[ct1_df_inds, f'mean syn size to {ct1_str}'] = ct1_sum_sizes_sorted/ ct1_syn_numbers_sorted
    #get number of partner cells to ct1
    num_ct1_partners = np.empty(len(unique_conn_ct1))
    for i, conn_cellid in enumerate(tqdm(unique_conn_ct1)):
        ct1_conn_partners = ct1_ssv_partners[np.where(ct1_ssv_partners == conn_cellid)[0]]
        ct1_conn_partners = ct1_conn_partners[np.where(ct1_conn_partners != conn_cellid)]
        num_ct1 = len(np.unique(ct1_conn_partners))
        num_ct1_partners[i] = num_ct1

    num_ct1_partners_sorted = num_ct1_partners[sort_inds_ct1]
    conn_result_df[f'number {ct1_str} partners'] = 0
    conn_result_df.loc[ct1_df_inds, f'number {ct1_str} partners'] = num_ct1_partners_sorted

    #get per cell information about synapses to ct2
    ct2_inds = np.where(m_cts == ct2)[0]
    ct2_ssv_partners = m_ssv_partners[ct2_inds]
    ct2_sizes = m_sizes[ct2_inds]
    ct2_cts = m_cts[ct2_inds]
    ct2_syn_numbers, ct2_sum_sizes, unique_conn_ct2 = get_ct_syn_number_sumsize(syn_sizes=ct2_sizes,
                                                                               syn_ssv_partners=ct2_ssv_partners,
                                                                               syn_cts=ct2_cts, ct=conn_ct)
    sort_inds_ct2 = np.argsort(unique_conn_ct2)
    unique_conn_ct2_sorted = unique_conn_ct2[sort_inds_ct2]
    ct2_syn_numbers_sorted = ct2_syn_numbers[sort_inds_ct2]
    ct2_sum_sizes_sorted = ct2_sum_sizes[sort_inds_ct2]
    ct2_df_inds = np.in1d(conn_result_df['cellid'], unique_conn_ct2_sorted)
    conn_result_df.loc[ct2_df_inds, f'syn number to {ct2_str}'] = ct2_syn_numbers_sorted
    conn_result_df.loc[ct2_df_inds, f'syn size to {ct2_str}'] = ct2_sum_sizes_sorted
    conn_result_df.loc[ct2_df_inds, f'mean syn size to {ct2_str}'] = ct2_sum_sizes_sorted / ct2_syn_numbers_sorted
    # get number of partner cells to ct1
    num_ct2_partners = np.empty(len(unique_conn_ct2))
    for i, conn_cellid in enumerate(tqdm(unique_conn_ct2)):
        ct2_conn_partners = ct2_ssv_partners[np.where(ct2_ssv_partners == conn_cellid)[0]]
        ct2_conn_partners = ct2_conn_partners[np.where(ct2_conn_partners != conn_cellid)]
        num_ct2 = len(np.unique(ct2_conn_partners))
        num_ct2_partners[i] = num_ct2

    num_ct2_partners_sorted = num_ct2_partners[sort_inds_ct2]
    conn_result_df[f'number {ct2_str} partners'] = 0
    conn_result_df.loc[ct2_df_inds, f'number {ct2_str} partners'] = num_ct2_partners_sorted
    conn_result_df['number cell partners'] = conn_result_df[f'number {ct2_str} partners'] + conn_result_df[f'number {ct1_str} partners']

    #get number of synapses to GP
    conn_result_df['syn number total'] = conn_result_df[f'syn number to {ct1_str}'] + conn_result_df[f'syn number to {ct2_str}']
    #get ratio per conn ct cell and put in dataframe: ct2/(ct2 + ct1)
    nonzero_inds = np.any([conn_result_df[f'syn number to {ct2_str}'] > 0, conn_result_df[f'syn number to {ct1_str}'] > 0], axis = 0)
    conn_result_df.loc[nonzero_inds, f'{ct1_str}, {ct2_str} ratio syn number'] = conn_result_df.loc[nonzero_inds, f'syn number to {ct2_str}'] / \
                                                             (conn_result_df.loc[nonzero_inds, f'syn number to {ct2_str}']  + conn_result_df.loc[nonzero_inds, f'syn number to {ct1_str}'])
    conn_result_df.loc[nonzero_inds, f'{ct1_str}, {ct2_str} ratio sum syn size'] =  conn_result_df.loc[nonzero_inds, f'syn size to {ct2_str}']/ \
                                                                 (conn_result_df.loc[nonzero_inds, f'syn size to {ct1_str}'] + conn_result_df.loc[nonzero_inds, f'syn size to {ct2_str}'])
    conn_result_df.loc[nonzero_inds, f'{ct1_str}, {ct2_str} ratio mean syn size'] = conn_result_df.loc[nonzero_inds, f'mean syn size to {ct2_str}']/ \
                                                                 (conn_result_df.loc[nonzero_inds, f'mean syn size to {ct1_str}'] + conn_result_df.loc[nonzero_inds, f'mean syn size to {ct2_str}'])
    #get abs(GP ratio - 0.5) for syn area ratio, syn size ratio
    conn_result_df.loc[nonzero_inds, f'abs({ct1_str}, {ct2_str} ratio - 0.5)'] = np.abs(conn_result_df.loc[nonzero_inds, f'{ct1_str}, {ct2_str} ratio syn number'] - 0.5)
    conn_result_df.loc[nonzero_inds, f'abs({ct1_str}, {ct2_str} area ratio - 0.5)'] = np.abs(
        conn_result_df.loc[nonzero_inds, f'{ct1_str}, {ct2_str} ratio sum syn size'] - 0.5)
    conn_result_df.to_csv(f'{f_name}/{ct_dict[conn_ct]}_morph_{ct1_str}_{ct2_str}ratio.csv')
    #get syn sizes for all cells
    lenct_syns = len(ct1_sizes) + len(ct2_sizes)
    conn_ct1_partners = ct1_ssv_partners[np.where(ct1_cts == conn_ct)]
    syn_sizes_df = pd.DataFrame(columns=['syn sizes', 'to celltype', 'cellid', f'{ct_dict[conn_ct]} group'], index=range(lenct_syns))
    syn_sizes_df.loc[0: len(ct1_sizes) - 1, 'syn sizes'] = ct1_sizes
    syn_sizes_df.loc[0: len(ct1_sizes) - 1, 'to celltype'] = ct1_str
    syn_sizes_df.loc[0: len(ct1_sizes) - 1, 'cellid'] = conn_ct1_partners
    conn_ct2_partners = ct2_ssv_partners[np.where(ct2_cts == conn_ct)]
    syn_sizes_df.loc[len(ct1_sizes): lenct_syns - 1, 'syn sizes'] = ct2_sizes
    syn_sizes_df.loc[len(ct1_sizes): lenct_syns - 1, 'to celltype'] = ct2_str
    syn_sizes_df.loc[len(ct1_sizes): lenct_syns - 1, 'cellid'] = conn_ct2_partners
    syn_sizes_df = syn_sizes_df.astype({'syn sizes': float})
    syn_sizes_df.to_csv(f'{f_name}/syn_sizes_to_{ct1_str}_{ct2_str}.csv')
    #get statistics for all sizes
    ranksum_sizes = pd.DataFrame(columns = [f'syn sizes to {ct1_str} vs {ct2_str}'], index = ['stats', 'p-value'])
    stats, p_value = ranksums(ct1_sizes, ct2_sizes)
    ranksum_sizes.loc['stats', f'syn sizes to {ct1_str} vs {ct2_str}'] = stats
    ranksum_sizes.loc['p-value', f'syn sizes to {ct1_str} vs {ct2_str}'] = p_value
    ranksum_sizes.to_csv(f'{f_name}/ranksum_sizes.csv')
    #plot syn sizes independent of cells
    sns.histplot(x='syn sizes', data=syn_sizes_df, hue='to celltype', palette=ct_palette, common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True)
    plt.ylabel('number of synapses')
    plt.xlabel('synaptic mesh area [µm²]')
    plt.savefig(f'{f_name}/synsizes_to_{ct1_str}_{ct2_str}_hist.png')
    plt.savefig(f'{f_name}/synsizes_to_{ct1_str}_{ct2_str}_hist.svg')
    plt.close()
    sns.histplot(x='syn sizes', data=syn_sizes_df, hue='to celltype', palette=ct_palette, common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, log_scale=True)
    plt.ylabel('number of synapses')
    plt.xlabel('synaptic mesh area [µm²]')
    plt.savefig(f'{f_name}/synsizes_to_{ct1_str}_{ct2_str}_log.png')
    plt.savefig(f'{f_name}/synsizes_to_{ct1_str}_{ct2_str}_hist_log.svg')
    plt.close()
    sns.histplot(x='syn sizes', data=syn_sizes_df, hue='to celltype', palette=ct_palette, common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, stat='percent')
    plt.ylabel('% of synapses')
    plt.xlabel('synaptic mesh area [µm²]')
    plt.savefig(f'{f_name}/synsizes_to_{ct1_str}_{ct2_str}_hist_perc.png')
    plt.savefig(f'{f_name}/synsizes_to_{ct1_str}_{ct2_str}_hist_perc.svg')
    plt.close()
    sns.histplot(x='syn sizes', data=syn_sizes_df, hue='to celltype', palette=ct_palette, common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, log_scale=True, stat='percent')
    plt.ylabel('% of synapses')
    plt.xlabel('synaptic mesh area [µm²]')
    plt.savefig(f'{f_name}/synsizes_to_{ct1_str}_{ct2_str}_hist_log_perc.png')
    plt.savefig(f'{f_name}/synsizes_to_{ct1_str}_{ct2_str}_hist_log_perc.svg')
    plt.close()
    sns.boxplot(data=syn_sizes_df, x='to celltype', y='syn sizes', palette=ct_palette)
    plt.ylabel('synaptic mesh area [µm²]')
    plt.savefig(f'{f_name}/synsizes_to_{ct1_str}_{ct2_str}_box.png')
    plt.savefig(f'{f_name}/synsizes_to_{ct1_str}_{ct2_str}_box.svg')
    plt.close()
    sns.stripplot(x='to celltype', y='syn sizes', data=syn_sizes_df, color='black', alpha=0.2,
                  dodge=True, size=2)
    sns.violinplot(x='to celltype', y='syn sizes', data=syn_sizes_df, inner="box",
                   palette=ct_palette)
    plt.ylabel('synaptic mesh area [µm²]')
    plt.savefig(f'{f_name}/synsizes_to_{ct1_str}_{ct2_str}_violin.png')
    plt.savefig(f'{f_name}/synsizes_to_{ct1_str}_{ct2_str}_violin.svg')
    plt.close()
    '''
    log.info('Step 5/7: Get cs ratio from MSN')
    cs_ssv_ids = np.load(f'/{analysis_params.file_locations}/cs_ssv_ids_filtered.npy')
    cs_ssv_mesh_areas = np.load(f'/{analysis_params.file_locations}/cs_ssv_mesh_areas_filtered.npy') / 2
    cs_ssv_neuron_partners = np.load(f'/{analysis_params.file_locations}/cs_ssv_neuron_partners_filtered.npy')
    cs_ssv_axoness = np.load(f'/{analysis_params.file_locations}/cs_ssv_axoness_filtered.npy')
    cs_ssv_celltypes = np.load(f'/{analysis_params.file_locations}/cs_ssv_celltypes_filtered.npy')
    cs_ssv_coords = np.load(f'/{analysis_params.file_locations}/cs_ssv_coords_filtered.npy')
    # filter so that only MSN axons to GPe/GPi dendrites are left
    cs_ssv_celltypes, cs_ssv_axoness, cs_ssv_neuron_partners, cs_ssv_mesh_areas, cs_ssv_coords = filter_contact_sites_axoness(
        cs_ssv_mesh_areas=cs_ssv_mesh_areas, cs_ssv_celltypes=cs_ssv_celltypes,
        cs_ssv_neuron_partners=cs_ssv_neuron_partners, cs_ssv_axoness=cs_ssv_axoness,
        cs_ssv_coords=cs_ssv_coords, pre_cts=[conn_ct
        ], post_cts=[gpe_ct, gpi_ct], axo_den_so=True, min_size=0.1)
    #filter so that only suitable ids left
    suit_ct_inds = np.all(np.in1d(cs_ssv_neuron_partners, all_suitable_ids).reshape(len(cs_ssv_neuron_partners), 2), axis=1)
    cs_ssv_celltypes = cs_ssv_celltypes[suit_ct_inds]
    cs_ssv_neuron_partners = cs_ssv_neuron_partners[suit_ct_inds]
    cs_ssv_mesh_areas = cs_ssv_mesh_areas[suit_ct_inds]
    cs_ssv_axoness = cs_ssv_axoness[suit_ct_inds]
    # get per cell information about cs to GPe
    gpe_inds = np.where(cs_ssv_celltypes == gpe_ct)[0]
    gpe_cs_ssv_partners = cs_ssv_neuron_partners[gpe_inds]
    gpe_cs_sizes = cs_ssv_mesh_areas[gpe_inds]
    gpe_cs_cts = cs_ssv_celltypes[gpe_inds]
    gpe_cs_numbers, gpe_cs_sum_sizes, unique_cs_msn_gpe = get_ct_syn_number_sumsize(syn_sizes=gpe_cs_sizes,
                                                                               syn_ssv_partners=gpe_cs_ssv_partners,
                                                                               syn_cts=gpe_cs_cts, ct=conn_ct
                                                                               )
    sort_inds_gpe = np.argsort(unique_cs_msn_gpe)
    unique_cs_msn_gpe_sorted = unique_cs_msn_gpe[sort_inds_gpe]
    gpe_cs_numbers_sorted = gpe_cs_numbers[sort_inds_gpe]
    gpe_cs_sum_sizes_sorted = gpe_cs_sum_sizes[sort_inds_gpe]
    gpe_df_inds = np.in1d(conn_result_df
    ['cellid'], unique_cs_msn_gpe_sorted)
    conn_result_df.loc[gpe_df_inds, 'cs number to GPe'] = gpe_cs_numbers_sorted
    conn_result_df.loc[gpe_df_inds, 'cs size to GPe'] = gpe_cs_sum_sizes_sorted
    conn_result_df.loc[gpe_df_inds, 'mean cs size to GPe'] = gpe_cs_sum_sizes_sorted / gpe_cs_numbers_sorted

    # get per cell information about synapses to GPi
    gpi_inds = np.where(cs_ssv_celltypes == gpi_ct)[0]
    gpi_cs_ssv_partners = cs_ssv_neuron_partners[gpi_inds]
    gpi_cs_sizes = cs_ssv_mesh_areas[gpi_inds]
    gpi_cs_cts = cs_ssv_celltypes[gpi_inds]
    gpi_cs_numbers, gpi_cs_sum_sizes, unique_cs_msn_gpi = get_ct_syn_number_sumsize(syn_sizes=gpi_cs_sizes,
                                                                               syn_ssv_partners=gpi_cs_ssv_partners,
                                                                               syn_cts=gpi_cs_cts, ct=conn_ct
                                                                               )
    sort_inds_gpi = np.argsort(unique_cs_msn_gpi)
    unique_cs_msn_gpi_sorted = unique_cs_msn_gpi[sort_inds_gpi]
    gpi_cs_numbers_sorted = gpi_cs_numbers[sort_inds_gpi]
    gpi_cs_sum_sizes_sorted = gpi_cs_sum_sizes[sort_inds_gpi]
    gpi_df_inds = np.in1d(conn_result_df['cellid'], unique_cs_msn_gpi_sorted)
    conn_result_df.loc[gpi_df_inds, 'cs number to GPi'] = gpi_cs_numbers_sorted
    conn_result_df.loc[gpi_df_inds, 'cs size to GPi'] = gpi_cs_sum_sizes_sorted
    conn_result_df.loc[gpi_df_inds, 'mean cs size to GPi'] = gpi_cs_sum_sizes_sorted / gpi_cs_numbers_sorted
    # get GP ratio per msn cell and put in dataframe: GPi/(GPi + GPe)
    nonzero_inds = np.any([conn_result_df['cs number to GPi'] > 0, conn_result_df['cs number to GPe'] > 0], axis = 0)
    conn_result_df.loc[nonzero_inds, 'GP ratio cs number'] = conn_result_df.loc[nonzero_inds, 'cs number to GPi'] / \
                                                             (conn_result_df.loc[nonzero_inds, 'cs number to GPi'] +
                                                              conn_result_df.loc[nonzero_inds, 'cs number to GPe'])
    conn_result_df.loc[nonzero_inds, 'GP ratio sum cs size'] = conn_result_df.loc[nonzero_inds, 'cs size to GPi'] / \
                                                               (conn_result_df.loc[nonzero_inds, 'cs size to GPe'] +
                                                                conn_result_df.loc[nonzero_inds, 'cs size to GPi'])
    # get cs size in relation to syn size
    #check that number of cs lager than that of syns
    assert (conn_result_df['cs number to GPe'].sum() >= conn_result_df['syn number to GPe'].sum())
    assert (conn_result_df['cs number to GPi'].sum() >= conn_result_df['syn number to GPi'].sum())
    #get cs number and size in relation to syn number and size
    gpe_nonzero_inds = conn_result_df['syn number to GPe'] > 0
    gpi_nonzero_inds = conn_result_df['syn number to GPi'] > 0
    conn_result_df.loc[gpe_nonzero_inds, 'GPe syn vs cs fraction number'] = conn_result_df.loc[gpe_nonzero_inds, 'syn number to GPe']/ \
                                                                        conn_result_df.loc[gpe_nonzero_inds, 'cs number to GPe']
    conn_result_df.loc[gpe_nonzero_inds, 'GPe syn vs cs fraction sum size'] = conn_result_df.loc[gpe_nonzero_inds, 'syn size to GPe'] / \
                                                                           conn_result_df.loc[gpe_nonzero_inds, 'cs size to GPe']
    conn_result_df.loc[gpi_nonzero_inds, 'GPi syn vs cs fraction number'] = conn_result_df.loc[gpi_nonzero_inds, 'syn number to GPi'] / \
                                                                           conn_result_df.loc[gpi_nonzero_inds, 'cs number to GPi']
    conn_result_df.loc[gpi_nonzero_inds, 'GPi syn vs cs fraction sum size'] = conn_result_df.loc[gpi_nonzero_inds, 'syn size to GPi'] / \
                                                                         conn_result_df.loc[gpi_nonzero_inds, 'cs size to GPi']

    conn_result_df.to_csv(f'{f_name}/msn_morph_GPratio_cs.csv')
    # get cs sizes for all cells
    lengp_cs = len(gpe_cs_sizes) + len(gpi_cs_sizes)
    cs_sizes_df = pd.DataFrame(columns=['cs sizes', 'to celltype'], index=range(lengp_cs))
    cs_sizes_df.loc[0: len(gpe_cs_sizes) - 1, 'cs sizes'] = gpe_cs_sizes
    cs_sizes_df.loc[0: len(gpe_cs_sizes) - 1, 'to celltype'] = 'GPe'
    cs_sizes_df.loc[len(gpe_cs_sizes): lengp_cs - 1, 'cs sizes'] = gpi_cs_sizes
    cs_sizes_df.loc[len(gpe_cs_sizes): lengp_cs - 1, 'to celltype'] = 'GPi'
    cs_sizes_df = cs_sizes_df.astype({'cs sizes': float})
    #get statistics for all sizes
    stats, p_value = ranksums(gpe_cs_sizes, gpi_cs_sizes)
    ranksum_sizes.loc['stats', 'cs sizes to GPe vs GPi'] = stats
    ranksum_sizes.loc['p-value', 'cs sizes to GPe vs GPi'] = p_value
    ranksum_sizes.to_csv(f'{f_name}/ranksum_sizes.csv')
    cs_sizes_df.to_csv(f'{f_name}/syn_sizes_toGP.csv')
    # plot syn sizes independent of cells
    sns.histplot(x='cs sizes', data=cs_sizes_df, hue='to celltype', palette=gp_palette, common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True)
    plt.ylabel('number of contact sites')
    plt.xlabel('contact site mesh area [µm²]')
    plt.savefig(f'{f_name}/cssizes_to_GP_hist.png')
    plt.savefig(f'{f_name}/cssizes_to_GP_hist.svg')
    plt.close()
    sns.histplot(x='cs sizes', data=cs_sizes_df, hue='to celltype', palette=gp_palette, common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, log_scale = True)
    plt.ylabel('number of contact sites')
    plt.xlabel('contact site mesh area [µm²]')
    plt.savefig(f'{f_name}/cssizes_to_GP_log.png')
    plt.savefig(f'{f_name}/cssizes_to_GP_hist_log.svg')
    plt.close()
    sns.histplot(x='cs sizes', data=cs_sizes_df, hue='to celltype', palette=gp_palette, common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, stat = 'percent')
    plt.ylabel('% of contact sites')
    plt.xlabel('contact site mesh area [µm²]')
    plt.savefig(f'{f_name}/cssizes_to_GP_hist_perc.png')
    plt.savefig(f'{f_name}/cssizes_to_GP_hist_perc.svg')
    plt.close()
    sns.histplot(x='cs sizes', data=cs_sizes_df, hue='to celltype', palette=gp_palette, common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, log_scale = True, stat = 'percent')
    plt.ylabel('% of contact sites')
    plt.xlabel('contact site mesh area [µm²]')
    plt.savefig(f'{f_name}/cssizes_to_GP_log_perc.png')
    plt.savefig(f'{f_name}/cssizes_to_GP_hist_log_perc.svg')
    plt.close()
    sns.boxplot(data=cs_sizes_df, x='to celltype', y='cs sizes', palette=gp_palette)
    plt.ylabel('contact site mesh area [µm²]')
    plt.savefig(f'{f_name}/cssizes_to_GP_box.png')
    plt.savefig(f'{f_name}/cssizes_to_GP_box.svg')
    plt.close()
    sns.stripplot(x='to celltype', y='cs sizes', data=cs_sizes_df, color='black', alpha=0.2,
                  dodge=True, size=2)
    sns.violinplot(x='to celltype', y='cs sizes', data=cs_sizes_df, inner="box",
                   palette=gp_palette)
    plt.ylabel('contact site mesh area [µm²]')
    plt.savefig(f'{f_name}/cssizes_to_GP_violin.png')
    plt.savefig(f'{f_name}/cssizes_to_GP_violin.svg')
    plt.close()
    '''
    log.info(f'Step 6/7: Plot morphological parameters vs {ct1_str}, {ct2_str} ratio as joint plot')
    # plot histograms for all cells, GP ratio only for those connected to GP
    for key in conn_result_df.keys():
        if 'cellid' in key or 'celltype' in key or 'cs' in key:
            continue
        if f'{ct1_str}, {ct2_str} ratio' in key:
            xhist = f'{ct2_str}/({ct2_str} + {ct1_str})'
        elif 'syn size' in key or 'cs size' in key:
            xhist = f'{key} [µm2]'
        elif 'length' in key:
            xhist = f'{key} [mm]'
        elif 'diameter' in key:
            xhist = f'{key} [µm]'
        elif 'density' in key:
            xhist = f'{key} [1/µm]'
        else:
            xhist = key
        sns.histplot(x=key, data=conn_result_df, color='black', common_norm=False,
                     fill=False, element="step", linewidth=3)
        plt.ylabel('count of cells')
        plt.xlabel(xhist)
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_hist.png')
        plt.close()
        sns.histplot(x=key, data=conn_result_df, color='black', common_norm=False,
                     fill=False, element="step", linewidth=3, stat='percent')
        plt.ylabel('percent of cells')
        plt.xlabel(xhist)
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_hist_perc.png')
        plt.close()

    example_cellids = [662789385, 542544908, 1340425305,  728819856, 1200237873, 27161078]
    example_inds = np.in1d(conn_result_df['cellid'], example_cellids)
    #plot correlation between morphological parameters and GP ratio
    #calculate spearman corr
    ratio_key_list = [f'{ct1_str}, {ct2_str} ratio syn number', f'{ct1_str}, {ct2_str} ratio sum syn size',
                      f'abs({ct1_str}, {ct2_str} ratio - 0.5)', f'abs({ct1_str}, {ct2_str} area ratio - 0.5)']
    spearman_result_df = pd.DataFrame(columns = ['stats', 'p-value'])
    zero_inds = np.all([conn_result_df[f'syn number to {ct2_str}'] == 0, conn_result_df[f'syn number to {ct1_str}'] == 0], axis = 0)
    plot_corr_df = conn_result_df[zero_inds == False]
    conn_ct_str = ct_dict[conn_ct]
    for key in conn_result_df.keys():
        if 'cellid' in key or 'celltype' in key or 'mean' in key or 'cs' in key:
            continue
        if f'{ct1_str}, {ct2_str} ratio' in key and 'syn' in key and 'cs' in key:
            continue
        if ct1_str in key or ct2_str in key:
            continue
        if 'syn size' in key:
            xhist = f'{key} [µm2]'
        elif 'length' in key:
            xhist = f'{key} [mm]'
        elif'diameter' in key:
            xhist = f'{key} [µm]'
        elif 'density' in key:
            xhist = f'{key} [1/µm]'
        else:
            xhist = key
        for rkey in ratio_key_list:
            if 'abs' in rkey and not ('number cell partners' in key or 'syn number total' in key):
                continue
            g = sns.JointGrid(data=plot_corr_df, x=key, y=rkey)
            g.plot_joint(sns.kdeplot, color = ct_palette[conn_ct_str])
            g.plot_joint(sns.scatterplot, color = 'black', alpha = 0.3)
            g.plot_marginals(sns.histplot, fill=False, element = 'step',
                             kde=False, bins='auto', color='black')
            g.ax_joint.xaxis.set_major_locator(plt.MaxNLocator(5))
            g.ax_joint.yaxis.set_major_locator(plt.MaxNLocator(5))
            g.ax_joint.set_xticks(g.ax_joint.get_xticks())
            g.ax_joint.set_yticks(g.ax_joint.get_yticks())
            g.ax_joint.set_xticklabels(["%.1f" % i for i in g.ax_joint.get_xticks()], fontsize=fontsize_jointplot)
            g.ax_joint.set_yticklabels(["%.1f" % i for i in g.ax_joint.get_yticks()], fontsize=fontsize_jointplot)
            g.ax_joint.set_xlabel(xhist)
            if 'number' in rkey:
                ylabel = f'{ct2_str}/({ct2_str} + {ct1_str}) syn numbers'
            else:
                ylabel = f'{ct2_str}/({ct2_str} + {ct1_str}) syn area'
            g.ax_joint.set_ylabel(ylabel)
            plt.savefig(f'{f_name}/{key}_{rkey}_all.png', dpi = 1200)
            plt.savefig(f'{f_name}/{key}_{rkey}_all.svg')
            plt.close()
            example_x = conn_result_df[key][example_inds]
            example_y = conn_result_df[rkey][example_inds]
            plt.scatter(conn_result_df[key], conn_result_df[rkey], color='gray')
            plt.scatter(example_x, example_y, color='red')
            plt.xlabel(xhist)
            plt.ylabel(ylabel)
            plt.savefig(f'{f_name}/{key}_{rkey}_scatter_examplecells.png')
            plt.close()
            spear_res = spearmanr(plot_corr_df[key], plot_corr_df[rkey], nan_policy='omit')
            spearman_result_df.loc[f'{key} vs {rkey}', 'stats'] = spear_res[0]
            spearman_result_df.loc[f'{key} vs {rkey}', 'p-value'] = spear_res[1]

    #plot spine density vs dendritic length
    g = sns.JointGrid(data=conn_result_df, x='dendritic length', y='spine density')
    g.plot_joint(sns.kdeplot, color=ct_palette[conn_ct_str])
    g.plot_joint(sns.scatterplot, color='black', alpha=0.3)
    g.plot_marginals(sns.histplot, fill=False, element = 'step',
                     kde=False, bins='auto', color='black')
    g.ax_joint.xaxis.set_major_locator(plt.MaxNLocator(5))
    g.ax_joint.yaxis.set_major_locator(plt.MaxNLocator(5))
    g.ax_joint.set_xticks(g.ax_joint.get_xticks())
    g.ax_joint.set_yticks(g.ax_joint.get_yticks())
    g.ax_joint.set_xticklabels(["%.1f" % i for i in g.ax_joint.get_xticks()], fontsize=fontsize_jointplot)
    g.ax_joint.set_yticklabels(["%.1f" % i for i in g.ax_joint.get_yticks()], fontsize=fontsize_jointplot)
    g.ax_joint.set_xlabel('dendritic length [mm]')
    g.ax_joint.set_ylabel('spine density [1/µm]')
    plt.savefig(f'{f_name}/den_len_spine_den_all.png', dpi = 1200)
    plt.savefig(f'{f_name}/den_len_spine_den_all.svg')
    plt.close()
    spear_res = spearmanr(conn_result_df['spine density'], conn_result_df['dendritic length'], nan_policy='omit')
    spearman_result_df.loc[f'spine density vs dendritic length', 'stats'] = spear_res[0]
    spearman_result_df.loc[f'spine density vs dendritic length', 'p-value'] = spear_res[1]
    '''
    #plot also GP syn ratio vs GP cs ratio
    g = sns.JointGrid(data=conn_result_df, x='GP ratio cs number', y='GP ratio syn number')
    g.plot_joint(sns.kdeplot, color="#EAAE34")
    g.plot_joint(sns.scatterplot, color='black', alpha=0.3)
    g.plot_marginals(sns.histplot, fill=False, element = 'step',
                     kde=False, bins='auto', color='black')
    g.ax_joint.set_xticks(g.ax_joint.get_xticks())
    g.ax_joint.set_yticks(g.ax_joint.get_yticks())
    g.ax_joint.set_xticklabels(["%.2f" % i for i in g.ax_joint.get_xticks()], fontsize=fontsize_jointplot)
    g.ax_joint.set_yticklabels(["%.2f" % i for i in g.ax_joint.get_yticks()], fontsize=fontsize_jointplot)
    g.ax_joint.set_xlabel('(GPe + GPi)/GPi cs numbers')
    g.ax_joint.set_ylabel('(GPe + GPi)/GPi syn numbers')
    plt.savefig(f'{f_name}/GP_ratio_cs_vs_syn_numbers_all.png')
    plt.savefig(f'{f_name}/GP_ratio_cs_vs_syn_numbers_all.svg')
    plt.close()
    g = sns.JointGrid(data=conn_result_df, x='GP ratio sum cs size', y='GP ratio sum syn size')
    g.plot_joint(sns.kdeplot, color="#EAAE34")
    g.plot_joint(sns.scatterplot, color='black', alpha=0.3)
    g.plot_marginals(sns.histplot, fill=False,
                     kde=False, bins='auto', color='black')
    g.ax_joint.set_xticks(g.ax_joint.get_xticks())
    g.ax_joint.set_yticks(g.ax_joint.get_yticks())
    g.ax_joint.set_xticklabels(["%.2f" % i for i in g.ax_joint.get_xticks()], fontsize=fontsize_jointplot)
    g.ax_joint.set_yticklabels(["%.2f" % i for i in g.ax_joint.get_yticks()], fontsize=fontsize_jointplot)
    g.ax_joint.set_xlabel('(GPe + GPi)/GPi cs area')
    g.ax_joint.set_ylabel('(GPe + GPi)/GPi syn area')
    plt.savefig(f'{f_name}/GP_ratio_cs_vs_syn_sum_size_all.png')
    plt.savefig(f'{f_name}/GP_ratio_cs_vs_syn_sum size_all.svg')
    plt.close()
    spear_res = spearmanr(conn_result_df['GP ratio syn number'], conn_result_df['GP ratio cs number'], nan_policy='omit')
    spearman_result_df.loc['GP ratio cs vs syn number', 'stats'] = spear_res[0]
    spearman_result_df.loc['GP ratio cs vs syn number', 'p-value'] = spear_res[1]
    spear_res = spearmanr(conn_result_df['GP ratio sum syn size'], conn_result_df['GP ratio sum cs size'], nan_policy='omit')
    spearman_result_df.loc['GP ratio cs vs sum sizes', 'stats'] = spear_res[0]
    spearman_result_df.loc['GP ratio cs vs sum sizes', 'p-value'] = spear_res[1]
    '''
    spearman_result_df.to_csv(f'{f_name}/spearman_corr_results.csv')

    log.info(f'Step 7/7: Divide {ct_dict[conn_ct]} into groups depending on connectivity, plot again and compare groups')
    ct1_zero = conn_result_df[f'syn number to {ct1_str}'] == 0
    ct2_zero = conn_result_df[f'syn number to {ct2_str}'] == 0
    no_ct = np.all([ct1_zero, ct2_zero], axis = 0)
    conn_result_df.loc[no_ct, 'celltype'] = f'{ct_dict[conn_ct]} none'
    both_ct = np.any([ct1_zero, ct2_zero], axis = 0) == False
    conn_result_df.loc[both_ct, 'celltype'] = f'{ct_dict[conn_ct]} both'
    only_ct1 = np.all([ct1_zero == False, ct2_zero], axis = 0)
    conn_result_df.loc[only_ct1, 'celltype'] = f'{ct_dict[conn_ct]} only {ct1_str}'
    only_ct2 = np.all([ct1_zero, ct2_zero == False], axis=0)
    conn_result_df.loc[only_ct2, 'celltype'] = f'{ct_dict[conn_ct]} only {ct2_str}'
    conn_result_df.to_csv(f'{f_name}/{ct_dict[conn_ct]}_spine_density_{ct1_str}_{ct2_str}ratio.csv')
    celltype_groups = conn_result_df.groupby('celltype')
    conn_subgroup_size = celltype_groups.size()
    log.info(f'There are the following {ct_dict[conn_ct]} subgroups with corresponding sizes: {conn_subgroup_size}')
    #create summary for number of subgroup and create plot
    conn_subgroup_percent = conn_subgroup_size * 100 / len(cell_ids)
    conn_groups_str = np.unique(conn_result_df['celltype'])
    conn_subgroup_df = pd.DataFrame(columns = ['celltype', f'number of {ct_dict[conn_ct]}cells', f'% of {ct_dict[conn_ct]} cells'], index = range(len(conn_groups_str)))
    conn_subgroup_df['celltype'] = conn_groups_str
    conn_subgroup_df[f'number of {ct_dict[conn_ct]} cells'] = np.array(conn_subgroup_size)
    conn_subgroup_df[f'% of {ct_dict[conn_ct]}cells'] = np.array(conn_subgroup_percent)
    conn_subgroup_df.to_csv(f'{f_name}/conn_subgroup_numbers.csv')
    conn_colors = ["#EAAE34", "black", "#707070", '#2F86A8']
    sns.barplot(data = conn_subgroup_df, x = 'celltype', y = f'number of {ct_dict[conn_ct]} cells', palette=conn_colors)
    plt.savefig(f'{f_name}/number_{ct_dict[conn_ct]}_cells_subgroup_bar.png')
    plt.savefig(f'{f_name}/number_{ct_dict[conn_ct]}_cells_subgroup_bar.svg')
    plt.close()
    # also add conn ct group to syn sizes df
    for conn_str in conn_groups_str:
        inds = np.in1d(syn_sizes_df['cellid'], conn_result_df['cellid'][conn_result_df['celltype'] == conn_str])
        syn_sizes_df.loc[inds, f'{ct_dict[conn_ct]} group'] = conn_str
    syn_sizes_df.to_csv(f'{f_name}/syn_sizes_to{ct1_str}_{ct2_str}.csv')
    #get kruskal-wallis (non-parametric test) for all keys
    kruskal_results_df = pd.DataFrame(columns = ['stats', 'p-value'])
    # get kruskal for all syn sizes between groups
    syn_sizes_groups = [group['syn sizes'].values for name, group in
                           syn_sizes_df.groupby(f'{ct_dict[conn_ct]} group')]
    kruskal_res = kruskal(*syn_sizes_groups, nan_policy='omit')
    kruskal_results_df.loc['syn sizes', 'stats'] = kruskal_res[0]
    kruskal_results_df.loc['syn sizes', 'p-value'] = kruskal_res[1]
    #get kruskal for all syn sizes dependend on if to ct1 or to ct2
    ct1_syn_sizes_df = syn_sizes_df[syn_sizes_df['to celltype'] == ct1_str]
    ct1_syn_sizes_groups = [group['syn sizes'].values for name, group in
                        ct1_syn_sizes_df.groupby(f'{ct_dict[conn_ct]} group')]
    ct2_syn_sizes_df = syn_sizes_df[syn_sizes_df['to celltype'] == ct2_str]
    ct2_syn_sizes_groups = [group['syn sizes'].values for name, group in
                            ct2_syn_sizes_df.groupby(f'{ct_dict[conn_ct]} group')]
    #get ranksum results for parameters compared between groups
    group_comps = list(combinations(range(len(conn_groups_str)), 2))
    ranksum_columns = [f'{conn_groups_str[gc[0]]} vs {conn_groups_str[gc[1]]}' for gc in group_comps]
    ranksum_group_df = pd.DataFrame(columns= ranksum_columns)
    #also do ranksum on syn sizes
    syn_sizes_group_str = np.unique(syn_sizes_df[f'{ct_dict[conn_ct]} group'])
    syn_group_combs = list(combinations(range(len(syn_sizes_group_str)), 2))
    for sg in syn_group_combs:
        ranksum_res = ranksums(syn_sizes_groups[sg[0]], syn_sizes_groups[sg[1]])
        ranksum_group_df.loc[f'syn sizes stats', f'{syn_sizes_group_str[sg[0]]} vs {syn_sizes_group_str[sg[1]]}'] = ranksum_res[0]
        ranksum_group_df.loc[f'syn sizes p-value', f'{syn_sizes_group_str[sg[0]]} vs {syn_sizes_group_str[sg[1]]}'] = ranksum_res[1]
    ranksum_res_ct1 = ranksums(ct1_syn_sizes_groups[0], ct2_syn_sizes_groups[1])
    ranksum_group_df.loc[f'indiv syn sizes to {ct1_str} stats', f'{conn_groups_str[0]} vs {conn_groups_str[2]}'] = ranksum_res_ct1[0]
    ranksum_group_df.loc[f'indiv syn sizes to {ct1_str} syn sizes p-value', f'{conn_groups_str[0]} vs {conn_groups_str[2]}'] = ranksum_res_ct1[
        1]
    ranksum_res_ct2 = ranksums(ct1_syn_sizes_groups[0], ct2_syn_sizes_groups[1])
    ranksum_group_df.loc[f'indiv syn sizes to {ct2_str} stats', f'{conn_groups_str[0]} vs {conn_groups_str[3]}'] = \
    ranksum_res_ct2[0]
    ranksum_group_df.loc[f'indiv syn sizes to {ct2_str} syn sizes p-value', f'{conn_groups_str[0]} vs {conn_groups_str[3]}'] = \
    ranksum_res_ct2[1]
    conn_palette = {ct: conn_colors[i] for i, ct in enumerate(conn_groups_str)}
    conn_result_df = conn_result_df[conn_result_df['dendritic length'] > 0]
    conn_result_df = conn_result_df.astype({'spine density': float, 'dendritic length': float,
                                            'number primary dendrites': int, 'number branching points': int,
                                            'ratio branching points vs primary dendrites': float,
                                            f'{ct1_str}, {ct2_str} ratio syn number': float, f'{ct1_str}, {ct2_str} ratio sum syn size': float,
                                            f'syn number to {ct1_str}': int, f'syn number to {ct2_str}': int, f'abs({ct1_str}, {ct2_str} ratio - 0.5)': float,
                                            f'abs({ct1_str}, {ct2_str} area ratio - 0.5)': float, 'syn number total':int})
    for key in conn_result_df.keys():
        if 'celltype' in key or 'cellid' in key or 'cs' in key:
            continue
        if f'{ct1_str}, {ct2_str} ratio' in key:
            axis_label = f'{ct2_str}/ ({ct2_str} + {ct1_str})'
        elif 'syn size' in key:
            axis_label = f'{key} [µm2]'
        elif 'length' in key or 'diameter' in key:
            axis_label = f'{key} [µm]'
        elif 'density' in key:
            axis_label = f'{key} [1/µm]'
        else:
            axis_label = key
        #get kruskal results for key
        celltype_key_groups = [group[key].values for name, group in
                               conn_result_df.groupby('celltype')]
        kruskal_res = kruskal(*celltype_key_groups, nan_policy='omit')
        kruskal_results_df.loc[key, 'stats'] = kruskal_res[0]
        kruskal_results_df.loc[key, 'p-value'] = kruskal_res[1]
        for gc in group_comps:
            ranksum_res = ranksums(celltype_key_groups[gc[0]], celltype_key_groups[gc[1]])
            ranksum_group_df.loc[f' {key} stats', f'{conn_groups_str[gc[0]]} vs {conn_groups_str[gc[1]]}'] = ranksum_res[0]
            ranksum_group_df.loc[f' {key} p-value', f'{conn_groups_str[gc[0]]} vs {conn_groups_str[gc[1]]}'] = ranksum_res[1]
        sns.barplot(data = conn_result_df, x = 'celltype', y = key, palette=conn_palette)
        plt.title(key)
        plt.ylabel(axis_label)
        plt.savefig(f'{f_name}/{key}_overview_bar.png')
        plt.savefig(f'{f_name}/{key}_overview_bar.svg')
        plt.close()
        sns.boxplot(data=conn_result_df, x='celltype', y=key, palette=conn_palette)
        plt.title(key)
        plt.savefig(f'{f_name}/{key}_overview_box.png')
        plt.savefig(f'{f_name}/{key}_overview_box.svg')
        plt.ylabel(axis_label)
        plt.close()
        sns.stripplot(x='celltype', y=key, data=conn_result_df, color='black', alpha=0.2,
                      dodge=True, size=2)
        sns.violinplot(x='celltype', y=key, data=conn_result_df, inner="box",
                           palette=conn_palette)
        plt.title(key)
        plt.ylabel(axis_label)
        plt.savefig(f'{f_name}/{key}_overview_violin.png')
        plt.savefig(f'{f_name}/{key}_overview_violin.svg')
        plt.close()
        sns.histplot(x=key, data=conn_result_df, hue = 'celltype', palette=conn_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True)
        plt.ylabel('number of cells')
        plt.xlabel(axis_label)
        plt.savefig(f'{f_name}/{key}_celltype_hist.png')
        plt.savefig(f'{f_name}/{key}_celltype_hist.svg')
        plt.close()
        sns.histplot(x=key, data=conn_result_df, hue='celltype', palette=conn_palette, common_norm=False,
                     fill=False, element="step", linewidth=3, legend=True, stat='percent')
        plt.ylabel('% of cells')
        plt.xlabel(axis_label)
        plt.savefig(f'{f_name}/{key}_celltype_hist_perc.png')
        plt.savefig(f'{f_name}/{key}_celltype_hist_perc.svg')
        plt.close()

    kruskal_results_df.to_csv(f'{f_name}/kruskal_results.csv')
    ranksum_group_df.to_csv(f'{f_name}/ranksums_{ct_dict[conn_ct]}_groups_results.csv')

    #overlay jointplot with density plot and plot again
    ratio_key_list = [f'{ct1_str}, {ct2_str} ratio syn number', f'{ct1_str}, {ct2_str} ratio sum syn size']
    for rkey in ratio_key_list:
        g = sns.JointGrid(data=conn_result_df, x='spine density', y=rkey, hue="celltype", palette=conn_palette
        )
        g.plot_joint(sns.kdeplot, hue = 'celltype', palette = conn_palette
        )
        g.plot_joint(sns.scatterplot, color = 'black', alpha = 0.3)
        g.plot_marginals(sns.histplot, fill=True, alpha=0.3,
                         kde=False, bins='auto', palette=conn_palette
                         )
        g.ax_joint.set_xticks(g.ax_joint.get_xticks())
        g.ax_joint.set_yticks(g.ax_joint.get_yticks())
        g.ax_joint.set_xticklabels(["%.2f" % i for i in g.ax_joint.get_xticks()], fontsize=fontsize_jointplot)
        g.ax_joint.set_yticklabels(["%.2f" % i for i in g.ax_joint.get_yticks()], fontsize=fontsize_jointplot)
        g.ax_joint.set_xlabel('spine density [1/µm]')
        if 'number' in rkey:
            g.ax_joint.set_ylabel(f'{ct2_str}/ ({ct2_str} + {ct1_str}) syn numbers')
        else:
            g.ax_joint.set_ylabel(f'{ct2_str}/ ({ct2_str} + {ct1_str}) syn area')
        plt.savefig(f'{f_name}/spine_density_{rkey}.png')
        plt.savefig(f'{f_name}/spine_density_{rkey}.svg')
        plt.close()

    #plot syn sizes of individual synapses for these groups
    sns.histplot(x='syn sizes', data=syn_sizes_df, hue=f'{ct_dict[conn_ct]} group', palette=conn_palette, common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, stat='percent')
    plt.ylabel('% of synapses')
    plt.xlabel('synaptic mesh area [µm²]')
    plt.savefig(f'{f_name}/synsizes_to_{ct1_str}_{ct2_str}_{ct_dict[conn_ct]}_groups_hist_perc.png')
    plt.savefig(f'{f_name}/synsizes_to_{ct1_str}_{ct2_str}_{ct_dict[conn_ct]}_groups_hist_perc.svg')
    plt.close()
    sns.histplot(x='syn sizes', data=syn_sizes_df, hue=f'{ct_dict[conn_ct]} group', palette=conn_palette, common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, log_scale=True, stat='percent')
    plt.ylabel('% of synapses')
    plt.xlabel('synaptic mesh area [µm²]')
    plt.savefig(f'{f_name}/synsizes_to_{ct1_str}_{ct2_str}_{ct_dict[conn_ct]}_groups_hist_log_perc.png')
    plt.savefig(f'{f_name}/synsizes_to_{ct1_str}_{ct2_str}_{ct_dict[conn_ct]}_groups_hist_log_perc.svg')
    plt.close()
    #only to ct1
    sns.histplot(x='syn sizes', data=ct1_syn_sizes_df, hue=f'{ct_dict[conn_ct]} group', palette=conn_palette, common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, stat='percent')
    plt.ylabel('% of synapses')
    plt.xlabel('synaptic mesh area [µm²]')
    plt.title(f'Syn sizes to {ct1_str}')
    plt.savefig(f'{f_name}/synsizes_to_{ct1_str}_{ct_dict[conn_ct]}_groups_hist_perc.png')
    plt.savefig(f'{f_name}/synsizes_to_{ct1_str}_{ct_dict[conn_ct]}_groups_hist_perc.svg')
    plt.close()
    sns.histplot(x='syn sizes', data=ct1_syn_sizes_df, hue=f'{ct_dict[conn_ct]} group', palette=conn_palette, common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, log_scale=True, stat='percent')
    plt.ylabel('% of synapses')
    plt.xlabel('synaptic mesh area [µm²]')
    plt.title(f'Syn sizes to {ct1_str}')
    plt.savefig(f'{f_name}/synsizes_to_{ct1_str}_{ct_dict[conn_ct]}_groups_hist_log_perc.png')
    plt.savefig(f'{f_name}/synsizes_to_{ct1_str}_{ct_dict[conn_ct]}_groups_hist_log_perc.svg')
    plt.close()
    # only to ct2
    sns.histplot(x='syn sizes', data=ct2_syn_sizes_df, hue=f'{ct_dict[conn_ct]} group', palette=conn_palette, common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, stat='percent')
    plt.ylabel('% of synapses')
    plt.xlabel('synaptic mesh area [µm²]')
    plt.title(f'Syn sizes to {ct2_str}')
    plt.savefig(f'{f_name}/synsizes_to_{ct2_str}_{ct_dict[conn_ct]}_groups_hist_perc.png')
    plt.savefig(f'{f_name}/synsizes_to_{ct2_str}_{ct_dict[conn_ct]}_groups_hist_perc.svg')
    plt.close()
    sns.histplot(x='syn sizes', data=ct2_syn_sizes_df, hue=f'{ct_dict[conn_ct]} group', palette=conn_palette, common_norm=False,
                 fill=False, element="step", linewidth=3, legend=True, log_scale=True, stat='percent')
    plt.ylabel('% of synapses')
    plt.xlabel('synaptic mesh area [µm²]')
    plt.title(f'Syn sizes to {ct2_str}')
    plt.savefig(f'{f_name}/synsizes_to_{ct2_str}_{ct_dict[conn_ct]}_groups_hist_log_perc.png')
    plt.savefig(f'{f_name}/synsizes_to_{ct2_str}_{ct_dict[conn_ct]}_groups_hist_log_perc.svg')
    plt.close()

    log.info(f'{ct_dict[conn_ct]} subgroup analysis of spine density and ratio done')
