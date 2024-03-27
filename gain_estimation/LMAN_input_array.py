#now write function here that uses information aboout firing rate from literature
#for INT1-3 uses values estimated with mitochondria volume density

#get one exmaple cell (e.g. large LMAN) and get its connectivity to full cells
#make this into array that incorporates postsynaptic cells surface area and spike threshold

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_general, \
        get_percell_number_sumsize
    from cajal.nvmescratch.users.arother.bio_analysis.general.result_helper import ConnMatrix
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from syconn.reps.segmentation import SegmentationDataset
    import os as os
    import pandas as pd
    import numpy as np
    import seaborn as sns
    from tqdm import tqdm

    # ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
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
    fontsize = 14
    min_syn_size = 0.1
    syn_prob_thresh = 0.6
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/gain_estimation/240327_j0251{version}_percell_input_matrix_mcl_%i_%s_fs%i" % (
        min_comp_len_cell, color_key, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('per_cell_conn_matrix_log', log_dir=f_name + '/logs/')
    log.info('Get per cell connectivity matrix')
    log.info(
        "min_comp_len = %i for full cells, colors = %s" % (
            min_comp_len_cell, color_key))
    log.info(f'min syn size = {min_syn_size} µm², syn prob threshold = {syn_prob_thresh}')
    input_cellids = [3171878]
    log.info(f'Calculate for LMAn example cellid: {input_cellids}')

    log.info('Step 1/4: Load dendritic somatic surface area from full cells')
    fontsize_denso = 20
    den_so_surface_area_path = f"cajal/scratch/users/arother/bio_analysis_results/gain_estimation/240325_j0251{version}_ct_den_so_surface_area_mcl_%i_%s_fs%i" % (
        min_comp_len_cell, color_key, fontsize_denso)
    denso_surface_area_df = pd.read_csv(f'{den_so_surface_area_path}/den_so_surface_area.csv')
    log.info(
        f'Step 2/5: matrix for dendritic and somatic surface area of full cells loaded from {den_so_surface_area_path}')
    # order surface area according to cellids 8similar ordering as matrix)
    denso_cellids = denso_surface_area_df['cellid']
    denso_cellids_order_inds = np.argsort(denso_cellids)
    denso_surface_area_df_ordered = denso_surface_area_df.loc[denso_cellids_order_inds]
    full_cell_ids = np.array(denso_surface_area_df_ordered['cellid'])

    log.info('Step 2/4: Prefilter synapses')
    all_suitable_ids = np.hstack([input_cellids, full_cell_ids])
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    # filter synapses for min_syn_size, syn_prob_thresh
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob = filter_synapse_caches_general(
        sd_synssv,
        syn_prob_thresh=syn_prob_thresh,
        min_syn_size=min_syn_size)
    synapse_cache = [m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob]
    # get rid of all synapses that no suitable cell of ct is involved in
    suit_ct_inds = np.all(np.in1d(m_ssv_partners, all_suitable_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[suit_ct_inds]
    m_ids = m_ids[suit_ct_inds]
    m_ssv_partners = m_ssv_partners[suit_ct_inds]
    m_sizes = m_sizes[suit_ct_inds]
    m_axs = m_axs[suit_ct_inds]
    # only use axo-dendritic synapses
    axs_inds = np.any(m_axs == 1, axis=1)
    m_cts = m_cts[axs_inds]
    m_ids = m_ids[axs_inds]
    m_axs = m_axs[axs_inds]
    m_ssv_partners = m_ssv_partners[axs_inds]
    m_sizes = m_sizes[axs_inds]
    den_so = np.array([0, 2])
    den_so_inds = np.any(np.in1d(m_axs, den_so).reshape(len(m_axs), 2), axis=1)
    m_cts = m_cts[den_so_inds]
    m_ids = m_ids[den_so_inds]
    m_axs = m_axs[den_so_inds]
    m_ssv_partners = m_ssv_partners[den_so_inds]
    m_sizes = m_sizes[den_so_inds]
    #make sure input cellids are only on axon side
    testct = np.in1d(m_ssv_partners, input_cellids).reshape(len(m_ssv_partners), 2)
    testax = np.in1d(m_axs, [1]).reshape(len(m_ssv_partners), 2)
    in_ax_inds = np.all(testct == testax, axis=1)
    m_ssv_partners = m_ssv_partners[in_ax_inds]
    m_axs = m_axs[in_ax_inds]
    m_sizes = m_sizes[in_ax_inds]

    log.info('Step 3/4: Get connectivity between each cell-pair')
    # make matrix where entry is synaptic area between two cells.
    # row is postsynapse
    input_matrix = pd.DataFrame(columns=input_cellids, index=full_cell_ids)
    # get all synapses where cell is part of
    for input_cellid in input_cellids:
        cell_inds = np.any(np.in1d(m_ssv_partners, input_cellid).reshape(len(m_ssv_partners), 2), axis=1)
        cell_ssv_partners = m_ssv_partners[cell_inds]
        cell_sizes = m_sizes[cell_inds]
        cell_axs = m_axs[cell_inds]
        # get ssvs cell inputs to
        post_ssvs = cell_ssv_partners[np.where(cell_axs != 1)]
        # get summed syn size for each cell-pair
        syn_numbers, syn_ssv_sizes, unique_ssv_ids = get_percell_number_sumsize(post_ssvs, cell_sizes)
        order_inds = np.argsort(unique_ssv_ids)
        ordered_pre_ids = unique_ssv_ids[order_inds]
        ordered_sizes = syn_ssv_sizes[order_inds]
        matrix_col_inds = np.in1d(full_cell_ids, ordered_pre_ids)
        input_matrix.loc[matrix_col_inds, input_cellid] = ordered_sizes

    input_matrix = input_matrix.fillna(0)
    input_matrix.to_csv(f'{f_name}/input_matrix_conn_percell.csv')
    log.info(f'max snyaptic area between two cells = {input_matrix.max().max():.2f} µm²')

    log.info('Step 4/4: Plot as heatmap')
    cmap_heatmap = sns.light_palette('black', as_cmap=True)
    inc_numbers_abs = ConnMatrix(data=input_matrix.transpose().astype(float),
                                 title='Summed syn sizes per cell', filename=f_name, cmap=cmap_heatmap)
    inc_numbers_abs.get_heatmap(save_svg=True, annot=False, fontsize=fontsize)

    log.info('Step 5/5: Get matrix calculated with dendritic, somatic surface areas, dt_spike_thresholds and syn value')
    # values for threshold differences to action potential
    # use values from Farries and Perkel, 2000
    # msn = -34.5, FS = -44.9 -> use for all INT types, LTS: -43.3
    # Farries and Perkel., 2002:
    # resting potential: msn = -72, LTS = -42, FS: -62, TAN = -51, GP = -47.7
    # AP threshold = msn -37.7, LTS: -43.5, FS: -40.9, TAN = -43.7, GP = NA
    # use this to calculate values, INT1-3 use FS, GP use 60 as resting potential, STN use also FS value
    dt_spike_threshold_ct = {'MSN': 34.3, 'STN': 21.1, 'TAN': 7.3, 'GPe': 12.3, 'GPi': 12.3, 'LTS': 1.5, 'INT1': 21.1,
                             'INT2': 21.1, 'INT3': 21.1}
    log.info(f'These spike thresholds are used in the matrix {dt_spike_threshold_ct}')
    # value to estimate syn current from surface area; from Holler et al., 2021
    syn_curr_val = 1.19
    input_matrix_res = input_matrix * syn_curr_val
    for ct in list(dt_spike_threshold_ct.keys()):
        ct_inds = np.where(denso_surface_area_df_ordered['celltype'] == ct)[0]
        denso_surface_area_df_ordered.loc[ct_inds, 'res variable'] = denso_surface_area_df_ordered.loc[ct_inds, 'denso surface area'] * dt_spike_threshold_ct[ct]

    log.info('Divide input matrix by dendritic and somatic surface area and dt_spice_threshold')
    matrix_values = np.array(input_matrix_res)
    res_vector = np.array(denso_surface_area_df_ordered['res variable'])
    res_matrix = matrix_values / res_vector.reshape(-1, 1)
    input_matrix_res = pd.DataFrame(data=res_matrix, columns=input_cellids, index=full_cell_ids)

    input_matrix_res.to_csv(f'{f_name}/input_matrix.csv')
    np.save(f'{f_name}/input_matrix_array.npy', res_matrix)

    log.info('Step 5/5: Plot matrix as heatmap')
    cmap_heatmap = sns.diverging_palette(179,341, s= 70, l = 35, as_cmap=True)
    inc_numbers_abs = ConnMatrix(data=input_matrix_res.transpose().astype(float),
                                 title='Input matrix', filename=f_name, cmap=cmap_heatmap)
    inc_numbers_abs.get_heatmap(save_svg=True, annot=False, fontsize=fontsize, center_zero=True)

    log.info('Analyses done')