#get per cell synaptic connectivity for all full cells

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_general, get_percell_number_sumsize
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
    fontsize = 14
    min_syn_size = 0.1
    syn_prob_thresh = 0.6
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/gain_estimation/240325_j0251{version}_percell_conn_matrix_mcl_%i_%s_fs%i" % (
        min_comp_len_cell, color_key, fontsize)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('per_cell_conn_matrix_log', log_dir=f_name + '/logs/')
    log.info('Get per cell connectivity matrix')
    log.info(
        "min_comp_len = %i for full cells, colors = %s" % (
            min_comp_len_cell, color_key))
    log.info(f'min syn size = {min_syn_size} µm², syn prob threshold = {syn_prob_thresh}')
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
    
    log.info('Step 2/4: Prefilter synapses')
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    #filter synapses for min_syn_size, syn_prob_thresh
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
    #only use axo-dendritic synapses
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

    log.info('Step 3/4: Get connectivity between each cell-pair')
    #make matrix where entry is synaptic area between two cells.
    #row is postsynapse
    ordered_suitable_ids = np.sort(all_suitable_ids)
    matrix_percell = pd.DataFrame(columns = ordered_suitable_ids, index = ordered_suitable_ids)
    for i, cellid in enumerate(tqdm(ordered_suitable_ids)):
        #get all synapses where cell is part of
        cell_inds = np.any(np.in1d(m_ssv_partners, cellid).reshape(len(m_ssv_partners), 2), axis = 1)
        cell_ssv_partners = m_ssv_partners[cell_inds]
        cell_sizes = m_sizes[cell_inds]
        cell_axs = m_axs[cell_inds]
        #get all synapses where cell is postsynapse
        testct = np.in1d(cell_ssv_partners, cellid).reshape(len(cell_ssv_partners), 2)
        testax = np.in1d(cell_axs, [2, 0]).reshape(len(cell_ssv_partners), 2)
        post_ct_inds = np.any(testct == testax, axis=1)
        postcell_ssv_partners = cell_ssv_partners[post_ct_inds]
        postcell_axs = cell_axs[post_ct_inds]
        postcell_sizes = cell_sizes[post_ct_inds]
        #get ssvs cell gets input from
        pre_ssvs = postcell_ssv_partners[np.where(postcell_axs == 1)]
        #get summed syn size for each cell-pair
        syn_numbers, syn_ssv_sizes, unique_ssv_ids = get_percell_number_sumsize(pre_ssvs, postcell_sizes)
        order_inds = np.argsort(unique_ssv_ids)
        ordered_pre_ids = unique_ssv_ids[order_inds]
        ordered_sizes = syn_ssv_sizes[order_inds]
        matrix_percell.loc[cellid, ordered_pre_ids] = ordered_sizes

    matrix_percell = matrix_percell.fillna(0)
    matrix_percell.to_csv(f'{f_name}/percell_conn_matrix.csv')
    log.info(f'max snyaptic area between two cells = {matrix_percell.max().max():.2f} µm²')

    log.info('Steo 4/4: Plot as heatmap')
    cmap_heatmap = sns.light_palette('black', as_cmap=True)
    inc_numbers_abs = ConnMatrix(data=matrix_percell.transpose().astype(float),
                                 title='Summed syn sizes per cell', filename=f_name, cmap=cmap_heatmap)
    inc_numbers_abs.get_heatmap(save_svg=True, annot=False, fontsize=fontsize)

    log.info('Analyses done')
