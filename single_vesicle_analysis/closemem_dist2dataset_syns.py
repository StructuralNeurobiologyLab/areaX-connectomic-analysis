#analyses of which synapses are close to
import scipy.spatial

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct, \
        filter_synapse_caches_general
    from syconn.handler.config import initialize_logging
    from syconn.reps.segmentation import SegmentationDataset
    from syconn import global_params
    from cajal.nvmescratch.users.arother.bio_analysis.general.vesicle_helper import get_non_synaptic_vesicle_coords
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from cajal.nvmescratch.users.arother.bio_analysis.general.result_helper import ConnMatrix
    import os as os
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from syconn.mp.mp_utils import start_multiprocess_imap
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.spatial import KDTree
    from syconn.handler.basics import load_pkl2obj


    #global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    #ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
    #           10: "NGF"}
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
    nonsyn_dist_threshold = 3000  # nm
    release_thresh = 2#µm
    cls = CelltypeColors(ct_dict = ct_dict)
    # color keys: 'BlRdGy', 'MudGrays', 'BlGrTe','TePkBr', 'BlYw'}
    color_key = 'TePkBrNGF'
    celltype = 0
    ct_str = ct_dict[celltype]
    fontsize = 20
    suitable_ids_only = False
    annot_matrix = True
    f_name = f"cajal/scratch/users/arother/bio_analysis_results/single_vesicle_analysis/250127_j0251{version}_{ct_str}_dist2matrix_mcl_%i_dt_%i_syn_%i_r%i_%s" % (
        min_comp_len, dist_threshold, nonsyn_dist_threshold, release_thresh, color_key)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'close_mem_2_syn_analysis_{ct_str}', log_dir=f_name + '/logs/')
    log.info("min_comp_len = %i, colors = %s" % (min_comp_len, color_key))
    log.info(f'min syn size = {min_syn_size} µm², syn prob thresh = {syn_prob_thresh}, threshold for close to membrane = {dist_threshold} nm, '
             f'non synaptic threshold = {nonsyn_dist_threshold} nm, threshold to putative release site = {release_thresh} µm')
    if full_cell:
        log.info('Only full cells will be processed')
    if with_glia:
        log.info('Putative glia synapses are not filtered out')
    else:
        log.info('Putative glia synapses are filtered out')
    if suitable_ids_only:
        id_path = 'cajal/scratch/users/arother/bio_analysis_results/general/' \
                  '240411_j0251v6_cts_percentages_mcl_200_ax50_synprob_0.60_TePkBrNGF_annot_bw_fs_20/suitable_ids_allct.pkl'
        log.info(f'Only suitable ids will be used, loaded from {id_path}')
        suitable_ids_dict = load_pkl2obj(id_path)
        suitable_ids = np.concatenate([suitable_ids_dict[key] for key in suitable_ids_dict])
        log.info(f'{len(suitable_ids)} cells were selected to be suitable for synapses')

    known_mergers = analysis_params.load_known_mergers()
    #misclassified_asto_ids = analysis_params.load_potential_astros()
    cache_name = analysis_params.file_locations
    all_cts = np.arange(0, len(ct_dict.keys()))

    log.info('Step 1/7: Filter suitable cellids')
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

    log.info('Step 2/7: Get coordinates of all close-membrane vesicles')
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

    log.info('Step 3/7: Prefilter synapses')
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    #filter for min syn size and syn prob to get all suitable synapses in dataset
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob = filter_synapse_caches_general(
        sd_synssv,
        syn_prob_thresh=syn_prob_thresh,
        min_syn_size=min_syn_size)
    #make sure only axo-dendritic or axo-somatic synapses
    axs_inds = np.any(m_axs == 1, axis=1)
    m_cts = m_cts[axs_inds]
    m_ids = m_ids[axs_inds]
    m_axs = m_axs[axs_inds]
    m_ssv_partners = m_ssv_partners[axs_inds]
    m_sizes = m_sizes[axs_inds]
    m_spiness = m_spiness[axs_inds]
    m_rep_coord = m_rep_coord[axs_inds]
    syn_prob = syn_prob[axs_inds]
    den_so = np.array([0, 2])
    den_so_inds = np.any(np.in1d(m_axs, den_so).reshape(len(m_axs), 2), axis=1)
    m_cts = m_cts[den_so_inds]
    m_ids = m_ids[den_so_inds]
    m_axs = m_axs[den_so_inds]
    m_ssv_partners = m_ssv_partners[den_so_inds]
    m_sizes = m_sizes[den_so_inds]
    m_spiness = m_spiness[den_so_inds]
    m_rep_coord = m_rep_coord[den_so_inds]
    syn_prob = syn_prob[den_so_inds]
    #if with_glia = False, filter out all synapses not between neurons
    if not with_glia:
        neuron_inds = np.all(np.in1d(m_cts, all_cts).reshape(len(m_cts), 2), axis = 1)
        m_cts = m_cts[neuron_inds]
        m_ids = m_ids[neuron_inds]
        m_axs = m_axs[neuron_inds]
        m_sizes = m_sizes[neuron_inds]
        m_ssv_partners = m_ssv_partners[neuron_inds]
        m_spiness = m_spiness[neuron_inds]
        m_rep_coord = m_rep_coord[neuron_inds]
        syn_prob = syn_prob[neuron_inds]
    if suitable_ids_only:
        suit_inds = np.all(np.in1d(m_ssv_partners, suitable_ids).reshape(len(m_ssv_partners), 2), axis = 1)
        m_cts = m_cts[suit_inds]
        m_ids = m_ids[suit_inds]
        m_axs = m_axs[suit_inds]
        m_sizes = m_sizes[suit_inds]
        m_ssv_partners = m_ssv_partners[suit_inds]
        m_spiness = m_spiness[suit_inds]
        m_rep_coord = m_rep_coord[suit_inds]
        syn_prob = syn_prob[suit_inds]
    #make sure synapses are not only between axon cts
    full_cts = all_cts[np.in1d(all_cts, axon_cts) == False]
    full_inds = np.any(np.in1d(m_cts, full_cts).reshape(len(m_cts), 2), axis = 1)
    m_cts = m_cts[full_inds]
    m_ids = m_ids[full_inds]
    m_axs = m_axs[full_inds]
    m_ssv_partners = m_ssv_partners[full_inds]
    m_sizes = m_sizes[full_inds ]
    m_spiness = m_spiness[full_inds]
    m_rep_coord = m_rep_coord[full_inds]
    syn_prob = syn_prob[full_inds]
    #now make sure, axon cell types are not post-synaptic
    testct = np.in1d(m_cts, full_cts).reshape(len(m_cts), 2)
    testax = np.in1d(m_axs, den_so).reshape(len(m_cts), 2)
    post_ct_inds = np.any(testct == testax, axis=1)
    m_cts = m_cts[post_ct_inds]
    m_ids = m_ids[post_ct_inds]
    m_axs = m_axs[post_ct_inds]
    m_ssv_partners = m_ssv_partners[post_ct_inds]
    m_sizes = m_sizes[post_ct_inds]
    m_spiness = m_spiness[post_ct_inds]
    m_rep_coord = m_rep_coord[post_ct_inds]
    syn_prob = syn_prob[post_ct_inds]
    synapse_cache = [m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob]
    #save information in dataframe for nomralisation later
    # get presynaptic cellids, celltypes
    ax_inds = np.where(m_axs == 1)
    pre_syn_cts = m_cts[ax_inds]
    pre_syn_cts_str = [ct_dict[ct] for ct in pre_syn_cts]
    denso_inds = np.where(m_axs != 1)
    post_syn_cts = m_cts[denso_inds]
    post_syn_cts_str = [ct_dict[ct] for ct in post_syn_cts]
    # make dataframe with all syns, syn sizes, pre and postsynaptic celltype
    columns = ['syn size', 'celltype pre', 'celltype post']
    all_syn_df = pd.DataFrame(columns=columns, index=range(len(m_sizes)))
    all_syn_df['syn size'] = m_sizes
    all_syn_df['celltype pre'] = pre_syn_cts_str
    all_syn_df['celltype post'] = post_syn_cts_str
    #now get only synapses outgoing from celltype to see which vesicles are non-synaptic
    ct_syn_cts, ct_syn_ids, ct_syn_axs, ct_syn_ssv_partners, ct_syn_sizes, ct_syn_spiness, ct_syn_rep_coord = filter_synapse_caches_for_ct(
        pre_cts=[celltype],
        post_cts=None,
        syn_prob_thresh=None,
        min_syn_size=None,
        axo_den_so=True,
        synapses_caches=synapse_cache)
    # filter so that only filtered cellids are included and are all presynaptic
    ct_inds = np.in1d(ct_syn_ssv_partners, cellids_close_mem).reshape(len(ct_syn_ssv_partners), 2)
    comp_inds = np.in1d(ct_syn_axs, 1).reshape(len(ct_syn_ssv_partners), 2)
    filtered_inds = np.all(ct_inds == comp_inds, axis=1)
    ct_syn_coords = ct_syn_rep_coord[filtered_inds]
    ct_syn_axs = ct_syn_axs[filtered_inds]
    ct_syn_ssv_partners = ct_syn_ssv_partners[filtered_inds]
    log.info(f'There are {len(m_sizes)} synapses in total, {len(ct_syn_coords)} are outgoing from {ct_str}')

    log.info('Step 4/7: Get non-synaptic vesicles')
    cell_input = [[cellid, ct_ves_ids, ct_ves_coords, ct_ves_map2ssvids, ct_syn_coords, nonsyn_dist_threshold] for cellid in cellids_close_mem]
    cell_output = start_multiprocess_imap(get_non_synaptic_vesicle_coords, cell_input)
    cell_output = np.array(cell_output, dtype='object')
    #output still list of arrays
    non_syn_ves_ids = cell_output[:, 0]
    non_syn_ves_coords = cell_output[:, 1]
    #get all ves ids
    non_syn_ves_ids_con = np.concatenate(non_syn_ves_ids)
    non_syn_ves_coords_con = np.concatenate(non_syn_ves_coords)
    log.info(f'{len(non_syn_ves_ids_con)} vesicles are close-membrane non-synaptic ({100*len(non_syn_ves_ids_con)/len(ct_ves_ids):.2f} %)')
    #make dataframe with all vesicles
    columns = ['cellid', 'ves coord x', 'ves coord y', 'ves coord z']
    non_syn_ves_coords_df = pd.DataFrame(columns = columns, index=non_syn_ves_ids_con)
    for i, cellid in enumerate(tqdm(cellids_close_mem)):
        cell_ves_ids = non_syn_ves_ids[i]
        cell_ves_coords = non_syn_ves_coords[i]
        non_syn_ves_coords_df.loc[cell_ves_ids, 'cellid'] = cellid
        non_syn_ves_coords_df.loc[cell_ves_ids, 'ves coord x'] = cell_ves_coords[:, 0]
        non_syn_ves_coords_df.loc[cell_ves_ids, 'ves coord y'] = cell_ves_coords[:, 1]
        non_syn_ves_coords_df.loc[cell_ves_ids, 'ves coord z'] = cell_ves_coords[:, 2]

    non_syn_ves_coords_df.to_csv(f'{f_name}/non_syn_{nonsyn_dist_threshold}_close_mem_{dist_threshold}_{ct_str}_ves_coords.csv')

    log.info(f'Step 5/7: Get synapses within {release_thresh} µm to close-membrane vesicles')
    non_syn_ves_coords_con_nm = non_syn_ves_coords_con * global_params.config['scaling']
    syn_coords_nm = m_rep_coord * global_params.config['scaling']
    #to do: check if correct indices and if easier to use query_ball_tree
    syns_tree = KDTree(syn_coords_nm)
    ves_tree = KDTree(non_syn_ves_coords_con_nm)
    synapse_inds = ves_tree.query_ball_tree(syns_tree, r = release_thresh * 1000)
    synapse_inds = np.unique(np.concatenate(synapse_inds).astype(np.uint32))
    close_syn_sizes = m_sizes[synapse_inds]
    close_syn_ssv_partners = m_ssv_partners[synapse_inds]
    close_syn_axs = m_axs[synapse_inds]
    close_syn_cts = m_cts[synapse_inds]
    close_syn_coords = m_rep_coord[synapse_inds]
    log.info(f'{len(close_syn_sizes)} synapses are close to a close-mebrane vesicle (dist <= {release_thresh} µm; '
             f'({100 * len(close_syn_sizes)/ len(m_sizes):.2f} %)')

    #get presynaptic cellids, celltypes
    ax_inds = np.where(close_syn_axs == 1)
    pre_syn_cellids_close = close_syn_ssv_partners[ax_inds]
    pre_syn_cts_close = close_syn_cts[ax_inds]
    pre_syn_cts_close_str = [ct_dict[ct] for ct in pre_syn_cts_close]
    denso_inds = np.where(close_syn_axs != 1)
    post_syn_cellids_close = close_syn_ssv_partners[denso_inds]
    post_syn_cts_close = close_syn_cts[denso_inds]
    post_syn_cts_close_str = [ct_dict[ct] for ct in post_syn_cts_close]
    #make dataframe with all syns, syn sizes, pre and postsynaptic celltype
    columns = ['syn size', 'celltype pre', 'cellid pre', 'celltype post', 'cellid post', 'coord x', 'coord y', 'coord z']
    syn_close_df = pd.DataFrame(columns=columns, index = range(len(close_syn_sizes)))
    syn_close_df['syn size'] = close_syn_sizes
    syn_close_df['celltype pre'] = pre_syn_cts_close_str
    syn_close_df['cellid pre'] = pre_syn_cellids_close
    syn_close_df['celltype post'] = post_syn_cts_close_str
    syn_close_df['cellid post'] = post_syn_cellids_close
    syn_close_df['coord x'] = close_syn_coords[:, 0]
    syn_close_df['coord y'] = close_syn_coords[:, 1]
    syn_close_df['coord z'] = close_syn_coords[:, 2]
    syn_close_df.to_csv(f'{f_name}/syn_close_df_dist{release_thresh}µm_{ct_str}.csv')

    log.info('Step 6/7: Get overview params and normalisations')
    ov_cts_str = np.unique(all_syn_df['celltype pre'])
    #get per pre and post celltype number and sum size of synapse
    #also get for all syns to calculate normalised one
    pre_post_ov_columns = ['number pre', 'sum size pre', 'number post', 'sum size post', 'celltype']
    all_syns_pre_post_overview_df = pd.DataFrame(columns = pre_post_ov_columns, index=ov_cts_str)
    all_syns_pre_post_overview_df['celltype'] = ov_cts_str
    all_syns_pre_group = all_syn_df.groupby('celltype pre')
    all_syns_post_group = all_syn_df.groupby('celltype post')
    pre_nums = all_syns_pre_group.size()
    all_syns_pre_post_overview_df.loc[pre_nums.index, 'number pre'] = pre_nums.values
    post_nums = all_syns_post_group.size()
    all_syns_pre_post_overview_df.loc[post_nums.index, 'number post'] = post_nums.values
    pre_areas = all_syns_pre_group['syn size'].sum()
    all_syns_pre_post_overview_df.loc[pre_areas.index, 'sum size pre'] = pre_areas.values
    post_areas = all_syns_post_group['syn size'].sum()
    all_syns_pre_post_overview_df.loc[post_areas.index, 'sum size post'] = post_areas.values
    all_syns_pre_post_overview_df.to_csv(f'{f_name}/all_syns_pre_post_ov.csv')
    #now get overview for close syns
    close_syns_pre_post_overview_df = pd.DataFrame(columns=pre_post_ov_columns, index=ov_cts_str)
    close_syns_pre_post_overview_df['celltype'] = ov_cts_str
    close_syns_pre_group = syn_close_df.groupby('celltype pre')
    close_syns_post_group = syn_close_df.groupby('celltype post')
    close_pre_nums = close_syns_pre_group.size()
    close_syns_pre_post_overview_df.loc[close_pre_nums.index, 'number pre'] = close_pre_nums.values
    close_post_nums = close_syns_post_group.size()
    close_syns_pre_post_overview_df.loc[close_post_nums.index, 'number post'] = close_post_nums.values
    close_pre_areas = close_syns_pre_group['syn size'].sum()
    close_syns_pre_post_overview_df.loc[close_pre_areas.index, 'sum size pre'] = close_pre_areas.values
    close_post_areas = close_syns_post_group['syn size'].sum()
    close_syns_pre_post_overview_df.loc[close_post_areas.index, 'sum size post'] = close_post_areas.values
    #calculate fraction of all cells
    for param in pre_post_ov_columns:
        if 'celltype' in param:
            continue
        percentage = 100 * close_syns_pre_post_overview_df[param] / all_syns_pre_post_overview_df[param]
        close_syns_pre_post_overview_df[f'percent {param}'] = percentage
    close_syns_pre_post_overview_df.to_csv(f'{f_name}/all_syns_pre_post_ov.csv')

    #get information per pre-and postcelltype (get matrix)
    #pre celltype is index; post is column
    ct_str_list = analysis_params.ct_str(with_glia=with_glia)
    full_ct_list = [ct_dict[ct] for ct in full_cts]
    all_syns_numbers = pd.DataFrame(columns=full_ct_list, index=ct_str_list)
    all_syns_sum_sizes = pd.DataFrame(columns = full_ct_list, index=ct_str_list)
    close_syn_numbers_rel = pd.DataFrame(columns = full_ct_list, index=ct_str_list)
    close_syn_sum_sizes_rel = pd.DataFrame(columns = full_ct_list, index=ct_str_list)
    close_syn_numbers_abs = pd.DataFrame(columns = full_ct_list, index=ct_str_list)
    close_syn_sum_sizes_abs = pd.DataFrame(columns = full_ct_list, index=ct_str_list)
    for pre_ct_str in tqdm(ct_str_list):
        all_syns_pre_df = all_syn_df[all_syn_df['celltype pre'] == pre_ct_str]
        close_syns_pre_df = syn_close_df[syn_close_df['celltype pre'] == pre_ct_str]
        for post_ct_str in full_ct_list:
            all_syns_pre_post_df = all_syns_pre_df[all_syns_pre_df['celltype post'] == post_ct_str]
            all_syns_numbers.loc[pre_ct_str, post_ct_str] = len(all_syns_pre_post_df)
            all_syns_sum_sizes.loc[pre_ct_str, post_ct_str] = all_syns_pre_post_df['syn size'].sum()
            close_syns_pre_post_df = close_syns_pre_df[close_syns_pre_df['celltype post'] == post_ct_str]
            close_syn_numbers_abs.loc[pre_ct_str, post_ct_str] = len(close_syns_pre_post_df)
            close_syn_sum_sizes_abs.loc[pre_ct_str, post_ct_str] = close_syns_pre_post_df['syn size'].sum()
            if len(all_syns_pre_post_df) == 0:
                close_syn_numbers_rel.loc[pre_ct_str, post_ct_str] = 0
                close_syn_sum_sizes_rel.loc[pre_ct_str, post_ct_str] = 0
            else:
                close_syn_numbers_rel.loc[pre_ct_str, post_ct_str] = 100 * len(close_syns_pre_post_df) / len(
                    all_syns_pre_post_df)
                close_syn_sum_sizes_rel.loc[pre_ct_str, post_ct_str] = 100 * close_syns_pre_post_df['syn size'].sum() / \
                                                                       all_syns_pre_post_df['syn size'].sum()

    all_syns_numbers.to_csv(f'{f_name}/all_syn_numbers.csv')
    all_syns_sum_sizes.to_csv(f'{f_name}/all_syns_sum_sizes.csv')
    close_syn_numbers_abs.to_csv(f'{f_name}/{ct_str}_close_syns_numbers.csv')
    close_syn_sum_sizes_abs.to_csv(f'{f_name}/{ct_str}_close_syns_sum_sizes.csv')
    close_syn_numbers_rel.to_csv(f'{f_name}/{ct_str}_close_syns_numbers_percent.csv')
    close_syn_sum_sizes_rel.to_csv(f'{f_name}/{ct_str}_close_syns_sum_sizes_percent.csv')

    log.info('Step 7/7: Plot results')
    #plot results for pre and postsynaptic celltype as barplot
    ct_palette = cls.ct_palette(key = color_key)
    for param in pre_post_ov_columns:
        if 'celltype' in param:
            continue
        sns.barplot(data=close_syns_pre_post_overview_df, x='celltype', y=param, palette=ct_palette, order=ct_str_list)
        plt.ylabel(param, fontsize=fontsize)
        plt.xlabel('celltype', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.title(f'{param}')
        plt.savefig(f'{f_name}/{param}_{ct_str}_bar.svg')
        plt.savefig(f'{f_name}/{param}_{ct_str}_bar.png')
        plt.close()

    #plot matrices
    save_svg = True
    cmap_heatmap = sns.light_palette('black', as_cmap=True)
    all_syns_numbers_cm = ConnMatrix(data=all_syns_numbers.astype(int),
                                 title='Numbers of all synapses', filename=f_name, cmap=cmap_heatmap)
    all_syns_numbers_cm.get_heatmap(save_svg=save_svg, annot=annot_matrix, fontsize=fontsize)
    all_syns_sum_sizes_cm = ConnMatrix(data=all_syns_sum_sizes.astype(float),
                                     title='Summed syn area of all synapses', filename=f_name, cmap=cmap_heatmap)
    all_syns_sum_sizes_cm.get_heatmap(save_svg=save_svg, annot=annot_matrix, fontsize=fontsize)
    close_syn_numbers_abs_cm = ConnMatrix(data=close_syn_numbers_abs.astype(int),
                                     title='Numbers of synapses close to ves', filename=f_name, cmap=cmap_heatmap)
    close_syn_numbers_abs_cm.get_heatmap(save_svg=save_svg, annot=annot_matrix, fontsize=fontsize)
    close_syn_sum_sizes_abs_cm = ConnMatrix(data=close_syn_sum_sizes_abs.astype(float),
                                     title='Summed syn area of synapses close to ves', filename=f_name, cmap=cmap_heatmap)
    close_syn_sum_sizes_abs_cm.get_heatmap(save_svg=save_svg, annot=annot_matrix, fontsize=fontsize)
    close_syn_numbers_rel_cm = ConnMatrix(data=close_syn_numbers_rel.astype(float),
                                     title='% of close to ves synapse numbers', filename=f_name, cmap=cmap_heatmap)
    close_syn_numbers_rel_cm.get_heatmap(save_svg=save_svg, annot=annot_matrix, fontsize=fontsize)
    close_syn_sum_sizes_rel_cm = ConnMatrix(data=close_syn_sum_sizes_rel.astype(float),
                                     title='% of close to ves synapse areas', filename=f_name, cmap=cmap_heatmap)
    close_syn_sum_sizes_rel_cm.get_heatmap(save_svg=save_svg, annot=annot_matrix, fontsize=fontsize)

    log.info('Analysis finsihed.')
