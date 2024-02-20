#check if synapses of two celltypes that are closely together on one axon project to the same or different cells
import scipy.spatial

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct, filter_synapse_caches_general
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    import os as os
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    import numpy as np
    from tqdm import tqdm
    from scipy.spatial import KDTree
    import seaborn as sns
    import matplotlib.pyplot as plt
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import CelltypeColors
    from scipy.stats import ranksums

    version = 'v6'
    bio_params = Analysis_Params(working_dir=global_params.wd, version=version)
    ct_dict = bio_params.ct_dict()
    global_params.wd = bio_params.working_dir()
    axon_cts = bio_params.axon_cts()
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)
    min_comp_len = 200
    min_comp_len_ax = 50
    syn_prob = 0.6
    min_syn_size = 0.1
    exclude_known_mergers = True
    #{0:'DA', 1:'LMAN', 2: 'HVC', 3:'MSN', 4:'STN', 5:'TAN', 6:'GPe', 7:'GPi', 8: 'LTS',
    #                      9:'INT1', 10:'INT2', 11:'INT3', 12:'ASTRO', 13:'OLIGO', 14:'MICRO', 15:'MIGR', 16:'FRAG'}
    #proj ct is celltype that makes synapses to ct1 and ct2, potentially in close proximity to each other
    #rec_ct is celltype that ct1 and ct2 project to and whose individual cells will be compared
    proj_ct = 1
    ct1 = 9
    ct2 = 3
    rec_ct = 6
    syn_dist_thresh = 10  # synapse distance threshold in µm
    comp_cts = [proj_ct, ct1, ct2, rec_ct]
    proj_ct_str = ct_dict[proj_ct]
    ct1_str = ct_dict[ct1]
    ct2_str = ct_dict[ct2]
    rec_ct_str = ct_dict[rec_ct]
    scaling = np.array([10, 10, 25]) #nm voxelsize
    color_key = 'STNGPINTv6'
    cls = CelltypeColors(ct_dict=ct_dict)
    ct_palette = cls.ct_palette(key=color_key)
    if np.any(np.in1d(comp_cts, axon_cts)):
        axon_ct_present = True
        f_name = f'cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/' \
                 f'240211_j0251{version}_close_syn_cellid_comp_{proj_ct_str}_{ct1_str}_{ct2_str}_{rec_ct_str}_{min_comp_len}_{min_comp_len_ax}_{syn_dist_thresh}'
    else:
        axon_ct_present = False
        f_name = f'cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/' \
                 f'240211_j0251{version}_close_syn_cellid_comp_{proj_ct_str}_{ct1_str}_{ct2_str}_{rec_ct_str}_{min_comp_len}_{syn_dist_thresh}'
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'Close synapse cells comparison {proj_ct_str}, {ct1_str}, {ct2_str}, {rec_ct_str}', log_dir=f_name + '/logs/')
    if axon_ct_present:
        log.info(f'min comp len cell = {min_comp_len}, min comp len ax = {min_comp_len_ax}, exclude known mergers = {exclude_known_mergers}, '
                 f'syn prob threshold = {syn_prob}, min synapse size = {min_syn_size}, threshold for distance of synapses from same axon: {syn_dist_thresh} µm')
    else:
        log.info(
            f'min comp len cell = {min_comp_len}, exclude known mergers = {exclude_known_mergers}, '
            f'syn prob threshold = {syn_prob}, min synapse size = {min_syn_size}, threshold for distance of synapses from same axon: {syn_dist_thresh} µm')

    log.info("Step 1/7: Load celltypes and check suitability")
    cts_str_analysis = [ct_dict[i] for i in comp_cts]
    if exclude_known_mergers:
        known_mergers = bio_params.load_known_mergers()
    suitable_ids_dict = {}
    all_suitable_ids = []
    for ct in tqdm(comp_cts):
        ct_str = ct_dict[ct]
        cell_dict = bio_params.load_cell_dict(ct)
        # get ids with min compartment length
        cellids = np.array(list(cell_dict.keys()))
        if exclude_known_mergers:
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            if ct == 2:
                misclassified_asto_ids = bio_params.load_potential_astros()
                astro_inds = np.in1d(cellids, misclassified_asto_ids) == False
                cellids = cellids[astro_inds]
        if ct in axon_cts:
            cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len_ax,
                                                    axon_only=True,
                                                    max_path_len=None)
        else:
            cellids_checked = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                    axon_only=False,
                                                    max_path_len=None)
        suitable_ids_dict[ct] = cellids_checked
        all_suitable_ids.append(cellids_checked)

    number_ids = [len(suitable_ids_dict[ct]) for ct in comp_cts]
    all_suitable_ids = np.hstack(all_suitable_ids)
    log.info(f"Suitable ids from celltypes {cts_str_analysis} were selected: {number_ids}")

    log.info('Step 2/7: Prefilter synapses for syn_prob, min_syn_size and suitable cellids')
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob = filter_synapse_caches_general(
        sd_synssv,
        syn_prob_thresh=syn_prob,
        min_syn_size=min_syn_size)
    # prefilter so that all synapses are between suitable ids
    suit_ids_ind = np.all(np.in1d(m_ssv_partners, all_suitable_ids).reshape(len(m_ssv_partners), 2), axis=1)
    m_ssv_partners = m_ssv_partners[suit_ids_ind]
    m_ids = m_ids[suit_ids_ind]
    m_sizes = m_sizes[suit_ids_ind]
    m_axs = m_axs[suit_ids_ind]
    m_rep_coord = m_rep_coord[suit_ids_ind]
    m_spiness = m_spiness[suit_ids_ind]
    m_cts = m_cts[suit_ids_ind]
    syn_prob = syn_prob[suit_ids_ind]
    synapse_cache = [m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob]

    log.info(f'Step 3/7: Get synapses from {proj_ct_str} to {ct1_str} and {ct2_str}')
    #proj_ct to ct1
    proj2ct1_cts, proj2ct1_syn_ids, proj2ct1_axs, proj2ct1_ssv_partners, proj2ct1_sizes, proj2ct1_spiness, proj2ct1_rep_coord = filter_synapse_caches_for_ct(pre_cts=[proj_ct],
                                                                                                        post_cts=[ct1],
                                                                                                        syn_prob_thresh=None,
                                                                                                        min_syn_size=None,
                                                                                                        axo_den_so=True,
                                                                                                        synapses_caches=synapse_cache)
    # get all cellids of proj_ct make synapses
    axo_inds = np.where(proj2ct1_axs == 1)
    axo_ssv_partners = proj2ct1_ssv_partners[axo_inds]
    proj_ids_2ct1 = np.unique(axo_ssv_partners)
    # check that all of them are really ct1 suitable ids
    assert (np.all(np.in1d(proj_ids_2ct1, suitable_ids_dict[proj_ct])))
    # get all cellids of ct1 that receive synapses from proj_ct
    denso_inds = np.where(proj2ct1_axs != 1)
    denso_ssv_partners = proj2ct1_ssv_partners[denso_inds]
    ct1_ids_proj = np.unique(denso_ssv_partners)
    # check that all of them are really ct1 suitable ids
    assert (np.all(np.in1d(ct1_ids_proj, suitable_ids_dict[ct1])))
    log.info(f'{len(proj_ids_2ct1)} {proj_ct_str} ids project to {len(ct1_ids_proj)} {ct1_str} ids with a total synaptic sum of {np.sum(proj2ct1_sizes):.2f} µm²')
    #proj_ct to ct2
    proj2ct2_cts, proj2ct2_syn_ids, proj2ct2_axs, proj2ct2_ssv_partners, proj2ct2_sizes, proj2ct2_spiness, proj2ct2_rep_coord = filter_synapse_caches_for_ct(
        pre_cts=[proj_ct],
        post_cts=[ct2],
        syn_prob_thresh=None,
        min_syn_size=None,
        axo_den_so=True,
        synapses_caches=synapse_cache)
    # get all cellids of proj_ct make synapses
    axo_inds = np.where(proj2ct2_axs == 1)
    axo_ssv_partners = proj2ct2_ssv_partners[axo_inds]
    proj_ids_2ct2 = np.unique(axo_ssv_partners)
    # check that all of them are really ct1 suitable ids
    assert (np.all(np.in1d(proj_ids_2ct2, suitable_ids_dict[proj_ct])))
    # get all cellids of ct1 that receive synapses from proj_ct
    denso_inds = np.where(proj2ct2_axs != 1)
    denso_ssv_partners = proj2ct2_ssv_partners[denso_inds]
    ct2_ids_proj = np.unique(denso_ssv_partners)
    # check that all of them are really ct1 suitable ids
    assert (np.all(np.in1d(ct2_ids_proj, suitable_ids_dict[ct2])))
    log.info(
        f'{len(proj_ids_2ct2)} {proj_ct_str} ids project to {len(ct2_ids_proj)} {ct2_str} ids with a total synaptic sum of {np.sum(proj2ct2_sizes):.2f} µm²')
    #get cellids that project to both celltypes and only corresponding synapses
    both_proj_ids = proj_ids_2ct1[np.in1d(proj_ids_2ct1, proj_ids_2ct2)]
    num_poth_proj_ids = len(both_proj_ids)
    log.info(f'{num_poth_proj_ids} project to {ct1_str} and {ct2_str} which is {100 * num_poth_proj_ids / len(suitable_ids_dict[proj_ct]):.2f} percent of all suitable {proj_ct_str} \n'
             f' ids ({100 * num_poth_proj_ids/ len(proj_ids_2ct1):.2f} percent of the ones projecting to {ct1_str}, \n'
             f' {100 * num_poth_proj_ids/ len(proj_ids_2ct2):.2f} percent to {ct2_str})')
    if num_poth_proj_ids < len(proj_ids_2ct1):
        both_inds = np.any(np.in1d(proj2ct1_ssv_partners, both_proj_ids).reshape(len(proj2ct1_ssv_partners), 2), axis = 1)
        proj2ct1_ssv_partners = proj2ct1_ssv_partners[both_inds]
        proj2ct1_sizes = proj2ct1_sizes[both_inds]
        proj2ct1_axs = proj2ct1_axs[both_inds]
        proj2ct1_rep_coord = proj2ct1_rep_coord[both_inds]
        proj2ct1_syn_ids = proj2ct1_syn_ids[both_inds]
    denso_inds = np.where(proj2ct1_axs != 1)
    denso_ssv_partners = proj2ct1_ssv_partners[denso_inds]
    ct1_ids_proj_both = np.unique(denso_ssv_partners)
    log.info(f'{len(ct1_ids_proj_both)} {ct1_str} cells get synapses from {proj_ct_str} ids that project also to {ct2_str} \n'
             f' This is {100 * len(ct1_ids_proj_both) / len(suitable_ids_dict[ct1]):.2f} percent of all {ct1_str} cells and '
             f' {100 * len(ct1_ids_proj_both) / len(ct1_ids_proj):.2f} percent of {ct1_str} cells that get {proj_ct_str} input')
    if num_poth_proj_ids < len(proj_ids_2ct2):
        both_inds = np.any(np.in1d(proj2ct2_ssv_partners, both_proj_ids).reshape(len(proj2ct2_ssv_partners), 2), axis=1)
        proj2ct2_ssv_partners = proj2ct2_ssv_partners[both_inds]
        proj2ct2_sizes = proj2ct2_sizes[both_inds]
        proj2ct2_axs = proj2ct2_axs[both_inds]
        proj2ct2_rep_coord = proj2ct2_rep_coord[both_inds]
        proj2ct2_syn_ids = proj2ct2_syn_ids[both_inds]
    denso_inds = np.where(proj2ct2_axs != 1)
    denso_ssv_partners = proj2ct2_ssv_partners[denso_inds]
    ct2_ids_proj_both = np.unique(denso_ssv_partners)
    log.info(f'{len(ct2_ids_proj_both)} {ct2_str} cells get synapses from {proj_ct_str} ids that project also to {ct1_str} \n'
        f' This is {100 * len(ct2_ids_proj_both) / len(suitable_ids_dict[ct2]):.2f} percent of all {ct1_str} cells and '
        f' {100 * len(ct2_ids_proj_both) / len(ct2_ids_proj):.2f} percent of {ct2_str} cells that get {proj_ct_str} input')
    log.info(f'The synaptic sum of {proj_ct_str} to {ct1_str} of ids to both celltypes is {np.sum(proj2ct1_sizes):.2f} µm².')
    log.info(f'The synaptic sum of {proj_ct_str} to {ct2_str} of ids to both celltypes is {np.sum(proj2ct2_sizes):.2f} µm².')

    log.info(f'Step 4/7: Get synapses of {ct1_str} and {ct2_str} that are with a distance of {syn_dist_thresh} µm on the same axon of {proj_ct_str}')
    #get synapses within the distance threshold via KDTree (not pathlenght on axon so far)
    syn_tree_ct1 = scipy.spatial.KDTree(proj2ct1_rep_coord * scaling)
    syn_tree_ct2 = scipy.spatial.KDTree(proj2ct2_rep_coord * scaling)
    inds_tree_ct2 = syn_tree_ct1.query_ball_tree(syn_tree_ct2, r = syn_dist_thresh * 1000)
    #remove ones that don't have any partner synapse
    non_zero_mask = [len(partners) > 0 for partners in inds_tree_ct2]
    partner_syns_ct1_sizes = proj2ct1_sizes[non_zero_mask]
    log.info(f'{len(partner_syns_ct1_sizes)} synapses have at least one synapse within {syn_dist_thresh} µm distance to a synapse from {proj_ct_str} to {ct2_str}.'
             f' This is {100 * len(partner_syns_ct1_sizes)/ len(proj2ct1_sizes):.2f} percent of synapses from {proj_ct_str} to {ct1_str}')
    num_partner_syns_ct2 = len(np.unique(np.concatenate(inds_tree_ct2)))
    log.info(
        f'{num_partner_syns_ct2} synapses have at least one synapse within {syn_dist_thresh} µm distance to a synapse from {proj_ct_str} to {ct1_str}.'
        f' This is {100 * num_partner_syns_ct2 / len(proj2ct2_sizes):.2f} percent of synapses from {proj_ct_str} to {ct2_str}')
    num_pairs = len(np.concatenate(inds_tree_ct2))
    log.info(f'In total there are {num_pairs} pairs of synapses between {proj_ct_str}-{ct1_str} and {proj_ct_str}-{ct2_str} synapses')
    #make dataframe with coordinates of partners, sizes etc
    #dictionary with pairs where proj_ct id is key and ct1, ct2ids are entries
    #make another dictionary with ct1_id as key and ct2_id as entry
    columns = [f'{proj_ct_str} id', f'{ct1_str} id', f'{ct2_str} id', f'{ct1_str} syn size', f'{ct2_str} syn size',
               f'{ct1_str} coord x', f'{ct1_str} coord y', f'{ct1_str} coord z', f'{ct2_str} coord x', f'{ct2_str} coord y', f'{ct2_str} coord z',
               f'{ct1_str} syn id', f'{ct2_str} syn id']
    syn_partners_df = pd.DataFrame(columns = columns, index = range(len(proj2ct1_ssv_partners)))
    proj_id_dict = {}
    ct1_id_dict = {}
    ct2_id_dict = {}
    ct1_proj_id_dict = {}
    ct2_proj_id_dict = {}
    ax_ind_ct1 = np.where(proj2ct1_axs == 1)[1]
    ax_ind_ct2 = np.where(proj2ct2_axs == 1)[1]
    ct1_ind = np.where(proj2ct1_axs != 1)[1]
    ct2_ind = np.where(proj2ct2_axs != 1)[1]
    for i, ct1_syn_id in enumerate(proj2ct1_syn_ids):
        #check first if all pais have same proj_ax
        if len(inds_tree_ct2[i]) == 0:
            continue
        proj_id_ct1 = proj2ct1_ssv_partners[i, ax_ind_ct1[i]]
        ct1_id = proj2ct1_ssv_partners[i, ct1_ind[i]]
        ct1_coords = proj2ct1_rep_coord[i]
        for ct2_syn_ind in inds_tree_ct2[i]:
            proj_id_ct2 = proj2ct2_ssv_partners[ct2_syn_ind, ax_ind_ct2[ct2_syn_ind]]
            ct2_id = proj2ct2_ssv_partners[ct2_syn_ind, ct2_ind[ct2_syn_ind]]
            if proj_id_ct2 != proj_id_ct1:
                continue
            if proj_id_ct1 not in list(proj_id_dict.keys()):
                proj_id_dict[proj_id_ct1] = []
            proj_id_dict[proj_id_ct1].append([ct1_id, ct2_id])
            if ct1_id not in list(ct1_id_dict.keys()):
                ct1_id_dict[ct1_id] = []
                ct1_proj_id_dict[ct1_id] = []
            ct1_id_dict[ct1_id].append(ct2_id)
            ct1_proj_id_dict[ct1_id].append(proj_id_ct1)
            if ct2_id not in list(ct2_id_dict.keys()):
                ct2_id_dict[ct2_id] = []
                ct2_proj_id_dict[ct2_id] = []
            ct2_id_dict[ct2_id].append(ct1_id)
            ct2_proj_id_dict[ct2_id].append(proj_id_ct1)
            syn_partners_df.loc[i, f'{proj_ct_str} id'] = proj_id_ct1
            syn_partners_df.loc[i, f'{ct1_str} id'] = ct1_id
            syn_partners_df.loc[i, f'{ct2_str} id'] = ct2_id
            syn_partners_df.loc[i, f'{ct1_str} syn size'] = proj2ct1_sizes[i]
            syn_partners_df.loc[i, f'{ct1_str} syn id'] = ct1_syn_id
            syn_partners_df.loc[i, f'{ct1_str} coord x'] = proj2ct1_rep_coord[i, 0]
            syn_partners_df.loc[i, f'{ct1_str} coord y'] = proj2ct1_rep_coord[i, 1]
            syn_partners_df.loc[i, f'{ct1_str} coord z'] = proj2ct1_rep_coord[i, 2]
            ct2_coords = proj2ct2_rep_coord[ct2_syn_ind]
            syn_partners_df.loc[i, f'{ct2_str} syn size'] = proj2ct2_sizes[ct2_syn_ind]
            syn_partners_df.loc[i, f'{ct2_str} syn id'] = proj2ct2_syn_ids[ct2_syn_ind]
            syn_partners_df.loc[i, f'{ct2_str} coord x'] = proj2ct2_rep_coord[ct2_syn_ind, 0]
            syn_partners_df.loc[i, f'{ct2_str} coord y'] = proj2ct2_rep_coord[ct2_syn_ind, 1]
            syn_partners_df.loc[i, f'{ct2_str} coord z'] = proj2ct2_rep_coord[ct2_syn_ind, 2]

    write_obj2pkl(f'{f_name}/{proj_ct_str}_dict2{ct1_str}_{ct2_str}.pkl', proj_id_dict)
    write_obj2pkl(f'{f_name}/{ct1_str}_dict2{ct2_str}_partners.pkl', ct1_id_dict)
    write_obj2pkl(f'{f_name}/{ct2_str}_dict2{ct1_str}_partners.pkl', ct2_id_dict)
    write_obj2pkl(f'{f_name}/{ct1_str}_dict_{proj_ct_str}_partners.pkl', ct1_proj_id_dict)
    write_obj2pkl(f'{f_name}/{ct2_str}_dict_{proj_ct_str}_partners.pkl', ct2_proj_id_dict)
    syn_partners_df = syn_partners_df.dropna()
    syn_partners_df = syn_partners_df.reset_index(drop = True)
    syn_partners_df.to_csv(f'{f_name}/{proj_ct_str}_{ct1_str}_{ct2_str}_partner_syns.csv')
    log.info(f'{len(syn_partners_df)} synapse partners exist from the same {proj_ct_str} axon. \n'
             f'This involves {len(proj_id_dict.keys())} {proj_ct_str} ids ({100 * len(proj_id_dict.keys())/ len(suitable_ids_dict[proj_ct]):.2f} percent), \n'
             f' {len(ct1_id_dict.keys())} {ct1_str} ids ({100 * len(ct1_id_dict.keys())/ len(suitable_ids_dict[ct1]):.2f} percent), \n'
             f' {len(ct2_id_dict.keys())} {ct2_str} ids ({100 * len(ct2_id_dict.keys())/ len(suitable_ids_dict[ct2]):.2f} percent).')
    num_unique_ct1_syns = len(np.unique(syn_partners_df[f'{ct1_str} syn id']))
    num_unique_ct2_syns = len(np.unique(syn_partners_df[f'{ct2_str} syn id']))
    log.info(
        f'{num_unique_ct1_syns} synapses have at least one synapse within {syn_dist_thresh} µm distance to a synapse from {proj_ct_str} to {ct2_str} from the same axon.'
        f' This is {100 * num_unique_ct1_syns / len(proj2ct1_sizes):.2f} percent of synapses from {proj_ct_str} to {ct1_str} and'
        f' {100 * num_unique_ct1_syns / len(partner_syns_ct1_sizes):.2f} percent of synapses that have a partner within the distance.')
    log.info(
        f'{num_unique_ct2_syns} synapses have at least one synapse within {syn_dist_thresh} µm distance to a synapse from {proj_ct_str} to {ct1_str} from the same axon.'
        f' This is {100 * num_unique_ct2_syns / len(proj2ct2_sizes):.2f} percent of synapses from {proj_ct_str} to {ct2_str} and'
        f' {100 * num_unique_ct2_syns / num_partner_syns_ct2:.2f} percent of synapses that have a partner within the distance.')
    #get information about how many ct2 cells one ct1 cells is connected to indirectly
    unique_ct1_cells = list(ct1_id_dict.keys())
    unique_ct2_cells = list(ct2_id_dict.keys())
    ct1_conn_partners_df = pd.DataFrame(columns = ['cellid', f'number syns from {proj_ct_str} with partner syn to {ct2_str}',
                                                   f'number {ct2_str} partners', f'number {proj_ct_str} axons'], index = range(len(unique_ct1_cells)))
    for i, ct1_id in enumerate(tqdm(unique_ct1_cells)):
        ct1_conn_partners_df.loc[i, 'cellid'] = ct1_id
        ct1_conn_partners_df.loc[i, f'number syns from {proj_ct_str} with partner syn to {ct2_str}'] = len(ct1_id_dict[ct1_id])
        ct1_conn_partners_df.loc[i, f'number {ct2_str} partners'] = len(np.unique(ct1_id_dict[ct1_id]))
        ct1_conn_partners_df.loc[i, f'number {proj_ct_str} axons'] = len(np.unique(ct1_proj_id_dict[ct1_id]))
    ct1_conn_partners_df.to_csv(f'{f_name}/{ct1_str}_number_cell_parners.csv')
    median_syn_percell = np.median(ct1_conn_partners_df[f'number syns from {proj_ct_str} with partner syn to {ct2_str}'])
    median_proj_ct_percell = np.median(ct1_conn_partners_df[f'number {proj_ct_str} axons'])
    median_otherct_percell = np.median(ct1_conn_partners_df[f'number {ct2_str} partners'])
    log.info(f'{ct1_str} cells have a median of {median_syn_percell} synapses where a {ct2_str} syn is close by on the same axon, \n'
             f' which connects them to a median of {median_otherct_percell} {ct2_str} cells with {median_proj_ct_percell} different {proj_ct_str} axons.')
    ct2_conn_partners_df = pd.DataFrame(
        columns=['cellid', f'number syns from {proj_ct_str} with partner syn to {ct1_str}',
                 f'number {ct1_str} partners', f'number {proj_ct_str} axons'], index = range(len(unique_ct2_cells)))
    for i, ct2_id in enumerate(tqdm(unique_ct2_cells)):
        ct2_conn_partners_df.loc[i, 'cellid'] = ct2_id
        ct2_conn_partners_df.loc[i, f'number syns from {proj_ct_str} with partner syn to {ct1_str}'] = len(ct2_id_dict[ct2_id])
        ct2_conn_partners_df.loc[i, f'number {ct1_str} partners'] = len(np.unique(ct2_id_dict[ct2_id]))
        ct2_conn_partners_df.loc[i, f'number {proj_ct_str} axons'] = len(np.unique(ct2_proj_id_dict[ct2_id]))
    ct2_conn_partners_df.to_csv(f'{f_name}/{ct2_str}_number_cell_parners.csv')
    median_syn_percell = np.median(
        ct2_conn_partners_df[f'number syns from {proj_ct_str} with partner syn to {ct1_str}'])
    median_proj_ct_percell = np.median(ct2_conn_partners_df[f'number {proj_ct_str} axons'])
    median_otherct_percell = np.median(ct2_conn_partners_df[f'number {ct1_str} partners'])
    log.info(
        f'{ct2_str} cells have a median of {median_syn_percell} synapses where a {ct1_str} syn is close by on the same axon, \n'
        f' which connects them to a median of {median_otherct_percell} {ct1_str} cells with {median_proj_ct_percell} different {proj_ct_str} axons.')
    #plot these parameters
    for param1, param2 in zip(ct1_conn_partners_df, ct2_conn_partners_df):
        if 'cellid' in param1:
            continue
        if 'syns' in param1:
            ylabel = 'number of synapses'
            ylabel_perc = 'percent of synapses'
        else:
            ylabel = 'number of cells'
            ylabel_perc = 'percent of cells'
        sns.histplot(x=param1, data=ct1_conn_partners_df, color=ct_palette[ct1_str], common_norm=False,
                     fill=False, element="step", linewidth=3)
        plt.ylabel(ylabel)
        plt.title(param1)
        plt.savefig(f'{f_name}/{ct1_str}_{param1}_hist.png')
        plt.close()
        sns.histplot(x=param2, data=ct2_conn_partners_df, color=ct_palette[ct2_str], common_norm=False,
                     fill=False, element="step", linewidth=3)
        plt.ylabel(ylabel)
        plt.title(param2)
        plt.savefig(f'{f_name}/{ct2_str}_{param2}_hist.png')
        plt.close()
        sns.histplot(x=param1, data=ct1_conn_partners_df, color=ct_palette[ct1_str], common_norm=False,
                     fill=False, element="step", linewidth=3, stat='percent')
        plt.ylabel(ylabel_perc)
        plt.title(param1)
        plt.savefig(f'{f_name}/{ct1_str}_{param1}_hist_perc.png')
        plt.close()
        sns.histplot(x=param2, data=ct2_conn_partners_df, color=ct_palette[ct2_str], common_norm=False,
                     fill=False, element="step", linewidth=3, stat='percent')
        plt.ylabel(ylabel_perc)
        plt.title(param2)
        plt.savefig(f'{f_name}/{ct2_str}_{param2}_hist_perc.png')
        plt.close()


    log.info(f'Step 5/7: Get synapses from {ct1_str} and {ct2_str} to {rec_ct_str}')
    #prefilter synapses
    ct12rec_cts, ct12rec_syn_ids, ct12rec_axs, ct12rec_ssv_partners, ct12rec_sizes, ct12rec_spiness, ct12rec_rep_coord = filter_synapse_caches_for_ct(
        pre_cts=[ct1],
        post_cts=[rec_ct],
        syn_prob_thresh=None,
        min_syn_size=None,
        axo_den_so=True,
        synapses_caches=synapse_cache)
    ct22rec_cts, ct22rec_syn_ids, ct22rec_axs, ct22rec_ssv_partners, ct22rec_sizes, ct22rec_spiness, ct22rec_rep_coord = filter_synapse_caches_for_ct(
        pre_cts=[ct2],
        post_cts=[rec_ct],
        syn_prob_thresh=None,
        min_syn_size=None,
        axo_den_so=True,
        synapses_caches=synapse_cache)
    #make sure only cellids that were identified in Step 3 are used
    id_inds = np.any(np.in1d(ct12rec_ssv_partners, unique_ct1_cells).reshape(len(ct12rec_ssv_partners), 2), axis = 1)
    ct12rec_syn_ids = ct12rec_syn_ids[id_inds]
    ct12rec_ssv_partners = ct12rec_ssv_partners[id_inds]
    ct12rec_sizes = ct12rec_sizes[id_inds]
    ct12rec_axs = ct12rec_axs[id_inds]
    id_inds = np.any(np.in1d(ct22rec_ssv_partners, unique_ct2_cells).reshape(len(ct22rec_ssv_partners), 2), axis=1)
    ct22rec_syn_ids = ct22rec_syn_ids[id_inds]
    ct22rec_ssv_partners = ct22rec_ssv_partners[id_inds]
    ct22rec_sizes = ct22rec_sizes[id_inds]
    ct22rec_axs = ct22rec_axs[id_inds]
    # get all cellids of proj_ct make synapses
    axo_inds = np.where(ct12rec_axs == 1)
    axo_ct1_partners = ct12rec_ssv_partners[axo_inds]
    ct1_ids_2rec = np.unique(axo_ct1_partners)
    # get all cellids of ct1 that project to rec ct
    denso_inds = np.where(ct12rec_axs != 1)
    denso_ct1_partners = ct12rec_ssv_partners[denso_inds]
    rec_ids_ct1 = np.unique(denso_ct1_partners)
    log.info(
        f'{len(ct1_ids_2rec)} {ct1_str} ids project to {len(rec_ids_ct1)} {rec_ct_str} ids with a total synaptic sum of {np.sum(ct12rec_sizes):.2f} µm²')
    log.info(f'These are {100 * len(ct1_ids_2rec)/ len(unique_ct1_cells):.2f} percent of {ct1_str} cells which have connected synapses and \n'
             f' {100 * len(rec_ids_ct1)/ len(suitable_ids_dict[rec_ct]):.2f} percent of {rec_ct_str} cells in total')
    # get all cellids of proj_ct make synapses
    axo_inds = np.where(ct22rec_axs == 1)
    axo_ct2_partners = ct22rec_ssv_partners[axo_inds]
    ct2_ids_2rec = np.unique(axo_ct2_partners)
    # get all cellids of ct2 that project to receiving ct ids
    denso_inds = np.where(ct22rec_axs != 1)
    denso_ct2_partners = ct22rec_ssv_partners[denso_inds]
    rec_ids_ct2 = np.unique(denso_ct2_partners)
    log.info(
        f'{len(ct2_ids_2rec)} {ct2_str} ids project to {len(rec_ids_ct2)} {rec_ct_str} ids with a total synaptic sum of {np.sum(ct22rec_sizes):.2f} µm²')
    log.info(
        f'These are {100 * len(ct2_ids_2rec) / len(unique_ct2_cells):.2f} percent of {ct2_str} cells which have connected synapses and \n'
        f' {100 * len(rec_ids_ct2) / len(suitable_ids_dict[rec_ct]):.2f} percent of {rec_ct_str} cells in total')
    #get information of each ct1, ct2 cell to which it is connected to
    sort_axo_inds = np.argsort(axo_ct1_partners)
    sorted_axo_ct1_partners = axo_ct1_partners[sort_axo_inds]
    sorted_denso_ct1_partners = denso_ct1_partners[sort_axo_inds]
    unique_ct1_ids, inds = np.unique(sorted_axo_ct1_partners, return_inverse=True)
    splits = np.cumsum(np.bincount(inds))[:-1]
    split_ct1_syn_rec_ids = np.split(sorted_denso_ct1_partners, splits)
    ct1_rec_id_dict = {unique_ct1_ids[i]: split_ct1_syn_rec_ids[i] for i in range(len(unique_ct1_ids))}
    sort_axo_inds = np.argsort(axo_ct2_partners)
    sorted_axo_ct2_partners = axo_ct2_partners[sort_axo_inds]
    sorted_denso_ct2_partners = denso_ct2_partners[sort_axo_inds]
    unique_ct2_ids, inds = np.unique(sorted_axo_ct2_partners, return_inverse=True)
    splits = np.cumsum(np.bincount(inds))[:-1]
    split_ct2_syn_rec_ids = np.split(sorted_denso_ct2_partners, splits)
    ct2_rec_id_dict = {unique_ct2_ids[i]: split_ct2_syn_rec_ids[i] for i in range(len(unique_ct2_ids))}
    write_obj2pkl(f'{f_name}/{ct1_str}_{rec_ct_str}_ids.pkl', ct1_rec_id_dict)
    write_obj2pkl(f'{f_name}/{ct2_str}_{rec_ct_str}_ids.pkl', ct2_rec_id_dict)
    #also save information on how many cells ct1, ct2 cells are connected to
    for i, ct1_id in enumerate(tqdm(unique_ct1_ids)):
        id_ind = np.where(ct1_conn_partners_df['cellid'] == ct1_id)[0]
        ct1_conn_partners_df.loc[id_ind, f'number {rec_ct_str} cells'] = len(np.unique(ct1_rec_id_dict[ct1_id]))
        ct1_conn_partners_df.loc[id_ind, f'number {rec_ct_str} syns'] = len(ct1_rec_id_dict[ct1_id])
    ct1_conn_partners_df = ct1_conn_partners_df.dropna()
    ct1_conn_partners_df = ct1_conn_partners_df.reset_index(drop=True)
    ct1_conn_partners_df.to_csv(f'{f_name}/{ct1_str}_number_cell_partners_{rec_ct_str}_only.csv')
    median_syn_percell = np.median(
        ct1_conn_partners_df[f'number {rec_ct_str} syns'])
    median_rec_ct_percell = np.median(ct1_conn_partners_df[f'number {rec_ct_str} cells'])
    log.info(
        f'{ct1_str} cells which have a partner cell with {ct2_str} have a median of {median_syn_percell} synapses to {rec_ct_str}, \n'
        f' which connects them to a median of {median_rec_ct_percell} {rec_ct_str} cells.')
    for i, ct2_id in enumerate(tqdm(unique_ct2_ids)):
        id_ind = np.where(ct2_conn_partners_df['cellid'] == ct2_id)[0]
        ct2_conn_partners_df.loc[id_ind, f'number {rec_ct_str} cells'] = len(np.unique(ct2_rec_id_dict[ct2_id]))
        ct2_conn_partners_df.loc[id_ind, f'number {rec_ct_str} syns'] = len(ct2_rec_id_dict[ct2_id])
    ct2_conn_partners_df.to_csv(f'{f_name}/{ct2_str}_number_cell_partners_{rec_ct_str}_only.csv')
    ct2_conn_partners_df = ct2_conn_partners_df.dropna()
    ct2_conn_partners_df = ct2_conn_partners_df.reset_index(drop=True)
    median_syn_percell = np.median(
        ct2_conn_partners_df[f'number {rec_ct_str} syns'])
    median_rec_ct_percell = np.median(ct2_conn_partners_df[f'number {rec_ct_str} cells'])
    log.info(
        f'{ct2_str} cells which have a partner cell with {ct1_str} have a median of {median_syn_percell} synapses to {rec_ct_str}, \n'
        f' which connects them to a median of {median_rec_ct_percell} {rec_ct_str} cells.')
    #plot information for number of syns to rec ct as above
    for param1 in ct1_conn_partners_df:
        if not rec_ct_str in param1:
            continue
        if 'syns' in param1:
            ylabel = 'number of synapses'
            ylabel_perc = 'percent of synapses'
        else:
            ylabel = 'number of cells'
            ylabel_perc = 'percent of cells'
        sns.histplot(x=param1, data=ct1_conn_partners_df, color=ct_palette[ct1_str], common_norm=False,
                     fill=False, element="step", linewidth=3)
        plt.ylabel(ylabel)
        plt.title(param1)
        plt.savefig(f'{f_name}/{ct1_str}_{param1}_hist.png')
        plt.close()
        sns.histplot(x=param2, data=ct2_conn_partners_df, color=ct_palette[ct2_str], common_norm=False,
                     fill=False, element="step", linewidth=3)
        plt.ylabel(ylabel)
        plt.title(param2)
        plt.savefig(f'{f_name}/{ct2_str}_{param2}_hist.png')
        plt.close()
        sns.histplot(x=param1, data=ct1_conn_partners_df, color=ct_palette[ct1_str], common_norm=False,
                     fill=False, element="step", linewidth=3, stat='percent')
        plt.ylabel(ylabel_perc)
        plt.title(param1)
        plt.savefig(f'{f_name}/{ct1_str}_{param1}_hist_perc.png')
        plt.close()
        sns.histplot(x=param2, data=ct2_conn_partners_df, color=ct_palette[ct2_str], common_norm=False,
                     fill=False, element="step", linewidth=3, stat='percent')
        plt.ylabel(ylabel_perc)
        plt.title(param2)
        plt.savefig(f'{f_name}/{ct2_str}_{param2}_hist_perc.png')
        plt.close()

    log.info(f'Step 6/7: Check if cells linked by close syns are projecting to the same {rec_ct_str} cells')
    #check for each pair of ct1 and ct2str if rec ct projections overlap
    #add binary, yes and no
    #also calculate percentage of overlap for ct1 and ct2 rec ct projection ids
    #get pairs of ct1 and ct2 str cells
    columns = [f'{ct1_str} id', f'{ct2_str} id', f'{rec_ct_str} ids different',
               f'percent of different {rec_ct_str} ids for {ct1_str}',
               f'percent of different {rec_ct_str} ids for {ct2_str}']
    ct1_ct2_pair_df = pd.DataFrame(columns = columns, index= range(len(syn_partners_df)))
    ct1_rec_dict_keys = list(ct1_rec_id_dict.keys())
    ct2_rec_dict_keys = list(ct2_rec_id_dict.keys())
    for i, ct1_id in enumerate(syn_partners_df[f'{ct1_str} id']):
        if ct1_id not in ct1_rec_dict_keys:
            continue
        ct2_id = syn_partners_df.loc[i, f'{ct2_str} id']
        if ct2_id not in ct2_rec_dict_keys:
            continue
        if ct1_id in ct1_ct2_pair_df[f'{ct1_str} id']:
            ct1_ind = np.where(ct1_ct2_pair_df[f'{ct1_str} id'] == ct1_id)[0]
            present_ct2_ids = ct1_ct2_pair_df.loc[ct1_ind, f'{ct2_str} id']
            if ct2_id in present_ct2_ids:
                continue
        ct1_ct2_pair_df.loc[i, f'{ct1_str} id'] = ct1_id
        ct1_ct2_pair_df.loc[i, f'{ct2_str} id'] = ct2_id
        ct1_rec_ids = np.unique(ct1_rec_id_dict[ct1_id])
        ct2_rec_ids = np.unique(ct2_rec_id_dict[ct2_id])
        mask_rec_ids_same = np.in1d(ct1_rec_ids, ct2_rec_ids)
        any_same_ids = np.any(mask_rec_ids_same)
        if any_same_ids == True:
            ct1_ct2_pair_df.loc[i, f'{rec_ct_str} ids different'] = False
        else:
            ct1_ct2_pair_df.loc[i, f'{rec_ct_str} ids different'] = True
        rec_same_ids = ct1_rec_ids[mask_rec_ids_same]
        ct1_perc_diff_ids = 100 - (100 * len(rec_same_ids)/ len(ct1_rec_ids))
        ct2_perc_diff_ids = 100 - (100 * len(rec_same_ids) / len(ct2_rec_ids))
        ct1_ct2_pair_df.loc[i, f'percent of different {rec_ct_str} ids for {ct1_str}'] = ct1_perc_diff_ids
        ct1_ct2_pair_df.loc[i, f'percent of different {rec_ct_str} ids for {ct2_str}'] = ct2_perc_diff_ids

    ct1_ct2_pair_df = ct1_ct2_pair_df.dropna()
    ct1_ct2_pair_df = ct1_ct2_pair_df.reset_index(drop=True)
    ct1_ct2_pair_df.to_csv(f'{f_name}/pairwise_{ct1_str}_{ct2_str}_{rec_ct_str}_same_ids.csv')
    num_pairs = len(ct1_ct2_pair_df)
    num_diff_pairs = len(ct1_ct2_pair_df[ct1_ct2_pair_df[f'{rec_ct_str} ids different']])
    log.info(f'{num_pairs} pairs of {ct1_str} cells and {ct2_str} cells are connected via a {proj_ct_str} synapse. \n'
             f' {num_diff_pairs} of them project only to different {rec_ct_str} ids ({100 * num_diff_pairs/ num_pairs:.2f} percent)')
    some_same_df = ct1_ct2_pair_df[ct1_ct2_pair_df[f'{rec_ct_str} ids different'] == False]
    if len(some_same_df) != 0:
        median_diff_ct1 = np.median(some_same_df[f'percent of different {rec_ct_str} ids for {ct1_str}'])
        median_diff_ct2 = np.median(some_same_df[f'percent of different {rec_ct_str} ids for {ct2_str}'])
        log.info(f'Of those who project to the same {rec_ct_str} cells, a median of {median_diff_ct1:.2f} percent for {ct1_str} are different \n'
                 f' and a median of {median_diff_ct2:.2f} percent for {ct2_str}')
        #plot results
        sns.histplot(x=f'percent of different {rec_ct_str} ids for {ct1_str}', data=ct1_ct2_pair_df, color=ct_palette[ct1_str], common_norm=False,
                     fill=False, element="step", linewidth=3)
        plt.ylabel('count of cell-pairs')
        plt.title(f'percent of different {rec_ct_str} ids for {ct1_str}')
        plt.savefig(f'{f_name}/{ct1_str}_perc_diff_{rec_ct_str}_hist.png')
        plt.close()
        sns.histplot(x=f'percent of different {rec_ct_str} ids for {ct1_str}', data=ct1_ct2_pair_df,
                     color=ct_palette[ct1_str], common_norm=False,
                     fill=False, element="step", linewidth=3, stat = 'percent')
        plt.ylabel('% of cell-pairs')
        plt.title(f'percent of different {rec_ct_str} ids for {ct1_str}')
        plt.savefig(f'{f_name}/{ct1_str}_perc_diff_{rec_ct_str}_hist_perc.png')
        plt.close()
        sns.histplot(x=f'percent of different {rec_ct_str} ids for {ct2_str}', data=ct1_ct2_pair_df,
                     color=ct_palette[ct2_str], common_norm=False,
                     fill=False, element="step", linewidth=3)
        plt.ylabel('count of cell-pairs')
        plt.title(f'percent of different {rec_ct_str} ids for {ct2_str}')
        plt.savefig(f'{f_name}/{ct2_str}_perc_diff_{rec_ct_str}_hist.png')
        plt.close()
        sns.histplot(x=f'percent of different {rec_ct_str} ids for {ct2_str}', data=ct1_ct2_pair_df,
                     color=ct_palette[ct2_str], common_norm=False,
                     fill=False, element="step", linewidth=3, stat='percent')
        plt.ylabel('% of cell-pairs')
        plt.title(f'percent of different {rec_ct_str} ids for {ct2_str}')
        plt.savefig(f'{f_name}/{ct2_str}_perc_diff_{rec_ct_str}_hist_perc.png')
        plt.close()

    #also get information per ct1 and per ct2 cell, not only per pais
    for i, ct1_id in enumerate(ct1_conn_partners_df['cellid']):
        ct2_ids = np.unique(ct1_id_dict[ct1_id])
        ct1_rec_ids_pc = np.unique(ct1_rec_id_dict[ct1_id])
        ct2_rec_ids_pc = []
        for ct2_id in ct2_ids:
            if ct2_id in ct2_rec_dict_keys:
                ct2_rec_ids_pc.append(ct2_rec_id_dict[ct2_id])
        if len(ct2_rec_ids_pc) == 0:
            ct1_conn_partners_df.loc[i, f'number {rec_ct_str} ids {ct2_str}'] = 0
            continue
        try:
            ct2_rec_ids_pc = np.unique(np.concatenate(ct2_rec_ids_pc))
        except ValueError:
            ct2_rec_ids_pc = np.unique(ct2_rec_ids_pc)
        ct1_conn_partners_df.loc[i, f'number {rec_ct_str} ids {ct2_str}'] = len(ct2_rec_ids_pc)
        mask_same_rec_ids = np.in1d(ct1_rec_ids_pc, ct2_rec_ids_pc)
        any_same_ids = np.any(mask_same_rec_ids)
        if any_same_ids == True:
            ct1_conn_partners_df.loc[i, f'{rec_ct_str} ids different'] = False
        else:
            ct1_conn_partners_df.loc[i, f'{rec_ct_str} ids different'] = True
        perc_diff_ids = 100 - (100 * len(ct1_rec_ids_pc[mask_same_rec_ids])/ len(ct1_rec_ids_pc))
        ct1_conn_partners_df.loc[i, f'percent {rec_ct_str} ids different'] = perc_diff_ids
    ct1_conn_partners_df = ct1_conn_partners_df.dropna()
    ct1_conn_partners_df = ct1_conn_partners_df.reset_index(drop = True)
    ct1_conn_partners_df.to_csv(f'{f_name}/{ct1_str}_number_cell_partners_{rec_ct_str}_only_with_parners_projecting2{rec_ct_str}.csv')
    log.info(f'{len(ct1_conn_partners_df)} {ct1_str} cells have {ct2_str} partners that connect to {rec_ct_str} and'
             f' connect to {rec_ct_str} themselves ({100 * len(ct1_conn_partners_df)/ len(suitable_ids_dict[ct1]):.2f}'
             f' percent of all {ct1_str} cells)')
    num_diff_ids = len(ct1_conn_partners_df[ct1_conn_partners_df[f'{rec_ct_str} ids different']])
    perc_diff_ids_ct1 = 100 * num_diff_ids / len(ct1_conn_partners_df)
    log.info(
        f'{perc_diff_ids_ct1:.2f} percent of {ct1_str} cells are paired with {ct2_str} cells which only project to '
        f'different {rec_ct_str} cells')
    median_rec_ids_via_otherct = np.median(ct1_conn_partners_df[f'number {rec_ct_str} ids {ct2_str}'])
    log.info(
        f'A median {ct1_str} cell is paired with {ct2_str} cells that in total connect to {median_rec_ids_via_otherct} '
        f'different {rec_ct_str} cells.')
    some_same_df = ct1_conn_partners_df[ct1_conn_partners_df[f'{rec_ct_str} ids different'] == False]
    if len(some_same_df) != 0:
        median_diff_ct1 = np.median(some_same_df[f'percent {rec_ct_str} ids different'])
        log.info(
            f'Of those who project to the same {rec_ct_str} cells, a median of {median_diff_ct1:.2f} percent for {ct1_str} are different')

    for i, ct2_id in enumerate(ct2_conn_partners_df['cellid']):
        ct1_ids = np.unique(ct2_id_dict[ct2_id])
        ct2_rec_ids_pc = np.unique(ct2_rec_id_dict[ct2_id])
        ct1_rec_ids_pc = []
        for ct1_id in ct1_ids:
            if ct1_id in ct1_rec_dict_keys:
                ct1_rec_ids_pc.append(ct1_rec_id_dict[ct1_id])
        if len(ct1_rec_ids_pc) == 0:
            ct2_conn_partners_df.loc[i, f'number {rec_ct_str} ids {ct1_str}'] = 0
            continue
        try:
            ct1_rec_ids_pc = np.unique(np.concatenate(ct1_rec_ids_pc))
        except ValueError:
            ct1_rec_ids_pc = np.unique(ct1_rec_ids_pc)
        ct2_conn_partners_df.loc[i, f'number {rec_ct_str} ids {ct1_str}'] = len(ct1_rec_ids_pc)
        mask_same_rec_ids = np.in1d(ct2_rec_ids_pc, ct1_rec_ids_pc)
        any_same_ids = np.any(mask_same_rec_ids)
        if any_same_ids == True:
            ct2_conn_partners_df.loc[i, f'{rec_ct_str} ids different'] = False
        else:
            ct2_conn_partners_df.loc[i, f'{rec_ct_str} ids different'] = True
        perc_diff_ids = 100 - (100 * len(ct2_rec_ids_pc[mask_same_rec_ids])/ len(ct2_rec_ids_pc))
        ct2_conn_partners_df.loc[i, f'percent {rec_ct_str} ids different'] = perc_diff_ids
    ct2_conn_partners_df = ct2_conn_partners_df.dropna()
    ct2_conn_partners_df = ct2_conn_partners_df.reset_index(drop=True)
    ct2_conn_partners_df.to_csv(
        f'{f_name}/{ct2_str}_number_cell_partners_{rec_ct_str}_only_with_parners_projecting2{rec_ct_str}.csv')
    log.info(f'{len(ct2_conn_partners_df)} {ct2_str} cells have {ct1_str} partners that connect to {rec_ct_str} and'
             f' connect to {rec_ct_str} themselves ({100 * len(ct2_conn_partners_df) / len(suitable_ids_dict[ct2]):.2f}'
             f' percent of all {ct2_str} cells)')
    num_diff_ids = len(ct2_conn_partners_df[ct2_conn_partners_df[f'{rec_ct_str} ids different']])
    perc_diff_ids_ct2 = 100 * num_diff_ids/ len(ct2_conn_partners_df)
    log.info(f'{perc_diff_ids_ct2:.2f} percent of {ct2_str} cells are paired with {ct1_str} cells which only project to '
             f'different {rec_ct_str} cells')
    median_rec_ids_via_otherct = np.median(ct2_conn_partners_df[f'number {rec_ct_str} ids {ct1_str}'])
    log.info(f'A median {ct2_str} cell is paired with {ct1_str} cells that in total connect to {median_rec_ids_via_otherct} '
             f'different {rec_ct_str} cells.')
    some_same_df = ct2_conn_partners_df[ct2_conn_partners_df[f'{rec_ct_str} ids different'] == False]
    if len(some_same_df) != 0:
        median_diff_ct2 = np.median(some_same_df[f'percent {rec_ct_str} ids different'])
        log.info(f'Of those who project to the same {rec_ct_str} cells, a median of {median_diff_ct2:.2f} percent for {ct2_str} are different')

    log.info(f'Step 6/7: Compare per cell information of {ct1_str}, {ct2_str} and plot results')
    #combine information on number of rec_ct from other celltype and percent of ids different for ct1 and ct2
    num_ct1_cells = len(ct1_conn_partners_df)
    num_cells = num_ct1_cells + len(ct2_conn_partners_df)
    columns = ['cellid', 'celltype', f'percent {rec_ct_str} ids different', f'number {rec_ct_str} ids other ct',
                f'number {rec_ct_str} cells', f'number {rec_ct_str} syns', f'number {proj_ct_str} axons', 'number partner cells other ct']
    ct1_ct2_percell_df = pd.DataFrame(columns =columns, index = range(num_cells))
    for param in ct1_conn_partners_df:
        if param in ct1_ct2_percell_df:
            ct1_ct2_percell_df.loc[0: num_ct1_cells - 1, param] = ct1_conn_partners_df[param]
            ct1_ct2_percell_df.loc[num_ct1_cells: num_cells - 1, param] = ct2_conn_partners_df[param]
    ct1_ct2_percell_df.loc[0: num_ct1_cells - 1, 'celltype'] = ct1_str
    ct1_ct2_percell_df.loc[num_ct1_cells: num_cells - 1, 'celltype'] = ct2_str
    ct1_ct2_percell_df.loc[0: num_ct1_cells - 1, f'number {rec_ct_str} ids other ct'] = ct1_conn_partners_df[
        f'number {rec_ct_str} ids {ct2_str}']
    ct1_ct2_percell_df.loc[num_ct1_cells: num_cells - 1, f'number {rec_ct_str} ids other ct'] = ct2_conn_partners_df[
        f'number {rec_ct_str} ids {ct1_str}']
    ct1_ct2_percell_df.loc[0: num_ct1_cells - 1, 'number partner cells other ct'] = ct1_conn_partners_df[
        f'number {ct2_str} partners']
    ct1_ct2_percell_df.loc[num_ct1_cells: num_cells - 1, 'number partner cells other ct'] = ct2_conn_partners_df[
        f'number {ct1_str} partners']
    ct1_ct2_percell_df.to_csv(f'{f_name}/percell_{ct1_str}_{ct2_str}_comp.csv')
    #get statistics
    ranksums_res_df = pd.DataFrame(columns = [f'{ct1_str} vs {ct2_str} stats', f'{ct1_str} vs {ct2_str} p-value'])
    for param in ct1_ct2_percell_df:
        if 'cellid' in param or 'celltype' in param:
            continue
        param_ct1 = ct1_ct2_percell_df[param][ct1_ct2_percell_df['celltype'] == ct1_str]
        param_ct2 = ct1_ct2_percell_df[param][ct1_ct2_percell_df['celltype'] == ct2_str]
        ranksums_res = ranksums(param_ct1, param_ct2)
        ranksums_res_df.loc[param, f'{ct1_str} vs {ct2_str} stats'] = ranksums_res[0]
        ranksums_res_df.loc[param, f'{ct1_str} vs {ct2_str} p-value'] = ranksums_res[1]
        ct1_ct2_percell_df[param] = ct1_ct2_percell_df[param].astype(float)
        sns.boxplot(data=ct1_ct2_percell_df, x='celltype', y=param, palette=ct_palette)
        plt.title(param)
        plt.savefig(f'{f_name}/{param}_{ct1_str}_{ct2_str}_box.png')
        plt.savefig(f'{f_name}/{param}_{ct1_str}_{ct2_str}_box.svg')
        plt.close()
        sns.stripplot(x='celltype', y=param, data=ct1_ct2_percell_df, color='black', alpha=0.2,
                      dodge=True, size=2)
        sns.violinplot(x='celltype', y=param, data=ct1_ct2_percell_df, inner="box",
                       palette=ct_palette)
        plt.title(param)
        plt.savefig(f'{f_name}/{param}_{ct1_str}_{ct2_str}_violin.png')
        plt.savefig(f'{f_name}/{param}_{ct1_str}_{ct2_str}_violin.svg')
        plt.close()
    ranksums_res_df.to_csv(f'{f_name}/ranksum_results_{ct1_str}_{ct2_str}.csv')

    log.info('Analysis finished')
