#check if synapses of two celltypes that are closely together on one axon project to the same or different cells
import scipy.spatial

if __name__ == '__main__':
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct, get_ct_syn_number_sumsize, filter_synapse_caches_general
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_colors import SubCT_Colors
    from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_params import Analysis_Params
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    import os as os
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    import numpy as np
    from tqdm import tqdm
    from scipy.spatial import KDTree
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import defaultdict

    #global_params.wd = "/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811"
    global_params.wd = '/cajal/nvmescratch/projects/data/songbird/j0251/j0251_72_seg_20210127_agglo2_syn_20220811_celltypes_20230822'
    sd_synssv = SegmentationDataset('syn_ssv', working_dir=global_params.config.working_dir)

    version = 'v6'
    bio_params = Analysis_Params(working_dir=global_params.wd, version=version)
    ct_dict = bio_params.ct_dict()
    axon_cts = bio_params.axon_cts()
    min_comp_len = 200
    min_comp_len_ax = 50
    syn_prob = 0.6
    min_syn_size = 0.1
    exclude_known_mergers = True
    #{0:'DA', 1:'LMAN', 2: 'HVC', 3:'MSN', 4:'STN', 5:'TAN', 6:'GPe', 7:'GPi', 8: 'LTS',
    #                      9:'INT1', 10:'INT2', 11:'INT3', 12:'ASTRO', 13:'OLIGO', 14:'MICRO', 15:'MIGR', 16:'FRAG'}
    #proj ct is celltype that makes synapses to ct1 and ct2, potentially in close proximity to each other
    #rec_ct is celltype that ct1 and ct2 project to and whose individual cells will be compared
    proj_ct = 2
    ct1 = 4
    ct2 = 3
    rec_ct = 7
    syn_dist_thresh = 2  # synapse distance threshold in µm
    comp_cts = [proj_ct, ct1, ct2, rec_ct]
    proj_ct_str = ct_dict[proj_ct]
    ct1_str = ct_dict[ct1]
    ct2_str = ct_dict[ct2]
    rec_ct_str = ct_dict[rec_ct]
    scaling = np.array([10, 10, 25]) #nm voxelsize

    if np.any(np.in1d(comp_cts, axon_cts)):
        axon_ct_present = True
        f_name = f'cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/' \
                 f'240201_j0251{version}_close_syn_cellid_comp_{proj_ct_str}_{ct1_str}_{ct2_str}_{rec_ct_str}_{min_comp_len}_{min_comp_len_ax}'
    else:
        axon_ct_present = False
        f_name = f'cajal/scratch/users/arother/bio_analysis_results/dir_indir_pathway_analysis/' \
                 f'240201_j0251{version}_close_syn_cellid_comp_{proj_ct_str}_{ct1_str}_{ct2_str}_{rec_ct_str}_{min_comp_len}'
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging(f'Close synapse cells comparison {proj_ct_str}, {ct1_str}, {ct2_str}, {rec_ct_str}', log_dir=f_name + '/logs/')
    if axon_ct_present:
        log.info(f'min comp len cell = {min_comp_len}, min comp len ax = {min_comp_len_ax}, exclude known mergers = {exclude_known_mergers}, '
                 f'syn prob threshold = {syn_prob}, min synapse size = {min_syn_size}, threshold for distance of synapses from same axon: {syn_dist_thresh}')
    else:
        log.info(
            f'min comp len cell = {min_comp_len}, exclude known mergers = {exclude_known_mergers}, '
            f'syn prob threshold = {syn_prob}, min synapse size = {min_syn_size}, threshold for distance of synapses from same axon: {syn_dist_thresh}')

    log.info("Step 1/5: Load celltypes and check suitability")
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

    log.info('Step 2/X: Prefilter synapses for syn_prob, min_syn_size and suitable cellids')
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

    log.info(f'Step 3/X: Get synapses from {proj_ct_str} to {ct1_str} and {ct2_str}')
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
    log.info(f'The synaptic sum of {proj_ct_str} to {ct1_str} of ids 2 both celltypes is {np.sum(proj2ct1_sizes):.2f}')
    log.info(f'The synaptic sum of {proj_ct_str} to {ct2_str} of ids 2 both celltypes is {np.sum(proj2ct2_sizes):.2f}')

    log.info(f'Step 3/X: Get synapses of {ct1_str} and {ct2_str} that are with a distance of {syn_dist_thresh} µm on the same axon of {proj_ct_str}')
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
    syn_partners_df = pd.Dataframe(columns = columns, index = range(len(proj2ct1_ssv_partners)))
    proj_id_dict = defaultdict(lambda: [])
    ct1_id_dict = defaultdict(lambda: [])
    ct2_id_dict = defaultdict(lambda: [])
    ax_ind_ct1 = np.where(proj2ct1_axs == 1)
    ax_ind_ct2 = np.where(proj2ct2_axs == 1)
    ct1_ind = np.where(proj2ct1_axs != 1)
    ct2_ind = np.where(proj2ct2_axs != 1)
    for i, ct1_syn_id in enumerate(proj2ct1_syn_ids):
        #check first if all pais have same proj_ax
        if len(inds_tree_ct2[i]) == 0:
            continue
        proj_id_ct1 = proj2ct1_ssv_partners[ax_ind_ct1[i]]
        ct1_id = proj2ct1_ssv_partners[ct1_ind[i]]
        ct1_coords = proj2ct1_rep_coord[i]
        for ct2_syn_ind in inds_tree_ct2[i]:
            proj_id_ct2 = proj2ct2_ssv_partners[ax_ind_ct2[ct2_syn_ind]]
            ct2_id = proj2ct2_ssv_partners[ct2_ind[ct2_syn_ind]]
            if proj_id_ct2 != proj_id_ct1:
                continue
            proj_id_dict[proj_id_ct1].append([ct1_id, ct2_id])
            ct1_id_dict[ct1_id].append(ct2_id)
            ct2_id_dict[ct2_id].append(ct1_id)
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

    #check that synapses that are close together are from the same proj axon
    #save number of synapses that are close to each other and number of cells that
    #are part of this
    #make dictionary with proj_axon_id as key and ct1 and ct2 id that are close togehter
    #another dictionary with ct1_id and ct2_id that is from same axon




    #Step 3: Check which synapses from ct1 are close to ct2,
    #save information on cellids that are close to each other

    #Step 4: Get synapses from ct1, ct2 to rec_ct and check which cellids each of them synapses to
    #save information on how many of the cellids that have receiving synapses close to each other
    #project to this celltype

    #Step 5: check if cellids with same input project to same rec_ct cells

    #Step 6: calculate statisitics and plot results
