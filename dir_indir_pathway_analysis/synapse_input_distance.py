#functions related to getting the distance to soma of synaptic inputs

import numpy as np
from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct, get_syn_input_distance_percell
from syconn.reps.super_segmentation import SuperSegmentationObject
import pandas as pd
from multiprocessing import pool

def get_syn_distances(ct_post, cellids_post, sd_synssv, syn_prob = 0.8, min_syn_size = 0.1, ct_pre = None, cellids_pre = None):
    '''
    Calculates distance of synapses to soma for two celltypes. Uses distance2soma
    :param ct_post: Celltype that receives synapses. distance2soma computed for this one.
    :param cellids_post: Filtered cellids from ct_post
    :param sd_synssv: synapse dataset
    :param syn_prob: synapse probabilitly threshold
    :param min_syn_size: minimum synapse size
    :param ct_pre: Presynaptic celltype, if none then same as ct_post
    :param cellids_pre: Filtered cellids from ct_pre, if none then cellids_post
    :return: array with cellids, median distances to soma per cell
    '''

    #filter synapses between celltypes
    if ct_pre is None:
        ct_pre = ct_post
        m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(sd_synssv,
                                                                                                        pre_cts=[ct_pre],
                                                                                                        post_cts=None,
                                                                                                        syn_prob_thresh=syn_prob,
                                                                                                        min_syn_size=min_syn_size,
                                                                                                        axo_den_so=True)
        suit_ct_inds = np.all(np.in1d(m_ssv_partners, cellids_post).reshape(len(m_ssv_partners), 2), axis=1)
        m_ssv_partners = m_ssv_partners[suit_ct_inds]
        m_sizes = m_sizes[suit_ct_inds]
        m_axs = m_axs[suit_ct_inds]
        m_rep_coord = m_rep_coord[suit_ct_inds]
    else:
        m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(sd_synssv,
                                                                                                            pre_cts=[ct_pre],
                                                                                                            post_cts=[ct_post],
                                                                                                            syn_prob_thresh=syn_prob,
                                                                                                            min_syn_size=min_syn_size,
                                                                                                            axo_den_so=True)
        suit_ct_inds = np.any(np.in1d(m_ssv_partners, cellids_post).reshape(len(m_ssv_partners), 2), axis=1)
        m_ssv_partners = m_ssv_partners[suit_ct_inds]
        m_sizes = m_sizes[suit_ct_inds]
        m_axs = m_axs[suit_ct_inds]
        m_rep_coord = m_rep_coord[suit_ct_inds]
        suit_ct_inds = np.any(np.in1d(m_ssv_partners, cellids_pre).reshape(len(m_ssv_partners), 2), axis=1)
        m_ssv_partners = m_ssv_partners[suit_ct_inds]
        m_sizes = m_sizes[suit_ct_inds]
        m_axs = m_axs[suit_ct_inds]
        m_rep_coord = m_rep_coord[suit_ct_inds]
    # get synaptic coordinates grouped per cellid_post
    sort_inds = np.where(m_axs != 1)
    post_ssvs = m_ssv_partners[sort_inds]
    ssv_inds, unique_post_ssvs = pd.factorize(post_ssvs)
    syn_ssv_sizes = np.bincount(ssv_inds, m_sizes)
    syn_numbers = np.bincount(ssv_inds)
    post_coords_pd = pd.DataFrame(m_rep_coord)
    per_postid_coords_grouped = post_coords_pd.groupby(by=ssv_inds)
    postid_coord_groups = per_postid_coords_grouped.groups
    #get distance of synapses per post id
    id_group_pair = [[unique_post_ssvs[i], m_rep_coord[postid_coord_groups[i]]] for i in range(len(unique_post_ssvs))]
    p = pool.Pool()
    outputs = p.map(get_syn_input_distance_percell, id_group_pair)
    outputs = np.array(outputs, dtype = object)
    post_ids = outputs[:, 0]
    median_distances_per_ids = outputs[:, 1]
    min_distances_per_ids = outputs[:, 2]
    max_distances_per_ids = outputs[:, 3]
    return post_ids, median_distances_per_ids, min_distances_per_ids, max_distances_per_ids, syn_numbers, syn_ssv_sizes





