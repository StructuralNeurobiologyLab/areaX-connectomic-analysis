import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os as os
import scipy
from collections import defaultdict
import time
from syconn.handler.config import initialize_logging
from syconn.handler.basics import load_pkl2obj
from tqdm import tqdm
from syconn.handler.basics import write_obj2pkl
from scipy.stats import ranksums
from syconn.proc.meshes import mesh_area_calc, compartmentalize_mesh_fromskel
from syconn.reps.super_segmentation import SuperSegmentationObject

def filter_synapse_caches_for_ct(sd_synssv, pre_cts, post_cts = None, syn_prob_thresh = 0.8, min_syn_size = 0.1, axo_den_so = True):
    """
    prefilter synapse caches accordi
    :param sd_synssv: segmentation dataset
    :param pre_cts: celltypes that can be on presynaptic side
    :param post_cts: celltypes that can be on postsynaptic side, if None, pre_cts can also be post
    :param syn_prob_thresh: threshold for synapse proabbility
    :param min_syn_size: minimal synapse size
    :param axo_den_so: if true only axo-dendritic oraxo-somatic synapses allowed
    :return: cached array with different parameters for celltype, axoness, ssv_partners, synapse sizes, spiness
    """
    syn_prob = sd_synssv.load_numpy_data("syn_prob")
    m = syn_prob > syn_prob_thresh
    m_ids = sd_synssv.ids[m]
    m_axs = sd_synssv.load_numpy_data("partner_axoness")[m]
    m_axs[m_axs == 3] = 1
    m_axs[m_axs == 4] = 1
    m_cts = sd_synssv.load_numpy_data("partner_celltypes")[m]
    m_ssv_partners = sd_synssv.load_numpy_data("neuron_partners")[m]
    m_sizes = sd_synssv.load_numpy_data("mesh_area")[m] / 2
    m_spiness = sd_synssv.load_numpy_data("partner_spiness")[m]
    # select only those of given_celltypes
    # if post and pre not specified both celltypes can be on both sides
    if post_cts is None:
        for ct in pre_cts:
            ct_inds = np.any(m_cts == ct, axis=1)
            m_cts = m_cts[ct_inds]
            m_ids = m_ids[ct_inds]
            m_axs = m_axs[ct_inds]
            m_ssv_partners = m_ssv_partners[ct_inds]
            m_sizes = m_sizes[ct_inds]
            m_spiness = m_spiness[ct_inds]
    else:
        #make sure to exclude pre and postsynaptic cells from wrong celltypes
        #exclude synapses without precelltypes
        ct_inds = np.any(np.in1d(m_cts, pre_cts).reshape(len(m_cts), 2), axis=1)
        m_cts = m_cts[ct_inds]
        m_ids = m_ids[ct_inds]
        m_axs = m_axs[ct_inds]
        m_ssv_partners = m_ssv_partners[ct_inds]
        m_sizes = m_sizes[ct_inds]
        m_spiness = m_spiness[ct_inds]
        #filter those where prects are not where axon is, only if axo_den_so
        if axo_den_so ==  True:
            testct = np.in1d(m_cts, pre_cts).reshape(len(m_cts), 2)
            testax = np.in1d(m_axs, 1).reshape(len(m_cts), 2)
            pre_ct_inds = np.any(testct == testax, axis = 1)
            m_cts = m_cts[pre_ct_inds]
            m_ids = m_ids[pre_ct_inds]
            m_axs = m_axs[pre_ct_inds]
            m_ssv_partners = m_ssv_partners[pre_ct_inds]
            m_sizes = m_sizes[pre_ct_inds]
            m_spiness = m_spiness[pre_ct_inds]
        # exclude synapses without postcelltypes
        ct_inds = np.any(np.in1d(m_cts, post_cts).reshape(len(m_cts), 2), axis=1)
        m_cts = m_cts[ct_inds]
        m_ids = m_ids[ct_inds]
        m_axs = m_axs[ct_inds]
        m_ssv_partners = m_ssv_partners[ct_inds]
        m_sizes = m_sizes[ct_inds]
        m_spiness = m_spiness[ct_inds]
        #filter those where postcts are where axon is, only if axo_den_so
        if axo_den_so ==  True:
            testct = np.in1d(m_cts, post_cts).reshape(len(m_cts), 2)
            testax = np.in1d(m_axs, [2,0]).reshape(len(m_cts), 2)
            post_ct_inds = np.any(testct == testax, axis=1)
            m_cts = m_cts[post_ct_inds]
            m_ids = m_ids[post_ct_inds]
            m_axs = m_axs[post_ct_inds]
            m_ssv_partners = m_ssv_partners[post_ct_inds]
            m_sizes = m_sizes[post_ct_inds]
            m_spiness = m_spiness[post_ct_inds]
    # filter those with size below min_syn_size
    size_inds = m_sizes > min_syn_size
    m_cts = m_cts[size_inds]
    m_ids = m_ids[size_inds]
    m_axs = m_axs[size_inds]
    m_ssv_partners = m_ssv_partners[size_inds]
    m_sizes = m_sizes[size_inds]
    m_spiness = m_spiness[size_inds]
    # only axo-dendritic or axo-somatic synapses allowed
    if axo_den_so:
        axs_inds = np.any(m_axs == 1, axis=1)
        m_cts = m_cts[axs_inds]
        m_ids = m_ids[axs_inds]
        m_axs = m_axs[axs_inds]
        m_ssv_partners = m_ssv_partners[axs_inds]
        m_sizes = m_sizes[axs_inds]
        m_spiness = m_spiness[axs_inds]
        den_so = np.array([0, 2])
        den_so_inds = np.any(np.in1d(m_axs, den_so).reshape(len(m_axs), 2), axis=1)
        m_cts = m_cts[den_so_inds]
        m_ids = m_ids[den_so_inds]
        m_axs = m_axs[den_so_inds]
        m_ssv_partners = m_ssv_partners[den_so_inds]
        m_sizes = m_sizes[den_so_inds]
        m_spiness = m_spiness[den_so_inds]
    return m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness

def estimate_input_ratio_ct(celltype1, cellids1, celltypes, , cellids, label_ct1 = None, label_cts = None, min_comp_len = 100, min_syn_size = 0.1, syn_prob_thresh = 0.8, only_input = True):
    """
    function to determine synaptic input form one celltype to other celltypes. Fractin of synapse amount and sum of synapse size in relation to overall synaptic input from cells with same
    minimum comparment length will be analysed.
    :param celltype1: celltype others are connected to
    :param cellids1: cellids for celltype1
    :param celltypes: list of celltypes connection to celltypes1
    :param cellids: list of cellids from other celltypes to be included
    :param label_ct1: label for celltype1 if deviating from ct_dict, e.g. if subpopulation
    :param label_cts: label for other celltypes if deviating from ct_dict
    :param min_comp_len: minimum compartment length for cells to be included in analysis
    :param min_syn_size: minimum synpase size for synapses to be included
    :param syn_prob_thresh: threshold for synapse probability
    :param only_input: if True then only connections to other celltype from celltype one will be analysed and not vice vera
    :return: dictionary
    """