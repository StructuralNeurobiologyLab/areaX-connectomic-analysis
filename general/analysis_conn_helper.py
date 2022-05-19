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
    :param sd_csssv: segmentation dataset for contact sites
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

def synapse_amount_sumsize_between2cts(celltype1, cellids1, cellids2, syn_ids, syn_cts, syn_ssv_partners, syn_sizes, syn_axs, seperate_soma_dens = False, fragments_pre = False):
    '''
        gives amount and summed synapse size for each cell from other celltpye and writes it in dictionary. Calculates synapses from celltype1 to celltype2.
        Function assumes synapses are already prefiltered with filter_synapse_caches for different thresholds and at least celltype involvement.
        (dendrite, soma) seperately.
        :param celltype1: celltype with outgoing synapses
        :param cellids1: cellids with outgoing synapses
        :param cellids2: cellids with incomng synapses
        :param m_cts: synapse ids
        :param syn_cts: celltypes of synaptic partners
        :param syn_ssv_partners: cellids of synaptic partners
        :param syn_sizes: synapse sizes
        :param syn_axs: axoness values of synaptic partners
        :param seperate_soma_dens: if True: seperate dictionaries for dendritic and somatic inputs
        :param if fragments_pre = True: all cellfragments will be included not only fullcells or cellids for presynapse
        :return: dictionary with cell_ids as keys and amount of synapses
        '''
    # get synapses where input celltype is axon
    if fragments_pre:
        testct = np.in1d(syn_ssv_partners, cellids1).reshape(len(syn_ssv_partners), 2)
        testax = np.in1d(syn_axs, 1).reshape(len(syn_ssv_partners), 2)
        pre_ct_inds = np.all(testct == testax, axis=1)
        syn_cts = syn_cts[pre_ct_inds]
        syn_ids = syn_ids[pre_ct_inds]
        syn_axs = syn_axs[pre_ct_inds]
        syn_ssv_partners = syn_ssv_partners[pre_ct_inds]
        syn_sizes = syn_sizes[pre_ct_inds]
    #get synapses where outgoing celltype gives dendrite, soma
    testct = np.in1d(syn_ssv_partners, cellids2).reshape(len(syn_ssv_partners), 2)
    testax = np.in1d(syn_axs, [2, 0]).reshape(len(syn_ssv_partners), 2)
    post_ct_inds = np.all(testct == testax, axis=1)
    m_cts = syn_cts[post_ct_inds]
    m_ids = syn_ids[post_ct_inds]
    m_axs = syn_axs[post_ct_inds]
    m_ssv_partners = syn_ssv_partners[post_ct_inds]
    m_sizes = syn_sizes[post_ct_inds]
    if seperate_soma_dens is True:
        #get unique cellids for cells recieving synapses, divide in dendritic and somatic synapses
        den_inds = np.where(m_axs == 0)
        som_inds = np.where(m_axs == 2)
        den_ct_inds = np.where(m_cts[den_inds] != celltype1)
        som_ct_inds = np.where(m_cts[som_inds] != celltype1)
        den_ssv_partners = m_ssv_partners[den_inds[0]]
        den_ssvs = den_ssv_partners[den_ct_inds, den_inds[1][den_ct_inds]][0]
        den_sizes = m_sizes[den_inds[0]][den_ct_inds]
        den_ssv_inds, unique_den_ssvs = pd.factorize(den_ssvs)
        den_syn_sizes = np.bincount(den_ssv_inds, den_sizes)
        den_amounts = np.bincount(den_ssv_inds)
        # get unique cellids from cells whose soma receive synapses, count them and sum up sizes
        som_ssv_partners = m_ssv_partners[som_inds[0]]
        som_ssvs = som_ssv_partners[som_ct_inds, som_inds[1][som_ct_inds]][0]
        som_sizes = m_sizes[som_inds[0]][som_ct_inds]
        som_ssv_inds, unique_som_ssvs = pd.factorize(som_ssvs)
        som_syn_sizes = np.bincount(som_ssv_inds, som_sizes)
        som_amounts = np.bincount(som_ssv_inds)
        # create dictionaries for soma, dendrite synapses
        den_dict = {cellid: {"amount": den_amounts[i], "summed size": den_syn_sizes[i]} for i, cellid in
                    enumerate(unique_den_ssvs)}
        soma_dict = {cellid: {"amount": som_amounts[i], "summed size": som_syn_sizes[i]} for i, cellid in
                     enumerate(unique_som_ssvs)}
        return den_dict, soma_dict
    else:
        # get unique cellids from cells who receive synapses, count them and sum up sizes
        rec_inds = np.where(m_axs != 1)
        rec_ct_inds = np.where(m_cts[rec_inds] != celltype1)
        receiv_ssvs = m_ssv_partners[rec_ct_inds, rec_inds[1][rec_ct_inds]][0]
        rec_sizes = m_sizes[rec_ct_inds]
        rec_ssv_inds, unique_rec_ssvs = pd.factorize(receiv_ssvs)
        rec_syn_sizes = np.bincount(rec_ssv_inds, rec_sizes)
        rec_amounts = np.bincount(rec_ssv_inds)
        rec_dict = {cellid: {"amount": rec_amounts[i], "summed size": rec_syn_sizes[i]} for i, cellid in
                    enumerate(unique_rec_ssvs)}
        return rec_dict


def filter_contact_caches_for_cellids(sd_cs_ssv, cellids1, cellids2):
    """
    filter contact sites to find contact sites between cells from cellids1 and cellids2.
    :param sd_cs_ssv: segmentation Dataset contact sites
    :param cellids1: cellids that should be part of one contact site
    :param cellids2: cellids that should be part of the other contact site
    :return:
    """
    raise ValueError
    cs_partners = sd_cs_ssv.cs_partners
    cs_ids = sd_cs_ssv.ids
    ct1_inds = np.any(np.in1d(cs_partners, cellids1).reshape(len(cs_partners), 2), axis=1)
    cs_partners = cs_partners[ct1_inds]
    cs_ids = cs_ids[ct1_inds]
    ct2_inds = np.any(np.in1d(cs_partners, cellids2).reshape(len(cs_partners), 2), axis=1)
    cs_partners = cs_partners[ct2_inds]
    cs_ids = cs_ids[ct2_inds]

    return cs_partners, cs_ids


