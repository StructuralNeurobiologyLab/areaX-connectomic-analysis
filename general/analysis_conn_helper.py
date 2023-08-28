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

def filter_synapse_caches_general(sd_synssv, syn_prob_thresh = 0.8, min_syn_size = 0.1):
    '''
    loads and filters synapse caches for min_syn_size and min_syn_size
    :param sd_synssv: segmentation dataset
    :param syn_prob_thresh: threshold for synapse probability
    :param min_syn_size: threshold for synapse size
    :param wd: path to working directory to load caches from .npy
    :return: m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord
    '''

    syn_prob = sd_synssv.load_numpy_data("syn_prob")
    m_ids = sd_synssv.ids
    m_axs = sd_synssv.load_numpy_data("partner_axoness")
    m_cts = sd_synssv.load_numpy_data("partner_celltypes")
    m_ssv_partners = sd_synssv.load_numpy_data("neuron_partners")
    m_sizes = sd_synssv.load_numpy_data("mesh_area") / 2
    m_spiness = sd_synssv.load_numpy_data("partner_spiness")
    m_rep_coord = sd_synssv.load_numpy_data("rep_coord")
    m = syn_prob > syn_prob_thresh
    m_ids = m_ids[m]
    m_axs = m_axs[m]
    m_cts = m_cts[m]
    m_ssv_partners = m_ssv_partners[m]
    syn_prob = syn_prob[m]
    m_sizes = m_sizes[m]
    m_spiness = m_spiness[m]
    m_rep_coord = m_rep_coord[m]
    size_inds = m_sizes > min_syn_size
    m_cts = m_cts[size_inds]
    m_ids = m_ids[size_inds]
    m_axs = m_axs[size_inds]
    m_ssv_partners = m_ssv_partners[size_inds]
    m_sizes = m_sizes[size_inds]
    m_spiness = m_spiness[size_inds]
    m_rep_coord = m_rep_coord[size_inds]
    syn_prob = syn_prob[size_inds]
    return m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob


def filter_synapse_caches_for_ct(pre_cts, post_cts = None, syn_prob_thresh = 0.8, min_syn_size = 0.1, axo_den_so = True, sd_synssv = None, synapses_caches = None, return_syn_prob = None):
    """
    prefilter synapse caches according to celltype and different parameters
    :param sd_synssv: segmentation dataset
    :param pre_cts: celltypes that can be on presynaptic side
    :param post_cts: celltypes that can be on postsynaptic side, if None, pre_cts can also be post
    :param sd_csssv: segmentation dataset for contact sites
    :param syn_prob_thresh: threshold for synapse probability, if None given will filter synapse probability like other
    parameters and return it
    :param min_syn_size: minimal synapse size
    :param axo_den_so: if true only axo-dendritic oraxo-somatic synapses allowed
    :param synapses_caches: if given, takes synapse parameters from cache and filers them
    :param return_syn_prob: if given, filter and return syn_prob
    :return: cached array with different parameters for celltype, axoness, ssv_partners, synapse sizes, spiness
    """
    if sd_synssv is None and synapses_caches is None:
        raise ValueError('Function needs either a Segmentation Dataset or Cached Synapse Parameters')
    if sd_synssv is not None:
        syn_prob = sd_synssv.load_numpy_data("syn_prob")
        #m_ids = sd_synssv.ids
        m_axs = sd_synssv.load_numpy_data("partner_axoness")
        m_cts = sd_synssv.load_numpy_data("partner_celltypes")
        m_ssv_partners = sd_synssv.load_numpy_data("neuron_partners")
        m_sizes = sd_synssv.load_numpy_data("mesh_area") / 2
        #m_spiness = sd_synssv.load_numpy_data("partner_spiness")
        m_rep_coord = sd_synssv.load_numpy_data("rep_coord")
    else:
        m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob = synapses_caches
    m_axs[m_axs == 3] = 1
    m_axs[m_axs == 4] = 1
    #filter for synanpse probabilty threshold
    if syn_prob_thresh is not None:
        m = syn_prob > syn_prob_thresh
        #m_ids = m_ids[m]
        m_axs = m_axs[m]
        m_cts = m_cts[m]
        m_ssv_partners = m_ssv_partners[m]
        m_sizes = m_sizes[m]
        #m_spiness = m_spiness[m]
        m_rep_coord = m_rep_coord[m]
    # select only those of given_celltypes
    # if post and pre not specified both celltypes can be on both sides
    if post_cts is None:
        for ct in pre_cts:
            ct_inds = np.any(m_cts == ct, axis=1)
            m_cts = m_cts[ct_inds]
           # m_ids = m_ids[ct_inds]
            m_axs = m_axs[ct_inds]
            m_ssv_partners = m_ssv_partners[ct_inds]
            m_sizes = m_sizes[ct_inds]
            #m_spiness = m_spiness[ct_inds]
            m_rep_coord = m_rep_coord[ct_inds]
            if return_syn_prob:
                syn_prob = syn_prob[ct_inds]
    else:
        #make sure to exclude pre and postsynaptic cells from wrong celltypes
        #exclude synapses without precelltypes
        ct_inds = np.any(np.in1d(m_cts, pre_cts).reshape(len(m_cts), 2), axis=1)
        m_cts = m_cts[ct_inds]
        #m_ids = m_ids[ct_inds]
        m_axs = m_axs[ct_inds]
        m_ssv_partners = m_ssv_partners[ct_inds]
        m_sizes = m_sizes[ct_inds]
        #m_spiness = m_spiness[ct_inds]
        m_rep_coord = m_rep_coord[ct_inds]
        if return_syn_prob:
            syn_prob = syn_prob[ct_inds]
        #filter those where prects are not where axon is, only if axo_den_so
        if axo_den_so ==  True:
            testct = np.in1d(m_cts, pre_cts).reshape(len(m_cts), 2)
            testax = np.in1d(m_axs, 1).reshape(len(m_cts), 2)
            pre_ct_inds = np.any(testct == testax, axis = 1)
            m_cts = m_cts[pre_ct_inds]
            #m_ids = m_ids[pre_ct_inds]
            m_axs = m_axs[pre_ct_inds]
            m_ssv_partners = m_ssv_partners[pre_ct_inds]
            m_sizes = m_sizes[pre_ct_inds]
            #m_spiness = m_spiness[pre_ct_inds]
            m_rep_coord = m_rep_coord[pre_ct_inds]
            if return_syn_prob:
                syn_prob = syn_prob[pre_ct_inds]
        # exclude synapses without postcelltypes
        ct_inds = np.any(np.in1d(m_cts, post_cts).reshape(len(m_cts), 2), axis=1)
        m_cts = m_cts[ct_inds]
        #m_ids = m_ids[ct_inds]
        m_axs = m_axs[ct_inds]
        m_ssv_partners = m_ssv_partners[ct_inds]
        m_sizes = m_sizes[ct_inds]
        #m_spiness = m_spiness[ct_inds]
        m_rep_coord = m_rep_coord[ct_inds]
        if return_syn_prob:
            syn_prob = syn_prob[ct_inds]
        #filter those where postcts are where axon is not, only if axo_den_so
        if axo_den_so ==  True:
            testct = np.in1d(m_cts, post_cts).reshape(len(m_cts), 2)
            testax = np.in1d(m_axs, [2,0]).reshape(len(m_cts), 2)
            post_ct_inds = np.any(testct == testax, axis=1)
            m_cts = m_cts[post_ct_inds]
            #m_ids = m_ids[post_ct_inds]
            m_axs = m_axs[post_ct_inds]
            m_ssv_partners = m_ssv_partners[post_ct_inds]
            m_sizes = m_sizes[post_ct_inds]
            #m_spiness = m_spiness[post_ct_inds]
            m_rep_coord = m_rep_coord[post_ct_inds]
            if return_syn_prob:
                syn_prob = syn_prob[post_ct_inds]
    # filter those with size below min_syn_size
    if min_syn_size is not None:
        size_inds = m_sizes > min_syn_size
        m_cts = m_cts[size_inds]
        #m_ids = m_ids[size_inds]
        m_axs = m_axs[size_inds]
        m_ssv_partners = m_ssv_partners[size_inds]
        m_sizes = m_sizes[size_inds]
        #m_spiness = m_spiness[size_inds]
        m_rep_coord = m_rep_coord[size_inds]
    if return_syn_prob:
        syn_prob = syn_prob[size_inds]
    # only axo-dendritic or axo-somatic synapses allowed
    if axo_den_so:
        axs_inds = np.any(m_axs == 1, axis=1)
        m_cts = m_cts[axs_inds]
        #m_ids = m_ids[axs_inds]
        m_axs = m_axs[axs_inds]
        m_ssv_partners = m_ssv_partners[axs_inds]
        m_sizes = m_sizes[axs_inds]
        #m_spiness = m_spiness[axs_inds]
        m_rep_coord = m_rep_coord[axs_inds]
        if return_syn_prob:
            syn_prob = syn_prob[axs_inds]
        den_so = np.array([0, 2])
        den_so_inds = np.any(np.in1d(m_axs, den_so).reshape(len(m_axs), 2), axis=1)
        m_cts = m_cts[den_so_inds]
        #m_ids = m_ids[den_so_inds]
        m_axs = m_axs[den_so_inds]
        m_ssv_partners = m_ssv_partners[den_so_inds]
        m_sizes = m_sizes[den_so_inds]
        #m_spiness = m_spiness[den_so_inds]
        m_rep_coord = m_rep_coord[den_so_inds]
        if return_syn_prob:
            syn_prob = syn_prob[den_so_inds]
    if return_syn_prob:
        return m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob
    else:
        #return m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord
        return m_cts, m_axs, m_ssv_partners, m_sizes, m_rep_coord

def filter_contact_sites_axoness(cs_ssv_mesh_areas, cs_ssv_celltypes, cs_ssv_neuron_partners, cs_ssv_axoness, cs_ssv_coords, pre_cts, post_cts = None, axo_den_so = True, min_size = 0.1):
    '''
    Funktion that filters cs_ssv numpy data similar to filter_synapse_caches. Needs information about cs_ssv celltypes and cs_ssv_axoness in array form.
    Will make sure that all cs_ssv have a minimum size, and that they contain only celltypes that are desired.
    :param cs_ssv_mesh_areas: mesh areas / size of contact area for cs_ssv.
    :param cs_ssv_celltypes: celltypes for both partners, 2D array
    :param cs_ssv_neuron_partners: ssv_ids for both partners, 2D array
    :param cs_ssv_axoness: axoness for both partners, 2D arrays, 0 = dendrite, 1 = axon, 2 = soma
    :param cs_ssv_coords: coordinates of cs_ssv
    :param pre_cts: celltypes for axons
    :param post_cts: celltypes for dendrite, soma, if None, pre_cts can also be post
    :param axo_den_so: if True only connections between axon and dendrite, or axon and soma analysed (similar to comparable synapses)
    :param min_size: minimum mesh area
    :return: filtered numpy arrays for mesh_area, celltype, neuron_partner, axoness
    '''
    #make sure only suitable sizes are filtered
    if min_size > 0:
        size_inds = cs_ssv_mesh_areas > min_size
        cs_ssv_mesh_areas = cs_ssv_mesh_areas[size_inds]
        cs_ssv_celltypes = cs_ssv_celltypes[size_inds]
        cs_ssv_neuron_partners = cs_ssv_neuron_partners[size_inds]
        cs_ssv_axoness = cs_ssv_axoness[size_inds]
        cs_ssv_coords = cs_ssv_coords[size_inds]
    #axoness should only idicate axon, soma, dendrite
    if 4 in np.unique(cs_ssv_axoness) or 5 in np.unique(cs_ssv_axoness):
        cs_ssv_axoness[cs_ssv_axoness == 4] = 1
        cs_ssv_axoness[cs_ssv_axoness == 5] = 1
    if post_cts is None:
        for ct in pre_cts:
            ct_inds = np.any(cs_ssv_celltypes == ct, axis=1)
            cs_ssv_celltypes = cs_ssv_celltypes[ct_inds]
            cs_ssv_mesh_areas = cs_ssv_mesh_areas[ct_inds]
            cs_ssv_axoness = cs_ssv_axoness[ct_inds]
            cs_ssv_neuron_partners = cs_ssv_neuron_partners[ct_inds]
            cs_ssv_coords = cs_ssv_coords[ct_inds]
    else:
        # make sure to exclude pre and postsynaptic cells from wrong celltypes
        # exclude cs_ssv without pre-celltypes
        ct_inds = np.any(np.in1d(cs_ssv_celltypes, pre_cts).reshape(len(cs_ssv_celltypes), 2), axis=1)
        cs_ssv_celltypes = cs_ssv_celltypes[ct_inds]
        cs_ssv_mesh_areas = cs_ssv_mesh_areas[ct_inds]
        cs_ssv_axoness = cs_ssv_axoness[ct_inds]
        cs_ssv_neuron_partners = cs_ssv_neuron_partners[ct_inds]
        cs_ssv_coords = cs_ssv_coords[ct_inds]
        # filter those where prects are not where axon is, only if axo_den_so
        if axo_den_so == True:
            testct = np.in1d(cs_ssv_celltypes, pre_cts).reshape(len(cs_ssv_celltypes), 2)
            testax = np.in1d(cs_ssv_axoness, 1).reshape(len(cs_ssv_axoness), 2)
            pre_ct_inds = np.any(testct == testax, axis=1)
            cs_ssv_celltypes = cs_ssv_celltypes[pre_ct_inds]
            cs_ssv_mesh_areas = cs_ssv_mesh_areas[pre_ct_inds]
            cs_ssv_axoness = cs_ssv_axoness[pre_ct_inds]
            cs_ssv_neuron_partners = cs_ssv_neuron_partners[pre_ct_inds]
            cs_ssv_coords = cs_ssv_coords[pre_ct_inds]
        # exclude synapses without postcelltypes
        ct_inds = np.any(np.in1d(cs_ssv_celltypes, post_cts).reshape(len(cs_ssv_celltypes), 2), axis=1)
        cs_ssv_celltypes = cs_ssv_celltypes[ct_inds]
        cs_ssv_mesh_areas = cs_ssv_mesh_areas[ct_inds]
        cs_ssv_axoness = cs_ssv_axoness[ct_inds]
        cs_ssv_neuron_partners = cs_ssv_neuron_partners[ct_inds]
        cs_ssv_coords = cs_ssv_coords[ct_inds]
        # filter those where postcts are where axon is not, only if axo_den_so
        if axo_den_so == True:
            testct = np.in1d(cs_ssv_celltypes, post_cts).reshape(len(cs_ssv_celltypes), 2)
            testax = np.in1d(cs_ssv_axoness, [2, 0]).reshape(len(cs_ssv_celltypes), 2)
            post_ct_inds = np.any(testct == testax, axis=1)
            cs_ssv_celltypes = cs_ssv_celltypes[post_ct_inds]
            cs_ssv_mesh_areas = cs_ssv_mesh_areas[post_ct_inds]
            cs_ssv_axoness = cs_ssv_axoness[post_ct_inds]
            cs_ssv_neuron_partners = cs_ssv_neuron_partners[post_ct_inds]
            cs_ssv_coords = cs_ssv_coords[post_ct_inds]
    # only axo-dendritic or axo-somatic synapses allowed
    if axo_den_so:
        axs_inds = np.any(cs_ssv_axoness == 1, axis=1)
        cs_ssv_celltypes = cs_ssv_celltypes[axs_inds]
        cs_ssv_mesh_areas = cs_ssv_mesh_areas[axs_inds]
        cs_ssv_axoness = cs_ssv_axoness[axs_inds]
        cs_ssv_neuron_partners = cs_ssv_neuron_partners[axs_inds]
        cs_ssv_coords = cs_ssv_coords[axs_inds]
        den_so = np.array([0, 2])
        den_so_inds = np.any(np.in1d(cs_ssv_axoness, den_so).reshape(len(cs_ssv_axoness), 2), axis=1)
        cs_ssv_celltypes = cs_ssv_celltypes[den_so_inds]
        cs_ssv_mesh_areas = cs_ssv_mesh_areas[den_so_inds]
        cs_ssv_axoness = cs_ssv_axoness[den_so_inds]
        cs_ssv_neuron_partners = cs_ssv_neuron_partners[den_so_inds]
        cs_ssv_coords = cs_ssv_coords[den_so_inds]
    return cs_ssv_celltypes, cs_ssv_axoness, cs_ssv_neuron_partners, cs_ssv_mesh_areas, cs_ssv_coords




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
    cs_partners = sd_cs_ssv.load_numpy_data("neuron_partners")
    cs_ids = sd_cs_ssv.ids
    cs_coords = sd_cs_ssv.load_numpy_data("rep_coord")
    ct1_inds = np.any(np.in1d(cs_partners, cellids1).reshape(len(cs_partners), 2), axis=1)
    cs_partners = cs_partners[ct1_inds]
    cs_ids = cs_ids[ct1_inds]
    cs_coords = cs_coords[ct1_inds]
    ct2_inds = np.any(np.in1d(cs_partners, cellids2).reshape(len(cs_partners), 2), axis=1)
    cs_partners = cs_partners[ct2_inds]
    cs_ids = cs_ids[ct2_inds]
    cs_coords = cs_coords[ct2_inds]

    return cs_partners, cs_ids, cs_coords

def get_contact_site_compartment_axoness_percell(cs_dict, compartment):
    """
    get contact sites related to a specific compartment per cell
    :param cellid: id of the cell
    :param compartment: 0 = dendrite, 1 = axon; compartment contact site should be close to
    :param cs_coords: coordinates of all contact sites
    :param cs_ids: ids of all contact sites
    :param cs_partners: ids of partner cells
    :return: dictionary with contact site ids per cell
    """
    cellid = cs_dict["cellid"]
    sso = SuperSegmentationObject(cellid)
    sso.load_skeleton()
    cell_nodes = sso.skeleton["nodes"] * sso.scaling
    axo = np.array(sso.skeleton["axoness_avg10000"])
    axo[axo == 3] = 1
    axo[axo == 4] = 1
    cs_partners = cs_dict["cs partners"]
    cs_coords = cs_dict["cs coords"] * sso.scaling
    cs_ids = cs_dict["cs ids"]
    kdtree = scipy.spatial.cKDTree(cell_nodes)
    close_node_ids = kdtree.query(cs_coords, k=1)[1].astype(int)
    close_node_comp = np.array(axo[close_node_ids])
    close_node_comp_inds = np.where(close_node_comp == compartment)
    cs_dict["cs partners"] = cs_partners[close_node_comp_inds]
    cs_dict["cs coords"] = cs_coords[close_node_comp_inds]
    cs_dict["cs ids"] = cs_ids[close_node_comp_inds]
    return cs_dict

def get_contact_size_axoness_per_cell(input):
    '''
    get information about which cs_ids is in which compartment per cell.
    :param cellid: id of the cell
    :param cs_ids: all cs_ids that axoness information is required
    :param cs_coords: all cs coords where axoness information in required
    :param cs_partners: all cs to be processed with information about neuron_partners
    :return: dictionary with cs_ids sorted into each compartment
    '''
    cellid, cs_ids, cs_coords, cs_partners = input
    #set up dictionary
    cell_dict = {cellid: {}}
    #get cs which cellid is a part of
    cell_inds = np.any(np.in1d(cs_partners, cellid).reshape(len(cs_partners), 2), axis = 1)
    cell_cs_ids = cs_ids[cell_inds]
    cell_cs_coords = cs_coords[cell_inds]
    #get axoness information from cell
    cell = SuperSegmentationObject(cellid)
    cell.load_skeleton()
    cell_nodes = cell.skeleton["nodes"] * cell.scaling
    axo = np.array(cell.skeleton["axoness_avg10000"])
    axo[axo == 3] = 1
    axo[axo == 4] = 1
    kdtree = scipy.spatial.cKDTree(cell_nodes)
    dists, close_node_ids = kdtree.query(cell_cs_coords * cell.scaling)
    axoness_close_node_ids = axo[close_node_ids]
    axon_cs_ids = cell_cs_ids[axoness_close_node_ids == 1]
    dendrite_cs_ids = cell_cs_ids[axoness_close_node_ids == 0]
    soma_cs_ids = cell_cs_ids[axoness_close_node_ids == 2]
    cell_dict[cellid]['axon cs ids'] = axon_cs_ids
    cell_dict[cellid]['dendrite cs ids'] = dendrite_cs_ids
    cell_dict[cellid]['soma cs ids'] = soma_cs_ids
    return cell_dict

def get_percell_number_sumsize(ssvs, syn_sizes):
    '''
    get number of synapses per cell and sum synapse size
    :param ssvs: arary of ssv ids
    :param syn_sizes: array of synapse sizes
    :return: syn_numbers, syn_sizes, unique_ssv_ids
    '''
    ssv_inds, unique_ssv_ids = pd.factorize(ssvs)
    syn_ssv_sizes = np.bincount(ssv_inds, syn_sizes)
    syn_numbers = np.bincount(ssv_inds)
    return syn_numbers, syn_ssv_sizes, unique_ssv_ids


def get_number_sum_size_synapses(syn_ids, syn_sizes, syn_ssv_partners, syn_axs, syn_cts, ct, cellids, filter_ax = None,
                                 filter_ids = None, return_syn_arrays = True, filter_pre_ids = None, filter_post_ids = None):
    '''
    Get number of synapses and sum of synapses sizes for each cell in array. If filter_ax then only the compartment wanted;
    if filter_ids then only from specific cellids.
    :param syn_ids: array of synapse ids
    :param syn_sizes: array with synapse sizes
    :param syn_ssv_partners: synaptic partner cellids
    :param syn_axs: synaptic axoness parameters, 0 = dendrite, 1 = axon, 2 = soma
    :param syn_cts: celltypes of synaptic partners
    :param ct: celltype that should be filtered with
    :param cellids: cellids for cells looked for
    :param filter_ax: if given, filters for given compartments, give list or array
    :param filter_ids: if given only uses synapses between given ids
    :param return_syn_arrays: if True returns id, sizes, ssv_partners, syn_ax arrays
    :param filter_pre_ids: if True, makes sure only certain ids are presynaptic
    :param filter_post_ids: if True: makes sure only certain ids are postsynaptic
    :return: number of synapses, sum size of synapses, filtered synaptic parameter arrays
    '''
    if filter_ax is not None:
        ct_inds = np.in1d(syn_ssv_partners, cellids).reshape(len(syn_ssv_partners), 2)
        comp_inds = np.in1d(syn_axs, filter_ax).reshape(len(syn_cts), 2)
        filtered_inds = np.any(ct_inds == comp_inds, axis=1)
        syn_cts = syn_cts[filtered_inds]
        syn_ids = syn_ids[filtered_inds]
        syn_axs = syn_axs[filtered_inds]
        syn_ssv_partners = syn_ssv_partners[filtered_inds]
        syn_sizes = syn_sizes[filtered_inds]
    if filter_ids is not None:
        id_inds = np.all(np.in1d(syn_ssv_partners, filter_ids).reshape(len(syn_ssv_partners), 2),
                              axis=1)
        syn_cts = syn_cts[id_inds]
        syn_ids = syn_ids[id_inds]
        syn_ssv_partners = syn_ssv_partners[id_inds]
        syn_sizes = syn_sizes[id_inds]
        syn_axs = syn_axs[id_inds]
    if filter_pre_ids is not None:
        ct_inds = np.in1d(syn_ssv_partners, filter_pre_ids).reshape(len(syn_ssv_partners), 2)
        comp_inds = np.in1d(syn_axs, 1).reshape(len(syn_ssv_partners), 2)
        filtered_inds = np.all(ct_inds == comp_inds, axis=1)
        syn_cts = syn_cts[filtered_inds]
        syn_ids = syn_ids[filtered_inds]
        syn_axs = syn_axs[filtered_inds]
        syn_ssv_partners = syn_ssv_partners[filtered_inds]
        syn_sizes = syn_sizes[filtered_inds]
    if filter_post_ids is not None:
        ct_inds = np.in1d(syn_ssv_partners, filter_post_ids).reshape(len(syn_ssv_partners), 2)
        comp_inds = np.in1d(syn_axs, [0,2]).reshape(len(syn_ssv_partners), 2)
        filtered_inds = np.all(ct_inds == comp_inds, axis=1)
        syn_cts = syn_cts[filtered_inds]
        syn_ids = syn_ids[filtered_inds]
        syn_axs = syn_axs[filtered_inds]
        syn_ssv_partners = syn_ssv_partners[filtered_inds]
        syn_sizes = syn_sizes[filtered_inds]
    if filter_ax is not None:
        #uses position of compartment to identify cells to group by
        if filter_ax[0] == 1:
            sort_inds =np.where(syn_axs == 1)
        else:
            sort_inds = np.where(syn_axs != 1)
    else:
        #uses celltype to identify cells to group by
        sort_inds = np.where(syn_cts == ct)
    ssvs = syn_ssv_partners[sort_inds]
    syn_numbers, syn_ssv_sizes, unique_ssv_ids = get_percell_number_sumsize(ssvs = ssvs, syn_sizes = syn_sizes)
    if return_syn_arrays:
        return syn_ids, syn_sizes, syn_ssv_partners, syn_axs, syn_cts, unique_ssv_ids, syn_ssv_sizes, syn_numbers
    else:
        return unique_ssv_ids, syn_ssv_sizes, syn_numbers

def get_syn_input_distance_percell(args):
    '''
    Get median, min and max distance to soma per cell for a given set of coordinates.
    Use shortestpath2soma. This function returns values in µm
    :param args: cellid, coordinates
    :return: cellid, median distance, min distance, max distance per cell
    '''
    cellid = args[0]
    coords = args[1]
    cell = SuperSegmentationObject(cellid)
    cell.load_skeleton()
    distance2soma = cell.shortestpath2soma(coordinates=coords)
    distance2soma = np.array(distance2soma) / 1000 #in µm
    median_distance = np.median(distance2soma)
    min_distance = np.min(distance2soma)
    max_distance = np.max(distance2soma)
    return [cellid, median_distance, min_distance, max_distance, distance2soma]

def get_compartment_syn_number_sumsize(syn_sizes, syn_ssv_partners, syn_axs, syn_spiness = None, ax_comp = None, spiness_comp = None, return_syn_sizes = False, sort_per_postsyn_ct = True):
    '''
    Get number of synapses and sum size per postsynaptic cell for a given compartment via ax_comp for axon, soma, dendrite or
    with ax_comp = 0 and spiness for dendritic shaft, spine neck, spine head. If no compartment is gives, computes total amount.
    :param syn_sizes: size of synapses
    :param syn_ssv_partners: synaptic partner neuron ids
    :param syn_axs: axoness values, 0 = dendrite, 1 = axon, 2 = soma
    :param syn_spiness: spiness values: 0 = spine neck, 1 = spine head, 2 = dendritic shaft, 3 = other
    :param ax_comp: which axoness compartment is wanted, if None uses dendrite and soma
    :param spiness_comp: spiness compartment wanted, ax_comp has to be set to 0 for dendritic compartments
    :param return_syn_sizes: if true, return filtered sizes array
    :param sort_per_postsyn_ct: if True gives parameters sorted per postsynaptic celltype, else per presynaptic celltype
    :return: number of synapses, sum of synapse sizes per cell, cellids
    '''
    if ax_comp is None:
        if sort_per_postsyn_ct:
            sort_inds = np.where(syn_axs != 1)
        else:
            sort_inds = np.where(syn_axs == 1)
        post_ssvs = syn_ssv_partners[sort_inds]
        ssv_inds, unique_post_ssvs = pd.factorize(post_ssvs)
        syn_ssv_sizes = np.bincount(ssv_inds, syn_sizes)
        syn_numbers = np.bincount(ssv_inds)
        comp_sizes = syn_ssv_sizes
    else:
        comp_inds = np.any(np.in1d(syn_axs, ax_comp).reshape(len(syn_axs), 2), axis=1)
        comp_ssv_partners = syn_ssv_partners[comp_inds]
        comp_sizes = syn_sizes[comp_inds]
        comp_axs = syn_axs[comp_inds]
        if syn_spiness is not None:
            comp_spiness = syn_spiness[comp_inds]
        if spiness_comp is not None:
            if syn_spiness is None:
                raise ValueError('Synaptic spiness info must be given to filter for spiness values')
            if ax_comp != 0:
                raise ValueError('When filter for spiness information, ax_comp must be set to 0')
            comp_inds = np.any(np.in1d(comp_spiness, spiness_comp).reshape(len(comp_spiness), 2), axis=1)
            comp_ssv_partners = comp_ssv_partners[comp_inds]
            comp_sizes = comp_sizes[comp_inds]
            comp_axs = comp_axs[comp_inds]
        if sort_per_postsyn_ct:
            sort_inds = np.where(comp_axs == ax_comp)
        else:
            sort_inds = np.where(comp_axs == 1)
        sort_ssvs = comp_ssv_partners[sort_inds]
        syn_numbers, syn_ssv_sizes, unique_post_ssvs = get_percell_number_sumsize(ssvs = sort_ssvs, syn_sizes = comp_sizes)
    if return_syn_sizes:
        return syn_numbers, syn_ssv_sizes, unique_post_ssvs, comp_sizes
    else:
        return syn_numbers, syn_ssv_sizes, unique_post_ssvs

def get_ct_syn_number_sumsize(syn_sizes, syn_ssv_partners, syn_cts, ct):
    '''
    Get number of synapses and sum size per cell for a given celltype that can be pre- or postsynaptic.
    Assumes that incoming arrays are filtered and all synapses can be used.
    :param syn_sizes: size of synapses
    :param syn_ssv_partners: synaptic partner neuron ids
    :param syn_cts: celltypes for synaptic partner neurons
    :param ct: celltype that should summed for
    :return: number of synapses, sum of synapse sizes per cell, cellids
    '''
    sort_inds = np.where(syn_cts == ct)
    ssvs = syn_ssv_partners[sort_inds]
    syn_numbers, syn_ssv_sizes, unique_post_ssvs = get_percell_number_sumsize(ssvs = ssvs, syn_sizes = syn_sizes)
    return syn_numbers, syn_ssv_sizes, unique_post_ssvs

def get_ct_information_npy(ssv_partners, cellids_array_full, celltype_array_full, desired_ssv_ids = None):
    '''
    For a given segmentation dataset, get information about celltypes from numpy array with all cellids and celltypes
    :param ssv_partners: numpy array with saved ssv_partners
    :param cellids_array_full: all cellids for SuperSegmentationDataset
    :param celltype_array_full: celltype numpy array for SuperSegmentationDataset, same length as cellids_array_full
    :param desired_ssv_ids: if given, will filter for these cellids
    :return:
    '''

    if desired_ssv_ids is not None:
        id_inds = np.in1d(cellids_array_full, desired_ssv_ids)
        cellids_array_full = cellids_array_full[id_inds]
        celltype_array_full = celltype_array_full[id_inds]
        id_inds2 = np.all(np.in1d(ssv_partners, desired_ssv_ids).reshape(len(ssv_partners), 2), axis = 1)
        ssv_partners = ssv_partners[id_inds2]
    celltypes = np.unique(celltype_array_full)
    celltype_partners = np.zeros((len(ssv_partners), 2)) -1
    for ct in celltypes:
        ct_ids = cellids_array_full[celltype_array_full == ct]
        ct_mask = np.in1d(ssv_partners, ct_ids).reshape(len(ssv_partners), 2)
        ct_inds = np.where(ct_mask == True)
        celltype_partners[ct_inds] = ct
    assert(np.all(np.unique(celltype_partners) == celltypes))
    return celltype_partners






