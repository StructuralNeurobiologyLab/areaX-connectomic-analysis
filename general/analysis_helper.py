
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


def get_compartment_length(sso, compartment, cell_graph):
    """
            calculates length of compartment in µm per cell using the skeleton if given the networkx graph of the cell.
            :param compartment: 0 = dendrite, 1 = axon, 2 = soma
            :param cell_graph: sso.weighted graph
            :return: comp_len in µm
            """
    non_comp_inds = np.nonzero(sso.skeleton["axoness_avg10000"] != compartment)[0]
    comp_graph = cell_graph.copy()
    comp_graph.remove_nodes_from(non_comp_inds)
    comp_length = comp_graph.size(weight="weight") / 1000  # in µm
    return comp_length


def get_spine_density(cellid , min_comp_len = 100, full_cell_dict = None):
    """
    calculates the spine density of the dendrite.Therefore, the amount of spines per µm dendrite is calculated.
     Amount of spines is the number of connected_components with spiness = spines.
     # spiness values: 0 = spine neck, 1 = spine head, 2 = dendritic shaft, 3 = other
    :param cell: super-segmentation object
    :param min_comp_len: minimum compartment length in µm
    :param full_cell_dict: dictionary with per cell parameter values, cell.id is key
    :return: amount of spines on dendrite, 0 if not having min_comp_len
    """
    cell = SuperSegmentationObject(cellid)
    cell.load_skeleton()
    g = cell.weighted_graph(add_node_attr=('axoness_avg10000',))
    # use axon and dendrite length dictionaries to lookup axon and dendrite lenght in future versions
    if full_cell_dict is not None:
        axon_length = full_cell_dict[cell.id]["axon length"]
    else:
        axon_length = get_compartment_length(cell, compartment = 1, cell_graph = g)
    if axon_length < min_comp_len:
        return 0
    if full_cell_dict is not None:
        dendrite_length = full_cell_dict[cell.id]["dendrite length"]
    else:
        dendrite_length = get_compartment_length(cell, compartment = 0, cell_graph = g)
    if dendrite_length < min_comp_len:
        return 0
    spine_shaftinds = np.nonzero(cell.skeleton["spiness"] == 2)[0]
    spine_otherinds = np.nonzero(cell.skeleton["spiness"] == 3)[0]
    nonspine_inds = np.hstack([spine_shaftinds, spine_otherinds])
    spine_head_inds = np.nonzero(cell.skeleton["spiness"] == 1)[0]
    spine_neck_inds = np.nonzero(cell.skeleton["spiness"] == 0)[0]
    spine_inds = np.hstack([spine_head_inds, spine_neck_inds])
    nospine_graph = g.copy()
    nospine_graph.remove_nodes_from(spine_inds)
    no_spine_dendrite_length = nospine_graph.size(weight="weight") / 1000
    spine_graph = g.copy()
    spine_graph.remove_nodes_from(nonspine_inds)
    spine_amount = len(list(nx.connected_component_subgraphs(spine_graph)))
    spine_density = spine_amount/no_spine_dendrite_length
    return spine_density

def get_compartment_radii(cell, comp_inds = None):
    """
    get radii from compartment graph of one cell
    :param comp_inds: indicies of compartment
    :return: comp_radii as array in µm
    """
    if not np.all(comp_inds) is None:
        comp_radii = cell.skeleton["diameters"][comp_inds] * 2 * cell.scaling[0] / 1000 #in µm
    else:
        comp_radii = cell.skeleton["diameters"]* 2 * cell.scaling[0] / 1000  # in µm
    return comp_radii

def get_compartment_bbvolume(comp_nodes):
    """
    calculates the bounding box volume of a given compartment
    :param comp_nodes: nodes belonging to a specific compartment
    :return: compartment volume as µm³
    """
    min_x = np.min(comp_nodes[:, 0])
    max_x = np.max(comp_nodes[:, 0])
    min_y = np.min(comp_nodes[:, 1])
    max_y = np.max(comp_nodes[:, 1])
    min_z = np.min(comp_nodes[:, 2])
    max_z = np.max(comp_nodes[:, 2])
    comp_volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z) * 10 ** (-9)  # in µm
    return comp_volume

def get_compartment_tortuosity_complete(comp_len, comp_nodes):
    """
    calculates tortuosity as ratio of pathlength vs bounding box length to the power of 2
    :param comp_len: pathlenght of compartment
    :param comp_nodes: compartment nodes
    :return: tortuosity
    """
    min_x = np.min(comp_nodes[:, 0])
    max_x = np.max(comp_nodes[:, 0])
    min_y = np.min(comp_nodes[:, 1])
    max_y = np.max(comp_nodes[:, 1])
    min_z = np.min(comp_nodes[:, 2])
    max_z = np.max(comp_nodes[:, 2])
    min = np.array([min_x, min_y, min_z])
    max = np.array([max_x, max_y, max_z])
    diagonal = np.linalg.norm(max - min)/ 1000 #in µm
    tortuosity = (comp_len/diagonal)**2
    return tortuosity

def get_compartment_tortuosity_sampled(comp_graph, comp_nodes, n_samples = 1000, n_radius = 1000):
    """
    calculates tortuosity as average of tortuosity in different samples of the compartment. Torutosity is the quadrat of the
    pathlength vs the bounding box diagonal
    :param comp_graph: compartment graph
    :param comp_nodes: compartment nodes in physical scale
    :param n_samples: amount of samples drawn
    :param n_radius: radius to be sampled in nm
    :return: averaged tortuosity
    """
    kdtree = scipy.spatial.cKDTree(comp_nodes)
    tortuosities = np.empty(n_samples)
    for i in range(n_samples):
        random_node_ind = np.random.choice(range(len(comp_nodes)))
        random_node = comp_nodes[random_node_ind]
        sample_ids = kdtree.query_ball_point(random_node, n_radius)
        if len(sample_ids) == 1:
            continue
        sample_graph = comp_graph.subgraph(sample_ids)
        sample_nodes = comp_nodes[sample_ids]
        sample_length = sample_graph.size(weight="weight") / 1000  # in µm
        min_x = np.min(sample_nodes[:, 0])
        max_x = np.max(sample_nodes[:, 0])
        min_y = np.min(sample_nodes[:, 1])
        max_y = np.max(sample_nodes[:, 1])
        min_z = np.min(sample_nodes[:, 2])
        max_z = np.max(sample_nodes[:, 2])
        min = np.array([min_x, min_y, min_z])
        max = np.array([max_x, max_y, max_z])
        sample_diagonal = np.linalg.norm(max - min) / 1000  # in µm
        sample_tortuosity = (sample_length/ sample_diagonal) ** 2
        tortuosities[i] = sample_tortuosity
    avg_tortuosity = np.nanmean(tortuosities)
    return avg_tortuosity

def get_myelin_fraction(cell, min_comp_len = 100):
    """
    calculate length and fraction of myelin for axon. Skeleton has to be loaded
    :param cell:super-segmentation object graph should be calculated on
    :param min_comp_len: compartment lengfh threshold
    :return: absolute length of mylein, relative length of myelin
    """
    non_axon_inds = np.nonzero(cell.skeleton["axoness_avg10000"] != 1)[0]
    non_myelin_inds = np.nonzero(cell.skeleton["myelin"] == 0)[0]
    g = cell.weighted_graph(add_node_attr=('axoness_avg10000', "myelin"))
    axon_graph = g.copy()
    axon_graph.remove_nodes_from(non_axon_inds)
    axon_length = axon_graph.size(weight="weight") / 1000  # in µm
    if axon_length < min_comp_len:
        return 0,0
    myelin_graph = axon_graph.copy()
    myelin_graph.remove_nodes_from(non_myelin_inds)
    absolute_myelin_length = myelin_graph.size(weight="weight") / 1000  # in µm
    relative_myelin_length = absolute_myelin_length / axon_length
    return absolute_myelin_length, relative_myelin_length

def get_organell_volume_density(cell, segmentation_object_ids, cached_so_ids,cached_so_rep_coord, cached_so_volume, full_cell_dict = None, k = 3, min_comp_len = 100):
    '''
    calculate density and volume density of a supersegmentation object per cell for axon and dendrite. Skeleton has to be loaded
    :param cell: super segmentation object
    :param segmentation_object_ids: organell ids per cell
    :param cached_so_ids: cached ids for organell of all cells
    :param cached_so_rep_coord: cached coordinates for organells of all cells
    :param cached_so_volume: cached organell volume for all cells
    :param full_cell_dict: lookup dictionary for per cell parameters, cell.id is key, if None given will be calculated
    :param k: number of nodes surrounding the organells compartment will be determined from
    :param min_comp_len: minimum compartment length
    :return: densities and volume densities for aoxn and dendrite
    '''

    kdtree = scipy.spatial.cKDTree(cell.skeleton["nodes"]*cell.scaling)
    sso_organell_inds = np.in1d(cached_so_ids, segmentation_object_ids)
    organell_volumes = cached_so_volume[sso_organell_inds] * 10 ** (-9) * np.prod(cell.scaling)  # convert to cubic µm
    so_rep_coord = cached_so_rep_coord[sso_organell_inds] * cell.scaling # in nm
    close_node_ids = kdtree.query(so_rep_coord, k=k)[1].astype(int)
    axo = np.array(cell.skeleton["axoness_avg10000"][close_node_ids])
    axo[axo == 3] = 1
    axo[axo == 4] = 1
    axon_unique = np.unique(np.where(axo == 1)[0], return_counts=True)
    axon_inds = axon_unique[0][axon_unique[1] > k / 2]
    axo_so_ids = segmentation_object_ids[axon_inds]
    den_unique = np.unique(np.where(axo == 0)[0], return_counts=True)
    den_inds = den_unique[0][den_unique[1] > k / 2]
    den_so_ids = segmentation_object_ids[den_inds]
    non_soma_inds = np.hstack([axon_inds, den_inds])
    segmentation_object_ids = segmentation_object_ids[non_soma_inds]
    if len(segmentation_object_ids) == 0:
        return 0, 0, 0, 0
    axo_so_amount = len(axo_so_ids)
    den_so_amount = len(den_so_ids)
    if full_cell_dict is not None:
        axon_length = full_cell_dict[cell.id]["axon length"]
    else:
        g = cell.weighted_graph(add_node_attr=('axoness_avg10000',))
        axon_length = get_compartment_length(cell, compartment=1, cell_graph=g)
    if axon_length < min_comp_len:
        return 0,0, 0, 0
    if full_cell_dict is not None:
        dendrite_length = full_cell_dict[cell.id]["dendrite length"]
    else:
        dendrite_length = get_compartment_length(cell, compartment=0, cell_graph=g)
    if dendrite_length < min_comp_len:
        return 0,0, 0, 0
    axo_so_density = axo_so_amount/ axon_length
    den_so_density = den_so_amount/dendrite_length
    axo_so_volume = np.sum(organell_volumes[axon_inds])
    den_so_volume = np.sum(organell_volumes[den_inds])
    axo_so_volume_density = axo_so_volume/ axon_length
    den_so_volume_density = den_so_volume/ dendrite_length
    return axo_so_density, den_so_density, axo_so_volume_density, den_so_volume_density

def get_compartment_mesh_area(cell):
    """
    get compartment mesh areas using compartmentalize_mesh and mesh_area_calc.
    :param cell: sso
    :return: dictionary with mesh_areas of axon, dendrite and soma
    """
    comp_meshes = compartmentalize_mesh_fromskel(cell)
    compartments = ["axon", "dendrite", "soma"]
    mesh_areas = {}
    for comp in compartments:
        mesh_areas[comp] = mesh_area_calc(comp_meshes[comp])

    return mesh_areas

def check_comp_lengths_ct(cellids, fullcelldict = None, min_comp_len = 200):
    """
    iterates of cellids and checks if their compartment length (axon, dendrite) are
    over a certain threshold.
    :param cellids: array or list with cellids to be checked
    :param fullcelldict: if given, will use this as a lookup dictionary
    :param min_comp_len: minimum compartment length in µm
    :return: cellids that fulfil requirement
    """
    checked_cells = np.zeros(len(cellids))
    for i, cellid in enumerate(tqdm(cellids)):
        if fullcelldict is not None:
            cell_axon_length = fullcelldict[cellid]["axon length"]
            if cell_axon_length < min_comp_len:
                continue
            cell_den_length = fullcelldict[cellid]["dendrite length"]
            if cell_den_length < min_comp_len:
                continue
        else:
            cell = SuperSegmentationObject(cellid)
            cell.load_skeleton()
            g = cell.weighted_graph(add_node_attr=('axoness_avg10000',))
            cell_axon_length = get_compartment_length(cell, compartment=1, cell_graph=g)
            if cell_axon_length < min_comp_len:
                continue
            cell_den_length = get_compartment_length(cell, compartment=0, cell_graph=g)
            if cell_den_length < min_comp_len:
                continue
        checked_cells[i] = cellid

    checked_cells = checked_cells[checked_cells > 0]
    return checked_cells

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


