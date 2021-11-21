
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


def get_compartment_length(sso, compartment, cell_graph):
    """
            calculates length of compartment in µm per cell using the skeleton if given the networkx graph of the cell.
            :param compartment: 0 = dendrite, 1 = axon, 2 = soma
            :param cell_graph: sso.weighted graph
            :param min_comp_len: minimum compartment length, if not return 0 [µm]
            :return: comp_len
            """
    non_comp_inds = np.nonzero(sso.skeleton["axoness_avg10000"] != compartment)[0]
    comp_graph = cell_graph.copy()
    comp_graph.remove_nodes_from(non_comp_inds)
    comp_length = comp_graph.size(weight="weight") / 1000  # in µm
    return comp_length


def get_spine_density(cell, min_comp_len = 100):
    """
    calculates the spine density of the dendrite.Therefore, the amount of spines per µm dendrite is calculated.
     Amount of spines is the number of connected_components with spiness = spines.
     # spiness values: 0 = spine neck, 1 = spine head, 2 = dendritic shaft, 3 = other
    :param cell: super-segmentation object
    :param min_comp_len: minimum compartment length in µm
    :return: amount of spines on dendrite, 0 if not having min_comp_len
    """
    cell.load_skeleton()
    g = cell.weighted_graph(add_node_attr=('axoness_avg10000',))
    axon_length = get_compartment_length(cell, compartment = 1, cell_graph = g)
    if axon_length < min_comp_len:
        return 0
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




