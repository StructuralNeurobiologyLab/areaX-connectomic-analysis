
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


def compartment_length_cell(sso, compartment, cell_graph):
    """
            calculates length of compartment per cell using the skeleton if given the networkx graph of the cell.
            :param compartment: 0 = dendrite, 1 = axon, 2 = soma
            :param cell_graph: sso.weighted graph
            :param min_comp_len: minimum compartment length, if not return 0
            :return: comp_len
            """
    non_comp_inds = np.nonzero(sso.skeleton["axoness_avg10000"] != compartment)[0]
    comp_graph = cell_graph.copy()
    comp_graph.remove_nodes_from(non_comp_inds)
    comp_length = comp_graph.size(weight="weight") / 1000  # in µm
    return comp_length


def counting_spines(cell, min_comp_len = 100):
    """
    determines the amount of spines using the skeleton. Amount of spines is the number of connected_components with spiness = spines.
    :param cell: super-segmentation object
    :param min_comp_len: minimum compartment length in µm
    :return: amount of spines on dendrite, 0 if not having min_comp_len
    """
    cell.load_skeleton()
    g = cell.weighted_graph(add_node_attr=('axoness_avg10000',))
    axon_length = compartment_length_cell(cell, compartment = 1, cell_graph = g)
    if axon_length < min_comp_len:
        return 0
    dendrite_length = compartment_length_cell(cell, compartment = 0, cell_graph = g)
    if dendrite_length < min_comp_len:
        return 0
    spine_shaftinds = np.nonzero(cell.skeleton["spiness"] == 0)[0]
    spine_otherinds = np.nonzero(cell.skeleton["spiness"] == 3)[0]
    nonspine_inds = np.hstack([spine_shaftinds, spine_otherinds])
    spine_graph = g.copy()
    spine_graph.remove_nodes_from(nonspine_inds)
    spine_amount = len(list(nx.connected_component_subgraphs(spine_graph)))
    return spine_amount