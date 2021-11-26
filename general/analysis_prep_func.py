from collections import defaultdict
import numpy as np
from u.arother.bio_analysis.general.analysis_helper import get_compartment_length
from tqdm import tqdm

def find_full_cells(ssd, celltype, soma_centre = True, syn_proba = 0.6, shortestpaths = True):
    """
    function finds full cells of a specific celltype if the cells have a dendrite, soma and axon in axoness_avg10000.
    :param ssd: segmentation dataset
    :param celltype: number of the celltype that is searched for; celltypes: j0126: STN=0, modulatory=1, MSN=2, LMAN=3, HVC=4, GP=5, INT=6
    # j0256: STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7, FS=8, LTS=9, NGF=10
    :param soma_centre: if True calculates average of soma skeleton notes as approximation to the soma centre
    :param shortestpath: returns shortest paths for all nodes that are not soma
    :param syn_proba: synapse probability
    :return: an array with cell_ids of the full_cells and if soma centre was calculated also a dictionary for each cell with its soma_centre
    """
    celltype_ids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == celltype]
    if soma_centre:
        soma_arr = np.zeros((len(celltype_ids), 3))
    full_cells = np.zeros((len(celltype_ids)))
    axon_length = np.zeros((len(celltype_ids)))
    dendrite_length = np.zeros((len(celltype_ids)))

    for i, cell in enumerate(tqdm(ssd.get_super_segmentation_object(celltype_ids))):
        cell.load_skeleton()
        axoness = cell.skeleton["axoness_avg10000"]
        axoness[axoness == 3] = 1
        axoness[axoness == 4] = 1
        unique_preds = np.unique(axoness)
        if not (0 in unique_preds and 1 in unique_preds and 2 in unique_preds):
            continue
        full_cells[i] = int(cell.id)
        # add compartment calculation for axon/ dendrite
        g = cell.weighted_graph()
        axon_length_cell = get_compartment_length(cell, compartment = 1, cell_graph = g)
        dendrite_length_cell = get_compartment_length(cell, compartment = 0, cell_graph = g)
        axon_length[i] = axon_length_cell
        dendrite_length[i] = dendrite_length_cell
        if soma_centre:
            soma_inds = np.nonzero(cell.skeleton["axoness_avg10000"] == 2)[0]
            positions = cell.skeleton["nodes"][soma_inds] * ssd.scaling #transform to nm
            soma_centre_coord = np.mean(positions, axis=0)
            soma_arr[i] = soma_centre_coord
        if shortestpaths:
            path_array = np.zeros(len(cell.skeleton["nodes"]))
            nonsoma_inds = np.nonzero(cell.skeleton["axoness_avg10000"] != 2)[0]
            coords = cell.skeleton["nodes"][nonsoma_inds]
            cell.skeleton["shortestpaths"] = np.zeros(len(cell.skeleton["nodes"]))
            shortespaths =  cell.shortestpath2soma(coords)
            cell.skeleton["shortestpaths"] = np.zeros(len(cell.skeleton["nodes"]))
            cell.skeleton["shortestpaths"][nonsoma_inds] = shortespaths

    inds = np.array(full_cells != 0)
    full_cells = full_cells[inds].astype(int)
    axon_length = axon_length[inds]
    dendrite_length = dendrite_length[inds]
    axon_dict = {int(full_cells[i]): axon_length[i] for i in range(0, len(full_cells))}
    dendrite_dict = {int(full_cells[i]): dendrite_length[i] for i in range(0, len(full_cells))}

    if soma_centre==True:
        soma_arr = soma_arr[inds]
        full_cells_dict = {int(full_cells[i]): soma_arr[i] for i in range(0, len(full_cells))}
        return full_cells, full_cells_dict, axon_dict, dendrite_dict
    else:
        return full_cells, axon_dict, dendrite_dict


#def create_syn_dict(sd_synssv, syn_prob = 0.6):


def indentify_branches(graph, min = 5, max = 20):
    """
    finds small branches in skeletons, see sparsify_skeleton_fast
    """
    for node in graph:
        if graph.degree(node) != 1:
            continue
        visiting_node = node
        while graph.degree(visiting_node <= 2):
            neighbours = [n for n in graph.neighbors(visiting_node)]

def synapse_amount_percell(celltype, sd_synssv,cellids, syn_proba):
    '''
    gives amount of synapses for each cell with defined synapse probability and writes it in a dictionary
    :param celltype: celltype analysis is wanted for
    :param sd_synssv: synapse daatset
    :param syn_proba: synapse probability
    :param cellids: cellids of cells wanted amount of synapses for
    :return: dictionary with cell_ids as keys and amount of synapses
    '''
    syn_prob = sd_synssv.load_cached_data("syn_prob")
    m = syn_prob > syn_proba
    m_cts = sd_synssv.load_cached_data("partner_celltypes")[m]
    m_ssv_partners = sd_synssv.load_cached_data("neuron_partners")[m]
    inds = np.any(m_cts == celltype, axis=1)
    m_ssv_cts = m_ssv_partners[inds]
    uniques = np.unique(m_ssv_cts, return_counts=True)
    unique_syn_ids = uniques[0]
    unique_counts = uniques[1]
    syn_amount_dict = {ui: unique_counts[i] for i, ui in enumerate(unique_syn_ids) if ui in cellids}
    return syn_amount_dict

def pernode__shortestpath(cell):
    """
    creates dictionary for all nodes in one cell that are not soma nodes, soma nodes are given the entry 0
    :param cell: sso nodes should be computed for
    :return: array with shortestpath2soma as entry
    """
    cell.load_skeleton()
    g = cell.weighted_graph()
    path_array = np.zeros(len(cell.skeleton["nodes"]))
    nonsoma_inds = np.nonzero(cell.skeleton["axoness_avg10000"] != 2)[0]
    coords = cell.skeleton["nodes"][nonsoma_inds]
    #shortespaths =  shortestpath2soma(coords)
    #add to skeleton as dictionary!