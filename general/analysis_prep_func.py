from collections import defaultdict
import numpy as np
from wholebrain.scratch.arother.bio_analysis.general.analysis_helper import get_compartment_length, get_compartment_mesh_area
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from syconn.proc.meshes import mesh_area_calc

def find_full_cells(ssd, celltype, soma_centre = True, shortestpaths = True):
    """
    function finds full cells of a specific celltype if the cells have a dendrite, soma and axon in axoness_avg10000.
    :param ssd: segmentation dataset
    :param celltype: number of the celltype that is searched for; celltypes: j0126: STN=0, modulatory=1, MSN=2, LMAN=3, HVC=4, GP=5, INT=6
    # j0251: STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7, FS=8, LTS=9, NGF=10
    :param soma_centre: if True calculates average of soma skeleton notes as approximation to the soma centre
    :param shortestpath: returns shortest paths for all nodes that are not soma
    :param syn_proba: synapse probability
    :return: an array with cell_ids of the full_cells and if soma centre was calculated also a dictionary for each cell with its soma_centre
    """
    celltype_ids = ssd.ssv_ids[ssd.load_numpy_data("celltype_cnn_e3") == celltype]
    if soma_centre:
        full_cell_dict = defaultdict(lambda: {"axon length": 0, "dendrite length": 0, "axon mesh surface area": 0,
                                              "dendrite mesh surface area": 0, "soma centre": np.zeros(3)})
    else:
        full_cell_dict = defaultdict(lambda: {"axon length": 0, "dendrite length": 0, "axon mesh surface area": 0,
                                              "dendrite mesh surface area": 0,
                                              "soma mesh surface area": 0})
    full_cells = np.zeros((len(celltype_ids)))
    raise ValueError

    for i, cell in enumerate(tqdm(ssd.get_super_segmentation_object(celltype_ids))):
        if i > 150:
            break
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
        full_cell_dict[cell.id]["axon length"] = axon_length_cell
        full_cell_dict[cell.id]["dendrite length"] = dendrite_length_cell
        mesh_surface_areas_cell = get_compartment_mesh_area(cell)
        full_cell_dict[cell.id]["axon mesh surface area"] = mesh_surface_areas_cell["axon"]
        full_cell_dict[cell.id]["denrite mesh surface area"] = mesh_surface_areas_cell["dendrite"]
        full_cell_dict[cell.id]["soma mesh surface area"] = mesh_surface_areas_cell["soma"]
        if soma_centre:
            soma_inds = np.nonzero(cell.skeleton["axoness_avg10000"] == 2)[0]
            positions = cell.skeleton["nodes"][soma_inds] * ssd.scaling #transform to nm
            soma_centre_coord = np.mean(positions, axis=0)
            full_cell_dict[cell.id]["soma centre"] = soma_centre_coord
        if shortestpaths:
            path_array = np.zeros(len(cell.skeleton["nodes"]))
            nonsoma_inds = np.nonzero(cell.skeleton["axoness_avg10000"] != 2)[0]
            coords = cell.skeleton["nodes"][nonsoma_inds]
            cell.skeleton["shortestpaths"] = np.zeros(len(cell.skeleton["nodes"]))
            shortespaths =  cell.shortestpath2soma(coords)
            cell.skeleton["shortestpaths"] = np.zeros(len(cell.skeleton["nodes"]))
            cell.skeleton["shortestpaths"][nonsoma_inds] = shortespaths

    full_cells = full_cells[full_cells > 0]

    return full_cells, full_cell_dict

def get_axon_length_area_perct(ssd, celltype):
    """
    writes axon length and axon surface mesh area in dictionary with each cell.
    :param ssd: ssd: segmentation dataset
    :param celltype: :param celltype: number of the celltype that is searched for; celltypes: j0126: STN=0, modulatory=1, MSN=2, LMAN=3, HVC=4, GP=5, INT=6
    # j0256: STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7, FS=8, LTS=9, NGF=10
    :return: dictionary with axon length and surface mesh area
    """
    celltype_ids = ssd.ssv_ids[ssd.load_numpy_data("celltype_cnn_e3") == celltype]
    axon_dict = defaultdict(lambda: {"axon length": 0, "axon mesh surface area": 0})
    for i, axon in enumerate(tqdm(ssd.get_super_segmentation_object(celltype_ids))):
        axon.load_skeleton()
        g = axon.weighted_graph()
        axon_length = get_compartment_length(axon, compartment = 1, cell_graph = g)
        axon_dict[axon.id]["axon length"] = axon_length
        axon_dict[axon.id]["axon mesh surface area"] = mesh_area_calc(axon.mesh)

    return axon_dict


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

def synapse_amount_percell(celltype, syn_cts, syn_ssv_partners, syn_sizes, syn_axs, axo_denso = True, all_comps = True):
    '''
    gives amount and summed synapse size for each cell and writes it in dictionary. Calculates outgoing  synapses and incoming synapses
    (dendrite, soma) seperately.
    :param celltype: celltype analysis is wanted for
    :param syn_cts: celltypes of synaptic partners
    :param syn_ssv_partners: cellids of synaptic partners
    :param syn_sizes: synapse sizes
    :param syn_axs: axoness values of synaptic partners
    :param axo_denso: if True, only axo-dendritic or axo-somatic connections
    :param all_comps: if True synapses from all compartments, otherwise only if celltype is axon
    :return: dictionary with cell_ids as keys and amount of synapses
    '''
    if axo_denso:
        #remove all synapses that are not axo-dendritic or axo_somatic
        axs_inds = np.any(syn_axs == 1, axis=1)
        syn_cts = syn_cts[axs_inds]
        syn_axs = syn_axs[axs_inds]
        syn_ssv_partners = syn_ssv_partners[axs_inds]
        syn_sizes = syn_sizes[axs_inds]
        den_so = np.array([0, 2])
        den_so_inds = np.any(np.in1d(syn_axs, den_so).reshape(len(syn_axs), 2), axis=1)
        syn_cts = syn_cts[den_so_inds]
        syn_axs = syn_axs[den_so_inds]
        syn_ssv_partners = syn_ssv_partners[den_so_inds]
        syn_sizes = syn_sizes[den_so_inds]
    #get synapses where celltype is involved
    inds = np.any(syn_cts == celltype, axis=1)
    ct_ssv_partners = syn_ssv_partners[inds]
    ct_cts = syn_cts[inds]
    ct_axs = syn_axs[inds]
    ct_sizes = syn_sizes[inds]
    #get indices from celltype, axon, dendrite, soma for sorting later
    ct_inds = np.where(ct_cts == celltype)
    axon_inds = np.where(ct_axs == 1)
    axo_ct_inds = np.where(ct_cts[axon_inds] == celltype)
    # get unique cellids from cells whose axons make connections, count them and sum up sizes
    axo_ssvs = ct_ssv_partners[axo_ct_inds, axon_inds[1][axo_ct_inds]][0]
    axo_sizes = ct_sizes[axo_ct_inds]
    axo_ssv_inds, unique_axo_ssvs = pd.factorize(axo_ssvs)
    axo_syn_sizes = np.bincount(axo_ssv_inds, axo_sizes)
    axo_amounts = np.bincount(axo_ssv_inds)
    # create dictionaries for axon
    axon_dict = {cellid: {"amount": axo_amounts[i], "summed size": axo_syn_sizes[i]} for i, cellid in
                 enumerate(unique_axo_ssvs)}
    if all_comps == True:
        den_inds = np.where(ct_axs == 0)
        som_inds = np.where(ct_axs == 2)
        den_ct_inds = np.where(ct_cts[den_inds] == celltype)
        som_ct_inds = np.where(ct_cts[som_inds] == celltype)
        # get unique cellids from cells whose dendrite receive synapses, count them and sum up sizes
        den_ssvs = ct_ssv_partners[den_ct_inds, den_inds[1][den_ct_inds]][0]
        den_sizes = ct_sizes[den_ct_inds]
        den_ssv_inds, unique_den_ssvs = pd.factorize(den_ssvs)
        den_syn_sizes = np.bincount(den_ssv_inds, den_sizes)
        den_amounts = np.bincount(den_ssv_inds)
        # get unique cellids from cells whose soma receive synapses, count them and sum up sizes
        som_ssvs = ct_ssv_partners[som_ct_inds, som_inds[1][som_ct_inds]][0]
        som_sizes = ct_sizes[som_ct_inds]
        som_ssv_inds, unique_som_ssvs = pd.factorize(som_ssvs)
        som_syn_sizes = np.bincount(som_ssv_inds, som_sizes)
        som_amounts = np.bincount(som_ssv_inds)
        # create dictionaries for soma, dendrite synapses
        den_dict = {cellid: {"amount": den_amounts[i], "summed size": den_syn_sizes[i]} for i, cellid in
                     enumerate(unique_den_ssvs)}
        soma_dict = {cellid: {"amount": som_amounts[i], "summed size": som_syn_sizes[i]} for i, cellid in
                     enumerate(unique_som_ssvs)}
        return axon_dict, den_dict, soma_dict
    else:
        return axon_dict

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




