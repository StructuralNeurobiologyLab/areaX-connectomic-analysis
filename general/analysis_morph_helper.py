
import numpy as np
import networkx as nx
import scipy
from tqdm import tqdm
from syconn.proc.meshes import mesh_area_calc, compartmentalize_mesh_fromskel
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.handler.basics import load_pkl2obj
from scipy.spatial import cKDTree
from syconn.proc.meshes import write_mesh2kzip
import pandas as pd
from collections import Counter
import matplotlib.colors as co

def get_cell_length(cellid):
    '''
    Calculate total length of cell or fragment from cell graph
    :param cellid: id of the cell
    :return: total length in µm
    '''
    cell = SuperSegmentationObject(cellid)
    cell_graph = cell.weighted_graph()
    total_length = cell_graph.size(weight="weight") / 1000  # in µm
    return total_length

def get_compartment_length(cellid, compartment, cell_graph=None, full_dict = None):
    """
            calculates length of compartment in µm per cell using the skeleton if given the networkx graph of the cell.
            :param compartment: 0 = dendrite, 1 = axon, 2 = soma
            :param cell_graph: sso.weighted graph
            :return: comp_len in µm
            """
    if full_dict is not None:
        if cellid in full_dict.keys():
            comp_dict = {0: 'dendrite', 1: 'axon', 2: 'soma'}
            comp_key = f'{comp_dict[compartment]} length'
            comp_length = full_dict[cellid][comp_key]
            return comp_length
    sso = SuperSegmentationObject(cellid)
    sso.load_skeleton()
    axoness = sso.skeleton["axoness_avg10000"]
    axoness[axoness == 3] = 1
    axoness[axoness == 4] = 1
    non_comp_inds = np.nonzero(axoness != compartment)[0]
    if cell_graph is None:
        cell_graph = sso.weighted_graph()
    comp_graph = cell_graph.copy()
    comp_graph.remove_nodes_from(non_comp_inds)
    comp_length = comp_graph.size(weight="weight") / 1000  # in µm
    return comp_length

def get_compartment_length_mp(comp_input):
    """
            calculates length of compartment in µm per cell using the skeleton if given the networkx graph of the cell.
            :param compartment: 0 = dendrite, 1 = axon, 2 = soma
            :param cell_graph: sso.weighted graph
            :return: comp_len in µm
            """
    cellid, compartment, cell_graph, full_dict = comp_input
    if full_dict is not None:
        if cellid in full_dict.keys():
            comp_dict = {0: 'dendrite', 1: 'axon', 2: 'soma'}
            comp_key = f'{comp_dict[compartment]} length'
            comp_length = full_dict[cellid][comp_key]
            return comp_length
    sso = SuperSegmentationObject(cellid)
    sso.load_skeleton()
    axoness = sso.skeleton["axoness_avg10000"]
    axoness[axoness == 3] = 1
    axoness[axoness == 4] = 1
    non_comp_inds = np.nonzero(axoness != compartment)[0]
    if cell_graph is None:
        cell_graph = sso.weighted_graph()
    comp_graph = cell_graph.copy()
    comp_graph.remove_nodes_from(non_comp_inds)
    comp_length = comp_graph.size(weight="weight") / 1000  # in µm
    return comp_length


def get_spine_density(input):
    """
    calculates the spine density of the dendrite.Therefore, the amount of spines per µm dendrite is calculated.
     Amount of spines is the number of connected_components with spiness = spines.
     # spiness values: 0 = spine neck, 1 = spine head, 2 = dendritic shaft, 3 = other
    :param cell: super-segmentation object
    :param min_comp_len: minimum compartment length in µm
    :param full_cell_dict: dictionary with per cell parameter values, already per cellid
    :return: amount of spines on dendrite, 0 if not having min_comp_len
    """
    cellid, min_comp_len, full_cell_dict = input
    if min_comp_len is None:
        min_comp_len = 100
    cell = SuperSegmentationObject(cellid)
    cell.load_skeleton()
    g = cell.weighted_graph(add_node_attr=('axoness_avg10000',))
    # use axon and dendrite length dictionaries to lookup axon and dendrite lenght in future versions
    if full_cell_dict is not None:
        try:
            axon_length = full_cell_dict["axon length"]
        except KeyError:
            all_cell_dict = load_pkl2obj("wholebrain/scratch/arother/j0251v4_prep/combined_fullcell_ax_dict.pkl")
            axon_length = all_cell_dict["axon length"]
    else:
        axon_length = get_compartment_length(cell, compartment = 1, cell_graph = g)
    if axon_length < min_comp_len:
        return 0
    if full_cell_dict is not None:
        try:
            dendrite_length = full_cell_dict["dendrite length"]
        except KeyError:
            dendrite_length = all_cell_dict["dendrite length"]
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
    subgraphs = (spine_graph.subgraph(c) for c in nx.connected_components(spine_graph))
    spine_amount = len(list(subgraphs))
    spine_density = spine_amount/no_spine_dendrite_length
    return [spine_density, no_spine_dendrite_length]


def get_compartment_radii(cell, comp_inds = None):
    """
    get radii from compartment graph of one cell
    :param comp_inds: indicies of compartment
    :return: comp_radii as array in µm
    """
    if not np.all(comp_inds) is None:
        comp_radii = cell.skeleton["diameters"][comp_inds] * cell.scaling[0] / 2000 #in µm and divided by 2 to get radius
    else:
        comp_radii = cell.skeleton["diameters"] * cell.scaling[0] / 2000  # in µm
    return comp_radii

def get_median_comp_radii_cell(cell_input):
    '''
    Get median radius of axon and if applicable also dendrite for cell.
    :param cell_input: cellid, only_axon if True then only axon median radius will be computed,
        no_spine if True, then spines will be removed before calculating median radius
    :return: axon median radius and dendrite median radius if applicable
    '''

    cellid, only_axon, no_spine = cell_input
    cell = SuperSegmentationObject(cellid)
    cell.load_skeleton()
    axon_inds = np.nonzero(cell.skeleton["axoness_avg10000"] == 1)[0]
    axon_radii_cell = get_compartment_radii(cell, comp_inds=axon_inds)
    ax_median_radius_cell = np.median(axon_radii_cell)
    if only_axon:
        return [ax_median_radius_cell, 0]
    else:
        dendrite_inds = np.nonzero(cell.skeleton["axoness_avg10000"] == 0)[0]
        if no_spine:
            spine_shaftinds = np.nonzero(cell.skeleton["spiness"] == 2)[0]
            spine_otherinds = np.nonzero(cell.skeleton["spiness"] == 3)[0]
            nonspine_inds = np.hstack([spine_shaftinds, spine_otherinds])
            dendrite_inds = dendrite_inds[np.in1d(dendrite_inds, nonspine_inds)]
        dendrite_radii_cell = get_compartment_radii(cell, comp_inds=dendrite_inds)
        den_median_radius_cell = np.median(dendrite_radii_cell)
        return [ax_median_radius_cell, den_median_radius_cell]


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
        sample_diagonal = np.linalg.norm(max - min) / 1000 # in µm
        if sample_diagonal == 0 or sample_diagonal < 10**(-5):
            continue
        sample_tortuosity = (sample_length/ sample_diagonal) ** 2
        if sample_tortuosity > 100:
            continue
        tortuosities[i] = sample_tortuosity
    avg_tortuosity = np.nanmean(tortuosities)

    return avg_tortuosity

def get_myelin_fraction(cell_input):
    """
    calculate length and fraction of myelin for axon. Skeleton has to be loaded
    :param cell:super-segmentation object graph should be calculated on
    :param min_comp_len: compartment lengfh threshold
    :return: absolute length of mylein, relative length of myelin
    """
    cellid, min_comp_len, load_skeleton, axon_only = cell_input
    if min_comp_len is None:
        min_comp_len = 100
    if axon_only is None:
        axon_only = False
    cell = SuperSegmentationObject(cellid)
    if load_skeleton:
        cell.load_skeleton()
    axoness = cell.skeleton["axoness_avg10000"]
    g = cell.weighted_graph(add_node_attr=('axoness_avg10000', "myelin"))
    axon_graph = g.copy()
    if not axon_only:
        axoness[axoness == 3] = 1
        axoness[axoness == 4] = 1
        non_axon_inds = np.nonzero(axoness != 1)[0]
        axon_graph.remove_nodes_from(non_axon_inds)
    axon_length = axon_graph.size(weight="weight") / 1000  # in µm
    if axon_length < min_comp_len:
        return np.nan, np.nan
    myelin_graph = axon_graph.copy()
    non_myelin_inds = np.nonzero(cell.skeleton["myelin"] == 0)[0]
    myelin_graph.remove_nodes_from(non_myelin_inds)
    absolute_myelin_length = myelin_graph.size(weight="weight") / 1000  # in µm
    relative_myelin_length = absolute_myelin_length / axon_length
    return [absolute_myelin_length, relative_myelin_length]

def get_percell_organell_volume_density(input):
    '''
    calculates volume density per cell given numpy arrays per cell.
    :param input: cellid, numpy array with all organell ids, numpy array with all volumes of organell,
    cell_dict is per cell dictionary with presaved values like axon_length, proj_axon (True if DA, HVC, LMAN)
    '''
    cellid, cached_so_ids, cached_so_volume, cell_dict, proj_axon, organelle_key = input
    cell = SuperSegmentationObject(cellid)
    segmentation_object_ids = cell.lookup_in_attribute_dict(organelle_key)
    sso_organell_inds = np.in1d(cached_so_ids, segmentation_object_ids)
    organell_volumes = np.sum(cached_so_volume[sso_organell_inds] * 10 ** (-9) * np.prod(cell.scaling))  # convert to cubic µm
    if cell_dict is None:
        cell_length = get_cell_length(cellid)
    else:
        if proj_axon:
            cell_length = cell_dict['axon length']
        else:
            cell_length = cell_dict["complete pathlength"]
    volume_density = organell_volumes/ cell_length
    return volume_density

def get_percell_organell_area_density(input):
    '''
    calculates surface area density per cell given numpy arrays per cell.
    :param input: cellid, numpy array with all organell ids, numpy array with all volumes of organell,
    cell_dict is per cell dictionary with presaved values like axon_length, proj_axon (True if DA, HVC, LMAN)
    '''
    cellid, cached_so_ids, cached_so_areas, cell_dict, proj_axon, organelle_key = input
    cell = SuperSegmentationObject(cellid)
    segmentation_object_ids = cell.lookup_in_attribute_dict(organelle_key)
    sso_organell_inds = np.in1d(cached_so_ids, segmentation_object_ids)
    organell_areas = np.sum(cached_so_areas[sso_organell_inds])  #in µm²
    if proj_axon:
        cell_surface_area = cell_dict['axon mesh surface area']
    else:
        cell_surface_area = mesh_area_calc(cell.mesh)
    mesh_area_density = organell_areas/ cell_surface_area
    return mesh_area_density

def get_organell_ids_comps(input):
    '''
    Get all organell ids for one cell seperated by compartment
    :param input: cellid, org_so_ids, org_rep_coords
    :return: axon_ids, den_ids, soma_ids
    '''
    cellid, org_so_ids, org_rep_coord, segmentation_object_ids = input
    cell = SuperSegmentationObject(cellid)
    cell.load_skeleton()
    kdtree = scipy.spatial.cKDTree(cell.skeleton["nodes"] * cell.scaling)
    sso_organell_inds = np.in1d(org_so_ids, segmentation_object_ids)
    sso_organell_ids = org_so_ids[sso_organell_inds]
    so_rep_coord = org_rep_coord[sso_organell_inds] * cell.scaling  # in nm
    close_node_ids = kdtree.query(so_rep_coord, k=1)[1].astype(int)
    axo = np.array(cell.skeleton["axoness_avg10000"][close_node_ids])
    axo[axo == 3] = 1
    axo[axo == 4] = 1
    org_soma_ids = sso_organell_ids[axo == 2]
    org_den_ids = sso_organell_ids[axo == 0]
    org_ax_ids = sso_organell_ids[axo == 1]
    comp_dict = {'axon': org_ax_ids, 'dendrite': org_den_ids, 'soma': org_soma_ids}
    return comp_dict

def get_organell_volume_density_comps(input):
    '''
    :param cell: super segmentation object
    :param segmentation_object_ids: organell ids per cell
    :param cached_so_ids: cached ids for organell of all cells
    :param cached_so_rep_coord: cached coordinates for organells of all cells
    :param cached_so_volume: cached organell volume for all cells
    :param full_cell_dict: lookup dictionary for per cell parameters, cell.id is key, if None given will be calculated
    :param k: number of nodes surrounding the organells compartment will be determined from
    :param axon_only: set True if assumption that whole cell is axon (e.g. for DA, lMAN, HVC)
    :return: densities and volume densities for aoxn and dendrite
    '''
    cellid, cached_so_ids, cached_so_rep_coord, cached_so_volume, full_cell_dict,k, min_comp_len, axon_only = input
    cell = SuperSegmentationObject(cellid)
    if full_cell_dict is not None:
        axon_length = full_cell_dict[cellid]["axon length"]
        if not axon_only:
            complete_length = full_cell_dict[cellid]['complete pathlength']
    else:
        g = cell.weighted_graph(add_node_attr=('axoness_avg10000',))
        axon_length = get_compartment_length(cell, compartment=1, cell_graph=g)
        if not axon_only:
            complete_length = get_cell_length(cellid)
    if axon_length < min_comp_len:
        return 0,0, 0, 0
    segmentation_object_ids = cell.mi_ids
    sso_organell_inds = np.in1d(cached_so_ids, segmentation_object_ids)
    organell_volumes = cached_so_volume[sso_organell_inds] * 10 ** (-9) * np.prod(cell.scaling)  # convert to cubic µm
    #if projecting axon no need for kdtree and mapping; all mito is used
    if axon_only:
        axo_so_density = len(cell.mis) / axon_length
        axo_so_volume_density = np.sum(organell_volumes)/ axon_length
        total_volume_density = axo_so_volume_density
        return [axo_so_density, axo_so_volume_density, 0, 0, total_volume_density]
    cell.load_skeleton()
    kdtree = scipy.spatial.cKDTree(cell.skeleton["nodes"]*cell.scaling)
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
    axo_so_density = axo_so_amount / axon_length
    axo_so_volume = np.sum(organell_volumes[axon_inds])
    axo_so_volume_density = axo_so_volume / axon_length
    total_volume_density = np.sum(organell_volumes) / complete_length
    den_so_amount = len(den_so_ids)
    if full_cell_dict is not None:
        dendrite_length = full_cell_dict[cell.id]["dendrite length"]
    else:
        dendrite_length = get_compartment_length(cell, compartment=0, cell_graph=g)
    if dendrite_length < min_comp_len:
        return 0,0, 0, 0
    den_so_density = den_so_amount/dendrite_length
    den_so_volume = np.sum(organell_volumes[den_inds])
    den_so_volume_density = den_so_volume/ dendrite_length
    return [axo_so_density, den_so_density, axo_so_volume_density, den_so_volume_density, total_volume_density]

def get_compartment_mesh_area(cell):
    """
    get compartment mesh areas using compartmentalize_mesh and mesh_area_calc.
    :param cell: sso
    :return: dictionary with mesh_areas of axon, dendrite and soma in µm²
    """
    #set nearest neighbour to one as this is also done to map comparmtent to synapses
    comp_meshes = compartmentalize_mesh_fromskel(cell, k = 1)
    compartments = ["axon", "dendrite", "soma"]
    mesh_areas = {}
    for comp in compartments:
        mesh_areas[comp] = mesh_area_calc(comp_meshes[comp])

    return mesh_areas

def check_comp_lengths_ct(cellids, fullcelldict = None, min_comp_len = 200, axon_only = False, max_path_len = None):
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
            try:
                cell_axon_length = fullcelldict[cellid]["axon length"]
            except KeyError:
                all_cell_dict = load_pkl2obj("cajal/nvmescratch/users/arother/j0251v4_prep/combined_fullcell_ax_dict.pkl")
                cell_axon_length = all_cell_dict[cellid]["axon length"]
            if cell_axon_length < min_comp_len:
                continue
            if not axon_only:
                try:
                    cell_den_length = fullcelldict[cellid]["dendrite length"]
                except KeyError:
                    cell_den_length = all_cell_dict[cellid]["dendrite length"]
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
        if max_path_len is not None:
            if axon_only == False:
                if fullcelldict is not None:
                    try:
                        full_path_length = fullcelldict[cellid]["complete pathlength"]
                    except KeyError:
                        cell = SuperSegmentationObject(cellid)
                        cell.load_skeleton()
                        g = cell.weighted_graph(add_node_attr=('axoness_avg10000',))
                        full_path_length = g.size(weight="weight") / 1000  # in µm
                    if full_path_length > max_path_len:
                        continue
            else:
                raise ValueError("max_path_length can only be set for full cells")
        checked_cells[i] = cellid

    checked_cells = checked_cells[checked_cells > 0].astype(int)
    return checked_cells

def get_compartment_nodes(ssoid, compartment):
    """
            calculates length of compartment in µm per cell using the skeleton if given the networkx graph of the cell.
            :param compartment: 0 = dendrite, 1 = axon, 2 = soma
            :param cell_graph: sso.weighted graph
            :return: comp_nodes with coordinates in pyhsical space (nm)
            """
    sso = SuperSegmentationObject(ssoid)
    sso.load_skeleton()
    comp_inds = np.nonzero(sso.skeleton["axoness_avg10000"] == compartment)[0]
    comp_nodes = sso.skeleton["nodes"][comp_inds] * sso.scaling
    return comp_nodes

def get_cell_nodes_ax(ssoid):
    """
            calculates length of compartment in µm per cell using the skeleton if given the networkx graph of the cell.
            :param compartment: 0 = dendrite, 1 = axon, 2 = soma
            :param cell_graph: sso.weighted graph
            :return: comp_len in µm
            """
    sso = SuperSegmentationObject(ssoid)
    sso.load_skeleton()
    cell_nodes = sso.skeleton["nodes"] * sso.scaling
    axo = np.array(sso.skeleton["axoness_avg10000"])
    axo[axo == 3] = 1
    axo[axo == 4] = 1
    return [cell_nodes, axo]

def remove_myelinated_part_axon(axonid):
    """
    removes part of axon which is myleniated and unbranched to get terminal branching part.
    Uses skeleton nodes and removes myelinated ones.
    :param axonid: id of axon
    :return: skeleton node positions of branched axon part
    """
    axon = SuperSegmentationObject(axonid)
    axon.load_skeleton()
    myelin_inds = np.nonzero(axon.skeleton["myelin"] == 1)[0]
    g = axon.weighted_graph()
    g.remove_nodes_from(myelin_inds)
    node_positions = [g.nodes[node]["position"] for node in g.nodes()]
    node_positions = np.array(node_positions) * axon.scaling
    return node_positions

def compute_overlap_skeleton(sso_id, sso_ids, all_node_positions, kdtree_radius = 10):
    '''
    Compare the overlap between skeelton nodes of one cell and a list of other cells.
    Uses kdrtee with given radius in µm. Returns percentage of overlap for one cell with other cells.
    :param sso_id: id of cell that other cells should be compared to
    :param sso_ids: ids the voerlap should be computed for, includes the sso_id
    :param all_node_positions: skeleton nodes of all ssoids, including sso_id given in first argument, in nm
    :param kdtree_radius: radius around which overlap will be computed for each skeleton node, in µm
    :return: overlapp between sso_id and other sso_ids
    '''

    sso_ind = int(np.where(sso_ids == sso_id)[0])
    sso_nodes = all_node_positions[sso_ind]
    overlap = np.zeros(len(sso_ids))
    kdtree_radius = kdtree_radius * 1000 #in nm
    #set overlap with itself to 1
    overlap[sso_ind] = 1
    #iterate over sso ids to get overlap of each of them
    for i, id in enumerate(sso_ids):
        if id == sso_id:
            continue
        nodes_2_compare = all_node_positions[i]
        kdtree = scipy.spatial.cKDTree(nodes_2_compare)
        overlapping_inds = kdtree.query_ball_point(sso_nodes, kdtree_radius)
        unique_overlapping_nodes = np.unique(np.hstack(overlapping_inds))
        overlap_cell = len(unique_overlapping_nodes)/ len(nodes_2_compare)
        overlap[i] = overlap_cell
    return overlap

def get_cell_close_surface_area(cell_input):
    '''
    To a given set of coordinates, see if the cell is in a given radius.
    if so, count the number of coordinates the cell is close to and calculate the summed surface area of the cell
    which is close to the coordinates.
    :param cell_input: cellid, coordinates, radius to coordinate
    :return: number of coordinates close, summed area close, cell surface area
    '''
    cellid, des_coords, radius = cell_input
    cell = SuperSegmentationObject(cellid)
    cell_mesh = cell.mesh
    indices, vertices, normals = cell_mesh
    # get cell surface mesh area in µm²
    cell_surface_area = mesh_area_calc(cell_mesh)
    #get coordinates close to cell surface
    vertices = vertices.reshape((-1, 3))
    coord_kdtree = cKDTree(des_coords)
    vert_tree = cKDTree(vertices)
    coord_inds = coord_kdtree.query_ball_tree(vert_tree, r = radius * 1000)
    con_coord_inds = np.concatenate(coord_inds)
    if len(con_coord_inds) == 0:
        return [0, 0, cell_surface_area]
    else:
        #get number of vesicles which is close to cell
        non_empty_coord_inds = [coord_ind for coord_ind in coord_inds if len(coord_ind) > 0]
        number_close_vesicles = len(non_empty_coord_inds)
        #calculate surface area close to vesicles
        unique_vertice_inds = np.unique(con_coord_inds).astype(int)
        #similar code to compartmentalize_mesh_fromskel
        #mask vertices
        vertex_mask = np.zeros(len(vertices))
        vertex_mask[unique_vertice_inds] = 1
        #mask indices
        ind_comp = vertex_mask[indices]
        indices = indices.reshape(-1, 3)
        ind_comp = ind_comp.reshape(-1, 3)
        ind_comp_maj = np.zeros((len(indices)), dtype=np.uint8)
        for ii in range(len(indices)):
            triangle = ind_comp[ii]
            cnt = Counter(triangle)
            ax, n = cnt.most_common(1)[0]
            ind_comp_maj[ii] = ax
        comp_ind = indices[ind_comp_maj == 1].flatten()
        unique_comp_ind = np.unique(comp_ind)
        remap_dict = {}
        for i in range(len(unique_comp_ind)):
            remap_dict[unique_comp_ind[i]] = i
        if len(normals) > 0:
            normals = normals.reshape(-1, 3)[unique_vertice_inds]
        comp_ind = np.array([remap_dict[i] for i in comp_ind], dtype=np.uint)
        close_vertices = vertices[unique_comp_ind].flatten()
        summed_surface_area = mesh_area_calc([comp_ind, close_vertices, normals])
        return [number_close_vesicles, summed_surface_area, cell_surface_area]


def generate_colored_mesh_from_skel_data(args):
    '''
    Generates mesh coloured according to axoness_avg10000 prediction of skeleton.
    Based partly on syconn2scripts.scripts.point_party.semseg_gt. Saves result as kzip
    :param args: cellid, path to folder where kzip should be stored, key to color
    :return:
    '''

    # color lookup from HA
    cellid, f_name, key, only_coarse, k, ct_dict = args
    if only_coarse:
        col_lookup = {0: (50, 135, 168, 255), 1: (232, 170, 71, 255), 2: (189, 55, 72, 255)}
    else:
        col_lookup = {0: (76, 92, 158, 255), 1: (255, 125, 125, 255), 2: (125, 255, 125, 255),
                      3: (113, 98, 227, 255),
                      4: (255, 255, 125, 255)}
    cell = SuperSegmentationObject(cellid)
    cell.load_skeleton()
    if ct_dict is not None:
        cell.load_attr_dict()
        ct_num = cell.attr_dict['celltype_pts_e3']
        celltype = ct_dict[ct_num]
    # load skeleton axoness, spiness attributes
    nodes = cell.skeleton['nodes'] * cell.scaling
    axoness_labels = cell.skeleton[key]
    if only_coarse:
        axoness_labels[axoness_labels == 3] = 1
        axoness_labels[axoness_labels == 4] = 1
    # load mesh and put skeleton annotations on mesh
    indices, vertices, normals = cell.mesh
    vertices = vertices.reshape((-1, 3))
    kdt = cKDTree(nodes)
    dists, node_inds = kdt.query(vertices, k = k)
    vert_axoness_labels = axoness_labels[node_inds]

    # save colored mesh
    cols = np.array([col_lookup[el] for el in vert_axoness_labels.squeeze()], dtype=np.uint8)
    if ct_dict is not None:
        kzip_out = f'{f_name}/{cellid}_{celltype}_colored_mesh_{key}'
    else:
        kzip_out = f'{f_name}/{cellid}_colored_mesh_{key}'
    kzip_out_skel = f'{f_name}/{cellid}_skel'
    write_mesh2kzip(kzip_out, indices.astype(np.float32), vertices.astype(np.float32), None, cols,
                    f'{cellid}.ply')
    cell.save_skeleton_to_kzip(kzip_out_skel, additional_keys=[key])
    return

def generate_colored_mesh_from_vert_labels(args):
    '''
    Generates mesh coloured according to vertex_label prediction of skeleton.
    Based partly on syconn2scripts.scripts.point_party.semseg_gt. Saves result as kzip
    :param args: cellid, path to folder where kzip should be stored, key to color
    :return:
    '''
    cellid, f_name, smooth, smooth_all = args
    #hexcode colors: 0 = #3287A8, 1 = #E8AA47, 2 = #BD3748,5 = #707070 (5 = unpredicted)
    col_lookup = {0: (50, 135, 168, 255), 1: (232, 170, 71, 255), 2: (189, 55, 72, 255), 5:(112, 112, 112, 255)}
    cell = SuperSegmentationObject(cellid)
    cell_ld = cell.label_dict('vertex')
    vert_axoness_labels = cell_ld['axoness']
    vert_axoness_labels[vert_axoness_labels == 3] = 1
    vert_axoness_labels[vert_axoness_labels == 4] = 1
    # load mesh and put skeleton annotations on mesh
    indices, vertices, normals = cell.mesh
    vertices = vertices.reshape((-1, 3))
    if smooth is not None:
        sf = np.round(smooth / 2)
        if smooth_all:
            for ind in range(len(vert_axoness_labels)):
                if ind - sf < 0:
                    ind_lower = 0
                else:
                    ind_lower = int(ind - sf)
                if ind + sf > len(vert_axoness_labels):
                    ind_upper = len(vert_axoness_labels)
                else:
                    ind_upper = int(ind + sf)
                axoness_ind = vert_axoness_labels[ind_lower: ind_upper]
                axoness_ind = axoness_ind[axoness_ind != 5]
                new_axoness = np.argmax(np.bincount(axoness_ind))
                vert_axoness_labels[ind] = new_axoness
        else:
            if 5 in vert_axoness_labels:
                #iteratively replace each label that is 5 with the majority before and after
                while 5 in vert_axoness_labels:
                    unpred_inds = np.where(vert_axoness_labels == 5)[0]
                    for ind in unpred_inds:
                        if ind - sf < 0:
                            ind_lower = 0
                        else:
                            ind_lower = int(ind - sf)
                        if ind + sf > len(vert_axoness_labels):
                            ind_upper = len(vert_axoness_labels)
                        else:
                            ind_upper = int(ind + sf)
                        axoness_ind = vert_axoness_labels[ind_lower: ind_upper]
                        axoness_ind = axoness_ind[axoness_ind != 5]
                        new_axoness = np.argmax(np.bincount(axoness_ind))
                        vert_axoness_labels[ind] = new_axoness
    # save colored mesh
    cols = np.array([col_lookup[el] for el in vert_axoness_labels.squeeze()], dtype=np.uint8)
    if smooth is None:
        kzip_out = f'{f_name}/{cellid}_colored_mesh_vert'
    else:
        if smooth_all:
            kzip_out = f'{f_name}/{cellid}_colored_mesh_vert_sm_all{smooth}'
        else:
            kzip_out = f'{f_name}/{cellid}_colored_mesh_vert_sm_unpred{smooth}'
    kzip_out_skel = f'{f_name}/{cellid}_skel'
    write_mesh2kzip(kzip_out, indices.astype(np.float32), vertices.astype(np.float32), None, cols,
                    f'{cellid}.ply')
    cell.save_skeleton_to_kzip(kzip_out_skel)
    return

def generate_colored_mesh_synprob_data(cell_input):
    '''
    Generates mesh coloured according to synapse_probability values for cells of synapses its part of.
    Does use all synapses cell is a part of, if synapses should be filtered this needs to be done before applying this function.
    Based partly on syconn2scripts.scripts.point_party.semseg_gt, similar to generate_colored_mesh_from_skel_data. Saves result as kzip
    :param args: cellid, path to folder where kzip should be stored, syn_ssv_partners, synapse coords, synapse probability, color lookup
    dictionary (categories to sort probability in should be keys)
    :return:
    '''
    cellid, f_name, syn_ssv_partners, syn_rep_coords, syn_prob, col_lookup = cell_input
    #filter synapses for cellid
    cell = SuperSegmentationObject(cellid)
    cell.load_attr_dict()
    ct_num = cell.attr_dict['celltype_pts_e3']
    cell_inds = np.where(syn_ssv_partners == cellid)[0]
    cell_syn_coords = syn_rep_coords[cell_inds]
    cell_syn_prob = syn_prob[cell_inds]
    #make kdTree from synapse_coords and map synprob_labels to vertices
    kdt = cKDTree(cell_syn_coords * cell.scaling)
    indices, vertices, normals = cell.mesh
    vertices = vertices.reshape((-1, 3))
    dists, node_inds = kdt.query(vertices)
    vert_synprob_labels = cell_syn_prob[node_inds]
    #caregorize labels to match color according to synapse probability
    #categories are in these case assumed to be lower bounds e.g. 0.0, 0.2, 0.4, 0.6, 0.8
    #this means 0.5 would be in category 0.4
    cats = list(col_lookup.keys())
    labels = list(col_lookup.keys())
    if 1.0 not in cats:
        cats.append(1.0)

    vert_synprob_labels_cats = np.array(pd.cut(vert_synprob_labels, cats, right = False, labels = labels))
    #set all nan values to -1, and add new color for meshes not mapped to any synprob
    if np.any(np.isnan(vert_synprob_labels_cats)):
        vert_synprob_labels_cats[np.isnan(vert_synprob_labels_cats)] = -1.0
        nan_col = '#707070'
        nan_col_rgba_int = co.to_rgba_array(nan_col)[0] * 255
        col_lookup[-1.0] = nan_col_rgba_int
    # save colored mesh
    cols = np.array([col_lookup[el] for el in vert_synprob_labels_cats], dtype=np.uint8)
    kzip_out = f'{f_name}/{cellid}_{ct_num}_colored_mesh'
    kzip_out_skel = f'{f_name}/{cellid}_{ct_num}_skel'
    write_mesh2kzip(kzip_out, indices.astype(np.float32), vertices.astype(np.float32), None, cols,
                    f'{cellid}_{ct_num}.ply')
    cell.save_skeleton_to_kzip(kzip_out_skel)
    return

def get_per_cell_mito_myelin_info(input):
    '''
    Function to get information about myelin fraction, axon mitochondria density and axon radius per cell.
    Also calculates the cell volume from cell.size in µm³
    :param input: cellid, cached mitochondria information, cell dict which cached cellids for cell
    :return: median radius per cell, mitochondria volume density, myelin fraction
    '''
    cellid, min_comp_len, mi_ssv_ids, mi_sizes, mi_axoness, full_cell_dict = input
    cell = SuperSegmentationObject(cellid)
    cell.load_skeleton()
    #cellid, min_comp_len, load_skeleton, axon_only
    cell_input = [cellid, min_comp_len, True, False]
    myelin_results = get_myelin_fraction(cell_input)
    abs_myelin_cell = myelin_results[0]
    rel_myelin_cell = myelin_results[1]
    if np.isnan(abs_myelin_cell):
        return [np.nan, np.nan, np.nan]
    #input for organell analysis: cellid, org_ssv_ids, org_sizes, org_axoness, full_cell_dict = params
    organell_input = [cellid, mi_ssv_ids, mi_sizes, mi_axoness, full_cell_dict, 1]
    mito_results = get_organelle_comp_density_presaved(organell_input)
    axo_mito_volume_density_cell = mito_results[0]
    total_mito_volume_density_cell = mito_results[1]
    axon_inds = np.nonzero(cell.skeleton["axoness_avg10000"] == 1)[0]
    axon_radii_cell = get_compartment_radii(cell, comp_inds=axon_inds)
    ax_median_radius_cell = np.median(axon_radii_cell)
    cell_volume = np.abs(cell.size) * np.prod(cell.scaling) * 10**(-9) #in µm³
    return [ax_median_radius_cell, axo_mito_volume_density_cell, rel_myelin_cell, cell_volume, total_mito_volume_density_cell]

def get_cell_comp_vert_coords(cellid, comp):
    '''
    Gets vertex coordinates from a mesh of a specific compartment based on
    the cells label dict.
    :param cellid: id of the cell
    :param comp: compartment to be selected for
    :return: vertex coordinates in physical space
    '''
    cell = SuperSegmentationObject(cellid)
    cell_ld = cell.label_dict('vertex')
    cell_ld_axoness = cell_ld['axoness']
    if comp == 1:
        #set en-passant bouton and terminal boutons to axon
        cell_ld_axoness[cell_ld_axoness == 3] = 1
        cell_ld_axoness[cell_ld_axoness == 4] = 1
    ind, vert, norm = cell.mesh
    vert_coords = vert.reshape((-1, 3))
    vert_comp_coords = vert_coords[cell_ld_axoness == comp]
    return vert_comp_coords


def get_cell_soma_radius(cellid, use_skel = False, use_median_centre = True):
    '''
    Gives an estimate about the soma radius. Calculates radius as median distance from soma
    centre to mesh vertices. Soma centre is calculated as average of the vertex coordiantes.
    Default is getting the vertex coordinates of soma directly from the vertex label dict via
    get_cell_comp_vert_coords. Can use the skeleton 'axoness_avg_10000' key to map the skeleton
    to the mesh with syconn.mesh.compartmentalize_mesh_fromskel but it less exact.
    :param cellid: id of cell
    :param use_skel: if True uses compartmentalize_mesh_fromskel to get soma mesh.
                    If soma coordinates far away from skeleton nodes it might not find them
                    If false uses get_cell_comp_vert_coords which uses the label dict of the vertices directly
    :param use_median_centre: if True uses the median to find the soma centre, otherwise average
    :return: soma center in physical coordinates, radius in nm
    '''

    if use_skel:
        cell = SuperSegmentationObject(cellid)
        cell.load_skeleton()
        cell_comp_meshes = compartmentalize_mesh_fromskel(cell, 'axoness_avg10000')
        soma_mesh = cell_comp_meshes['soma']
        ind, vert, norm = soma_mesh
        soma_vert_coords = vert.reshape((-1, 3))
    else:
        soma_vert_coords = get_cell_comp_vert_coords(cellid, comp=2)
    if len(soma_vert_coords) == 0:
        return [[np.nan, np.nan, np.nan], np.nan]
    if use_median_centre:
        soma_vert_avg = np.median(soma_vert_coords, axis=0)
    else:
        soma_vert_avg = np.mean(soma_vert_coords, axis=0)
    dist2centre = np.linalg.norm(soma_vert_coords - soma_vert_avg, axis = 1)
    radius = np.median(dist2centre) / 1000
    return [soma_vert_avg, radius]

def get_dendrite_info_cell(input):
    '''
    Get information about the dendrtíes of a cell with a specified total minimum length.
    Calculates total dendritic length without spines, number of primary dendrites.
    Number of branching points needs to be interpretet carefully as it needs a manual
    threshold. With a threshold too small there are spines included which should be
    avoided but with a threshold too high real branching points are missed out.
    The threshold used will always be a compromise between the two, giving as many real
    branching points but trying to avoid most spines or other branches.
    :param input: combination of the three parameters below
    :param cellid: id of cell
    :param min_comp_len: minimum total length of dendrite
    :return: total dendritic length, number of primary dendrites
    '''
    cellid, min_comp_len = input
    if min_comp_len is None:
        min_comp_len = 0
    cell = SuperSegmentationObject(cellid)
    cell.load_skeleton()
    g = cell.weighted_graph(add_node_attr=('axoness_avg10000',))
    #code from get_compartment_length
    #get dendrite compartments
    axoness = cell.skeleton["axoness_avg10000"]
    axoness[axoness == 3] = 1
    axoness[axoness == 4] = 1
    non_comp_inds = np.nonzero(axoness != 0)[0]
    dendrite_graph = g.copy()
    dendrite_graph.remove_nodes_from(non_comp_inds)
    #remove spines
    spine_head_inds = np.nonzero(cell.skeleton["spiness"] == 1)[0]
    spine_neck_inds = np.nonzero(cell.skeleton["spiness"] == 0)[0]
    spine_inds = np.hstack([spine_head_inds, spine_neck_inds])
    dendrite_no_spine_graph = dendrite_graph.copy()
    dendrite_no_spine_graph.remove_nodes_from(spine_inds)
    dendrite_length = dendrite_no_spine_graph.size(weight="weight") / 1000  # in µm
    if dendrite_length < min_comp_len:
        return [np.nan, np.nan, np.nan]
    #get number of primary dendrites
    dendrite_subgraphs = list(dendrite_graph.subgraph(c) for c in nx.connected_components(dendrite_graph))
    primary_dendrite_number = len(dendrite_subgraphs)
    #get number of branching points
    degrees = np.zeros(len(g))
    for ix, node_id in enumerate(g.nodes):
        degrees[ix] = g.degree[node_id]
    pot_branching_points = np.where(degrees > 2)[0]
    branching_points = []
    for indiv_dendrite in dendrite_subgraphs:
        sub_branching_points = pot_branching_points[np.in1d(pot_branching_points, indiv_dendrite.nodes)]
        for b in sub_branching_points:
            test_sub = indiv_dendrite.copy()
            test_sub.remove_node(b)
            sub_subs = list(test_sub.subgraph(c) for c in nx.connected_components(test_sub))
            if len(sub_subs) >= 3:
                lenghts = [sub.size(weight = 'weight') / 1000 for sub in sub_subs]
                if min(lenghts) > 8:
                    branching_points.append(b)
    number_branching_points = len(branching_points)
    return dendrite_length, primary_dendrite_number, number_branching_points

def map_axoness_cellid2org(params):
    '''
    Map axoness (axon, dendrite, soma) of a cellid to the organell associated with it;
    also map cellid and celltype.
    :param input: cellid, org_ids, org_coords
    :return: org_ids and axoness per cell
    '''

    cellid, org_ids, org_coords, organelle_key = params
    cell = SuperSegmentationObject(cellid)
    cell.load_skeleton()
    cell_mi_ids = cell.lookup_in_attribute_dict(organelle_key)
    cell_org_ind = np.in1d(org_ids, cell_mi_ids)
    cell_org_coords = org_coords[cell_org_ind]
    cell_org_ids = org_ids[cell_org_ind]
    kdtree = scipy.spatial.cKDTree(cell.skeleton["nodes"] * cell.scaling)
    close_node_ids = kdtree.query(cell_org_coords * cell.scaling, k=1)[1].astype(int)
    axo = np.array(cell.skeleton["axoness_avg10000"][close_node_ids])
    axo[axo == 3] = 1
    axo[axo == 4] = 1
    org_ssv_ids = np.zeros(len(axo), dtype=np.int64) + cellid
    return [cell_org_ids, axo, org_ssv_ids]

def map_cellid2org(params):
    '''
    Map cellid and celltype to the organell associated with it;
    For full cells use 'map_axoness_cellid2org' to also map comparetment directly;
    Use for projecting axon celltypes instead of 'map_axoness_cellid2org'
    :param input: cellid, org_ids (need both to have in same order as other values e.g. size which is not used here)
    :return: org_ids and axoness per cell
    '''

    cellid, org_ids, organelle_key = params
    cell = SuperSegmentationObject(cellid)
    cell_mi_ids = cell.lookup_in_attribute_dict(organelle_key)
    cell_org_ind = np.in1d(org_ids, cell_mi_ids)
    cell_org_ids = org_ids[cell_org_ind]
    org_ssv_ids = np.zeros(len(cell_org_ids), dtype=np.int64) + cellid
    return [cell_org_ids, org_ssv_ids]

def get_mito_density_presaved(params):
    '''
    Get mito density from presaved mito arrays with cellid. If proj_axon = True,
    uses 'axon_length' and not total length for volume density
    :param params: cellid, mi_ids, mi_ssv_ids, mi_sizes, full_cell_dict (per cell), proj_axon
    :return: mito_volume_density
    '''

    scaling = [10, 10, 25]
    cellid, mi_ssv_ids, mi_sizes, full_cell_dict, proj_axon = params
    cell_mi_inds = np.in1d(mi_ssv_ids, cellid)
    cell_mi_sizes = mi_sizes[cell_mi_inds]
    if proj_axon:
        length = full_cell_dict['axon length']
    else:
        length = full_cell_dict["complete pathlength"]
    #convert to µm³
    cell_mi_volume = np.sum(cell_mi_sizes) *10 ** (-9) * np.prod(scaling)
    mito_volume_density = cell_mi_volume/ length
    return mito_volume_density

def get_org_density_volume_presaved(params):
    '''
    Get organell density from presaved mito arrays with cellid.
    if organelle is ER, then cellid is er is and mapping not necessary.
    In this case, the org_ssv_ids are none
    :param params: cellid, org_ids, org_sizes, organelle name
    :return: mito_volume_density
    '''

    scaling = [10, 10, 25]
    cellid, org_ids, org_sizes, org = params
    if org == 'er':
        cell_org_sizes = org_sizes[org_ids == cellid]
    else:
        cell_org_inds = np.in1d(org_ids, cellid)
        cell_org_sizes = org_sizes[cell_org_inds]
    cell = SuperSegmentationObject(cellid)
    cell_volume = np.abs(cell.size)*10 ** (-9) * np.prod(scaling)
    #convert to µm³
    cell_org_volume = np.sum(cell_org_sizes) *10 ** (-9) * np.prod(scaling)
    cell_org_volume_density = cell_org_volume / cell_volume
    return cell_org_volume_density

def get_mito_comp_density_presaved(params):
    '''
        Get mito density from presaved mito arrays with cellid for each compartment and the complete cell.
        :param params: cellid, mi_ids, mi_ssv_ids, mi_sizes, mi_axoness, full_cell_dict (per cell)
        :return: mito_volume_density for axon, dendrite and full cell
        '''
    scaling = [10, 10, 25]
    cellid, mi_ssv_ids, mi_sizes, mi_axoness, full_cell_dict = params
    cell_mi_inds = np.in1d(mi_ssv_ids, cellid)
    cell_mi_sizes = mi_sizes[cell_mi_inds]
    cell_mi_axoness = mi_axoness[cell_mi_inds]
    full_length = full_cell_dict["complete pathlength"]
    #convert t0 µm³
    cell_mi_volume = np.sum(cell_mi_sizes) * 10 ** (-9) * np.prod(scaling)
    full_mito_volume_density = cell_mi_volume / full_length
    axon_mito_sizes = cell_mi_sizes[cell_mi_axoness == 1]
    den_mito_sizes = cell_mi_sizes[cell_mi_axoness == 0]
    axon_mi_volume = np.sum(axon_mito_sizes) * 10 ** (-9) * np.prod(scaling)
    den_mi_volume = np.sum(den_mito_sizes) * 10 ** (-9) * np.prod(scaling)
    axon_mito_volume_density = axon_mi_volume / full_cell_dict['axon length']
    den_mito_volume_density = den_mi_volume/ full_cell_dict['dendrite length']
    return [axon_mito_volume_density, den_mito_volume_density, full_mito_volume_density]

def get_organelle_comp_density_presaved(params):
    '''
        Get organelle density from presaved mito arrays with cellid for each compartment and the complete cell.
        :param params: cellid, org_ids, org_ssv_ids, org_sizes, org_axoness, full_cell_dict (per cell)
        :return: org_volume_density for axon, dendrite and full cell
        '''
    scaling = [10, 10, 25]
    comp_dict = {0:'dendrite', 1:'axon', 2:'soma'}
    cellid, org_ssv_ids, org_sizes, org_axoness, full_cell_dict, compartment = params
    cell_org_inds = np.in1d(org_ssv_ids, cellid)
    cell_org_sizes = org_sizes[cell_org_inds]
    cell_org_axoness = org_axoness[cell_org_inds]
    full_length = full_cell_dict["complete pathlength"]
    #convert t0 µm³
    cell_org_volume = np.sum(cell_org_sizes) * 10 ** (-9) * np.prod(scaling)
    full_org_volume_density = cell_org_volume / full_length
    comp_org_sizes = cell_org_sizes[cell_org_axoness == compartment]
    comp_org_volume = np.sum(comp_org_sizes) * 10 ** (-9) * np.prod(scaling)
    if compartment == 2:
        soma_radius = full_cell_dict['soma radius']
        soma_volume = (4/3) * np.pi * soma_radius**3
        comp_volume_density = comp_org_volume / soma_volume
    else:
        comp_length = full_cell_dict[f'{comp_dict[compartment]} length']
        comp_volume_density = comp_org_volume / comp_length
    return [comp_volume_density, full_org_volume_density]

def get_organelle_comp_area_density_presaved(params):
    '''
        Get organelle density from presaved mito arrays with cellid for each compartment and the complete cell.
        :param params: cellid, org_ids, org_ssv_ids, org_sizes, org_axoness, full_cell_dict (per cell)
        :return: org_volume_density for axon, dendrite and full cell
        '''
    comp_dict = {0:'dendrite', 1:'axon', 2:'soma'}
    cellid, org_ssv_ids, org_mesh_areas, org_axoness, full_cell_dict, compartment = params
    cell_org_inds = np.in1d(org_ssv_ids, cellid)
    cell_org_mesh_areas = org_mesh_areas[cell_org_inds]
    cell_org_axoness = org_axoness[cell_org_inds]
    comp_org_mesh_areas = cell_org_mesh_areas[cell_org_axoness == compartment]
    comp_area_density = np.sum(comp_org_mesh_areas) /  full_cell_dict[f'{comp_dict[compartment]} mesh surface area']
    return comp_area_density

def check_cutoff_dendrites(cell_input):
    '''
    Function that loads cell and checks if any of their dendrites is within a certain radius away from the dataset borders.
    As the raw data is often not perfectly cubed resulting in blank space (for j0251 ca 5 µm) at the sides and segmentation also sometimes does not reach the borders,
    a threshold of 7 µm is recommended.
    :param cell_input: cellid and radius to determine distance to dataset border (in nm), dataset borders as 2D array, can be called
    extracted from config with: np.array(global_params.config.entries['cube_of_interest_bb'] * scaling)
    :return: cellid if cell has no cutoff dendrites
    '''

    cellid, dist_thresh, dataset_borders = cell_input
    #get dendritic skeleton nodes
    dendrite_nodes = get_compartment_nodes(cellid, compartment=0)
    dataset_borders_thresh = np.array([dataset_borders[0] + dist_thresh, dataset_borders[1] - dist_thresh])
    lower_bounds = np.any(dendrite_nodes < dataset_borders_thresh[0])
    upper_bounds = np.any(dendrite_nodes > dataset_borders_thresh[1])
    #return True if cell is not outside borders
    if np.any([lower_bounds, upper_bounds]):
        return [cellid, False]
    else:
        return [cellid, True]

















