
import numpy as np
import networkx as nx
import scipy
from tqdm import tqdm
from syconn.proc.meshes import mesh_area_calc, compartmentalize_mesh_fromskel
from syconn.reps.super_segmentation import SuperSegmentationObject
from syconn.handler.basics import load_pkl2obj, write_obj2pkl


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
        try:
            axon_length = full_cell_dict[cell.id]["axon length"]
        except KeyError:
            all_cell_dict = load_pkl2obj("wholebrain/scratch/arother/j0251v4_prep/combined_fullcell_ax_dict.pkl")
            axon_length = all_cell_dict[cell.id]["axon length"]
    else:
        axon_length = get_compartment_length(cell, compartment = 1, cell_graph = g)
    if axon_length < min_comp_len:
        return 0
    if full_cell_dict is not None:
        try:
            dendrite_length = full_cell_dict[cell.id]["dendrite length"]
        except KeyError:
            dendrite_length = all_cell_dict[cell.id]["dendrite length"]
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
        sample_diagonal = np.linalg.norm(max - min) / 1000 # in µm
        if sample_diagonal == 0 or sample_diagonal < 10**(-5):
            continue
        sample_tortuosity = (sample_length/ sample_diagonal) ** 2
        if sample_tortuosity > 100:
            continue
        tortuosities[i] = sample_tortuosity
    avg_tortuosity = np.nanmean(tortuosities)

    return avg_tortuosity

def get_myelin_fraction(cellid, cell = None, min_comp_len = 100, load_skeleton = False):
    """
    calculate length and fraction of myelin for axon. Skeleton has to be loaded
    :param cell:super-segmentation object graph should be calculated on
    :param min_comp_len: compartment lengfh threshold
    :return: absolute length of mylein, relative length of myelin
    """
    if cell is None:
        cell = SuperSegmentationObject(cellid)
    if load_skeleton:
        cell.load_skeleton()
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
    return [absolute_myelin_length, relative_myelin_length]

def get_organell_volume_density(cellid, cached_so_ids,cached_so_rep_coord, cached_so_volume, cell = None, full_cell_dict = None,skeleton_loaded = False, k = 3, min_comp_len = 100):
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
    if cell is None:
        cell = SuperSegmentationObject(cellid)
    segmentation_object_ids = cell.mi_ids
    if skeleton_loaded == False:
        cell.load_skeleton()
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
    return np.array([axo_so_density, den_so_density, axo_so_volume_density, den_so_volume_density])

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
            try:
                cell_axon_length = fullcelldict[cellid]["axon length"]
            except KeyError:
                all_cell_dict = load_pkl2obj("wholebrain/scratch/arother/j0251v4_prep/combined_fullcell_ax_dict.pkl")
                cell_axon_length = all_cell_dict[cellid]["axon length"]
            if cell_axon_length < min_comp_len:
                continue
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
        checked_cells[i] = cellid

    checked_cells = checked_cells[checked_cells > 0].astype(int)
    return checked_cells

def get_compartment_nodes(ssoid, compartment):
    """
            calculates length of compartment in µm per cell using the skeleton if given the networkx graph of the cell.
            :param compartment: 0 = dendrite, 1 = axon, 2 = soma
            :param cell_graph: sso.weighted graph
            :return: comp_len in µm
            """
    sso = SuperSegmentationObject(ssoid)
    sso.load_skeleton()
    comp_inds = np.nonzero(sso.skeleton["axoness_avg10000"] == compartment)[0]
    comp_nodes = sso.skeleton["nodes"][comp_inds] * sso.scaling
    return comp_nodes





