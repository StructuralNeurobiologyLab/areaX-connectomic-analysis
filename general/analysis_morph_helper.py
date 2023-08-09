
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


def get_compartment_length(sso, compartment, cell_graph):
    """
            calculates length of compartment in µm per cell using the skeleton if given the networkx graph of the cell.
            :param compartment: 0 = dendrite, 1 = axon, 2 = soma
            :param cell_graph: sso.weighted graph
            :return: comp_len in µm
            """
    axoness = sso.skeleton["axoness_avg10000"]
    axoness[axoness == 3] = 1
    axoness[axoness == 4] = 1
    non_comp_inds = np.nonzero(axoness != compartment)[0]
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
    subgraphs = (spine_graph.subgraph(c) for c in nx.connected_components(spine_graph))
    spine_amount = len(list(subgraphs))
    spine_density = spine_amount/no_spine_dendrite_length
    return spine_density

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
    axoness = cell.skeleton["axoness_avg10000"]
    axoness[axoness == 3] = 1
    axoness[axoness == 4] = 1
    non_axon_inds = np.nonzero(axoness != 1)[0]
    non_myelin_inds = np.nonzero(cell.skeleton["myelin"] == 0)[0]
    g = cell.weighted_graph(add_node_attr=('axoness_avg10000', "myelin"))
    axon_graph = g.copy()
    axon_graph.remove_nodes_from(non_axon_inds)
    axon_length = axon_graph.size(weight="weight") / 1000  # in µm
    if axon_length < min_comp_len:
        return np.nan, np.nan
    myelin_graph = axon_graph.copy()
    myelin_graph.remove_nodes_from(non_myelin_inds)
    absolute_myelin_length = myelin_graph.size(weight="weight") / 1000  # in µm
    relative_myelin_length = absolute_myelin_length / axon_length
    return [absolute_myelin_length, relative_myelin_length]

def get_organell_volume_density(input):
    '''
    :param cell: super segmentation object
    :param segmentation_object_ids: organell ids per cell
    :param cached_so_ids: cached ids for organell of all cells
    :param cached_so_rep_coord: cached coordinates for organells of all cells
    :param cached_so_volume: cached organell volume for all cells
    :param full_cell_dict: lookup dictionary for per cell parameters, cell.id is key, if None given will be calculated
    :param k: number of nodes surrounding the organells compartment will be determined from
    :return: densities and volume densities for aoxn and dendrite
    '''
    cellid, cached_so_ids, cached_so_rep_coord, cached_so_volume, full_cell_dict,k, min_comp_len, axon_only = input
    cell = SuperSegmentationObject(cellid)
    segmentation_object_ids = cell.mi_ids
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
    if axon_only == False:
        den_unique = np.unique(np.where(axo == 0)[0], return_counts=True)
        den_inds = den_unique[0][den_unique[1] > k / 2]
        den_so_ids = segmentation_object_ids[den_inds]
        non_soma_inds = np.hstack([axon_inds, den_inds])
        segmentation_object_ids = segmentation_object_ids[non_soma_inds]
    if len(segmentation_object_ids) == 0:
        return 0, 0, 0, 0
    axo_so_amount = len(axo_so_ids)
    if full_cell_dict is not None:
        axon_length = full_cell_dict[cell.id]["axon length"]
    else:
        g = cell.weighted_graph(add_node_attr=('axoness_avg10000',))
        axon_length = get_compartment_length(cell, compartment=1, cell_graph=g)
    if axon_length < min_comp_len:
        return 0,0, 0, 0
    axo_so_density = axo_so_amount / axon_length
    axo_so_volume = np.sum(organell_volumes[axon_inds])
    axo_so_volume_density = axo_so_volume / axon_length
    if axon_only:
        return [axo_so_density, axo_so_volume_density, 0, 0]
    else:
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
        return [axo_so_density, den_so_density, axo_so_volume_density, den_so_volume_density]

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
                all_cell_dict = load_pkl2obj("wholebrain/scratch/arother/j0251v4_prep/combined_fullcell_ax_dict.pkl")
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
            :return: comp_len in µm
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

def generate_colored_mesh_from_skel_data(args):
    '''
    Generates mesh coloured according to axoness_avg10000 prediction of skeleton.
    Based partly on syconn2scripts.scripts.point_party.semseg_gt. Saves result as kzip
    :param args: cellid, path to folder where kzip should be stored, key to color
    :return:
    '''
    # color lookup from HA
    col_lookup = {0: (76, 92, 158, 255), 1: (255, 125, 125, 255), 2: (125, 255, 125, 255),
                  3: (113, 98, 227, 255),
                  4: (255, 255, 125, 255)}
    cellid, f_name, key = args

    cell = SuperSegmentationObject(cellid)
    cell.load_skeleton()
    # load skeleton axoness, spiness attributes
    nodes = cell.skeleton['nodes'] * cell.scaling
    axoness_labels = cell.skeleton[key]
    # load mesh and put skeleton annotations on mesh
    indices, vertices, normals = cell.mesh
    vertices = vertices.reshape((-1, 3))
    kdt = cKDTree(nodes)
    dists, node_inds = kdt.query(vertices)
    vert_axoness_labels = axoness_labels[node_inds]

    # save colored mesh
    cols = np.array([col_lookup[el] for el in vert_axoness_labels.squeeze()], dtype=np.uint8)
    kzip_out = f'{f_name}/{cellid}_colored_mesh'
    kzip_out_skel = f'{f_name}/{cellid}_skel'
    write_mesh2kzip(kzip_out, indices.astype(np.float32), vertices.astype(np.float32), None, cols,
                    f'{cellid}.ply')
    cell.save_skeleton_to_kzip(kzip_out_skel, additional_keys=[key])
    return

def generate_colored_mesh_synprob_data(args):
    '''
    Generates mesh coloured according to synapse_probability values for cells of synapses its part of.
    Does use all synapses cell is a part of, if synapses should be filtered this needs to be done before applying this function.
    Based partly on syconn2scripts.scripts.point_party.semseg_gt, similar to generate_colored_mesh_from_skel_data. Saves result as kzip
    :param args: cellid, path to folder where kzip should be stored, syn_ssv_partners, synapse coords, synapse probability, color lookup
    dictionary (categories to sort probability in should be keys)
    :return:
    '''
    cellid, f_name, syn_ssv_partners, syn_rep_coords, syn_prob, col_lookup = args
    #filter synapses for cellid
    cell = SuperSegmentationObject(cellid)
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
    # save colored mesh
    cols = np.array([col_lookup[el] for el in vert_synprob_labels_cats], dtype=np.uint8)
    kzip_out = f'{f_name}/{cellid}_colored_mesh'
    kzip_out_skel = f'{f_name}/{cellid}_skel'
    write_mesh2kzip(kzip_out, indices.astype(np.float32), vertices.astype(np.float32), None, cols,
                    f'{cellid}.ply')
    cell.save_skeleton_to_kzip(kzip_out_skel)
    return

def get_per_cell_mito_myelin_info(input):
    '''
    Function to get information about myelin fraction, mitochondria density and axon radius per cell
    :param input: cellid, cached mitochondria information, cell dict which cached cellids for celltype
    :return: median radius per cell, mitochondria volume density, myelin fraction
    '''
    cellid, min_comp_len, cached_mito_ids, cached_mito_rep_coords, cached_mito_volumes, full_cell_dict = input
    cell = SuperSegmentationObject(cellid)
    cell.load_skeleton()
    myelin_results = get_myelin_fraction(cellid=cellid, cell=cell, min_comp_len=min_comp_len, load_skeleton=False)
    abs_myelin_cell = myelin_results[0]
    rel_myelin_cell = myelin_results[1]
    if np.isnan(abs_myelin_cell):
        return [np.nan, np.nan, np.nan]
    #input for organell analysis: cellid, cached_so_ids, cached_so_rep_coord,
    # cached_so_volume, full_cell_dict,k, min_comp_len, axon_only
    organell_input = [cellid, cached_mito_ids, cached_mito_rep_coords,
                      cached_mito_volumes, full_cell_dict, 3, min_comp_len, False]
    mito_results = get_organell_volume_density(organell_input)
    den_mito_density_cell = mito_results[1]
    axo_mito_volume_density_cell = mito_results[2]
    axon_inds = np.nonzero(cell.skeleton["axoness_avg10000"] == 1)[0]
    axon_radii_cell = get_compartment_radii(cell, comp_inds=axon_inds)
    ax_median_radius_cell = np.median(axon_radii_cell)
    return [ax_median_radius_cell, axo_mito_volume_density_cell, rel_myelin_cell]

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

def get_cell_soma_radius(cellid, use_skel = False):
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
    soma_vert_avg = np.mean(soma_vert_coords, axis=0)
    dist2centre = np.linalg.norm(soma_vert_coords - soma_vert_avg, axis = 1)
    radius = np.median(dist2centre) / 1000
    return [soma_vert_avg, radius]







