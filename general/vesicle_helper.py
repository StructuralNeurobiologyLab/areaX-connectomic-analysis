from syconn.reps.super_segmentation import SuperSegmentationObject
import numpy as np
import scipy

def get_ves_distance_per_cell(cell_input):
    '''
    Function to filter single vesicles per cell according to coordinates and return the corresponding distances to matrix
    in nm. Filters vesicles with certain distance to matrix if filtering parameter given.
    :param cell_input: list of inputs including cellid, ves_coords, mapped_ssv_ids, ves_dist2matrix
    :return: number of vesicles, vesicle number per pathlength
    '''
    cellid = cell_input[0]
    ves_coords = cell_input[1]
    mapped_ssv_ids = cell_input[2]
    ves_dist2matrix = cell_input[3]
    distance_threshold = cell_input[4]
    axon_pathlength = cell_input[5]
    #load cell skeleton, filter all vesicles not close to axon
    cell = SuperSegmentationObject(cellid)
    cell.load_skeleton()
    cell_ves_ind = np.in1d(mapped_ssv_ids, cellid)
    cell_ves_coords = ves_coords[cell_ves_ind]
    cell_dist2matrix = ves_dist2matrix[cell_ves_ind]
    kdtree = scipy.spatial.cKDTree(cell.skeleton["nodes"] * cell.scaling)
    close_node_ids = kdtree.query(cell_ves_coords * cell.scaling, k=1)[1].astype(int)
    axo = np.array(cell.skeleton["axoness_avg10000"][close_node_ids])
    axo[axo == 3] = 1
    axo[axo == 4] = 1
    cell_axo_ves_coords = cell_ves_coords[axo == 1]
    cell_axo_dist2matrix = cell_dist2matrix[axo == 1]
    #now filter according to distance to matrix
    cell_axo_ves_coords_thresh = cell_axo_ves_coords[cell_axo_dist2matrix < distance_threshold]
    number_vesicles = len(cell_axo_ves_coords)
    number_vesicles_close = len(cell_axo_ves_coords_thresh)
    #calculate density
    vesicle_density = number_vesicles / axon_pathlength
    vesicle_density_close = number_vesicles_close / axon_pathlength
    return [vesicle_density, vesicle_density_close]

def get_ves_distance_multiple_per_cell(cell_input):
    '''
    Function to filter single vesicles per cell according to coordinates and return the corresponding distances to matrix
    in nm. Filters vesicles with certain distance to matrix if filtering parameter given. Give multiple distance thresholds to see
    dependency on that parameter.
    :param cell_input: list of inputs including cellid, ves_coords, mapped_ssv_ids, ves_dist2matrix
    :return: number of vesicles, vesicle number per pathlength
    '''
    cellid = cell_input[0]
    ves_coords = cell_input[1]
    mapped_ssv_ids = cell_input[2]
    ves_dist2matrix = cell_input[3]
    distance_thresholds = cell_input[4]
    axon_pathlength = cell_input[5]
    #load cell skeleton, filter all vesicles not close to axon
    cell = SuperSegmentationObject(cellid)
    cell.load_skeleton()
    cell_ves_ind = np.in1d(mapped_ssv_ids, cellid)
    cell_ves_coords = ves_coords[cell_ves_ind]
    cell_dist2matrix = ves_dist2matrix[cell_ves_ind]
    kdtree = scipy.spatial.cKDTree(cell.skeleton["nodes"] * cell.scaling)
    close_node_ids = kdtree.query(cell_ves_coords * cell.scaling, k=1)[1].astype(int)
    axo = np.array(cell.skeleton["axoness_avg10000"][close_node_ids])
    axo[axo == 3] = 1
    axo[axo == 4] = 1
    cell_axo_ves_coords = cell_ves_coords[axo == 1]
    cell_axo_dist2matrix = cell_dist2matrix[axo == 1]
    number_vesicles = len(cell_axo_ves_coords)
    # calculate density
    vesicle_density = number_vesicles / axon_pathlength
    #now filter according to distance to matrix
    close_densities = np.zeros(len(distance_thresholds))
    for i, dt in enumerate(distance_thresholds):
        cell_axo_ves_coords_thresh = cell_axo_ves_coords[cell_axo_dist2matrix < dt]
        number_vesicles_close = len(cell_axo_ves_coords_thresh)
        close_densities[i] = number_vesicles_close / axon_pathlength
    output = np.hstack(np.array([vesicle_density, close_densities]))
    return output

def get_synapse_proximity_vesicle_percell(cell_input):
    '''
    Function to filter single vesicles per cell according to coordinates and return the percentage of vesicles which is within
    a certain distance to the cell membrane and not close to the synapse (within a certain distance to the synapse).
    The function assumes synapse parameters are already prefiltered according to a minimum synapse size,
    minimum synapse probability, only axo-dendritic/somatic if desired (use analysis_conn_helper.filter_synapse_caches_for_ct for this prefiltering).
    Function makes sure all synapses relating to the cell are presynaptic/ on its axon.
    :param cell_input: list of inputs including cellid, ves_coords, mapped_ssv_ids, ves_dist2matrix, synapse_coordinates, synapse_axoness, syn_partner_ssvs, distance threshold
                        for distance to membrane, threshold for distance to synapse
    :return: vesicle density close to synapse, vesicle density further from synapse than threshold, % of vesicles not close to synapse
    '''
    cellid = cell_input[0]
    ves_coords = cell_input[1]
    mapped_ssv_ids = cell_input[2]
    ves_dist2matrix = cell_input[3]
    distance_threshold = cell_input[4]
    syn_coords = cell_input[5]
    syn_axs = cell_input[6]
    syn_partner_ssvs = cell_input[7]
    syn_distance_threshold = cell_input[8]
    axon_pathlength = cell_input[9]
    #load cell skeleton, filter all vesicles not close to axon
    cell = SuperSegmentationObject(cellid)
    cell.load_skeleton()
    cell_ves_ind = np.in1d(mapped_ssv_ids, cellid)
    cell_ves_coords = ves_coords[cell_ves_ind]
    cell_dist2matrix = ves_dist2matrix[cell_ves_ind]
    kdtree = scipy.spatial.cKDTree(cell.skeleton["nodes"] * cell.scaling)
    close_node_ids = kdtree.query(cell_ves_coords * cell.scaling, k=1)[1].astype(int)
    axo = np.array(cell.skeleton["axoness_avg10000"][close_node_ids])
    axo[axo == 3] = 1
    axo[axo == 4] = 1
    cell_axo_ves_coords = cell_ves_coords[axo == 1]
    cell_axo_dist2matrix = cell_dist2matrix[axo == 1]
    #now filter according to distance to matrix
    cell_axo_ves_coords_thresh = cell_axo_ves_coords[cell_axo_dist2matrix < distance_threshold]
    number_vesicles_close = len(cell_axo_ves_coords_thresh)
    #filter synapses

    #calculate density
    vesicle_density_close = number_vesicles_close / axon_pathlength
    return [number_vesicles, number_vesicles_close, vesicle_density, vesicle_density_close]