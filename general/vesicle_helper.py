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
    number_vesicles = len(cell_axo_ves_coords)
    # calculate density
    vesicle_density = number_vesicles / axon_pathlength
    #now filter according to distance to matrix
    close_densities = np.zeros(len(distance_threshold))
    for i, dt in enumerate(distance_threshold):
        cell_axo_ves_coords_thresh = cell_axo_ves_coords[cell_axo_dist2matrix < dt]
        number_vesicles_close = len(cell_axo_ves_coords_thresh)
        close_densities[i] = number_vesicles_close / axon_pathlength
    output = np.hstack([vesicle_density, close_densities])
    return output

def get_synapse_proximity_vesicle_percell(cell_input):
    '''
    Function to filter single vesicles per cell according to coordinates and return the percentage of vesicles which is within
    a certain distance to the cell membrane and not close to the synapse (within a certain distance to the synapse).
    The function assumes synapse parameters are already prefiltered according to a minimum synapse size,
    minimum synapse probability, only axo-dendritic/somatic if desired (use analysis_conn_helper.filter_synapse_caches_for_ct for this prefiltering).
    Function makes sure all synapses relating to the cell are presynaptic/ on its axon.
    Thresholds should be given as nm.
    :param cell_input: list of inputs including cellid, ves_coords, mapped_ssv_ids, ves_dist2matrix, synapse_coordinates, synapse_axoness, syn_partner_ssvs, distance threshold
                        for distance to membrane, threshold for distance to synapse
    :return: fraction of vesicles (at membrane) not close to synapse, density of vesicles (membrane) not
             close to synapse, density of ves (membrane) close to synapse
    '''
    cellid = cell_input[0]
    ves_coords = cell_input[1]
    mapped_ssv_ids = cell_input[2]
    ves_dist2matrix = cell_input[3]
    distance_threshold = cell_input[4]
    syn_coords = cell_input[5]
    syn_axs = cell_input[6]
    syn_ssv_partners = cell_input[7]
    syn_distance_threshold = cell_input[8]
    axon_pathlength = cell_input[9]
    # filter synapses, similar to filtering in analysis_conn_helper
    ct_inds = np.in1d(syn_ssv_partners, cellid).reshape(len(syn_ssv_partners), 2)
    comp_inds = np.in1d(syn_axs, 1).reshape(len(syn_ssv_partners), 2)
    filtered_inds = np.all(ct_inds == comp_inds, axis=1)
    syn_coords = syn_coords[filtered_inds]
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
    if number_vesicles_close == 0:
        return [np.nan, 0, 0]
    # get number vesicles close to membrane
    #make kdTree out of vesicle coords
    ves_kdtree = scipy.spatial.cKDTree(cell_axo_ves_coords_thresh * cell.scaling)
    #search which vesicle indices are within certain distance of synapse
    ves_inds = ves_kdtree.query_ball_point(syn_coords * cell.scaling, r = syn_distance_threshold)
    if len(ves_inds) == 0:
        number_syn_ves_close = 0
    else:
        ves_inds_flatten = np.hstack(ves_inds).astype(int)
        ves_coords_syns = cell_axo_ves_coords_thresh[ves_inds_flatten]
        number_syn_ves_close = len(ves_coords_syns)
    #TO DO: indices sorted to closest synapses, get synapse size
    fraction_non_syn_ves = (number_vesicles_close - number_syn_ves_close) / number_vesicles_close
    density_non_syn_ves = (number_vesicles_close - number_syn_ves_close) / axon_pathlength
    density_syn_ves = number_syn_ves_close / axon_pathlength
    return [fraction_non_syn_ves, density_non_syn_ves, density_syn_ves]