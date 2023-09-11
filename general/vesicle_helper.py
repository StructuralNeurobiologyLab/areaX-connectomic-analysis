import pandas as pd
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
    dependency on that parameter. This function assumes cells are in the desired compartment.
    :param cell_input: list of inputs including cellid, ves_coords, mapped_ssv_ids, ves_dist2matrix
    :return: number of vesicles, vesicle number per pathlength
    '''
    cellid = cell_input[0]
    ves_coords = cell_input[1]
    mapped_ssv_ids = cell_input[2]
    ves_dist2matrix = cell_input[3]
    distance_threshold = cell_input[4]
    axon_pathlength = cell_input[5]
    #filer vesicles according to cell
    cell_ves_ind = np.in1d(mapped_ssv_ids, cellid)
    cell_ves_coords = ves_coords[cell_ves_ind]
    cell_dist2matrix = ves_dist2matrix[cell_ves_ind]
    number_vesicles = len(cell_ves_coords)
    # calculate density
    vesicle_density = number_vesicles / axon_pathlength
    #now filter according to distance to matrix
    close_densities = np.zeros(len(distance_threshold))
    for i, dt in enumerate(distance_threshold):
        cell_axo_ves_coords_thresh = cell_ves_coords[cell_dist2matrix < dt]
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
    Thresholds should be given as nm. This function assumes that only vesicles which are in an axon/ or other desired compartment are used.
    :param cell_input: list of inputs including cellid, ves_coords, mapped_ssv_ids, ves_dist2matrix,
            synapse_coordinates, synapse_axoness, syn_partner_ssvs, distance threshold
                        for distance to membrane, threshold for distance to synapse
    :return: fraction of vesicles (at membrane) not close to synapse, density of vesicles (membrane) not
             close to synapse, density of ves (membrane) close to synapse
    '''
    scaling = [10, 10, 25]
    cellid = cell_input[0]
    ves_coords = cell_input[1]
    mapped_ssv_ids = cell_input[2]
    ves_dist2matrix = cell_input[3]
    distance_threshold = cell_input[4]
    syn_coords = cell_input[5]
    syn_axs = cell_input[6]
    syn_ssv_partners = cell_input[7]
    syn_distance_threshold = cell_input[8]
    non_syn_distance_threshold = cell_input[9]
    axon_pathlength = cell_input[10]
    # filter synapses, similar to filtering in analysis_conn_helper
    ct_inds = np.in1d(syn_ssv_partners, cellid).reshape(len(syn_ssv_partners), 2)
    comp_inds = np.in1d(syn_axs, 1).reshape(len(syn_ssv_partners), 2)
    filtered_inds = np.all(ct_inds == comp_inds, axis=1)
    syn_coords = syn_coords[filtered_inds]
    if len(syn_coords) == 0:
        return [np.nan, np.nan, np.nan]
    #filter vesicles belonging to cell
    cell_ves_ind = np.in1d(mapped_ssv_ids, cellid)
    cell_ves_coords = ves_coords[cell_ves_ind]
    cell_dist2matrix = ves_dist2matrix[cell_ves_ind]
    #now filter according to distance to matrix
    cell_ves_coords_thresh = cell_ves_coords[cell_dist2matrix < distance_threshold]
    number_vesicles_close = len(cell_ves_coords_thresh)
    if number_vesicles_close == 0:
        return [np.nan, 0, 0]
    # get number vesicles close to membrane
    #make kdTree out of vesicle coords
    ves_kdtree = scipy.spatial.cKDTree(cell_ves_coords_thresh * scaling)
    #search which vesicle indices are within certain distance of synapse
    ves_inds = ves_kdtree.query_ball_point(syn_coords * scaling, r = syn_distance_threshold)
    if len(ves_inds) == 0:
        number_syn_ves_close = 0
    else:
        ves_inds_flatten = np.unique(np.hstack(ves_inds)).astype(int)
        ves_coords_syns = cell_ves_coords_thresh[ves_inds_flatten]
        number_syn_ves_close = len(ves_coords_syns)
    ves_far_inds = ves_kdtree.query_ball_point(syn_coords * scaling, r = non_syn_distance_threshold)
    if len(ves_far_inds) == 0:
        number_nonsyn_ves_close = number_vesicles_close
    else:
        ves_far_inds_flatten = np.unique(np.hstack(ves_far_inds)).astype(int)
        ves_coords_synsmore = cell_ves_coords_thresh[ves_far_inds_flatten]
        number_nonsyn_ves_close = number_vesicles_close - len(ves_coords_synsmore)
    fraction_non_syn_ves = number_nonsyn_ves_close / number_vesicles_close
    density_non_syn_ves = number_nonsyn_ves_close / axon_pathlength
    density_syn_ves = number_syn_ves_close / axon_pathlength
    return [fraction_non_syn_ves, density_non_syn_ves, density_syn_ves]

def get_ves_synsize_percell(cell_input):
    '''
    Function to filter single vesicles per cell according to coordinates and return number of vesicles per synapse, all and the ones within
    a certain distance to the membrane. This function assumes vesicles are already in the desired compartment
    The function assumes synapse parameters are already prefiltered according to a minimum synapse size,
    minimum synapse probability, only axo-dendritic/somatic if desired (use analysis_conn_helper.filter_synapse_caches_for_ct for this prefiltering).
    Function makes sure all synapses relating to the cell are presynaptic/ on its axon.
    Thresholds should be given as nm.
    :param cell_input: list of inputs including cellid, ves_coords, mapped_ssv_ids, ves_dist2matrix, synapse_coordinates, synapse_axoness, syn_partner_ssvs, distance threshold
                        for distance to membrane, threshold for distance to synapse
    :return: a Dataframe with the cellid, synapse sizes, number of vesicles per synapse in total, number of membrane-close vesicles per synapse
    '''
    #adapt for each dataset
    scaling = [10, 10, 25]
    cellid = cell_input[0]
    ves_coords = cell_input[1]
    mapped_ssv_ids = cell_input[2]
    ves_dist2matrix = cell_input[3]
    distance_threshold = cell_input[4]
    syn_coords = cell_input[5]
    syn_axs = cell_input[6]
    syn_ssv_partners = cell_input[7]
    syn_sizes = cell_input[8]
    syn_distance_threshold = cell_input[9]
    # filter synapses, similar to filtering in analysis_conn_helper
    ct_inds = np.in1d(syn_ssv_partners, cellid).reshape(len(syn_ssv_partners), 2)
    comp_inds = np.in1d(syn_axs, 1).reshape(len(syn_ssv_partners), 2)
    filtered_inds = np.all(ct_inds == comp_inds, axis=1)
    syn_coords = syn_coords[filtered_inds]
    syn_sizes = syn_sizes[filtered_inds]
    num_syns = len(syn_sizes)
    #filter vesicles related to cell
    cell_ves_ind = np.in1d(mapped_ssv_ids, cellid)
    cell_ves_coords = ves_coords[cell_ves_ind]
    cell_dist2matrix = ves_dist2matrix[cell_ves_ind]
    # get number of vesicles within certain distance to synapse
    # make kdTree out of vesicle coords
    ves_kdtree = scipy.spatial.cKDTree(cell_ves_coords * scaling)
    ves_inds = ves_kdtree.query_ball_point(syn_coords * scaling, r=syn_distance_threshold)
    number_ves_per_synapse = np.array([len(ves_inds[i]) for i in range(num_syns)])
    columns = ['cellid', 'synapse size [µm²]', 'number of vesicles', 'number of membrane-close vesicles']
    output_df = pd.DataFrame(columns = columns, index = range(num_syns))
    output_df['synapse size [µm²]'] = syn_sizes
    output_df['number of vesicles'] = number_ves_per_synapse
    output_df['cellid'] = cellid
    #now filter according to distance to matrix
    cell_axo_ves_coords_thresh = cell_ves_coords[cell_dist2matrix < distance_threshold]
    #search which vesicle indices are within certain distance of synapse and close to membrane
    ves_close_kdtree = scipy.spatial.cKDTree(cell_axo_ves_coords_thresh * scaling)
    ves_close_inds = ves_close_kdtree.query_ball_point(syn_coords * scaling, r = syn_distance_threshold)
    if len(ves_close_inds) == 0:
        output_df['number of membrane-close vesicles'] = 0
    else:
        number_close_per_synapse = np.array([len(ves_close_inds[i]) for i in range(len(syn_coords))])
        output_df['number of membrane-close vesicles'] = number_close_per_synapse
    return output_df

def get_vesicle_distance_information_per_cell(params):
    '''
    filter vesicles if they are in axon of the cell and saves information about distance to
    membrane. Filters synapses that are in axon of cell and calculates distance to
    next synapse per vesicle. Saves information about all vesicles in dataframe.
    This function filters vesicles and selects only those which are in the axon of the cells.
    :param input: cellid, vesicle_coords, vesicle_distance2membrane, ves_ssv_mapping, synapse_ssv_partners, synapse_axoness, synapse_coords
    :return: all vesicles with coords in Dataframe
    '''
    cellid = params[0]
    ves_coords = params[1]
    mapped_ssv_ids = params[2]
    ves_dist2matrix = params[3]
    syn_coords = params[4]
    syn_axs = params[5]
    syn_ssv_partners = params[6]
    celltype = params[7]
    # filter synapses, similar to filtering in analysis_conn_helper
    ct_inds = np.in1d(syn_ssv_partners, cellid).reshape(len(syn_ssv_partners), 2)
    comp_inds = np.in1d(syn_axs, 1).reshape(len(syn_ssv_partners), 2)
    filtered_inds = np.all(ct_inds == comp_inds, axis=1)
    syn_coords = syn_coords[filtered_inds]
    #filter vesicles for cell
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
    num_vesicles = len(cell_axo_ves_coords)
    # write in dataframe
    columns = ['cellid', 'celltype', 'ves coord x', 'ves coord y', 'ves coord z', 'dist 2 membrane', 'dist 2 synapse']
    output_df = pd.DataFrame(columns=columns, index=range(num_vesicles))
    output_df['cellid'] = cellid
    output_df['celltype'] = celltype
    output_df['ves coord x'] = cell_axo_ves_coords[:, 0]
    output_df['ves coord y'] = cell_axo_ves_coords[:, 1]
    output_df['ves coord z'] = cell_axo_ves_coords[:, 2]
    output_df['dist 2 membrane'] = cell_axo_dist2matrix
    #get distance of vesicles to synapses
    if len(syn_coords) > 0:
        syn_kdtree = scipy.spatial.cKDTree(syn_coords * cell.scaling)
        #search which vesicle indices are within certain distance of synapse
        ves_dist2syn = syn_kdtree.query(cell_axo_ves_coords * cell.scaling)[0]
        output_df['dist 2 synapse'] = ves_dist2syn
    return output_df

def map_axoness2ves(params):
    '''
    Map axoness (axon, dendrite, soma) of a cellid to the vesicles associated with it
    :param input: cellid, vesicle_ids, vesicle_coords, ssv_ids mapped to vesicles
    :return: vesicle_ids and axoness per cell
    '''

    cellid, ves_ids, ves_coords, mapped_ssv_ids = params
    cell = SuperSegmentationObject(cellid)
    cell.load_skeleton()
    cell_ves_ind = np.in1d(mapped_ssv_ids, cellid)
    cell_ves_coords = ves_coords[cell_ves_ind]
    cell_ves_ids = ves_ids[cell_ves_ind]
    kdtree = scipy.spatial.cKDTree(cell.skeleton["nodes"] * cell.scaling)
    close_node_ids = kdtree.query(cell_ves_coords * cell.scaling, k=1)[1].astype(int)
    axo = np.array(cell.skeleton["axoness_avg10000"][close_node_ids])
    axo[axo == 3] = 1
    axo[axo == 4] = 1
    return [cell_ves_ids, axo]


