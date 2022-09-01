if __name__ == '__main__':
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    from syconn.reps.segmentation import SegmentationDataset
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import networkx as nx
    import pandas as pd
    import os as os
    import scipy
    import time
    from syconn.handler.config import initialize_logging
    from syconn.handler.basics import load_pkl2obj
    from tqdm import tqdm
    from syconn.handler.basics import write_obj2pkl
    from scipy.stats import ranksums
    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv",working_dir=global_params.config.working_dir)
    sd_mitossv = SegmentationDataset("mi",working_dir=global_params.config.working_dir)
    sd_vcssv = SegmentationDataset("vc",working_dir=global_params.config.working_dir)

    def mito_spiness_percell(sso, cached_mito_ids, cached_mito_rep_coord, cached_mito_mesh_bb, cached_mito_volume, min_comp_len = 200, k = 3):
        """
        sorts mitochonria into proximal (<50 µm) or distal (>100 µm) and saves parameter like mitochondrium volume density, mitochondrium length density, amount of mitochondria and average distance to soma
        compare mito parameter to axon_dendrite_synsize, spiness
        returns dictionary with morphological parameters lie axon, dendritic length, amount of spines, number of dendrites and
        dictionary with mito_ids, mito_volume, mito_length, mito_pathlength, axon_mito_ids, dendrite_mtio_ids, distal_mito_ids, proximal_mito_ids
        """
        sso.load_skeleton()
        non_axon_inds = np.nonzero(sso.skeleton["axoness_avg10000"] != 1)[0]
        g = sso.weighted_graph(add_node_attr=('axoness_avg10000', "spiness"))
        axon_graph = g.copy()
        axon_graph.remove_nodes_from(non_axon_inds)
        axon_length = axon_graph.size(weight = "weight") / 1000 #in µm
        if axon_length < min_comp_len:
            return 0, 0
        non_dendrite_inds = np.nonzero(sso.skeleton["axoness_avg10000"] != 0)[0]
        dendrite_graph = g.copy()
        dendrite_graph.remove_nodes_from(non_dendrite_inds)
        dendrite_length = dendrite_graph.size(weight = "weight") / 1000 #in µm
        if dendrite_length < min_comp_len:
            return 0, 0
        amount_dendrite_subgraphs = len(list(nx.connected_component_subgraphs(dendrite_graph)))
        spine_shaftinds = np.nonzero(sso.skeleton["spiness"] == 0)[0]
        spine_otherinds = np.nonzero(sso.skeleton["spiness"] == 3)[0]
        nonspine_inds = np.hstack([spine_shaftinds, spine_otherinds])
        spine_graph = dendrite_graph.copy()
        spine_graph.remove_nodes_from(nonspine_inds)
        spine_amount_skeleton = len(list(nx.connected_component_subgraphs(spine_graph)))
        morph_params = {"axon_length": axon_length, "den_length": dendrite_length, "spine_amount": spine_amount_skeleton, "am_den_subgraphs": amount_dendrite_subgraphs}
        kdtree = scipy.spatial.cKDTree(sso.skeleton["nodes"]*sso.scaling)
        all_mito_ids = sso.mi_ids
        #mitochrondrial index and density (every µm)
        #get mito parameters per cell
        sso_mito_inds = np.in1d(cached_mito_ids, all_mito_ids)
        all_mito_volume = cached_mito_volume[sso_mito_inds] * 10 ** (-9) * np.prod(sso.scaling)  # convert to cubic µm
        all_mi_rep_coord = cached_mito_rep_coord[sso_mito_inds] * sso.scaling #in nm
        mi_mesh_bb = cached_mito_mesh_bb[sso_mito_inds]
        #give all coordinates of one cell in together for shorter running time
        close_node_ids = kdtree.query(all_mi_rep_coord, k = k)[1].astype(int)
        axo = np.array(sso.skeleton["axoness_avg10000"][close_node_ids])
        axo[axo == 3] = 1
        axo[axo == 4] = 1
        axon_unique = np.unique(np.where(axo == 1)[0], return_counts = True)
        axon_inds = axon_unique[0][axon_unique[1] > k/2]
        all_axo_mito_ids = all_mito_ids[axon_inds]
        den_unique = np.unique(np.where(axo == 0)[0], return_counts=True)
        den_inds = den_unique[0][den_unique[1] > k / 2]
        all_den_mito_ids = all_mito_ids[den_inds]
        non_soma_inds = np.hstack([axon_inds, den_inds])
        all_mito_ids = all_mito_ids[non_soma_inds]
        if len(all_mito_ids) == 0:
            return 0, 0
        all_mi_rep_coord = all_mi_rep_coord[non_soma_inds]
        mi_mesh_bb = mi_mesh_bb[non_soma_inds]
        mi_diff = mi_mesh_bb[:, 1] - mi_mesh_bb[:, 0]
        mi_diff_start = all_mi_rep_coord - mi_mesh_bb[:, 0]
        mi_diff_end = mi_mesh_bb[:, 1] -  all_mi_rep_coord
        all_mito_length = np.linalg.norm(mi_diff, axis=1) / 1000  # in µm
        all_mito_length_start = np.linalg.norm(mi_diff_start, axis=1) / 1000  # in µm
        all_mito_length_end = np.linalg.norm(mi_diff_end, axis=1) / 1000  # in µm
        all_mito_volume = all_mito_volume[non_soma_inds]
        all_mito_pathlength = sso.shortestpath2soma(all_mi_rep_coord)[0] / 1000 #in µm
        mi_start_pathlength = all_mito_pathlength - all_mito_length_start
        mi_end_pathlength = all_mito_pathlength + all_mito_length_end
        prox_inds = mi_start_pathlength < 50
        dist_inds = mi_end_pathlength > 100
        prox_mito_ids = all_mito_ids[prox_inds]
        dist_mito_ids = all_mito_ids[dist_inds]
        mito_params = {"all_ids": all_mito_ids, "axo_ids": all_axo_mito_ids, "den_ids": all_den_mito_ids, "dist_ids": dist_mito_ids, "prox_ids": prox_mito_ids,
                       "pathlength": all_mito_pathlength, "length": all_mito_length, "volume": all_mito_volume}

        return morph_params, mito_params

    def synapse_per_cell(sso, cached_ids, cached_ssv_partners, cached_partner_axoness, cached_sizes, cached_rep_coord, cached_vc_ids, cached_vc_sizes,
                         cached_vc_mesh_bb, cached_vc_repcoord, min_syn_size = 0.1):
        """
        iterate over synapses per cell and returns ids with minimal size and syn_probabilty threshold. Compares to cached array with synapse properties to sort into axon,
        dendrite. Calculates distance to soma to sort into distant and proximal.
        axoness: 0 = dendrite, 1 = axon, 2 = soma
        :param sso: super segmentation object
        :param min_syn_size: threshold for minimal synapse size
        :param cached_ids: synapses ids from same celltype, thresholded by synapse probability
        :param cached_partner_axoness: axoness synapse on both neurons, of synapses from same celltype, thresholded by synapse probability
        :param cached_ssv_partners: ssv.id of both neurons, of synapses from same celltpye, thresholded by synapse probability
        :return: synapse_ids, synapse_axon_ids, synapse_dendrite_ids, distal_synape_ids, proximal_synapse_ids, synapse_sizes
        """
        syn_ssvs = sso.syn_ssv
        if len(syn_ssvs) == 0:
            return 0,0
        # get ids from cell
        sso_syn_ids0 = np.where(cached_ssv_partners == sso.id)[0]
        all_syn_ids = cached_ids[sso_syn_ids0]
        syn_rep_coords = cached_rep_coord[sso_syn_ids0]
        all_syn_sizes = cached_sizes[sso_syn_ids0]
        partner_axoness = cached_partner_axoness[sso_syn_ids0]
        ssv_partners = cached_ssv_partners[sso_syn_ids0]
        # filter those that don have minimum size
        min_syn_inds = all_syn_sizes > min_syn_size
        all_syn_ids = all_syn_ids[min_syn_inds]
        all_syn_sizes = all_syn_sizes[min_syn_inds]
        syn_rep_coords = syn_rep_coords[min_syn_inds]
        partner_axoness = partner_axoness[min_syn_inds]
        ssv_partners = ssv_partners[min_syn_inds]
        #determine axoness of synapse
        sso_syn_ids1 = np.where(ssv_partners == sso.id)
        axoness = partner_axoness[sso_syn_ids1]
        axo_inds = axoness == 1
        den_inds = axoness == 0
        axo_syn_ids = all_syn_ids[axo_inds]
        den_syn_ids = all_syn_ids[den_inds]
        non_soma_inds = axo_inds + den_inds
        all_syn_ids = all_syn_ids[non_soma_inds]
        if len(all_syn_ids) == 0:
            return 0, 0
        syn_sizes = all_syn_sizes[non_soma_inds]
        syn_rep_coords = syn_rep_coords[non_soma_inds] * sso.scaling
        syn_rep_coord_pathlength = sso.shortestpath2soma(syn_rep_coords)[0] / 1000  # in µm
        dist_inds = syn_rep_coord_pathlength > 100
        dist_syn_ids = all_syn_ids[dist_inds]
        prox_inds = syn_rep_coord_pathlength < 50
        prox_syn_ids = all_syn_ids[prox_inds]
        syn_dict = {"all_ids": all_syn_ids, "axo_ids": axo_syn_ids, "den_ids": den_syn_ids, "prox_ids": prox_syn_ids,
                    "dist_ids": dist_syn_ids, "syn_sizes": syn_sizes}
        # get vc_id and size for sso id
        sso_vc_ids = sso.vc_ids
        if len(sso_vc_ids) == 0:
            return 0, 0
        vc_inds = np.in1d(cached_vc_ids, sso_vc_ids)
        sso_vc_sizes = cached_vc_sizes[vc_inds]  * 10 ** (-9) * np.prod(sso.scaling) #µm
        sso_vc_repcoord = cached_vc_repcoord[vc_inds] * sso.scaling # in nm
        sso_vc_mesh_bb = cached_vc_mesh_bb[vc_inds]
        syn_kdtree = scipy.spatial.cKDTree(syn_rep_coords)
        distance_to_synapse, closest_syn_inds = syn_kdtree.query(sso_vc_repcoord, k = 1)
        closest_syn_inds = closest_syn_inds.astype(int)
        closest_syn_ids = all_syn_ids[closest_syn_inds]
        axo_vc_inds = np.in1d(closest_syn_ids, axo_syn_ids)
        close_axo_syn_ids = closest_syn_ids[axo_vc_inds]
        axo_vc_ids = sso_vc_ids[axo_vc_inds]
        axo_vc_sizes = sso_vc_sizes[axo_vc_inds]
        axo_vc_mesh_bb = sso_vc_mesh_bb[axo_vc_inds]
        distance_to_synapse = distance_to_synapse[axo_vc_inds] / 1000 #µm
        axo_vc_repcoord = sso_vc_repcoord[axo_vc_inds]
        if len(axo_vc_repcoord) == 0:
            return 0, 0
        distance_to_soma = sso.shortestpath2soma(axo_vc_repcoord)[0]/ 1000
        vc_diff_start = axo_vc_repcoord - axo_vc_mesh_bb[:, 0]
        vc_diff_end = axo_vc_mesh_bb[:, 1] - axo_vc_repcoord
        vc_lengths_start = np.linalg.norm(vc_diff_start, axis = 1) / 1000 #in µm
        vc_lengths_end = np.linalg.norm(vc_diff_end, axis=1) / 1000  # in µm
        vc_start = distance_to_soma - vc_lengths_start
        vc_end = distance_to_soma + vc_lengths_end
        prox_vc_inds = vc_start < 50
        prox_vc_ids = axo_vc_ids[prox_vc_inds]
        dist_vc_inds = vc_end > 100
        dist_vc_ids = axo_vc_ids[dist_vc_inds]
        axo_vc_dict = {"axo_ids":axo_vc_ids, "prox_ids": prox_vc_ids, "dist_ids": dist_vc_ids,
        "vc_sizes": axo_vc_sizes, "distance2synapse": distance_to_synapse}
        return syn_dict, axo_vc_dict

    def ct_mito_syn_spiness_analysis(ssd, celltype, min_comp_len = 200, syn_prob_thresh = 0.6, syn_size_thresh = 0.1, close2syn_thresh = 1):
        """
        analysis of mitochondria length in proximal [until 50 µm] and distal[from 100 µm onwards] dendrites.
        Parameters that will be plotted are mitochondrial lenth (norm of bounding box diameter, might need a more exact measurement later),
        mitochondrial volume(mi.size), mitochondrial density (amount of mitochondria per µm axon/dendrite), mitochondrial index (mitochondrial
        length per µm axon/dendritic length) (see chandra et al., 2019 for comparison)
        :param ssd: SuperSegmentationDataset
        :param celltype: 0:"STN", 1:"DA", 2:"MSN", 3:"LMAN", 4:"HVC", 5:"GP", 6:"FS", 7:"TAN", 8:"INT"
        celltypes: j0256: STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7,
    #                      FS=8, LTS=9, NGF=10
                compartment: 0 = dendrite, 1 = axon, 2 = soma
        :return: mitochondrial parameters in proximal and sital dendrites
        """
        start = time.time()
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   10: "NGF"}
        f_name = ("u/arother/test_folder/210707_j0251v3_mito_spiness_%s_mcl%i_c2s_%i" % (ct_dict[celltype], min_comp_len, close2syn_thresh))
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('mitchondrial_synapse_analysis', log_dir=f_name + '/logs/')
        time_stamps = [time.time()]
        step_idents = ['t-0']
        comp_dict = {1: 'axons', 0: 'dendrites'}
        # cellids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == celltype]
        if celltype == 5:
            cellids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_handpicked.pkl" % ct_dict[celltype])
        else:
            cellids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_arr.pkl" % ct_dict[celltype])
        #prepare mito caches
        cached_mito_ids = sd_mitossv.ids
        cached_mito_mesh_bb = sd_mitossv.load_cached_data("mesh_bb")
        cached_mito_rep_coords = sd_mitossv.load_cached_data("rep_coord")
        cached_mito_volumes = sd_mitossv.load_cached_data("size")
        #prepare vesicle cloud caches
        cached_vc_ids = sd_vcssv.ids
        cached_vc_rep_coords = sd_vcssv.load_cached_data("rep_coord")
        cached_vc_sizes = sd_vcssv.load_cached_data("size")
        cached_vc_mesh_bb = sd_vcssv.load_cached_data("mesh_bb")
        #prepare synapse caches with synapse threshold for right celltype
        syn_prob = sd_synssv.load_cached_data("syn_prob")
        m = syn_prob > syn_prob_thresh
        m_ids = sd_synssv.ids[m]
        m_axs = sd_synssv.load_cached_data("partner_axoness")[m]
        m_axs[m_axs == 3] = 1
        m_axs[m_axs == 4] = 1
        m_cts = sd_synssv.load_cached_data("partner_celltypes")[m]
        m_ssv_partners = sd_synssv.load_cached_data("neuron_partners")[m]
        m_sizes = sd_synssv.load_cached_data("mesh_area")[m] / 2
        m_rep_coords = sd_synssv.load_cached_data("rep_coord")[m]
        ct_inds = np.any(m_cts == celltype, axis=1)
        m_ids = m_ids[ct_inds]
        m_axs = m_axs[ct_inds]
        m_ssv_partners = m_ssv_partners[ct_inds]
        m_sizes = m_sizes[ct_inds]
        m_rep_coords = m_rep_coords[ct_inds]
        #prepare empty arrays for parameters to plot
        spine_density = np.zeros(len(cellids))
        #mito parameters: amount, median
        axo_mito_amount = np.zeros(len(cellids))
        den_mito_amount = np.zeros(len(cellids))
        axo_prox_mito_amount = np.zeros(len(cellids))
        axo_dist_mito_amount = np.zeros(len(cellids))
        den_prox_mito_amount = np.zeros(len(cellids))
        den_dist_mito_amount = np.zeros(len(cellids))
        axo_median_mito_length = np.zeros(len(cellids))
        den_median_mito_length = np.zeros(len(cellids))
        axo_prox_median_mito_length = np.zeros(len(cellids))
        axo_dist_median_mito_length = np.zeros(len(cellids))
        den_prox_median_mito_length = np.zeros(len(cellids))
        den_dist_median_mito_length = np.zeros(len(cellids))
        axo_median_mito_volume = np.zeros(len(cellids))
        den_median_mito_volume = np.zeros(len(cellids))
        axo_prox_median_mito_volume = np.zeros(len(cellids))
        axo_dist_median_mito_volume = np.zeros(len(cellids))
        den_prox_median_mito_volume = np.zeros(len(cellids))
        den_dist_median_mito_volume = np.zeros(len(cellids))
        # mito parameters: mito densities
        axo_mito_amount_density = np.zeros(len(cellids))
        den_mito_amount_density = np.zeros(len(cellids))
        axo_prox_mito_amount_density = np.zeros(len(cellids))
        axo_dist_mito_amount_density = np.zeros(len(cellids))
        den_prox_mito_amount_density = np.zeros(len(cellids))
        den_dist_mito_amount_density = np.zeros(len(cellids))
        axo_mito_length_density = np.zeros(len(cellids))
        den_mito_length_density = np.zeros(len(cellids))
        axo_prox_mito_length_density = np.zeros(len(cellids))
        axo_dist_mito_length_density = np.zeros(len(cellids))
        den_prox_mito_length_density = np.zeros(len(cellids))
        den_dist_mito_length_density = np.zeros(len(cellids))
        axo_mito_volume_density = np.zeros(len(cellids))
        den_mito_volume_density = np.zeros(len(cellids))
        axo_prox_mito_volume_density = np.zeros(len(cellids))
        axo_dist_mito_volume_density = np.zeros(len(cellids))
        den_prox_mito_volume_density = np.zeros(len(cellids))
        den_dist_mito_volume_density = np.zeros(len(cellids))
        # synapse parameters: amount, median
        axo_synapse_amount = np.zeros(len(cellids))
        den_synapse_amount = np.zeros(len(cellids))
        axo_prox_synapse_amount = np.zeros(len(cellids))
        axo_dist_synapse_amount = np.zeros(len(cellids))
        den_prox_synapse_amount = np.zeros(len(cellids))
        den_dist_synapse_amount = np.zeros(len(cellids))
        axo_median_syn_size = np.zeros(len(cellids))
        den_median_syn_size = np.zeros(len(cellids))
        axo_prox_median_syn_size = np.zeros(len(cellids))
        axo_dist_median_syn_size = np.zeros(len(cellids))
        den_prox_median_syn_size = np.zeros(len(cellids))
        den_dist_median_syn_size = np.zeros(len(cellids))
        #synapse amount and size densities
        axo_syn_amount_density = np.zeros(len(cellids))
        den_syn_amount_density = np.zeros(len(cellids))
        axo_prox_syn_am_density = np.zeros(len(cellids))
        axo_dist_syn_am_density = np.zeros(len(cellids))
        den_prox_syn_am_density = np.zeros(len(cellids))
        den_dist_syn_am_density = np.zeros(len(cellids))
        axo_syn_size_density = np.zeros(len(cellids))
        den_syn_size_density = np.zeros(len(cellids))
        axo_prox_syn_size_density = np.zeros(len(cellids))
        axo_dist_syn_size_density = np.zeros(len(cellids))
        den_prox_syn_size_density = np.zeros(len(cellids))
        den_dist_syn_size_density = np.zeros(len(cellids))
        # vesicle clouds parameters: amount, median size, median distance to synapse
        axo_vc_amount = np.zeros(len(cellids))
        axo_prox_vc_amount = np.zeros(len(cellids))
        axo_dist_vc_amount = np.zeros(len(cellids))
        axo_median_vc_volume = np.zeros(len(cellids))
        axo_prox_median_vc_volume = np.zeros(len(cellids))
        axo_dist_median_vc_volume = np.zeros(len(cellids))
        axo_median_vc_dist2syn = np.zeros(len(cellids))
        axo_prox_median_vc_dist2syn = np.zeros(len(cellids))
        axo_dist_median_vc_dist2syn = np.zeros(len(cellids))
        axo_vc_close2syn_ratio = np.zeros(len(cellids))
        axo_prox_vc_close2syn_ratio = np.zeros(len(cellids))
        axo_dist_vc_close2syn_ratio = np.zeros(len(cellids))
        # vesicle cloud, amount, size density
        axo_vc_amount_density = np.zeros(len(cellids))
        axo_prox_vc_am_density = np.zeros(len(cellids))
        axo_dist_vc_am_density = np.zeros(len(cellids))
        axo_vc_vol_density = np.zeros(len(cellids))
        axo_prox_vc_vol_density = np.zeros(len(cellids))
        axo_dist_vc_vol_density = np.zeros(len(cellids))
        comp_dict = {1: 'axons', 0: 'dendrites'}
        log.info('Step 1/2 generating mitochondrial and synaptic parameters per cell')
        for i, cell in enumerate(tqdm(ssd.get_super_segmentation_object(cellids))):
            morph_dict_cell, mito_dict_cell = mito_spiness_percell(cell, cached_mito_ids=cached_mito_ids, cached_mito_mesh_bb=cached_mito_mesh_bb,
                                                                   cached_mito_rep_coord=cached_mito_rep_coords,cached_mito_volume=cached_mito_volumes, min_comp_len=min_comp_len)
            if type(morph_dict_cell) == int:
                continue
            syn_dict_cell, vc_dict_cell = synapse_per_cell(cell, cached_ids=m_ids, cached_partner_axoness=m_axs, cached_ssv_partners=m_ssv_partners,
                                             cached_sizes=m_sizes, cached_rep_coord=m_rep_coords, cached_vc_ids=cached_vc_ids,
                                             cached_vc_sizes= cached_vc_sizes, cached_vc_mesh_bb=cached_vc_mesh_bb,
                                                           cached_vc_repcoord= cached_vc_rep_coords,min_syn_size=syn_size_thresh)
            if type(syn_dict_cell) == int:
                continue
            den_length = morph_dict_cell["den_length"]
            axon_length = morph_dict_cell["axon_length"]
            # spine density
            spine_density[i] = morph_dict_cell["spine_amount"]/ den_length
            # calcuate mito parameters: mito density of amount, length, volume, average amount, length, volume
            mito_ids = mito_dict_cell["all_ids"]
            axon_mito_inds = np.in1d(mito_ids, mito_dict_cell["axo_ids"])
            axon_mito_ids = mito_ids[axon_mito_inds]
            den_mito_inds = np.in1d(mito_ids, mito_dict_cell["den_ids"])
            den_mito_ids = mito_ids[den_mito_inds]
            axo_mito_amount[i] = len(axon_mito_ids)
            den_mito_amount[i] = len(den_mito_ids)
            axo_prox_mito_ids = axon_mito_ids[np.in1d(axon_mito_ids, mito_dict_cell["prox_ids"])]
            axo_prox_mito_inds = np.in1d(mito_ids, axo_prox_mito_ids)
            axo_prox_mito_amount[i] = len(axo_prox_mito_ids)
            axo_dist_mito_ids = axon_mito_ids[np.in1d(axon_mito_ids, mito_dict_cell["dist_ids"])]
            axo_dist_mito_inds = np.in1d(mito_ids, axo_dist_mito_ids)
            axo_dist_mito_amount[i] = len(axo_dist_mito_ids)
            den_prox_mito_ids = den_mito_ids[np.in1d(den_mito_ids, mito_dict_cell["prox_ids"])]
            den_prox_mito_inds = np.in1d(mito_ids, den_prox_mito_ids)
            den_prox_mito_amount[i] = len(den_prox_mito_ids)
            den_dist_mito_ids = den_mito_ids[np.in1d(den_mito_ids, mito_dict_cell["dist_ids"])]
            den_dist_mito_inds = np.in1d(mito_ids, den_dist_mito_ids)
            den_dist_mito_amount[i] = len(den_dist_mito_ids)
            #mito amount densities
            axo_mito_amount_density[i] = axo_mito_amount[i]/ axon_length
            den_mito_amount_density[i] = den_mito_amount[i]/den_length
            axo_prox_mito_amount_density[i] = axo_prox_mito_amount[i]/ 50
            axo_dist_mito_amount_density[i] = axo_dist_mito_amount[i]/ (axon_length - 100)
            primary_dendrites = morph_dict_cell["am_den_subgraphs"]
            den_prox_mito_amount_density[i] = den_prox_mito_amount[i]/ (50*primary_dendrites)
            den_dist_mito_amount_density[i] = den_dist_mito_amount[i]/ (den_length - 100*primary_dendrites)
            # mito median length and length density in µm/µm
            mito_lengths = mito_dict_cell["length"]
            axo_median_mito_length[i]= np.median(mito_lengths[axon_mito_inds])
            den_median_mito_length[i] = np.median(mito_lengths[den_mito_inds])
            axo_prox_median_mito_length[i] = np.median(mito_lengths[axo_prox_mito_inds])
            axo_dist_median_mito_length[i] = np.median(mito_lengths[axo_dist_mito_inds])
            den_prox_median_mito_length[i] = np.median(mito_lengths[den_prox_mito_inds])
            den_dist_median_mito_length[i] = np.median(mito_lengths[den_dist_mito_inds])
            axo_mito_length_density[i] = np.sum(mito_lengths[axon_mito_inds]) / axon_length
            den_mito_length_density[i] = np.sum(mito_lengths[den_mito_inds])/ den_length
            axo_prox_mito_length_density[i] = np.sum(mito_lengths[axo_prox_mito_inds])/ 50
            axo_dist_mito_length_density[i]  = np.sum(mito_lengths[axo_dist_mito_inds])/(axon_length - 100)
            den_prox_mito_length_density[i] = np.sum(mito_lengths[den_prox_mito_inds]) / (50*primary_dendrites)
            den_dist_mito_length_density[i] = np.sum(mito_lengths[den_dist_mito_inds])/(den_length - 100*primary_dendrites)
            # mito median volume and volume density in µm3/µm
            mito_volumes = mito_dict_cell["volume"]
            axo_median_mito_volume[i] = np.median(mito_volumes[axon_mito_inds])
            den_median_mito_volume[i] = np.median(mito_volumes[den_mito_inds])
            axo_prox_median_mito_volume[i] = np.median(mito_volumes[axo_prox_mito_inds])
            axo_dist_median_mito_volume[i] = np.median(mito_volumes[axo_dist_mito_inds])
            den_prox_median_mito_volume[i] = np.median(mito_volumes[den_prox_mito_inds])
            den_dist_median_mito_volume[i] = np.median(mito_volumes[den_dist_mito_inds])
            axo_mito_volume_density[i] = np.sum(mito_volumes[axon_mito_inds]) / axon_length
            den_mito_volume_density[i] = np.sum(mito_volumes[den_mito_inds]) / den_length
            axo_prox_mito_volume_density[i] = np.sum(mito_volumes[axo_prox_mito_inds]) / 50
            axo_dist_mito_volume_density[i] = np.sum(mito_volumes[axo_dist_mito_inds]) / (axon_length - 100)
            den_prox_mito_volume_density[i] = np.sum(mito_volumes[den_prox_mito_inds]) / (50 * primary_dendrites)
            den_dist_mito_volume_density[i] = np.sum(mito_volumes[den_dist_mito_inds]) / (den_length - 100 * primary_dendrites)
            # synapse amount and amount density
            syn_ids = syn_dict_cell["all_ids"]
            axon_syn_inds = np.in1d(syn_ids, syn_dict_cell["axo_ids"])
            axon_syn_ids = syn_ids[axon_syn_inds]
            den_syn_inds = np.in1d(syn_ids, syn_dict_cell["den_ids"])
            den_syn_ids = syn_ids[den_syn_inds]
            axo_synapse_amount[i] = len(axon_syn_ids)
            den_synapse_amount[i] = len(den_syn_ids)
            axo_prox_syn_ids = axon_syn_ids[np.in1d(axon_syn_ids, syn_dict_cell["prox_ids"])]
            axo_prox_syn_inds = np.in1d(syn_ids, axo_prox_syn_ids)
            axo_prox_synapse_amount[i] = len(axo_prox_syn_ids)
            axo_dist_syn_ids = axon_syn_ids[np.in1d(axon_syn_ids, syn_dict_cell["dist_ids"])]
            axo_dist_syn_inds = np.in1d(syn_ids, axo_dist_syn_ids)
            axo_dist_synapse_amount[i] = len(axo_dist_syn_ids)
            den_prox_syn_ids = den_syn_ids[np.in1d(den_syn_ids, syn_dict_cell["prox_ids"])]
            den_prox_syn_inds = np.in1d(syn_ids, den_prox_syn_ids)
            den_prox_synapse_amount[i] = len(den_prox_syn_ids)
            den_dist_syn_ids = den_syn_ids[np.in1d(den_syn_ids, syn_dict_cell["dist_ids"])]
            den_dist_syn_inds = np.in1d(syn_ids, den_prox_syn_ids)
            den_dist_synapse_amount[i] = len(den_dist_syn_ids)
            axo_syn_amount_density[i] = axo_synapse_amount[i] / axon_length
            den_syn_amount_density[i] = den_synapse_amount[i] / den_length
            axo_prox_syn_am_density[i] = axo_prox_synapse_amount[i] / 50
            axo_dist_syn_am_density[i] = axo_dist_synapse_amount[i] / (axon_length - 100)
            den_prox_syn_am_density[i] = den_prox_synapse_amount[i] / (50 * primary_dendrites)
            den_dist_syn_am_density[i] = den_dist_synapse_amount[i]/ (den_length - 100 * primary_dendrites)
            #median synapse size and density in µm2/µm
            syn_sizes = syn_dict_cell["syn_sizes"]
            axo_median_syn_size[i] = np.median(syn_sizes[axon_syn_inds])
            den_median_syn_size[i] = np.median(syn_sizes[den_syn_inds])
            axo_prox_median_syn_size[i] = np.median(syn_sizes[axo_prox_syn_inds])
            axo_dist_median_syn_size[i] = np.median(syn_sizes[axo_dist_syn_inds])
            den_prox_median_syn_size[i] = np.median(syn_sizes[den_prox_syn_inds])
            den_dist_median_syn_size[i] = np.median(mito_volumes[den_dist_mito_inds])
            axo_syn_size_density[i] = np.sum(syn_sizes[axon_syn_inds]) / axon_length
            den_syn_size_density[i] = np.sum(syn_sizes[den_syn_inds]) / den_length
            axo_prox_syn_size_density[i] = np.sum(syn_sizes[axo_prox_syn_inds]) / 50
            axo_dist_syn_size_density[i] = np.sum(syn_sizes[axo_dist_syn_inds]) / (axon_length - 100)
            den_prox_syn_size_density[i] = np.sum(syn_sizes[den_prox_syn_inds]) / (50 * primary_dendrites)
            den_dist_syn_size_density[i] = np.sum(syn_sizes[den_dist_syn_inds]) / (
                        den_length - 100 * primary_dendrites)
            #vesicle amount and amount density
            axo_vc_ids = vc_dict_cell["axo_ids"]
            axo_vc_amount[i] = len(axo_vc_ids)
            axo_prox_vc_ids = vc_dict_cell["prox_ids"]
            axo_prox_vc_inds = np.in1d(axo_vc_ids, axo_prox_vc_ids)
            axo_prox_vc_amount[i] = len(axo_prox_vc_ids)
            axo_dist_vc_ids = vc_dict_cell["dist_ids"]
            axo_dist_vc_amount[i] = len(axo_dist_vc_ids)
            axo_dist_vc_inds = np.in1d(axo_vc_ids, axo_dist_vc_ids)
            axo_vc_amount_density[i] = axo_vc_amount[i]/axon_length
            axo_prox_vc_am_density[i] = axo_prox_vc_amount[i]/50
            axo_dist_vc_am_density[i] = axo_dist_vc_amount[i]/ (axon_length - 100)
            #vesicle median volume, volume density
            axo_vc_vol = vc_dict_cell["vc_sizes"]
            axo_median_vc_volume[i] = np.median(axo_vc_vol)
            axo_prox_vc_vol = axo_vc_vol[axo_prox_vc_inds]
            axo_dist_vc_vol = axo_vc_vol[axo_dist_vc_inds]
            axo_prox_median_vc_volume[i] = np.median(axo_prox_vc_vol)
            axo_dist_median_vc_volume[i] = np.median(axo_dist_vc_vol)
            axo_vc_vol_density[i] = np.sum(axo_vc_vol) / axon_length
            axo_prox_vc_vol_density[i] = np.sum(axo_prox_vc_vol) / 50
            axo_dist_vc_vol_density[i] = np.sum(axo_dist_vc_vol) / (axon_length - 100)
            #vesicle clouse distance 2 synapse, clsoe 2 synapse ratio
            vc_dist2syn = vc_dict_cell["distance2synapse"]
            axo_median_vc_dist2syn[i] = np.median(vc_dist2syn)
            axo_prox_vc_dist2syn = vc_dist2syn[axo_prox_vc_inds]
            axo_dist_vc_dist2syn = vc_dist2syn[axo_dist_vc_inds]
            axo_prox_median_vc_dist2syn[i] = np.median(axo_prox_vc_dist2syn)
            axo_dist_median_vc_dist2syn[i] = np.median(axo_dist_vc_dist2syn)
            close2syn_inds = vc_dist2syn <= close2syn_thresh
            close2syn_ids = axo_vc_ids[close2syn_inds]
            axo_vc_close2syn_ratio[i] = len(close2syn_ids)/ len(axo_vc_ids)
            prox_close2syn_inds = axo_prox_vc_dist2syn <= close2syn_thresh
            prox_close2syn_ids = axo_prox_vc_ids[prox_close2syn_inds]
            if len(axo_prox_vc_ids) == 0:
                axo_prox_vc_close2syn_ratio[i] = 0
            else:
                axo_prox_vc_close2syn_ratio[i] = len(prox_close2syn_ids)/ len(axo_prox_vc_ids)
            dist_close2syn_inds = axo_dist_vc_dist2syn <= close2syn_thresh
            dist_close2syn_ids = axo_dist_vc_ids[dist_close2syn_inds]
            if len(axo_dist_vc_ids) == 0:
                axo_dist_vc_close2syn_ratio[i] = 0
            else:
                axo_dist_vc_close2syn_ratio[i] = len(dist_close2syn_ids)/ len(axo_dist_vc_ids)
            
        nonzero_inds = spine_density > 0
        cellids = cellids[nonzero_inds]
        spine_density = spine_density[nonzero_inds]

        #write resulting arrays to dictionary to save as pkl for plotting changes
        axo_synapse_dict = {"syn amount density": axo_syn_amount_density,
                            "prox syn amount density": axo_prox_syn_am_density,
                            "dist syn amount density": axo_dist_syn_am_density,
                            "syn amount": axo_synapse_amount, "prox syn amount": axo_prox_synapse_amount, "dist syn amount": axo_dist_synapse_amount,
                            "syn size density": axo_syn_size_density,
                            "prox syn size density": axo_prox_syn_size_density,
                            "dist syn size density": axo_dist_syn_size_density,
                            "median syn size": axo_median_syn_size, "prox median syn size": axo_prox_median_syn_size,
                            "dist median syn size": axo_dist_median_syn_size, "spine density": spine_density, "cellids": cellids}
        for key in axo_synapse_dict.keys():
            if key == "cellids" or key == "spine density":
                continue
            axo_synapse_dict[key]= axo_synapse_dict[key][nonzero_inds]
        den_synapse_dict = {"syn amount density": den_syn_amount_density,
                            "prox syn amount density": den_prox_syn_am_density,
                            "dist syn amount density": den_dist_syn_am_density,
                            "syn amount": den_synapse_amount, "prox syn amount": den_prox_synapse_amount,
                            "dist syn amount": den_dist_synapse_amount,
                            "syn size density": den_syn_size_density,
                            "prox syn size density": den_prox_syn_size_density,
                            "dist syn size density": den_dist_syn_size_density,
                            "median syn size": den_median_syn_size, "prox median syn size": den_prox_median_syn_size,
                            "dist median syn size": den_dist_median_syn_size, "spine density": spine_density, "cellids": cellids}
        for key in den_synapse_dict.keys():
            if key == "cellids" or key == "spine density":
                continue
            den_synapse_dict[key]= den_synapse_dict[key][nonzero_inds]
        axo_mito_dict = {"mito amount density": axo_mito_amount_density,
                            "prox mito amount density": axo_prox_mito_amount_density,
                            "dist mito amount density": axo_dist_mito_amount_density,
                            "mito amount": axo_mito_amount, "prox mito amount": axo_prox_mito_amount,
                            "dist mito amount": axo_dist_mito_amount,
                            "mito length density": axo_mito_length_density,
                            "prox mito length density": axo_prox_mito_length_density,
                            "dist mito length density": axo_dist_mito_length_density,
                            "median mito length": axo_median_mito_length, "prox median mito length": axo_prox_median_mito_length,
                            "dist median mito length": axo_dist_median_mito_length, "mito volume density": axo_mito_volume_density,
                            "prox mito volume density": axo_prox_mito_volume_density,
                            "dist mito volume density": axo_dist_mito_volume_density,
                            "median mito volume": axo_median_mito_volume, "prox median mito volume": axo_prox_median_mito_volume,
                            "dist median mito volume": axo_dist_median_mito_volume, "cellids": cellids}
        for key in axo_mito_dict.keys():
            if key == "cellids":
                continue
            axo_mito_dict[key]= axo_mito_dict[key][nonzero_inds]
        den_mito_dict = {"mito amount density": den_mito_amount_density,
                         "prox mito amount density": den_prox_mito_amount_density,
                         "dist mito amount density": den_dist_mito_amount_density,
                         "mito amount": den_mito_amount, "prox mito amount": den_prox_mito_amount,
                         "dist mito amount": den_dist_mito_amount,
                         "mito length density": den_mito_length_density,
                         "prox mito length density": den_prox_mito_length_density,
                         "dist mito length density": den_dist_mito_length_density,
                         "median mito length": den_median_mito_length,
                         "prox median mito length": den_prox_median_mito_length,
                         "dist median mito length": den_dist_median_mito_length,
                         "mito volume density": den_mito_volume_density,
                         "prox mito volume density": den_prox_mito_volume_density,
                         "dist mito volume density": den_dist_mito_volume_density,
                         "median mito volume": den_median_mito_volume,
                         "prox median mito volume": den_prox_median_mito_volume,
                         "dist median mito volume": den_dist_median_mito_volume, "cellids":cellids}
        for key in den_mito_dict.keys():
            if key == "cellids":
                continue
            den_mito_dict[key]= den_mito_dict[key][nonzero_inds]
        axo_vc_dict = {"vc amount density": axo_vc_amount_density, "vc amount": axo_vc_amount_density,
                       "prox vc amount": axo_prox_vc_amount, "prox vc amount density": axo_prox_vc_am_density,
                       "dist vc amount": axo_dist_vc_amount, "dist vc amount density": axo_dist_vc_am_density,
                       "vc median volume": axo_median_vc_volume, "vc volume density": axo_vc_vol_density,
                       "prox vc median volume": axo_prox_median_vc_volume, "prox vc volume density": axo_prox_vc_vol_density,
                       "dist vc median volume": axo_dist_median_vc_volume, "dist vc volume density": axo_dist_vc_vol_density,
                       "vc median distance to syn": axo_median_vc_dist2syn, " prox vc median distance to syn": axo_prox_median_vc_dist2syn,
                       "dist vc median distance to syn": axo_dist_median_vc_dist2syn, "vc close to syn ratio": axo_vc_close2syn_ratio,
                       "prox vc close to syn ratio": axo_prox_vc_close2syn_ratio, "dist vc close to syn ratio": axo_dist_vc_close2syn_ratio,
                       "cellids": cellids}
        for key in axo_vc_dict.keys():
            if key == "cellids":
                continue
            axo_vc_dict[key]= axo_vc_dict[key][nonzero_inds]
        write_obj2pkl("%s/%s_axo_synapse_dict.pkl" % (f_name, ct_dict[celltype]), axo_synapse_dict)
        write_obj2pkl("%s/%s_den_synapse_dict.pkl" % (f_name, ct_dict[celltype]), den_synapse_dict)
        write_obj2pkl("%s/%s_axo_mito_dict.pkl" % (f_name, ct_dict[celltype]), axo_mito_dict)
        write_obj2pkl("%s/%s_den_mito_dict.pkl" % (f_name, ct_dict[celltype]), den_mito_dict)
        write_obj2pkl("%s/%s_axo_vc_dict.pkl" % (f_name, ct_dict[celltype]), axo_vc_dict)

        axo_dist_mito_threshold = 0.08
        axo_low_dist_mito_am_density_inds = np.where(axo_mito_dict["dist mito amount density"] <= axo_dist_mito_threshold)[0]
        axo_high_dist_mito_am_density_inds = np.where(axo_mito_dict["dist mito amount density"] > axo_dist_mito_threshold)[0]
        axo_prox_mito_threshold = 0.5
        axo_low_prox_mito_am_density_inds = np.where(axo_mito_dict["prox mito amount density"]<= axo_prox_mito_threshold)[0]
        axo_high_prox_mito_am_density_inds = np.where(axo_mito_dict["prox mito amount density"] > axo_prox_mito_threshold)[0]
        axo_mito_amden_median = np.median(axo_mito_dict["mito amount density"])
        axo_low_mito_am_density_inds = np.where(axo_mito_dict["mito amount density"] <= axo_mito_amden_median)[0]
        axo_high_mito_am_density_inds = np.where(axo_mito_dict["mito amount density"] > axo_mito_amden_median)[0]


        # violin plot with upper and lower half

        high_low_mito_am_den_spiness = pd.DataFrame()
        high_low_mito_am_den_spiness["cellids"] = cellids
        high_low_mito_am_den_spiness["spiness"] = spine_density
        high_low_mito_am_den_spiness["axo mito amount density distal"] = str()
        high_low_mito_am_den_spiness.loc[
            axo_low_dist_mito_am_density_inds, "axo mito amount density distal"] = "low mito amount density"
        high_low_mito_am_den_spiness.loc[
            axo_high_dist_mito_am_density_inds, "axo mito amount density distal"] = "high mito amount density"
        high_low_mito_am_den_spiness["axo mito amount density proximal"] = str()
        high_low_mito_am_den_spiness.loc[
            axo_low_prox_mito_am_density_inds, "axo mito amount density proximal"] = "low mito amount density"
        high_low_mito_am_den_spiness.loc[
            axo_high_prox_mito_am_density_inds, "axo mito amount density proximal"] = "high mito amount density"
        high_low_mito_am_den_spiness["axo synapse amount density"] = axo_synapse_dict["syn amount density"]
        high_low_mito_am_den_spiness["axo synapse size density"] = axo_synapse_dict["syn size density"]
        high_low_mito_am_den_spiness.loc[
            axo_high_mito_am_density_inds, "axo mito amount density"] = "high mito amount density"
        high_low_mito_am_den_spiness.loc[
            axo_low_mito_am_density_inds, "axo mito amount density"] = "low mito amount density"
        write_obj2pkl("%s/axo_high_low_mito_am_den_spines_df.pkl" % f_name, high_low_mito_am_den_spiness)
        high_low_mito_am_den_spiness.to_csv("%s/axo_high_low_mito_am_den_spiness_%s.csv" % (f_name, ct_dict[celltype]))

        axo_dist_stat, axo_dist_p_val = ranksums(spine_density[axo_low_dist_mito_am_density_inds],
                                                 spine_density[axo_high_dist_mito_am_density_inds])
        axo_prox_stat, axo_prox_p_val = ranksums(spine_density[axo_low_prox_mito_am_density_inds],
                                                 spine_density[axo_high_prox_mito_am_density_inds])
        axo_dist_synam_stat, axo_dist_synam_p_val = ranksums(
            axo_synapse_dict["syn amount density"][axo_low_dist_mito_am_density_inds],
            axo_synapse_dict["syn amount density"][axo_high_dist_mito_am_density_inds])
        axo_prox_synam_stat, axo_prox_synam_p_val = ranksums(
            axo_synapse_dict["syn amount density"][axo_low_prox_mito_am_density_inds],
            axo_synapse_dict["syn amount density"][axo_high_prox_mito_am_density_inds])
        axo_dist_synsi_stat, axo_dist_synsi_p_val = ranksums(
            axo_synapse_dict["syn size density"][axo_low_dist_mito_am_density_inds],
            axo_synapse_dict["syn size density"][axo_high_dist_mito_am_density_inds])
        axo_prox_synsi_stat, axo_prox_synsi_p_val = ranksums(
            axo_synapse_dict["syn size density"][axo_low_prox_mito_am_density_inds],
            axo_synapse_dict["syn size density"][axo_high_prox_mito_am_density_inds])
        axo_mito_spiness_stat, axo_mito_spiness_p_val = ranksums(spine_density[axo_low_mito_am_density_inds],
                                                                 spine_density[axo_high_mito_am_density_inds])
        axo_mito_synam_stat, axo_mito_synam_p_val = ranksums(axo_synapse_dict["syn amount density"][axo_low_mito_am_density_inds],
                                                             axo_synapse_dict["syn amount density"][axo_high_mito_am_density_inds])
        axo_mito_synsi_stat, axo_mito_synsi_p_val = ranksums(axo_synapse_dict["syn size density"][axo_low_mito_am_density_inds],
                                                             axo_synapse_dict["syn size density"][axo_high_mito_am_density_inds])
        ranksum_results = pd.DataFrame(
            columns=["high vs low spiness distal mitos", "high vs low spiness proximal mitos",
                     "high vs low syn amount proximal mitos",
                     "high vs low syn amount distal mitos", "high vs low syn size distal mitos",
                     "high vs low syn size proximal mitos", "high vs low spiness mitos",
                     "high vs low syn amount mitos", "high vs low syn size mitos"], index=["stats", "p value"])
        ranksum_results["high vs low spiness distal mitos"]["stats"] = axo_dist_stat
        ranksum_results["high vs low spiness distal mitos"]["p value"] = axo_dist_p_val
        ranksum_results["high vs low spiness proximal mitos"]["stats"] = axo_prox_stat
        ranksum_results["high vs low spiness proximal mitos"]["p value"] = axo_prox_p_val
        ranksum_results["high vs low syn amount distal mitos"]["stats"] = axo_dist_synam_stat
        ranksum_results["high vs low syn amount distal mitos"]["p value"] = axo_dist_synam_p_val
        ranksum_results["high vs low syn amount proximal mitos"]["stats"] = axo_prox_synam_stat
        ranksum_results["high vs low syn amount proximal mitos"]["p value"] = axo_prox_synam_p_val
        ranksum_results["high vs low syn size distal mitos"]["stats"] = axo_dist_synsi_stat
        ranksum_results["high vs low syn size distal mitos"]["p value"] = axo_dist_synsi_p_val
        ranksum_results["high vs low syn size proximal mitos"]["stats"] = axo_prox_synsi_stat
        ranksum_results["high vs low syn size proximal mitos"]["p value"] = axo_prox_synsi_p_val
        ranksum_results["high vs low spiness mitos"]["stats"] = axo_mito_spiness_stat
        ranksum_results["high vs low spiness mitos"]["p value"] = axo_mito_spiness_p_val
        ranksum_results["high vs low syn amount mitos"]["stats"] = axo_mito_synam_stat
        ranksum_results["high vs low syn amount mitos"]["p value"] = axo_mito_synam_p_val
        ranksum_results["high vs low syn size mitos"]["stats"] = axo_mito_synsi_stat
        ranksum_results["high vs low syn size mitos"]["p value"] = axo_mito_synsi_p_val
        ranksum_results.to_csv("%s/wilcoxon_ranksum_results.csv" % f_name)


        celltime = time.time() - start
        print("%.2f sec for processing cells" % celltime)
        time_stamps.append(time.time())
        step_idents.append('mitochondrial parameters per cell')
        log.info("Step 2/2 plot graphs per celltype")
        #violin plots
        spiness_pal = {"high mito amount density": "darkturquoise",
                       "low mito amount density": "gray"}

        sns.violinplot(x="axo mito amount density distal", y="spiness", data=high_low_mito_am_den_spiness,
                      palette=spiness_pal, inner="box")
        sns.stripplot(x="axo mito amount density distal", y="spiness", data=high_low_mito_am_den_spiness, color="black",
                      alpha=0.1)
        plt.title('axonal mito amount density vs spiness in %s > 100 µm from soma' % ct_dict[celltype])
        filename = ("%s/vio_axo_dist_spine_dist_mitoamden_%s.png" % (f_name, ct_dict[celltype]))
        plt.savefig(filename)
        plt.close()

        sns.violinplot(x="axo mito amount density proximal", y="spiness", data=high_low_mito_am_den_spiness,
                       palette=spiness_pal, inner="box")
        sns.stripplot(x="axo mito amount density proximal", y="spiness", data=high_low_mito_am_den_spiness,
                      color="black",
                      alpha=0.1)
        plt.title('axonal mito amount density vs spiness in %s < 50 µm from soma' % ct_dict[celltype])
        filename = ("%s/vio_axo_dist_spine_prox_mitoamden_%s.png" % (f_name, ct_dict[celltype]))
        plt.savefig(filename)
        plt.close()

        sns.violinplot(x="axo mito amount density distal", y="axo synapse amount density",
                       data=high_low_mito_am_den_spiness,
                       palette=spiness_pal, inner="box")
        sns.stripplot(x="axo mito amount density distal", y="axo synapse amount density",
                      data=high_low_mito_am_den_spiness, color="black",
                      alpha=0.1)
        plt.title(
            'axonal mito amount density vs axo synapse amount density in %s > 100 µm from soma' % ct_dict[celltype])
        filename = ("%s/vio_axo_syn_amden_dist_mitoamden_%s.png" % (f_name, ct_dict[celltype]))
        plt.savefig(filename)
        plt.close()

        sns.violinplot(x="axo mito amount density proximal", y="axo synapse amount density",
                       data=high_low_mito_am_den_spiness,
                       palette=spiness_pal, inner="box")
        sns.stripplot(x="axo mito amount density proximal", y="axo synapse amount density",
                      data=high_low_mito_am_den_spiness, color="black",
                      alpha=0.1)
        plt.title(
            'axonal mito amount density vs axo synapse amount density in %s < 50 µm from soma' % ct_dict[celltype])
        filename = ("%s/vio_axo_syn_amden_prox_mitoamden_%s.png" % (f_name, ct_dict[celltype]))
        plt.savefig(filename)
        plt.close()

        sns.violinplot(x="axo mito amount density distal", y="axo synapse size density",
                       data=high_low_mito_am_den_spiness,
                       palette=spiness_pal, inner="box")
        sns.stripplot(x="axo mito amount density distal", y="axo synapse size density",
                      data=high_low_mito_am_den_spiness, color="black",
                      alpha=0.1)
        plt.title('axonal mito amount density vs axo synapse size density in %s > 100 µm from soma' % ct_dict[celltype])
        filename = ("%s/vio_axo_syn_siden_dist_mitoamden_%s.png" % (f_name, ct_dict[celltype]))
        plt.savefig(filename)
        plt.close()

        sns.violinplot(x="axo mito amount density proximal", y="axo synapse size density",
                       data=high_low_mito_am_den_spiness,
                       palette=spiness_pal, inner="box")
        sns.stripplot(x="axo mito amount density proximal", y="axo synapse size density",
                      data=high_low_mito_am_den_spiness, color="black",
                      alpha=0.1)
        plt.title('axonal mito amount density vs axo synapse size density in %s < 50 µm from soma' % ct_dict[celltype])
        filename = ("%s/vio_axo_syn_siden_prox_mitoamden_%s.png" % (f_name, ct_dict[celltype]))
        plt.savefig(filename)
        plt.close()

        sns.violinplot(x="axo mito amount density", y="spiness",
                       data=high_low_mito_am_den_spiness,
                       palette=spiness_pal, inner="box")
        sns.stripplot(x="axo mito amount density", y="spiness", data=high_low_mito_am_den_spiness,
                      color="black",
                      alpha=0.1)
        plt.title('axonal mito amount density vs spine density in %s' % ct_dict[celltype])
        filename = ("%s/vio_spiness_axo_mitoamden_%s.png" % (f_name, ct_dict[celltype]))
        plt.savefig(filename)
        plt.close()

        sns.violinplot(x="axo mito amount density", y="axo synapse amount density",
                       data=high_low_mito_am_den_spiness,
                       palette=spiness_pal, inner="box")
        sns.stripplot(x="axo mito amount density", y="axo synapse amount density", data=high_low_mito_am_den_spiness,
                      color="black",
                      alpha=0.1)
        plt.title('axonal mito amount density vs axo synapse amount density in %s' % ct_dict[celltype])
        filename = ("%s/vio_axo_syn_amden_axo_mitoamden_%s.png" % (f_name, ct_dict[celltype]))
        plt.savefig(filename)
        plt.close()

        sns.violinplot(x="axo mito amount density", y="axo synapse size density",
                       data=high_low_mito_am_den_spiness,
                       palette=spiness_pal, inner="box")
        sns.stripplot(x="axo mito amount density", y="axo synapse size density", data=high_low_mito_am_den_spiness,
                      color="black",
                      alpha=0.1)
        plt.title('axonal mito amount density vs axo synapse size density in %s' % ct_dict[celltype])
        filename = ("%s/vio_axo_syn_siden_axo_mitoamden_%s.png" % (f_name, ct_dict[celltype]))
        plt.savefig(filename)
        plt.close()


        #plot each of the parameters as distplot on their own
        #spine density
        sns.distplot(spine_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/spineden_%s.png' % (f_name, ct_dict[celltype]))
        plt.title('Spine density in %s dendrites' % (ct_dict[celltype]))
        plt.xlabel('amount of spines per µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        # distplots as loop over keys for synaptic and mitochondrial parameters
        # also make scatterplots
        for key in axo_synapse_dict.keys():
            if key == "spine density" or key == "cellids":
                continue
            sns.distplot(axo_synapse_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                         kde=False)
            avg_filename = ('%s/axo_%s_%s.png' % (f_name, key, ct_dict[celltype]))
            plt.title('%s in %s axons' % (key, ct_dict[celltype]))
            if "amount" in key:
                if "density" in key:
                    plt.xlabel("synapse amount per µm")
                else:
                    plt.xlabel("amount of synapses")
            else:
                if "density" in key:
                    plt.xlabel("synapse size density [µm²/µm]")
                else:
                    plt.xlabel("synapse size [µm²]")
            plt.ylabel('count of cells')
            plt.savefig(avg_filename)
            plt.close()
            if "density" not in key:
                continue
            #plot densities as scatter plots against spine density
            plt.scatter(x=spine_density, y=axo_synapse_dict[key], c="black", alpha=0.7)
            avg_filename = ('%s/spineden_vs_axo_%s_%s.png' % (f_name, key, ct_dict[celltype]))
            plt.title(
                'Spine density vs axon %s in %s' % (key, ct_dict[celltype]))
            plt.xlabel('amount of spines per µm')
            if "amount" in key:
                plt.ylabel('synapse amount per µm')
            else:
                plt.ylabel("synapse size density [µm²/µm]")
            plt.savefig(avg_filename)
            plt.close()
            #plot density against vesicle cloud parameters
            for vc_key in axo_vc_dict.keys():
                if "density" not in vc_key or "syn" not in vc_key:
                    continue
                if "distal" in key and "distal" not in vc_key:
                    continue
                if "proximal" in key and "proximal" not in vc_key:
                    continue
                plt.scatter(x=axo_synapse_dict[key], y=axo_vc_dict[vc_key], c="black", alpha=0.7)
                avg_filename = ('%s/axo_%s_vs_axo_%s_%s.png' % (f_name, key,vc_key, ct_dict[celltype]))
                plt.title(
                    'Axon %s vs %s in %s' % (key,vc_key, ct_dict[celltype]))
                if "amount" in key:
                    plt.xlabel('amount of synapses per µm')
                    if "amount" in vc_key:
                        plt.ylabel('vesicle cloud amount per µm')
                else:
                    plt.xlabel("synapse size density [µm²/µm]")
                    if "volume" in vc_key:
                        plt.ylabel("vesicle cloud volume density [µm³/µm]")
                if "distance" in vc_key:
                    plt.ylabel("distance from vc to closest syn [µm]")
                else:
                    plt.ylabel("ratio of vc close to syn (> %.2f µm)" % close2syn_thresh)
                plt.savefig(avg_filename)
                plt.close()
            #plot syn densities vs mito densities
            for m_key in axo_mito_dict.keys():
                if "density" not in m_key:
                    continue
                if "distal" in key and "distal" not in m_key:
                    continue
                if "proximal" in key and "proximal" not in m_key:
                    continue
                plt.scatter(x=axo_synapse_dict[key], y=axo_mito_dict[m_key], c="black", alpha=0.7)
                avg_filename = ('%s/axo_%s_vs_axo_%s_%s.png' % (f_name, key, m_key, ct_dict[celltype]))
                plt.title(
                    'Axon %s vs %s in %s' % (key, m_key, ct_dict[celltype]))
                if "amount" in key:
                    plt.xlabel('amount of synapses per µm')
                    if "amount" in m_key:
                        plt.ylabel('mitochondria amount per µm')
                else:
                    plt.xlabel("synapse size density [µm²/µm]")
                    if "volume" in m_key:
                        plt.ylabel("mitochondria volume density [µm³/µm]")
                    else:
                        plt.ylabel("mitochondria length density [µm/µm]")
                plt.savefig(avg_filename)
                plt.close()


        for key in den_synapse_dict.keys():
            if key == "spine density" or key == "cellids":
                continue
            sns.distplot(den_synapse_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                         kde=False)
            avg_filename = ('%s/den_%s_%s.png' % (f_name, key, ct_dict[celltype]))
            plt.title('%s in %s dendrites' % (key, ct_dict[celltype]))
            if "amount" in key:
                if "density" in key:
                    plt.xlabel("synapse amount per µm")
                else:
                    plt.xlabel("amount of synapses")
            else:
                if "density" in key:
                    plt.xlabel("synapse size density [µm²/µm]")
                else:
                    plt.xlabel("synapse size [µm²]")
            plt.ylabel('count of cells')
            plt.savefig(avg_filename)
            plt.close()
            if "density" not in key:
                continue
            #plot densities as scatter plots against spine density
            plt.scatter(x=spine_density, y=den_synapse_dict[key], c="black", alpha=0.7)
            avg_filename = ('%s/spineden_vs_den_%s_%s.png' % (f_name, key, ct_dict[celltype]))
            plt.title(
                'Spine density vs dendrite %s in %s' % (key, ct_dict[celltype]))
            plt.xlabel('amount of spines per µm')
            if "amount" in key:
                plt.ylabel('synapse amount per µm')
            else:
                plt.ylabel("synapse size density [µm²/µm]")
            plt.savefig(avg_filename)
            plt.close()
            # plot syn densities vs mito densities
            for m_key in axo_mito_dict.keys():
                if "density" not in m_key:
                    continue
                if "distal" in key and "distal" not in m_key:
                    continue
                if "proximal" in key and "proximal" not in m_key:
                    continue
                plt.scatter(x=den_synapse_dict[key], y=den_mito_dict[m_key], c="black", alpha=0.7)
                avg_filename = ('%s/den_%s_vs_den_%s_%s.png' % (f_name, key, m_key, ct_dict[celltype]))
                plt.title(
                    'Dendrite %s vs %s in %s' % (key, m_key, ct_dict[celltype]))
                if "amount" in key:
                    plt.xlabel('amount of synapses per µm')
                    if "amount" in m_key:
                        plt.ylabel('mitochondria amount per µm')
                else:
                    plt.xlabel("synapse size density [µm²/µm]")
                    if "volume" in m_key:
                        plt.ylabel("mitochondria volume density [µm³/µm]")
                    else:
                        plt.ylabel("mitochondria length density [µm/µm]")
                plt.savefig(avg_filename)
                plt.close()

        for key in axo_mito_dict.keys():
            if key == "cellids":
                continue
            sns.distplot(axo_mito_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                         kde=False)
            avg_filename = ('%s/axo_%s_%s.png' % (f_name, key, ct_dict[celltype]))
            plt.title('%s in %s in axons' % (key, ct_dict[celltype]))
            if "amount" in key:
                if "density" in key:
                    plt.xlabel("mitochondria amount per µm")
                else:
                    plt.xlabel("amount of mitochondria")
            elif "length" in key:
                if "density" in key:
                    plt.xlabel("mitochondria length density [µm/µm]")
                else:
                    plt.xlabel("mitochondria length [µm]")
            else:
                if "density" in key:
                    plt.xlabel("mitochondria volume density [µm³/µm]")
                else:
                    plt.xlabel("mitochondria volume [µm³]")
            plt.ylabel('count of cells')
            plt.savefig(avg_filename)
            plt.close()
            if "density" not in key:
                continue
            #plot densities as scatter plots against spine density
            plt.scatter(x=spine_density, y=axo_mito_dict[key], c="black", alpha=0.7)
            avg_filename = ('%s/spineden_vs_axo_%s_%s.png' % (f_name, key, ct_dict[celltype]))
            plt.title(
                'Spine density vs axon %s in %s' % (key, ct_dict[celltype]))
            plt.xlabel('amount of spines per µm')
            if "amount" in key:
                plt.ylabel('mitochondria amount per µm')
            elif "length" in key:
                plt.ylabel("mitochondria length density [µm/µm]")
            else:
                plt.ylabel("mitochondria volume density [µm³/µm]")
            plt.savefig(avg_filename)
            plt.close()
            # plot density against vesicle cloud parameters
            for vc_key in axo_vc_dict.keys():
                if "density" not in vc_key or "syn" not in vc_key:
                    continue
                if "distal" in key and "distal" not in vc_key:
                    continue
                if "proximal" in key and "proximal" not in vc_key:
                    continue
                plt.scatter(x=axo_mito_dict[key], y=axo_vc_dict[vc_key], c="black", alpha=0.7)
                avg_filename = ('%s/axo_%s_vs_axo_%s_%s.png' % (f_name, key, vc_key, ct_dict[celltype]))
                plt.title(
                    'Axon %s vs %s in %s' % (key, vc_key, ct_dict[celltype]))
                if "amount" in key:
                    plt.xlabel('amount of mitochondria per µm')
                    if "amount" in vc_key:
                        plt.ylabel('vesicle cloud amount per µm')
                elif "length" in key:
                    plt.xlabel("mitochondria length density [µm/µm]")
                    if "volume" in vc_key:
                        plt.ylabel("vesicle cloud volume density [µm³/µm]")
                else:
                    plt.xlabel("mitochondria volume density [µm²/µm]")
                    if "volume" in vc_key:
                        plt.ylabel("vesicle cloud volume density [µm³/µm]")
                if "distance" in vc_key:
                    plt.ylabel("distance from vc to closest syn [µm]")
                else:
                    plt.ylabel("ratio of vc close to syn (< %.2f µm)" % close2syn_thresh)
                plt.savefig(avg_filename)
                plt.close()


        for key in den_mito_dict.keys():
            if key == "cellids":
                continue
            sns.distplot(den_mito_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                         kde=False)
            avg_filename = ('%s/den_%s_%s.png' % (f_name, key, ct_dict[celltype]))
            plt.title('%s in %s in dendrites' % (key, ct_dict[celltype]))
            if "amount" in key:
                if "density" in key:
                    plt.xlabel("mitochondria amount per µm")
                else:
                    plt.xlabel("amount of mitochondria")
            elif "length" in key:
                if "density" in key:
                    plt.xlabel("mitochondria length density [µm/µm]")
                else:
                    plt.xlabel("mitochondria length [µm]")
            else:
                if "density" in key:
                    plt.xlabel("mitochondria volume density [µm³/µm]")
                else:
                    plt.xlabel("mitochondria volume [µm³]")
            plt.ylabel('count of cells')
            plt.savefig(avg_filename)
            plt.close()
            if "density" not in key:
                continue
            #plot densities as scatter plots against spine density
            plt.scatter(x=spine_density, y=den_mito_dict[key], c="black", alpha=0.7)
            avg_filename = ('%s/spineden_vs_den_%s_%s.png' % (f_name, key, ct_dict[celltype]))
            plt.title(
                'Spine density vs dendrite %s in %s' % (key, ct_dict[celltype]))
            plt.xlabel('amount of spines per µm')
            if "amount" in key:
                plt.ylabel('mitochondria amount per µm')
            elif "length" in key:
                plt.ylabel("mitochondria length density [µm/µm]")
            else:
                plt.ylabel("mitochondria volume density [µm³/µm]")
            plt.savefig(avg_filename)
            plt.close()

        for key in axo_vc_dict.keys():
            if key == "cellids":
                continue
            sns.distplot(axo_vc_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                         kde=False)
            avg_filename = ('%s/axo_%s_%s.png' % (f_name, key, ct_dict[celltype]))
            plt.title('%s in %s in dendrites' % (key, ct_dict[celltype]))
            if "amount" in key:
                if "density" in key:
                    plt.xlabel("vesicle cloud amount per µm")
                else:
                    plt.xlabel("amount of vesicle clouds")
            elif "volume" in key:
                if "density" in key:
                    plt.xlabel("vc volume density [µm³/µm]")
                else:
                    plt.xlabel("vc volume [µm³]")
            elif "distance" in key:
                plt.xlabel("vc distance to synapse [µm]")
            else:
                plt.xlabel("ratio of vcs close to synapses (< %.2f µm)" % close2syn_thresh)
            plt.ylabel('count of cells')
            plt.savefig(avg_filename)
            plt.close()
            if "density" not in key:
                continue
            #plot densities as scatter plots against spine density
            plt.scatter(x=spine_density, y=axo_vc_dict[key], c="black", alpha=0.7)
            avg_filename = ('%s/spineden_vs_axo_%s_%s.png' % (f_name, key, ct_dict[celltype]))
            plt.title(
                'Spine density vs axon %s in %s' % (key, ct_dict[celltype]))
            plt.xlabel('amount of spines per µm')
            if "amount" in key:
                plt.ylabel('vesicle cloud amount per µm')
            else:
                plt.ylabel("vesicle cloud volume density [µm³/µm]")
            plt.savefig(avg_filename)
            plt.close()


        time_stamps = [time.time()]
        step_idents = ['t-3']
        plottime = time.time() - celltime
        print("%.2f min, %.2f sec for plotting" % (plottime // 60, plottime % 60))

        log.info('fast mito, syn analysis in %s finished' %  ct_dict[celltype])


    ct_mito_syn_spiness_analysis(ssd, celltype = 5 , min_comp_len=200, syn_prob_thresh = 0.6, syn_size_thresh = 0.1, close2syn_thresh=0.5)
