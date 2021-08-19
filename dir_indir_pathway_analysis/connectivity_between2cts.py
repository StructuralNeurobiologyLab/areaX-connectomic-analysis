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
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)

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

    def synapses_between2cts(ssd, sd_synssv, celltype1, celltype2, full_cells = True, handpicked1 = True, handpicked2 = True,
                             min_comp_len = 100, min_syn_size = 0.1, syn_prob_thresh = 0.6):
        '''
        looks at basic connectivty parameters between two celltypes such as amount of synapses, average of synapses between cell types but also
        the average from one cell to the same other cell. Also looks at distribution of axo_dendritic synapses onto spines/shaft and the percentage of axo-somatic
        synapses. Uses cached synapse properties. Uses compartment_length per cell to ignore cells with not enough axon/dendrite
        :param ssd: super-segmentation dataset
        :param sd_synssv: segmentation dataset for synapses.
        :param celltype1, celltype2: celltypes to be compared. j0256: STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7,
    #                      FS=8, LTS=9, NGF=10
        :param full_cells: if True, load preprocessed full cells
        :param handpicked1, handpicked2: if True, load manually selected full cells. Can select for each celltype
        :param min_comp_len: minimum length for axon/dendrite to have to include cell in analysis
        :param min_syn_size: minimum size for synapses
        :param syn_prob_thresh: threshold for synapse probability
        :return:
        '''
        start = time.time()
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   10: "NGF"}
        f_name = "u/arother/bio_analysis_results/dir_indir_pathway_analysis/210818_j0251v3_syn_conn_%s_%s_mcl%i_sysi_%i_st_%i" % (
        ct_dict[celltype1], ct_dict[celltype2], min_comp_len, min_syn_size, syn_prob_thresh)
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('compartment volume estimation', log_dir=f_name + '/logs/')
        log.info("parameters: celltype1 = %s, celltype2 = %s, min_comp_length = %.i, min_syn_size = %i, syn_prob_thresh = %i" %
                 (ct_dict[celltype1], ct_dict[celltype2], min_comp_len, min_syn_size, syn_prob_thresh))
        time_stamps = [time.time()]
        step_idents = ['t-0']
        if full_cells:
            if handpicked1:
                cellids1 = load_pkl2obj(
                    "/wholebrain/scratch/arother/j0251v3_prep/handpicked_%.3s_arr.pkl" % ct_dict[celltype1])
            else:
                cellids1 = load_pkl2obj(
                    "/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_arr.pkl" % ct_dict[celltype1])
            if handpicked2:
                cellids2 = load_pkl2obj(
                    "/wholebrain/scratch/arother/j0251v3_prep/handpicked_%.3s_arr.pkl" % ct_dict[celltype2])
            else:
                cellids2 = load_pkl2obj(
                    "/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_arr.pkl" % ct_dict[celltype2])
        else:
            cellids1 = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == celltype1]
            cellids2 = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == celltype2]

        ct1_axon_length = np.zeros(len(cellids1))
        ct2_axon_length = np.zeros(len(cellids2))
        log.info("Step 1/4 Iterate over %s to check min_comp_len" % ct_dict[celltype1])
        for i, cell in enumerate(tqdm(ssd.get_super_segmentation_object(cellids1))):
            cell.load_skeleton()
            g = cell.weighted_graph(add_node_attr=('axoness_avg10000',))
            cell_axon_length = compartment_length_cell(cell, compartment=1, cell_graph=g)
            if cell_axon_length < min_comp_len:
                continue
            cell_den_length = compartment_length_cell(cell, compartment=0, cell_graph=g)
            if cell_den_length < min_comp_len:
                continue
            ct1_axon_length[i] = cell_axon_length

        ct1_axon_length = ct1_axon_length[ct1_axon_length > 0]
        cellids1 = cellids1[ct1_axon_length > 0]

        ct1time = time.time() - start
        print("%.2f sec for iterating through %s cells" % (ct_dict[celltype1], ct1time))
        time_stamps.append(time.time())
        step_idents.append('iterating over %s cells' % ct_dict[celltype1])

        log.info("Step 2/4 Iterate over %s to check min_comp_len" % ct_dict[celltype2])
        for i, cell in enumerate(tqdm(ssd.get_super_segmentation_object(cellids2))):
            cell.load_skeleton()
            g = cell.weighted_graph(add_node_attr=('axoness_avg10000',))
            cell_axon_length = compartment_length_cell(cell, compartment=1, cell_graph=g)
            if cell_axon_length < min_comp_len:
                continue
            cell_den_length = compartment_length_cell(cell, compartment=0, cell_graph=g)
            if cell_den_length < min_comp_len:
                continue
            ct2_axon_length[i] = cell_axon_length

        ct2_axon_length = ct2_axon_length[ct2_axon_length > 0]
        cellids2 = cellids2[ct2_axon_length > 0]

        ct2time = time.time() - ct1time
        print("%.2f sec for iterating through %s cells" % (ct_dict[celltype2], ct2time))
        time_stamps.append(time.time())
        step_idents.append('iterating over %s cells' % ct_dict[celltype2])

        log.info("Step 3/4 get synaptic connectivity parameters")
        log.info("Step 3a: prefilter synapse caches")
        # prepare synapse caches with synapse threshold
        syn_prob = sd_synssv.load_cached_data("syn_prob")
        m = syn_prob > syn_prob_thresh
        m_ids = sd_synssv.ids[m]
        m_axs = sd_synssv.load_cached_data("partner_axoness")[m]
        m_axs[m_axs == 3] = 1
        m_axs[m_axs == 4] = 1
        m_cts = sd_synssv.load_cached_data("partner_celltypes")[m]
        m_ssv_partners = sd_synssv.load_cached_data("neuron_partners")[m]
        m_sizes = sd_synssv.load_cached_data("mesh_area")[m] / 2
        m_spiness = sd_synssv.load_cached_data("partner_spiness")[m]
        #select only those of celltype1 and celltype2
        ct1_inds = np.any(m_cts == celltype1, axis=1)
        m_cts = m_cts[ct1_inds]
        m_ids = m_ids[ct1_inds]
        m_axs = m_axs[ct1_inds]
        m_ssv_partners = m_ssv_partners[ct1_inds]
        m_sizes = m_sizes[ct1_inds]
        m_spiness = m_spiness[ct1_inds]
        ct2_inds = np.any(m_cts == celltype2, axis=1)
        m_cts = m_cts[ct2_inds]
        m_ids = m_ids[ct2_inds]
        m_axs = m_axs[ct2_inds]
        m_ssv_partners = m_ssv_partners[ct2_inds]
        m_sizes = m_sizes[ct2_inds]
        m_spiness = m_spiness[ct2_inds]
        # filter those with size below min_syn_size
        size_inds = m_sizes > min_syn_size
        m_cts = m_cts[size_inds]
        m_ids = m_ids[size_inds]
        m_axs = m_axs[size_inds]
        m_ssv_partners = m_ssv_partners[size_inds]
        m_sizes = m_sizes[size_inds]
        m_spiness = m_spiness[size_inds]
        # only axo-axonic or axo-somatic synapses allowed
        axs_inds = np.any(m_axs == 1, axis=1)
        m_cts = m_cts[axs_inds]
        m_ids = m_ids[axs_inds]
        m_axs = m_axs[axs_inds]
        m_ssv_partners = m_ssv_partners[axs_inds]
        m_sizes = m_sizes[axs_inds]
        m_spiness = m_spiness[axs_inds]

        prepsyntime = time.time() - ct2time
        print("%.2f sec for preprocessing synapses" % prepsyntime)
        time_stamps.append(time.time())
        step_idents.append('preprocessing synapses')

        log.info("Step 3b: iterate over synapses to get synaptic connectivity parameters")
        for i, syn_id in enumerate(tqdm(m_ids)):
            syn_ax = m_axs[i]
            if syn_ax[0] == syn_ax[1]:  # no axo-axonic synapses
                continue
            #remove cells that are not in cellids:

            if syn_ax[0] == 1:
                ct1, ct2 = m_cts[i]
                ssv1, ssv2 = m_ssv_partners[i]
            else:
                ct2, ct1 = m_cts[i]
                ssv2, ssv1 = m_ssv_partners[i]








#write analysis to compare connectivity of two celltypes
# compare amount of synapses, average synapse area, average amount of synapses from one cell to another, average size of synapses from one cell to another
#also compare percentage onto spine vs shaft from axo_dendritic synapses
#amount and average size of soma synapses, percentage of soma synapses from totl