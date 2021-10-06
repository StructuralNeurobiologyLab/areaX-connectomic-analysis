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
    from collections import defaultdict
    import time
    from syconn.handler.config import initialize_logging
    from syconn.handler.basics import load_pkl2obj
    from tqdm import tqdm
    from syconn.handler.basics import write_obj2pkl
    from scipy.stats import ranksums
    from u.arother.bio_analysis.general.analysis_helper import compartment_length_cell
    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)



    def synapses_between2cts(ssd, sd_synssv, celltype1, celltype2, full_cells = True, handpicked1 = True, handpicked2 = True, percentile = 0, lower_percentile = False,
                             min_comp_len = 100, min_syn_size = 0.1, syn_prob_thresh = 0.6):
        '''
        looks at basic connectivty parameters between two celltypes such as amount of synapses, average of synapses between cell types but also
        the average from one cell to the same other cell. Also looks at distribution of axo_dendritic synapses onto spines/shaft and the percentage of axo-somatic
        synapses. Uses cached synapse properties. Uses compartment_length per cell to ignore cells with not enough axon/dendrite
        spiness values: 0 = dendritic shaft, 1 = spine head, 2 = spine neck, 3 = other
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
        f_name = "u/arother/bio_analysis_results/dir_indir_pathway_analysis/210923_j0251v3_syn_conn_%s_2_%s_mcl%i_sysi_%.2f_st_%.2f" % (
             ct_dict[celltype1], ct_dict[celltype2], min_comp_len, min_syn_size, syn_prob_thresh)
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('compartment volume estimation', log_dir=f_name + '/logs/')
        log.info("parameters: celltype1 = %s, celltype2 = %s, min_comp_length = %.i, min_syn_size = %.2f, syn_prob_thresh = %.2f" %
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

        ct1_inds = ct1_axon_length > 0
        ct1_axon_length = ct1_axon_length[ct1_inds]
        cellids1 = cellids1[ct1_inds]

        ct1time = time.time() - start
        print("%.2f sec for iterating through %s cells" % (ct1time, ct_dict[celltype1]))
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

        ct2_inds = ct2_axon_length > 0
        ct2_axon_length = ct2_axon_length[ct2_inds]
        cellids2 = cellids2[ct2_inds]

        ct2time = time.time() - ct1time
        print("%.2f sec for iterating through %s cells" % (ct2time, ct_dict[celltype2]))
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
        # only axo-dendritic or axo-somatic synapses allowed
        axs_inds = np.any(m_axs == 1, axis=1)
        m_cts = m_cts[axs_inds]
        m_ids = m_ids[axs_inds]
        m_axs = m_axs[axs_inds]
        m_ssv_partners = m_ssv_partners[axs_inds]
        m_sizes = m_sizes[axs_inds]
        m_spiness = m_spiness[axs_inds]
        den_so = np.array([0, 2])
        den_so_inds = np.any(np.in1d(m_axs, den_so).reshape(len(m_axs), 2),axis = 1)
        m_cts = m_cts[den_so_inds]
        m_ids = m_ids[den_so_inds]
        m_axs = m_axs[den_so_inds]
        m_ssv_partners = m_ssv_partners[den_so_inds]
        m_sizes = m_sizes[den_so_inds]
        m_spiness = m_spiness[den_so_inds]

        prepsyntime = time.time() - ct2time
        print("%.2f sec for preprocessing synapses" % prepsyntime)
        time_stamps.append(time.time())
        step_idents.append('preprocessing synapses')

        log.info("Step 3b: iterate over synapses to get synaptic connectivity parameters")
        ct2_2_ct1_syn_dict = {"cellids": cellids1, "syn amount": np.zeros(len(cellids1)), "sum syn size": np.zeros(len(cellids1)),
                        "amount shaft syn": np.zeros(len(cellids1)), "sum size shaft syn": np.zeros(len(cellids1)),
                              "amount spine head syn": np.zeros(len(cellids1)),
                              "sum size spine head syn": np.zeros(len(cellids1)),
                              "amount soma syn": np.zeros(len(cellids1)), "sum size soma syn": np.zeros(len(cellids1)),
                             "amount spine neck syn": np.zeros(len(cellids1)),
                              "sum size spine neck syn": np.zeros(len(cellids1))}
        ct2_2_ct1_percell_syn_amount = np.zeros((len(cellids1), len(cellids2))).astype(float)
        ct2_2_ct1_percell_syn_size = np.zeros((len(cellids1), len(cellids2))).astype(float)
        ct1_2_ct2_syn_dict = {"cellids": cellids2, "syn amount": np.zeros(len(cellids2)), "sum syn size": np.zeros(len(cellids2)),
                        "amount shaft syn": np.zeros(len(cellids2)), "sum size shaft syn": np.zeros(len(cellids2)),
                              "amount spine head syn": np.zeros(len(cellids2)),
                              "sum size spine head syn": np.zeros(len(cellids2)),
                              "amount soma syn": np.zeros(len(cellids2)), "sum size soma syn": np.zeros(len(cellids2)),
                               "amount spine neck syn": np.zeros(len(cellids2)),
                              "sum size spine neck syn": np.zeros(len(cellids2))}
        ct1_2_ct2_percell_syn_amount = np.zeros((len(cellids2), len(cellids1))).astype(float)
        ct1_2_ct2_percell_syn_size = np.zeros((len(cellids2), len(cellids1))).astype(float)
        for i, syn_id in enumerate(tqdm(m_ids)):
            syn_ax = m_axs[i]
            #remove cells that are not in cellids:
            if not np.any(np.in1d(m_ssv_partners[i], cellids1)):
                continue
            if not np.any(np.in1d(m_ssv_partners[i], cellids2)):
                continue
            if syn_ax[0] == 1:
                ct_ax, ct_deso = m_cts[i]
                ssv_ax, ssv_deso = m_ssv_partners[i]
                spin_ax, spin_deso = m_spiness[i]
                ax, deso = syn_ax
            else:
                ct_deso, ct_ax = m_cts[i]
                ssv_deso, ssv_ax = m_ssv_partners[i]
                spin_deso, spin_ax = m_spiness[i]
                deso, ax = syn_ax
            if ct_ax == ct_deso:
                continue
            syn_size = m_sizes[i]
            if ct_ax == celltype1:
                cell2_ind = np.where(ct1_2_ct2_syn_dict["cellids"] == ssv_deso)[0]
                cell1_ind = np.where(ct2_2_ct1_syn_dict["cellids"] == ssv_ax)[0]
                ct1_2_ct2_syn_dict["syn amount"][cell2_ind] += 1
                ct1_2_ct2_syn_dict["sum syn size"][cell2_ind] += syn_size

                ct1_2_ct2_percell_syn_amount[cell2_ind, cell1_ind] += 1
                ct1_2_ct2_percell_syn_size[cell2_ind, cell1_ind] += syn_size
                if deso == 0:
                    if spin_deso == 0:
                        ct1_2_ct2_syn_dict["amount shaft syn"][cell2_ind] += 1
                        ct1_2_ct2_syn_dict["sum size shaft syn"][cell2_ind] += syn_size
                    elif spin_deso == 1:
                        ct1_2_ct2_syn_dict["amount spine head syn"][cell2_ind] += 1
                        ct1_2_ct2_syn_dict["sum size spine head syn"][cell2_ind] += syn_size
                    elif spin_deso == 2:
                        ct1_2_ct2_syn_dict["amount spine neck syn"][cell2_ind] += 1
                        ct1_2_ct2_syn_dict["sum size spine neck syn"][cell2_ind] += syn_size
                else:
                    ct1_2_ct2_syn_dict["amount soma syn"][cell2_ind] += 1
                    ct1_2_ct2_syn_dict["sum size soma syn"][cell2_ind] += syn_size
            else:
                cell1_ind = np.where(ct2_2_ct1_syn_dict["cellids"] == ssv_deso)[0]
                cell2_ind = np.where(ct1_2_ct2_syn_dict["cellids"] == ssv_ax)[0]
                ct2_2_ct1_syn_dict["syn amount"][cell1_ind] += 1
                ct2_2_ct1_syn_dict["sum syn size"][cell1_ind] += syn_size
                ct2_2_ct1_percell_syn_amount[cell1_ind, cell2_ind] += 1
                ct2_2_ct1_percell_syn_size[cell1_ind, cell2_ind] += syn_size
                if deso == 0:
                    if spin_deso == 0:
                        ct2_2_ct1_syn_dict["amount shaft syn"][cell1_ind] += 1
                        ct2_2_ct1_syn_dict["sum size shaft syn"][cell1_ind] += syn_size
                    elif spin_deso == 1:
                        ct2_2_ct1_syn_dict["amount spine head syn"][cell1_ind] += 1
                        ct2_2_ct1_syn_dict["sum size spine head syn"][cell1_ind] += syn_size
                    elif spin_deso == 2:
                        ct2_2_ct1_syn_dict["amount spine neck syn"][cell1_ind] += 1
                        ct2_2_ct1_syn_dict["sum size spine neck syn"][cell1_ind] += syn_size
                else:
                    ct2_2_ct1_syn_dict["amount soma syn"][cell1_ind] += 1
                    ct2_2_ct1_syn_dict["sum size soma syn"][cell1_ind] += syn_size

        syntime = time.time() - prepsyntime
        print("%.2f sec for processing synapses" % syntime)
        time_stamps.append(time.time())
        step_idents.append('processing synapses')

        log.info("Step 4/4: calculate last parameters (average syn size, average syn amount and syn size between 2 cells) and plotting")


        ct2_syn_inds = ct1_2_ct2_syn_dict["syn amount"] > 0
        for key in ct1_2_ct2_syn_dict.keys():
            ct1_2_ct2_syn_dict[key] = ct1_2_ct2_syn_dict[key][ct2_syn_inds]

        ct1_2_ct2_syn_dict["avg syn size"] = ct1_2_ct2_syn_dict["sum syn size"] / ct1_2_ct2_syn_dict["syn amount"]
        ct1_2_ct2_syn_dict["avg size shaft syn"] = ct1_2_ct2_syn_dict["sum size shaft syn"] / ct1_2_ct2_syn_dict["amount shaft syn"]
        ct1_2_ct2_syn_dict["avg size spine head syn"] = ct1_2_ct2_syn_dict["sum size spine head syn"] / ct1_2_ct2_syn_dict[
            "amount spine head syn"]
        ct1_2_ct2_syn_dict["avg size spine neck syn"] = ct1_2_ct2_syn_dict["sum size spine neck syn"] / ct1_2_ct2_syn_dict[
            "amount spine neck syn"]
        ct1_2_ct2_syn_dict["avg size soma syn"] = ct1_2_ct2_syn_dict["sum size soma syn"] / ct1_2_ct2_syn_dict[
            "amount soma syn"]
        ct1_2_ct2_syn_dict["percentage shaft syn amount"] = ct1_2_ct2_syn_dict["amount shaft syn"]/ ct1_2_ct2_syn_dict["syn amount"] * 100
        ct1_2_ct2_syn_dict["percentage spine head syn amount"] = ct1_2_ct2_syn_dict["amount spine head syn"] / ct1_2_ct2_syn_dict[
            "syn amount"] * 100
        ct1_2_ct2_syn_dict["percentage spine neck syn amount"] = ct1_2_ct2_syn_dict["amount spine neck syn"] / ct1_2_ct2_syn_dict[
            "syn amount"] * 100
        ct1_2_ct2_syn_dict["percentage soma syn amount"] = ct1_2_ct2_syn_dict["amount soma syn"] / ct1_2_ct2_syn_dict[
            "syn amount"] * 100
        ct1_2_ct2_syn_dict["percentage shaft syn size"] = ct1_2_ct2_syn_dict["sum size shaft syn"] / ct1_2_ct2_syn_dict[
            "sum syn size"] * 100
        ct1_2_ct2_syn_dict["percentage spine head syn size"] = ct1_2_ct2_syn_dict["sum size spine head syn"] / \
                                                             ct1_2_ct2_syn_dict[
                                                                 "sum syn size"] * 100
        ct1_2_ct2_syn_dict["percentage spine neck syn size"] = ct1_2_ct2_syn_dict["sum size spine neck syn"] / \
                                                             ct1_2_ct2_syn_dict[
                                                                 "sum syn size"] * 100
        ct1_2_ct2_syn_dict["percentage soma syn size"] = ct1_2_ct2_syn_dict["sum size soma syn"] / ct1_2_ct2_syn_dict[
            "sum syn size"] * 100

        ct1_syn_inds = ct2_2_ct1_syn_dict["syn amount"] > 0
        for key in ct2_2_ct1_syn_dict.keys():
            ct2_2_ct1_syn_dict[key] = ct2_2_ct1_syn_dict[key][ct1_syn_inds]

        ct2_2_ct1_syn_dict["avg syn size"] = ct2_2_ct1_syn_dict["sum syn size"] / ct2_2_ct1_syn_dict["syn amount"]
        ct2_2_ct1_syn_dict["avg size shaft syn"] = ct2_2_ct1_syn_dict["sum size shaft syn"] / ct2_2_ct1_syn_dict[
            "amount shaft syn"]
        ct2_2_ct1_syn_dict["avg size spine head syn"] = ct2_2_ct1_syn_dict["sum size spine head syn"] / \
                                                        ct2_2_ct1_syn_dict[
                                                            "amount spine head syn"]
        ct2_2_ct1_syn_dict["avg size spine neck syn"] = ct2_2_ct1_syn_dict["sum size spine neck syn"] / \
                                                        ct2_2_ct1_syn_dict[
                                                            "amount spine neck syn"]
        ct2_2_ct1_syn_dict["avg size soma syn"] = ct2_2_ct1_syn_dict["sum size soma syn"] / ct2_2_ct1_syn_dict[
            "amount soma syn"]
        ct2_2_ct1_syn_dict["percentage shaft syn amount"] = ct2_2_ct1_syn_dict["amount shaft syn"] / \
                                                            ct2_2_ct1_syn_dict["syn amount"] * 100
        ct2_2_ct1_syn_dict["percentage spine head syn amount"] = ct2_2_ct1_syn_dict["amount spine head syn"] / \
                                                                 ct2_2_ct1_syn_dict[
                                                                     "syn amount"] * 100
        ct2_2_ct1_syn_dict["percentage spine neck syn amount"] = ct2_2_ct1_syn_dict["amount spine neck syn"] / \
                                                                 ct2_2_ct1_syn_dict[
                                                                     "syn amount"] * 100
        ct2_2_ct1_syn_dict["percentage soma syn amount"] = ct2_2_ct1_syn_dict["amount soma syn"] / \
                                                           ct2_2_ct1_syn_dict[
                                                               "syn amount"] * 100
        ct2_2_ct1_syn_dict["percentage shaft syn size"] = ct2_2_ct1_syn_dict["sum size shaft syn"] / \
                                                          ct2_2_ct1_syn_dict[
                                                              "sum syn size"] * 100
        ct2_2_ct1_syn_dict["percentage spine head syn size"] = ct2_2_ct1_syn_dict["sum size spine head syn"] / \
                                                               ct2_2_ct1_syn_dict[
                                                                   "sum syn size"] * 100
        ct2_2_ct1_syn_dict["percentage spine neck syn size"] = ct2_2_ct1_syn_dict["sum size spine neck syn"] / \
                                                               ct2_2_ct1_syn_dict[
                                                                   "sum syn size"] * 100
        ct2_2_ct1_syn_dict["percentage soma syn size"] = ct2_2_ct1_syn_dict["sum size soma syn"] / \
                                                         ct2_2_ct1_syn_dict[
                                                             "sum syn size"] * 100

        ct1_2_ct2_syn_dict.pop("sum syn size")
        ct1_2_ct2_syn_dict.pop("sum size soma syn")
        ct1_2_ct2_syn_dict.pop("sum size spine head syn")
        ct1_2_ct2_syn_dict.pop("sum size spine neck syn")
        ct1_2_ct2_syn_dict.pop("sum size shaft syn")

        ct2_2_ct1_syn_dict.pop("sum syn size")
        ct2_2_ct1_syn_dict.pop("sum size soma syn")
        ct2_2_ct1_syn_dict.pop("sum size spine head syn")
        ct2_2_ct1_syn_dict.pop("sum size spine neck syn")
        ct2_2_ct1_syn_dict.pop("sum size shaft syn")

        #average synapse amount and size for one cell form ct1 to one cell of ct2
        ct1_2_ct2_percell_syn_amount = ct1_2_ct2_percell_syn_amount[ct2_syn_inds]
        ct1_2_ct2_percell_syn_amount[ct1_2_ct2_percell_syn_amount == 0] = np.nan
        ct1_2_ct2_syn_dict["avg amount one cell"] = np.nanmean(ct1_2_ct2_percell_syn_amount, axis = 1)

        ct1_2_ct2_percell_syn_size = ct1_2_ct2_percell_syn_size[ct2_syn_inds]
        ct1_2_ct2_percell_syn_size[ct1_2_ct2_percell_syn_size == 0] = np.nan
        ct1_2_ct2_syn_dict["avg syn size one cell"] = np.nanmean(ct1_2_ct2_percell_syn_size, axis = 1) / ct1_2_ct2_syn_dict["avg amount one cell"]

        ct2_2_ct1_percell_syn_amount = ct2_2_ct1_percell_syn_amount[ct1_syn_inds]
        ct2_2_ct1_percell_syn_amount[ct2_2_ct1_percell_syn_amount == 0] = np.nan
        ct2_2_ct1_syn_dict["avg amount one cell"] = np.nanmean(ct2_2_ct1_percell_syn_amount, axis=1)

        ct2_2_ct1_percell_syn_size = ct2_2_ct1_percell_syn_size[ct1_syn_inds]
        ct2_2_ct1_percell_syn_size[ct2_2_ct1_percell_syn_size == 0] = np.nan
        ct2_2_ct1_syn_dict["avg syn size one cell"] = np.nanmean(ct2_2_ct1_percell_syn_size, axis=1) / \
                                                      ct2_2_ct1_syn_dict["avg amount one cell"]
            

        write_obj2pkl("%s/%s_2_%s_dict.pkl" % (f_name, ct_dict[celltype1], ct_dict[celltype2]), ct1_2_ct2_syn_dict)
        ct1_2_ct2_pd = pd.DataFrame(ct1_2_ct2_syn_dict)
        ct1_2_ct2_pd.to_csv("%s/%s_2_%s_dict.csv" % (f_name, ct_dict[celltype1], ct_dict[celltype2]))

        write_obj2pkl("%s/%s_2_%s_dict.pkl" % (f_name, ct_dict[celltype2], ct_dict[celltype1]), ct2_2_ct1_syn_dict)
        ct2_2_ct1_pd = pd.DataFrame(ct2_2_ct1_syn_dict)
        ct2_2_ct1_pd.to_csv("%s/%s_2_%s_dict.csv" % (f_name, ct_dict[celltype2], ct_dict[celltype1]))

        #plot parameters as distplot
        for key in ct1_2_ct2_syn_dict.keys():
            if "ids" in key:
                continue
            sns.distplot(ct1_2_ct2_syn_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "steelblue"},
                         kde=False)
            plt.ylabel("count of cells")
            if "amount" in key:
                plt.xlabel("amount of synapses")
            elif "size" in key:
                plt.xlabel("average synapse size [µm²]")
            elif "percentage" in key:
                plt.xlabel("percentage of synapses")
            plt.title("%s from %s to %s" % (key, ct_dict[celltype1], ct_dict[celltype2]))
            plt.savefig("%s/%s_%s_2_%s.png" % (f_name, key, ct_dict[celltype1], ct_dict[celltype2]))
            plt.close()
            sns.distplot(ct2_2_ct1_syn_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "steelblue"},
                         kde=False, bins=10)
            plt.ylabel("count of cells")
            if "amount" in key:
                if "percentage" in key:
                    plt.xlabel("percentage of synapses")
                else:
                    plt.xlabel("amount of synapses")
            elif "size" in key:
                if "percentage" in key:
                    plt.xlabel("percentage of synapse size")
                else:
                    plt.xlabel("average synapse size [µm²]")

            plt.title("%s from %s to %s" % (key, ct_dict[celltype2], ct_dict[celltype1]))
            plt.savefig("%s/%s_%s_2_%s.png" % (f_name, key, ct_dict[celltype2], ct_dict[celltype1]))
            plt.close()

        #make violin plots for amount and size (absolute, relative) for different compartments
        x_labels = ["spine head", "spine neck", "shaft", "soma"]
        ticks = np.arange(4)
        sns.violinplot(data = [ct1_2_ct2_pd["amount spine head syn"], ct1_2_ct2_pd["amount spine neck syn"], ct1_2_ct2_pd["amount shaft syn"], ct1_2_ct2_pd["amount soma syn"]], inner= "box")
        sns.stripplot(data=[ct1_2_ct2_pd["amount spine head syn"], ct1_2_ct2_pd["amount spine neck syn"], ct1_2_ct2_pd["amount shaft syn"], ct1_2_ct2_pd["amount soma syn"]], color="black", alpha=0.2)
        plt.xticks(ticks = ticks, labels=x_labels)
        plt.ylabel("amount of synapses")
        plt.title("synapse amount from %s to %s" % (ct_dict[celltype1], ct_dict[celltype2]))
        plt.savefig("%s/syn_amount_violin_%s_2_%s.png" % (f_name, ct_dict[celltype1], ct_dict[celltype2]))
        plt.close()

        sns.boxplot(data=[ct1_2_ct2_pd["amount spine head syn"], ct1_2_ct2_pd["amount spine neck syn"],
                             ct1_2_ct2_pd["amount shaft syn"], ct1_2_ct2_pd["amount soma syn"]])
        sns.stripplot(data=[ct1_2_ct2_pd["amount spine head syn"], ct1_2_ct2_pd["amount spine neck syn"],
                            ct1_2_ct2_pd["amount shaft syn"], ct1_2_ct2_pd["amount soma syn"]], color="black",
                      alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("amount of synapses")
        plt.title("synapse amount from %s to %s" % (ct_dict[celltype1], ct_dict[celltype2]))
        plt.savefig("%s/syn_amount_box_%s_2_%s.png" % (f_name, ct_dict[celltype1], ct_dict[celltype2]))
        plt.close()


        sns.violinplot(data=[ct1_2_ct2_pd["avg size spine head syn"], ct1_2_ct2_pd["avg size spine neck syn"],
                             ct1_2_ct2_pd["avg size shaft syn"], ct1_2_ct2_pd["avg size soma syn"]], inner="box")
        sns.stripplot(data=[ct1_2_ct2_pd["avg size spine head syn"], ct1_2_ct2_pd["avg size spine neck syn"],
                            ct1_2_ct2_pd["avg size shaft syn"], ct1_2_ct2_pd["avg size soma syn"]], color="black",
                      alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("average size of synapses [µm²]")
        plt.title("average synapse size from %s to %s" % (ct_dict[celltype1], ct_dict[celltype2]))
        plt.savefig("%s/syn_size_violin_%s_2_%s.png" % (f_name, ct_dict[celltype1], ct_dict[celltype2]))
        plt.close()

        sns.boxplot(data=[ct1_2_ct2_pd["avg size spine head syn"], ct1_2_ct2_pd["avg size spine neck syn"],
                             ct1_2_ct2_pd["avg size shaft syn"], ct1_2_ct2_pd["avg size soma syn"]])
        sns.stripplot(data=[ct1_2_ct2_pd["avg size spine head syn"], ct1_2_ct2_pd["avg size spine neck syn"],
                            ct1_2_ct2_pd["avg size shaft syn"], ct1_2_ct2_pd["avg size soma syn"]], color="black",
                      alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("average size of synapses [µm²]")
        plt.title("average synapse size from %s to %s" % (ct_dict[celltype1], ct_dict[celltype2]))
        plt.savefig("%s/syn_size_box_%s_2_%s.png" % (f_name, ct_dict[celltype1], ct_dict[celltype2]))
        plt.close()

        sns.violinplot(data=[ct1_2_ct2_pd["percentage spine head syn amount"], ct1_2_ct2_pd["percentage spine neck syn amount"],
                             ct1_2_ct2_pd["percentage shaft syn amount"], ct1_2_ct2_pd["percentage soma syn amount"]], inner="box")
        sns.stripplot(data=[ct1_2_ct2_pd["percentage spine head syn amount"], ct1_2_ct2_pd["percentage spine neck syn amount"],
                            ct1_2_ct2_pd["percentage shaft syn amount"], ct1_2_ct2_pd["percentage soma syn amount"]], color="black",
                      alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("percentage of synapses")
        plt.title("percentage of synapses from %s to %s" % (ct_dict[celltype1], ct_dict[celltype2]))
        plt.savefig("%s/syn_perc_violin_%s_2_%s.png" % (f_name,  ct_dict[celltype1], ct_dict[celltype2]))
        plt.close()

        sns.boxplot(
            data=[ct1_2_ct2_pd["percentage spine head syn amount"], ct1_2_ct2_pd["percentage spine neck syn amount"],
                  ct1_2_ct2_pd["percentage shaft syn amount"], ct1_2_ct2_pd["percentage soma syn amount"]])
        sns.stripplot(
            data=[ct1_2_ct2_pd["percentage spine head syn amount"], ct1_2_ct2_pd["percentage spine neck syn amount"],
                  ct1_2_ct2_pd["percentage shaft syn amount"], ct1_2_ct2_pd["percentage soma syn amount"]], color="black",
            alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("percentage of synapses")
        plt.title("percentage of synapses from %s to %s" % (ct_dict[celltype1], ct_dict[celltype2]))
        plt.savefig("%s/syn_perc_box_%s_2_%s.png" % (f_name, ct_dict[celltype1], ct_dict[celltype2]))
        plt.close()

        sns.violinplot(
            data=[ct1_2_ct2_pd["percentage spine head syn size"], ct1_2_ct2_pd["percentage spine neck syn size"],
                  ct1_2_ct2_pd["percentage shaft syn size"], ct1_2_ct2_pd["percentage soma syn size"]], inner="box")
        sns.stripplot(
            data=[ct1_2_ct2_pd["percentage spine head syn size"], ct1_2_ct2_pd["percentage spine neck syn size"],
                  ct1_2_ct2_pd["percentage shaft syn size"], ct1_2_ct2_pd["percentage soma syn size"]], color="black",
            alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("percentage of synapse size")
        plt.title("percentage of synapse size from %s to %s" % (ct_dict[celltype1], ct_dict[celltype2]))
        plt.savefig("%s/syn_size_perc_violin_%s_2_%s.png" % (f_name,ct_dict[celltype1], ct_dict[celltype2]))
        plt.close()

        sns.boxplot(
            data=[ct1_2_ct2_pd["percentage spine head syn size"], ct1_2_ct2_pd["percentage spine neck syn size"],
                  ct1_2_ct2_pd["percentage shaft syn size"], ct1_2_ct2_pd["percentage soma syn size"]])
        sns.stripplot(
            data=[ct1_2_ct2_pd["percentage spine head syn size"], ct1_2_ct2_pd["percentage spine neck syn size"],
                  ct1_2_ct2_pd["percentage shaft syn size"], ct1_2_ct2_pd["percentage soma syn size"]], color="black",
            alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("percentage of synapse size")
        plt.title("percentage of synapse size from %s to %s" % (ct_dict[celltype1], ct_dict[celltype2]))
        plt.savefig("%s/syn_size_perc_box_%s_2_%s.png" % (f_name, ct_dict[celltype1], ct_dict[celltype2]))
        plt.close()

        sns.violinplot(data=[ct2_2_ct1_pd["amount spine head syn"], ct2_2_ct1_pd["amount spine neck syn"],
                             ct2_2_ct1_pd["amount shaft syn"], ct2_2_ct1_pd["amount soma syn"]], inner="box")
        sns.stripplot(data=[ct2_2_ct1_pd["amount spine head syn"], ct2_2_ct1_pd["amount spine neck syn"],
                            ct2_2_ct1_pd["amount shaft syn"], ct2_2_ct1_pd["amount soma syn"]], color="black",
                      alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("amount of synapses")
        plt.title("synapse amount from %s to %s" % (ct_dict[celltype2], ct_dict[celltype1]))
        plt.savefig("%s/syn_amount_violin_%s_2_%s.png" % (f_name, ct_dict[celltype2], ct_dict[celltype1]))
        plt.close()

        sns.boxplot(data=[ct2_2_ct1_pd["amount spine head syn"], ct2_2_ct1_pd["amount spine neck syn"],
                             ct2_2_ct1_pd["amount shaft syn"], ct2_2_ct1_pd["amount soma syn"]])
        sns.stripplot(data=[ct2_2_ct1_pd["amount spine head syn"], ct2_2_ct1_pd["amount spine neck syn"],
                            ct2_2_ct1_pd["amount shaft syn"], ct2_2_ct1_pd["amount soma syn"]], color="black",
                      alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("amount of synapses")
        plt.title("synapse amount from %s to %s" % (ct_dict[celltype2], ct_dict[celltype1]))
        plt.savefig("%s/syn_amount_box_%s_2_%s.png" % (f_name, ct_dict[celltype2], ct_dict[celltype1]))
        plt.close()

        sns.violinplot(data=[ct2_2_ct1_pd["avg size spine head syn"], ct2_2_ct1_pd["avg size spine neck syn"],
                             ct2_2_ct1_pd["avg size shaft syn"], ct2_2_ct1_pd["avg size soma syn"]], inner="box")
        sns.stripplot(data=[ct2_2_ct1_pd["avg size spine head syn"], ct2_2_ct1_pd["avg size spine neck syn"],
                            ct2_2_ct1_pd["avg size shaft syn"], ct2_2_ct1_pd["avg size soma syn"]], color="black",
                      alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("average size of synapses [µm²]")
        plt.title("average synapse size from %s to %s" % (ct_dict[celltype2], ct_dict[celltype1]))
        plt.savefig("%s/syn_size_violin_%s_2_%s.png" % (f_name, ct_dict[celltype2], ct_dict[celltype1]))
        plt.close()

        sns.boxplot(data=[ct2_2_ct1_pd["avg size spine head syn"], ct2_2_ct1_pd["avg size spine neck syn"],
                             ct2_2_ct1_pd["avg size shaft syn"], ct2_2_ct1_pd["avg size soma syn"]])
        sns.stripplot(data=[ct2_2_ct1_pd["avg size spine head syn"], ct2_2_ct1_pd["avg size spine neck syn"],
                            ct2_2_ct1_pd["avg size shaft syn"], ct2_2_ct1_pd["avg size soma syn"]], color="black",
                      alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("average size of synapses [µm²]")
        plt.title("average synapse size from %s to %s" % (ct_dict[celltype2], ct_dict[celltype1]))
        plt.savefig("%s/syn_size_box_%s_2_%s.png" % (f_name, ct_dict[celltype2], ct_dict[celltype1]))
        plt.close()

        sns.violinplot(
            data=[ct2_2_ct1_pd["percentage spine head syn amount"], ct2_2_ct1_pd["percentage spine neck syn amount"],
                  ct2_2_ct1_pd["percentage shaft syn amount"], ct2_2_ct1_pd["percentage soma syn amount"]], inner="box")
        sns.stripplot(
            data=[ct2_2_ct1_pd["percentage spine head syn amount"], ct2_2_ct1_pd["percentage spine neck syn amount"],
                  ct2_2_ct1_pd["percentage shaft syn amount"], ct2_2_ct1_pd["percentage soma syn amount"]],
            color="black",
            alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("percentage of synapses")
        plt.title("percentage of synapses from %s to %s" % (ct_dict[celltype2], ct_dict[celltype1]))
        plt.savefig("%s/syn_perc_violin_%s_2_%s.png" % (f_name, ct_dict[celltype2], ct_dict[celltype1]))
        plt.close()

        sns.boxplot(
            data=[ct2_2_ct1_pd["percentage spine head syn amount"], ct2_2_ct1_pd["percentage spine neck syn amount"],
                  ct2_2_ct1_pd["percentage shaft syn amount"], ct2_2_ct1_pd["percentage soma syn amount"]])
        sns.stripplot(
            data=[ct2_2_ct1_pd["percentage spine head syn amount"], ct2_2_ct1_pd["percentage spine neck syn amount"],
                  ct2_2_ct1_pd["percentage shaft syn amount"], ct2_2_ct1_pd["percentage soma syn amount"]],
            color="black",
            alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("percentage of synapses")
        plt.title("percentage of synapses from %s to %s" % (ct_dict[celltype2], ct_dict[celltype1]))
        plt.savefig("%s/syn_perc_box_%s_2_%s.png" % (f_name, ct_dict[celltype2], ct_dict[celltype1]))
        plt.close()

        sns.violinplot(
            data=[ct2_2_ct1_pd["percentage spine head syn size"], ct2_2_ct1_pd["percentage spine neck syn size"],
                  ct2_2_ct1_pd["percentage shaft syn size"], ct2_2_ct1_pd["percentage soma syn size"]], inner="box")
        sns.stripplot(
            data=[ct2_2_ct1_pd["percentage spine head syn size"], ct2_2_ct1_pd["percentage spine neck syn size"],
                  ct2_2_ct1_pd["percentage shaft syn size"], ct2_2_ct1_pd["percentage soma syn size"]], color="black",
            alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("percentage of synapse size")
        plt.title("percentage of synapse size from %s to %s" % (ct_dict[celltype2], ct_dict[celltype1]))
        plt.savefig("%s/syn_size_perc_violin_%s_2_%s.png" % (f_name, ct_dict[celltype2], ct_dict[celltype1]))
        plt.close()

        sns.boxplot(
            data=[ct2_2_ct1_pd["percentage spine head syn size"], ct2_2_ct1_pd["percentage spine neck syn size"],
                  ct2_2_ct1_pd["percentage shaft syn size"], ct2_2_ct1_pd["percentage soma syn size"]])
        sns.stripplot(
            data=[ct2_2_ct1_pd["percentage spine head syn size"], ct2_2_ct1_pd["percentage spine neck syn size"],
                  ct2_2_ct1_pd["percentage shaft syn size"], ct2_2_ct1_pd["percentage soma syn size"]], color="black",
            alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("percentage of synapse size")
        plt.title("percentage of synapse size from %s to %s" % (ct_dict[celltype2], ct_dict[celltype1]))
        plt.savefig("%s/syn_size_perc_box_%s_2_%s.png" % (f_name, ct_dict[celltype2], ct_dict[celltype1]))
        plt.close()

        plottime = time.time() - syntime
        print("%.2f sec for calculating parameters, plotting" % plottime)
        time_stamps.append(time.time())
        step_idents.append('calculating last parameters, plotting')

        log.info("Connectivity analysis between 2 celltypes (%s, %s) finished" % (ct_dict[celltype1], ct_dict[celltype2]))

    def compare_connectvity(comp_ct1, comp_ct2, connected_ct = None, foldername_ct1 = None, foldername_ct2 = None, min_comp_len = 100):
        '''
        compares connectivity parameters between two celltypes or connectivity of a third celltype to the two celltypes. Connectivity parameters are calculated in
        synapses_between2cts. Parameters include synapse amount and average synapse size, as well as amount and average synapse size in shaft, soma, spine head and spine neck.
        P-value will be calculated with scipy.stats.ranksum.
        :param comp_ct1, comp_ct2: celltypes to compare to each other
        :param connected_ct: if given, connectivity of this celltype to two others will be compared
        :param foldername_ct1, foldername_ct2: foldernames where parameters of connectivity are stored
        :param min_comp_len: minimum compartment length
        :return:
        '''
        start = time.time()
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   10: "NGF"}
        if connected_ct != None:
            f_name = "u/arother/bio_analysis_results/dir_indir_pathway_analysis/210923_j0251v3__%s_%s_%s_syn_con_comp_mcl%i" % (
                ct_dict[comp_ct1], ct_dict[comp_ct2], ct_dict[connected_ct], min_comp_len)
        else:
            f_name = "u/arother/bio_analysis_results/dir_indir_pathway_analysis/210923_j0251v3__%s_%s_syn_con_comp_mcl%i" % (
            ct_dict[comp_ct1], ct_dict[comp_ct2], min_comp_len)
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('compare connectivty between two celltypes', log_dir=f_name + '/logs/')
        if connected_ct != None:
            log.info("parameters: celltype1 = %s,celltype2 = %s , connected ct = %s, min_comp_length = %.i" % (
            ct_dict[comp_ct1], ct_dict[comp_ct2], ct_dict[connected_ct], min_comp_len))
        else:
            log.info("parameters: celltype1 = %s,celltype2 = %s, min_comp_length = %.i" % (
                ct_dict[comp_ct1], ct_dict[comp_ct2], min_comp_len))
        time_stamps = [time.time()]
        step_idents = ['t-0']
        if connected_ct != None:
            ct2_syn_dict = load_pkl2obj("%s/%s_2_%s_dict.pkl" % (foldername_ct2, ct_dict[connected_ct], ct_dict[comp_ct2]))
            ct1_syn_dict = load_pkl2obj("%s/%s_2_%s_dict.pkl" % (foldername_ct1, ct_dict[connected_ct], ct_dict[comp_ct1]))
        else:
            ct2_syn_dict = load_pkl2obj(
                "%s/%s_2_%s_dict.pkl" % (foldername_ct2, ct_dict[comp_ct1], ct_dict[comp_ct2]))
            ct1_syn_dict = load_pkl2obj(
                "%s/%s_2_%s_dict.pkl" % (foldername_ct1, ct_dict[comp_ct2], ct_dict[comp_ct1]))
        syn_dict_keys = list(ct1_syn_dict.keys())
        log.info("compute statistics for comparison, create violinplot and histogram")
        ranksum_results = pd.DataFrame(columns=syn_dict_keys[1:], index=["stats", "p value"])
        colours_pal = {ct_dict[comp_ct1]: "mediumorchid", ct_dict[comp_ct2]: "springgreen"}

        # plot distribution of compartments togehter
        len_ct1 = len(ct1_syn_dict["amount spine head syn"])
        len_ct2 = len(ct2_syn_dict["amount spine head syn"])
        sum_len = len_ct1 + len_ct2
        results_for_plotting_comps = pd.DataFrame(
            columns=["syn amount", "avg syn size", "percentage amount", "percentage syn size", "celltype",
                     "compartment"], index=range(sum_len * 4))
        results_for_plotting_comps["compartment"] = str()
        results_for_plotting_comps.loc[0:sum_len - 1, "compartment"] = "spine head"
        results_for_plotting_comps.loc[0:len_ct1 - 1, "celltype"] = ct_dict[comp_ct1]
        results_for_plotting_comps.loc[len_ct1: sum_len - 1, "celltype"] = ct_dict[comp_ct2]
        results_for_plotting_comps.loc[0:len_ct1 - 1, "syn amount"] = ct1_syn_dict["amount spine head syn"]
        results_for_plotting_comps.loc[len_ct1:sum_len - 1, "syn amount"] = ct2_syn_dict["amount spine head syn"]
        results_for_plotting_comps.loc[0:len_ct1 - 1, "avg syn size"] = ct1_syn_dict["avg size spine head syn"]
        results_for_plotting_comps.loc[len_ct1:sum_len - 1,"avg syn size"] = ct2_syn_dict["avg size spine head syn"]
        results_for_plotting_comps.loc[0:len_ct1 - 1, "percentage amount"] = ct1_syn_dict["percentage spine head syn amount"]
        results_for_plotting_comps.loc[len_ct1:sum_len - 1,"percentage amount"] = ct2_syn_dict["percentage spine head syn amount"]
        results_for_plotting_comps.loc[0:len_ct1 - 1,"percentage syn size"] = ct1_syn_dict["percentage spine head syn size"]
        results_for_plotting_comps.loc[len_ct1:sum_len - 1,"percentage syn size"] = ct2_syn_dict["percentage spine head syn size"]
        results_for_plotting_comps.loc[sum_len:sum_len * 2 - 1, "compartment"] = "spine neck"
        results_for_plotting_comps.loc[sum_len:sum_len + len_ct1 - 1, "celltype"] = ct_dict[comp_ct1]
        results_for_plotting_comps.loc[sum_len + len_ct1:sum_len*2 - 1, "celltype"] = ct_dict[comp_ct2]
        results_for_plotting_comps.loc[sum_len:sum_len + len_ct1- 1,"syn amount"] = ct1_syn_dict["amount spine neck syn"]
        results_for_plotting_comps.loc[sum_len + len_ct1:sum_len*2 - 1,"syn amount"] = ct2_syn_dict["amount spine neck syn"]
        results_for_plotting_comps.loc[sum_len: sum_len + len_ct1 - 1, "avg syn size"] = ct1_syn_dict["avg size spine neck syn"]
        results_for_plotting_comps.loc[sum_len + len_ct1:sum_len*2 - 1,"avg syn size"] = ct2_syn_dict["avg size spine neck syn"]
        results_for_plotting_comps.loc[sum_len:sum_len + len_ct1 - 1, "percentage amount"] = ct1_syn_dict["percentage spine neck syn amount"]
        results_for_plotting_comps.loc[sum_len + len_ct1:sum_len*2 - 1,"percentage amount"] = ct2_syn_dict["percentage spine neck syn amount"]
        results_for_plotting_comps.loc[sum_len: sum_len + len_ct1 - 1, "percentage syn size"] = ct1_syn_dict["percentage spine neck syn size"]
        results_for_plotting_comps.loc[sum_len + len_ct1:sum_len*2 - 1,"percentage syn size"] = ct2_syn_dict["percentage spine neck syn size"]
        results_for_plotting_comps.loc[sum_len * 2:sum_len * 3 - 1, "compartment"] = "shaft"
        results_for_plotting_comps.loc[sum_len * 2:sum_len * 2 + len_ct1 - 1, "celltype"] = ct_dict[comp_ct1]
        results_for_plotting_comps.loc[sum_len * 2 + len_ct1:sum_len * 3 - 1, "celltype"] = ct_dict[comp_ct2]
        results_for_plotting_comps.loc[sum_len * 2: sum_len * 2 + len_ct1 - 1,"syn amount"] = ct1_syn_dict["amount shaft syn"]
        results_for_plotting_comps.loc[sum_len * 2 + len_ct1:sum_len * 3 - 1, "syn amount"] = ct2_syn_dict["amount shaft syn"]
        results_for_plotting_comps.loc[sum_len * 2: sum_len * 2 + len_ct1 - 1,"avg syn size"] = ct1_syn_dict["avg size shaft syn"]
        results_for_plotting_comps.loc[sum_len * 2 + len_ct1:sum_len * 3 - 1,"avg syn size"] = ct2_syn_dict["avg size shaft syn"]
        results_for_plotting_comps.loc[sum_len * 2:sum_len * 2 + len_ct1 - 1,"percentage amount"] = ct1_syn_dict["percentage shaft syn amount"]
        results_for_plotting_comps.loc[sum_len * 2 + len_ct1:sum_len * 3 - 1,"percentage amount"] = ct2_syn_dict["percentage shaft syn amount"]
        results_for_plotting_comps.loc[sum_len * 2: sum_len * 2 + len_ct1 - 1,"percentage syn size"] = ct1_syn_dict["percentage shaft syn size"]
        results_for_plotting_comps.loc[sum_len * 2 + len_ct1:sum_len * 3 - 1,"percentage syn size"] = ct2_syn_dict["percentage shaft syn size"]
        results_for_plotting_comps.loc[sum_len * 3:sum_len * 4 - 1, "compartment"] = "soma"
        results_for_plotting_comps.loc[sum_len * 3:sum_len * 3 + len_ct1 - 1, "celltype"] = ct_dict[comp_ct1]
        results_for_plotting_comps.loc[sum_len * 3 + len_ct1:sum_len * 4 - 1, "celltype"] = ct_dict[comp_ct2]
        results_for_plotting_comps.loc[sum_len * 3: sum_len * 3 + len_ct1 - 1,"syn amount"] = ct1_syn_dict["amount soma syn"]
        results_for_plotting_comps.loc[sum_len * 3 + len_ct1:sum_len * 4 - 1,"syn amount"] = ct2_syn_dict["amount soma syn"]
        results_for_plotting_comps.loc[sum_len * 3: sum_len * 3 + len_ct1 - 1,"avg syn size"] = ct1_syn_dict["avg size soma syn"]
        results_for_plotting_comps.loc[sum_len * 3 + len_ct1:sum_len * 4 - 1,"avg syn size"] = ct2_syn_dict["avg size soma syn"]
        results_for_plotting_comps.loc[sum_len * 3: sum_len * 3 + len_ct1 - 1,"percentage amount"] = ct1_syn_dict["percentage soma syn amount"]
        results_for_plotting_comps.loc[sum_len * 3 + len_ct1:sum_len * 4 - 1,"percentage amount"] = ct2_syn_dict["percentage soma syn amount"]
        results_for_plotting_comps.loc[sum_len * 3: sum_len * 3 + len_ct1 - 1,"percentage syn size"] = ct1_syn_dict["percentage soma syn size"]
        results_for_plotting_comps.loc[sum_len * 3 + len_ct1:sum_len*4 - 1,"percentage syn size"] = ct2_syn_dict["percentage soma syn size"]
        results_for_plotting_comps["syn amount"] = results_for_plotting_comps["syn amount"].astype("float64")
        results_for_plotting_comps["avg syn size"] = results_for_plotting_comps["avg syn size"].astype("float64")
        results_for_plotting_comps["percentage amount"] = results_for_plotting_comps["percentage amount"].astype('float64')
        results_for_plotting_comps["percentage syn size"] = results_for_plotting_comps["percentage syn size"].astype("float64")

        if connected_ct != None:
            results_for_plotting_comps.to_csv("%s/%s_2_%s_%s_syn_compartments.csv" % (f_name, ct_dict[connected_ct], ct_dict[comp_ct1], ct_dict[comp_ct2]))
        else:
            results_for_plotting_comps.to_csv("%s/%s_%s_syn_compartments.csv" % (
            f_name, ct_dict[comp_ct1], ct_dict[comp_ct2]))

        for key in results_for_plotting_comps.keys():
            if "celltype" in key or "compartment" in key:
                continue
            sns.stripplot(x="compartment", y=key, data=results_for_plotting_comps, hue="celltype", color="black",alpha=0.3, dodge=True)
            ax = sns.violinplot(x = "compartment", y = key, data=results_for_plotting_comps.reset_index(), inner="box", palette=colours_pal, hue="celltype")
            handles, labels = ax.get_legend_handles_labels()
            plt.legend(handles[0:2], labels[0:2])
            if connected_ct != None:
                plt.title('%s, %s to %s/ %s' % (key, ct_dict[connected_ct], ct_dict[comp_ct1], ct_dict[comp_ct2]))
            else:
                plt.title('%s, between %s and %s in different compartments' % (key, ct_dict[comp_ct1], ct_dict[comp_ct2]))
            if "amount" in key:
                if "percentage" in key:
                    plt.ylabel("percentage of synapses")
                else:
                    plt.ylabel("amount of synapses")
            elif "size" in key:
                if "percentage" in key:
                    plt.ylabel("percentage of synapse size")
                else:
                    plt.ylabel("average synapse size [µm²]")
            if connected_ct != None:
                filename = ("%s/%s_comps_violin_%s_2_%s_%s.png" % (
                f_name, key, ct_dict[connected_ct], ct_dict[comp_ct1], ct_dict[comp_ct2]))
            else:
                filename = ("%s/%s_comps_violin_%s_%s.png" % (f_name, key, ct_dict[comp_ct1], ct_dict[comp_ct2]))
            plt.savefig(filename)
            plt.close()
            sns.boxplot(x="compartment", y=key, data=results_for_plotting_comps, palette=colours_pal,
                           hue="celltype")
            if connected_ct != None:
                plt.title('%s, %s to %s/ %s' % (key, ct_dict[connected_ct], ct_dict[comp_ct1], ct_dict[comp_ct2]))
            else:
                plt.title(
                    '%s, between %s and %s in different compartments' % (key, ct_dict[comp_ct1], ct_dict[comp_ct2]))
            if "amount" in key:
                if "percentage" in key:
                    plt.ylabel("percentage of synapses")
                else:
                    plt.ylabel("amount of synapses")
            elif "size" in key:
                if "percentage" in key:
                    plt.ylabel("percentage of synapse size")
                else:
                    plt.ylabel("average synapse size [µm²]")
            if connected_ct != None:
                filename = ("%s/%s_comps_box_%s_2_%s_%s.png" % (
                    f_name, key, ct_dict[connected_ct], ct_dict[comp_ct1], ct_dict[comp_ct2]))
            else:
                filename = ("%s/%s_comps_box_%s_%s.png" % (f_name, key, ct_dict[comp_ct1], ct_dict[comp_ct2]))
            plt.savefig(filename)
            plt.close()

        for key in ct1_syn_dict.keys():
            if "ids" in key:
                continue
            # calculate p_value for parameter
            stats, p_value = ranksums(ct1_syn_dict[key], ct2_syn_dict[key])
            ranksum_results.loc["stats", key] = stats
            ranksum_results.loc["p value", key] = p_value
            # plot parameter as violinplot
            if not "spine" in key and not "soma" in key and not "shaft" in key:
                results_for_plotting = pd.DataFrame(columns=[ct_dict[comp_ct1], ct_dict[comp_ct2]], index=range(sum_len))
                results_for_plotting.loc[0:len(ct1_syn_dict[key]) - 1, ct_dict[comp_ct1]] = ct1_syn_dict[key]
                results_for_plotting.loc[0:len(ct2_syn_dict[key]) - 1, ct_dict[comp_ct2]] = ct2_syn_dict[key]
                sns.violinplot(data=results_for_plotting, inner="box", palette=colours_pal)
                sns.stripplot(data=results_for_plotting, color="black", alpha=0.2)
                if connected_ct != None:
                    plt.title('%s, %s to %s/ %s' % (key, ct_dict[connected_ct], ct_dict[comp_ct1], ct_dict[comp_ct2]))
                else:
                    plt.title('%s, between %s and %s' % (key, ct_dict[comp_ct1], ct_dict[comp_ct2]))
                if "amount" in key:
                    if "percentage" in key:
                        plt.ylabel("percentage of synapses")
                    else:
                        plt.ylabel("amount of synapses")
                elif "size" in key:
                    if "percentage" in key:
                        plt.ylabel("percentage of synapse size")
                    else:
                        plt.ylabel("average synapse size [µm²]")
                if connected_ct != None:
                    filename = ("%s/%s_violin_%s_2_%s_%s.png" % (f_name, key, ct_dict[connected_ct], ct_dict[comp_ct1], ct_dict[comp_ct2]))
                else:
                    filename = ("%s/%s_violin_%s_%s.png" % (f_name, key, ct_dict[comp_ct1], ct_dict[comp_ct2]))
                plt.savefig(filename)
                plt.close()
                sns.boxplot(data=results_for_plotting, palette=colours_pal)
                if connected_ct != None:
                    plt.title('%s, %s to %s/ %s' % (key, ct_dict[connected_ct], ct_dict[comp_ct1], ct_dict[comp_ct2]))
                else:
                    plt.title('%s, between %s and %s' % (key, ct_dict[comp_ct1], ct_dict[comp_ct2]))
                if "amount" in key:
                    if "percentage" in key:
                        plt.ylabel("percentage of synapses")
                    else:
                        plt.ylabel("amount of synapses")
                elif "size" in key:
                    if "percentage" in key:
                        plt.ylabel("percentage of synapse size")
                    else:
                        plt.ylabel("average synapse size [µm²]")
                if connected_ct != None:
                    filename = ("%s/%s_box_%s_2_%s_%s.png" % (
                    f_name, key, ct_dict[connected_ct], ct_dict[comp_ct1], ct_dict[comp_ct2]))
                else:
                    filename = ("%s/%s_box_%s_%s.png" % (f_name, key, ct_dict[comp_ct1], ct_dict[comp_ct2]))
                plt.savefig(filename)
                plt.close()
            sns.distplot(ct1_syn_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                         kde=False, label=ct_dict[comp_ct1], bins=10)
            sns.distplot(ct2_syn_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                         kde=False, label=ct_dict[comp_ct2], bins=10)
            plt.legend()
            if connected_ct != None:
                plt.title('%s, %s to %s/ %s' % (key, ct_dict[connected_ct], ct_dict[comp_ct1], ct_dict[comp_ct2]))
            else:
                plt.title('%s, between %s and %s' % (key, ct_dict[comp_ct1], ct_dict[comp_ct2]))
            plt.ylabel("count of cells")
            if "amount" in key:
                if "percentage" in key:
                    plt.xlabel("percentage of synapses")
                else:
                    plt.xlabel("amount of synapses")
            elif "size" in key:
                if "percentage" in key:
                    plt.xlabel("percentage of synapse size")
                else:
                    plt.xlabel("average synapse size [µm²]")
            if connected_ct != None:
                filename = ("%s/%s_hist_%s_2_%s_%s.png" % (
                    f_name, key, ct_dict[connected_ct], ct_dict[comp_ct1], ct_dict[comp_ct2]))
            else:
                filename = ("%s/%s_hist_%s_%s.png" % (f_name, key, ct_dict[comp_ct1], ct_dict[comp_ct2]))
            plt.savefig(filename)
            plt.close()
            sns.distplot(ct1_syn_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                         kde=False, label=ct_dict[comp_ct1], bins=10, norm_hist=True)
            sns.distplot(ct2_syn_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                         kde=False, label=ct_dict[comp_ct2], bins=10, norm_hist=True)
            plt.legend()
            if connected_ct != None:
                plt.title('%s, %s to %s/ %s' % (key, ct_dict[connected_ct], ct_dict[comp_ct1], ct_dict[comp_ct2]))
            else:
                plt.title('%s, between %s and %s' % (key, ct_dict[comp_ct1], ct_dict[comp_ct2]))
            plt.ylabel("fraction of cells")
            if "length" in key:
                plt.xlabel("pathlength in µm")
            elif "vol" in key:
                if "percentage" in key:
                    plt.xlabel("% of whole dataset")
                else:
                    plt.xlabel("volume in µm³")
            else:
                plt.xlabel("distance in µm")
            if connected_ct != None:
                filename = ("%s/%s_hist_norm_%s_2_%s_%s.png" % (
                    f_name, key, ct_dict[connected_ct], ct_dict[comp_ct1], ct_dict[comp_ct2]))
            else:
                filename = ("%s/%s_hist_norm_%s_%s.png" % (f_name, key, ct_dict[comp_ct1], ct_dict[comp_ct2]))
            plt.savefig(filename)
            plt.close()

        if connected_ct != None:
            ranksum_results.to_csv("%s/ranksum_%s_2_%s_%s.csv" % (f_name, ct_dict[connected_ct], ct_dict[comp_ct1], ct_dict[comp_ct2]))
        else:
            ranksum_results.to_csv("%s/ranksum_%s_%s.csv" % (f_name, ct_dict[comp_ct1], ct_dict[comp_ct2]))

        #also compare outgoing connections from celltype
        if connected_ct != None:
            ct2_syn_dict = load_pkl2obj("%s/%s_2_%s_dict.pkl" % (foldername_ct2, ct_dict[comp_ct2], ct_dict[connected_ct]))
            ct1_syn_dict = load_pkl2obj(
                "%s/%s_2_%s_dict.pkl" % (foldername_ct1, ct_dict[comp_ct1], ct_dict[connected_ct]))
            len_ct1 = len(ct1_syn_dict["amount spine head syn"])
            len_ct2 = len(ct2_syn_dict["amount spine head syn"])
            sum_len = len_ct1 + len_ct2
            results_for_plotting_comps = pd.DataFrame(
                columns=["syn amount", "avg syn size", "percentage amount", "percentage syn size", "celltype",
                         "compartment"], index=range(sum_len * 4))
            results_for_plotting_comps["compartment"] = str()
            results_for_plotting_comps.loc[0:sum_len - 1, "compartment"] = "spine head"
            results_for_plotting_comps.loc[0:len_ct1 - 1, "celltype"] = ct_dict[comp_ct1]
            results_for_plotting_comps.loc[len_ct1: sum_len - 1, "celltype"] = ct_dict[comp_ct2]
            results_for_plotting_comps.loc[0:len_ct1 - 1, "syn amount"] = ct1_syn_dict["amount spine head syn"]
            results_for_plotting_comps.loc[len_ct1:sum_len - 1, "syn amount"] = ct2_syn_dict["amount spine head syn"]
            results_for_plotting_comps.loc[0:len_ct1 - 1, "avg syn size"] = ct1_syn_dict["avg size spine head syn"]
            results_for_plotting_comps.loc[len_ct1:sum_len - 1, "avg syn size"] = ct2_syn_dict[
                "avg size spine head syn"]
            results_for_plotting_comps.loc[0:len_ct1 - 1, "percentage amount"] = ct1_syn_dict[
                "percentage spine head syn amount"]
            results_for_plotting_comps.loc[len_ct1:sum_len - 1, "percentage amount"] = ct2_syn_dict[
                "percentage spine head syn amount"]
            results_for_plotting_comps.loc[0:len_ct1 - 1, "percentage syn size"] = ct1_syn_dict[
                "percentage spine head syn size"]
            results_for_plotting_comps.loc[len_ct1:sum_len - 1, "percentage syn size"] = ct2_syn_dict[
                "percentage spine head syn size"]
            results_for_plotting_comps.loc[sum_len:sum_len * 2 - 1, "compartment"] = "spine neck"
            results_for_plotting_comps.loc[sum_len:sum_len + len_ct1 - 1, "celltype"] = ct_dict[comp_ct1]
            results_for_plotting_comps.loc[sum_len + len_ct1:sum_len * 2 - 1, "celltype"] = ct_dict[comp_ct2]
            results_for_plotting_comps.loc[sum_len:sum_len + len_ct1 - 1, "syn amount"] = ct1_syn_dict[
                "amount spine neck syn"]
            results_for_plotting_comps.loc[sum_len + len_ct1:sum_len * 2 - 1, "syn amount"] = ct2_syn_dict[
                "amount spine neck syn"]
            results_for_plotting_comps.loc[sum_len: sum_len + len_ct1 - 1, "avg syn size"] = ct1_syn_dict[
                "avg size spine neck syn"]
            results_for_plotting_comps.loc[sum_len + len_ct1:sum_len * 2 - 1, "avg syn size"] = ct2_syn_dict[
                "avg size spine neck syn"]
            results_for_plotting_comps.loc[sum_len:sum_len + len_ct1 - 1, "percentage amount"] = ct1_syn_dict[
                "percentage spine neck syn amount"]
            results_for_plotting_comps.loc[sum_len + len_ct1:sum_len * 2 - 1, "percentage amount"] = ct2_syn_dict[
                "percentage spine neck syn amount"]
            results_for_plotting_comps.loc[sum_len: sum_len + len_ct1 - 1, "percentage syn size"] = ct1_syn_dict[
                "percentage spine neck syn size"]
            results_for_plotting_comps.loc[sum_len + len_ct1:sum_len * 2 - 1, "percentage syn size"] = ct2_syn_dict[
                "percentage spine neck syn size"]
            results_for_plotting_comps.loc[sum_len * 2:sum_len * 3 - 1, "compartment"] = "shaft"
            results_for_plotting_comps.loc[sum_len * 2:sum_len * 2 + len_ct1 - 1, "celltype"] = ct_dict[comp_ct1]
            results_for_plotting_comps.loc[sum_len * 2 + len_ct1:sum_len * 3 - 1, "celltype"] = ct_dict[comp_ct2]
            results_for_plotting_comps.loc[sum_len * 2: sum_len * 2 + len_ct1 - 1, "syn amount"] = ct1_syn_dict[
                "amount shaft syn"]
            results_for_plotting_comps.loc[sum_len * 2 + len_ct1:sum_len * 3 - 1, "syn amount"] = ct2_syn_dict[
                "amount shaft syn"]
            results_for_plotting_comps.loc[sum_len * 2: sum_len * 2 + len_ct1 - 1, "avg syn size"] = ct1_syn_dict[
                "avg size shaft syn"]
            results_for_plotting_comps.loc[sum_len * 2 + len_ct1:sum_len * 3 - 1, "avg syn size"] = ct2_syn_dict[
                "avg size shaft syn"]
            results_for_plotting_comps.loc[sum_len * 2:sum_len * 2 + len_ct1 - 1, "percentage amount"] = ct1_syn_dict[
                "percentage shaft syn amount"]
            results_for_plotting_comps.loc[sum_len * 2 + len_ct1:sum_len * 3 - 1, "percentage amount"] = ct2_syn_dict[
                "percentage shaft syn amount"]
            results_for_plotting_comps.loc[sum_len * 2: sum_len * 2 + len_ct1 - 1, "percentage syn size"] = \
            ct1_syn_dict["percentage shaft syn size"]
            results_for_plotting_comps.loc[sum_len * 2 + len_ct1:sum_len * 3 - 1, "percentage syn size"] = ct2_syn_dict[
                "percentage shaft syn size"]
            results_for_plotting_comps.loc[sum_len * 3:sum_len * 4 - 1, "compartment"] = "soma"
            results_for_plotting_comps.loc[sum_len * 3:sum_len * 3 + len_ct1 - 1, "celltype"] = ct_dict[comp_ct1]
            results_for_plotting_comps.loc[sum_len * 3 + len_ct1:sum_len * 4 - 1, "celltype"] = ct_dict[comp_ct2]
            results_for_plotting_comps.loc[sum_len * 3: sum_len * 3 + len_ct1 - 1, "syn amount"] = ct1_syn_dict[
                "amount soma syn"]
            results_for_plotting_comps.loc[sum_len * 3 + len_ct1:sum_len * 4 - 1, "syn amount"] = ct2_syn_dict[
                "amount soma syn"]
            results_for_plotting_comps.loc[sum_len * 3: sum_len * 3 + len_ct1 - 1, "avg syn size"] = ct1_syn_dict[
                "avg size soma syn"]
            results_for_plotting_comps.loc[sum_len * 3 + len_ct1:sum_len * 4 - 1, "avg syn size"] = ct2_syn_dict[
                "avg size soma syn"]
            results_for_plotting_comps.loc[sum_len * 3: sum_len * 3 + len_ct1 - 1, "percentage amount"] = ct1_syn_dict[
                "percentage soma syn amount"]
            results_for_plotting_comps.loc[sum_len * 3 + len_ct1:sum_len * 4 - 1, "percentage amount"] = ct2_syn_dict[
                "percentage soma syn amount"]
            results_for_plotting_comps.loc[sum_len * 3: sum_len * 3 + len_ct1 - 1, "percentage syn size"] = \
            ct1_syn_dict["percentage soma syn size"]
            results_for_plotting_comps.loc[sum_len * 3 + len_ct1:sum_len * 4 - 1, "percentage syn size"] = ct2_syn_dict[
                "percentage soma syn size"]
            results_for_plotting_comps["syn amount"] = results_for_plotting_comps["syn amount"].astype("float64")
            results_for_plotting_comps["avg syn size"] = results_for_plotting_comps["avg syn size"].astype("float64")
            results_for_plotting_comps["percentage amount"] = results_for_plotting_comps["percentage amount"].astype(
                'float64')
            results_for_plotting_comps["percentage syn size"] = results_for_plotting_comps[
                "percentage syn size"].astype("float64")


            results_for_plotting_comps.to_csv("%s/%s_%s_2_%s_syn_compartments_outgoing.csv" % (
                f_name, ct_dict[comp_ct1], ct_dict[comp_ct2], ct_dict[connected_ct]))

            for key in results_for_plotting_comps.keys():
                if "celltype" in key or "compartment" in key:
                    continue

                sns.stripplot(x="compartment", y=key, data=results_for_plotting_comps, hue="celltype", color="black",
                              alpha=0.3, dodge=True)
                ax = sns.violinplot(x="compartment", y=key, data=results_for_plotting_comps.reset_index(), inner="box",
                                    palette=colours_pal, hue="celltype")
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(handles[0:2], labels[0:2])
                plt.title('%s, %s, %s to %s' % (key, ct_dict[comp_ct1], ct_dict[comp_ct2], ct_dict[connected_ct]))
                if "amount" in key:
                    if "percentage" in key:
                        plt.ylabel("percentage of synapses")
                    else:
                        plt.ylabel("amount of synapses")
                elif "size" in key:
                    if "percentage" in key:
                        plt.ylabel("percentage of synapse size")
                    else:
                        plt.ylabel("average synapse size [µm²]")
                filename = ("%s/%s_comps_violin_%s_%s_2_%s_outgoing.png" % (
                            f_name, key, ct_dict[comp_ct1], ct_dict[comp_ct2], ct_dict[connected_ct]))
                plt.savefig(filename)
                plt.close()
                sns.boxplot(x="compartment", y=key, data=results_for_plotting_comps, palette=colours_pal,
                            hue="celltype")
                plt.title('%s, %s, %s to %s' % (key, ct_dict[comp_ct1], ct_dict[comp_ct2], ct_dict[connected_ct]))
                if "amount" in key:
                    if "percentage" in key:
                        plt.ylabel("percentage of synapses")
                    else:
                        plt.ylabel("amount of synapses")
                elif "size" in key:
                    if "percentage" in key:
                        plt.ylabel("percentage of synapse size")
                    else:
                        plt.ylabel("average synapse size [µm²]")
                filename = ("%s/%s_comps_box_%s_%s_2_%s.png" % (
                            f_name, key, ct_dict[comp_ct1], ct_dict[comp_ct2], ct_dict[connected_ct]))
                plt.savefig(filename)
                plt.close()

            for key in ct1_syn_dict.keys():
                if "ids" in key:
                    continue
                # calculate p_value for parameter
                stats, p_value = ranksums(ct1_syn_dict[key], ct2_syn_dict[key])
                ranksum_results.loc["stats", key] = stats
                ranksum_results.loc["p value", key] = p_value
                # plot parameter as violinplot
                if not "spine" in key and not "soma" in key and not "shaft" in key:
                    results_for_plotting = pd.DataFrame(columns=[ct_dict[comp_ct1], ct_dict[comp_ct2]],
                                                        index=range(sum_len))
                    results_for_plotting.loc[0:len(ct1_syn_dict[key]) - 1, ct_dict[comp_ct1]] = ct1_syn_dict[key]
                    results_for_plotting.loc[0:len(ct2_syn_dict[key]) - 1, ct_dict[comp_ct2]] = ct2_syn_dict[key]
                    sns.violinplot(data=results_for_plotting, inner="box", palette=colours_pal)
                    sns.stripplot(data=results_for_plotting, color="black", alpha=0.2)
                    plt.title('%s, %s, %s to %s' % (key, ct_dict[comp_ct1], ct_dict[comp_ct2], ct_dict[connected_ct]))
                    if "amount" in key:
                        if "percentage" in key:
                            plt.ylabel("percentage of synapses")
                        else:
                            plt.ylabel("amount of synapses")
                    elif "size" in key:
                        if "percentage" in key:
                            plt.ylabel("percentage of synapse size")
                        else:
                            plt.ylabel("average synapse size [µm²]")
                    filename = ("%s/%s_violin_%s_%s_2_%s_outgoing.png" % (
                            f_name, key, ct_dict[comp_ct1], ct_dict[comp_ct2], ct_dict[connected_ct]))
                    plt.savefig(filename)
                    plt.close()
                    sns.boxplot(data=results_for_plotting, palette=colours_pal)
                    plt.title('%s, %s, %s to %s' % (key, ct_dict[comp_ct1], ct_dict[comp_ct2], ct_dict[connected_ct]))
                    if "amount" in key:
                        if "percentage" in key:
                            plt.ylabel("percentage of synapses")
                        else:
                            plt.ylabel("amount of synapses")
                    elif "size" in key:
                        if "percentage" in key:
                            plt.ylabel("percentage of synapse size")
                        else:
                            plt.ylabel("average synapse size [µm²]")
                    filename = ("%s/%s_box_%s_%s_2_%s_outgoing.png" % (
                                f_name, key, ct_dict[comp_ct1], ct_dict[comp_ct2], ct_dict[connected_ct]))
                    plt.savefig(filename)
                    plt.close()
                sns.distplot(ct1_syn_dict[key],
                             hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                             kde=False, label=ct_dict[comp_ct1], bins=10)
                sns.distplot(ct2_syn_dict[key],
                             hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                             kde=False, label=ct_dict[comp_ct2], bins=10)
                plt.legend()
                plt.title('%s, %s, %s to %s' % (key, ct_dict[comp_ct1], ct_dict[comp_ct2], ct_dict[connected_ct]))
                plt.ylabel("count of cells")
                if "amount" in key:
                    if "percentage" in key:
                        plt.xlabel("percentage of synapses")
                    else:
                        plt.xlabel("amount of synapses")
                elif "size" in key:
                    if "percentage" in key:
                        plt.xlabel("percentage of synapse size")
                    else:
                        plt.xlabel("average synapse size [µm²]")
                filename = ("%s/%s_hist_%s_%s_2_%s_outgoing.png" % (
                            f_name, key, ct_dict[comp_ct1], ct_dict[comp_ct2], ct_dict[connected_ct]))
                plt.savefig(filename)
                plt.close()
                sns.distplot(ct1_syn_dict[key],
                             hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                             kde=False, label=ct_dict[comp_ct1], bins=10, norm_hist=True)
                sns.distplot(ct2_syn_dict[key],
                             hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                             kde=False, label=ct_dict[comp_ct2], bins=10, norm_hist=True)
                plt.legend()
                plt.title('%s, %s, %s to %s' % (key, ct_dict[comp_ct1], ct_dict[comp_ct2], ct_dict[connected_ct]))
                plt.ylabel("fraction of cells")
                if "length" in key:
                    plt.xlabel("pathlength in µm")
                elif "vol" in key:
                    if "percentage" in key:
                        plt.xlabel("% of whole dataset")
                    else:
                        plt.xlabel("volume in µm³")
                else:
                    plt.xlabel("distance in µm")
                filename = ("%s/%s_hist_norm_%s_%s_2_%s_outgoing.png" % (
                            f_name, key, ct_dict[comp_ct1], ct_dict[comp_ct2], ct_dict[connected_ct]))
                plt.savefig(filename)
                plt.close()

            ranksum_results.to_csv("%s/ranksum_%s_%s_2_%s_outgoing.csv" % (f_name, ct_dict[comp_ct1], ct_dict[comp_ct2], ct_dict[connected_ct]))

        plottime = time.time() - start
        print("%.2f sec for statistics and plotting" % plottime)
        time_stamps.append(time.time())
        step_idents.append('comparing celltypes')
        log.info("comparing celltypes via connectivity finished")


    def synapses_between2percentiles(ssd, sd_synssv, celltype,
                             percentile=0,
                             min_comp_len=100, min_syn_size=0.1, syn_prob_thresh=0.6):
        '''
        looks at basic connectivty parameters between two percentiles (preprocessed) within one celltype such as amount of synapses, average of synapses between cell types but also
        the average from one cell to the same other cell. Also looks at distribution of axo_dendritic synapses onto spines/shaft and the percentage of axo-somatic
        synapses. Uses cached synapse properties. Uses compartment_length per cell to ignore cells with not enough axon/dendrite
        spiness values: 0 = dendritic shaft, 1 = spine head, 2 = spine neck, 3 = other
        :param ssd: super-segmentation dataset
        :param sd_synssv: segmentation dataset for synapses.
        :param celltype: celltypes to be compared. j0256: STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7,
    #                      FS=8, LTS=9, NGF=10
        :param percentile: percentile which will be compared to 100 - percentile (need to be of full cells)
        :param min_comp_len: minimum length for axon/dendrite to have to include cell in analysis
        :param min_syn_size: minimum size for synapses
        :param syn_prob_thresh: threshold for synapse probability
        :return:
        '''
        start = time.time()
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   10: "NGF"}
        f_name = "u/arother/bio_analysis_results/dir_indir_pathway_analysis/211001_j0251v3_syn_conn_%s_p%i_mcl%i_sysi_%.2f_st_%.2f" % (
            ct_dict[celltype], percentile, min_comp_len, min_syn_size, syn_prob_thresh)
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('compartment volume estimation', log_dir=f_name + '/logs/')
        log.info(
            "parameters: celltype1 = %s, percentile = %i, min_comp_length = %.i, min_syn_size = %.2f, syn_prob_thresh = %.2f" %
            (ct_dict[celltype], percentile, min_comp_len, min_syn_size, syn_prob_thresh))
        time_stamps = [time.time()]
        step_idents = ['t-0']

        cellids1 = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_arr_%i_l.pkl" % (ct_dict[celltype],
                        percentile))
        cellids2 = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_arr_%i_h.pkl" % (ct_dict[celltype],
                                                                                                     100 - percentile))

        if percentile == 100 - percentile:
            percentile -= 1
        ct1_axon_length = np.zeros(len(cellids1))
        ct2_axon_length = np.zeros(len(cellids2))
        log.info("Step 1/4 Iterate over percentile %i in %s to check min_comp_len" % (percentile, ct_dict[celltype]))
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

        ct1_inds = ct1_axon_length > 0
        ct1_axon_length = ct1_axon_length[ct1_inds]
        cellids1 = cellids1[ct1_inds]

        ct1time = time.time() - start
        print("%.2f sec for iterating through %i percentile of %s cells" % (ct1time, percentile, ct_dict[celltype]))
        time_stamps.append(time.time())
        step_idents.append('iterating over %i percentile of %s cells' % (percentile, ct_dict[celltype]))

        log.info("Step 2/4 Iterate over percentiles %i in %s to check min_comp_len" % (100 - percentile, ct_dict[celltype]))
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

        ct2_inds = ct2_axon_length > 0
        ct2_axon_length = ct2_axon_length[ct2_inds]
        cellids2 = cellids2[ct2_inds]

        ct2time = time.time() - ct1time
        print("%.2f sec for iterating through %i percentile of %s cells" % (ct2time, 100 - percentile, ct_dict[celltype]))
        time_stamps.append(time.time())
        step_idents.append('iterating over %i percentile of %s cells' % (100 - percentile, ct_dict[celltype]))

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
        # select only those of celltype1 and celltype2
        ct_inds = np.any(m_cts == celltype, axis=1)
        m_cts = m_cts[ct_inds]
        m_ids = m_ids[ct_inds]
        m_axs = m_axs[ct_inds]
        m_ssv_partners = m_ssv_partners[ct_inds]
        m_sizes = m_sizes[ct_inds]
        m_spiness = m_spiness[ct_inds]
        # filter those with size below min_syn_size
        size_inds = m_sizes > min_syn_size
        m_cts = m_cts[size_inds]
        m_ids = m_ids[size_inds]
        m_axs = m_axs[size_inds]
        m_ssv_partners = m_ssv_partners[size_inds]
        m_sizes = m_sizes[size_inds]
        m_spiness = m_spiness[size_inds]
        # only axo-dendritic or axo-somatic synapses allowed
        axs_inds = np.any(m_axs == 1, axis=1)
        m_cts = m_cts[axs_inds]
        m_ids = m_ids[axs_inds]
        m_axs = m_axs[axs_inds]
        m_ssv_partners = m_ssv_partners[axs_inds]
        m_sizes = m_sizes[axs_inds]
        m_spiness = m_spiness[axs_inds]
        den_so = np.array([0, 2])
        den_so_inds = np.any(np.in1d(m_axs, den_so).reshape(len(m_axs), 2), axis=1)
        m_cts = m_cts[den_so_inds]
        m_ids = m_ids[den_so_inds]
        m_axs = m_axs[den_so_inds]
        m_ssv_partners = m_ssv_partners[den_so_inds]
        m_sizes = m_sizes[den_so_inds]
        m_spiness = m_spiness[den_so_inds]

        prepsyntime = time.time() - ct2time
        print("%.2f sec for preprocessing synapses" % prepsyntime)
        time_stamps.append(time.time())
        step_idents.append('preprocessing synapses')

        log.info("Step 3b: iterate over synapses to get synaptic connectivity parameters")
        ct2_2_ct1_syn_dict = {"cellids": cellids1, "syn amount": np.zeros(len(cellids1)),
                              "sum syn size": np.zeros(len(cellids1)),
                              "amount shaft syn": np.zeros(len(cellids1)),
                              "sum size shaft syn": np.zeros(len(cellids1)),
                              "amount spine head syn": np.zeros(len(cellids1)),
                              "sum size spine head syn": np.zeros(len(cellids1)),
                              "amount soma syn": np.zeros(len(cellids1)), "sum size soma syn": np.zeros(len(cellids1)),
                              "amount spine neck syn": np.zeros(len(cellids1)),
                              "sum size spine neck syn": np.zeros(len(cellids1))}
        ct2_2_ct1_percell_syn_amount = np.zeros((len(cellids1), len(cellids2))).astype(float)
        ct2_2_ct1_percell_syn_size = np.zeros((len(cellids1), len(cellids2))).astype(float)
        ct1_2_ct2_syn_dict = {"cellids": cellids2, "syn amount": np.zeros(len(cellids2)),
                              "sum syn size": np.zeros(len(cellids2)),
                              "amount shaft syn": np.zeros(len(cellids2)),
                              "sum size shaft syn": np.zeros(len(cellids2)),
                              "amount spine head syn": np.zeros(len(cellids2)),
                              "sum size spine head syn": np.zeros(len(cellids2)),
                              "amount soma syn": np.zeros(len(cellids2)), "sum size soma syn": np.zeros(len(cellids2)),
                              "amount spine neck syn": np.zeros(len(cellids2)),
                              "sum size spine neck syn": np.zeros(len(cellids2))}
        ct1_2_ct2_percell_syn_amount = np.zeros((len(cellids2), len(cellids1))).astype(float)
        ct1_2_ct2_percell_syn_size = np.zeros((len(cellids2), len(cellids1))).astype(float)
        for i, syn_id in enumerate(tqdm(m_ids)):
            syn_ax = m_axs[i]
            # remove cells that are not in cellids:
            if not np.any(np.in1d(m_ssv_partners[i], cellids1)):
                continue
            if not np.any(np.in1d(m_ssv_partners[i], cellids2)):
                continue
            if syn_ax[0] == 1:
                ssv_ax, ssv_deso = m_ssv_partners[i]
                spin_ax, spin_deso = m_spiness[i]
                ax, deso = syn_ax
            else:
                ssv_deso, ssv_ax = m_ssv_partners[i]
                spin_deso, spin_ax = m_spiness[i]
                deso, ax = syn_ax
            syn_size = m_sizes[i]
            if ssv_ax in cellids1:
                cell2_ind = np.where(ct1_2_ct2_syn_dict["cellids"] == ssv_deso)[0]
                cell1_ind = np.where(ct2_2_ct1_syn_dict["cellids"] == ssv_ax)[0]
                ct1_2_ct2_syn_dict["syn amount"][cell2_ind] += 1
                ct1_2_ct2_syn_dict["sum syn size"][cell2_ind] += syn_size

                ct1_2_ct2_percell_syn_amount[cell2_ind, cell1_ind] += 1
                ct1_2_ct2_percell_syn_size[cell2_ind, cell1_ind] += syn_size
                if deso == 0:
                    if spin_deso == 0:
                        ct1_2_ct2_syn_dict["amount shaft syn"][cell2_ind] += 1
                        ct1_2_ct2_syn_dict["sum size shaft syn"][cell2_ind] += syn_size
                    elif spin_deso == 1:
                        ct1_2_ct2_syn_dict["amount spine head syn"][cell2_ind] += 1
                        ct1_2_ct2_syn_dict["sum size spine head syn"][cell2_ind] += syn_size
                    elif spin_deso == 2:
                        ct1_2_ct2_syn_dict["amount spine neck syn"][cell2_ind] += 1
                        ct1_2_ct2_syn_dict["sum size spine neck syn"][cell2_ind] += syn_size
                else:
                    ct1_2_ct2_syn_dict["amount soma syn"][cell2_ind] += 1
                    ct1_2_ct2_syn_dict["sum size soma syn"][cell2_ind] += syn_size
            else:
                cell1_ind = np.where(ct2_2_ct1_syn_dict["cellids"] == ssv_deso)[0]
                cell2_ind = np.where(ct1_2_ct2_syn_dict["cellids"] == ssv_ax)[0]
                ct2_2_ct1_syn_dict["syn amount"][cell1_ind] += 1
                ct2_2_ct1_syn_dict["sum syn size"][cell1_ind] += syn_size
                ct2_2_ct1_percell_syn_amount[cell1_ind, cell2_ind] += 1
                ct2_2_ct1_percell_syn_size[cell1_ind, cell2_ind] += syn_size
                if deso == 0:
                    if spin_deso == 0:
                        ct2_2_ct1_syn_dict["amount shaft syn"][cell1_ind] += 1
                        ct2_2_ct1_syn_dict["sum size shaft syn"][cell1_ind] += syn_size
                    elif spin_deso == 1:
                        ct2_2_ct1_syn_dict["amount spine head syn"][cell1_ind] += 1
                        ct2_2_ct1_syn_dict["sum size spine head syn"][cell1_ind] += syn_size
                    elif spin_deso == 2:
                        ct2_2_ct1_syn_dict["amount spine neck syn"][cell1_ind] += 1
                        ct2_2_ct1_syn_dict["sum size spine neck syn"][cell1_ind] += syn_size
                else:
                    ct2_2_ct1_syn_dict["amount soma syn"][cell1_ind] += 1
                    ct2_2_ct1_syn_dict["sum size soma syn"][cell1_ind] += syn_size

        syntime = time.time() - prepsyntime
        print("%.2f sec for processing synapses" % syntime)
        time_stamps.append(time.time())
        step_idents.append('processing synapses')

        log.info(
            "Step 4/4: calculate last parameters (average syn size, average syn amount and syn size between 2 cells) and plotting")

        ct2_syn_inds = ct1_2_ct2_syn_dict["syn amount"] > 0
        for key in ct1_2_ct2_syn_dict.keys():
            ct1_2_ct2_syn_dict[key] = ct1_2_ct2_syn_dict[key][ct2_syn_inds]

        ct1_2_ct2_syn_dict["avg syn size"] = ct1_2_ct2_syn_dict["sum syn size"] / ct1_2_ct2_syn_dict["syn amount"]
        ct1_2_ct2_syn_dict["avg size shaft syn"] = ct1_2_ct2_syn_dict["sum size shaft syn"] / ct1_2_ct2_syn_dict[
            "amount shaft syn"]
        ct1_2_ct2_syn_dict["avg size spine head syn"] = ct1_2_ct2_syn_dict["sum size spine head syn"] / \
                                                        ct1_2_ct2_syn_dict[
                                                            "amount spine head syn"]
        ct1_2_ct2_syn_dict["avg size spine neck syn"] = ct1_2_ct2_syn_dict["sum size spine neck syn"] / \
                                                        ct1_2_ct2_syn_dict[
                                                            "amount spine neck syn"]
        ct1_2_ct2_syn_dict["avg size soma syn"] = ct1_2_ct2_syn_dict["sum size soma syn"] / ct1_2_ct2_syn_dict[
            "amount soma syn"]
        ct1_2_ct2_syn_dict["percentage shaft syn amount"] = ct1_2_ct2_syn_dict["amount shaft syn"] / ct1_2_ct2_syn_dict[
            "syn amount"] * 100
        ct1_2_ct2_syn_dict["percentage spine head syn amount"] = ct1_2_ct2_syn_dict["amount spine head syn"] / \
                                                                 ct1_2_ct2_syn_dict[
                                                                     "syn amount"] * 100
        ct1_2_ct2_syn_dict["percentage spine neck syn amount"] = ct1_2_ct2_syn_dict["amount spine neck syn"] / \
                                                                 ct1_2_ct2_syn_dict[
                                                                     "syn amount"] * 100
        ct1_2_ct2_syn_dict["percentage soma syn amount"] = ct1_2_ct2_syn_dict["amount soma syn"] / ct1_2_ct2_syn_dict[
            "syn amount"] * 100
        ct1_2_ct2_syn_dict["percentage shaft syn size"] = ct1_2_ct2_syn_dict["sum size shaft syn"] / ct1_2_ct2_syn_dict[
            "sum syn size"] * 100
        ct1_2_ct2_syn_dict["percentage spine head syn size"] = ct1_2_ct2_syn_dict["sum size spine head syn"] / \
                                                               ct1_2_ct2_syn_dict[
                                                                   "sum syn size"] * 100
        ct1_2_ct2_syn_dict["percentage spine neck syn size"] = ct1_2_ct2_syn_dict["sum size spine neck syn"] / \
                                                               ct1_2_ct2_syn_dict[
                                                                   "sum syn size"] * 100
        ct1_2_ct2_syn_dict["percentage soma syn size"] = ct1_2_ct2_syn_dict["sum size soma syn"] / ct1_2_ct2_syn_dict[
            "sum syn size"] * 100

        ct1_syn_inds = ct2_2_ct1_syn_dict["syn amount"] > 0
        for key in ct2_2_ct1_syn_dict.keys():
            ct2_2_ct1_syn_dict[key] = ct2_2_ct1_syn_dict[key][ct1_syn_inds]

        ct2_2_ct1_syn_dict["avg syn size"] = ct2_2_ct1_syn_dict["sum syn size"] / ct2_2_ct1_syn_dict["syn amount"]
        ct2_2_ct1_syn_dict["avg size shaft syn"] = ct2_2_ct1_syn_dict["sum size shaft syn"] / ct2_2_ct1_syn_dict[
            "amount shaft syn"]
        ct2_2_ct1_syn_dict["avg size spine head syn"] = ct2_2_ct1_syn_dict["sum size spine head syn"] / \
                                                        ct2_2_ct1_syn_dict[
                                                            "amount spine head syn"]
        ct2_2_ct1_syn_dict["avg size spine neck syn"] = ct2_2_ct1_syn_dict["sum size spine neck syn"] / \
                                                        ct2_2_ct1_syn_dict[
                                                            "amount spine neck syn"]
        ct2_2_ct1_syn_dict["avg size soma syn"] = ct2_2_ct1_syn_dict["sum size soma syn"] / ct2_2_ct1_syn_dict[
            "amount soma syn"]
        ct2_2_ct1_syn_dict["percentage shaft syn amount"] = ct2_2_ct1_syn_dict["amount shaft syn"] / \
                                                            ct2_2_ct1_syn_dict["syn amount"] * 100
        ct2_2_ct1_syn_dict["percentage spine head syn amount"] = ct2_2_ct1_syn_dict["amount spine head syn"] / \
                                                                 ct2_2_ct1_syn_dict[
                                                                     "syn amount"] * 100
        ct2_2_ct1_syn_dict["percentage spine neck syn amount"] = ct2_2_ct1_syn_dict["amount spine neck syn"] / \
                                                                 ct2_2_ct1_syn_dict[
                                                                     "syn amount"] * 100
        ct2_2_ct1_syn_dict["percentage soma syn amount"] = ct2_2_ct1_syn_dict["amount soma syn"] / \
                                                           ct2_2_ct1_syn_dict[
                                                               "syn amount"] * 100
        ct2_2_ct1_syn_dict["percentage shaft syn size"] = ct2_2_ct1_syn_dict["sum size shaft syn"] / \
                                                          ct2_2_ct1_syn_dict[
                                                              "sum syn size"] * 100
        ct2_2_ct1_syn_dict["percentage spine head syn size"] = ct2_2_ct1_syn_dict["sum size spine head syn"] / \
                                                               ct2_2_ct1_syn_dict[
                                                                   "sum syn size"] * 100
        ct2_2_ct1_syn_dict["percentage spine neck syn size"] = ct2_2_ct1_syn_dict["sum size spine neck syn"] / \
                                                               ct2_2_ct1_syn_dict[
                                                                   "sum syn size"] * 100
        ct2_2_ct1_syn_dict["percentage soma syn size"] = ct2_2_ct1_syn_dict["sum size soma syn"] / \
                                                         ct2_2_ct1_syn_dict[
                                                             "sum syn size"] * 100

        ct1_2_ct2_syn_dict.pop("sum syn size")
        ct1_2_ct2_syn_dict.pop("sum size soma syn")
        ct1_2_ct2_syn_dict.pop("sum size spine head syn")
        ct1_2_ct2_syn_dict.pop("sum size spine neck syn")
        ct1_2_ct2_syn_dict.pop("sum size shaft syn")

        ct2_2_ct1_syn_dict.pop("sum syn size")
        ct2_2_ct1_syn_dict.pop("sum size soma syn")
        ct2_2_ct1_syn_dict.pop("sum size spine head syn")
        ct2_2_ct1_syn_dict.pop("sum size spine neck syn")
        ct2_2_ct1_syn_dict.pop("sum size shaft syn")

        # average synapse amount and size for one cell form ct1 to one cell of ct2
        ct1_2_ct2_percell_syn_amount = ct1_2_ct2_percell_syn_amount[ct2_syn_inds]
        ct1_2_ct2_percell_syn_amount[ct1_2_ct2_percell_syn_amount == 0] = np.nan
        ct1_2_ct2_syn_dict["avg amount one cell"] = np.nanmean(ct1_2_ct2_percell_syn_amount, axis=1)

        ct1_2_ct2_percell_syn_size = ct1_2_ct2_percell_syn_size[ct2_syn_inds]
        ct1_2_ct2_percell_syn_size[ct1_2_ct2_percell_syn_size == 0] = np.nan
        ct1_2_ct2_syn_dict["avg syn size one cell"] = np.nanmean(ct1_2_ct2_percell_syn_size, axis=1) / \
                                                      ct1_2_ct2_syn_dict["avg amount one cell"]

        ct2_2_ct1_percell_syn_amount = ct2_2_ct1_percell_syn_amount[ct1_syn_inds]
        ct2_2_ct1_percell_syn_amount[ct2_2_ct1_percell_syn_amount == 0] = np.nan
        ct2_2_ct1_syn_dict["avg amount one cell"] = np.nanmean(ct2_2_ct1_percell_syn_amount, axis=1)

        ct2_2_ct1_percell_syn_size = ct2_2_ct1_percell_syn_size[ct1_syn_inds]
        ct2_2_ct1_percell_syn_size[ct2_2_ct1_percell_syn_size == 0] = np.nan
        ct2_2_ct1_syn_dict["avg syn size one cell"] = np.nanmean(ct2_2_ct1_percell_syn_size, axis=1) / \
                                                      ct2_2_ct1_syn_dict["avg amount one cell"]

        write_obj2pkl("%s/%s_%i_2_%i_dict.pkl" % (f_name, ct_dict[celltype], percentile, 100 - percentile), ct1_2_ct2_syn_dict)
        ct1_2_ct2_pd = pd.DataFrame(ct1_2_ct2_syn_dict)
        ct1_2_ct2_pd.to_csv("%s/%s_%i_2_%i_dict.csv" % (f_name, ct_dict[celltype], percentile, 100 - percentile))

        write_obj2pkl("%s/%s_%i_2_%i_dict.pkl" % (f_name, ct_dict[celltype], 100 - percentile, percentile), ct2_2_ct1_syn_dict)
        ct2_2_ct1_pd = pd.DataFrame(ct2_2_ct1_syn_dict)
        ct2_2_ct1_pd.to_csv("%s/%s_%i_2_%i_dict.csv" % (f_name, ct_dict[celltype], 100 - percentile, percentile))

        # plot parameters as distplot
        for key in ct1_2_ct2_syn_dict.keys():
            if "ids" in key:
                continue
            sns.distplot(ct1_2_ct2_syn_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "steelblue"},
                         kde=False)
            plt.ylabel("count of cells")
            if "amount" in key:
                plt.xlabel("amount of synapses")
            elif "size" in key:
                plt.xlabel("average synapse size [µm²]")
            elif "percentage" in key:
                plt.xlabel("percentage of synapses")
            plt.title("%s from %i to %i in %s" % (key, percentile, 100 - percentile, ct_dict[celltype]))
            plt.savefig("%s/%s_%i_2_%i_%s.png" % (f_name, key, percentile, 100 - percentile, ct_dict[celltype]))
            plt.close()
            sns.distplot(ct2_2_ct1_syn_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "steelblue"},
                         kde=False, bins=10)
            plt.ylabel("count of cells")
            if "amount" in key:
                if "percentage" in key:
                    plt.xlabel("percentage of synapses")
                else:
                    plt.xlabel("amount of synapses")
            elif "size" in key:
                if "percentage" in key:
                    plt.xlabel("percentage of synapse size")
                else:
                    plt.xlabel("average synapse size [µm²]")

            plt.title("%s from %i to %i in %s" % (key, 100 - percentile, percentile, ct_dict[celltype]))
            plt.savefig("%s/%s_%i_2_%i_%s.png" % (f_name, key, 100 - percentile, percentile, ct_dict[celltype]))
            plt.close()

        # make violin plots for amount and size (absolute, relative) for different compartments
        x_labels = ["spine head", "spine neck", "shaft", "soma"]
        ticks = np.arange(4)
        sns.violinplot(data=[ct1_2_ct2_pd["amount spine head syn"], ct1_2_ct2_pd["amount spine neck syn"],
                             ct1_2_ct2_pd["amount shaft syn"], ct1_2_ct2_pd["amount soma syn"]], inner="box")
        sns.stripplot(data=[ct1_2_ct2_pd["amount spine head syn"], ct1_2_ct2_pd["amount spine neck syn"],
                            ct1_2_ct2_pd["amount shaft syn"], ct1_2_ct2_pd["amount soma syn"]], color="black",
                      alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("amount of synapses")
        plt.title("synapse amount from %i to % i in %s" % (percentile, 100 - percentile, ct_dict[celltype]))
        plt.savefig("%s/syn_amount_violin_%i_2_%i_%s.png" % (f_name, percentile, 100 - percentile, ct_dict[celltype]))
        plt.close()

        sns.boxplot(data=[ct1_2_ct2_pd["amount spine head syn"], ct1_2_ct2_pd["amount spine neck syn"],
                          ct1_2_ct2_pd["amount shaft syn"], ct1_2_ct2_pd["amount soma syn"]])
        sns.stripplot(data=[ct1_2_ct2_pd["amount spine head syn"], ct1_2_ct2_pd["amount spine neck syn"],
                            ct1_2_ct2_pd["amount shaft syn"], ct1_2_ct2_pd["amount soma syn"]], color="black",
                      alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("amount of synapses")
        plt.title("synapse amount from %i to %i in %s" % (percentile, 100 - percentile, ct_dict[celltype]))
        plt.savefig("%s/syn_amount_box_%i_2_%i_%s.png" % (f_name, percentile, 100 - percentile, ct_dict[celltype]))
        plt.close()

        sns.violinplot(data=[ct1_2_ct2_pd["avg size spine head syn"], ct1_2_ct2_pd["avg size spine neck syn"],
                             ct1_2_ct2_pd["avg size shaft syn"], ct1_2_ct2_pd["avg size soma syn"]], inner="box")
        sns.stripplot(data=[ct1_2_ct2_pd["avg size spine head syn"], ct1_2_ct2_pd["avg size spine neck syn"],
                            ct1_2_ct2_pd["avg size shaft syn"], ct1_2_ct2_pd["avg size soma syn"]], color="black",
                      alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("average size of synapses [µm²]")
        plt.title("average synapse size from %i to %i in %s" % (percentile, 100 - percentile, ct_dict[celltype]))
        plt.savefig("%s/syn_size_violin_%i_2_%i_%s.png" % (f_name, percentile, 100 - percentile, ct_dict[celltype]))
        plt.close()

        sns.boxplot(data=[ct1_2_ct2_pd["avg size spine head syn"], ct1_2_ct2_pd["avg size spine neck syn"],
                          ct1_2_ct2_pd["avg size shaft syn"], ct1_2_ct2_pd["avg size soma syn"]])
        sns.stripplot(data=[ct1_2_ct2_pd["avg size spine head syn"], ct1_2_ct2_pd["avg size spine neck syn"],
                            ct1_2_ct2_pd["avg size shaft syn"], ct1_2_ct2_pd["avg size soma syn"]], color="black",
                      alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("average size of synapses [µm²]")
        plt.title("average synapse size from %i to %i in %s" % (percentile, 100 - percentile, ct_dict[celltype]))
        plt.savefig("%s/syn_size_box_%i_2_%i_%s.png" % (f_name, percentile, 100 - percentile, ct_dict[celltype]))
        plt.close()

        sns.violinplot(
            data=[ct1_2_ct2_pd["percentage spine head syn amount"], ct1_2_ct2_pd["percentage spine neck syn amount"],
                  ct1_2_ct2_pd["percentage shaft syn amount"], ct1_2_ct2_pd["percentage soma syn amount"]], inner="box")
        sns.stripplot(
            data=[ct1_2_ct2_pd["percentage spine head syn amount"], ct1_2_ct2_pd["percentage spine neck syn amount"],
                  ct1_2_ct2_pd["percentage shaft syn amount"], ct1_2_ct2_pd["percentage soma syn amount"]],
            color="black",
            alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("percentage of synapses")
        plt.title("percentage of synapses from %i to %i in %s" % (percentile, 100 - percentile, ct_dict[celltype]))
        plt.savefig("%s/syn_perc_violin_%i_2_%i_%s.png" % (f_name, percentile, 100 - percentile, ct_dict[celltype]))
        plt.close()

        sns.boxplot(
            data=[ct1_2_ct2_pd["percentage spine head syn amount"], ct1_2_ct2_pd["percentage spine neck syn amount"],
                  ct1_2_ct2_pd["percentage shaft syn amount"], ct1_2_ct2_pd["percentage soma syn amount"]])
        sns.stripplot(
            data=[ct1_2_ct2_pd["percentage spine head syn amount"], ct1_2_ct2_pd["percentage spine neck syn amount"],
                  ct1_2_ct2_pd["percentage shaft syn amount"], ct1_2_ct2_pd["percentage soma syn amount"]],
            color="black",
            alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("percentage of synapses")
        plt.title("percentage of synapses from %i to %i in %s" % (percentile, 100 - percentile, ct_dict[celltype]))
        plt.savefig("%s/syn_perc_box_%i_2_%i_%s.png" % (f_name, percentile, 100 - percentile, ct_dict[celltype]))
        plt.close()

        sns.violinplot(
            data=[ct1_2_ct2_pd["percentage spine head syn size"], ct1_2_ct2_pd["percentage spine neck syn size"],
                  ct1_2_ct2_pd["percentage shaft syn size"], ct1_2_ct2_pd["percentage soma syn size"]], inner="box")
        sns.stripplot(
            data=[ct1_2_ct2_pd["percentage spine head syn size"], ct1_2_ct2_pd["percentage spine neck syn size"],
                  ct1_2_ct2_pd["percentage shaft syn size"], ct1_2_ct2_pd["percentage soma syn size"]], color="black",
            alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("percentage of synapse size")
        plt.title("percentage of synapse size from %i to %i in %s" % (percentile, 100 - percentile, ct_dict[celltype]))
        plt.savefig("%s/syn_size_perc_violin_%i_2_%i_%s.png" % (f_name, percentile, 100 - percentile, ct_dict[celltype]))
        plt.close()

        sns.boxplot(
            data=[ct1_2_ct2_pd["percentage spine head syn size"], ct1_2_ct2_pd["percentage spine neck syn size"],
                  ct1_2_ct2_pd["percentage shaft syn size"], ct1_2_ct2_pd["percentage soma syn size"]])
        sns.stripplot(
            data=[ct1_2_ct2_pd["percentage spine head syn size"], ct1_2_ct2_pd["percentage spine neck syn size"],
                  ct1_2_ct2_pd["percentage shaft syn size"], ct1_2_ct2_pd["percentage soma syn size"]], color="black",
            alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("percentage of synapse size")
        plt.title("percentage of synapse size from %i to %i in %s" % (percentile, 100 - percentile, ct_dict[celltype]))
        plt.savefig("%s/syn_size_perc_box_%i_2_%i_%s.png" % (f_name, percentile, 100 - percentile, ct_dict[celltype]))
        plt.close()

        sns.violinplot(data=[ct2_2_ct1_pd["amount spine head syn"], ct2_2_ct1_pd["amount spine neck syn"],
                             ct2_2_ct1_pd["amount shaft syn"], ct2_2_ct1_pd["amount soma syn"]], inner="box")
        sns.stripplot(data=[ct2_2_ct1_pd["amount spine head syn"], ct2_2_ct1_pd["amount spine neck syn"],
                            ct2_2_ct1_pd["amount shaft syn"], ct2_2_ct1_pd["amount soma syn"]], color="black",
                      alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("amount of synapses")
        plt.title("synapse amount from %i to %i in %s" % (100 - percentile, percentile, ct_dict[celltype]))
        plt.savefig("%s/syn_amount_violin_%i_2_%i_%s.png" % (f_name, 100 - percentile, percentile, ct_dict[celltype]))
        plt.close()

        sns.boxplot(data=[ct2_2_ct1_pd["amount spine head syn"], ct2_2_ct1_pd["amount spine neck syn"],
                          ct2_2_ct1_pd["amount shaft syn"], ct2_2_ct1_pd["amount soma syn"]])
        sns.stripplot(data=[ct2_2_ct1_pd["amount spine head syn"], ct2_2_ct1_pd["amount spine neck syn"],
                            ct2_2_ct1_pd["amount shaft syn"], ct2_2_ct1_pd["amount soma syn"]], color="black",
                      alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("amount of synapses")
        plt.title("synapse amount from %i to %i in %s" % (100 - percentile, percentile, ct_dict[celltype]))
        plt.savefig("%s/syn_amount_box_%i_2_%i_%s.png" % (f_name, 100 - percentile, percentile, ct_dict[celltype]))
        plt.close()

        sns.violinplot(data=[ct2_2_ct1_pd["avg size spine head syn"], ct2_2_ct1_pd["avg size spine neck syn"],
                             ct2_2_ct1_pd["avg size shaft syn"], ct2_2_ct1_pd["avg size soma syn"]], inner="box")
        sns.stripplot(data=[ct2_2_ct1_pd["avg size spine head syn"], ct2_2_ct1_pd["avg size spine neck syn"],
                            ct2_2_ct1_pd["avg size shaft syn"], ct2_2_ct1_pd["avg size soma syn"]], color="black",
                      alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("average size of synapses [µm²]")
        plt.title("average synapse size from %i to %i in %s" % (100 - percentile, percentile, ct_dict[celltype]))
        plt.savefig("%s/syn_size_violin_%i_2_%i_%s.png" % (f_name, 100 - percentile, percentile, ct_dict[celltype]))
        plt.close()

        sns.boxplot(data=[ct2_2_ct1_pd["avg size spine head syn"], ct2_2_ct1_pd["avg size spine neck syn"],
                          ct2_2_ct1_pd["avg size shaft syn"], ct2_2_ct1_pd["avg size soma syn"]])
        sns.stripplot(data=[ct2_2_ct1_pd["avg size spine head syn"], ct2_2_ct1_pd["avg size spine neck syn"],
                            ct2_2_ct1_pd["avg size shaft syn"], ct2_2_ct1_pd["avg size soma syn"]], color="black",
                      alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("average size of synapses [µm²]")
        plt.title("average synapse size from %i to %i in %s" % (100 - percentile, percentile, ct_dict[celltype]))
        plt.savefig("%s/syn_size_box_%i_2_%i_%s.png" % (f_name, 100 - percentile, percentile, ct_dict[celltype]))
        plt.close()

        sns.violinplot(
            data=[ct2_2_ct1_pd["percentage spine head syn amount"], ct2_2_ct1_pd["percentage spine neck syn amount"],
                  ct2_2_ct1_pd["percentage shaft syn amount"], ct2_2_ct1_pd["percentage soma syn amount"]], inner="box")
        sns.stripplot(
            data=[ct2_2_ct1_pd["percentage spine head syn amount"], ct2_2_ct1_pd["percentage spine neck syn amount"],
                  ct2_2_ct1_pd["percentage shaft syn amount"], ct2_2_ct1_pd["percentage soma syn amount"]],
            color="black",
            alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("percentage of synapses")
        plt.title("percentage of synapses from %i to %i in %s" % (100 - percentile, percentile, ct_dict[celltype]))
        plt.savefig("%s/syn_perc_violin_%i_2_%i_%s.png" % (f_name, 100 - percentile, percentile, ct_dict[celltype]))
        plt.close()

        sns.boxplot(
            data=[ct2_2_ct1_pd["percentage spine head syn amount"], ct2_2_ct1_pd["percentage spine neck syn amount"],
                  ct2_2_ct1_pd["percentage shaft syn amount"], ct2_2_ct1_pd["percentage soma syn amount"]])
        sns.stripplot(
            data=[ct2_2_ct1_pd["percentage spine head syn amount"], ct2_2_ct1_pd["percentage spine neck syn amount"],
                  ct2_2_ct1_pd["percentage shaft syn amount"], ct2_2_ct1_pd["percentage soma syn amount"]],
            color="black",
            alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("percentage of synapses")
        plt.title("percentage of synapses from %i to %i in %s" % (100 - percentile, percentile, ct_dict[celltype]))
        plt.savefig("%s/syn_perc_box_%i_2_%i_%s.png" % (f_name, 100 - percentile, percentile, ct_dict[celltype]))
        plt.close()

        sns.violinplot(
            data=[ct2_2_ct1_pd["percentage spine head syn size"], ct2_2_ct1_pd["percentage spine neck syn size"],
                  ct2_2_ct1_pd["percentage shaft syn size"], ct2_2_ct1_pd["percentage soma syn size"]], inner="box")
        sns.stripplot(
            data=[ct2_2_ct1_pd["percentage spine head syn size"], ct2_2_ct1_pd["percentage spine neck syn size"],
                  ct2_2_ct1_pd["percentage shaft syn size"], ct2_2_ct1_pd["percentage soma syn size"]], color="black",
            alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("percentage of synapse size")
        plt.title("percentage of synapse size from %i to %i %s" % (100 - percentile, percentile, ct_dict[celltype]))
        plt.savefig("%s/syn_size_perc_violin_%i_2_%i_%s.png" % (f_name, 100 - percentile, percentile, ct_dict[celltype]))
        plt.close()

        sns.boxplot(
            data=[ct2_2_ct1_pd["percentage spine head syn size"], ct2_2_ct1_pd["percentage spine neck syn size"],
                  ct2_2_ct1_pd["percentage shaft syn size"], ct2_2_ct1_pd["percentage soma syn size"]])
        sns.stripplot(
            data=[ct2_2_ct1_pd["percentage spine head syn size"], ct2_2_ct1_pd["percentage spine neck syn size"],
                  ct2_2_ct1_pd["percentage shaft syn size"], ct2_2_ct1_pd["percentage soma syn size"]], color="black",
            alpha=0.2)
        plt.xticks(ticks=ticks, labels=x_labels)
        plt.ylabel("percentage of synapse size")
        plt.title("percentage of synapse size from %i to %i in %s" % (100 - percentile, percentile, ct_dict[celltype]))
        plt.savefig("%s/syn_size_perc_box_%i_2_%i_%s.png" % (f_name, 100 - percentile, percentile, ct_dict[celltype]))
        plt.close()

        plottime = time.time() - syntime
        print("%.2f sec for calculating parameters, plotting" % plottime)
        time_stamps.append(time.time())
        step_idents.append('calculating last parameters, plotting')

        log.info(
            "Connectivity analysis between 2 percentiles (%s, %s) in %s finished" % (percentile, 100 - percentile, ct_dict[celltype]))


    def compare_connectvity_percentiles(celltype, percentile, connected_ct=None, foldername_ct1=None, foldername_ct2=None,
                            min_comp_len=100):
        '''
        compares connectivity parameters between two percentiles (percentile, 100 - percentile) within one celltypes or connectivity of another celltype to the two celltypes. Connectivity parameters are calculated in
        synapses_between2cts. Parameters include synapse amount and average synapse size, as well as amount and average synapse size in shaft, soma, spine head and spine neck.
        P-value will be calculated with scipy.stats.ranksum.
        :param celltype celltype of percentiles to be compared
        :param percentile: given percentile will be compared to 100 - percentile
        :param connected_ct: if given, connectivity of this celltype to two others will be compared
        :param foldername_ct1, foldername_ct2: foldernames where parameters of connectivity are stored
        :param min_comp_len: minimum compartment length
        :return:
        '''
        start = time.time()
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   10: "NGF"}
        if connected_ct != None:
            f_name = "u/arother/bio_analysis_results/dir_indir_pathway_analysis/211001_j0251v3__%s_p%i_%s_syn_con_comp_mcl%i" % (
                ct_dict[celltype], percentile, ct_dict[connected_ct], min_comp_len)
        else:
            f_name = "u/arother/bio_analysis_results/dir_indir_pathway_analysis/211001_j0251v3__%s_p%i_syn_con_comp_mcl%i" % (
                ct_dict[celltype], percentile, min_comp_len)
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('compare connectivty between two celltypes', log_dir=f_name + '/logs/')
        if connected_ct != None:
            log.info("parameters: celltype = %s, percentile = %i , connected ct = %s, min_comp_length = %.i" % (
                ct_dict[celltype], percentile, ct_dict[connected_ct], min_comp_len))
        else:
            log.info("parameters: celltype = %s, percentile = %i, min_comp_length = %.i" % (
                ct_dict[celltype], percentile, min_comp_len))
        time_stamps = [time.time()]
        step_idents = ['t-0']
        if percentile == 100 - percentile:
            percentile -= 1
        if connected_ct != None:
            ct2_syn_dict = load_pkl2obj(
                "%s/%s_2_%i_%s_dict.pkl" % (foldername_ct2, ct_dict[connected_ct], 100 - percentile, ct_dict[celltype]))
            ct1_syn_dict = load_pkl2obj(
                "%s/%s_2_%i_%s_dict.pkl" % (foldername_ct1, ct_dict[connected_ct], percentile, ct_dict[celltype]))
        else:
            ct2_syn_dict = load_pkl2obj(
                "%s/%s_%i_2_%i_dict.pkl" % (foldername_ct2, ct_dict[celltype], percentile, 100 - percentile))
            ct1_syn_dict = load_pkl2obj(
                "%s/%s_%i_2_%i_dict.pkl" % (foldername_ct1, ct_dict[celltype], 100 - percentile, percentile))
        syn_dict_keys = list(ct1_syn_dict.keys())
        log.info("compute statistics for comparison, create violinplot and histogram")
        ranksum_results = pd.DataFrame(columns=syn_dict_keys[1:], index=["stats", "p value"])
        colours_pal = {percentile: "gray", 100 - percentile: "darkturquoise"}

        # plot distribution of compartments togehter
        len_ct1 = len(ct1_syn_dict["amount spine head syn"])
        len_ct2 = len(ct2_syn_dict["amount spine head syn"])
        sum_len = len_ct1 + len_ct2
        results_for_plotting_comps = pd.DataFrame(
            columns=["syn amount", "avg syn size", "percentage amount", "percentage syn size", "percentile",
                     "compartment"], index=range(sum_len * 4))
        results_for_plotting_comps["compartment"] = str()
        results_for_plotting_comps.loc[0:sum_len - 1, "compartment"] = "spine head"
        results_for_plotting_comps.loc[0:len_ct1 - 1, "percentile"] = percentile
        results_for_plotting_comps.loc[len_ct1: sum_len - 1, "percentile"] = 100 - percentile
        results_for_plotting_comps.loc[0:len_ct1 - 1, "syn amount"] = ct1_syn_dict["amount spine head syn"]
        results_for_plotting_comps.loc[len_ct1:sum_len - 1, "syn amount"] = ct2_syn_dict["amount spine head syn"]
        results_for_plotting_comps.loc[0:len_ct1 - 1, "avg syn size"] = ct1_syn_dict["avg size spine head syn"]
        results_for_plotting_comps.loc[len_ct1:sum_len - 1, "avg syn size"] = ct2_syn_dict["avg size spine head syn"]
        results_for_plotting_comps.loc[0:len_ct1 - 1, "percentage amount"] = ct1_syn_dict[
            "percentage spine head syn amount"]
        results_for_plotting_comps.loc[len_ct1:sum_len - 1, "percentage amount"] = ct2_syn_dict[
            "percentage spine head syn amount"]
        results_for_plotting_comps.loc[0:len_ct1 - 1, "percentage syn size"] = ct1_syn_dict[
            "percentage spine head syn size"]
        results_for_plotting_comps.loc[len_ct1:sum_len - 1, "percentage syn size"] = ct2_syn_dict[
            "percentage spine head syn size"]
        results_for_plotting_comps.loc[sum_len:sum_len * 2 - 1, "compartment"] = "spine neck"
        results_for_plotting_comps.loc[sum_len:sum_len + len_ct1 - 1, "percentile"] = percentile
        results_for_plotting_comps.loc[sum_len + len_ct1:sum_len * 2 - 1, "percentile"] = 100 - percentile
        results_for_plotting_comps.loc[sum_len:sum_len + len_ct1 - 1, "syn amount"] = ct1_syn_dict[
            "amount spine neck syn"]
        results_for_plotting_comps.loc[sum_len + len_ct1:sum_len * 2 - 1, "syn amount"] = ct2_syn_dict[
            "amount spine neck syn"]
        results_for_plotting_comps.loc[sum_len: sum_len + len_ct1 - 1, "avg syn size"] = ct1_syn_dict[
            "avg size spine neck syn"]
        results_for_plotting_comps.loc[sum_len + len_ct1:sum_len * 2 - 1, "avg syn size"] = ct2_syn_dict[
            "avg size spine neck syn"]
        results_for_plotting_comps.loc[sum_len:sum_len + len_ct1 - 1, "percentage amount"] = ct1_syn_dict[
            "percentage spine neck syn amount"]
        results_for_plotting_comps.loc[sum_len + len_ct1:sum_len * 2 - 1, "percentage amount"] = ct2_syn_dict[
            "percentage spine neck syn amount"]
        results_for_plotting_comps.loc[sum_len: sum_len + len_ct1 - 1, "percentage syn size"] = ct1_syn_dict[
            "percentage spine neck syn size"]
        results_for_plotting_comps.loc[sum_len + len_ct1:sum_len * 2 - 1, "percentage syn size"] = ct2_syn_dict[
            "percentage spine neck syn size"]
        results_for_plotting_comps.loc[sum_len * 2:sum_len * 3 - 1, "compartment"] = "shaft"
        results_for_plotting_comps.loc[sum_len * 2:sum_len * 2 + len_ct1 - 1, "percentile"] = percentile
        results_for_plotting_comps.loc[sum_len * 2 + len_ct1:sum_len * 3 - 1, "percentile"] = 100 - percentile
        results_for_plotting_comps.loc[sum_len * 2: sum_len * 2 + len_ct1 - 1, "syn amount"] = ct1_syn_dict[
            "amount shaft syn"]
        results_for_plotting_comps.loc[sum_len * 2 + len_ct1:sum_len * 3 - 1, "syn amount"] = ct2_syn_dict[
            "amount shaft syn"]
        results_for_plotting_comps.loc[sum_len * 2: sum_len * 2 + len_ct1 - 1, "avg syn size"] = ct1_syn_dict[
            "avg size shaft syn"]
        results_for_plotting_comps.loc[sum_len * 2 + len_ct1:sum_len * 3 - 1, "avg syn size"] = ct2_syn_dict[
            "avg size shaft syn"]
        results_for_plotting_comps.loc[sum_len * 2:sum_len * 2 + len_ct1 - 1, "percentage amount"] = ct1_syn_dict[
            "percentage shaft syn amount"]
        results_for_plotting_comps.loc[sum_len * 2 + len_ct1:sum_len * 3 - 1, "percentage amount"] = ct2_syn_dict[
            "percentage shaft syn amount"]
        results_for_plotting_comps.loc[sum_len * 2: sum_len * 2 + len_ct1 - 1, "percentage syn size"] = ct1_syn_dict[
            "percentage shaft syn size"]
        results_for_plotting_comps.loc[sum_len * 2 + len_ct1:sum_len * 3 - 1, "percentage syn size"] = ct2_syn_dict[
            "percentage shaft syn size"]
        results_for_plotting_comps.loc[sum_len * 3:sum_len * 4 - 1, "compartment"] = "soma"
        results_for_plotting_comps.loc[sum_len * 3:sum_len * 3 + len_ct1 - 1, "percentile"] = percentile
        results_for_plotting_comps.loc[sum_len * 3 + len_ct1:sum_len * 4 - 1, "percentile"] = 100 - percentile
        results_for_plotting_comps.loc[sum_len * 3: sum_len * 3 + len_ct1 - 1, "syn amount"] = ct1_syn_dict[
            "amount soma syn"]
        results_for_plotting_comps.loc[sum_len * 3 + len_ct1:sum_len * 4 - 1, "syn amount"] = ct2_syn_dict[
            "amount soma syn"]
        results_for_plotting_comps.loc[sum_len * 3: sum_len * 3 + len_ct1 - 1, "avg syn size"] = ct1_syn_dict[
            "avg size soma syn"]
        results_for_plotting_comps.loc[sum_len * 3 + len_ct1:sum_len * 4 - 1, "avg syn size"] = ct2_syn_dict[
            "avg size soma syn"]
        results_for_plotting_comps.loc[sum_len * 3: sum_len * 3 + len_ct1 - 1, "percentage amount"] = ct1_syn_dict[
            "percentage soma syn amount"]
        results_for_plotting_comps.loc[sum_len * 3 + len_ct1:sum_len * 4 - 1, "percentage amount"] = ct2_syn_dict[
            "percentage soma syn amount"]
        results_for_plotting_comps.loc[sum_len * 3: sum_len * 3 + len_ct1 - 1, "percentage syn size"] = ct1_syn_dict[
            "percentage soma syn size"]
        results_for_plotting_comps.loc[sum_len * 3 + len_ct1:sum_len * 4 - 1, "percentage syn size"] = ct2_syn_dict[
            "percentage soma syn size"]
        results_for_plotting_comps["syn amount"] = results_for_plotting_comps["syn amount"].astype("float64")
        results_for_plotting_comps["avg syn size"] = results_for_plotting_comps["avg syn size"].astype("float64")
        results_for_plotting_comps["percentage amount"] = results_for_plotting_comps["percentage amount"].astype(
            'float64')
        results_for_plotting_comps["percentage syn size"] = results_for_plotting_comps["percentage syn size"].astype(
            "float64")

        if connected_ct != None:
            results_for_plotting_comps.to_csv("%s/%s_2_%i_%s_syn_compartments.csv" % (
            f_name, ct_dict[connected_ct], percentile, ct_dict[celltype]))
        else:
            results_for_plotting_comps.to_csv("%s/%i_%s_syn_compartments.csv" % (
                f_name, percentile, ct_dict[celltype]))

        for key in results_for_plotting_comps.keys():
            if "celltype" in key or "compartment" in key:
                continue
            sns.stripplot(x="compartment", y=key, data=results_for_plotting_comps, hue="percentile", color="black",
                          alpha=0.3, dodge=True)
            ax = sns.violinplot(x="compartment", y=key, data=results_for_plotting_comps.reset_index(), inner="box",
                                palette=colours_pal, hue="percentile")
            handles, labels = ax.get_legend_handles_labels()
            plt.legend(handles[0:2], labels[0:2])
            if connected_ct != None:
                plt.title('%s, %s to %i/ %i in %s' % (key, ct_dict[connected_ct], percentile, 100 - percentile, ct_dict[celltype]))
            else:
                plt.title(
                    '%s, between %i and %i in %s in different compartments' % (key, percentile, 100 - percentile, ct_dict[celltype]))
            if "amount" in key:
                if "percentage" in key:
                    plt.ylabel("percentage of synapses")
                else:
                    plt.ylabel("amount of synapses")
            elif "size" in key:
                if "percentage" in key:
                    plt.ylabel("percentage of synapse size")
                else:
                    plt.ylabel("average synapse size [µm²]")
            if connected_ct != None:
                filename = ("%s/%s_comps_violin_%s_2_%i_%i_%s.png" % (
                    f_name, key, ct_dict[connected_ct], percentile, 100 - percentile, ct_dict[celltype]))
            else:
                filename = ("%s/%s_comps_violin_%i_%i_%s.png" % (f_name, key, percentile, 100 - percentile, ct_dict[celltype]))
            plt.savefig(filename)
            plt.close()
            sns.boxplot(x="compartment", y=key, data=results_for_plotting_comps, palette=colours_pal,
                        hue="percentile")
            if connected_ct != None:
                plt.title('%s, %s to %i/ %i in %s' % (key, ct_dict[connected_ct], percentile, 100 - percentile, ct_dict[celltype]))
            else:
                plt.title(
                    '%s, between %i and %i in %s in different compartments' % (key, percentile, 100 - percentile, ct_dict[celltype]))
            if "amount" in key:
                if "percentage" in key:
                    plt.ylabel("percentage of synapses")
                else:
                    plt.ylabel("amount of synapses")
            elif "size" in key:
                if "percentage" in key:
                    plt.ylabel("percentage of synapse size")
                else:
                    plt.ylabel("average synapse size [µm²]")
            if connected_ct != None:
                filename = ("%s/%s_comps_box_%s_2_%i_%i_%s.png" % (
                    f_name, key, ct_dict[connected_ct], percentile, 100 - percentile, ct_dict[celltype]))
            else:
                filename = ("%s/%s_comps_box_%i_%i_%s.png" % (f_name, key, percentile, 100 - percentile, ct_dict[celltype]))
            plt.savefig(filename)
            plt.close()

        for key in ct1_syn_dict.keys():
            if "ids" in key:
                continue
            # calculate p_value for parameter
            stats, p_value = ranksums(ct1_syn_dict[key], ct2_syn_dict[key])
            ranksum_results.loc["stats", key] = stats
            ranksum_results.loc["p value", key] = p_value
            # plot parameter as violinplot
            if not "spine" in key and not "soma" in key and not "shaft" in key:
                results_for_plotting = pd.DataFrame(columns=[percentile, 100 - percentile],
                                                    index=range(sum_len))
                results_for_plotting.loc[0:len(ct1_syn_dict[key]) - 1, percentile] = ct1_syn_dict[key]
                results_for_plotting.loc[0:len(ct2_syn_dict[key]) - 1, 100 - percentile] = ct2_syn_dict[key]
                sns.violinplot(data=results_for_plotting, inner="box", palette=colours_pal)
                sns.stripplot(data=results_for_plotting, color="black", alpha=0.2)
                if connected_ct != None:
                    plt.title('%s, %s to %i/ %i in %s' % (key, ct_dict[connected_ct], percentile, 100 - percentile, ct_dict[celltype]))
                else:
                    plt.title('%s, between %i and %i in %s' % (key, percentile, 100 - percentile, ct_dict[celltype]))
                if "amount" in key:
                    if "percentage" in key:
                        plt.ylabel("percentage of synapses")
                    else:
                        plt.ylabel("amount of synapses")
                elif "size" in key:
                    if "percentage" in key:
                        plt.ylabel("percentage of synapse size")
                    else:
                        plt.ylabel("average synapse size [µm²]")
                if connected_ct != None:
                    filename = ("%s/%s_violin_%s_2_%i_%i_%s.png" % (
                    f_name, key, ct_dict[connected_ct], percentile, 100 - percentile, ct_dict[celltype]))
                else:
                    filename = ("%s/%s_violin_%i_%i_%s.png" % (f_name, key, percentile, 100 - percentile, ct_dict[celltype]))
                plt.savefig(filename)
                plt.close()
                sns.boxplot(data=results_for_plotting, palette=colours_pal)
                if connected_ct != None:
                    plt.title('%s, %s to %i/ %i in %s' % (key, ct_dict[connected_ct], percentile, 100 - percentile, ct_dict[celltype]))
                else:
                    plt.title('%s, between %i and %i in %s' % (key, percentile, 100 - percentile, ct_dict[celltype]))
                if "amount" in key:
                    if "percentage" in key:
                        plt.ylabel("percentage of synapses")
                    else:
                        plt.ylabel("amount of synapses")
                elif "size" in key:
                    if "percentage" in key:
                        plt.ylabel("percentage of synapse size")
                    else:
                        plt.ylabel("average synapse size [µm²]")
                if connected_ct != None:
                    filename = ("%s/%s_box_%s_2_%i_%i_%s.png" % (
                        f_name, key, ct_dict[connected_ct], percentile, 100 - percentile, ct_dict[celltype]))
                else:
                    filename = ("%s/%s_box_%i_%i_%s.png" % (f_name, key, percentile, 100 - percentile, ct_dict[celltype]))
                plt.savefig(filename)
                plt.close()
            sns.distplot(ct1_syn_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "gray"},
                         kde=False, label=percentile, bins=10)
            sns.distplot(ct2_syn_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "darkturquoise"},
                         kde=False, label=100 - percentile, bins=10)
            plt.legend()
            if connected_ct != None:
                plt.title('%s, %s to %i/ %i in %s' % (key, ct_dict[connected_ct], percentile, 100 - percentile, ct_dict[celltype]))
            else:
                plt.title('%s, between %i and %i in %s' % (key, percentile, 100 - percentile, ct_dict[celltype]))
            plt.ylabel("count of cells")
            if "amount" in key:
                if "percentage" in key:
                    plt.xlabel("percentage of synapses")
                else:
                    plt.xlabel("amount of synapses")
            elif "size" in key:
                if "percentage" in key:
                    plt.xlabel("percentage of synapse size")
                else:
                    plt.xlabel("average synapse size [µm²]")
            if connected_ct != None:
                filename = ("%s/%s_hist_%s_2_%i_%i_%s.png" % (
                    f_name, key, ct_dict[connected_ct], percentile, 100 - percentile, ct_dict[celltype]))
            else:
                filename = ("%s/%s_hist_%i_%i_%s.png" % (f_name, key, percentile, 100 - percentile, ct_dict[celltype]))
            plt.savefig(filename)
            plt.close()
            sns.distplot(ct1_syn_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                         kde=False, label=percentile, bins=10, norm_hist=True)
            sns.distplot(ct2_syn_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                         kde=False, label=100 - percentile, bins=10, norm_hist=True)
            plt.legend()
            if connected_ct != None:
                plt.title('%s, %s to %i/ %i in %s' % (key, ct_dict[connected_ct], percentile, 100 - percentile, ct_dict[celltype]))
            else:
                plt.title('%s, between %i and %i in %s' % (key, percentile, 100 - percentile, ct_dict[celltype]))
            plt.ylabel("fraction of cells")
            if "length" in key:
                plt.xlabel("pathlength in µm")
            elif "vol" in key:
                if "percentage" in key:
                    plt.xlabel("% of whole dataset")
                else:
                    plt.xlabel("volume in µm³")
            else:
                plt.xlabel("distance in µm")
            if connected_ct != None:
                filename = ("%s/%s_hist_norm_%s_2_%i_%i_%s.png" % (
                    f_name, key, ct_dict[connected_ct], percentile, 100 - percentile, ct_dict[celltype]))
            else:
                filename = ("%s/%s_hist_norm_%i_%i_%s.png" % (f_name, key, percentile, 100 - percentile, ct_dict[celltype]))
            plt.savefig(filename)
            plt.close()

        if connected_ct != None:
            ranksum_results.to_csv(
                "%s/ranksum_%s_2_%i_%i_%s.csv" % (f_name, ct_dict[connected_ct], percentile, 100 - percentile, ct_dict[celltype]))
        else:
            ranksum_results.to_csv("%s/ranksum_%i_%i_%s.csv" % (f_name, percentile, 100 - percentile, ct_dict[celltype]))

        # also compare outgoing connections from celltype
        if connected_ct != None:
            ct2_syn_dict = load_pkl2obj(
                "%s/%i_%s_2_%s_dict.pkl" % (foldername_ct2, 100 - percentile, ct_dict[celltype], ct_dict[connected_ct]))
            ct1_syn_dict = load_pkl2obj(
                "%s/%i_%s_2_%s_dict.pkl" % (foldername_ct1, percentile, ct_dict[celltype], ct_dict[connected_ct]))
            len_ct1 = len(ct1_syn_dict["amount spine head syn"])
            len_ct2 = len(ct2_syn_dict["amount spine head syn"])
            sum_len = len_ct1 + len_ct2
            results_for_plotting_comps = pd.DataFrame(
                columns=["syn amount", "avg syn size", "percentage amount", "percentage syn size", "percentile",
                         "compartment"], index=range(sum_len * 4))
            results_for_plotting_comps["compartment"] = str()
            results_for_plotting_comps.loc[0:sum_len - 1, "compartment"] = "spine head"
            results_for_plotting_comps.loc[0:len_ct1 - 1, "percentile"] = percentile
            results_for_plotting_comps.loc[len_ct1: sum_len - 1, "percentile"] = 100 - percentile
            results_for_plotting_comps.loc[0:len_ct1 - 1, "syn amount"] = ct1_syn_dict["amount spine head syn"]
            results_for_plotting_comps.loc[len_ct1:sum_len - 1, "syn amount"] = ct2_syn_dict["amount spine head syn"]
            results_for_plotting_comps.loc[0:len_ct1 - 1, "avg syn size"] = ct1_syn_dict["avg size spine head syn"]
            results_for_plotting_comps.loc[len_ct1:sum_len - 1, "avg syn size"] = ct2_syn_dict[
                "avg size spine head syn"]
            results_for_plotting_comps.loc[0:len_ct1 - 1, "percentage amount"] = ct1_syn_dict[
                "percentage spine head syn amount"]
            results_for_plotting_comps.loc[len_ct1:sum_len - 1, "percentage amount"] = ct2_syn_dict[
                "percentage spine head syn amount"]
            results_for_plotting_comps.loc[0:len_ct1 - 1, "percentage syn size"] = ct1_syn_dict[
                "percentage spine head syn size"]
            results_for_plotting_comps.loc[len_ct1:sum_len - 1, "percentage syn size"] = ct2_syn_dict[
                "percentage spine head syn size"]
            results_for_plotting_comps.loc[sum_len:sum_len * 2 - 1, "compartment"] = "spine neck"
            results_for_plotting_comps.loc[sum_len:sum_len + len_ct1 - 1, "percentile"] = percentile
            results_for_plotting_comps.loc[sum_len + len_ct1:sum_len * 2 - 1, "percentile"] = 100 - percentile
            results_for_plotting_comps.loc[sum_len:sum_len + len_ct1 - 1, "syn amount"] = ct1_syn_dict[
                "amount spine neck syn"]
            results_for_plotting_comps.loc[sum_len + len_ct1:sum_len * 2 - 1, "syn amount"] = ct2_syn_dict[
                "amount spine neck syn"]
            results_for_plotting_comps.loc[sum_len: sum_len + len_ct1 - 1, "avg syn size"] = ct1_syn_dict[
                "avg size spine neck syn"]
            results_for_plotting_comps.loc[sum_len + len_ct1:sum_len * 2 - 1, "avg syn size"] = ct2_syn_dict[
                "avg size spine neck syn"]
            results_for_plotting_comps.loc[sum_len:sum_len + len_ct1 - 1, "percentage amount"] = ct1_syn_dict[
                "percentage spine neck syn amount"]
            results_for_plotting_comps.loc[sum_len + len_ct1:sum_len * 2 - 1, "percentage amount"] = ct2_syn_dict[
                "percentage spine neck syn amount"]
            results_for_plotting_comps.loc[sum_len: sum_len + len_ct1 - 1, "percentage syn size"] = ct1_syn_dict[
                "percentage spine neck syn size"]
            results_for_plotting_comps.loc[sum_len + len_ct1:sum_len * 2 - 1, "percentage syn size"] = ct2_syn_dict[
                "percentage spine neck syn size"]
            results_for_plotting_comps.loc[sum_len * 2:sum_len * 3 - 1, "compartment"] = "shaft"
            results_for_plotting_comps.loc[sum_len * 2:sum_len * 2 + len_ct1 - 1, "percentile"] = percentile
            results_for_plotting_comps.loc[sum_len * 2 + len_ct1:sum_len * 3 - 1, "percentile"] = 100 - percentile
            results_for_plotting_comps.loc[sum_len * 2: sum_len * 2 + len_ct1 - 1, "syn amount"] = ct1_syn_dict[
                "amount shaft syn"]
            results_for_plotting_comps.loc[sum_len * 2 + len_ct1:sum_len * 3 - 1, "syn amount"] = ct2_syn_dict[
                "amount shaft syn"]
            results_for_plotting_comps.loc[sum_len * 2: sum_len * 2 + len_ct1 - 1, "avg syn size"] = ct1_syn_dict[
                "avg size shaft syn"]
            results_for_plotting_comps.loc[sum_len * 2 + len_ct1:sum_len * 3 - 1, "avg syn size"] = ct2_syn_dict[
                "avg size shaft syn"]
            results_for_plotting_comps.loc[sum_len * 2:sum_len * 2 + len_ct1 - 1, "percentage amount"] = ct1_syn_dict[
                "percentage shaft syn amount"]
            results_for_plotting_comps.loc[sum_len * 2 + len_ct1:sum_len * 3 - 1, "percentage amount"] = ct2_syn_dict[
                "percentage shaft syn amount"]
            results_for_plotting_comps.loc[sum_len * 2: sum_len * 2 + len_ct1 - 1, "percentage syn size"] = \
                ct1_syn_dict["percentage shaft syn size"]
            results_for_plotting_comps.loc[sum_len * 2 + len_ct1:sum_len * 3 - 1, "percentage syn size"] = ct2_syn_dict[
                "percentage shaft syn size"]
            results_for_plotting_comps.loc[sum_len * 3:sum_len * 4 - 1, "compartment"] = "soma"
            results_for_plotting_comps.loc[sum_len * 3:sum_len * 3 + len_ct1 - 1, "percentile"] = percentile
            results_for_plotting_comps.loc[sum_len * 3 + len_ct1:sum_len * 4 - 1, "percentile"] = 100 - percentile
            results_for_plotting_comps.loc[sum_len * 3: sum_len * 3 + len_ct1 - 1, "syn amount"] = ct1_syn_dict[
                "amount soma syn"]
            results_for_plotting_comps.loc[sum_len * 3 + len_ct1:sum_len * 4 - 1, "syn amount"] = ct2_syn_dict[
                "amount soma syn"]
            results_for_plotting_comps.loc[sum_len * 3: sum_len * 3 + len_ct1 - 1, "avg syn size"] = ct1_syn_dict[
                "avg size soma syn"]
            results_for_plotting_comps.loc[sum_len * 3 + len_ct1:sum_len * 4 - 1, "avg syn size"] = ct2_syn_dict[
                "avg size soma syn"]
            results_for_plotting_comps.loc[sum_len * 3: sum_len * 3 + len_ct1 - 1, "percentage amount"] = ct1_syn_dict[
                "percentage soma syn amount"]
            results_for_plotting_comps.loc[sum_len * 3 + len_ct1:sum_len * 4 - 1, "percentage amount"] = ct2_syn_dict[
                "percentage soma syn amount"]
            results_for_plotting_comps.loc[sum_len * 3: sum_len * 3 + len_ct1 - 1, "percentage syn size"] = \
                ct1_syn_dict["percentage soma syn size"]
            results_for_plotting_comps.loc[sum_len * 3 + len_ct1:sum_len * 4 - 1, "percentage syn size"] = ct2_syn_dict[
                "percentage soma syn size"]
            results_for_plotting_comps["syn amount"] = results_for_plotting_comps["syn amount"].astype("float64")
            results_for_plotting_comps["avg syn size"] = results_for_plotting_comps["avg syn size"].astype("float64")
            results_for_plotting_comps["percentage amount"] = results_for_plotting_comps["percentage amount"].astype(
                'float64')
            results_for_plotting_comps["percentage syn size"] = results_for_plotting_comps[
                "percentage syn size"].astype("float64")

            results_for_plotting_comps.to_csv("%s/%i_%i_%s_2_%s_syn_compartments_outgoing.csv" % (
                f_name, percentile, 100 - percentile, ct_dict[celltype], ct_dict[connected_ct]))

            for key in results_for_plotting_comps.keys():
                if "celltype" in key or "compartment" in key:
                    continue

                sns.stripplot(x="compartment", y=key, data=results_for_plotting_comps, hue="celltype", color="black",
                              alpha=0.3, dodge=True)
                ax = sns.violinplot(x="compartment", y=key, data=results_for_plotting_comps.reset_index(), inner="box",
                                    palette=colours_pal, hue="celltype")
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(handles[0:2], labels[0:2])
                plt.title('%s, %i, %i %s to %s' % (key, percentile, 100 - percentile, ct_dict[celltype], ct_dict[connected_ct]))
                if "amount" in key:
                    if "percentage" in key:
                        plt.ylabel("percentage of synapses")
                    else:
                        plt.ylabel("amount of synapses")
                elif "size" in key:
                    if "percentage" in key:
                        plt.ylabel("percentage of synapse size")
                    else:
                        plt.ylabel("average synapse size [µm²]")
                filename = ("%s/%s_comps_violin_%s_%i_%i_%s_2_%s_outgoing.png" % (
                    f_name, key, percentile, 100 - percentile, ct_dict[celltype], ct_dict[connected_ct]))
                plt.savefig(filename)
                plt.close()
                sns.boxplot(x="compartment", y=key, data=results_for_plotting_comps, palette=colours_pal,
                            hue="celltype")
                plt.title('%s, %i, %i %s to %s' % (key, percentile, 100 - percentile, ct_dict[celltype], ct_dict[connected_ct]))
                if "amount" in key:
                    if "percentage" in key:
                        plt.ylabel("percentage of synapses")
                    else:
                        plt.ylabel("amount of synapses")
                elif "size" in key:
                    if "percentage" in key:
                        plt.ylabel("percentage of synapse size")
                    else:
                        plt.ylabel("average synapse size [µm²]")
                filename = ("%s/%s_comps_box_%i_%i_%s_2_%s.png" % (
                    f_name, key, percentile, 100 - percentile, ct_dict[celltype], ct_dict[connected_ct]))
                plt.savefig(filename)
                plt.close()

            for key in ct1_syn_dict.keys():
                if "ids" in key:
                    continue
                # calculate p_value for parameter
                stats, p_value = ranksums(ct1_syn_dict[key], ct2_syn_dict[key])
                ranksum_results.loc["stats", key] = stats
                ranksum_results.loc["p value", key] = p_value
                # plot parameter as violinplot
                if not "spine" in key and not "soma" in key and not "shaft" in key:
                    results_for_plotting = pd.DataFrame(columns=[percentile, 100 - percentile],
                                                        index=range(sum_len))
                    results_for_plotting.loc[0:len(ct1_syn_dict[key]) - 1, percentile] = ct1_syn_dict[key]
                    results_for_plotting.loc[0:len(ct2_syn_dict[key]) - 1, 100 - percentile] = ct2_syn_dict[key]
                    sns.violinplot(data=results_for_plotting, inner="box", palette=colours_pal)
                    sns.stripplot(data=results_for_plotting, color="black", alpha=0.2)
                    plt.title('%s, %i, %i %s to %s' % (key, percentile, 100 - percentile, ct_dict[celltype], ct_dict[connected_ct]))
                    if "amount" in key:
                        if "percentage" in key:
                            plt.ylabel("percentage of synapses")
                        else:
                            plt.ylabel("amount of synapses")
                    elif "size" in key:
                        if "percentage" in key:
                            plt.ylabel("percentage of synapse size")
                        else:
                            plt.ylabel("average synapse size [µm²]")
                    filename = ("%s/%s_violin_%i_%i_%s_2_%s_outgoing.png" % (
                        f_name, key, percentile, 100 - percentile, ct_dict[celltype], ct_dict[connected_ct]))
                    plt.savefig(filename)
                    plt.close()
                    sns.boxplot(data=results_for_plotting, palette=colours_pal)
                    plt.title('%s, %i, %i %s to %s' % (key, percentile, 100 - percentile, ct_dict[connected_ct]))
                    if "amount" in key:
                        if "percentage" in key:
                            plt.ylabel("percentage of synapses")
                        else:
                            plt.ylabel("amount of synapses")
                    elif "size" in key:
                        if "percentage" in key:
                            plt.ylabel("percentage of synapse size")
                        else:
                            plt.ylabel("average synapse size [µm²]")
                    filename = ("%s/%s_box_%si_%i_%s_2_%s_outgoing.png" % (
                        f_name, key, percentile, 100 - percentile, ct_dict[celltype], ct_dict[connected_ct]))
                    plt.savefig(filename)
                    plt.close()
                sns.distplot(ct1_syn_dict[key],
                             hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                             kde=False, label=percentile, bins=10)
                sns.distplot(ct2_syn_dict[key],
                             hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                             kde=False, label=100 - percentile, bins=10)
                plt.legend()
                plt.title('%s, %i, %i %s to %s' % (key, percentile, 100 - percentile, ct_dict[celltype], ct_dict[connected_ct]))
                plt.ylabel("count of cells")
                if "amount" in key:
                    if "percentage" in key:
                        plt.xlabel("percentage of synapses")
                    else:
                        plt.xlabel("amount of synapses")
                elif "size" in key:
                    if "percentage" in key:
                        plt.xlabel("percentage of synapse size")
                    else:
                        plt.xlabel("average synapse size [µm²]")
                filename = ("%s/%s_hist_%i_%i_%s_2_%s_outgoing.png" % (
                    f_name, key, percentile, 100 - percentile, ct_dict[celltype], ct_dict[connected_ct]))
                plt.savefig(filename)
                plt.close()
                sns.distplot(ct1_syn_dict[key],
                             hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                             kde=False, label=percentile, bins=10, norm_hist=True)
                sns.distplot(ct2_syn_dict[key],
                             hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                             kde=False, label=100 - percentile, bins=10, norm_hist=True)
                plt.legend()
                plt.title('%s, %i, %i %s to %s' % (key, percentile, 100 - percentile, ct_dict[celltype], ct_dict[connected_ct]))
                plt.ylabel("fraction of cells")
                if "length" in key:
                    plt.xlabel("pathlength in µm")
                elif "vol" in key:
                    if "percentage" in key:
                        plt.xlabel("% of whole dataset")
                    else:
                        plt.xlabel("volume in µm³")
                else:
                    plt.xlabel("distance in µm")
                filename = ("%s/%s_hist_norm_%i_%i_%s_2_%s_outgoing.png" % (
                    f_name, key, percentile, 100 - percentile, ct_dict[celltype], ct_dict[connected_ct]))
                plt.savefig(filename)
                plt.close()

            ranksum_results.to_csv("%s/ranksum_%i_%i_%s_2_%s_outgoing.csv" % (
            f_name, percentile, 100 - percentile, ct_dict[celltype], ct_dict[connected_ct]))

        plottime = time.time() - start
        print("%.2f sec for statistics and plotting" % plottime)
        time_stamps.append(time.time())
        step_idents.append('comparing celltypes')
        log.info("comparing celltypes via connectivity finished")


    #synapses_between2cts(ssd, sd_synssv, celltype1=6, celltype2=2, full_cells=True, handpicked1=True, handpicked2 = False)
    #synapses_between2cts(ssd, sd_synssv, celltype1=7, celltype2=2, full_cells=True, handpicked1=True, handpicked2=False)
    #synapses_between2cts(ssd, sd_synssv, celltype1=5, celltype2=6, full_cells=True, handpicked1=True, handpicked2=True)
    #synapses_between2cts(ssd, sd_synssv, celltype1=0, celltype2=6, full_cells=True, handpicked1=False, handpicked2=True)
    #synapses_between2cts(ssd, sd_synssv, celltype1=0, celltype2=7, full_cells=True, handpicked1=False, handpicked2=True)
    foldername = "u/arother/bio_analysis_results/dir_indir_pathway_analysis/"
    #ct1_filename = "%s/210823_j0251v3_syn_conn_GPe_2_GPi_mcl100_sysi_0.10_st_0.60" % foldername
    #compare_connectvity(comp_ct1=6, comp_ct2=7, foldername_ct1=ct1_filename, foldername_ct2=ct1_filename)
    #ct1_filename = "%s/210923_j0251v3_syn_conn_GPe_2_MSN_mcl100_sysi_0.10_st_0.60" % foldername
    #ct2_filename = "%s/210923_j0251v3_syn_conn_GPi_2_MSN_mcl100_sysi_0.10_st_0.60" % foldername
    #compare_connectvity(comp_ct1=6, comp_ct2=7, connected_ct=2, foldername_ct1=ct1_filename, foldername_ct2=ct2_filename)
    #ct1_filename = "%s/210823_j0251v3_syn_conn_STN_2_GPe_mcl100_sysi_0.10_st_0.60" % foldername
    #ct2_filename = "%s/210823_j0251v3_syn_conn_STN_2_GPi_mcl100_sysi_0.10_st_0.60" % foldername
    #compare_connectvity(comp_ct1=6, comp_ct2=7, connected_ct=0, foldername_ct1=ct1_filename,
                        #foldername_ct2=ct2_filename)
    percentiles = [10, 25, 50]
    for percentile in percentiles:
        #synapses_between2percentiles(ssd, sd_synssv=sd_synssv, celltype=2, percentile=percentile)
        ct1_filename = "%s/211001_j0251v3_syn_conn_MSN_p%i_mcl100_sysi_0.10_st_0.60" % (foldername, percentile)
        compare_connectvity_percentiles(celltype=2, percentile=percentile, foldername_ct1=ct1_filename, foldername_ct2=ct1_filename)




