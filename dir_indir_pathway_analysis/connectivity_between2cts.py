
import numpy as np
import pandas as pd
import os as os
import time
from syconn.handler.config import initialize_logging
from syconn.handler.basics import load_pkl2obj
from tqdm import tqdm
from syconn.handler.basics import write_obj2pkl
from scipy.stats import ranksums
from wholebrain.scratch.arother.bio_analysis.general.analysis_helper import get_compartment_length, check_comp_lengths_ct, filter_synapse_caches_for_ct
from wholebrain.scratch.arother.bio_analysis.general.result_helper import ResultsForPlotting, ComparingResultsForPLotting, plot_nx_graph, ComparingMultipleForPLotting



def synapses_between2cts(sd_synssv, celltype1, filename, cellids1, celltype2 = None, cellids2 = None, full_cells = True, percentile_ct1 = None,
                         min_comp_len = 100, min_syn_size = 0.1, syn_prob_thresh = 0.8, label_ct1 = None, label_ct2 = None):
    '''
    looks at basic connectivty parameters between two celltypes such as amount of synapses, average of synapses between cell types but also
    the average from one cell to the same other cell. Also looks at distribution of axo_dendritic synapses onto spines/shaft and the percentage of axo-somatic
    synapses. Uses cached synapse properties. Uses compartment_length per cell to ignore cells with not enough axon/dendrite
    # spiness values: 0 = spine neck, 1 = spine head, 2 = dendritic shaft, 3 = other
    axoness values: 0 = dendrite, 1 = axon, 2 = soma
    :param sd_synssv: segmentation dataset for synapses.
    :param celltype1, celltype2: celltypes to be compared. j0256: STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7,
                      FS=8, LTS=9, NGF=10
    :param cellids1, cellids2: cellids for celltypes 1 and 2
    :param full_cells: if True: full_cell_dict will be tried to load for celltypes
    :param percentile_ct1: if given, different subpopulations within a celltype will be compared
    :param min_comp_len: minimum length for axon/dendrite to have to include cell in analysis
    :param min_syn_size: minimum size for synapses
    :param syn_prob_thresh: threshold for synapse probability
    :param label_ct1, label_ct2: label of celltypes or subgroups not in ct_dict
    :return: f_name: foldername in which results are stored
    '''

    start = time.time()
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    if label_ct1 is None:
        ct1_str = ct_dict[celltype1]
    else:
        ct1_str = label_ct1
    if label_ct2 is None:
        if celltype2 is not None and percentile_ct1 is None:
            ct2_str = ct_dict[celltype2]
        elif percentile_ct1 is not None and celltype2 is None:
            if percentile_ct1 == 50:
                raise ValueError("Due to ambiguity, value has to be either 49 or 51")
            ct1_str = ct_dict[celltype1] + " p%.2i" % percentile_ct1
            ct2_str = ct_dict[celltype1] + " p%.2i" % (100 - percentile_ct1)
        elif percentile_ct1 is not None and celltype2 is not None:
            ct2_str = ct_dict[celltype2]
            if percentile_ct1 == 50:
                raise ValueError("Due to ambiguity, value has to be either 49 or 51")
            ct1_str = ct_dict[celltype1] + " p%.2i" % percentile_ct1
        else:
            raise ValueError("either celltypes or percentiles must be compared")
    else:
        ct2_str = label_ct2

    f_name = "%s/syn_conn_%s_2_%s_mcl%i_sysi_%.2f_st_%.2f" % (
            filename, ct1_str, ct2_str, min_comp_len, min_syn_size, syn_prob_thresh)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('compartment volume estimation', log_dir=f_name + '/logs/')
    log.info("parameters: celltype1 = %s, celltype2 = %s, min_comp_length = %.i, min_syn_size = %.2f, syn_prob_thresh = %.2f" %
             (ct1_str, ct2_str, min_comp_len, min_syn_size, syn_prob_thresh))
    time_stamps = [time.time()]
    step_idents = ['t-0']
    if full_cells:
        try:
            full_cell_dict_ct1 = load_pkl2obj("/wholebrain/scratch/arother/j0251v4_prep/full_%.3s_dict.pkl" % ct_dict[celltype1])
        except FileNotFoundError:
            print("preprocessed parameters not available for ct1")
    if celltype2 is not None and celltype2 != celltype1:
        try:
            full_cell_dict_ct2 = load_pkl2obj(
                "/wholebrain/scratch/arother/j0251v4_prep/full_%.3s_dict.pkl" % ct_dict[celltype2])
        except FileNotFoundError:
            print("preprocessed parameters not available for ct2")
    else:
        full_cell_dict_ct2 = full_cell_dict_ct1

    log.info("Step 1/4 Iterate over %s to check min_comp_len" % ct1_str)
    # use axon and dendrite length dictionaries to lookup axon and dendrite lenght in future versions
    cellids1 = check_comp_lengths_ct(cellids1, fullcelldict = full_cell_dict_ct1, min_comp_len = min_comp_len)

    ct1time = time.time() - start
    print("%.2f sec for iterating through %s cells" % (ct1time, ct1_str))
    time_stamps.append(time.time())
    step_idents.append('iterating over %s cells' % ct1_str)

    log.info("Step 2/4 Iterate over %s to check min_comp_len" % ct2_str)
    cellids2 = check_comp_lengths_ct(cellids2, fullcelldict = full_cell_dict_ct2, min_comp_len = min_comp_len)



    ct2time = time.time() - ct1time
    print("%.2f sec for iterating through %s cells" % (ct2time, ct2_str))
    time_stamps.append(time.time())
    step_idents.append('iterating over %s cells' % ct2_str)


    log.info("Step 3/4 get synaptic connectivity parameters")
    log.info("Step 3a: prefilter synapse caches")
    # prepare synapse caches with synapse threshold
    if celltype2 is not None and celltype2 != celltype1:
        m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness = filter_synapse_caches_for_ct(sd_synssv, pre_cts = [celltype1, celltype2],
                                                                                               syn_prob_thresh = syn_prob_thresh, min_syn_size = min_syn_size, axo_den_so = True)
    else:
        m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness = filter_synapse_caches_for_ct(sd_synssv,
                                                                                               pre_cts=[celltype1],
                                                                                               syn_prob_thresh=syn_prob_thresh,
                                                                                               min_syn_size=min_syn_size,
                                                                                               axo_den_so=True)

    prepsyntime = time.time() - ct2time
    print("%.2f sec for preprocessing synapses" % prepsyntime)
    time_stamps.append(time.time())
    step_idents.append('preprocessing synapses')



    log.info("Step 3b: iterate over synapses to get synaptic connectivity parameters")
    param_labels = ["amount synapses", "sum size synapses"]
    # spiness values: 0 = spine neck, 1 = spine head, 2 = dendritic shaft, 3 = other
    comp_labels = ["spine neck", "spine head", "shaft", "soma"]

    ct2_2_ct1_syn_dict = {"cellids": cellids1}
    ct1_2_ct2_syn_dict = {"cellids": cellids2}
    for pi in param_labels:
        ct2_2_ct1_syn_dict[pi] = np.zeros(len(cellids1))
        ct1_2_ct2_syn_dict[pi] = np.zeros(len(cellids2))
        for ci in comp_labels:
            ct2_2_ct1_syn_dict[pi + " - " + ci] = np.zeros(len(cellids1))
            ct1_2_ct2_syn_dict[pi + " - " + ci] = np.zeros(len(cellids2))

    ct2_2_ct1_percell_syn_amount = np.zeros((len(cellids1), len(cellids2))).astype(float)
    ct2_2_ct1_percell_syn_size = np.zeros((len(cellids1), len(cellids2))).astype(float)
    ct1_2_ct2_percell_syn_amount = np.zeros((len(cellids2), len(cellids1))).astype(float)
    ct1_2_ct2_percell_syn_size = np.zeros((len(cellids2), len(cellids1))).astype(float)
    ct1_2_ct2_all_syn_sizes = np.zeros(len(m_ids))
    ct2_2_ct1_all_syn_sizes = np.zeros(len(m_ids))

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
        if ct_ax == ct_deso and celltype2 is not None:
            continue
        syn_size = m_sizes[i]
        if ssv_ax in cellids1:
            cell2_ind = np.where(ct1_2_ct2_syn_dict["cellids"] == ssv_deso)[0]
            cell1_ind = np.where(ct2_2_ct1_syn_dict["cellids"] == ssv_ax)[0]
            ct1_2_ct2_syn_dict[param_labels[0]][cell2_ind] += 1
            ct1_2_ct2_syn_dict[param_labels[1]][cell2_ind] += syn_size
            ct1_2_ct2_all_syn_sizes[i] = syn_size
            ct1_2_ct2_percell_syn_amount[cell2_ind, cell1_ind] += 1
            ct1_2_ct2_percell_syn_size[cell2_ind, cell1_ind] += syn_size
            if deso == 0:
                if spin_deso <= 2:
                    ct1_2_ct2_syn_dict[param_labels[0] + " - " + comp_labels[spin_deso]][cell2_ind] += 1
                    ct1_2_ct2_syn_dict[param_labels[1] + " - " + comp_labels[spin_deso]][cell2_ind] += syn_size
            else:
                ct1_2_ct2_syn_dict[param_labels[0] + " - " + "soma"][cell2_ind] += 1
                ct1_2_ct2_syn_dict[param_labels[1] + " - " + "soma"][cell2_ind] += syn_size
        else:
            cell1_ind = np.where(ct2_2_ct1_syn_dict["cellids"] == ssv_deso)[0]
            cell2_ind = np.where(ct1_2_ct2_syn_dict["cellids"] == ssv_ax)[0]
            ct2_2_ct1_syn_dict[param_labels[0]][cell1_ind] += 1
            ct2_2_ct1_syn_dict[param_labels[1]][cell1_ind] += syn_size
            ct2_2_ct1_percell_syn_amount[cell1_ind, cell2_ind] += 1
            ct2_2_ct1_percell_syn_size[cell1_ind, cell2_ind] += syn_size
            ct2_2_ct1_all_syn_sizes[i] = syn_size
            if deso == 0:
                if spin_deso <= 2:
                    ct2_2_ct1_syn_dict[param_labels[0] + " - " + comp_labels[spin_deso]][cell1_ind] += 1
                    ct2_2_ct1_syn_dict[param_labels[1] + " - " + comp_labels[spin_deso]][cell1_ind] += syn_size
            else:
                ct2_2_ct1_syn_dict[param_labels[0] + " - " + "soma"][cell1_ind] += 1
                ct2_2_ct1_syn_dict[param_labels[1] + " - " + "soma"][cell1_ind] += syn_size

    syntime = time.time() - prepsyntime
    print("%.2f sec for processing synapses" % syntime)
    time_stamps.append(time.time())
    step_idents.append('processing synapses')

    log.info("Step 4/4: calculate last parameters (average syn size, average syn amount and syn size between 2 cells) and plotting")

    ct2_2_ct1_all_syn_sizes = ct2_2_ct1_all_syn_sizes[ct2_2_ct1_all_syn_sizes > 0]
    ct1_2_ct2_all_syn_sizes = ct1_2_ct2_all_syn_sizes[ct1_2_ct2_all_syn_sizes > 0]

    ct2_syn_inds = ct1_2_ct2_syn_dict["amount synapses"] > 0
    ct1_syn_inds = ct2_2_ct1_syn_dict["amount synapses"] > 0

    for key in ct1_2_ct2_syn_dict.keys():
        ct1_2_ct2_syn_dict[key] = ct1_2_ct2_syn_dict[key][ct2_syn_inds]
        ct2_2_ct1_syn_dict[key] = ct2_2_ct1_syn_dict[key][ct1_syn_inds]

    ct1_2_ct2_syn_dict["average synapse size"] = ct1_2_ct2_syn_dict["sum size synapses"] / ct1_2_ct2_syn_dict["amount synapses"]
    ct2_2_ct1_syn_dict["average synapse size"] = ct2_2_ct1_syn_dict["sum size synapses"] / ct2_2_ct1_syn_dict[
        "amount synapses"]

    for ci in comp_labels:
        ct1_2_ct2_syn_dict["average synapse size - " + ci] = ct1_2_ct2_syn_dict["sum size synapses - " + ci]/ ct1_2_ct2_syn_dict["amount synapses - " + ci]
        ct2_2_ct1_syn_dict["average synapse size - " + ci] = ct2_2_ct1_syn_dict["sum size synapses - " + ci] / \
                                                             ct2_2_ct1_syn_dict["amount synapses - " + ci]
        ct1_2_ct2_syn_dict["percentage synapse size - " + ci] = ct1_2_ct2_syn_dict["sum size synapses - " + ci] / \
                                                             ct1_2_ct2_syn_dict["sum size synapses"] * 100
        ct2_2_ct1_syn_dict["percentage synapse size - " + ci] = ct2_2_ct1_syn_dict["sum size synapses - " + ci] / \
                                                             ct2_2_ct1_syn_dict["sum size synapses"] * 100
        ct1_2_ct2_syn_dict["percentage synapse amount - " + ci] = ct1_2_ct2_syn_dict["amount synapses - " + ci] / \
                                                                ct1_2_ct2_syn_dict["amount synapses"] * 100
        ct2_2_ct1_syn_dict["percentage synapse amount - " + ci] = ct2_2_ct1_syn_dict["amount synapses - " + ci] / \
                                                                ct2_2_ct1_syn_dict["amount synapses"] * 100


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

    #calculate amount of synapses, sum of synapse size in relation to dendritic pathlength, dendritic surface area
    dendritic_pathlengths_ct1 = np.zeros(len(ct2_2_ct1_syn_dict["cellids"]))
    dendritic_surface_area_ct1 = np.zeros(len(ct2_2_ct1_syn_dict["cellids"]))
    overall_amount_synapses_ct1 = np.zeros(len(ct2_2_ct1_syn_dict["cellids"]))
    overall_sum_synapses_ct1 = np.zeros(len(ct2_2_ct1_syn_dict["cellids"]))
    for i, cellid in enumerate(ct2_2_ct1_syn_dict["cellids"]):
        dendritic_pathlengths_ct1[i] = full_cell_dict_ct1[cellid]["dendrite length"]
        dendritic_surface_area_ct1[i] = full_cell_dict_ct1[cellid]["dendrite mesh surface area"]
        overall_amount_synapses_ct1[i] = full_cell_dict_ct1[cellid]["dendrite synapse amount"] + full_cell_dict_ct1[cellid]["soma synapse amount"]
        overall_sum_synapses_ct1[i] = full_cell_dict_ct1[cellid]["dendrite summed synapse size"] + full_cell_dict_ct1[cellid]["soma summed synapse size"]

    ct2_2_ct1_syn_dict["amount synapses per dendritic pathlength"] = (ct2_2_ct1_syn_dict["amount synapses"] - ct2_2_ct1_syn_dict["amount synapses - soma"]) /dendritic_pathlengths_ct1
    ct2_2_ct1_syn_dict["amount synapses per dendritic surface area"] = (ct2_2_ct1_syn_dict[
                                                               "amount synapses"] - ct2_2_ct1_syn_dict["amount synapses - soma"])/ dendritic_surface_area_ct1

    ct2_2_ct1_syn_dict["sum size synapses per dendritic pathlength"] = (ct2_2_ct1_syn_dict[
                                                                         "sum size synapses"] - ct2_2_ct1_syn_dict["sum size synapses - soma"]) / dendritic_pathlengths_ct1
    ct2_2_ct1_syn_dict["sum size synapses per dendritic surface area"] = (ct2_2_ct1_syn_dict[
                                                                           "sum size synapses"] - ct2_2_ct1_syn_dict["sum size synapses - soma"])/ dendritic_surface_area_ct1
    ct2_2_ct1_syn_dict["percentage amount synapses"] = (ct2_2_ct1_syn_dict["amount synapses"] / overall_amount_synapses_ct1) * 100
    ct2_2_ct1_syn_dict["percentage sum size synapses"] = (ct2_2_ct1_syn_dict["sum size synapses"]/ overall_sum_synapses_ct1)* 100

    dendritic_pathlengths_ct2 = np.zeros(len(ct1_2_ct2_syn_dict["cellids"]))
    dendritic_surface_area_ct2 = np.zeros(len(ct1_2_ct2_syn_dict["cellids"]))
    overall_amount_synapses_ct2 = np.zeros(len(ct1_2_ct2_syn_dict["cellids"]))
    overall_sum_synapses_ct2 = np.zeros(len(ct1_2_ct2_syn_dict["cellids"]))
    for i, cellid in enumerate(ct1_2_ct2_syn_dict["cellids"]):
        dendritic_pathlengths_ct2[i] = full_cell_dict_ct2[cellid]["dendrite length"]
        dendritic_surface_area_ct2[i] = full_cell_dict_ct2[cellid]["dendrite mesh surface area"]
        overall_amount_synapses_ct2[i] = full_cell_dict_ct2[cellid]["dendrite synapse amount"] + full_cell_dict_ct2[cellid]["soma synapse amount"]
        overall_sum_synapses_ct2[i] = full_cell_dict_ct2[cellid]["dendrite summed synapse size"] + full_cell_dict_ct2[cellid]["soma summed synapse size"]

    ct1_2_ct2_syn_dict["amount synapses per dendritic pathlength"] = (ct1_2_ct2_syn_dict[
                                                                         "amount synapses"] - ct1_2_ct2_syn_dict["amount synapses - soma"])/ dendritic_pathlengths_ct2
    ct1_2_ct2_syn_dict["amount synapses per dendritic surface area"] = (ct1_2_ct2_syn_dict[
                                                                           "amount synapses"] - ct1_2_ct2_syn_dict["amount synapses - soma"])/ dendritic_surface_area_ct2

    ct1_2_ct2_syn_dict["sum size synapses per dendritic pathlength"] = (ct1_2_ct2_syn_dict[
                                                                           "sum size synapses"] - ct1_2_ct2_syn_dict["sum size synapses - soma"])/ dendritic_pathlengths_ct2
    ct1_2_ct2_syn_dict["sum size synapses per dendritic surface area"] = (ct1_2_ct2_syn_dict[
                                                                             "sum size synapses"] - ct1_2_ct2_syn_dict["sum size synapses - soma"])/ dendritic_surface_area_ct2
    ct1_2_ct2_syn_dict["percentage amount synapses"] = (ct1_2_ct2_syn_dict[
                                                            "amount synapses"] / overall_amount_synapses_ct2) * 100
    ct1_2_ct2_syn_dict["percentage sum size synapses"] = (ct1_2_ct2_syn_dict[
                                                              "sum size synapses"] / overall_sum_synapses_ct2) * 100

    for ci in comp_labels:
        ct1_2_ct2_syn_dict.pop("sum size synapses - " + ci)
        ct2_2_ct1_syn_dict.pop("sum size synapses - " + ci)

    ct1_2_ct2_pd = pd.DataFrame(ct1_2_ct2_syn_dict)
    ct1_2_ct2_pd.to_csv("%s/%s_2_%s_dict.csv" % (f_name, ct1_str, ct2_str))


    ct2_2_ct1_pd = pd.DataFrame(ct2_2_ct1_syn_dict)
    ct2_2_ct1_pd.to_csv("%s/%s_2_%s_dict.csv" % (f_name, ct2_str, ct1_str))

    # group average amount one cell by amount of synapses
    # make barplot
    if len(ct1_2_ct2_percell_syn_amount) != 0 and len(ct2_2_ct1_percell_syn_amount) != 0:
        ct1_2_ct2_max_multisyn = int(np.nanmax(ct1_2_ct2_percell_syn_amount))
        ct2_2_ct1_max_multisyn = int(np.nanmax(ct1_2_ct2_percell_syn_amount))
        max_multisyn = int(np.nanmax(np.array([ct1_2_ct2_max_multisyn, ct2_2_ct1_max_multisyn])))
        multisyn_amount = range(1, max_multisyn + 1)
        ct1_2_ct2_multi_syn_amount = {}
        ct2_2_ct1_multi_syn_amount = {}
        ct1_2_ct2_multi_syn_sumsize = {}
        ct2_2_ct1_multi_syn_sumsize = {}
        for i in multisyn_amount:
            if i <= ct1_2_ct2_max_multisyn:
                ct1_2_ct2_multi_syn_amount[i] = len(np.where(ct1_2_ct2_percell_syn_amount == i)[0])
                ct1_2_ct2_multi_syn_sumsize[i] = np.sum(
                    ct1_2_ct2_percell_syn_size[np.where(ct1_2_ct2_percell_syn_amount == i)])
            if i <= ct2_2_ct1_max_multisyn:
                ct2_2_ct1_multi_syn_amount[i] = len(np.where(ct2_2_ct1_percell_syn_amount == i)[0])
                ct2_2_ct1_multi_syn_sumsize[i] = np.sum(ct2_2_ct1_percell_syn_size[np.where(ct2_2_ct1_percell_syn_amount == i)])

        if percentile_ct1 is not None:
            multisyn_plotting_amount = ComparingResultsForPLotting(celltype1=ct1_str,
                                                                   celltype2=ct2_str, filename=f_name,
                                                                   dictionary1=ct2_2_ct1_multi_syn_amount,
                                                                   dictionary2=ct1_2_ct2_multi_syn_amount, color1="#EAAE34",
                                                                   color2="#2F86A8")
            multisyn_plotting_sumsize = ComparingResultsForPLotting(celltype1=ct1_str,
                                                                    celltype2=ct2_str, filename=f_name,
                                                                    dictionary1=ct2_2_ct1_multi_syn_sumsize,
                                                                    dictionary2=ct1_2_ct2_multi_syn_sumsize, color1="#EAAE34",
                                                                   color2="#2F86A8")
        else:
            multisyn_plotting_amount = ComparingResultsForPLotting(celltype1=ct1_str,
                                                                   celltype2=ct2_str,
                                                                   filename=f_name,
                                                                   dictionary1=ct2_2_ct1_multi_syn_amount,
                                                                   dictionary2=ct1_2_ct2_multi_syn_amount)
            multisyn_plotting_sumsize = ComparingResultsForPLotting(celltype1=ct1_str,
                                                                    celltype2=ct2_str, filename=f_name,
                                                                    dictionary1=ct2_2_ct1_multi_syn_sumsize,
                                                                    dictionary2=ct1_2_ct2_multi_syn_sumsize)

        sum_max_multisyn = int(ct2_2_ct1_max_multisyn + ct1_2_ct2_max_multisyn)
        multisyn_df = pd.DataFrame(columns=["multisynapse amount", "sum size synapses", "amount of cells", "celltype"],
                                   index=range(sum_max_multisyn))
        multisyn_df.loc[0: ct2_2_ct1_max_multisyn - 1, "celltype"] = ct1_str
        multisyn_df.loc[ct2_2_ct1_max_multisyn: sum_max_multisyn - 1, "celltype"] = ct2_str
        multisyn_df.loc[0: ct2_2_ct1_max_multisyn - 1, "multisynapse amount"] = range(1, ct2_2_ct1_max_multisyn + 1)
        multisyn_df.loc[ct2_2_ct1_max_multisyn: sum_max_multisyn - 1, "multisynapse amount"] = range(1,
                                                                                                     ct1_2_ct2_max_multisyn + 1)
        for i, key in enumerate(ct2_2_ct1_multi_syn_amount.keys()):
            multisyn_df.loc[i, "amount of cells"] = ct2_2_ct1_multi_syn_amount[key]
            multisyn_df.loc[i, "sum size synapses"] = ct2_2_ct1_multi_syn_sumsize[key]
            multisyn_df.loc[ct2_2_ct1_max_multisyn + i, "amount of cells"] = ct1_2_ct2_multi_syn_amount[key]
            multisyn_df.loc[ct2_2_ct1_max_multisyn + i, "sum size synapses"] = ct1_2_ct2_multi_syn_sumsize[key]

        multisyn_df.to_csv("%s/multi_synapses_%s_%s.csv" % (f_name, ct1_str, ct2_str))
        multisyn_plotting_amount.plot_bar_hue(key = "multisynapse amount", x = "amount of cells", results_df = multisyn_df, hue = "celltype")
        multisyn_plotting_sumsize.plot_bar_hue(key="multisynapse amount", x="sum size synapses", results_df=multisyn_df,
                                              hue="celltype")

    # put all synsizes array into dictionary (but not into dataframe)
    ct1_2_ct2_syn_dict["all synapse sizes"] = ct1_2_ct2_all_syn_sizes
    ct2_2_ct1_syn_dict["all synapse sizes"] = ct2_2_ct1_all_syn_sizes

    #put multisynapse dictionary into dict but not dataframe
    if len(ct1_2_ct2_percell_syn_amount) != 0 and len(ct2_2_ct1_percell_syn_amount):
        ct1_2_ct2_syn_dict["multisynapse amount"] = ct1_2_ct2_multi_syn_amount
        ct1_2_ct2_syn_dict["multisynapse sum size"] = ct1_2_ct2_multi_syn_sumsize
        ct2_2_ct1_syn_dict["multisynapse amount"] = ct2_2_ct1_multi_syn_amount
        ct2_2_ct1_syn_dict["multisynapse sum size"] = ct2_2_ct1_multi_syn_sumsize

    write_obj2pkl("%s/%s_2_%s_dict.pkl" % (f_name, ct1_str, ct2_str), ct1_2_ct2_syn_dict)
    write_obj2pkl("%s/%s_2_%s_dict.pkl" % (f_name, ct2_str, ct1_str), ct2_2_ct1_syn_dict)

    ct1_2_ct2_resultsdict = ResultsForPlotting(celltype=ct2_str, filename=f_name,
                                               dictionary=ct1_2_ct2_syn_dict)
    ct2_2_ct1_resultsdict = ResultsForPlotting(celltype=ct1_str, filename=f_name,
                                               dictionary=ct1_2_ct2_syn_dict)

    #plot parameters as distplot
    # also make plots for amount and size (absolute, relative) for different compartments
    ticks = np.arange(4)
    ct1_2_ct2_resultsdict.multiple_param_labels(comp_labels, ticks)
    ct2_2_ct1_resultsdict.multiple_param_labels(comp_labels, ticks)

    for key in ct1_2_ct2_syn_dict.keys():
        if "ids" in key or "multi" in key:
            continue
        if "all" in key:
            ct1_2_ct2_resultsdict.plot_hist(key=key, subcell="synapse", cells= False, celltype2=ct1_str)
            ct2_2_ct1_resultsdict.plot_hist(key=key, subcell="synapse", cells= False, celltype2=ct2_str)
        else:
            ct1_2_ct2_resultsdict.plot_hist(key=key, subcell="synapse", celltype2=ct1_str)
            ct2_2_ct1_resultsdict.plot_hist(key=key, subcell="synapse", celltype2=ct2_str)
        if comp_labels[0] in key:
            key_split = key.split("-")
            key2 = key_split[0] + "- " + comp_labels[1]
            key3 = key_split[0] + "- " + comp_labels[2]
            key4 = key_split[0] + "- " + comp_labels[3]
            param_list_ct2= [ct1_2_ct2_pd[key], ct1_2_ct2_pd[key2], ct1_2_ct2_pd[key3], ct1_2_ct2_pd[key4]]
            ct1_2_ct2_resultsdict.plot_violin_params(key = key_split[0], param_list = param_list_ct2, subcell = "synapse", stripplot= True, celltype2 = ct1_str, outgoing = False)
            ct1_2_ct2_resultsdict.plot_box_params(key=key_split[0], param_list=param_list_ct2, subcell="synapse",
                                                     stripplot=False, celltype2= ct1_str, outgoing = False)
            param_list_ct1 = [ct2_2_ct1_pd[key], ct2_2_ct1_pd[key2], ct2_2_ct1_pd[key3], ct2_2_ct1_pd[key4]]
            ct2_2_ct1_resultsdict.plot_violin_params(key=key_split[0], param_list=param_list_ct1, subcell="synapse",
                                                     stripplot=True, celltype2=ct2_str, outgoing=False)
            ct2_2_ct1_resultsdict.plot_box_params(key=key_split[0], param_list=param_list_ct1, subcell="synapse",
                                                  stripplot=False, celltype2=ct2_str, outgoing=False)

    plottime = time.time() - syntime
    print("%.2f sec for calculating parameters, plotting" % plottime)
    time_stamps.append(time.time())
    step_idents.append('calculating last parameters, plotting')

    log.info("Connectivity analysis between 2 celltypes (%s, %s) finished" % (ct1_str, ct2_str))

    return f_name

def synapses_ax2ct(sd_synssv, celltype1, filename, cellids1, celltype2, cellids2, full_cells_ct2= True,
                         min_comp_len = 100, min_syn_size = 0.1, syn_prob_thresh = 0.8, label_ct1 = None, label_ct2 = None):
    '''
    looks at basic connectivty parameters between two celltypes (one axon: ct1) such as amount of synapses, average of synapses between cell types but also
    the average from one cell to the same other cell. Also looks at distribution of axo_dendritic synapses onto spines/shaft and the percentage of axo-somatic
    synapses. Uses cached synapse properties. Uses compartment_length per cell to ignore cells with not enough axon/dendrite
    # spiness values: 0 = spine neck, 1 = spine head, 2 = dendritic shaft, 3 = other
    axoness values: 0 = dendrite, 1 = axon, 2 = soma
    :param sd_synssv: segmentation dataset for synapses.
    :param celltype1, celltype2: celltypes to be compared. j0256: STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7,
                      FS=8, LTS=9, NGF=10
    :param cellids1, cellids2: cellids for celltypes 1 and 2
    :param full_cells_ct2: if True: full_cell_dict will be tried to load for celltype 2
    :param min_comp_len: minimum length for axon/dendrite to have to include cell in analysis
    :param min_syn_size: minimum size for synapses
    :param syn_prob_thresh: threshold for synapse probability
    :param label_ct1, label_ct2: label of celltypes or subgroups not in ct_dict
    :return: f_name: foldername in which results are stored
    '''

    start = time.time()
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    if label_ct1 is None:
        ct1_str = ct_dict[celltype1]
    else:
        ct1_str = label_ct1
    if label_ct2 is None:
        ct2_str = ct_dict[celltype2]
    else:
        ct2_str = label_ct2

    f_name = "%s/syn_conn_%s_2_%s_mcl%i_sysi_%.2f_st_%.2f" % (
            filename, ct1_str, ct2_str, min_comp_len, min_syn_size, syn_prob_thresh)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('compartment volume estimation', log_dir=f_name + '/logs/')
    log.info("parameters: celltype1 = %s, celltype2 = %s, min_comp_length = %.i, min_syn_size = %.2f, syn_prob_thresh = %.2f" %
             (ct1_str, ct2_str, min_comp_len, min_syn_size, syn_prob_thresh))
    time_stamps = [time.time()]
    step_idents = ['t-0']
    ax_dict = load_pkl2obj("/wholebrain/scratch/arother/j0251v4_prep/ax_%.3s_dict.pkl" % ct_dict[celltype1])
    if full_cells_ct2:
        try:
            full_cell_dict_ct2 = load_pkl2obj("/wholebrain/scratch/arother/j0251v4_prep/full_%.3s_dict.pkl" % ct_dict[celltype2])
        except FileNotFoundError:
            print("preprocessed parameters not available for ct1")


    log.info("Step 1/4 Iterate over %s to check min_comp_len" % ct1_str)
    # use axon and dendrite length dictionaries to lookup axon and dendrite length in future versions
    checked_cells = np.zeros(len(cellids1))
    for i, cellid in enumerate(tqdm(cellids1)):
        cell_axon_length = ax_dict[cellid]["axon length"]
        if cell_axon_length < min_comp_len:
            continue
        checked_cells[i] = cellid
    cellids1 = checked_cells[checked_cells > 0]

    ct1time = time.time() - start
    print("%.2f sec for iterating through %s cells" % (ct1time, ct1_str))
    time_stamps.append(time.time())
    step_idents.append('iterating over %s cells' % ct1_str)

    log.info("Step 2/4 Iterate over %s to check min_comp_len" % ct2_str)
    cellids2 = check_comp_lengths_ct(cellids2, fullcelldict = full_cell_dict_ct2, min_comp_len = min_comp_len)



    ct2time = time.time() - ct1time
    print("%.2f sec for iterating through %s cells" % (ct2time, ct2_str))
    time_stamps.append(time.time())
    step_idents.append('iterating over %s cells' % ct2_str)


    log.info("Step 3/4 get synaptic connectivity parameters")
    log.info("Step 3a: prefilter synapse caches")
    # prepare synapse caches with synapse threshold
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness = filter_synapse_caches_for_ct(sd_synssv, pre_cts = [celltype1], post_cts = [celltype2],
                                                                                           syn_prob_thresh = syn_prob_thresh, min_syn_size = min_syn_size, axo_den_so = True)
    prepsyntime = time.time() - ct2time
    print("%.2f sec for preprocessing synapses" % prepsyntime)
    time_stamps.append(time.time())
    step_idents.append('preprocessing synapses')


    log.info("Step 3b: iterate over synapses to get synaptic connectivity parameters")
    param_labels = ["amount synapses", "sum size synapses"]
    # spiness values: 0 = spine neck, 1 = spine head, 2 = dendritic shaft, 3 = other
    comp_labels = ["spine neck", "spine head", "shaft", "soma"]

    ct1_2_ct2_syn_dict = {"cellids": cellids2}
    for pi in param_labels:
        ct1_2_ct2_syn_dict[pi] = np.zeros(len(cellids2))
        for ci in comp_labels:
            ct1_2_ct2_syn_dict[pi + " - " + ci] = np.zeros(len(cellids2))

    ct1_2_ct2_percell_syn_amount = np.zeros((len(cellids2), len(cellids1))).astype(float)
    ct1_2_ct2_percell_syn_size = np.zeros((len(cellids2), len(cellids1))).astype(float)
    ct1_2_ct2_all_syn_sizes = np.zeros(len(m_ids))

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
        if ct_ax == ct_deso and celltype2 is not None:
            continue
        syn_size = m_sizes[i]
        if ssv_ax in cellids1:
            cell2_ind = np.where(ct1_2_ct2_syn_dict["cellids"] == ssv_deso)[0]
            cell1_ind = np.where(cellids1 == ssv_ax)[0]
            ct1_2_ct2_syn_dict[param_labels[0]][cell2_ind] += 1
            ct1_2_ct2_syn_dict[param_labels[1]][cell2_ind] += syn_size
            ct1_2_ct2_all_syn_sizes[i] = syn_size
            ct1_2_ct2_percell_syn_amount[cell2_ind, cell1_ind] += 1
            ct1_2_ct2_percell_syn_size[cell2_ind, cell1_ind] += syn_size
            if deso == 0:
                if spin_deso <= 2:
                    ct1_2_ct2_syn_dict[param_labels[0] + " - " + comp_labels[spin_deso]][cell2_ind] += 1
                    ct1_2_ct2_syn_dict[param_labels[1] + " - " + comp_labels[spin_deso]][cell2_ind] += syn_size
            else:
                ct1_2_ct2_syn_dict[param_labels[0] + " - " + "soma"][cell2_ind] += 1
                ct1_2_ct2_syn_dict[param_labels[1] + " - " + "soma"][cell2_ind] += syn_size

    syntime = time.time() - prepsyntime
    print("%.2f sec for processing synapses" % syntime)
    time_stamps.append(time.time())
    step_idents.append('processing synapses')

    log.info("Step 4/4: calculate last parameters (average syn size, average syn amount and syn size between 2 cells) and plotting")
    ct1_2_ct2_all_syn_sizes = ct1_2_ct2_all_syn_sizes[ct1_2_ct2_all_syn_sizes > 0]

    ct2_syn_inds = ct1_2_ct2_syn_dict["amount synapses"] > 0

    for key in ct1_2_ct2_syn_dict.keys():
        ct1_2_ct2_syn_dict[key] = ct1_2_ct2_syn_dict[key][ct2_syn_inds]

    ct1_2_ct2_syn_dict["average synapse size"] = ct1_2_ct2_syn_dict["sum size synapses"] / ct1_2_ct2_syn_dict["amount synapses"]

    for ci in comp_labels:
        ct1_2_ct2_syn_dict["average synapse size - " + ci] = ct1_2_ct2_syn_dict["sum size synapses - " + ci]/ ct1_2_ct2_syn_dict["amount synapses - " + ci]
        ct1_2_ct2_syn_dict["percentage synapse size - " + ci] = ct1_2_ct2_syn_dict["sum size synapses - " + ci] / \
                                                             ct1_2_ct2_syn_dict["sum size synapses"] * 100
        ct1_2_ct2_syn_dict["percentage synapse amount - " + ci] = ct1_2_ct2_syn_dict["amount synapses - " + ci] / \
                                                                ct1_2_ct2_syn_dict["amount synapses"] * 100


    #average synapse amount and size for one cell form ct1 to one cell of ct2
    ct1_2_ct2_percell_syn_amount = ct1_2_ct2_percell_syn_amount[ct2_syn_inds]
    ct1_2_ct2_percell_syn_amount[ct1_2_ct2_percell_syn_amount == 0] = np.nan
    ct1_2_ct2_syn_dict["avg amount one cell"] = np.nanmean(ct1_2_ct2_percell_syn_amount, axis = 1)

    ct1_2_ct2_percell_syn_size = ct1_2_ct2_percell_syn_size[ct2_syn_inds]
    ct1_2_ct2_percell_syn_size[ct1_2_ct2_percell_syn_size == 0] = np.nan
    ct1_2_ct2_syn_dict["avg syn size one cell"] = np.nanmean(ct1_2_ct2_percell_syn_size, axis = 1) / ct1_2_ct2_syn_dict["avg amount one cell"]


    #calculate amount of synapses, sum of synapse size in relation to dendritic pathlength, dendritic surface area
    dendritic_pathlengths_ct2 = np.zeros(len(ct1_2_ct2_syn_dict["cellids"]))
    dendritic_surface_area_ct2 = np.zeros(len(ct1_2_ct2_syn_dict["cellids"]))
    overall_amount_synapses_ct2 = np.zeros(len(ct1_2_ct2_syn_dict["cellids"]))
    overall_sum_synapses_ct2 = np.zeros(len(ct1_2_ct2_syn_dict["cellids"]))
    for i, cellid in enumerate(ct1_2_ct2_syn_dict["cellids"]):
        dendritic_pathlengths_ct2[i] = full_cell_dict_ct2[cellid]["dendrite length"]
        dendritic_surface_area_ct2[i] = full_cell_dict_ct2[cellid]["dendrite mesh surface area"]
        overall_amount_synapses_ct2[i] = full_cell_dict_ct2[cellid]["dendrite synapse amount"] + full_cell_dict_ct2[cellid]["soma synapse amount"]
        overall_sum_synapses_ct2[i] = full_cell_dict_ct2[cellid]["dendrite summed synapse size"] + full_cell_dict_ct2[cellid]["soma summed synapse size"]

    ct1_2_ct2_syn_dict["amount synapses per dendritic pathlength"] = (ct1_2_ct2_syn_dict[
                                                                         "amount synapses"] - ct1_2_ct2_syn_dict["amount synapses - soma"])/ dendritic_pathlengths_ct2
    ct1_2_ct2_syn_dict["amount synapses per dendritic surface area"] = (ct1_2_ct2_syn_dict[
                                                                           "amount synapses"] - ct1_2_ct2_syn_dict["amount synapses - soma"])/ dendritic_surface_area_ct2

    ct1_2_ct2_syn_dict["sum size synapses per dendritic pathlength"] = (ct1_2_ct2_syn_dict[
                                                                           "sum size synapses"] - ct1_2_ct2_syn_dict["sum size synapses - soma"])/ dendritic_pathlengths_ct2
    ct1_2_ct2_syn_dict["sum size synapses per dendritic surface area"] = (ct1_2_ct2_syn_dict[
                                                                             "sum size synapses"] - ct1_2_ct2_syn_dict["sum size synapses - soma"])/ dendritic_surface_area_ct2
    ct1_2_ct2_syn_dict["percentage amount synapses"] = (ct1_2_ct2_syn_dict[
                                                            "amount synapses"] / overall_amount_synapses_ct2) * 100
    ct1_2_ct2_syn_dict["percentage sum size synapses"] = (ct1_2_ct2_syn_dict[
                                                              "sum size synapses"] / overall_sum_synapses_ct2) * 100

    for ci in comp_labels:
        ct1_2_ct2_syn_dict.pop("sum size synapses - " + ci)

    ct1_2_ct2_pd = pd.DataFrame(ct1_2_ct2_syn_dict)
    ct1_2_ct2_pd.to_csv("%s/%s_2_%s_dict.csv" % (f_name, ct1_str, ct2_str))


    # group average amount one cell by amount of synapses
    # make barplot
    if len(ct1_2_ct2_percell_syn_amount) != 0:
        max_multisyn = int(np.nanmax(ct1_2_ct2_percell_syn_amount))
        multisyn_amount = range(1, max_multisyn + 1)
        ct1_2_ct2_multi_syn_amount = {}
        ct1_2_ct2_multi_syn_sumsize = {}
        for i in multisyn_amount:
            ct1_2_ct2_multi_syn_amount[i] = len(np.where(ct1_2_ct2_percell_syn_amount == i)[0])
            ct1_2_ct2_multi_syn_sumsize[i] = np.sum(
                ct1_2_ct2_percell_syn_size[np.where(ct1_2_ct2_percell_syn_amount == i)])

        multisyn_plotting_amount = ResultsForPlotting(celltype = ct2_str, filename = f_name, dictionary = ct1_2_ct2_multi_syn_amount)
        multisyn_plotting_sumsize = ResultsForPlotting(celltype=ct2_str, filename=f_name,
                                                      dictionary=ct1_2_ct2_multi_syn_sumsize)

        multisyn_df = pd.DataFrame(columns=["multisynapse amount", "sum size synapses", "amount of cells", "celltype axon", "celltype tareted cell"],
                                   index=range(max_multisyn))
        multisyn_df.loc[0: max_multisyn - 1, "celltype axon"] = ct1_str
        multisyn_df.loc[0: max_multisyn - 1, "celltype tareted cell"] = ct2_str

        multisyn_df.loc[0: max_multisyn - 1, "multisynapse amount"] = range(1, max_multisyn + 1)
        for i, key in enumerate(ct1_2_ct2_multi_syn_amount.keys()):
            multisyn_df.loc[i, "amount of cells"] = ct1_2_ct2_multi_syn_amount[key]
            multisyn_df.loc[i, "sum size synapses"] = ct1_2_ct2_multi_syn_sumsize[key]

        multisyn_df.to_csv("%s/multi_synapses_%s_%s.csv" % (f_name, ct1_str, ct2_str))
        multisyn_plotting_amount.plot_bar(key = "multisynapse amount", x = "amount of cells", results_df = multisyn_df)
        multisyn_plotting_sumsize.plot_bar(key="multisynapse amount", x="sum size synapses", results_df=multisyn_df)

    # put all synsizes array into dictionary (but not into dataframe)
    ct1_2_ct2_syn_dict["all synapse sizes"] = ct1_2_ct2_all_syn_sizes

    #put multisynapse dictionary into dict but not dataframe
    if len(ct1_2_ct2_percell_syn_amount) != 0:
        ct1_2_ct2_syn_dict["multisynapse amount"] = ct1_2_ct2_multi_syn_amount
        ct1_2_ct2_syn_dict["multisynapse sum size"] = ct1_2_ct2_multi_syn_sumsize

    write_obj2pkl("%s/%s_2_%s_dict.pkl" % (f_name, ct1_str, ct2_str), ct1_2_ct2_syn_dict)

    ct1_2_ct2_resultsdict = ResultsForPlotting(celltype=ct2_str, filename=f_name,
                                               dictionary=ct1_2_ct2_syn_dict)

    #plot parameters as distplot
    # also make plots for amount and size (absolute, relative) for different compartments
    ticks = np.arange(4)
    ct1_2_ct2_resultsdict.multiple_param_labels(comp_labels, ticks)

    for key in ct1_2_ct2_syn_dict.keys():
        if "ids" in key or "multi" in key:
            continue
        if "all" in key:
            ct1_2_ct2_resultsdict.plot_hist(key=key, subcell="synapse", cells= False, celltype2=ct1_str)
        else:
            ct1_2_ct2_resultsdict.plot_hist(key=key, subcell="synapse", celltype2=ct1_str)
        if comp_labels[0] in key:
            key_split = key.split("-")
            key2 = key_split[0] + "- " + comp_labels[1]
            key3 = key_split[0] + "- " + comp_labels[2]
            key4 = key_split[0] + "- " + comp_labels[3]
            param_list_ct2= [ct1_2_ct2_pd[key], ct1_2_ct2_pd[key2], ct1_2_ct2_pd[key3], ct1_2_ct2_pd[key4]]
            ct1_2_ct2_resultsdict.plot_violin_params(key = key_split[0], param_list = param_list_ct2, subcell = "synapse", stripplot= True, celltype2 = ct1_str, outgoing = False)
            ct1_2_ct2_resultsdict.plot_box_params(key=key_split[0], param_list=param_list_ct2, subcell="synapse",
                                                     stripplot=False, celltype2= ct1_str, outgoing = False)
    plottime = time.time() - syntime
    print("%.2f sec for calculating parameters, plotting" % plottime)
    time_stamps.append(time.time())
    step_idents.append('calculating last parameters, plotting')

    log.info("Connectivity analysis between 2 celltypes (%s, %s) finished" % (ct1_str, ct2_str))

    return f_name

def compare_connectivity(comp_ct1, filename, comp_ct2 = None, connected_ct = None, percentile = None, foldername_ct1 = None, foldername_ct2 = None, min_comp_len = 100, label_ct1 = None, label_ct2 = None, label_conn_ct = None):
    '''
    compares connectivity parameters between two celltypes or connectivity of a third celltype to the two celltypes. Connectivity parameters are calculated in
    synapses_between2cts. Parameters include synapse amount and average synapse size, as well as amount and average synapse size in shaft, soma, spine head and spine neck.
    P-value will be calculated with scipy.stats.ranksum.
    :param comp_ct1, comp_ct2: celltypes to compare to each other
    :param connected_ct: if given, connectivity of this celltype to two others will be compared
    :param percentile: if given not two celltypes but different population swithin a celltype will be compared
    :param foldername_ct1, foldername_ct2: foldernames where parameters of connectivity are stored
    :param min_comp_len: minimum compartment length
    :param label_ct1, label_ct2, label_conn_ct: celltype labels deviating from ct_dict e.g. for subpopulations
    :return: summed synapse sizes
    '''
    start = time.time()
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    axon_cts = [1, 3, 4]
    if percentile is None and comp_ct2 is None:
        raise ValueError("either celltypes or percentiles must be compared")
    if label_ct1 is None:
        ct1_str = ct_dict[comp_ct1]
    else:
        ct1_str = label_ct1
    if comp_ct2 is not None:
        if label_ct2 is None:
            ct2_str = ct_dict[comp_ct2]
        else:
            ct2_str = label_ct2
    if percentile is not None:
        if percentile == 50:
            raise ValueError("Due to ambiguity, value has to be either 49 or 51")
        else:
            ct1_str= ct_dict[comp_ct1] + " p%.2i" % percentile
            ct2_str = ct_dict[comp_ct1] + " p%.2i" % (100 - percentile)
    if connected_ct is not None:
        if label_conn_ct is None:
            conn_ct_str = ct_dict[connected_ct]
        else:
            conn_ct_str = label_conn_ct
        f_name = "%s/comp_conn_%s_%s_%s_syn_con_comp_mcl%i" % (
            filename, ct1_str, ct2_str, conn_ct_str, min_comp_len)
    else:
        f_name = "%s/comp_conn_%s_%s_syn_con_comp_mcl%i" % (
            filename, ct1_str, ct2_str, min_comp_len)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('compare connectivty between two celltypes', log_dir=f_name + '/logs/')
    if connected_ct is not None:
        log.info("parameters: celltype1 = %s,celltype2 = %s , connected ct = %s, min_comp_length = %.i" % (
            ct1_str, ct2_str, conn_ct_str, min_comp_len))
    else:
        log.info("parameters: celltype1 = %s,celltype2 = %s, min_comp_length = %.i" % (
           ct1_str, ct2_str, min_comp_len))
    time_stamps = [time.time()]
    step_idents = ['t-0']
    if connected_ct is not None:
        ct2_syn_dict = load_pkl2obj("%s/%s_2_%s_dict.pkl" % (foldername_ct2, conn_ct_str, ct2_str))
        ct1_syn_dict = load_pkl2obj("%s/%s_2_%s_dict.pkl" % (foldername_ct1, conn_ct_str, ct1_str))
    else:
        ct2_syn_dict = load_pkl2obj(
            "%s/%s_2_%s_dict.pkl" % (foldername_ct2, ct1_str, ct2_str))
        ct1_syn_dict = load_pkl2obj(
            "%s/%s_2_%s_dict.pkl" % (foldername_ct1, ct2_str, ct1_str))
    syn_dict_keys = list(ct1_syn_dict.keys())
    log.info("compute statistics for comparison, create violinplot and histogram")
    ranksum_results = pd.DataFrame(columns=syn_dict_keys[1:], index=["stats", "p value"])


    #put dictionaries into ComparingResultsForPlotting to make plotting of results easier
    if percentile is not None:
        results_comparison = ComparingResultsForPLotting(celltype1=ct1_str,
                                                         celltype2=ct2_str, filename=f_name,
                                                         dictionary1=ct1_syn_dict, dictionary2=ct2_syn_dict,
                                                         color1="#EAAE34",
                                                         color2="#2F86A8")
    else:
        results_comparison = ComparingResultsForPLotting(celltype1=ct1_str, celltype2=ct2_str,
                                                         filename=f_name, dictionary1=ct1_syn_dict,
                                                         dictionary2=ct2_syn_dict, color1 = "#592A87", color2 = "#2AC644")
    if "multisynapse amount" in ct1_syn_dict.keys():
        result_df_multi_params = results_comparison.result_df_categories(label_category= "compartment")

        if connected_ct is not None:
            result_df_multi_params.to_csv("%s/%s_2_%s_%s_syn_compartments.csv" % (f_name, conn_ct_str, ct1_str, ct2_str))
        else:
            result_df_multi_params.to_csv("%s/%s_%s_syn_compartments.csv" % (
                f_name, ct1_str, ct2_str))

        for key in result_df_multi_params.keys():
            if "celltype" in key or "compartment" in key:
                continue
            if connected_ct is not None:
                results_comparison.plot_violin_hue(x = "compartment", key = key, subcell="synapse", results_df = result_df_multi_params, hue = "celltype", conn_celltype= conn_ct_str, outgoing=False, stripplot=True)
                results_comparison.plot_box_hue(x = "compartment", key = key, subcell="synapse", results_df = result_df_multi_params, hue = "celltype", conn_celltype= conn_ct_str, outgoing=False, stripplot=False)
            else:
                results_comparison.plot_violin_hue(x="compartment", key=key, subcell="synapse", results_df=result_df_multi_params, hue="celltype",
                                                   stripplot=True)
                results_comparison.plot_box_hue(x="compartment", key=key, subcell="synapse", results_df=result_df_multi_params, hue="celltype",
                                                stripplot=False)

    for key in ct1_syn_dict.keys():
        if "ids" in key or "multi" in key:
            continue
        # calculate p_value for parameter
        stats, p_value = ranksums(ct1_syn_dict[key], ct2_syn_dict[key])
        ranksum_results.loc["stats", key] = stats
        ranksum_results.loc["p value", key] = p_value
        # plot parameter as violinplot
        if "-" not in key:
            results_for_plotting = results_comparison.result_df_per_param(key)
            if connected_ct is not None:
                results_comparison.plot_violin(key, results_for_plotting, subcell="synapse", stripplot=True, conn_celltype=conn_ct_str, outgoing=False)
                results_comparison.plot_box(key, results_for_plotting, subcell="synapse", stripplot= False,
                                        conn_celltype=conn_ct_str, outgoing = False)
            else:
                results_comparison.plot_violin(key, results_for_plotting, subcell="synapse", stripplot=True)
                results_comparison.plot_box(key, results_for_plotting, subcell="synapse", stripplot=False)
        if connected_ct is not None:
            if "all" in key:
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, cells=False, norm_hist=False,
                                                        conn_celltype=conn_ct_str, outgoing=False)
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, cells=False, norm_hist=True,
                                                        conn_celltype=conn_ct_str, outgoing=False)
            else:
                results_comparison.plot_hist_comparison(key, subcell = "synapse", bins = 10, norm_hist=False, conn_celltype=conn_ct_str, outgoing=False)
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, norm_hist=True,
                                                        conn_celltype=conn_ct_str, outgoing=False)
        else:
            if "all" in key:
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, cells=False, norm_hist=False,
                                                        outgoing=False)
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, cells=False, norm_hist=True,
                                                        outgoing=False)
            else:
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, norm_hist=False)
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, norm_hist=True)

    if connected_ct is not None:
        ranksum_results.to_csv("%s/ranksum_%s_2_%s_%s.csv" % (f_name, conn_ct_str, ct1_str, ct2_str))
    else:
        ranksum_results.to_csv("%s/ranksum_%s_%s.csv" % (f_name, ct1_str, ct2_str))

    #compare multisynapses in boxplot
    # only if connected_ct as otherwise already in synapses_between2cts
    if connected_ct is not None:
        if "multisynapse amount" in ct1_syn_dict.keys():
            ct1_max_multisyn = len(ct1_syn_dict["multisynapse amount"].keys())
            ct2_max_multisyn = len(ct2_syn_dict["multisynapse amount"].keys())
            sum_max_multisyn = ct1_max_multisyn + ct2_max_multisyn
            multisyn_df = pd.DataFrame(columns=["multisynapse amount", "sum size synapses", "amount of cells", "celltype"],
                                       index=range(sum_max_multisyn))
            multisyn_df.loc[0: ct1_max_multisyn - 1, "celltype"] = ct1_str
            multisyn_df.loc[ct1_max_multisyn: sum_max_multisyn - 1, "celltype"] = ct2_str
            multisyn_df.loc[0: ct1_max_multisyn - 1, "multisynapse amount"] = range(1, ct1_max_multisyn + 1)
            multisyn_df.loc[ct1_max_multisyn: sum_max_multisyn - 1, "multisynapse amount"] = range(1, ct2_max_multisyn + 1)
            for i, key in enumerate(ct1_syn_dict["multisynapse amount"].keys()):
                multisyn_df.loc[i, "amount of cells"] = ct1_syn_dict["multisynapse amount"][key]
                multisyn_df.loc[i, "sum size synapses"] = ct1_syn_dict["multisynapse sum size"][key]
                try:
                    multisyn_df.loc[ct1_max_multisyn + i, "amount of cells"] = ct2_syn_dict["multisynapse amount"][key]
                    multisyn_df.loc[ct1_max_multisyn + i, "sum size synapses"] = ct2_syn_dict["multisynapse sum size"][key]
                except KeyError:
                    continue

            multisyn_df.to_csv("%s/multi_synapses_%s_2_%s_%s.csv" % (f_name, conn_ct_str, ct1_str, ct2_str))
            results_comparison.plot_bar_hue(key="multisynapse amount", x="amount of cells", results_df=multisyn_df,
                                                  hue="celltype", conn_celltype = conn_ct_str, outgoing = False)
            results_comparison.plot_bar_hue(key="multisynapse amount", x="sum size synapses", results_df=multisyn_df,
                                                   hue="celltype", conn_celltype = conn_ct_str, outgoing = False)

    #calculate summed synapse size per celltype
    summed_synapse_sizes = {}
    if connected_ct is not None:
        summed_synapse_sizes[(conn_ct_str, ct1_str)] = np.sum(ct1_syn_dict["sum size synapses"])
        summed_synapse_sizes[(conn_ct_str, ct2_str)] = np.sum(ct2_syn_dict["sum size synapses"])
    else:
        summed_synapse_sizes[(ct2_str, ct1_str)] = np.sum(ct1_syn_dict["sum size synapses"])
        summed_synapse_sizes[(ct1_str, ct2_str)] = np.sum(ct2_syn_dict["sum size synapses"])


    #also compare outgoing connections from celltype, only needed if connected ct is not axon
    if connected_ct is not None and connected_ct not in axon_cts:
        ct2_syn_dict = load_pkl2obj("%s/%s_2_%s_dict.pkl" % (foldername_ct2, ct2_str, conn_ct_str))
        ct1_syn_dict = load_pkl2obj(
            "%s/%s_2_%s_dict.pkl" % (foldername_ct1, ct1_str, conn_ct_str))

        if percentile is not None:
            results_comparison = ComparingResultsForPLotting(celltype1=ct1_str,
                                                             celltype2=ct2_str, filename=f_name,
                                                             dictionary1=ct1_syn_dict, dictionary2=ct2_syn_dict,
                                                             color1="#EAAE34",
                                                             color2="#2F86A8")
        else:
            results_comparison = ComparingResultsForPLotting(celltype1=ct1_str,
                                                             celltype2=ct2_str,
                                                             filename=f_name, dictionary1=ct1_syn_dict,
                                                             dictionary2=ct2_syn_dict, color1 = "#592A87", color2 = "#2AC644")

        result_df_multi_params = results_comparison.result_df_categories(label_category="compartment")
        result_df_multi_params.to_csv("%s/%s_%s_2_%s_syn_compartments_outgoing.csv" % (
            f_name, ct1_str, ct2_str, conn_ct_str))

        for key in result_df_multi_params.keys():
            if "celltype" in key or "compartment" in key:
                continue
            results_comparison.plot_violin_hue(x="compartment", key=key, subcell="synapse", results_df=result_df_multi_params,
                                               hue="celltype", conn_celltype=conn_ct_str,
                                               outgoing=True, stripplot=True)
            results_comparison.plot_box_hue(x="compartment", key=key, subcell="synapse", results_df=result_df_multi_params, hue="celltype",
                                            conn_celltype=conn_ct_str, outgoing=True,
                                            stripplot=False)

        for key in ct1_syn_dict.keys():
            if "ids" in key or "multi" in key:
                continue
            # calculate p_value for parameter
            stats, p_value = ranksums(ct1_syn_dict[key], ct2_syn_dict[key])
            ranksum_results.loc["stats", key] = stats
            ranksum_results.loc["p value", key] = p_value
            # plot parameter as violinplot
            if not "spine" in key and not "soma" in key and not "shaft" in key:
                results_for_plotting = results_comparison.result_df_per_param(key)
                results_comparison.plot_violin(key, results_for_plotting, subcell="synapse", stripplot=True,
                                               conn_celltype=conn_ct_str, outgoing=True)
                results_comparison.plot_box(key, results_for_plotting, subcell="synapse", stripplot=False,
                                            conn_celltype=conn_ct_str, outgoing=True)
            if "all" in key:
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, cells=False, norm_hist=False,
                                                        conn_celltype=conn_ct_str, outgoing=True)
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, cells=False, norm_hist=True,
                                                        conn_celltype=conn_ct_str, outgoing=True)
            else:
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, norm_hist=False,
                                                        conn_celltype=conn_ct_str, outgoing=True)
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, norm_hist=True,
                                                        conn_celltype=conn_ct_str, outgoing=True)

        ranksum_results.to_csv("%s/ranksum_%s_%s_2_%s_outgoing.csv" % (f_name, ct1_str, ct2_str, conn_ct_str))

        summed_synapse_sizes[(ct1_str, conn_ct_str)] = np.sum(ct1_syn_dict["sum size synapses"])
        summed_synapse_sizes[(ct2_str, conn_ct_str)] = np.sum(ct2_syn_dict["sum size synapses"])

        #multisynapse barplot for outgoing synapses
        ct1_max_multisyn = len(ct1_syn_dict["multisynapse amount"].keys())
        ct2_max_multisyn = len(ct2_syn_dict["multisynapse amount"].keys())
        sum_max_multisyn = ct1_max_multisyn + ct2_max_multisyn
        multisyn_df = pd.DataFrame(columns=["multisynapse amount", "sum size synapses", "amount of cells", "celltype"],
                                   index=range(sum_max_multisyn))
        multisyn_df.loc[0: ct1_max_multisyn - 1, "celltype"] = ct1_str
        multisyn_df.loc[ct1_max_multisyn: sum_max_multisyn - 1, "celltype"] = ct2_str
        multisyn_df.loc[0: ct1_max_multisyn - 1, "multisynapse amount"] = range(1, ct1_max_multisyn + 1)
        multisyn_df.loc[ct1_max_multisyn: sum_max_multisyn - 1, "multisynapse amount"] = range(1, ct2_max_multisyn + 1)
        for i, key in enumerate(ct1_syn_dict["multisynapse amount"].keys()):
            multisyn_df.loc[i, "amount of cells"] = ct1_syn_dict["multisynapse amount"][key]
            multisyn_df.loc[i, "sum size synapses"] = ct1_syn_dict["multisynapse sum size"][key]
            try:
                multisyn_df.loc[ct1_max_multisyn + i, "amount of cells"] = ct2_syn_dict["multisynapse amount"][key]
                multisyn_df.loc[ct1_max_multisyn + i, "sum size synapses"] = ct2_syn_dict["multisynapse sum size"][key]
            except KeyError:
                continue

        multisyn_df.to_csv(
            "%s/multi_synapses_%s_%s_2_%s.csv" % (f_name, ct1_str, ct2_str, conn_ct_str))
        results_comparison.plot_bar_hue(key="multisynapse amount", x="amount of cells", results_df=multisyn_df,
                                        hue="celltype", conn_celltype=conn_ct_str,
                                        outgoing=True)
        results_comparison.plot_bar_hue(key="multisynapse amount", x="sum size synapses", results_df=multisyn_df,
                                        hue="celltype", conn_celltype=conn_ct_str,
                                        outgoing=True)

    sum_synapses_pd = pd.DataFrame(summed_synapse_sizes, index = [0])
    if connected_ct is not None:
        sum_synapses_pd.to_csv("%s/%s_%s_%s_sum_synapses_per_ct.csv" % (f_name, ct1_str,ct2_str, conn_ct_str))
    else:
        sum_synapses_pd.to_csv("%s/%s_%s_sum_synapses_per_ct.csv" % (
        f_name, ct1_str, ct2_str))

    #make networkx graph for sum of synapses per celltype
    if connected_ct is not None:
        filename = "%s/sum_synapse_size_ct_%s_%s_%s_nxgraph.png" % (f_name, ct1_str, ct2_str, conn_ct_str)
    else:
        filename = "%s/sum_synapse_size_ct_%s_%s_nxgraph.png" % (
        f_name, ct1_str, ct2_str)
    plot_nx_graph(results_dictionary = summed_synapse_sizes, filename = filename, title = "sum of synapse size")

    plottime = time.time() - start
    print("%.2f sec for statistics and plotting" % plottime)
    time_stamps.append(time.time())
    step_idents.append('comparing celltypes')
    log.info("comparing celltypes via connectivity finished")

    return summed_synapse_sizes

def compare_connectivity_multiple(comp_cts, filename, foldernames, connected_ct, min_comp_len = 100, label_cts = None, label_conn_ct = None, colours = None):
    '''
    compares connectivity parameters between several celltypes or connectivity of a celltype to the several (>2) celltypes. Connectivity parameters are calculated in
    synapses_between2cts. Parameters include synapse amount and average synapse size, as well as amount and average synapse size in shaft, soma, spine head and spine neck.
    P-value will be calculated with scipy.stats.ranksum.
    :param comp_cts: list of celltypes to compare to each other
    :param filename: path were results should be stored
    :param foldernames: list of foldernames where connectivity parameters are stored
    :param connected_ct: if given, connectivity of this celltype to two others will be compared
    :param percentile: if given not two celltypes but different population swithin a celltype will be compared
    :param min_comp_len: minimum compartment length
    :param label_cts: list of celltype labels to be compared if subpopulations, if None: celltype labels will be used
    :param label_conn_ct: celltype labels deviating from ct_dict e.g. for subpopulations
    :param colours = list of colors that should be used for plotting, same length as comp_cts
    :return: summed synapse sizes
    '''
    start = time.time()
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    axon_cts = [1, 3, 4]
    if label_cts is None:
        label_cts = [ct_dict[i] for i in comp_cts]
    if connected_ct is not None:
        if label_conn_ct is None:
            conn_ct_str = ct_dict[connected_ct]
        else:
            conn_ct_str = label_conn_ct
        f_name = "%s/comp_conn_%s_%s_%s_syn_con_comp_mcl%i" % (
            filename, label_cts[0], label_cts[1], conn_ct_str, min_comp_len)
    else:
        f_name = "%s/comp_conn_%s_%s_syn_con_comp_mcl%i" % (
            filename, label_cts[0], label_cts[1], min_comp_len)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('compare connectivty between two celltypes', log_dir=f_name + '/logs/')
    log.info("parameters: amount of celltypes to be compared: %i, ct1 = %s, ct2 = %s , connected ct = %s, min_comp_length = %.i" % (
        len(comp_cts), label_cts[0], label_cts[1], conn_ct_str, min_comp_len))
    time_stamps = [time.time()]
    step_idents = ['t-0']
    syn_dict_list = [load_pkl2obj("%s/%s_2_%s_dict.pkl" % (foldernames[i], conn_ct_str, label_cts[i] for i in range(len(comp_cts))))]
    syn_dicts = {[conn_ct_str, label_cts[i]]: syn_dict_list[1] for i in range(len(comp_cts))}
    ct_connections = list(syn_dicts.keys())
    log.info("compute statistics for comparison, create violinplot and histogram")
    ranksum_results = pd.DataFrame(columns=ct_connections, index=range(len(ct_connections) * 2))


    #put dictionaries into ComparingResultsForPlotting to make plotting of results easier
    if colours is None:
        blues = ["#0ECCEB", "#0A95AB", "#06535F", "#065E6C", "#043C45"]
        colours = blues[:len(comp_cts)]
    results_comparison = ComparingMultipleForPLotting(ct_list = label_cts, filename=f_name,
                                                     dictionary_list = syn_dict_list,
                                                     colour_list = colours)

    if "multisynapse amount" in ct1_syn_dict.keys():
        result_df_multi_params = results_comparison.result_df_categories(label_category= "compartment")

        if connected_ct is not None:
            result_df_multi_params.to_csv("%s/%s_2_%s_%s_syn_compartments.csv" % (f_name, conn_ct_str, ct1_str, ct2_str))
        else:
            result_df_multi_params.to_csv("%s/%s_%s_syn_compartments.csv" % (
                f_name, ct1_str, ct2_str))

        for key in result_df_multi_params.keys():
            if "celltype" in key or "compartment" in key:
                continue
            if connected_ct is not None:
                results_comparison.plot_violin_hue(x = "compartment", key = key, subcell="synapse", results_df = result_df_multi_params, hue = "celltype", conn_celltype= conn_ct_str, outgoing=False, stripplot=True)
                results_comparison.plot_box_hue(x = "compartment", key = key, subcell="synapse", results_df = result_df_multi_params, hue = "celltype", conn_celltype= conn_ct_str, outgoing=False, stripplot=False)
            else:
                results_comparison.plot_violin_hue(x="compartment", key=key, subcell="synapse", results_df=result_df_multi_params, hue="celltype",
                                                   stripplot=True)
                results_comparison.plot_box_hue(x="compartment", key=key, subcell="synapse", results_df=result_df_multi_params, hue="celltype",
                                                stripplot=False)

    for key in ct1_syn_dict.keys():
        if "ids" in key or "multi" in key:
            continue
        # calculate p_value for parameter
        stats, p_value = ranksums(ct1_syn_dict[key], ct2_syn_dict[key])
        ranksum_results.loc["stats", key] = stats
        ranksum_results.loc["p value", key] = p_value
        # plot parameter as violinplot
        if "-" not in key:
            results_for_plotting = results_comparison.result_df_per_param(key)
            if connected_ct is not None:
                results_comparison.plot_violin(key, results_for_plotting, subcell="synapse", stripplot=True, conn_celltype=conn_ct_str, outgoing=False)
                results_comparison.plot_box(key, results_for_plotting, subcell="synapse", stripplot= False,
                                        conn_celltype=conn_ct_str, outgoing = False)
            else:
                results_comparison.plot_violin(key, results_for_plotting, subcell="synapse", stripplot=True)
                results_comparison.plot_box(key, results_for_plotting, subcell="synapse", stripplot=False)
        if connected_ct is not None:
            if "all" in key:
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, cells=False, norm_hist=False,
                                                        conn_celltype=conn_ct_str, outgoing=False)
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, cells=False, norm_hist=True,
                                                        conn_celltype=conn_ct_str, outgoing=False)
            else:
                results_comparison.plot_hist_comparison(key, subcell = "synapse", bins = 10, norm_hist=False, conn_celltype=conn_ct_str, outgoing=False)
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, norm_hist=True,
                                                        conn_celltype=conn_ct_str, outgoing=False)
        else:
            if "all" in key:
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, cells=False, norm_hist=False,
                                                        outgoing=False)
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, cells=False, norm_hist=True,
                                                        outgoing=False)
            else:
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, norm_hist=False)
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, norm_hist=True)

    if connected_ct is not None:
        ranksum_results.to_csv("%s/ranksum_%s_2_%s_%s.csv" % (f_name, conn_ct_str, ct1_str, ct2_str))
    else:
        ranksum_results.to_csv("%s/ranksum_%s_%s.csv" % (f_name, ct1_str, ct2_str))

    #compare multisynapses in boxplot
    # only if connected_ct as otherwise already in synapses_between2cts
    if connected_ct is not None:
        if "multisynapse amount" in ct1_syn_dict.keys():
            ct1_max_multisyn = len(ct1_syn_dict["multisynapse amount"].keys())
            ct2_max_multisyn = len(ct2_syn_dict["multisynapse amount"].keys())
            sum_max_multisyn = ct1_max_multisyn + ct2_max_multisyn
            multisyn_df = pd.DataFrame(columns=["multisynapse amount", "sum size synapses", "amount of cells", "celltype"],
                                       index=range(sum_max_multisyn))
            multisyn_df.loc[0: ct1_max_multisyn - 1, "celltype"] = ct1_str
            multisyn_df.loc[ct1_max_multisyn: sum_max_multisyn - 1, "celltype"] = ct2_str
            multisyn_df.loc[0: ct1_max_multisyn - 1, "multisynapse amount"] = range(1, ct1_max_multisyn + 1)
            multisyn_df.loc[ct1_max_multisyn: sum_max_multisyn - 1, "multisynapse amount"] = range(1, ct2_max_multisyn + 1)
            for i, key in enumerate(ct1_syn_dict["multisynapse amount"].keys()):
                multisyn_df.loc[i, "amount of cells"] = ct1_syn_dict["multisynapse amount"][key]
                multisyn_df.loc[i, "sum size synapses"] = ct1_syn_dict["multisynapse sum size"][key]
                try:
                    multisyn_df.loc[ct1_max_multisyn + i, "amount of cells"] = ct2_syn_dict["multisynapse amount"][key]
                    multisyn_df.loc[ct1_max_multisyn + i, "sum size synapses"] = ct2_syn_dict["multisynapse sum size"][key]
                except KeyError:
                    continue

            multisyn_df.to_csv("%s/multi_synapses_%s_2_%s_%s.csv" % (f_name, conn_ct_str, ct1_str, ct2_str))
            results_comparison.plot_bar_hue(key="multisynapse amount", x="amount of cells", results_df=multisyn_df,
                                                  hue="celltype", conn_celltype = conn_ct_str, outgoing = False)
            results_comparison.plot_bar_hue(key="multisynapse amount", x="sum size synapses", results_df=multisyn_df,
                                                   hue="celltype", conn_celltype = conn_ct_str, outgoing = False)

    #calculate summed synapse size per celltype
    summed_synapse_sizes = {}
    if connected_ct is not None:
        summed_synapse_sizes[(conn_ct_str, ct1_str)] = np.sum(ct1_syn_dict["sum size synapses"])
        summed_synapse_sizes[(conn_ct_str, ct2_str)] = np.sum(ct2_syn_dict["sum size synapses"])
    else:
        summed_synapse_sizes[(ct2_str, ct1_str)] = np.sum(ct1_syn_dict["sum size synapses"])
        summed_synapse_sizes[(ct1_str, ct2_str)] = np.sum(ct2_syn_dict["sum size synapses"])


    #also compare outgoing connections from celltype, only needed if connected ct is not axon
    if connected_ct is not None and connected_ct not in axon_cts:
        ct2_syn_dict = load_pkl2obj("%s/%s_2_%s_dict.pkl" % (foldername_ct2, ct2_str, conn_ct_str))
        ct1_syn_dict = load_pkl2obj(
            "%s/%s_2_%s_dict.pkl" % (foldername_ct1, ct1_str, conn_ct_str))

        if percentile is not None:
            results_comparison = ComparingResultsForPLotting(celltype1=ct1_str,
                                                             celltype2=ct2_str, filename=f_name,
                                                             dictionary1=ct1_syn_dict, dictionary2=ct2_syn_dict,
                                                             color1="#EAAE34",
                                                             color2="#2F86A8")
        else:
            results_comparison = ComparingResultsForPLotting(celltype1=ct1_str,
                                                             celltype2=ct2_str,
                                                             filename=f_name, dictionary1=ct1_syn_dict,
                                                             dictionary2=ct2_syn_dict, color1 = "#592A87", color2 = "#2AC644")

        result_df_multi_params = results_comparison.result_df_categories(label_category="compartment")
        result_df_multi_params.to_csv("%s/%s_%s_2_%s_syn_compartments_outgoing.csv" % (
            f_name, ct1_str, ct2_str, conn_ct_str))

        for key in result_df_multi_params.keys():
            if "celltype" in key or "compartment" in key:
                continue
            results_comparison.plot_violin_hue(x="compartment", key=key, subcell="synapse", results_df=result_df_multi_params,
                                               hue="celltype", conn_celltype=conn_ct_str,
                                               outgoing=True, stripplot=True)
            results_comparison.plot_box_hue(x="compartment", key=key, subcell="synapse", results_df=result_df_multi_params, hue="celltype",
                                            conn_celltype=conn_ct_str, outgoing=True,
                                            stripplot=False)

        for key in ct1_syn_dict.keys():
            if "ids" in key or "multi" in key:
                continue
            # calculate p_value for parameter
            stats, p_value = ranksums(ct1_syn_dict[key], ct2_syn_dict[key])
            ranksum_results.loc["stats", key] = stats
            ranksum_results.loc["p value", key] = p_value
            # plot parameter as violinplot
            if not "spine" in key and not "soma" in key and not "shaft" in key:
                results_for_plotting = results_comparison.result_df_per_param(key)
                results_comparison.plot_violin(key, results_for_plotting, subcell="synapse", stripplot=True,
                                               conn_celltype=conn_ct_str, outgoing=True)
                results_comparison.plot_box(key, results_for_plotting, subcell="synapse", stripplot=False,
                                            conn_celltype=conn_ct_str, outgoing=True)
            if "all" in key:
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, cells=False, norm_hist=False,
                                                        conn_celltype=conn_ct_str, outgoing=True)
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, cells=False, norm_hist=True,
                                                        conn_celltype=conn_ct_str, outgoing=True)
            else:
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, norm_hist=False,
                                                        conn_celltype=conn_ct_str, outgoing=True)
                results_comparison.plot_hist_comparison(key, subcell="synapse", bins=10, norm_hist=True,
                                                        conn_celltype=conn_ct_str, outgoing=True)

        ranksum_results.to_csv("%s/ranksum_%s_%s_2_%s_outgoing.csv" % (f_name, ct1_str, ct2_str, conn_ct_str))

        summed_synapse_sizes[(ct1_str, conn_ct_str)] = np.sum(ct1_syn_dict["sum size synapses"])
        summed_synapse_sizes[(ct2_str, conn_ct_str)] = np.sum(ct2_syn_dict["sum size synapses"])

        #multisynapse barplot for outgoing synapses
        ct1_max_multisyn = len(ct1_syn_dict["multisynapse amount"].keys())
        ct2_max_multisyn = len(ct2_syn_dict["multisynapse amount"].keys())
        sum_max_multisyn = ct1_max_multisyn + ct2_max_multisyn
        multisyn_df = pd.DataFrame(columns=["multisynapse amount", "sum size synapses", "amount of cells", "celltype"],
                                   index=range(sum_max_multisyn))
        multisyn_df.loc[0: ct1_max_multisyn - 1, "celltype"] = ct1_str
        multisyn_df.loc[ct1_max_multisyn: sum_max_multisyn - 1, "celltype"] = ct2_str
        multisyn_df.loc[0: ct1_max_multisyn - 1, "multisynapse amount"] = range(1, ct1_max_multisyn + 1)
        multisyn_df.loc[ct1_max_multisyn: sum_max_multisyn - 1, "multisynapse amount"] = range(1, ct2_max_multisyn + 1)
        for i, key in enumerate(ct1_syn_dict["multisynapse amount"].keys()):
            multisyn_df.loc[i, "amount of cells"] = ct1_syn_dict["multisynapse amount"][key]
            multisyn_df.loc[i, "sum size synapses"] = ct1_syn_dict["multisynapse sum size"][key]
            try:
                multisyn_df.loc[ct1_max_multisyn + i, "amount of cells"] = ct2_syn_dict["multisynapse amount"][key]
                multisyn_df.loc[ct1_max_multisyn + i, "sum size synapses"] = ct2_syn_dict["multisynapse sum size"][key]
            except KeyError:
                continue

        multisyn_df.to_csv(
            "%s/multi_synapses_%s_%s_2_%s.csv" % (f_name, ct1_str, ct2_str, conn_ct_str))
        results_comparison.plot_bar_hue(key="multisynapse amount", x="amount of cells", results_df=multisyn_df,
                                        hue="celltype", conn_celltype=conn_ct_str,
                                        outgoing=True)
        results_comparison.plot_bar_hue(key="multisynapse amount", x="sum size synapses", results_df=multisyn_df,
                                        hue="celltype", conn_celltype=conn_ct_str,
                                        outgoing=True)

    sum_synapses_pd = pd.DataFrame(summed_synapse_sizes, index = [0])
    if connected_ct is not None:
        sum_synapses_pd.to_csv("%s/%s_%s_%s_sum_synapses_per_ct.csv" % (f_name, ct1_str,ct2_str, conn_ct_str))
    else:
        sum_synapses_pd.to_csv("%s/%s_%s_sum_synapses_per_ct.csv" % (
        f_name, ct1_str, ct2_str))

    #make networkx graph for sum of synapses per celltype
    if connected_ct is not None:
        filename = "%s/sum_synapse_size_ct_%s_%s_%s_nxgraph.png" % (f_name, ct1_str, ct2_str, conn_ct_str)
    else:
        filename = "%s/sum_synapse_size_ct_%s_%s_nxgraph.png" % (
        f_name, ct1_str, ct2_str)
    plot_nx_graph(results_dictionary = summed_synapse_sizes, filename = filename, title = "sum of synapse size")

    plottime = time.time() - start
    print("%.2f sec for statistics and plotting" % plottime)
    time_stamps.append(time.time())
    step_idents.append('comparing celltypes')
    log.info("comparing celltypes via connectivity finished")

    return summed_synapse_sizes







