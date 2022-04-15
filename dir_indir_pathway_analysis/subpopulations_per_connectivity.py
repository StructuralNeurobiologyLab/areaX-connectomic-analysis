# function to seperate MSN into groups based on connectivity to GPe/i
# four groups: GPe only, GPi only, GPe/i both and none
# plot amount and sum of synapses, cellids of GPe/i they synapse onto

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os as os
import time
from syconn.handler.config import initialize_logging
from syconn.handler.basics import load_pkl2obj
from tqdm import tqdm
from syconn.handler.basics import write_obj2pkl
from scipy.stats import ranksums
from wholebrain.scratch.arother.bio_analysis.general.analysis_helper import get_compartment_length, check_comp_lengths_ct, filter_synapse_caches_for_ct
from wholebrain.scratch.arother.bio_analysis.general.result_helper import ComparingMultipleForPLotting

def sort_by_connectivity(sd_synssv, ct1, ct2, ct3, cellids1, cellids2, cellids3, f_name, f_name_saving = None, min_comp_len = 200, syn_prob_thresh = 0.8, min_syn_size = 0.1):
    """
    sort one celltype into 4 groups based on connectivty to two other celltypes. Groups will be only one of them, neither or both.
    Also synapse amount, sum of synaptic area and cellids of the cells they synapsed onto will be looked at.
    :param sd_synssv: segmentation dataset for synapses.
    :param ct1: celltype to be sorted
    :param ct2, ct3: celltypes ct1 is connected to
    :param cellids1, cellids2, cellids3: cellids of corresponding celltypes, will be checked for
    minimal compartment length
    :param f_name: filename where plots should be saved
    :param f_name_saving: where cellids should be saved, if not given then f_name
    :param full_celldict1, full_celldict2, full_celledcit3: dictionaries with parameters of correpsonding celltypes
    :param min_comp_len: minimal compartment length for axon and dendrite
    :param syn_prob_thresh: synapse probability threshold
    :param min_syn_size: minimum synaose size
    :return:
    """
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    full_celldict1 = load_pkl2obj("/wholebrain/scratch/arother/j0251v4_prep/full_%.3s_dict.pkl" % ct_dict[ct1])
    full_celldict2 = load_pkl2obj("/wholebrain/scratch/arother/j0251v4_prep/full_%.3s_dict.pkl" % ct_dict[ct2])
    full_celldict3 = load_pkl2obj("/wholebrain/scratch/arother/j0251v4_prep/full_%.3s_dict.pkl" % ct_dict[ct3])
    if f_name_saving is None:
        f_name_saving = f_name
    log = initialize_logging('subpopulation grouping per connectivity', log_dir=f_name + '/logs/')
    log.info(
        "parameters: celltype1 = %s, celltype2 = %s, celltype3 = %s, min_comp_length = %.i, min_syn_size = %.2f, syn_prob_thresh = %.2f" %
        (ct_dict[ct1], ct_dict[ct2], ct_dict[ct3], min_comp_len, min_syn_size, syn_prob_thresh))
    time_stamps = [time.time()]
    step_idents = ['t-0']

    log.info("Step 1/4: Check compartment length of cells from three celltypes")
    cellids1 = check_comp_lengths_ct(cellids1, fullcelldict=full_celldict1, min_comp_len=min_comp_len)
    cellids2 = check_comp_lengths_ct(cellids2, fullcelldict=full_celldict2, min_comp_len=min_comp_len)
    cellids3 = check_comp_lengths_ct(cellids3, fullcelldict=full_celldict3, min_comp_len=min_comp_len)

    log.info("Step 2/4: Prefilter synapses for synapses between these celltypes")
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness = filter_synapse_caches_for_ct(sd_synssv,
                                                                                           pre_cts=[ct1], post_cts = [ct2, ct3],
                                                                                           syn_prob_thresh=syn_prob_thresh,
                                                                                           min_syn_size=min_syn_size,
                                                                                           axo_den_so=True)

    time_stamps = [time.time()]
    step_idents = ['t-0']

    log.info("Step 3/4: Sort celltype into groups based on connectivity")
    #filter synapses that are not from cellids 1
    ct1ids_inds = np.any(np.in1d(m_ssv_partners, cellids1).reshape(len(m_cts), 2), axis=1)
    m_cts = m_cts[ct1ids_inds]
    m_ids = m_ids[ct1ids_inds]
    m_axs = m_axs[ct1ids_inds]
    m_ssv_partners = m_ssv_partners[ct1ids_inds]
    m_sizes = m_sizes[ct1ids_inds]
    m_spiness = m_spiness[ct1ids_inds]
    #filter synapses that are not from cellids 2 or 3
    ct2ids_inds = np.any(np.in1d(m_ssv_partners, np.hstack([cellids2, cellids3])).reshape(len(m_ssv_partners), 2), axis=1)
    m_cts = m_cts[ct2ids_inds]
    m_ids = m_ids[ct2ids_inds]
    m_axs = m_axs[ct2ids_inds]
    m_ssv_partners = m_ssv_partners[ct2ids_inds]
    m_sizes = m_sizes[ct2ids_inds]
    m_spiness = m_spiness[ct2ids_inds]
    #sort according to connectivity
    #seperate synapses into connecteted to cellids2 and cellids3
    conn_ct2_inds = np.any(np.in1d(m_ssv_partners, cellids2).reshape(len(m_ssv_partners), 2), axis=1)
    conn_ct3_inds = np.any(np.in1d(m_ssv_partners, cellids3).reshape(len(m_ssv_partners), 2), axis=1)
    ct2_ssv_partners = m_ssv_partners[conn_ct2_inds]
    ct2_sizes = m_sizes[conn_ct2_inds]
    ct2_spiness = m_spiness[conn_ct2_inds]
    ct2_cts = m_cts[conn_ct2_inds]
    ct3_ssv_partners = m_ssv_partners[conn_ct3_inds]
    ct3_sizes = m_sizes[conn_ct3_inds]
    ct3_spiness = m_spiness[conn_ct3_inds]
    ct3_cts = m_cts[conn_ct3_inds]
    #get unique ids for cellids1 from each of them
    ct1_ct2_inds = np.where(ct2_cts == ct1)
    ct1_ct2_ssvsids = ct2_ssv_partners[ct1_ct2_inds]
    ct1ct2_ssv_inds, unique_ct1_ct2ssvs = pd.factorize(ct1_ct2_ssvsids)
    ct1ct2_syn_sumsizes = np.bincount(ct1ct2_ssv_inds, ct2_sizes)
    ct1ct2_syn_amounts = np.bincount(ct1ct2_ssv_inds)
    ct1_ct3_inds = np.where(ct3_cts == ct1)
    ct1_ct3_ssvsids = ct3_ssv_partners[ct1_ct3_inds]
    ct1ct3_ssv_inds, unique_ct1_ct3ssvs = pd.factorize(ct1_ct3_ssvsids)
    ct1ct3_syn_sumsizes = np.bincount(ct1ct3_ssv_inds, ct3_sizes)
    ct1ct3_syn_amounts = np.bincount(ct1ct3_ssv_inds)
    #split in three groups: oonly ct2, only ct3, both
    both_inds = np.in1d(unique_ct1_ct2ssvs, unique_ct1_ct3ssvs)
    both_inds2 = np.in1d(unique_ct1_ct3ssvs, unique_ct1_ct2ssvs)
    arg_ct2 = np.argsort(unique_ct1_ct2ssvs[both_inds], axis = 0)
    arg_ct3 = np.argsort(unique_ct1_ct3ssvs[both_inds2], axis = 0)
    sorted_ct2_ids = np.take_along_axis(unique_ct1_ct2ssvs[both_inds], arg_ct2, axis= 0)
    both_cellids = sorted_ct2_ids
    sorted_ct2_amounts = np.take_along_axis(ct1ct2_syn_amounts[both_inds], arg_ct2, axis= 0)
    sorted_ct2_sumsizes = np.take_along_axis(ct1ct2_syn_sumsizes[both_inds], arg_ct2, axis=0)
    sorted_ct3_amounts = np.take_along_axis(ct1ct3_syn_amounts[both_inds2], arg_ct3, axis=0)
    sorted_ct3_sumsizes = np.take_along_axis(ct1ct3_syn_sumsizes[both_inds2], arg_ct3, axis=0)
    both_syn_sumsizes = sorted_ct2_sumsizes + sorted_ct3_sumsizes
    both_syn_amounts = sorted_ct2_amounts + sorted_ct3_amounts
    only_2ct2_cellids = unique_ct1_ct2ssvs[both_inds == False]
    only_ct1ct2_syn_sumsizes = ct1ct2_syn_sumsizes[both_inds == False]
    only_ct1ct2_syn_amounts = ct1ct2_syn_amounts[both_inds == False]
    only_2ct3_cellids = unique_ct1_ct3ssvs[both_inds2 == False]
    only_ct1ct3_syn_sumsizes = ct1ct3_syn_sumsizes[both_inds2 == False]
    only_ct1ct3_syn_amounts = ct1ct3_syn_amounts[both_inds2 == False]
    #get ct2, ct3 cellids each cell is connected to
    ct1_ct2_partner_inds = np.where(ct2_cts == ct2)
    ct1_ct2_partners = ct2_ssv_partners[ct1_ct2_partner_inds]
    ct1_ct3_partner_inds = np.where(ct3_cts == ct3)
    ct1_ct3_partners = ct3_ssv_partners[ct1_ct3_partner_inds]
    # make dictionary for each group
    only_ct2_dict = {cellid: {"synapse amount": only_ct1ct2_syn_amounts[i], "sum size synapses": only_ct1ct2_syn_sumsizes[i], "ct partners": ct1_ct2_partners[np.where(ct1_ct2_ssvsids == cellid)]} for i, cellid in enumerate(only_2ct2_cellids)}
    only_ct3_dict = {
        cellid: {"synapse amount": only_ct1ct3_syn_amounts[i], "sum size synapses": only_ct1ct3_syn_sumsizes[i], "ct partners": ct1_ct3_partners[np.where(ct1_ct3_ssvsids == cellid)]} for
        i, cellid in enumerate(only_2ct3_cellids)}
    both_dict = {cellid: {"synapse amount": both_syn_amounts[i], "sum size synapses": both_syn_sumsizes[i], "ct partners": np.unique(np.hstack([ct1_ct2_partners[np.where(ct1_ct2_ssvsids == cellid)], ct1_ct3_partners[np.where(ct1_ct3_ssvsids == cellid)]]))} for i, cellid in enumerate(both_cellids)}
    #those that are in fullcells of ct1 and do not have nay connectivity here are in group four
    connected_cellids1 = np.hstack([only_2ct3_cellids, only_2ct2_cellids, both_cellids])
    not_conn_inds = np.in1d(cellids1, connected_cellids1) == False
    not_connected_ids = cellids1[not_conn_inds]



    time_stamps = [time.time()]
    step_idents = ['t-0']

    log.info("Step 4/4: Compute statistics and plot results")

    conn_df = pd.DataFrame(columns=["cellids", "connection to cts", "synapse amount", "sum size synapses", "amount partners", "avg synapse amount per partner", "avg synapse size per partner"], index=range(len(connected_cellids1)))
    conn_df.loc[0:len(only_2ct2_cellids) - 1, "cellids"] = only_2ct2_cellids
    conn_df.loc[0:len(only_2ct2_cellids) - 1, "connection to cts"] = "only %s" % ct_dict[ct2]
    conn_df.loc[0:len(only_2ct2_cellids) - 1, "synapse amount"] = only_ct1ct2_syn_amounts
    conn_df.loc[0:len(only_2ct2_cellids) - 1, "sum size synapses"] = only_ct1ct2_syn_sumsizes
    conn_df.loc[len(only_2ct2_cellids):len(only_2ct2_cellids) + len(only_2ct3_cellids) - 1, "cellids"] = only_2ct3_cellids
    conn_df.loc[len(only_2ct2_cellids):len(only_2ct2_cellids) + len(only_2ct3_cellids) - 1, "connection to cts"] = "only %s" % ct_dict[ct3]
    conn_df.loc[len(only_2ct2_cellids):len(only_2ct2_cellids) + len(only_2ct3_cellids) - 1, "synapse amount"] = only_ct1ct3_syn_amounts
    conn_df.loc[len(only_2ct2_cellids):len(only_2ct2_cellids) + len(only_2ct3_cellids) - 1, "sum size synapses"] = only_ct1ct3_syn_sumsizes
    conn_df.loc[len(only_2ct2_cellids) + len(only_2ct3_cellids): len(connected_cellids1) - 1,
    "cellids"] = both_cellids
    conn_df.loc[len(only_2ct2_cellids) + len(only_2ct3_cellids): len(connected_cellids1) - 1,
    "connection to cts"] = "both"
    conn_df.loc[len(only_2ct2_cellids) + len(only_2ct3_cellids): len(connected_cellids1) - 1,
    "synapse amount"] = both_syn_amounts
    conn_df.loc[len(only_2ct2_cellids) + len(only_2ct3_cellids): len(connected_cellids1) - 1,
    "sum size synapses"] = both_syn_sumsizes
    key_list = list(conn_df.keys())
    key_list.remove("connection to cts")
    ct2_arr_dict = {key: np.zeros(len(only_2ct2_cellids)) for key in key_list}
    ct3_arr_dict = {key: np.zeros(len(only_2ct3_cellids)) for key in key_list}
    both_arr_dict = {key: np.zeros(len(both_cellids)) for key in key_list}
    ct2_arr_dict["cellids"] = only_2ct2_cellids
    ct3_arr_dict["cellids"] = only_2ct3_cellids
    both_arr_dict["cellids"] = both_cellids
    for cellid in cellids1:
        if cellid in only_2ct2_cellids:
            amount_partners = len(only_ct2_dict[cellid]["ct partners"])
            avg_syn_amount_partner = only_ct2_dict[cellid]["synapse amount"] / len(only_ct2_dict[cellid]["ct partners"])
            avg_syn_size_partner = only_ct2_dict[cellid]["sum size synapses"] / len(only_ct2_dict[cellid]["ct partners"])
            conn_df.loc[np.where(only_2ct2_cellids == cellid)[0], "amount partners"] = amount_partners
            conn_df.loc[np.where(only_2ct2_cellids == cellid)[0], "avg synapse amount per partner"] = avg_syn_amount_partner
            conn_df.loc[np.where(only_2ct2_cellids == cellid)[0], "avg synapse size per partner"] = avg_syn_size_partner
            only_ct2_dict[cellid]["amount partners"] = amount_partners
            only_ct2_dict[cellid]["avg synapse amount per partner"] = avg_syn_amount_partner
            only_ct2_dict[cellid]["avg synapse size per partner"] = avg_syn_size_partner
            ind = np.where(ct2_arr_dict["cellids"] == cellid)
            for key in key_list:
                if "cellids" in key:
                    continue
                ct2_arr_dict[key][ind] = only_ct2_dict[cellid][key]
        elif cellid in only_2ct3_cellids:
            amount_partners = len(only_ct3_dict[cellid]["ct partners"])
            avg_syn_amount_partner = only_ct3_dict[cellid]["synapse amount"] / len(only_ct3_dict[cellid]["ct partners"])
            avg_syn_size_partner = only_ct3_dict[cellid]["sum size synapses"] / len(
                only_ct3_dict[cellid]["ct partners"])
            conn_df.loc[np.where(only_2ct3_cellids == cellid)[0] + len(only_2ct2_cellids), "amount partners"] = amount_partners
            conn_df.loc[np.where(only_2ct3_cellids == cellid)[0] + len(only_2ct2_cellids), "avg synapse amount per partner"] = avg_syn_amount_partner
            conn_df.loc[np.where(only_2ct3_cellids == cellid)[0] + len(only_2ct2_cellids), "avg synapse size per partner"] = avg_syn_size_partner
            only_ct3_dict[cellid]["amount partners"] = amount_partners
            only_ct3_dict[cellid]["avg synapse amount per partner"] = avg_syn_amount_partner
            only_ct3_dict[cellid]["avg synapse size per partner"] = avg_syn_size_partner
            ind = np.where(ct3_arr_dict["cellids"] == cellid)
            for key in key_list:
                if "cellids" in key:
                    continue
                ct3_arr_dict[key][ind] = only_ct3_dict[cellid][key]
        elif cellid in both_cellids:
            amount_partners = len(both_dict[cellid]["ct partners"])
            avg_syn_amount_partner = both_dict[cellid]["synapse amount"] / len(both_dict[cellid]["ct partners"])
            avg_syn_size_partner = both_dict[cellid]["sum size synapses"] / len(
                both_dict[cellid]["ct partners"])
            conn_df.loc[np.where(both_cellids == cellid)[0] + len(only_2ct2_cellids) + len(
                only_2ct3_cellids), "amount partners"] = amount_partners
            conn_df.loc[np.where(both_cellids == cellid)[0] + len(only_2ct2_cellids) + len(
                only_2ct3_cellids), "avg synapse amount per partner"] = avg_syn_amount_partner
            conn_df.loc[np.where(both_cellids == cellid)[0] + len(only_2ct2_cellids) + len(
                only_2ct3_cellids), "avg synapse size per partner"] = avg_syn_size_partner
            both_dict[cellid]["amount partners"] = amount_partners
            both_dict[cellid]["avg synapse amount per partner"] = avg_syn_amount_partner
            both_dict[cellid]["avg synapse size per partner"] = avg_syn_size_partner
            ind = np.where(both_arr_dict["cellids"] == cellid)
            for key in key_list:
                if "cellids" in key:
                    continue
                both_arr_dict[key][ind] = both_dict[cellid][key]
        else:
            continue

    #state type of columns with float entries to prevent plotting issues
    for key in key_list[1:]:
        conn_df[key] = conn_df[key].astype(float)

    conn_df.to_csv("%s/result_params.csv" % f_name)

    write_obj2pkl("%s/full_%s_2_%s_dict_%i.pkl" % (f_name_saving, ct_dict[ct1], ct_dict[ct2], min_comp_len),
                  only_ct2_dict)
    write_obj2pkl("%s/full_%s_2_%s_dict_%i.pkl" % (f_name_saving, ct_dict[ct1], ct_dict[ct3], min_comp_len),
                  only_ct3_dict)
    write_obj2pkl(
        "%s/full_%s_2_%s%s_dict_%i.pkl" % (f_name_saving, ct_dict[ct1], ct_dict[ct2], ct_dict[ct3], min_comp_len),
        both_dict)
    write_obj2pkl("%s/full_%s_2_%s_arr_%i.pkl" % (f_name_saving, ct_dict[ct1], ct_dict[ct2], min_comp_len),
                  only_2ct2_cellids)
    write_obj2pkl("%s/full_%s_2_%s_arr_%i.pkl" % (f_name_saving, ct_dict[ct1], ct_dict[ct3], min_comp_len),
                  only_2ct3_cellids)
    write_obj2pkl(
        "%s/full_%s_2_%s%s_arr_%i.pkl" % (f_name_saving, ct_dict[ct1], ct_dict[ct2], ct_dict[ct3], min_comp_len),
        both_dict)
    write_obj2pkl(
        "%s/full_%s_no_conn_%s%s_arr_%i.pkl" % (f_name_saving, ct_dict[ct1], ct_dict[ct2], ct_dict[ct3], min_comp_len),
        not_connected_ids)

    write_obj2pkl("%s/%s_2_%s_dict.pkl" % (f_name, ct_dict[ct1], ct_dict[ct2]), ct2_arr_dict)
    write_obj2pkl("%s/%s_2_%s_dict.pkl" % (f_name, ct_dict[ct1], ct_dict[ct3]), ct3_arr_dict)
    write_obj2pkl("%s/%s_2_both%s%s_dict.pkl" % (f_name, ct_dict[ct1], ct_dict[ct2], ct_dict[ct3]), both_arr_dict)

    conn_synapses = ComparingMultipleForPLotting(ct_list = ["only %s" % ct_dict[ct2], "only %s" % ct_dict[ct3], "both"], filename = f_name, dictionary_list = [ct2_arr_dict, ct3_arr_dict, both_arr_dict], colour_list = ["#EAAE34", '#2F86A8', "#707070"])

    r_columns = ["connected to only %s vs only %s" % (ct_dict[ct2], ct_dict[ct3]), "connected to both vs only %s" % ct_dict[ct2], "connected to both vs only %s" % ct_dict[ct3]]
    ranksum_results = pd.DataFrame(columns= r_columns)
    for key in ct2_arr_dict:
        if "cellids" in key:
            continue
        # calculate p_value for parameter
        stats_23, p_value_23 = ranksums(ct2_arr_dict[key], ct3_arr_dict[key])
        stats_b2, p_value_b2 = ranksums(both_arr_dict[key], ct2_arr_dict[key])
        stats_b3, p_value_b3 = ranksums(both_arr_dict[key], ct3_arr_dict[key])
        ranksum_results.loc["stats - " + key, r_columns[0]] = stats_23
        ranksum_results.loc["p value - " + key, r_columns[0]] = p_value_23
        ranksum_results.loc["stats - " + key, r_columns[1]] = stats_b2
        ranksum_results.loc["p value - " + key, r_columns[1]] = p_value_b2
        ranksum_results.loc["stats - " + key, r_columns[2]] = stats_b3
        ranksum_results.loc["p value - " + key, r_columns[2]] = p_value_b3
        conn_synapses.plot_violin(key = key, x = "connection to cts", result_df = conn_df, subcell="synapse", stripplot=True)
        conn_synapses.plot_box(key=key, x = "connection to cts", result_df=conn_df, subcell="synapse", stripplot=False)
        conn_synapses.plot_hist_comparison(key=key, subcell="synapse", cells=True, norm_hist = True, bins= 10)

    ranksum_results.to_csv("%s/ranksum_results.csv" % f_name)

    return only_2ct2_cellids, only_2ct3_cellids, both_cellids, not_connected_ids



