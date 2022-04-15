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
from wholebrain.scratch.arother.bio_analysis.general.result_helper import ResultsForPlotting, ComparingResultsForPLotting, plot_nx_graph

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
    log = initialize_logging('subpopulation goruping per connectivity', log_dir=f_name + '/logs/')
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
    both_cellids = unique_ct1_ct2ssvs[both_inds]
    both_syn_sumsizes = ct1ct2_syn_sumsizes[both_inds] + ct1ct3_syn_sumsizes[both_inds2]
    both_syn_amounts = ct1ct2_syn_amounts[both_inds] + ct1ct3_syn_amounts[both_inds2]
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

    write_obj2pkl("%sfull_%s_2_%s_dict_%i.pkl" % (f_name_saving, ct_dict[ct1], ct_dict[ct2], min_comp_len), only_ct2_dict)
    write_obj2pkl("%sfull_%s_2_%s_dict_%i.pkl" % (f_name_saving, ct_dict[ct1], ct_dict[ct3], min_comp_len),
                  only_ct3_dict)
    write_obj2pkl("%sfull_%s_2_%s%s_dict_%i.pkl" % (f_name_saving, ct_dict[ct1], ct_dict[ct2], ct_dict[ct3], min_comp_len),
                  both_dict)
    write_obj2pkl("%sfull_%s_2_%s_arr_%i.pkl" % (f_name_saving, ct_dict[ct1], ct_dict[ct2], min_comp_len),
                  only_2ct2_cellids)
    write_obj2pkl("%sfull_%s_2_%s_arr_%i.pkl" % (f_name_saving, ct_dict[ct1], ct_dict[ct3], min_comp_len),
                  only_2ct3_cellids)
    write_obj2pkl("%sfull_%s_2_%s%s_arr_%i.pkl" % (f_name_saving, ct_dict[ct1], ct_dict[ct2], ct_dict[ct3], min_comp_len),
                  both_dict)
    write_obj2pkl("%sfull_%s_no_conn_%s%s_arr_%i.pkl" % (f_name_saving, ct_dict[ct1], ct_dict[ct2], ct_dict[ct3], min_comp_len),
                  not_connected_ids)

    time_stamps = [time.time()]
    step_idents = ['t-0']

    log.info("Step 4/4: Compute statistics and plot results")

    conn_df = pd.DataFrame(columns=["cellids", "connection to cts", "synapse amount", "sum synapse size", "amount partners", "avg syn amount per partner", "avg syn size per partner"], index=range(len(connected_cellids1)))
    conn_df.loc[0:len(only_2ct2_cellids) - 1, "cellids"] = only_2ct2_cellids
    conn_df.loc[0:len(only_2ct2_cellids) - 1, "connection to cts"] = "only %s" % ct_dict[ct2]
    conn_df.loc[0:len(only_2ct2_cellids) - 1, "synapse amount"] = only_ct1ct2_syn_amounts
    conn_df.loc[0:len(only_2ct2_cellids) - 1, "sum synapse size"] = only_ct1ct2_syn_sumsizes
    conn_df.loc[len(only_2ct2_cellids):len(only_2ct2_cellids) + len(only_2ct3_cellids) - 1, "cellids"] = only_2ct3_cellids
    conn_df.loc[len(only_2ct2_cellids):len(only_2ct2_cellids) + len(only_2ct3_cellids) - 1, "connection to cts"] = "only %s" % ct_dict[ct3]
    conn_df.loc[len(only_2ct2_cellids):len(only_2ct2_cellids) + len(only_2ct3_cellids) - 1, "synapse amount"] = only_ct1ct3_syn_amounts
    conn_df.loc[len(only_2ct2_cellids):len(only_2ct2_cellids) + len(only_2ct3_cellids) - 1, "sum synapse size"] = only_ct1ct3_syn_sumsizes
    conn_df.loc[len(only_2ct2_cellids) + len(only_2ct3_cellids): len(connected_cellids1) - 1,
    "cellids"] = both_cellids
    conn_df.loc[len(only_2ct2_cellids) + len(only_2ct3_cellids): len(connected_cellids1) - 1,
    "connection to cts"] = "both"
    conn_df.loc[len(only_2ct2_cellids) + len(only_2ct3_cellids): len(connected_cellids1) - 1,
    "synapse amount"] = both_syn_amounts
    conn_df.loc[len(only_2ct2_cellids) + len(only_2ct3_cellids): len(connected_cellids1) - 1,
    "sum synapse size"] = both_syn_sumsizes
    for cellid in cellids1:
        if cellid in only_2ct2_cellids:
            conn_df.loc[np.where(only_2ct2_cellids == cellid)[0], "amount partners"] = len(
                only_ct2_dict[cellid]["ct partners"])
            conn_df.loc[np.where(only_2ct2_cellids == cellid)[0], "avg syn amount per partner"] = only_ct2_dict[cellid]["synapse amount"] / len(
                only_ct2_dict[cellid]["ct partners"])
            conn_df.loc[np.where(only_2ct2_cellids == cellid)[0], "avg syn size per partner"] = only_ct2_dict[cellid]["sum size synapses"] / len(
                only_ct2_dict[cellid]["ct partners"])
        elif cellid in only_2ct3_cellids:
            conn_df.loc[np.where(only_2ct3_cellids == cellid)[0] + len(only_2ct2_cellids), "amount partners"] = len(
                only_ct3_dict[cellid]["ct partners"])
            conn_df.loc[np.where(only_2ct3_cellids == cellid)[0] + len(only_2ct2_cellids), "avg syn amount per partner"] = only_ct3_dict[cellid][
                                                                                                      "synapse amount"] / len(
                only_ct3_dict[cellid]["ct partners"])
            conn_df.loc[np.where(only_2ct3_cellids == cellid)[0] + len(only_2ct2_cellids), "avg syn size per partner"] = only_ct3_dict[cellid][
                                                                                                    "sum size synapses"] / len(
                only_ct3_dict[cellid]["ct partners"])
        elif cellid in both_cellids:
            conn_df.loc[np.where(both_cellids == cellid)[0] + len(only_2ct2_cellids) + len(
                only_2ct3_cellids), "amount partners"] = len(
                both_dict[cellid]["ct partners"])
            conn_df.loc[np.where(both_cellids == cellid)[0] + len(only_2ct2_cellids) + len(
                only_2ct3_cellids), "avg syn amount per partner"] = both_dict[cellid]["synapse amount"] / len(
                both_dict[cellid]["ct partners"])
            conn_df.loc[np.where(both_cellids == cellid)[0] + len(only_2ct2_cellids) + len(
                only_2ct3_cellids), "avg syn size per partner"] = both_dict[cellid]["sum size synapses"] / len(both_dict[cellid]["ct partners"])
        else:
            continue

    conn_df.to_csv("%sresult_params.csv" % f_name)



    raise ValueError

    #save GPe, GPiids, they are connected to, compartments they are connected to
    #save data in dataframe and table
    #plot results and compute statistical values between groups
    #also include percentage of overall synamount for outgoing synapses
    #maybe even incoming synapses as GPe/i
    #save dictionaries and arrays for further use
    #then rewrite other functions to compare up to 4 groups

    return only_2ct2_cellids, only_2ct3_cellids, both_cellids, not_connected_ids



