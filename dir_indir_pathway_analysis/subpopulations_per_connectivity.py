#functions to sort groups based onto connectivity

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os as os
import seaborn as sns
import time
from syconn.handler.config import initialize_logging
from syconn.handler.basics import load_pkl2obj
from tqdm import tqdm
from syconn.handler.basics import write_obj2pkl
from scipy.stats import ranksums
from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_morph_helper import get_compartment_length, check_comp_lengths_ct, get_compartment_nodes, get_cell_nodes_ax
from cajal.nvmescratch.users.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct, synapse_amount_sumsize_between2cts, filter_contact_caches_for_cellids, get_contact_site_axoness_percell
from cajal.nvmescratch.users.arother.bio_analysis.general.result_helper import ComparingMultipleForPLotting, ResultsForPlotting
from multiprocessing import pool
from functools import partial

def sort_by_connectivity(sd_synssv, ct1, ct2, ct3, cellids1, cellids2, cellids3, f_name, celldicts, sd_csssv = None, f_name_saving = None, min_comp_len = 200, syn_prob_thresh = 0.8, min_syn_size = 0.1):
    """
    sort one celltype into 4 groups based on connectivty to two other celltypes. Groups will be only one of them, neither or both.
    Also synapse amount, sum of synaptic area and cellids of the cells they synapsed onto will be looked at.
    :param sd_synssv: segmentation dataset for synapses.
    :param ct1: celltype to be sorted
    :param ct2, ct3: celltypes ct1 is connected to
    :param cellids1, cellids2, cellids3: cellids of corresponding celltypes, will be checked for
    minimal compartment length
    :param f_name: filename where plots should be saved
    :param celldicts: list of three dictionaries for loading full celldicts
    :param sd_csssv: segmentation dataset for contact sites
    :param f_name_saving: where cellids should be saved, if not given then f_name
    :param full_celldict1, full_celldict2, full_celledcit3: dictionaries with parameters of correpsonding celltypes
    :param min_comp_len: minimal compartment length for axon and dendrite
    :param syn_prob_thresh: synapse probability threshold
    :param min_syn_size: minimum synaose size
    :return:
    """
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    full_celldict1 , full_celldict2, full_celldict3 = celldicts
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
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord = filter_synapse_caches_for_ct(sd_synssv = sd_synssv,
                                                                                           pre_cts=[ct1], post_cts = [ct2, ct3],
                                                                                           syn_prob_thresh=syn_prob_thresh,
                                                                                           min_syn_size=min_syn_size,
                                                                                           axo_den_so=True)
    if sd_csssv is not None:
        post_cellids = np.hstack([cellids2, cellids3])
        cs_partners, cs_ids, cs_coords = filter_contact_caches_for_cellids(sd_csssv, cellids1 = cellids1, cellids2 = post_cellids)
        #order contact site ids, cs partners and cs coords per cell
        #make list of dictionaries
        # find only contact sites that are on cellid1 axon
        cs_cell_dict_list_ct1 = []
        for ci, cellid in enumerate(tqdm(cellids1)):
            percell_cs_partner_inds = np.where(cs_partners == cellid)[0]
            cell_dict = {}
            cell_dict["cs partners"] = cs_partners[percell_cs_partner_inds]
            cell_dict["cs ids"] = cs_ids[percell_cs_partner_inds]
            cell_dict["cs coords"] = cs_coords[percell_cs_partner_inds]
            cell_dict["cellid"] = cellid
            cs_cell_dict_list_ct1.append(cell_dict)
        p = pool.Pool()
        cs_result_dicts_ct1 = p.map(partial(get_contact_site_axoness_percell, compartment = 1), tqdm(cs_cell_dict_list_ct1))
        #find contact sites that are close to postcellid dendrites
        cs_cell_dict_list_post = []
        for ci, cellid in enumerate(tqdm(post_cellids)):
            percell_cs_partner_inds = np.where(cs_partners == cellid)[0]
            cell_dict = {}
            cell_dict["cs partners"] = cs_partners[percell_cs_partner_inds]
            cell_dict["cs ids"] = cs_ids[percell_cs_partner_inds]
            cell_dict["cs coords"] = cs_coords[percell_cs_partner_inds]
            cell_dict["cellid"] = cellid
            cs_cell_dict_list_post.append(cell_dict)
        p = pool.Pool()
        cs_result_dicts_post = p.map(partial(get_contact_site_axoness_percell, compartment=0), tqdm(cs_cell_dict_list_post))
        cs_ids_ct1 = []
        for ci, cellid in enumerate(tqdm(cellids1)):
            cs_ids_ct1.append(cs_result_dicts_ct1[ci]["cs ids"])
        cs_ids_post = []
        for ci, cellid in enumerate(tqdm(post_cellids)):
            cs_ids_post.append(cs_result_dicts_post[ci]["cs ids"])
        cs_ids_ct1 = np.hstack(np.array(cs_ids_ct1, dtype=object))
        cs_ids_post = np.hstack(np.array(cs_ids_post, dtype=object))
        if len(cs_ids_ct1) > len(cs_ids_post):
            suitable_cs_inds_ct = np.in1d(cs_ids_ct1, cs_ids_post)
            suitable_cs_ids = cs_ids_ct1[suitable_cs_inds_ct]
        else:
            suitable_cs_inds_ct = np.in1d(cs_ids_post, cs_ids_ct1)
            suitable_cs_ids = cs_ids_post[suitable_cs_inds_ct]
        suitable_cs_inds = np.in1d(cs_ids, suitable_cs_ids)
        suitable_cs_coords = cs_coords[suitable_cs_inds]
        suitable_cs_partners = cs_partners[suitable_cs_inds]
        suitable_cs_ids = cs_ids[suitable_cs_inds]
        for ci, cellid in enumerate(tqdm(cellids1)):
            cs_inds = np.in1d(cs_result_dicts_ct1[ci]["cs ids"], suitable_cs_ids)
            cs_result_dicts_ct1[ci]["cs ids"] = cs_result_dicts_ct1[ci]["cs ids"][cs_inds]
            cs_result_dicts_ct1[ci]["cs partners"] = cs_result_dicts_ct1[ci]["cs partners"][cs_inds]
            cs_result_dicts_ct1[ci]["cs coords"] = cs_result_dicts_ct1[ci]["cs coords"][cs_inds]
        for ci, cellid in enumerate(tqdm(post_cellids)):
            cs_inds = np.in1d(cs_result_dicts_post[ci]["cs ids"], suitable_cs_ids)
            cs_result_dicts_post[ci]["cs ids"] = cs_result_dicts_post[ci]["cs ids"][cs_inds]
            cs_result_dicts_post[ci]["cs partners"] = cs_result_dicts_post[ci]["cs partners"][cs_inds]
            cs_result_dicts_post[ci]["cs coords"] = cs_result_dicts_post[ci]["cs coords"][cs_inds]

    time_stamps = [time.time()]
    step_idents = ['prefilter synapses done']

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
    only_ct2_dict = {cellid: {"synapse amount": only_ct1ct2_syn_amounts[i], "sum size synapses": only_ct1ct2_syn_sumsizes[i], "ct partners": np.unique(ct1_ct2_partners[np.where(ct1_ct2_ssvsids == cellid)])} for i, cellid in enumerate(only_2ct2_cellids)}
    only_ct3_dict = {
        cellid: {"synapse amount": only_ct1ct3_syn_amounts[i], "sum size synapses": only_ct1ct3_syn_sumsizes[i], "ct partners": np.unique(ct1_ct3_partners[np.where(ct1_ct3_ssvsids == cellid)])} for
        i, cellid in enumerate(only_2ct3_cellids)}
    both_dict = {cellid: {"synapse amount": both_syn_amounts[i], "sum size synapses": both_syn_sumsizes[i], "ct partners": np.unique(np.hstack([ct1_ct2_partners[np.where(ct1_ct2_ssvsids == cellid)], ct1_ct3_partners[np.where(ct1_ct3_ssvsids == cellid)]]))} for i, cellid in enumerate(both_cellids)}
    #those that are in fullcells of ct1 and do not have nay connectivity here are in group four
    connected_cellids1 = np.hstack([only_2ct3_cellids, only_2ct2_cellids, both_cellids])
    not_conn_inds = np.in1d(cellids1, connected_cellids1) == False
    not_connected_ids = cellids1[not_conn_inds]

    if sd_csssv is not None:
        #relate cs ids back to synapses and per cell
        not_conn_dict = {}
        for ci, cellid in enumerate(tqdm(cellids1)):
            ct2_ct3_cs_inds = np.where(cs_result_dicts_ct1[ci]["cs partners"] != cellid)
            ct2_ct3_cs_cellids = cs_result_dicts_ct1[ci]["cs partners"][ct2_ct3_cs_inds]
            ct2_cs_inds = np.in1d(ct2_ct3_cs_cellids, cellids2)
            ct2_cs_ids = cs_result_dicts_ct1[ci]["cs ids"][ct2_cs_inds]
            ct2_cs_cellids = ct2_ct3_cs_cellids[ct2_cs_inds]
            unique_ct2_cs_cellids = np.unique(ct2_cs_cellids)
            ct3_cs_inds = np.in1d(ct2_ct3_cs_cellids, cellids3)
            ct3_cs_ids = cs_result_dicts_ct1[ci]["cs ids"][ct3_cs_inds]
            ct3_cs_cellids = ct2_ct3_cs_cellids[ct3_cs_inds]
            unique_ct3_cs_cellids = np.unique(ct3_cs_cellids)
            if cellid in only_2ct2_cellids:
                only_ct2_dict[cellid]["sum contact sites"] = len(cs_result_dicts_ct1[ci]["cs ids"])
                only_ct2_dict[cellid]["sum contact sites %s" % ct_dict[ct2]] = len(ct2_cs_ids)
                only_ct2_dict[cellid]["average contact sites %s per cell" % ct_dict[ct2]] = len(ct2_cs_ids)/ len(unique_ct2_cs_cellids)
                only_ct2_dict[cellid]["sum contact sites %s" % ct_dict[ct3]] = len(ct3_cs_ids)
                try:
                    only_ct2_dict[cellid]["average contact sites %s per cell" % ct_dict[ct3]] = len(ct3_cs_ids) / len(
                        unique_ct3_cs_cellids)
                except ZeroDivisionError:
                    only_ct2_dict[cellid]["average contact sites %s per cell" % ct_dict[ct3]] = 0
                only_ct2_dict[cellid]["percentage synapse amount to contact sites"] = (only_ct2_dict[cellid]["synapse amount"] * 100)/ len(ct2_cs_ids)
                only_ct2_dict[cellid]["percentage synapse amount to contact sites %s"  % ct_dict[ct2]] = only_ct2_dict[cellid]["percentage synapse amount to contact sites"]
            elif cellid in only_2ct3_cellids:
                only_ct3_dict[cellid]["sum contact sites"] = len(cs_result_dicts_ct1[ci]["cs ids"])
                only_ct3_dict[cellid]["sum contact sites %s" % ct_dict[ct2]] = len(ct2_cs_ids)
                try:
                    only_ct3_dict[cellid]["average contact sites %s per cell" % ct_dict[ct2]] = len(ct2_cs_ids) / len(
                        unique_ct2_cs_cellids)
                except ZeroDivisionError:
                    only_ct3_dict[cellid]["average contact sites %s per cell" % ct_dict[ct2]] = 0
                only_ct3_dict[cellid]["sum contact sites %s" % ct_dict[ct3]] = len(ct3_cs_ids)
                only_ct3_dict[cellid]["average contact sites %s per cell" % ct_dict[ct3]] = len(ct3_cs_ids) / len(
                    unique_ct3_cs_cellids)
                only_ct3_dict[cellid]["percentage synapse amount to contact sites"] = (only_ct3_dict[cellid][
                                                                                           "synapse amount"] * 100) / len(
                    ct3_cs_ids)
                only_ct3_dict[cellid]["percentage synapse amount to contact sites %s" % ct_dict[ct3]] = \
                only_ct3_dict[cellid]["percentage synapse amount to contact sites"]
            elif cellid in both_cellids:
                both_dict[cellid]["sum contact sites"] = len(cs_result_dicts_ct1[ci]["cs ids"])
                both_dict[cellid]["sum contact sites %s" % ct_dict[ct2]] = len(ct2_cs_ids)
                try:
                    both_dict[cellid]["average contact sites %s per cell" % ct_dict[ct2]] = len(ct2_cs_ids) / len(
                        unique_ct2_cs_cellids)
                except ZeroDivisionError:
                    both_dict[cellid]["average contact sites %s per cell" % ct_dict[ct2]] = 0
                both_dict[cellid]["sum contact sites %s" % ct_dict[ct3]] = len(ct3_cs_ids)
                try:
                    both_dict[cellid]["average contact sites %s per cell" % ct_dict[ct3]] = len(ct3_cs_ids) / len(
                        unique_ct3_cs_cellids)
                except ZeroDivisionError:
                    both_dict[cellid]["average contact sites %s per cell" % ct_dict[ct3]] = 0
                both_dict[cellid]["percentage synapse amount to contact sites"] = (both_dict[cellid][
                                                                                           "synapse amount"] * 100) / (len(
                    ct2_cs_ids) + len(ct3_cs_ids))
                syn_amount_cell_ind = np.where(both_cellids == cellid)
                ct2_syn_amount_cell = sorted_ct2_amounts[syn_amount_cell_ind]
                ct3_syn_amount_cell = sorted_ct3_amounts[syn_amount_cell_ind]
                both_dict[cellid]["percentage synapse amount to contact sites %s" % ct_dict[ct2]] = ((ct2_syn_amount_cell * 100) / len(ct2_cs_ids))[0]
                both_dict[cellid]["percentage synapse amount to contact sites %s" % ct_dict[ct3]] = ((ct3_syn_amount_cell * 100) / len(
                    ct3_cs_ids))[0]
            else:
                not_conn_dict[cellid] = {}
                not_conn_dict[cellid]["sum contact sites"] = len(cs_result_dicts_ct1[ci]["cs ids"])
                not_conn_dict[cellid]["sum contact sites %s" % ct_dict[ct2]] = len(ct2_cs_ids)
                try:
                    not_conn_dict[cellid]["average contact sites %s per cell" % ct_dict[ct2]] = len(ct2_cs_ids) / len(
                        unique_ct2_cs_cellids)
                except ZeroDivisionError:
                    not_conn_dict[cellid]["average contact sites %s per cell" % ct_dict[ct2]] = 0
                not_conn_dict[cellid]["sum contact sites %s" % ct_dict[ct3]] = len(ct3_cs_ids)
                try:
                    not_conn_dict[cellid]["average contact sites %s per cell" % ct_dict[ct3]] = len(ct3_cs_ids) / len(
                        unique_ct3_cs_cellids)
                except ZeroDivisionError:
                    not_conn_dict[cellid]["average contact sites %s per cell" % ct_dict[ct3]] = 0

    time_stamps = [time.time()]
    step_idents = ['t-0']

    log.info(f'Cell numbers: only {ct_dict[ct2]}: {len(only_2ct2_cellids)}, only {ct_dict[ct3]}: {len(only_2ct3_cellids)}, both: {len(both_cellids)}, none: {len(not_connected_ids)}')
    columns = [f'only {ct_dict[ct2]}', f'only {ct_dict[ct3]}', 'both', 'none']
    number_pd = pd.DataFrame(columns = columns, index = range(0))
    number_pd.loc[0, f'only {ct_dict[ct2]}'] = len(only_2ct2_cellids)
    number_pd.loc[0, f'only {ct_dict[ct3]}'] = len(only_2ct3_cellids)
    number_pd.loc[0, 'both'] =len(both_cellids)
    number_pd.loc[0, 'none'] = len(not_connected_ids)
    number_pd.to_csv(f'{f_name}/cell_numbers.csv')


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
        conn_synapses.plot_hist_comparison(key=key, subcell="synapse", result_df=conn_df, hue = 'connection to cts', cells=True, norm_hist = True, bins= 10)
        conn_synapses.plot_hist_comparison(key=key, subcell="synapse", result_df=conn_df, hue = 'connection to cts', cells=True, norm_hist=False, bins=10)

    ranksum_results.to_csv("%s/ranksum_results.csv" % f_name)

    if sd_csssv is not None:
        log.info("Step 4b/4: Compute statistics for contact sites and plot results")
        cs_columnns = ["cellids", "connection to cts", "sum contact sites", "sum contact sites %s" % ct_dict[ct2], "average contact sites %s per cell" % ct_dict[ct2],
                       "sum contact sites %s" % ct_dict[ct3], "average contact sites %s per cell" % ct_dict[ct3], "percentage synapse amount to contact sites",
                       "percentage synapse amount to contact sites %s" % ct_dict[ct2], "percentage synapse amount to contact sites %s" % ct_dict[ct3]]
        conn_df = pd.DataFrame(
            columns=cs_columnns,
            index=len(cellids1))
        conn_df.loc[0:len(only_2ct2_cellids) - 1, "cellids"] = only_2ct2_cellids
        conn_df.loc[0:len(only_2ct2_cellids) - 1, "connection to cts"] = "only %s" % ct_dict[ct2]
        conn_df.loc[0:len(only_2ct2_cellids) - 1, "synapse amount"] = only_ct1ct2_syn_amounts
        conn_df.loc[0:len(only_2ct2_cellids) - 1, "sum size synapses"] = only_ct1ct2_syn_sumsizes
        conn_df.loc[len(only_2ct2_cellids):len(only_2ct2_cellids) + len(only_2ct3_cellids) - 1,
        "cellids"] = only_2ct3_cellids
        conn_df.loc[len(only_2ct2_cellids):len(only_2ct2_cellids) + len(only_2ct3_cellids) - 1,
        "connection to cts"] = "only %s" % ct_dict[ct3]
        conn_df.loc[len(only_2ct2_cellids):len(only_2ct2_cellids) + len(only_2ct3_cellids) - 1,
        "synapse amount"] = only_ct1ct3_syn_amounts
        conn_df.loc[len(only_2ct2_cellids):len(only_2ct2_cellids) + len(only_2ct3_cellids) - 1,
        "sum size synapses"] = only_ct1ct3_syn_sumsizes
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
                avg_syn_amount_partner = only_ct2_dict[cellid]["synapse amount"] / len(
                    only_ct2_dict[cellid]["ct partners"])
                avg_syn_size_partner = only_ct2_dict[cellid]["sum size synapses"] / len(
                    only_ct2_dict[cellid]["ct partners"])
                conn_df.loc[np.where(only_2ct2_cellids == cellid)[0], "amount partners"] = amount_partners
                conn_df.loc[
                    np.where(only_2ct2_cellids == cellid)[0], "avg synapse amount per partner"] = avg_syn_amount_partner
                conn_df.loc[
                    np.where(only_2ct2_cellids == cellid)[0], "avg synapse size per partner"] = avg_syn_size_partner
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
                avg_syn_amount_partner = only_ct3_dict[cellid]["synapse amount"] / len(
                    only_ct3_dict[cellid]["ct partners"])
                avg_syn_size_partner = only_ct3_dict[cellid]["sum size synapses"] / len(
                    only_ct3_dict[cellid]["ct partners"])
                conn_df.loc[np.where(only_2ct3_cellids == cellid)[0] + len(
                    only_2ct2_cellids), "amount partners"] = amount_partners
                conn_df.loc[np.where(only_2ct3_cellids == cellid)[0] + len(
                    only_2ct2_cellids), "avg synapse amount per partner"] = avg_syn_amount_partner
                conn_df.loc[np.where(only_2ct3_cellids == cellid)[0] + len(
                    only_2ct2_cellids), "avg synapse size per partner"] = avg_syn_size_partner
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

        # state type of columns with float entries to prevent plotting issues
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
            "%s/full_%s_no_conn_%s%s_arr_%i.pkl" % (
            f_name_saving, ct_dict[ct1], ct_dict[ct2], ct_dict[ct3], min_comp_len),
            not_connected_ids)

        write_obj2pkl("%s/%s_2_%s_dict.pkl" % (f_name, ct_dict[ct1], ct_dict[ct2]), ct2_arr_dict)
        write_obj2pkl("%s/%s_2_%s_dict.pkl" % (f_name, ct_dict[ct1], ct_dict[ct3]), ct3_arr_dict)
        write_obj2pkl("%s/%s_2_both%s%s_dict.pkl" % (f_name, ct_dict[ct1], ct_dict[ct2], ct_dict[ct3]), both_arr_dict)

        conn_synapses = ComparingMultipleForPLotting(
            ct_list=["only %s" % ct_dict[ct2], "only %s" % ct_dict[ct3], "both"], filename=f_name,
            dictionary_list=[ct2_arr_dict, ct3_arr_dict, both_arr_dict], colour_list=["#EAAE34", '#2F86A8', "#707070"])

        r_columns = ["connected to only %s vs only %s" % (ct_dict[ct2], ct_dict[ct3]),
                     "connected to both vs only %s" % ct_dict[ct2], "connected to both vs only %s" % ct_dict[ct3]]
        ranksum_results = pd.DataFrame(columns=r_columns)
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
            conn_synapses.plot_violin(key=key, x="connection to cts", result_df=conn_df, subcell="synapse",
                                      stripplot=True)
            conn_synapses.plot_box(key=key, x="connection to cts", result_df=conn_df, subcell="synapse",
                                   stripplot=False)
            conn_synapses.plot_hist_comparison(key=key, subcell="synapse", cells=True, norm_hist=True, bins=10)
            conn_synapses.plot_hist_comparison(key=key, subcell="synapse", cells=True, norm_hist=False, bins=10)

        ranksum_results.to_csv("%s/ranksum_results.csv" % f_name)


    return only_2ct2_cellids, only_2ct3_cellids, both_cellids, not_connected_ids

def get_ct_via_inputfraction(sd_synssv, pre_ct, post_cts, pre_cellids, post_cellids, filename, celltype_threshold, pre_label = None, post_labels = None, min_comp_len = 200, min_syn_size = 0.1, syn_prob_thresh = 0.8, compare2mcl = True):
    """
    Get the input fraction of one celltype to one or multiple other celltypes. The fraction of synapse amount and sum of synapses is calculated in relation to
    synaptic amount of sum of synaptic synapses from all input to one cell (only from cells/axons that fulfill minimum comaprtment requirements theirselfes).
    :param sd_synssv: segmentation dataset for synapses.
    :param pre_ct: input celltype
    :param post_cts: list celltypes the input should be compared to
    :param pre_cellids: cellids of input celltype
    :param post_cellids: list of cellids from output celltypes
    :param filename: path to dictionary where results should be saved
    :param celltype_threshold: threshold of which fraction belongs to celltype searched for
    :param pre_label: celllabel if deviating from label in ct_dict e.g. subpopulations
    :param post_labels: list of labels if deviating from label in ct_dict, same length as output_cts
    :param min_comp_len: minimum compartment length for cells to e included in analysis
    :param min_syn_size: minimum synapse size
    :param syn_prob_thresh: threshold for synapse probability
    :param compare2mcl: if True: compare to synapses only from cells with same min comparment length
    :return: cellids for cells that match threshold, result dictionary
    """
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    pre_celldict = load_pkl2obj("/wholebrain/scratch/arother/j0251v4_prep/full_%.3s_dict.pkl" % ct_dict[pre_ct])
    post_celldicts = [load_pkl2obj("/wholebrain/scratch/arother/j0251v4_prep/full_%.3s_dict.pkl" % ct_dict[i]) for i in post_cts]
    post_ct_amount = len(post_cts)
    if pre_label is None:
        pre_label = ct_dict[pre_ct]
    if post_labels is None:
        post_labels = [ct_dict[i] for  i in post_cts]


    filename = "%s/get_ct_via_%s_input_mcl_%i_mss_%i_ct_%i_sp_%i" % (filename, pre_label, min_comp_len, min_syn_size, celltype_threshold, syn_prob_thresh)

    log = initialize_logging('subpopulation grouping per connectivity', log_dir=filename + '/logs/')
    log.info(
        "parameters: input celltype = %s, output celltype1 = %s, output celltype2 = %s, amount of output celltypes = %i, min_comp_length = %.i, min_syn_size = %.2f, syn_prob_thresh = %.2f" %
        (pre_label, post_labels[0], post_labels[1], post_ct_amount, min_comp_len, min_syn_size, syn_prob_thresh))
    time_stamps = [time.time()]
    step_idents = ['t-0']

    # check if cellids match minimum compartment length
    log.info("Step 1/5: Check compartment length of cells from %i celltypes" % (post_ct_amount + 1))
    if compare2mcl:
        pre_cellids = check_comp_lengths_ct(pre_cellids, fullcelldict=pre_celldict, min_comp_len=min_comp_len)
    post_cellids = [check_comp_lengths_ct(post_cellids[i], fullcelldict = post_celldicts[i], min_comp_len = min_comp_len) for i in range(post_ct_amount)]
    post_lengths = np.array([len(post_cellids[i]) for i in range(post_ct_amount)])
    max_post_length = np.max(post_lengths)
    post_cellids_2D = np.zeros((post_ct_amount, max_post_length))
    for i in range(post_ct_amount):
        post_cellids_2D[i][0:post_lengths[i]] = post_cellids[i]
    flattened_post_cellids = np.concatenate(post_cellids).astype(int)
    time_stamps = [time.time()]
    step_idents = ['check compartment length from all %i celltypes' % (post_ct_amount + 1)]


    log.info("Step 2/5: Prefilter synapses for synapses between these celltypes")
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness = filter_synapse_caches_for_ct(sd_synssv,
                                                                                           pre_cts=[pre_ct],
                                                                                           post_cts=post_cts,
                                                                                           syn_prob_thresh=syn_prob_thresh,
                                                                                           min_syn_size=min_syn_size,
                                                                                           axo_den_so=True)

    time_stamps = [time.time()]
    step_idents = ['prefilter synapses done']

    log.info("Step 3/5: Get amount and sum size from %s for each cell" % pre_label)

    #for each cell: determine synapse amount and sum of synapse sizes per cell from input celltype
    if compare2mcl:
        from_ct1_syn_dict = synapse_amount_sumsize_between2cts(celltype1=pre_ct, cellids1=pre_cellids,
                                                               cellids2=flattened_post_cellids,
                                                               syn_ids=m_ids, syn_cts=m_cts,
                                                               syn_ssv_partners=m_ssv_partners,
                                                               syn_sizes=m_sizes, syn_axs=m_axs,
                                                               seperate_soma_dens=False, fragments_pre = False)
    else:
        from_ct1_syn_dict = synapse_amount_sumsize_between2cts(celltype1=pre_ct, cellids1=pre_cellids,
                                                               cellids2=flattened_post_cellids,
                                                               syn_ids=m_ids, syn_cts=m_cts,
                                                               syn_ssv_partners=m_ssv_partners,
                                                               syn_sizes=m_sizes, syn_axs=m_axs,
                                                               seperate_soma_dens=False, fragments_pre=True)

    time_stamps = [time.time()]
    step_idents = ['getting %s amount and summed size per cell done' % pre_label]

    #get total synapse amount and synapse sum size per cell for min_comp_len
    log.info("Step 4/5: Calculate fraction of input for each cell")
    synapse_amount_fraction = np.zeros(len(flattened_post_cellids))
    synapse_sumsize_fraction = np.zeros(len(flattened_post_cellids))
    celltypes = np.empty(len(flattened_post_cellids)).astype(str)
    fraction_dict = {i: {} for i in post_labels}
    if not ("dendrite synapse amount %i" % min_comp_len) in post_celldicts[post_cts[0]][flattened_post_cellids[0]]:
        #if it is not calculated, yet, calculate as in analysis prep
        raise KeyError("no synapse amount and summed size calculated for this compartment length requirement")
    for i, cellid in enumerate(tqdm(flattened_post_cellids)):
        try:
            msn_syn_amount = from_ct1_syn_dict[cellid]["amount"]
            msn_summed_syn_size = from_ct1_syn_dict[cellid]["summed size"]
        except KeyError:
            msn_syn_amount = 0
            msn_summed_syn_size = 0
        celltype = int(np.where(post_cellids_2D == cellid)[0])
        if compare2mcl:
            overall_synapse_amount = post_celldicts[celltype][cellid]["dendrite synapse amount %i" % min_comp_len] + post_celldicts[celltype][cellid]["soma synapse amount %i" % min_comp_len]
            overall_summed_synsize = post_celldicts[celltype][cellid]["dendrite summed synapse size %i" % min_comp_len] + post_celldicts[celltype][cellid]["soma summed synapse size %i" % min_comp_len]
        else:
            overall_synapse_amount = post_celldicts[celltype][cellid]["dendrite synapse amount"] + \
                                     post_celldicts[celltype][cellid]["soma synapse amount"]
            overall_summed_synsize = post_celldicts[celltype][cellid]["dendrite summed synapse size"] + post_celldicts[celltype][cellid]["soma summed synapse size"]
        if overall_synapse_amount > 0:
            synapse_amount_fraction[i] = msn_syn_amount / overall_synapse_amount
            synapse_sumsize_fraction[i] = msn_summed_syn_size/overall_summed_synsize
        celltypes[i] = post_labels[celltype]
        fraction_dict[post_labels[celltype]][cellid] = {"synapse amount fraction": synapse_amount_fraction[i], "synapse summed size fraction": synapse_sumsize_fraction[i]}

    write_obj2pkl("%s/fraction_dict.pkl" % filename, fraction_dict)
    results_dict = {"synapse amount fraction": synapse_amount_fraction,
                    "synapse summed size fraction": synapse_sumsize_fraction, "cellids": flattened_post_cellids, "predicted celltype": celltypes}
    pd_results = pd.DataFrame(results_dict)
    pd_results.to_csv("%s/results.csv" % filename)
    high_input_inds = synapse_sumsize_fraction >= celltype_threshold
    high_input_cellids = flattened_post_cellids[high_input_inds]

    time_stamps = [time.time()]
    step_idents = ['input fraction for each cell calculated']
    log.info("%i cells over celltype threshold %i" % (len(high_input_cellids), celltype_threshold))


    log.info("Step 5/5: Plot results in histogram")
    result_plots = ResultsForPlotting(celltype = post_labels[0], filename = filename, dictionary = results_dict)
    for key in results_dict:
        if "cellids" in key or "celltype" in key:
            continue
        result_plots.plot_hist(key = key, subcell = "synapse", cells = True, norm_hist = False, bins = 20, xlabel = None, celltype2 = pre_label, outgoing = False)
        result_plots.plot_hist(key=key, subcell="synapse", cells=True, norm_hist=True, bins=20, xlabel=None,
                               celltype2=pre_label, outgoing=False)

    time_stamps = [time.time()]
    step_idents = ['results plotted, analysis finished']
    log.info("Finding celltype based on input fraction of %s done" % pre_label)


    return  high_input_cellids, results_dict






