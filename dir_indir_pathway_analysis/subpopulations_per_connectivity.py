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

def sort_by_connectivity(sd_synssv, ct1, ct2, ct3, cellids1, cellids2, cellids3, full_celldict1, full_celldict2, full_celldict3, f_name, f_name_saving = None, min_comp_len = 200, syn_prob_thresh = 0.8, min_syn_size = 0.1):
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
    if f_name_saving is None:
        f_name_saving = f_name
    log = initialize_logging('subpopulation goruping per connectivity', log_dir=f_name + '/logs/')
    log.info(
        "parameters: celltype1 = %s, celltype2 = %s, celltype3 = %s, min_comp_length = %.i, min_syn_size = %.2f, syn_prob_thresh = %.2f" %
        (ct_dict[ct1], ct_dict[ct2], ct_dict[ct3], min_comp_len, min_syn_size, syn_prob_thresh))
    time_stamps = [time.time()]
    step_idents = ['t-0']

    log.info("Step 1/X Check compartment length of cells from three celltypes")
    cellids1 = check_comp_lengths_ct(cellids1, fullcelldict=full_celldict1, min_comp_len=min_comp_len)
    cellids2 = check_comp_lengths_ct(cellids2, fullcelldict=full_celldict2, min_comp_len=min_comp_len)
    cellids3 = check_comp_lengths_ct(cellids3, fullcelldict=full_celldict3, min_comp_len=min_comp_len)

    log.info("Step 2/X Prefilter synapses for synapses between these celltypes")
    m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness = filter_synapse_caches_for_ct(sd_synssv,
                                                                                           pre_cts=[ct1], post_cts = [ct2, ct3],
                                                                                           syn_prob_thresh=syn_prob_thresh,
                                                                                           min_syn_size=min_syn_size,
                                                                                           axo_den_so=True)

#take cached synapse arrays and sort them into four groups
# save cellids per group as array
# save table and dictionary which GPs they connect to, their amount and sum of synapses
# also save compartment

