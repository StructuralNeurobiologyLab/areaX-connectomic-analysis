
from syconn import global_params
from syconn.reps.super_segmentation import SuperSegmentationDataset
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os as os
import scipy
import time
from syconn.handler.config import initialize_logging
from syconn.handler.basics import load_pkl2obj
from tqdm import tqdm
from syconn.handler.basics import write_obj2pkl
from scipy.stats import ranksums
from wholebrain.scratch.arother.bio_analysis.general.result_helper import ResultsForPlotting, ComparingResultsForPLotting
from wholebrain.scratch.arother.bio_analysis.general.analysis_helper import get_compartment_length, get_compartment_bbvolume, \
    get_compartment_radii, get_compartment_tortuosity_complete, get_compartment_tortuosity_sampled
global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"

ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

def comp_aroborization(sso, compartment, cell_graph, min_comp_len = 100, full_cell_dict = None):
    """
    calculates bounding box from min and max dendrite values in each direction to get a volume estimation per compartment.
    :param sso: cell
    :param compartment: 0 = dendrite, 1 = axon, 2 = soma
    :param cell_graph: sso.weighted graph
    :param min_comp_len: minimum compartment length, if not return 0
    :param full_cell_dict: dictionary holding per cell parameters for lookup, cell.id is key
    :return: comp_len, comp_volume in µm³
    """
    # use full cell dict to lookup versions
    comp_dict = {1: "axon", 0: "dendrite"}
    if full_cell_dict is not None:
        comp_length = full_cell_dict[sso.id][comp_dict[compartment] + " length"]
    else:
        comp_length = get_compartment_length(sso, compartment, cell_graph)
    if comp_length < min_comp_len:
        return 0, 0, 0, 0, 0
    comp_inds = np.nonzero(sso.skeleton["axoness_avg10000"] == compartment)[0]
    comp_nodes = sso.skeleton["nodes"][comp_inds] * sso.scaling
    comp_volume = get_compartment_bbvolume(comp_nodes)
    comp_radii = get_compartment_radii(sso, comp_inds)
    median_comp_radius = np.median(comp_radii)
    tortuosity_complete = get_compartment_tortuosity_complete(comp_length, comp_nodes)
    tortosity_sampled = get_compartment_tortuosity_sampled(cell_graph, comp_nodes)
    return comp_length, comp_volume, median_comp_radius, tortuosity_complete, tortosity_sampled

def axon_dendritic_arborization_cell(sso, min_comp_len = 100, full_cell_dict = None):
    '''
    analysis the spatial distribution of the axonal/dendritic arborization per cell if they fulfill the minimum compartment length.
    To estimate the volume a dendritic arborization spans, the bounding box around the axon or dendrite is estimated by its min and max values in each direction.
    Uses comp_arborization.
    :param min_comp_len: minimum compartment length of axon and dendrite
    :param full_cell_dict: dictionary with cached values. cell.id is key
    :return: overall axonal/dendritic length [µm], axonal/dendritic volume [µm³]
    '''
    sso.load_skeleton()
    g = sso.weighted_graph(add_node_attr=('axoness_avg10000',))
    if full_cell_dict is not None:
        axon_length, axon_volume, ax_median_radius, ax_tortuosity_complete, ax_tortuosity_sampled = comp_aroborization(
            sso, compartment=1, cell_graph=g, min_comp_len=min_comp_len, full_cell_dict = full_cell_dict)
    else:
        axon_length, axon_volume, ax_median_radius, ax_tortuosity_complete, ax_tortuosity_sampled = comp_aroborization(sso, compartment=1, cell_graph=g, min_comp_len = min_comp_len)
    if axon_length == 0:
        return 0, 0
    if full_cell_dict is not None:
        dendrite_length, dendrite_volume, dendrite_median_radius, dendrite_tortuosity_complete, dendrite_tortuosity_sampled = comp_aroborization(sso, compartment=0, cell_graph=g,
                                                                              min_comp_len=min_comp_len,full_cell_dict= full_cell_dict)
    else:
        dendrite_length, dendrite_volume, dendrite_median_radius, dendrite_tortuosity_complete, dendrite_tortuosity_sampled = comp_aroborization(
            sso, compartment=0, cell_graph=g,
            min_comp_len=min_comp_len)
    if dendrite_length == 0:
        return 0, 0
    ax_dict = {"length": axon_length, "volume": axon_volume, "median radius": ax_median_radius, "tortuosity complete": ax_tortuosity_complete, "tortuosity sampled": ax_tortuosity_sampled}
    den_dict = {"length": dendrite_length, "volume": dendrite_volume, "median radius": dendrite_median_radius, "tortuosity complete": dendrite_tortuosity_complete, "tortuosity sampled": dendrite_tortuosity_sampled}
    return ax_dict, den_dict

def axon_den_arborization_ct(ssd, celltype, filename, min_comp_len = 100, full_cells = True, handpicked = True, percentile = None):
    '''
    estimate the axonal and dendritic aroborization by celltype. Uses axon_dendritic_arborization to get the aoxnal/dendritic bounding box volume per cell
    via comp_arborization. Plots the volume per compartment and the overall length as histograms.
    :param ssd: super segmentation dataset
    :param celltype: j0256: STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7,
#                      FS=8, LTS=9, NGF=10
    :param min_comp_len: minimum compartment length in µm
    :param full_cells: loads preprocessed cells that have axon, soma and dendrite
    :param handpicked: loads cells that were manually checked
    :param if percentile given, percentile of the cell population can be compared, if preprocessed, in case of 50 have to give either 49 or 51
    :return: f_name: foldername in which results are stored
    '''

    start = time.time()
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    if full_cells:
        full_cell_dict = load_pkl2obj(
                "/wholebrain/scratch/arother/j0251v4_prep/full_%.3s_dict.pkl" % ct_dict[celltype])
    if percentile is not None:
        if percentile == 50:
            raise ValueError("Due to ambiguity, value has to be either 49 or 51")
        else:
            ct_dict[celltype] = ct_dict[celltype] + " p%.2i" % percentile
    f_name = "%s/%s_comp_volume_mcl%i" % (filename, ct_dict[celltype], min_comp_len)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('compartment volume estimation', log_dir=f_name + '/logs/')
    log.info("parameters: celltype = %s, min_comp_length = %.i" % (ct_dict[celltype], min_comp_len))
    time_stamps = [time.time()]
    step_idents = ['t-0']
    if full_cells:
        if handpicked:
            try:
                cellids = load_pkl2obj(
                    "/wholebrain/scratch/arother/j0251v4_prep/handpicked_%s_arr_c%i.pkl" % (ct_dict[celltype], min_comp_len))
            except FileNotFoundError:
                cellids = load_pkl2obj(
                    "/wholebrain/scratch/arother/j0251v4_prep/handpicked_%s_arr.pkl" % ct_dict[celltype])
        else:
            try:
                cellids = load_pkl2obj("/wholebrain/scratch/arother/j0251v4_prep/full_%s_arr_c%i.pkl" % (ct_dict[celltype], min_comp_len))
            except FileNotFoundError:
                cellids = load_pkl2obj("/wholebrain/scratch/arother/j0251v4_prep/full_%s_arr.pkl" % ct_dict[celltype])
    else:
        if percentile is not None:
            raise ValueError("percentiles can only be used on preprocessed cellids")
        else:
            cellids = ssd.ssv_ids[ssd.load_numpy_data("celltype_cnn_e3") == celltype]
    log.info('Step 1/2 calculating volume estimate for axon/dendrite per cell')
    axon_length_ct = np.zeros(len(cellids))
    dendrite_length_ct = np.zeros(len(cellids))
    axon_vol_ct = np.zeros(len(cellids))
    dendrite_vol_ct = np.zeros(len(cellids))
    axon_med_radius_ct = np.zeros(len(cellids))
    dendrite_med_radius_ct = np.zeros(len(cellids))
    axon_tortuosity_complete_ct = np.zeros(len(cellids))
    dendrite_tortuosity_complete_ct = np.zeros(len(cellids))
    axon_tortuosity_sampled_ct = np.zeros(len(cellids))
    dendrite_tortuosity_sampled_ct = np.zeros(len(cellids))
    if full_cells:
        soma_centres = np.zeros((len(cellids), 3))
    for i, cell in enumerate(tqdm(ssd.get_super_segmentation_object(cellids))):
        axon_dict, dendrite_dict = axon_dendritic_arborization_cell(cell, min_comp_len = min_comp_len, full_cell_dict= full_cell_dict,)
        if type(axon_dict) == int:
            continue
        axon_length_ct[i] = axon_dict["length"]
        dendrite_length_ct[i] = dendrite_dict["length"]
        axon_vol_ct[i] = axon_dict["volume"]
        dendrite_vol_ct[i] = dendrite_dict["volume"]
        axon_med_radius_ct[i] = axon_dict["median radius"]
        dendrite_med_radius_ct[i] = dendrite_dict["median radius"]
        axon_tortuosity_complete_ct[i] = axon_dict["tortuosity complete"]
        dendrite_tortuosity_complete_ct[i] = dendrite_dict["tortuosity complete"]
        axon_tortuosity_sampled_ct[i] = axon_dict["tortuosity sampled"]
        dendrite_tortuosity_sampled_ct[i] = dendrite_dict["tortuosity sampled"]
        if full_cells:
            soma_centres[i] = full_cell_dict[cell.id]["soma centre"]

    celltime = time.time() - start
    print("%.2f sec for iterating through cells" % celltime)
    time_stamps.append(time.time())
    step_idents.append('calculating bounding box volume per cell')

    log.info('Step 2/2 processing and plotting ct arrays')
    nonzero_inds = axon_length_ct > 0
    axon_length_ct = axon_length_ct[nonzero_inds]
    dendrite_length_ct = dendrite_length_ct[nonzero_inds]
    axon_vol_ct = axon_vol_ct[nonzero_inds]
    dendrite_vol_ct = dendrite_vol_ct[nonzero_inds]
    axon_med_radius_ct = axon_med_radius_ct[nonzero_inds]
    dendrite_med_radius_ct = dendrite_med_radius_ct[nonzero_inds]
    axon_tortuosity_complete_ct = axon_tortuosity_complete_ct[nonzero_inds]
    dendrite_tortuosity_complete_ct = dendrite_tortuosity_complete_ct[nonzero_inds]
    axon_tortuosity_sampled_ct = axon_tortuosity_sampled_ct[nonzero_inds]
    dendrite_tortuosity_sampled_ct = dendrite_tortuosity_sampled_ct[nonzero_inds]
    cellids = cellids[nonzero_inds]
    ds_size = [256, 256, 394] #size of whole dataset
    ds_vol = np.prod(ds_size)
    axon_vol_perc = axon_vol_ct/ds_vol * 100
    dendrite_vol_perc = dendrite_vol_ct/ds_vol * 100



    if full_cells:
        soma_centres = soma_centres[nonzero_inds]
        distances_between_soma = scipy.spatial.distance.cdist(soma_centres, soma_centres, metric = "euclidean") / 1000
        distances_between_soma = distances_between_soma[distances_between_soma > 0].reshape(len(cellids), len(cellids) - 1)
        avg_soma_distance_per_cell = np.mean(distances_between_soma, axis=1)
        pairwise_soma_distances = scipy.spatial.distance.pdist(soma_centres, metric = "euclidean") / 1000
        ct_vol_comp_dict = {"cell ids": cellids,"axon length": axon_length_ct, "dendrite length": dendrite_length_ct,
                            "axon volume bb": axon_vol_ct, "dendrite volume bb": dendrite_vol_ct,
                            "axon volume percentage": axon_vol_perc, "dendrite volume percentage": dendrite_vol_perc,
                            "mean soma distance": avg_soma_distance_per_cell, "axon median radius": axon_med_radius_ct,
                            "dendrite median radius": dendrite_med_radius_ct, "axon tortuosity complete": axon_tortuosity_complete_ct,
                            "dendrite tortuosity complete": dendrite_tortuosity_complete_ct, "axon tortuosity sampled": axon_tortuosity_sampled_ct,
                            "dendrite tortuosity sampled": dendrite_tortuosity_sampled_ct}
    else:
        ct_vol_comp_dict = {"cell ids": cellids, "axon length": axon_length_ct,
                            "dendrite length": dendrite_length_ct,
                            "axon volume bb": axon_vol_ct, "dendrite volume bb": dendrite_vol_ct,
                            "axon volume percentage": axon_vol_perc,
                            "dendrite volume percentage": dendrite_vol_perc, "axon median radius": axon_med_radius_ct,
                            "dendrite median radius": dendrite_med_radius_ct, "axon tortuosity complete": axon_tortuosity_complete_ct,
                            "dendrite tortuosity complete": dendrite_tortuosity_complete_ct, "axon tortuosity sampled": axon_tortuosity_sampled_ct,
                            "dendrite tortuosity sampled": dendrite_tortuosity_sampled_ct}
    vol_comp_pd = pd.DataFrame(ct_vol_comp_dict)
    vol_comp_pd.to_csv("%s/ct_vol_comp.csv" % f_name)

    #write soma centre coords and pairwise distances into dictionary but not dataframe (different length than other parameters
    if full_cells:
        ct_vol_comp_dict["soma centre coords"] = soma_centres
        ct_vol_comp_dict["pairwise soma distance"] = pairwise_soma_distances

    vol_result_dict = ResultsForPlotting(celltype = ct_dict[celltype], filename = f_name, dictionary = ct_vol_comp_dict)

    for key in ct_vol_comp_dict.keys():
        if "ids" in key:
            continue
        if "axon" in key:
            vol_result_dict.plot_hist(key=key, subcell="axon")
        elif "dendrite" in key:
            vol_result_dict.plot_hist(key = key, subcell="dendrite")
        elif "soma distance" in key:
            if "pairwise" in key:
                vol_result_dict.plot_hist(key=key, subcell="soma", cells=False)
            else:
                vol_result_dict.plot_hist(key= key, subcell="soma")

    write_obj2pkl("%s/ct_vol_comp.pkl" % f_name, ct_vol_comp_dict)

    plottime = time.time() - celltime
    print("%.2f sec for plotting" % plottime)
    time_stamps.append(time.time())
    step_idents.append('processing arrays per celltype, plotting')
    log.info("compartment volume estimation per celltype finished")

    return f_name

def compare_compartment_volume_ct(celltype1, filename, celltype2= None, percentile = None, filename1 = None, filename2 = None, min_comp_len = 100):
    '''
    compares estimated compartment volumes (by bounding box) between two celltypes that have been generated by axon_den_arborization_ct.
    Data will be compared in histogram and violinplots. P-Values are computed by ranksum test.
    :param celltype1: j0256: STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7,
#                      FS=8, LTS=9, NGF=10
    :param celltype2: compared against celltype 2,not needed if percentiles are compared
    :param percentile: if percentile given not two celltypes but two different populations within a celltye will be compared, has to be either 49 or 51 not 50
    :param filename1, filename2: only if data preprocessed: filename were preprocessed data is stored
    :param min_comp_len: minimum comparment length used by analysis
    :return:
    '''
    start = time.time()
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    if percentile is None and celltype2 is None:
        raise ValueError("either celltypes or percentiles must be compared")
    ct1_str = ct_dict[celltype1]
    if celltype2 is not None:
        ct2_str = ct_dict[celltype2]
    if percentile is not None:
        if percentile == 50:
            raise ValueError("Due to ambiguity, value has to be either 49 or 51")
        else:
            ct1_str = ct_dict[celltype1] + " p%.2i" % percentile
            ct2_str = ct_dict[celltype1] + " p%.2i" % (100 - percentile)
    f_name = "%s/comp_compartment_%s_%s_comp_volume_mcl%i" % (
        filename, ct1_str,ct2_str, min_comp_len)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('compare compartment volumes between two celltypes', log_dir=f_name + '/logs/')
    log.info("parameters: celltype1 = %s,celltype2 = %s min_comp_length = %.i" % (ct1_str, ct2_str, min_comp_len))
    time_stamps = [time.time()]
    step_idents = ['t-0']
    ct1_comp_dict = load_pkl2obj("%s/ct_vol_comp.pkl"% filename1)
    ct2_comp_dict = load_pkl2obj("%s/ct_vol_comp.pkl"% filename2)
    comp_dict_keys = list(ct1_comp_dict.keys())
    if "soma centre coords" in comp_dict_keys:
        log.info("compute mean soma distance between %s and %s" % (ct1_str, ct2_str))
        ct1_soma_coords = ct1_comp_dict["soma centre coords"]
        ct2_soma_coords = ct2_comp_dict["soma centre coords"]
        ct1_distances2ct2 = scipy.spatial.distance.cdist(ct1_soma_coords, ct2_soma_coords, metric="euclidean") / 1000
        ct1avg_soma_distance2ct2_per_cell = np.mean(ct1_distances2ct2, axis=1)
        ct2_distances2ct1 = scipy.spatial.distance.cdist(ct2_soma_coords, ct1_soma_coords,
                                                         metric="euclidean") / 1000
        ct2avg_soma_distance2ct1_per_cell = np.mean(ct2_distances2ct1, axis=1)
        ct_soma_coords = np.concatenate((ct1_soma_coords, ct2_soma_coords))
        pairwise_distances_cts = scipy.spatial.distance.pdist(ct_soma_coords, metric = "euclidean") / 1000
        ct1_comp_dict["avg soma distance to other celltype"] = ct1avg_soma_distance2ct2_per_cell
        ct2_comp_dict["avg soma distance to other celltype"] = ct2avg_soma_distance2ct1_per_cell
        ct1_comp_dict["pairwise soma distance to other celltype"] = pairwise_distances_cts
        ct2_comp_dict["pairwise_soma distance to other celltype"] = pairwise_distances_cts
        ct1_comp_dict.pop("soma centre coords")
        ct2_comp_dict.pop("soma centre coords")
        comp_dict_keys = list(ct1_comp_dict.keys())
    log.info("compute statistics for comparison, create violinplot and histogram")
    ranksum_results = pd.DataFrame(columns=comp_dict_keys[1:], index=["stats", "p value"])
    if percentile is not None:
        results_comparision = ComparingResultsForPLotting(celltype1=ct1_str,
                                                          celltype2=ct2_str, filename=f_name,
                                                          dictionary1=ct1_comp_dict, dictionary2=ct2_comp_dict,
                                                          color1="gray", color2="darkturquoise")
    else:
        results_comparision = ComparingResultsForPLotting(celltype1 = ct1_str, celltype2 = ct2_str, filename = f_name, dictionary1 = ct1_comp_dict, dictionary2 = ct2_comp_dict, color1 = "mediumorchid", color2 = "springgreen")
    for key in ct1_comp_dict.keys():
        if "ids" in key or ("pairwise" in key and "other" in key):
            continue
        #calculate p_value for parameter
        stats, p_value = ranksums(ct1_comp_dict[key], ct2_comp_dict[key])
        ranksum_results.loc["stats", key] = stats
        ranksum_results.loc["p value", key] = p_value
        #plot parameter as violinplot
        if "axon" in key:
            subcell = "axon"
        elif "dendrite" in key:
            subcell = "dendrite"
        else:
            subcell = "soma"
        if "pairwise" in key:
            column_labels = ["distances within %s" % ct1_str, "distances within %s" % ct2_str, "distances between %s and %s" % (ct1_str, ct2_str)]
            results_for_plotting = results_comparision.result_df_per_param(key, key2 = "pairwise soma distance to other celltype", column_labels= column_labels)
            s1, p1 = ranksums(ct1_comp_dict[key], ct1_comp_dict["pairwise soma distance to other celltype"])
            s2, p2 = ranksums(ct2_comp_dict[key], ct1_comp_dict["pairwise soma distance to other celltype"])
            ranksum_results.loc["stats", "pairwise among %s to mixed" % ct1_str] = s1
            ranksum_results.loc["p value", "pairwise among %s to mixed" % ct1_str] = p1
            ranksum_results.loc["stats", "pairwise among %s to mixed" % ct2_str] = s2
            ranksum_results.loc["p value", "pairwise among %s to mixed" % ct2_str] = p2
            ptitle = "pairwise soma distances within and between %s and %s" % (ct1_str, ct2_str)
            results_comparision.plot_hist_comparison(key, subcell, add_key = "pairwise soma distance to other celltype", cells=False, title=ptitle, norm_hist=False)
            results_comparision.plot_hist_comparison(key, subcell, add_key="pairwise soma distance to other celltype",
                                                     cells=False, title=ptitle, norm_hist=True)
        else:
            results_for_plotting = results_comparision.result_df_per_param(key)
            results_comparision.plot_hist_comparison(key, subcell, bins=10, norm_hist=False)
            results_comparision.plot_hist_comparison(key, subcell, bins=10, norm_hist=True)
            results_comparision.plot_violin(key, results_for_plotting, subcell, stripplot=True)
            results_comparision.plot_box(key, results_for_plotting, subcell, stripplot=False)


    ranksum_results.to_csv("%s/ranksum_%s_%s.csv" % (f_name,ct1_str,ct2_str))

    plottime = time.time() - start
    print("%.2f sec for statistics and plotting" % plottime)
    time_stamps.append(time.time())
    step_idents.append('comparing celltypes')
    log.info("compartment volume comparison finished")
