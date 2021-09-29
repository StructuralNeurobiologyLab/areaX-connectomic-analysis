if __name__ == '__main__':
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
    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

    def comp_aroborization(sso, compartment, cell_graph, min_comp_len = 100):
        """
        calculates bounding box from min and max dendrite values in each direction to get a volume estimation per compartment.
        :param sso: cell
        :param compartment: 0 = dendrite, 1 = axon, 2 = soma
        :param cell_graph: sso.weighted graph
        :param min_comp_len: minimum compartment length, if not return 0
        :return: comp_len, comp_volume in µm³
        """
        non_comp_inds = np.nonzero(sso.skeleton["axoness_avg10000"] != compartment)[0]
        comp_graph = cell_graph.copy()
        comp_graph.remove_nodes_from(non_comp_inds)
        comp_length = comp_graph.size(weight="weight") / 1000  # in µm
        if comp_length < min_comp_len:
            return 0, 0
        comp_inds = np.nonzero(sso.skeleton["axoness_avg10000"] == compartment)[0]
        comp_nodes = sso.skeleton["nodes"][comp_inds] * sso.scaling
        min_x = np.min(comp_nodes[:,0])
        max_x = np.max(comp_nodes[:, 0])
        min_y = np.min(comp_nodes[:, 1])
        max_y = np.max(comp_nodes[:, 1])
        min_z = np.min(comp_nodes[:, 2])
        max_z = np.max(comp_nodes[:, 2])
        comp_volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z) * 10**(-9) #in µm
        return comp_length, comp_volume

    def axon_dendritic_arborization_cell(sso, min_comp_len = 100):
        '''
        analysis the spatial distribution of the axonal/dendritic arborization per cell if they fulfill the minimum compartment length.
        To estimate the volume a dendritic arborization spans, the bounding box around the axon or dendrite is estimated by its min and max values in each direction.
        Uses comp_arborization.
        :param min_comp_len: minimum compartment length of axon and dendrite
        :return: overall axonal/dendritic length [µm], axonal/dendritic volume [µm³]
        '''
        sso.load_skeleton()
        g = sso.weighted_graph(add_node_attr=('axoness_avg10000',))
        axon_length, axon_volume = comp_aroborization(sso, compartment=1, cell_graph=g, min_comp_len = min_comp_len)
        if axon_length == 0:
            return 0,0, 0, 0
        dendrite_length, dendrite_volume = comp_aroborization(sso, compartment=0, cell_graph=g,
                                                                              min_comp_len=min_comp_len)
        if dendrite_length == 0:
            return 0, 0, 0, 0
        return axon_length, axon_volume, dendrite_length, dendrite_volume

    def axon_den_arborization_ct(ssd, celltype, min_comp_len = 100, full_cells = True, handpicked = True, percentile = 0, low_percentile = False):
        '''
        estimate the axonal and dendritic aroborization by celltype. Uses axon_dendritic_arborization to get the aoxnal/dendritic bounding box volume per cell
        via comp_arborization. Plots the volume per compartment and the overall length as histograms.
        :param ssd: super segmentation dataset
        :param celltype: j0256: STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7,
    #                      FS=8, LTS=9, NGF=10
        :param min_comp_len: minimum compartment length in µm
        :param full_cells: loads preprocessed cells that have axon, soma and dendrite
        :param handpicked: loads cells that were manually checked
        :param if percentile given, percentile of the cell population can be compared, if preprocessed
        :param: low_percentile: True if percentile of cell population analysed is in lower half.
        :return:
        '''

        start = time.time()
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   10: "NGF"}
        if percentile != None:
            if low_percentile:
                f_name = "u/arother/bio_analysis_results/dir_indir_pathway_analysis/210929_j0251v3_%s_comp_volume_mcl%i_p%il" % (
                ct_dict[celltype], min_comp_len, percentile)
            else:
                f_name = "u/arother/bio_analysis_results/dir_indir_pathway_analysis/210929_j0251v3_%s_comp_volume_mcl%i_p%ih" % (
                    ct_dict[celltype], min_comp_len, percentile)
        else:
            f_name = "u/arother/bio_analysis_results/dir_indir_pathway_analysis/210929_j0251v3_%s_comp_volume_mcl%i" % (ct_dict[celltype], min_comp_len)
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('compartment volume estimation', log_dir=f_name + '/logs/')
        log.info("parameters: celltype = %s, min_comp_length = %.i" % (ct_dict[celltype], min_comp_len))
        time_stamps = [time.time()]
        step_idents = ['t-0']
        if full_cells:
            soma_centre_dict = load_pkl2obj(
                    "/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_dict.pkl" % ct_dict[celltype])

            if handpicked:
                cellids = load_pkl2obj(
                    "/wholebrain/scratch/arother/j0251v3_prep/handpicked_%.3s_arr.pkl" % ct_dict[celltype])
            if percentile != 0:
                if low_percentile:
                    cellids = load_pkl2obj(
                    "/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_arr_%i_l.pkl" % (ct_dict[celltype], percentile))
                else:
                    cellids = load_pkl2obj(
                        "/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_arr_%i_h.pkl" % (ct_dict[celltype],
                        percentile))
            else:
                cellids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_arr.pkl" % ct_dict[celltype])
        else:
            cellids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == celltype]
        log.info('Step 1/2 calculating volume estimate for axon/dendrite per cell')
        axon_length_ct = np.zeros(len(cellids))
        dendrite_length_ct = np.zeros(len(cellids))
        axon_vol_ct = np.zeros(len(cellids))
        dendrite_vol_ct = np.zeros(len(cellids))
        if full_cells:
            soma_centres = np.zeros((len(cellids), 3))
        for i, cell in enumerate(tqdm(ssd.get_super_segmentation_object(cellids))):
            axon_length, axon_volume, dendrite_length, dendrite_volume = axon_dendritic_arborization_cell(cell, min_comp_len = min_comp_len)
            if axon_length == 0:
                continue
            axon_length_ct[i] = axon_length
            dendrite_length_ct[i] = dendrite_length
            axon_vol_ct[i] = axon_volume
            dendrite_vol_ct[i] = dendrite_volume
            if full_cells:
                soma_centres[i] = soma_centre_dict[cell.id]

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
            ct_vol_comp_dict = {"cell ids": cellids,"axon length": axon_length_ct, "dendrite length": dendrite_length_ct,
                                "axon volume bb": axon_vol_ct, "dendrite volume bb": dendrite_vol_ct,
                                "axon volume percentage": axon_vol_perc, "dendrite volume percentage": dendrite_vol_perc,
                                "mean soma distance": avg_soma_distance_per_cell}
        else:
            ct_vol_comp_dict = {"cell ids": cellids, "axon length": axon_length_ct,
                                "dendrite length": dendrite_length_ct,
                                "axon volume bb": axon_vol_ct, "dendrite volume bb": dendrite_vol_ct,
                                "axon volume percentage": axon_vol_perc,
                                "dendrite volume percentage": dendrite_vol_perc}
        vol_comp_pd = pd.DataFrame(ct_vol_comp_dict)
        vol_comp_pd.to_csv("%s/ct_vol_comp.csv" % f_name)


        for key in ct_vol_comp_dict.keys():
            if "ids" in key:
                continue
            sns.distplot(ct_vol_comp_dict[key], hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "steelblue"},
                         kde=False)
            plt.ylabel("count of cells")
            if "length" in key:
                plt.xlabel("pathlength in µm")
            elif "vol" in key:
                if "percentage" in key:
                    plt.xlabel("% of whole dataset")
                else:
                    plt.xlabel("volume in µm³")
            else:
                plt.xlabel("distance in µm")
            plt.title("%s" % key)
            plt.savefig("%s/%s.png" % (f_name, key))
            plt.close()

        if full_cells:
            ct_vol_comp_dict["soma centre coords"] = soma_centres

        write_obj2pkl("%s/ct_vol_comp.pkl" % f_name, ct_vol_comp_dict)

        plottime = time.time() - celltime
        print("%.2f sec for plotting" % plottime)
        time_stamps.append(time.time())
        step_idents.append('processing arrays per celltype, plotting')
        log.info("compartment volume estimation per celltype finished")

    def compare_compartment_volume_ct(celltype1, celltype2, filename1 = None, filename2 = None, min_comp_len = 100):
        '''
        compares estimated compartment volumes (by bounding box) between two celltypes that have been generated by axon_den_arborization_ct.
        Data will be compared in histogram and violinplots. P-Values are computed by ranksum test.
        :param celltype1: j0256: STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7,
    #                      FS=8, LTS=9, NGF=10
        :param celltype2: compared against celltype 2
        :param filename1, filename2: only if data preprocessed: filename were preprocessed data is stored
        :param min_comp_len: minimum comparment length used by analysis
        :return:
        '''
        start = time.time()
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   10: "NGF"}
        f_name = "u/arother/bio_analysis_results/dir_indir_pathway_analysis/210929_j0251v3__%s_%s_comp_volume_mcl%i" % (
        ct_dict[celltype1],ct_dict[celltype2], min_comp_len)
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('compare compartment volumes between two celltypes', log_dir=f_name + '/logs/')
        log.info("parameters: celltype1 = %s,celltype2 = %s min_comp_length = %.i" % (ct_dict[celltype1], ct_dict[celltype2], min_comp_len))
        time_stamps = [time.time()]
        step_idents = ['t-0']
        ct1_comp_dict = load_pkl2obj("%s/ct_vol_comp.pkl"% filename1)
        ct2_comp_dict = load_pkl2obj("%s/ct_vol_comp.pkl"% filename2)
        comp_dict_keys = list(ct1_comp_dict.keys())
        if "soma centre coords" in comp_dict_keys:
            log.info("compute mean soma distance between %s and %s" % (ct_dict[celltype1], ct_dict[celltype2]))
            ct1_soma_coords = ct1_comp_dict["soma centre coords"]
            ct2_soma_coords = ct2_comp_dict["soma centre coords"]
            ct1_distances2ct2 = scipy.spatial.distance.cdist(ct1_soma_coords, ct2_soma_coords, metric="euclidean") / 1000
            ct1avg_soma_distance2ct2_per_cell = np.mean(ct1_distances2ct2, axis=1)
            ct2_distances2ct1 = scipy.spatial.distance.cdist(ct2_soma_coords, ct1_soma_coords,
                                                             metric="euclidean") / 1000
            ct2avg_soma_distance2ct1_per_cell = np.mean(ct2_distances2ct1, axis=1)
            ct1_comp_dict["avg soma distance to other celltype"] = ct1avg_soma_distance2ct2_per_cell
            ct2_comp_dict["avg soma distance to other celltype"] = ct2avg_soma_distance2ct1_per_cell
            ct1_comp_dict.pop("soma centre coords")
            ct2_comp_dict.pop("soma centre coords")
            comp_dict_keys = list(ct1_comp_dict.keys())
        log.info("compute statistics for comparison, create violinplot and histogram")
        ranksum_results = pd.DataFrame(columns=comp_dict_keys[1:], index=["stats", "p value"])
        colours_pal = {ct_dict[celltype1]: "mediumorchid", ct_dict[celltype2]: "springgreen"}
        max_len = np.max(np.array([len(ct1_comp_dict[comp_dict_keys[0]]), len(ct2_comp_dict[comp_dict_keys[0]])]))
        for key in ct1_comp_dict.keys():
            if "ids" in key:
                continue
            #calculate p_value for parameter
            stats, p_value = ranksums(ct1_comp_dict[key], ct2_comp_dict[key])
            ranksum_results.loc["stats", key] = stats
            ranksum_results.loc["p value", key] = p_value
            #plot parameter as violinplot
            results_for_plotting = pd.DataFrame(columns=[ct_dict[celltype1], ct_dict[celltype2]], index=range(max_len))
            results_for_plotting.loc[0:len(ct1_comp_dict[key]) - 1, ct_dict[celltype1]] = ct1_comp_dict[key]
            results_for_plotting.loc[0:len(ct2_comp_dict[key]) - 1, ct_dict[celltype2]] = ct2_comp_dict[key]
            sns.violinplot(data= results_for_plotting, inner="box", palette=colours_pal)
            sns.stripplot(data=results_for_plotting, color="black", alpha=0.2)
            plt.title('%s' % key)
            if "length" in key:
                plt.ylabel("pathlength in µm")
            elif "vol" in key:
                if "percentage" in key:
                    plt.ylabel("% of whole dataset")
                else:
                    plt.ylabel("volume in µm³")
            else:
                plt.ylabel("distance in µm")
            filename = ("%s/%s_violin.png" % (f_name, key))
            plt.savefig(filename)
            plt.close()
            sns.boxplot(data=results_for_plotting, palette=colours_pal)
            plt.title('%s' % key)
            if "length" in key:
                plt.ylabel("pathlength in µm")
            elif "vol" in key:
                if "percentage" in key:
                    plt.ylabel("% of whole dataset")
                else:
                    plt.ylabel("volume in µm³")
            else:
                plt.ylabel("distance in µm")
            filename = ("%s/%s_box.png" % (f_name, key))
            plt.savefig(filename)
            plt.close()
            sns.distplot(ct1_comp_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                         kde=False, label=ct_dict[celltype1], bins=10)
            sns.distplot(ct2_comp_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                         kde=False, label=ct_dict[celltype2], bins=10)
            plt.legend()
            plt.title('%s' % key)
            plt.ylabel("count of cells")
            if "length" in key:
                plt.xlabel("pathlength in µm")
            elif "vol" in key:
                if "percentage" in key:
                    plt.xlabel("% of whole dataset")
                else:
                    plt.xlabel("volume in µm³")
            else:
                plt.xlabel("distance in µm")
            filename = ("%s/%s_hist.png" % (f_name, key))
            plt.savefig(filename)
            plt.close()
            sns.distplot(ct1_comp_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                         kde=False, label=ct_dict[celltype1], bins=10, norm_hist=True)
            sns.distplot(ct2_comp_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                         kde=False, label=ct_dict[celltype2], bins=10, norm_hist=True)
            plt.legend()
            plt.title('%s' % key)
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
            filename = ("%s/%s_hist_norm.png" % (f_name, key))
            plt.savefig(filename)
            plt.close()

        ranksum_results.to_csv("%s/ranksum_%s_%s.csv" % (f_name, ct_dict[celltype1], ct_dict[celltype2]))

        plottime = time.time() - start
        print("%.2f sec for statistics and plotting" % plottime)
        time_stamps.append(time.time())
        step_idents.append('comparing celltypes')
        log.info("compartment volume comparison finished")

    def compare_compartment_volume_percentiles(celltype, percentile, filename1 = None, filename2 = None, min_comp_len = 100):
        '''
        compares estimated compartment volumes (by bounding box) between different fractions of one celltype that have been generated by axon_den_arborization_ct.
        Data will be compared in histogram and violinplots. P-Values are computed by ranksum test.
        :param celltype: j0256: STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7,
    #                      FS=8, LTS=9, NGF=10
        :param percentile: percentile given and 100 - percentile of a given celltye will be compared
        :param filename1, filename2: only if data preprocessed: filename were preprocessed data is stored
        :param min_comp_len: minimum comparment length used by analysis
        :return:
        '''
        start = time.time()
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   10: "NGF"}
        f_name = "u/arother/bio_analysis_results/dir_indir_pathway_analysis/210929_j0251v3__%s_comp_volume_mcl%i_p%i" % (
        ct_dict[celltype], min_comp_len, percentile)
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('compare compartment volumes between two celltypes', log_dir=f_name + '/logs/')
        log.info("parameters: celltype1 = %s, min_comp_length = %.i, percentile = %i" % (ct_dict[celltype], min_comp_len, percentile))
        time_stamps = [time.time()]
        step_idents = ['t-0']
        p1_comp_dict = load_pkl2obj("%s/ct_vol_comp.pkl"% filename1)
        p2_comp_dict = load_pkl2obj("%s/ct_vol_comp.pkl"% filename2)
        comp_dict_keys = list(p1_comp_dict.keys())
        if "soma centre coords" in comp_dict_keys:
            log.info("compute mean soma distance between percentiles %i and %i of %s" % (percentile, 100 - percentile, ct_dict[celltype]))
            p1_soma_coords = p1_comp_dict["soma centre coords"]
            p2_soma_coords = p2_comp_dict["soma centre coords"]
            p1_distances2p2 = scipy.spatial.distance.cdist(p1_soma_coords, p2_soma_coords, metric="euclidean") / 1000
            p1avg_soma_distance2p2_per_cell = np.mean(p1_distances2p2, axis=1)
            p2_distances2p1 = scipy.spatial.distance.cdist(p2_soma_coords, p1_soma_coords,
                                                             metric="euclidean") / 1000
            p2avg_soma_distance2p1_per_cell = np.mean(p2_distances2p1, axis=1)
            p1_comp_dict["avg soma distance to other celltype"] = p1avg_soma_distance2p2_per_cell
            p2_comp_dict["avg soma distance to other celltype"] = p2avg_soma_distance2p1_per_cell
            p1_comp_dict.pop("soma centre coords")
            p2_comp_dict.pop("soma centre coords")
            comp_dict_keys = list(p1_comp_dict.keys())
        log.info("compute statistics for comparison, create violinplot and histogram")
        ranksum_results = pd.DataFrame(columns=comp_dict_keys[1:], index=["stats", "p value"])
        colours_pal = {percentile: "mediumorchid", 100 - percentile: "springgreen"}
        max_len = np.max(np.array([len(p1_comp_dict[comp_dict_keys[0]]), len(p2_comp_dict[comp_dict_keys[0]])]))
        for key in p1_comp_dict.keys():
            if "ids" in key:
                continue
            #calculate p_value for parameter
            stats, p_value = ranksums(p1_comp_dict[key], p2_comp_dict[key])
            ranksum_results.loc["stats", key] = stats
            ranksum_results.loc["p value", key] = p_value
            #plot parameter as violinplot
            results_for_plotting = pd.DataFrame(columns=[percentile, 100 - percentile], index=range(max_len))
            results_for_plotting.loc[0:len(p1_comp_dict[key]) - 1, percentile] = p1_comp_dict[key]
            results_for_plotting.loc[0:len(p2_comp_dict[key]) - 1, 100 - percentile] = p2_comp_dict[key]
            sns.violinplot(data= results_for_plotting, inner="box", palette=colours_pal)
            sns.stripplot(data=results_for_plotting, color="black", alpha=0.2)
            plt.title('%s' % key)
            if "length" in key:
                plt.ylabel("pathlength in µm")
            elif "vol" in key:
                if "percentage" in key:
                    plt.ylabel("% of whole dataset")
                else:
                    plt.ylabel("volume in µm³")
            else:
                plt.ylabel("distance in µm")
            filename = ("%s/%s_violin.png" % (f_name, key))
            plt.savefig(filename)
            plt.close()
            sns.boxplot(data=results_for_plotting, palette=colours_pal)
            plt.title('%s' % key)
            if "length" in key:
                plt.ylabel("pathlength in µm")
            elif "vol" in key:
                if "percentage" in key:
                    plt.ylabel("% of whole dataset")
                else:
                    plt.ylabel("volume in µm³")
            else:
                plt.ylabel("distance in µm")
            filename = ("%s/%s_box.png" % (f_name, key))
            plt.savefig(filename)
            plt.close()
            sns.distplot(p1_comp_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                         kde=False, label=percentile, bins=10)
            sns.distplot(p2_comp_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                         kde=False, label=100 - percentile, bins=10)
            plt.legend()
            plt.title('%s in %s, percentiles %i vs %i' % (key, ct_dict[celltype], percentile, 100 - percentile))
            plt.ylabel("count of cells")
            if "length" in key:
                plt.xlabel("pathlength in µm")
            elif "vol" in key:
                if "percentage" in key:
                    plt.xlabel("% of whole dataset")
                else:
                    plt.xlabel("volume in µm³")
            else:
                plt.xlabel("distance in µm")
            filename = ("%s/%s_hist.png" % (f_name, key))
            plt.savefig(filename)
            plt.close()
            sns.distplot(p1_comp_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                         kde=False, label=percentile, bins=10, norm_hist=True)
            sns.distplot(p2_comp_dict[key],
                         hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                         kde=False, label=100 - percentile, bins=10, norm_hist=True)
            plt.legend()
            plt.title('%s in %s, percentiles %i vs %i' % (key, ct_dict[celltype], percentile, 100 - percentile))
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
            filename = ("%s/%s_hist_norm.png" % (f_name, key))
            plt.savefig(filename)
            plt.close()

        ranksum_results.to_csv("%s/ranksum_%s_%i_%i.csv" % (f_name, ct_dict[celltype], percentile, 100 - percentile))

        plottime = time.time() - start
        print("%.2f sec for statistics and plotting" % plottime)
        time_stamps.append(time.time())
        step_idents.append('comparing percentiles in %s' % ct_dict[celltype])
        log.info("compartment volume comparison finished")


    #axon_den_arborization_ct(ssd, celltype=7)
    #axon_den_arborization_ct(ssd, celltype=2, full_cells= True, handpicked=False)
    foldername = "u/arother/bio_analysis_results/dir_indir_pathway_analysis/"
    #percentiles = [10, 25, 50]
    percentiles = [50]
    for percentile in percentiles:
        axon_den_arborization_ct(ssd, celltype=2, full_cells=True, handpicked=False, percentile=percentile, low_percentile=True)
        axon_den_arborization_ct(ssd, celltype=2, full_cells=True, handpicked=False, percentile=100 - percentile,
                                 low_percentile=False)
        p1_filename = "%s/210929_j0251v3_MSN_comp_volume_mcl100_p%il" % (foldername, percentile)
        p2_filename = "%s/210929_j0251v3_MSN_comp_volume_mcl100_p%ih" % (foldername, 100 - percentile)
        compare_compartment_volume_percentiles(celltype=2, percentile=percentile, filename1=p1_filename,
                                  filename2=p2_filename)