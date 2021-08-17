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
            return 0, 0, 0
        comp_inds = np.nonzero(sso.skeleton["axoness_avg10000"] == compartment)[0]
        comp_nodes = sso.skeleton["nodes"][comp_inds] * sso.scaling
        min_x = np.min[comp_nodes[:,0]]
        max_x = np.max[comp_nodes[:, 0]]
        min_y = np.min[comp_nodes[:, 1]]
        max_y = np.max[comp_nodes[:, 1]]
        min_z = np.min[comp_nodes[:, 2]]
        max_z = np.max[comp_nodes[:, 2]]
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

    def axon_den_arborization_ct(ssd, celltype, min_comp_len = 100, full_cells = True, handpicked = True):
        '''
        estimate the axonal and dendritic aroborization by celltype. Uses axon_dendritic_arborization to get the aoxnal/dendritic bounding box volume per cell
        via comp_arborization. Plots the volume per compartment and the overall length as histograms.
        :param ssd: super segmentation dataset
        :param celltype: j0256: STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7,
    #                      FS=8, LTS=9, NGF=10
        :param min_comp_len: minimum compartment length in µm
        :param full_cells: loads preprocessed cells that have axon, soma and dendrite
        :param handpicked: loads cells that were manually checked
        :return:
        '''

        start = time.time()
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   10: "NGF"}
        f_name = "u/arother/bio_analysis_results/210817_j0251v3__%s_comp_volume_mcl%i" % (ct_dict[celltype], min_comp_len)
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('compartment volume estimation', log_dir=f_name + '/logs/')
        log.info("parameters: min_comp_length = %.i" % min_comp_len)
        time_stamps = [time.time()]
        step_idents = ['t-0']
        if full_cells:
            soma_centre_dict = load_pkl2obj(
                    "/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_dict.pkl" % ct_dict[celltype])

            if handpicked:
                cellids = load_pkl2obj(
                    "/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_handpicked.pkl" % ct_dict[celltype])
            else:
                cellids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_arr.pkl" % ct_dict[celltype])
        else:
            cellids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == celltype]
        log.info('Step 1/2 calculating volume estimate for axon/dendrite per cell')
        axon_length_ct = np.zeros(len(cellids))
        dendrite_length_ct = np.zeros(cellids)
        axon_vol_ct = np.zeros(len(cellids))
        dendrite_vol_ct = np.zeros(len(cellids))
        if full_cells:
            soma_centres = np.zeros(len(cellids))
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
        if full_cells:
            soma_centres = soma_centres[nonzero_inds]
            distances_between_soma = scipy.spatial.distance.pdist(soma_centres, metric = "euclidean")
            avg_dist_soma = np.mean(distances_between_soma) / 1000
            ct_vol_comp_dict = {"cell ids": cellids,"axon length": axon_length_ct, "dendrite length": dendrite_length_ct,
                                "axon volume bb": axon_vol_ct, "dendrite volume bb": dendrite_vol_ct, "mean soma distance": avg_dist_soma}
        else:
            ct_vol_comp_dict = {"cell ids": cellids,"axon length": axon_length_ct, "dendrite length": dendrite_length_ct,
                                "axon volume bb": axon_vol_ct, "dendrite volume bb": dendrite_vol_ct}
        write_obj2pkl("%s/ct_vol_comp.pkl" % f_name, ct_vol_comp_dict)
        vol_comp_pd = pd.DataFrame(ct_vol_comp_dict)
        vol_comp_pd.to_csv("%s/ct_vol_comp.csv")

        for key in ct_vol_comp_dict.keys():
            if key is "cell ids":
                continue
            sns.distplot(ct_vol_comp_dict[key], hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "steelblue"},
                         kde=False)
            plt.ylabel("count of cells")
            if "length" in key:
                plt.ylabel("pathlength in µm")
            if "vol" in key:
                plt.ylabel("volume in µm³")
            else:
                plt.ylabel("distance in µm")
            plt.title("%s" % key)
            plt.savefig("%s/%s.png" % (f_name, key))

        plottime = time.time() - celltime
        print("%.2f sec for plotting" % plottime)
        time_stamps.append(time.time())
        step_idents.append('processing arrays per celltype, plotting')
        log.info("compartment volume estimation per celltype finished")