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
    import time
    from syconn.handler.config import initialize_logging
    from syconn.handler.basics import load_pkl2obj, write_obj2pkl
    #global_params.wd = "/ssdscratch/pschuber/songbird/j0126/assembled_core_relabeled_base_merges_relabeled_to_v4b_base_20180214_full_agglo_cbsplit_with_reconnects_no_soma_merger_manual_edges_removed"
    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv",working_dir=global_params.config.working_dir)

##two possibilites of doing the analysis:
    # A) iterating over shortestpath2soma and count branching points in distance intervals
    #+ gets rid of limitation of growing back axons in Gutierrez and Davies, 2007
    #- relies on skeletons with shortestpath2soma
    #B) creating concentric circles and count number of branching and terminating points within these
    # + independent of pathlength as only iwthin concentric circles
    # - limitation with growing back axon still there
    # - problems with skeleton still exits e.g. non-biological termination points
    #To DO plot braching and terminating points as density -> in relation to complete dendrite pathlength

    def sholl_prep(sso, compartment):
        """
        For preperation of the sholl analysis function computes starting value from networkx which is the amount of desired compartments (default: dendrites) emerging from the soma.
        It also computes the shortestpath2soma for nodes that are not degree = 2 e.g. are branching points or terminating points and saves them in array for sholl analysis.
               compartment = compartment that should be looked at, default = 0 (dendrites) to look at dendritic tree, Spine predictions: 0: neck, 1: head, 2: shaft, 3: other(dendrite, axon)
        :return: return shortestpath, degrees
        """
        #compute starting points for dendrite or axon
        #remove all compartments with value 1 = axon, value 2 = soma: (see function majority_vote_compartments in github)
        # if want to look at axons: remove values 0 = dendrite and 2 = soma, spiness= 0 = dendritic shaft, 1 = spine head, 2 = spine neck, 3 = other
        #from create_sso_skeleton_fast
        if compartment == 0:
            unwanted_compartment = 1
        elif compartment == 1:
            unwanted_compartment = 0
        else:
            print('unknown compartment value %.i. Compartment has to be either 0 or 1' % compartment)
            raise ValueError
        #TO DO: remove all nodes in spines
        #also remove small branches that might be spines? or leave all spines in to compare?
        # To DO: introduce spine parameter, only important or dendrites
        g = sso.weighted_graph(add_node_attr=('axoness_avg10000',"spiness"))
        desired_compartment_graph = g.copy()
        if compartment != 1:
            for n, d in g.nodes(data=True):
                axoness = d["axoness_avg10000"]
                if axoness == 3 or axoness == 4:
                    axoness = 1
                if d['axoness_avg10000'] == 2 or d['axoness_avg10000'] == unwanted_compartment:
                    desired_compartment_graph.remove_node(n)
                if d["spiness"] == 1 or d["spiness"] == 2:
                    if n not in desired_compartment_graph.nodes():
                        continue
                    desired_compartment_graph.remove_node(n)
            x0 = len(list(nx.connected_component_subgraphs(desired_compartment_graph)))
            if x0 == 0:
                x0 = 1
        else: #only accept one axon emerging from soma
            x0 = 1
        # starting value = dendrites emerging from soma = number of connected_components
        # compute number of branching points, termination points for pathlength in intervals length r:
        # termination points: G.degree[node] = 1 -> only one edge, branching points: G.degree[node] >= 3
        #option 1: iterate over nodes and sort points into corresponding intersections: have to have limit of length given
        #option 2: interate over pathlength until there are no nodes left but then need to iterate over nodes in graph more than once
        #option 1
        axoness = sso.skeleton["axoness_avg10000"]
        axoness[axoness == 3] = 1
        axoness[axoness == 4] = 1
        compartment_inds = np.nonzero(axoness == compartment)[0]
        if compartment == 0:
            spiness_inds1 = np.nonzero(sso.skeleton["spiness"] == 0)[0]
            spiness_inds2 = np.nonzero(sso.skeleton["spiness"] == 3)[0]
            spiness_inds = np.hstack(np.array([spiness_inds1, spiness_inds2]))
            compartment_inds = compartment_inds[np.in1d(compartment_inds, spiness_inds)]
        coords = sso.skeleton["nodes"][compartment_inds]
        degrees = np.zeros(len(coords))
        for ix, node_id in enumerate(compartment_inds):
            degrees[ix] = g.degree[node_id]
        n_coords = coords[degrees != 2]
        degrees = degrees[degrees != 2]
        shortest_paths = sso.shortestpath2soma(n_coords)
        shortest_paths = np.array(shortest_paths)
        return x0, shortest_paths, degrees


    def ct_sholl_analysis(ssd, celltype,  compartment, r = 30000):
        """
        computes sholl_analysis per celltype and per cell with shortest_paths and degrees of each node (degree != 2).
        Function uses path_length in different intervals to compute length of dendritic tree and produce values for a Sholl profile. It counts the amount of branching points, termination points (see Gutierrez and Davies, 2017)
        via edges in networkx for intervals with difference r (default 30 µm = 30000 nm).
        Branching points are points that have at least 3 edges (3 edges = binary branching point = 1, more than three edges = n-1 branching points), Termination points are points with only one edge.
        Start values are emerging dendrites from soma which is the amount of connected compartments after removal of soma, axons.
        :param ssd: SuperSegmentationDataset
        :param j0126: STN=0, modulatory=1, MSN=2, LMAN=3, HVC=4, GP=5, INT=6
                r = length of intersections
                j0251:ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7:"GPi", 8:"FS", 9:"LTS", 10:"NGF"}
        :return: sholl_dict, bt_dict, avg_b, med_b = mean/median amount of branching points per celltype, avg_t, med_t = mean/median amount of terminating points per celltype
        """
        #per celltyype function
        # for cell == celltype:
            #sholl_analysis(cell)
        start = time.time()
        f_name = "u/arother/test_folder/210414_sholl_analysis_j0251v3GPhandpicked_r30000"
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('sholl_analysis', log_dir=f_name + '/logs/')
        time_stamps = [time.time()]
        step_idents = ['t-0']
        #cellids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == celltype]
        sholl_dict = {}
        comp_dict = {1: 'axons', 0: 'dendrites'}
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7:"GPi", 8:"FS", 9:"LTS", 10:"NGF"}
        ct_dict_GP = {6: "GP"}
        #cellids = load_pkl2obj("/wholebrain/scratch/arother/j0256_prep/full_%3s_arr.pkl" % ct_dict[celltype])
        #gpeids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%3s_arr.pkl" % ct_dict[celltype[0]])
        #gpiids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%3s_arr.pkl" % ct_dict[celltype[1]])
        gpeids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/handpicked_%3s_arr.pkl" % ct_dict[celltype[0]])
        gpiids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/handpicked_%3s_arr.pkl" % ct_dict[celltype[1]])
        gpesomacentr = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%3s_dict.pkl" % ct_dict[celltype[0]])
        gpisomacentr = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%3s_dict.pkl" % ct_dict[celltype[1]])
        cellids = np.hstack(np.array([gpeids, gpiids]))
        celltype = celltype[0]
        dataset_voxelsize = np.array([27119, 27350, 15494])
        dataset_nm = dataset_voxelsize * ssd.scaling
        log.info('Step 1/2 generating sholl per cell')
        for pi, cell in enumerate(ssd.get_super_segmentation_object(cellids)):
            cell.load_skeleton()
            #make sure only cells are used that are not at the edges
            #dataset size: 0.256 mm x 0.256 mm x 0.394 mm, laut conf: in voxel: 27119, 27359, 15494
            '''
            if cell.id in gpeids:
                soma_center = gpesomacentr[cell.id]
            else:
                soma_center = gpisomacentr[cell.id]
            if np.any(soma_center < 20000):
                continue
            if np.any(soma_center > dataset_nm - 20000):
                continue
            '''
            x0, shortest_paths, degrees = sholl_prep(cell, compartment=compartment)
            if len(shortest_paths) == 0:
                continue
            # rmax dependent on longest shortest path
            rmax = np.max(shortest_paths)
            # intersections for sholl analysis
            inter_amount = int(rmax / r) + 1
            bt_count = np.zeros((inter_amount, 3))# [0] = distance to soma in intersections, [1] = branching points, [2] = terminal points
            for i in range(inter_amount):
                bt_count[i][0] = (r * (i + 1)) / 1000
            # counting branching and terminating points
            for ix in range(len(degrees)):
                degree = degrees[ix]
                shortest_path = shortest_paths[ix]
                if shortest_path == 0:
                    continue
                intersection = int(shortest_path / r)
                if intersection > inter_amount:
                    print('pathlength  %.7f longer than rmax = %.7f' % (shortest_path, rmax))
                    raise ValueError
                if degree == 1:  # terminal point
                    bt_count[intersection][2] += 1
                elif degree >= 3:
                    n = degree - 2  # number of branching points (binary branching point = degree 3)
                    bt_count[intersection][1] += n
            sholl_array = np.zeros((inter_amount, 2))  # [0] = distance to soma in intersections, [1] = X for intersections
            # sholl analysis per cell
            for i in range(inter_amount):
                sholl_array[i][0] = bt_count[i][0]
                if i == 0:
                    sholl_array[i][1] = x0 + bt_count[i][1] - bt_count[i][2]
                else:
                    sholl_array[i][1] = sholl_array[i - 1][1] + bt_count[i][1] - bt_count[i][2]
            # plot sholl_array per cell
            columns = ['pathlength in µm', ('count of %.9s' % comp_dict[compartment])]
            sholl_dataset = pd.DataFrame(data=sholl_array, columns=columns)
            sns.catplot(x='pathlength in µm', y='count of %.9s' % comp_dict[compartment], color="blue",
                        data=sholl_dataset, kind='bar')
            foldername = '%s/sholl_percell_%s_n' % (f_name, ct_dict[celltype])
            if not os.path.exists(foldername):
                os.mkdir(foldername)
            plt.savefig('%s/sholl_%.i_%s_%s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
            plt.close()
            # add to dictionary per celltype
            sholl_dict[cell.id] = dict()
            sholl_dict[cell.id]["sholl"] = sholl_array
            sholl_dict[cell.id]["bt"] = bt_count
            sholl_dict[cell.id]["shortest_paths"] = shortest_paths
            sholl_dict[cell.id]["degrees"] = degrees
            sholl_dict[cell.id]["X0"] = x0
            percentage = 100 * (pi / len(cellids))
            print("%.2f percent" % percentage)
            # intersections in celltype
        celltime = time.time() - start
        print("%.2f sec for processing cells" % celltime)
        time_stamps.append(time.time())
        step_idents.append('sholl per cell')
        log.info("Step 2/2 plot graphs per celltype")
        intersections = []
        for key in sholl_dict:
            for i in range(len(sholl_dict[key]["sholl"])):
                if sholl_dict[key]["sholl"][i][0] not in intersections:
                    intersections.append(sholl_dict[key]["sholl"][i][0])
        intersections = np.array(intersections)
        intersections = np.sort(intersections)

        #graphs for max sholl_analysis per cell: graph plots amount of cells that have this maximum vs. intersection bin
        # TO DO: find way to convert distributions into one distribution: make bins smaller and do PCA on them or take bin with maximum amount of counts & variance?;
        #graphs needed: furthest bin of sholl, highest bin of sholl = maximum, median of sholl; average number of terminating ppoints and branching points, PCA on Sholl with small binsize
        max_dis = np.zeros(len(sholl_dict.keys())) #maximum of sholl_plot: in which intersection are the most dendrites: plot bin
        avg_dis = np.zeros(len(sholl_dict.keys()))
        var_dis = np.zeros(len(sholl_dict.keys()))# which bin holds median of dendrites? Amount of cells whose median is in same bin will be plotted
        furthestbin_dis = np.zeros(len(sholl_dict.keys()))#maximum bin used, how long is longest
        avg_b_am = np.zeros(len(sholl_dict.keys()))
        avg_t_am = np.zeros(len(sholl_dict.keys()))
        sum_b = np.zeros(len(sholl_dict.keys()))
        sum_t = np.zeros(len(sholl_dict.keys()))
        gpemax_dis = np.zeros(len(sholl_dict.keys()))
        gpeavg_dis = np.zeros(len(sholl_dict.keys()))
        gpevar_dis = np.zeros(len(sholl_dict.keys()))
        gpefurthestbin_dis = np.zeros(len(sholl_dict.keys()))
        gpeavg_b_am = np.zeros(len(sholl_dict.keys()))
        gpeavg_t_am = np.zeros(len(sholl_dict.keys()))
        gpesum_b = np.zeros(len(sholl_dict.keys()))
        gpesum_t = np.zeros(len(sholl_dict.keys()))
        gpimax_dis = np.zeros(len(sholl_dict.keys()))
        gpiavg_dis = np.zeros(len(sholl_dict.keys()))
        gpivar_dis = np.zeros(len(sholl_dict.keys()))
        gpifurthestbin_dis = np.zeros(len(sholl_dict.keys()))
        gpiavg_b_am = np.zeros(len(sholl_dict.keys()))
        gpiavg_t_am = np.zeros(len(sholl_dict.keys()))
        gpisum_b = np.zeros(len(sholl_dict.keys()))
        gpisum_t = np.zeros(len(sholl_dict.keys()))
        gpe_sids = np.zeros(len(sholl_dict.keys()))
        gpi_sids = np.zeros(len(sholl_dict.keys()))
        for ii, cell in enumerate(sholl_dict):
            sholl = sholl_dict[cell]["sholl"]
            X0 = sholl_dict[cell]["X0"]
            if type(sholl) != np.ndarray:
                continue
            if sholl.size == 0:
                continue
            in_val = sholl[:, 1]
            max = np.max(in_val)
            max_i = np.where(in_val == max)[0]
            max_dis[ii] = sholl[int(max_i[-1])][0]
            length = len(in_val)
            furthestbin_dis[ii] = (sholl[length - 1][0])
            avg = np.mean(sholl, axis= 0)
            avg_dis[ii] = (avg[0])
            var = np.var(sholl, axis = 0)
            var_dis[ii] = var[0]
            #branching, terminating points
            bt = sholl_dict[cell]["bt"]
            sum_b[ii] = np.sum(bt[:,1])
            sum_t[ii] = np.sum(bt[:,2])
            # only if dendrites: average branching and terminating points per initial compartment
            if compartment == 0:
                if sum_b[ii] != 0:
                    avg_b_am[ii] = sum_b[ii] / X0
                if sum_t[ii] != 0:
                    avg_t_am[ii] = sum_t[ii] / X0
            if cell in gpeids:
                gpe_sids[ii] = cellids[ii]
                gpemax_dis[ii] = max_dis[ii]
                gpefurthestbin_dis[ii] = furthestbin_dis[ii]
                gpeavg_dis[ii] = avg_dis[ii]
                gpevar_dis[ii] = var_dis[ii]
                gpesum_b[ii] = sum_b[ii]
                gpesum_t[ii] = sum_t[ii]
                if compartment == 0:
                    gpeavg_b_am[ii] = avg_b_am[ii]
                    gpeavg_t_am[ii] = avg_t_am[ii]
            elif cell in gpiids:
                gpi_sids[ii] = cellids[ii]
                gpimax_dis[ii] = max_dis[ii]
                gpifurthestbin_dis[ii] = furthestbin_dis[ii]
                gpiavg_dis[ii] = avg_dis[ii]
                gpivar_dis[ii] = var_dis[ii]
                gpisum_b[ii] = sum_b[ii]
                gpisum_t[ii] = sum_t[ii]
                if compartment == 0:
                    gpiavg_b_am[ii] = avg_b_am[ii]
                    gpiavg_t_am[ii] = avg_t_am[ii]

        zsum_b = np.where(sum_b <= 0)
        zsum_t = np.where(sum_t <= 0)
        zmax_dis = np.where(max_dis <= 0)
        zavg_dis = np.where(avg_dis <= 0)
        zvar_dis = np.where(var_dis <= 0)
        zfurthest = np.where(furthestbin_dis <= 0)
        arr_inds = np.unique(np.hstack([zsum_b, zsum_t, zmax_dis, zavg_dis, zvar_dis, zfurthest]))
        sum_b = np.delete(sum_b, arr_inds)
        sum_t = np.delete(sum_t, arr_inds)
        max_dis = np.delete(max_dis, arr_inds)
        avg_dis = np.delete(avg_dis, arr_inds)
        var_dis = np.delete(var_dis, arr_inds)
        furthestbin_dis = np.delete(furthestbin_dis, arr_inds)

        gpezsum_b = np.where(gpesum_b <= 0)
        gpezsum_t = np.where(gpesum_t <= 0)
        gpezmax_dis = np.where(gpemax_dis <= 0)
        gpezavg_dis = np.where(gpeavg_dis <= 0)
        gpezvar_dis = np.where(gpevar_dis <= 0)
        gpezfurthest = np.where(gpefurthestbin_dis <= 0)
        gpearr_inds = np.unique(np.hstack([gpezsum_b, gpezsum_t, gpezmax_dis, gpezavg_dis, gpezvar_dis, gpezfurthest]))
        gpesum_b = np.delete(gpesum_b, gpearr_inds)
        gpesum_t = np.delete(gpesum_t, gpearr_inds)
        gpemax_dis = np.delete(gpemax_dis, gpearr_inds)
        gpeavg_dis = np.delete(gpeavg_dis, gpearr_inds)
        gpevar_dis = np.delete(gpevar_dis, gpearr_inds)
        gpefurthestbin_dis = np.delete(gpefurthestbin_dis, gpearr_inds)
        # get ids of GPE whose longest bin is shorter than 200
        gpe_sids = np.delete(gpe_sids, gpearr_inds)
        gpe_sh_ids = gpe_sids[gpefurthestbin_dis < 200]
        gpe_lg_ids = gpe_sids[gpefurthestbin_dis > 200]
        write_obj2pkl("%s/gpe_sh_arr_%s.pkl" % (f_name, comp_dict[compartment]), gpe_sh_ids)
        write_obj2pkl("%s/gpe_lg_arr_%s.pkl" % (f_name, comp_dict[compartment]), gpe_lg_ids)
        #gpis
        gpizsum_b = np.where(gpisum_b <= 0)
        gpizsum_t = np.where(gpisum_t <= 0)
        gpizmax_dis = np.where(gpimax_dis <= 0)
        gpizavg_dis = np.where(gpiavg_dis <= 0)
        gpizvar_dis = np.where(gpivar_dis <= 0)
        gpizfurthest = np.where(gpifurthestbin_dis <= 0)
        gpiarr_inds = np.unique(np.hstack([gpizsum_b, gpizsum_t, gpizmax_dis, gpizavg_dis, gpizvar_dis, gpizfurthest]))
        gpisum_b = np.delete(gpisum_b, gpiarr_inds)
        gpisum_t = np.delete(gpisum_t, gpiarr_inds)
        gpimax_dis = np.delete(gpimax_dis, gpiarr_inds)
        gpiavg_dis = np.delete(gpiavg_dis, gpiarr_inds)
        gpivar_dis = np.delete(gpivar_dis, gpiarr_inds)
        gpifurthestbin_dis = np.delete(gpifurthestbin_dis, gpiarr_inds)
        # get ids of GPi whose longest bin is shorter than 200
        gpi_sids = np.delete(gpi_sids, gpiarr_inds)
        gpi_sh_ids = gpi_sids[gpifurthestbin_dis < 200]
        gpi_lg_ids = gpi_sids[gpifurthestbin_dis > 200]
        write_obj2pkl("%s/gpi_sh_arr_%s.pkl" % (f_name, comp_dict[compartment]), gpi_sh_ids)
        write_obj2pkl("%s/gpi_lg_arr_%s.pkl" % (f_name, comp_dict[compartment]), gpi_lg_ids)
        if compartment == 0:
            avg_b_am = np.delete(avg_b_am, arr_inds)
            avg_t_am = np.delete(avg_dis, arr_inds)
            gpeavg_b_am = np.delete(gpeavg_b_am, gpearr_inds)
            gpeavg_t_am = np.delete(gpeavg_t_am, gpearr_inds)
            gpiavg_b_am = np.delete(gpiavg_b_am, gpiarr_inds)
            gpiavg_t_am = np.delete(gpiavg_t_am, gpiarr_inds)
            sns.distplot(avg_b_am, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"}, kde=False, label="GP")
            sns.distplot(gpeavg_b_am, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"}, kde=False,label="GPe")
            sns.distplot(gpiavg_b_am, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"}, kde=False, label="GPi")
            avb_filename = ('%s/avgb_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
            plt.title('average amount of branching points per dendrite in %.4s' % ct_dict_GP[celltype])
            plt.xlabel('amount of branching points/ dendrite')
            plt.ylabel('count of cells')
            plt.legend()
            plt.savefig(avb_filename)
            plt.close()
            sns.distplot(avg_t_am, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"}, kde=False, label="GP")
            sns.distplot(gpeavg_t_am, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"}, kde=False, label="GPe")
            sns.distplot(gpiavg_t_am, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"}, kde=False, label="GPi")
            avt_filename = ('%s/avt_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
            plt.title('average amount of terminating points per dendrite  in %.4s' % ct_dict_GP[celltype])
            plt.xlabel('amount of terminating points per dendrite')
            plt.ylabel('count of cells')
            plt.legend()
            plt.savefig(avt_filename)
            plt.close()
            #log plots
            sns.distplot(np.log10(avg_b_am), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"}, kde=False, label="GP")
            sns.distplot(np.log10(gpeavg_b_am), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"}, kde=False, label="GPe")
            sns.distplot(np.log10(gpiavg_b_am), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"}, kde=False, label="GPi")
            avb_filename = ('%s/logavgb_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
            plt.title('log of average amount of branching points per dendrite in %.4s' % ct_dict_GP[celltype])
            plt.xlabel('log of amount of branching points/ dendrite in 10^')
            plt.ylabel('count of cells')
            plt.legend()
            plt.savefig(avb_filename)
            plt.close()
            sns.distplot(np.log10(avg_t_am), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"}, kde=False, label="GP")
            sns.distplot(np.log10(gpeavg_t_am), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"}, kde=False, label="GPe")
            sns.distplot(np.log10(gpiavg_t_am), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"}, kde=False, label="GPi")
            avt_filename = ('%s/logavt_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
            plt.title('log of average amount of terminating points per dendrite  in %.4s' % ct_dict_GP[celltype])
            plt.xlabel('log of amount of terminating points per dendrite in 10^')
            plt.ylabel('count of cells')
            plt.legend()
            plt.savefig(avt_filename)
            plt.close()
            plt.scatter(x = gpeavg_b_am, y=gpesum_b, c="mediumorchid", label="GPe")
            plt.scatter(x = gpiavg_b_am, y=gpisum_b, c="springgreen", label = "GPi")
            sc_filename = ('%s/scavgbsumb_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
            plt.title('sum of branching points vs branching points per dendrite  in %.4s' % ct_dict_GP[celltype])
            plt.xlabel('amount of branching points per dendrite')
            plt.ylabel('sum of branching points')
            plt.legend()
            plt.savefig(sc_filename)
            plt.close()
            plt.scatter(x=np.log10(gpeavg_b_am), y=np.log10(gpesum_b), c="mediumorchid", label="GPe")
            plt.scatter(x=np.log10(gpiavg_b_am), y=np.log10(gpisum_b), c="springgreen", label="GPi")
            sc_filename = ('%s/logscavgbsumb_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
            plt.title('log sum of branching points vs branching points per dendrite  in %.4s' % ct_dict_GP[celltype])
            plt.xlabel('log amount of branching points per dendrite in 10^')
            plt.ylabel('log sum of branching points in 10^')
            plt.legend()
            plt.savefig(sc_filename)
            plt.close()
            plt.scatter(x=gpeavg_b_am, y=gpemax_dis, c="mediumorchid", label="GPe")
            plt.scatter(x=gpiavg_b_am, y=gpimax_dis, c="springgreen", label="GPi")
            sc_filename = ('%s/scmaxdisavgb_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
            plt.title('maximum of sholl vs branching points per dendrite  in %.4s' % ct_dict_GP[celltype])
            plt.xlabel('amount of branching points per dendrite')
            plt.ylabel('sholl max. pathlength in µm')
            plt.legend()
            plt.savefig(sc_filename)
            plt.close()
            plt.scatter(x=np.log10(gpeavg_b_am), y=np.log10(gpemax_dis), c="mediumorchid", label="GPe")
            plt.scatter(x=np.log10(gpiavg_b_am), y=np.log10(gpimax_dis), c="springgreen", label="GPi")
            sc_filename = ('%s/logscmaxdisavgb_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
            plt.title('log maximum of sholl vs branching points per dendrite  in %.4s' % ct_dict_GP[celltype])
            plt.xlabel('log amount of branching points per dendrite in 10^')
            plt.ylabel('log of sholl max. pathlength in 10^ µm')
            plt.legend()
            plt.savefig(sc_filename)
            plt.close()
        sns.distplot(max_dis, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"}, kde=False, label="GP")
        sns.distplot(gpemax_dis, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"}, kde=False, label="GPe")
        sns.distplot(gpimax_dis, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"}, kde=False, label="GPi")
        max_filename = ('%s/maxsholl_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('bins of maxima in sholl analysis in %.4s' % ct_dict_GP[celltype])
        plt.xlabel('pathlength in µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(max_filename)
        plt.close()
        sns.distplot(max_dis, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"}, kde=False,
                     label="GP", norm_hist=True)
        sns.distplot(gpemax_dis, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpimax_dis, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        max_filename = ('%s/relmaxsholl_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('bins of maxima in sholl analysis in %.4s' % ct_dict_GP[celltype])
        plt.xlabel('pathlength in µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(max_filename)
        plt.close()
        sns.distplot(furthestbin_dis, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"}, kde=False, label="GP")
        sns.distplot(gpefurthestbin_dis, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"}, kde=False, label="GPe")
        sns.distplot(gpifurthestbin_dis, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"}, kde=False, label="GPi")
        furthest_filename = ('%s/furthestsholl_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('bins of longest dendrites in sholl analysis in %s' % ct_dict_GP[celltype])
        plt.xlabel('pathlength in µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(furthest_filename)
        plt.close()
        sns.distplot(furthestbin_dis, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, label="GP", norm_hist=True)
        sns.distplot(gpefurthestbin_dis,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"}, kde=False,
                     label="GPe", norm_hist=True)
        sns.distplot(gpifurthestbin_dis,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"}, kde=False,
                     label="GPi", norm_hist=True)
        furthest_filename = ('%s/relfurthestsholl_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('bins of longest dendrites in sholl analysis in %s' % ct_dict_GP[celltype])
        plt.xlabel('pathlength in µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(furthest_filename)
        plt.close()
        sns.distplot(avg_dis, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"}, kde=False, label = "GP")
        sns.distplot(gpeavg_dis, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"}, kde=False, label="GPe")
        sns.distplot(gpiavg_dis, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"}, kde=False, label="GPi")
        avg_filename = ('%s/avgsholl_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('average in sholl analysis in %s' % ct_dict_GP[celltype])
        plt.xlabel('pathlength in µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()
        sns.distplot(avg_dis, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"}, kde=False,
                     label="GP",norm_hist=True)
        sns.distplot(gpeavg_dis, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe",norm_hist=True)
        sns.distplot(gpiavg_dis, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi",norm_hist=True)
        avg_filename = ('%s/relavgsholl_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('average in sholl analysis in %s' % ct_dict_GP[celltype])
        plt.xlabel('pathlength in µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()
        sns.distplot(var_dis, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"}, kde=False, label="GP")
        sns.distplot(gpevar_dis, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"}, kde=False, label="GPe")
        sns.distplot(gpivar_dis, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"}, kde=False, label="GPi")
        avg_filename = ('%s/varsholl_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('variance in sholl analysis in %s' % ct_dict_GP[celltype])
        plt.xlabel('variance')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()
        sns.distplot(sum_b, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"}, kde=False, label="GP")
        sns.distplot(gpesum_b, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"}, kde=False, label="GPe")
        sns.distplot(gpisum_b, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"}, kde=False, label="GPi")
        sumb_filename = ('%s/sumb_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('sum of branching points in %.4s' % ct_dict_GP[celltype])
        plt.xlabel("number of branching points")
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(sumb_filename)
        plt.close()
        sns.distplot(sum_b, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"}, kde=False,
                     label="GP",norm_hist=True)
        sns.distplot(gpesum_b, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe",norm_hist=True)
        sns.distplot(gpisum_b, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi",norm_hist=True)
        sumb_filename = ('%s/relsumb_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('sum of branching points in %.4s' % ct_dict_GP[celltype])
        plt.xlabel("number of branching points")
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(sumb_filename)
        plt.close()
        sns.distplot(sum_t, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"}, kde=False, label="GP")
        sns.distplot(gpesum_t, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"}, kde=False, label="GPe")
        sns.distplot(gpisum_t, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"}, kde=False, label="GPi")
        sumt_filename = ('%s/sumt_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('sum of terminating points in %.4s' % ct_dict_GP[celltype])
        plt.xlabel("number of terminating points")
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(sumt_filename)
        plt.close()
        #log plots
        sns.distplot(np.log10(max_dis), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"}, kde=False, label="GP")
        sns.distplot(np.log10(gpemax_dis), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"}, kde=False, label="GPe")
        sns.distplot(np.log10(gpimax_dis),hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"}, kde=False, label="GPi")
        max_filename = ('%s/logmaxsholl_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('log of bins of maxima in sholl analysis in %.4s' % ct_dict_GP[celltype])
        plt.xlabel('log of pathlength in µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(max_filename)
        plt.close()
        sns.distplot(np.log10(furthestbin_dis), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"}, kde=False, label="GP")
        sns.distplot(np.log10(gpefurthestbin_dis), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"}, kde=False, label="GPe")
        sns.distplot(np.log10(gpifurthestbin_dis), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"}, kde=False, label="GPi")
        furthest_filename = ('%s/logfurthestsholl_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('log of bins of longest dendrites in sholl analysis in %s' % ct_dict_GP[celltype])
        plt.xlabel('log of pathlength in µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(furthest_filename)
        plt.close()
        sns.distplot(np.log10(avg_dis), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"}, kde=False, label="GP")
        sns.distplot(np.log10(gpeavg_dis), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"}, kde=False, label="GPe")
        sns.distplot(np.log10(gpiavg_dis), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"}, kde=False, label="GPi")
        avg_filename = ('%s/logavgsholl_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('log of average in sholl analysis in %s' % ct_dict_GP[celltype])
        plt.xlabel('log of pathlength in 10^ µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()
        sns.distplot(np.log10(var_dis), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"}, kde=False, label="GP")
        sns.distplot(np.log10(gpevar_dis), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"}, kde=False, label="GPe")
        sns.distplot(np.log10(gpivar_dis), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"}, kde=False, label="GPi")
        avg_filename = ('%s/logvarsholl_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('log of variance in sholl analysis in %s' % ct_dict_GP[celltype])
        plt.xlabel('log of variance in 10^')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()
        sns.distplot(np.log10(sum_b), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"}, kde=False, label="GP")
        sns.distplot(np.log10(gpesum_b), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"}, kde=False, label="GPe")
        sns.distplot(np.log10(gpisum_b), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"}, kde=False, label="GPi")
        sumb_filename = ('%s/logsumb_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('log of sum of branching points in %.4s' % ct_dict_GP[celltype])
        plt.xlabel("log of number of branching pointsin 10^")
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(sumb_filename)
        plt.close()
        sns.distplot(np.log10(sum_b), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, label="GP",norm_hist=True)
        sns.distplot(np.log10(gpesum_b),
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"}, kde=False,
                     label="GPe",norm_hist=True)
        sns.distplot(np.log10(gpisum_b),
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"}, kde=False,
                     label="GPi",norm_hist=True)
        sumb_filename = ('%s/rellogsumb_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('log of sum of branching points in %.4s' % ct_dict_GP[celltype])
        plt.xlabel("log of number of branching pointsin 10^")
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(sumb_filename)
        plt.close()
        sns.distplot(np.log10(sum_t), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"}, kde=False, label="GP")
        sns.distplot(np.log10(gpesum_t), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"}, kde=False, label="GPe")
        sns.distplot(np.log10(gpisum_t), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"}, kde=False, label="GPi")
        sumt_filename = ('%s/logsumt_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('log of sum of terminating points in %.4s' % ct_dict_GP[celltype])
        plt.xlabel("log of number of terminating points in 10^")
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(sumt_filename)
        plt.close()

        plt.scatter(x=gpesum_b, y=gpemax_dis, c="mediumorchid", label="GPe")
        plt.scatter(x=gpisum_b, y=gpimax_dis, c="springgreen", label="GPi")
        sc_filename = ('%s/scmaxdissumb_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('maximum of sholl vs sum of branching points in %.4s' % ct_dict_GP[celltype])
        plt.xlabel('sum of branching points')
        plt.ylabel('sholl max. pathlength in µm')
        plt.legend()
        plt.savefig(sc_filename)
        plt.close()
        plt.scatter(x=np.log10(gpesum_b), y=np.log10(gpemax_dis), c="mediumorchid", label="GPe")
        plt.scatter(x=np.log10(gpisum_b), y=np.log10(gpimax_dis), c="springgreen", label="GPi")
        sc_filename = ('%s/logscmaxdissumb_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('log maximum of sholl vs sum of branching points in %.4s' % ct_dict_GP[celltype])
        plt.xlabel('log sum of branching points in 10^')
        plt.ylabel('log of sholl max. pathlength in µm in 10^')
        plt.legend()
        plt.savefig(sc_filename)
        plt.close()

        plt.scatter(x=gpesum_b, y=gpefurthestbin_dis, c="mediumorchid", label="GPe")
        plt.scatter(x=gpisum_b, y=gpifurthestbin_dis, c="springgreen", label="GPi")
        sc_filename = ('%s/scfudissumb_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('furthest sholl vs sum of branching points in %.4s' % ct_dict_GP[celltype])
        plt.xlabel('sum of branching points')
        plt.ylabel('furthest sholl pathlength in µm')
        plt.legend()
        plt.savefig(sc_filename)
        plt.close()
        plt.scatter(x=np.log10(gpesum_b), y=np.log10(gpefurthestbin_dis), c="mediumorchid", label="GPe")
        plt.scatter(x=np.log10(gpisum_b), y=np.log10(gpifurthestbin_dis), c="springgreen", label="GPi")
        sc_filename = ('%s/logscfudissumb_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('log furthest of sholl vs sum of branching points in %.4s' % ct_dict_GP[celltype])
        plt.xlabel('log sum of branching points in 10^')
        plt.ylabel('furthest sholl pathlength in 10^ µm')
        plt.legend()
        plt.savefig(sc_filename)
        plt.close()

        plt.scatter(x=gpeavg_dis, y=gpevar_dis, c="mediumorchid", alpha=0.3, label="GPe")
        plt.scatter(x=gpiavg_dis, y=gpivar_dis, c="springgreen", alpha=0.3, label="GPi")
        sc_filename = ('%s/scvaravg_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('average of sholl vs variance in %.4s' % ct_dict_GP[celltype])
        plt.xlabel('average of sholl in pathlength in µm')
        plt.ylabel('variance of sholl in µm')
        plt.legend()
        plt.savefig(sc_filename)
        plt.close()
        plt.scatter(x=np.log10(gpeavg_dis), y=np.log10(gpevar_dis), c="mediumorchid", alpha=0.3, label="GPe")
        plt.scatter(x=np.log10(gpiavg_dis), y=np.log10(gpivar_dis), c="springgreen", alpha=0.3, label="GPi")
        sc_filename = ('%s/logscvaravg_%s_%s_n.png' % (f_name, ct_dict_GP[celltype], comp_dict[compartment]))
        plt.title('log average of sholl vs variance in %.4s' % ct_dict_GP[celltype])
        plt.xlabel('log average of sholl in pathlength in 10^ µm')
        plt.ylabel('log variance of sholl in 10^ µm')
        plt.legend()
        plt.savefig(sc_filename)
        plt.close()

        plottime = time.time() - celltime
        print("%.2f sec for plotting" % plottime)
        time_stamps.append(time.time())
        step_idents.append('plots per celltype and compartment')
        log.info("finished plots for %s in compartment: %s" % (ct_dict[celltype], comp_dict[compartment]))
        return sholl_dict, sum_t, sum_b

    gp_sholl_dict, gp_sb, gp_st = ct_sholl_analysis(ssd,r= 10000, celltype=[6,7], compartment=0)
    gp_sholl_dicta, gp_sba, gp_sta = ct_sholl_analysis(ssd, celltype=[6,7], compartment=1, r=10000)

    raise ValueError
