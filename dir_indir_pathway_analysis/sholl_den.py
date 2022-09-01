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
    from syconn.handler.basics import load_pkl2obj
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

    def sholl_prep(sso, compartment):
        """
        For preperation of the sholl analysis function computes starting value from networkx which is the amount of desired compartments (default: dendrites) emerging from the soma.
        It also computes the shortestpath2soma for nodes that are not degree = 2 e.g. are branching points or terminating points and saves them in array for sholl analysis.
               compartment = compartment that should be looked at, default = 0 (dendrites) to look at dendritic tree, Spine predictions: 0: neck, 1: head, 2: shaft, 3: other(dendrite, axon)
        :return: return shortestpath, degrees
        """
        #compute starting points for dendrite or axon
        #remove all compartments with value 1 = axon, value 2 = soma: (see function majority_vote_compartments in github)
        # if want to look at axons: remove values 0 = dendrite and 2 = soma
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
                if d["spiness"] == 0 or d["spiness"] == 1:
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


    def ct_sholl_analysis(ssd, celltype, compartment, r = 30000):
        """
        computes sholl_analysis per celltype and per cell with shortest_paths and degrees of each node (degree != 2).
        Function uses path_length in different intervals to compute length of dendritic tree and produce values for a Sholl profile. It counts the amount of branching points, termination points (see Gutierrez and Davies, 2017)
        via edges in networkx for intervals with difference r (default 30 µm = 30000 nm).
        Branching points are points that have at least 3 edges (3 edges = binary branching point = 1, more than three edges = n-1 branching points), Termination points are points with only one edge.
        Start values are emerging dendrites from soma which is the amount of connected compartments after removal of soma, axons.
        :param ssd: SuperSegmentationDataset
        :param j0126: STN=0, modulatory=1, MSN=2, LMAN=3, HVC=4, GP=5, INT=6
                r = length of intersections
        :return: sholl_dict, bt_dict, avg_b, med_b = mean/median amount of branching points per celltype, avg_t, med_t = mean/median amount of terminating points per celltype
        """
        #per celltyype function
        # for cell == celltype:
            #sholl_analysis(cell)
        start = time.time()
        f_name = "u/arother/test_folder/201203_sholl_analysisMSN_j0251v3_r5000"
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('sholl_analysis', log_dir=f_name + '/logs/')
        time_stamps = [time.time()]
        step_idents = ['t-0']
        #cellids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == celltype]
        sholl_dict = {}
        comp_dict = {1: 'axons', 0: 'dendrites'}
        ct_dict = {0: "STN", 1: "MOD", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "GP", 6: "INT"}
        cellids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%3s_arr.pkl" % ct_dict[celltype])
        log.info('Step 1/2 generating sholl per cell')
        for pi, cell in enumerate(ssd.get_super_segmentation_object(cellids)):
            cell.load_skeleton()
            """
            axoness = cell.skeleton["axoness_avg10000"]
            axoness[axoness == 3] = 1
            axoness[axoness == 4] = 1
            unique_preds = np.unique(axoness)
            if not (0 in unique_preds and 1 in unique_preds and 2 in unique_preds):
                continue
            """
            x0, shortest_paths, degrees = sholl_prep(cell, compartment=compartment)
            # rmax dependent on longest shortest path
            if len(shortest_paths) == 0:
                continue
            rmax = np.max(shortest_paths)
            # intersections for sholl analysis
            inter_amount = int(rmax / r) + 1
            bt_count = np.zeros((inter_amount, 3))  # [0] = distance to soma in intersections, [1] = branching points, [2] = terminal points
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
                avg_b_am[ii] = sum_b[ii]/X0
                avg_t_am[ii] = sum_t[ii]/X0
        zsum_b = np.where(sum_b <= 0)
        zsum_t = np.where(sum_t <= 0)
        zmax_dis = np.where(max_dis <= 0)
        zavg_dis =  np.where(avg_dis <= 0)
        zvar_dis = np.where(var_dis <= 0)
        zfurthest = np.where(furthestbin_dis <= 0)
        arr_inds = np.unique(np.hstack([zsum_b, zsum_t, zmax_dis, zavg_dis, zvar_dis, zfurthest]))
        sum_b = np.delete(sum_b, arr_inds)
        sum_t = np.delete(sum_t, arr_inds)
        max_dis = np.delete(max_dis, arr_inds)
        avg_dis = np.delete(avg_dis, arr_inds)
        var_dis = np.delete(var_dis, arr_inds)
        furthestbin_dis = np.delete(furthestbin_dis, arr_inds)
        if compartment == 0:
            avg_b_am = np.delete(avg_b_am, arr_inds)
            avg_t_am = np.delete(avg_t_am, arr_inds)
            sns.distplot(avg_b_am, color="blue", kde=False)
            avb_filename = ('%s/avgb_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
            plt.title('average amount of branching points per dendrite in %.4s' % ct_dict[celltype])
            plt.xlabel('amount of branching points/ dendrite')
            plt.ylabel('count of cells')
            plt.savefig(avb_filename)
            plt.close()
            sns.distplot(avg_t_am, color="blue", kde=False)
            avt_filename = ('%s/avt_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
            plt.title('average amount of terminating points per dendrite  in %.4s' % ct_dict[celltype])
            plt.xlabel('amount of terminating points per dendrite')
            plt.ylabel('count of cells')
            plt.savefig(avt_filename)
            plt.close()
            #log plots
            sns.distplot(np.log10(avg_b_am), color="blue", kde=False)
            avb_filename = ('%s/logavgb_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
            plt.title('log of average amount of branching points per dendrite in %.4s' % ct_dict[celltype])
            plt.xlabel('log of amount of branching points/ dendrite in 10^')
            plt.ylabel('count of cells')
            plt.savefig(avb_filename)
            plt.close()
            sns.distplot(np.log10(avg_t_am), color="blue", kde=False)
            avt_filename = ('%s/logavt_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
            plt.title('log of average amount of terminating points per dendrite  in %.4s' % ct_dict[celltype])
            plt.xlabel('log of amount of terminating points per dendrite in 10^')
            plt.ylabel('count of cells')
            plt.savefig(avt_filename)
            plt.close()
            plt.scatter(x=avg_b_am, y=sum_b, c="skyblue")
            sc_filename = ('%s/scavgbsumb_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
            plt.title('sum of branching points vs branching points per dendrite  in %.4s' % ct_dict[celltype])
            plt.xlabel('amount of branching points per dendrite in 10^')
            plt.ylabel('sum of branching points')
            plt.savefig(sc_filename)
            plt.close()
            plt.scatter(x=np.log10(avg_b_am), y=np.log10(sum_b), c="skyblue")
            sc_filename = ('%s/logscavgbsumb_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
            plt.title('log sum of branching points vs branching points per dendrite  in %.4s' % ct_dict[celltype])
            plt.xlabel('log amount of branching points per dendrite in 10^')
            plt.ylabel('log sum of branching points in 10^')
            plt.savefig(sc_filename)
            plt.close()
            plt.scatter(x=avg_b_am, y=max_dis, c="skyblue")
            sc_filename = ('%s/scmaxdisavgb_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
            plt.title('maximum of sholl vs branching points per dendrite  in %.4s' % ct_dict[celltype])
            plt.xlabel('amount of branching points per dendrite')
            plt.ylabel('sholl max. pathlength in µm')
            plt.savefig(sc_filename)
            plt.close()
            plt.scatter(x=np.log10(avg_b_am), y=np.log10(max_dis), c="skyblue")
            sc_filename = ('%s/logscmaxdisavgb_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
            plt.title('log maximum of sholl vs branching points per dendrite  in %.4s' % ct_dict[celltype])
            plt.xlabel('log amount of branching points per dendritein 10^')
            plt.ylabel('log of sholl max. pathlength in µm')
            plt.savefig(sc_filename)
            plt.close()
        sns.distplot(max_dis, color="blue", kde=False)
        max_filename = ('%s/maxsholl_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('bins of maxima in sholl analysis in %.4s' % ct_dict[celltype])
        plt.xlabel('pathlength in µm')
        plt.ylabel('count of cells')
        plt.savefig(max_filename)
        plt.close()
        sns.distplot(furthestbin_dis, color="blue", kde=False)
        furthest_filename = ('%s/furthestsholl_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('bins of longest dendrites in sholl analysis in %s' % ct_dict[celltype])
        plt.xlabel('pathlength in µm')
        plt.ylabel('count of cells')
        plt.savefig(furthest_filename)
        plt.close()
        sns.distplot(avg_dis, color="blue", kde=False)
        avg_filename = ('%s/avgsholl_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('average in sholl analysis in %s' % ct_dict[celltype])
        plt.xlabel('pathlength in µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()
        sns.distplot(var_dis, color="blue", kde=False)
        avg_filename = ('%s/varsholl_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('variance in sholl analysis in %s' % ct_dict[celltype])
        plt.xlabel('variance')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()
        sns.distplot(sum_b, color="blue", kde=False)
        sumb_filename = ('%s/sumb_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('sum of branching points in %.4s' % ct_dict[celltype])
        plt.xlabel("number of branching points")
        plt.ylabel('count of cells')
        plt.savefig(sumb_filename)
        plt.close()
        sns.distplot(sum_t, color="blue", kde=False)
        sumt_filename = ('%s/sumt_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('sum of terminating points in %.4s' % ct_dict[celltype])
        plt.xlabel("number of terminating points")
        plt.ylabel('count of cells')
        plt.savefig(sumt_filename)
        plt.close()
        #log plots
        sns.distplot(np.log10(max_dis), color="blue", kde=False)
        max_filename = ('%s/logmaxsholl_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of bins of maxima in sholl analysis in %.4s' % ct_dict[celltype])
        plt.xlabel('log of pathlength in µm')
        plt.ylabel('count of cells')
        plt.savefig(max_filename)
        plt.close()
        sns.distplot(np.log10(furthestbin_dis), color="blue", kde=False)
        furthest_filename = ('%s/logfurthestsholl_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of bins of longest dendrites in sholl analysis in %s' % ct_dict[celltype])
        plt.xlabel('log of pathlength in 10^ µm')
        plt.ylabel('count of cells')
        plt.savefig(furthest_filename)
        plt.close()
        sns.distplot(np.log10(avg_dis), color="blue", kde=False)
        avg_filename = ('%s/logavgsholl_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of average in sholl analysis in %s' % ct_dict[celltype])
        plt.xlabel('log of pathlength in 10^ µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()
        sns.distplot(np.log10(var_dis), color="blue", kde=False)
        avg_filename = ('%s/logvarsholl_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of variance in sholl analysis in %s' % ct_dict[celltype])
        plt.xlabel('log of variance in 10^')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()
        sns.distplot(np.log10(sum_b), color="blue", kde=False)
        sumb_filename = ('%s/logsumb_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of sum of branching points in %.4s' % ct_dict[celltype])
        plt.xlabel("log of number of branching points in 10^")
        plt.ylabel('count of cells')
        plt.savefig(sumb_filename)
        plt.close()
        sns.distplot(np.log10(sum_t), color="blue", kde=False)
        sumt_filename = ('%s/logsumt_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log of sum of terminating points in %.4s' % ct_dict[celltype])
        plt.xlabel("log of number of terminating points in 10^")
        plt.ylabel('count of cells')
        plt.savefig(sumt_filename)
        plt.close()

        plt.scatter(x=sum_b, y=max_dis, c="skyblue")
        sc_filename = ('%s/scmaxdissumb_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('maximum of sholl vs sum of branching points in %.4s' % ct_dict[celltype])
        plt.xlabel('sum of branching points')
        plt.ylabel('sholl max. pathlength in µm')
        plt.savefig(sc_filename)
        plt.close()
        plt.scatter(x=np.log10(sum_b), y=np.log10(max_dis), c="skyblue")
        sc_filename = ('%s/logscmaxdissumb_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log maximum of sholl vs sum of branching points in %.4s' % ct_dict[celltype])
        plt.xlabel('log sum of branching points in 10^')
        plt.ylabel('log of sholl max. pathlength in 10^ µm')
        plt.savefig(sc_filename)
        plt.close()

        plt.scatter(x=sum_b, y=furthestbin_dis, c="skyblue")
        sc_filename = ('%s/scfudissumb_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('furthest sholl vs sum of branching points in %.4s' % ct_dict[celltype])
        plt.xlabel('sum of branching points')
        plt.ylabel('furthest sholl pathlength in µm')
        plt.savefig(sc_filename)
        plt.close()
        plt.scatter(x=np.log10(sum_b), y=np.log10(furthestbin_dis), c="skyblue")
        sc_filename = ('%s/logscfudissumb_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log furthest of sholl vs sum of branching points in %.4s' % ct_dict[celltype])
        plt.xlabel('log sum of branching points in 10^')
        plt.ylabel('furthest sholl pathlength in µm')
        plt.savefig(sc_filename)
        plt.close()

        plt.scatter(x=avg_dis, y=var_dis, c="skyblue")
        sc_filename = ('%s/scvaravg_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('average of sholl vs variance in %.4s' % ct_dict[celltype])
        plt.xlabel('average of sholl in pathlength in µm')
        plt.ylabel('variance of sholl in µm')
        plt.savefig(sc_filename)
        plt.close()
        plt.scatter(x=np.log10(avg_dis), y=np.log10(var_dis), c="skyblue")
        sc_filename = ('%s/logscvaravg_%s_%s_n.png' % (f_name, ct_dict[celltype], comp_dict[compartment]))
        plt.title('log average of sholl vs variance in %.4s' % ct_dict[celltype])
        plt.xlabel('log average of sholl in pathlength in 10^ µm')
        plt.ylabel('log variance of sholl in 10^ µm')
        plt.savefig(sc_filename)
        plt.close()
        plottime = time.time() - celltime
        print("%.2f sec for plotting" % plottime)
        time_stamps.append(time.time())
        step_idents.append('plots per celltype and compartment')
        log.info("finished plots for %s in compartment: %s" % (ct_dict[celltype], comp_dict[compartment]))
        return sholl_dict, sum_t, sum_b

    msn_sholl_dict, msn_sb, msn_st = ct_sholl_analysis(ssd, celltype=2, compartment=0, r= 5000)
    msn_sholl_dicta, msn_sba, msn_sta = ct_sholl_analysis(ssd, celltype=2, compartment=1, r=5000)
    #gp_sholl_dict, gp_avb, gp_av_t, gp_m_b, gp_m_t = ct_sholl_analysis(ssd, celltype=5)





    raise ValueError
