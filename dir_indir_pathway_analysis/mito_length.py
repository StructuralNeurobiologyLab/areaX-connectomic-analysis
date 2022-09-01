#script to compare length of mitochondria as in paper chandra et al., 2019
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
    global_params.wd = "/wholebrain/scratch/arother/assembled_core_relabeled_base_merges_relabeled_to_v4b_base_20180214_full_agglo_cbsplit_with_reconnects_no_soma_merger_manual_edges_removed"

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv",working_dir=global_params.config.working_dir)

    def mito_prox_dis(sso, compartment = 0, radius = 50):
        """
        compartments: 0 = dendrite, 1 = axon, soma = 2
        """
        if compartment == 0:
            unwanted_compartment = 1
        elif compartment == 1:
            unwanted_compartment = 0
        else:
            print('unknown compartment value %.i. Compartment has to be either 0 or 1' % compartment)
            raise ValueError
        sso.load_skeleton_kimimaro(skel="knx_skeleton")
        sso.load_skeleton_kimimaro(skel="knx_skeleton_dict")
        g = sso.knx_skeleton
        desired_compartment_graph = g.copy()
        for i, d in enumerate(sso.knx_skeleton_dict["axoness_avg10000"]):
            if d == 2 or d == unwanted_compartment:
                desired_compartment_graph.remove_node(i)
        prox_list = []
        dist_list = []
        prox_d = []
        dist_d = []
        mis = sso.mis
        for mi in mis:
            #from syconn.reps.super_segmentation_object attr_for_coors
            kdtree = scipy.spatial.cKDTree(sso.skeleton["nodes"] * sso.scaling)
            close_node_ids = kdtree.query_ball_point(mi.bounding_box, radius)
            axoness_list = []
            for id in close_node_ids:
                axoness_list.append(sso.knx_skeleton_dict["axoness_avg10000"])
            axo = np.array(axoness_list)
            if np.all(axo) == unwanted_compartment or np.all(axo) == 2:
                continue
            if len(np.where(axo == compartment)) < len(axo)/2:
                continue
            dists = sso.shortestpath2soma(mi.mesh_bb)
            dist1 = dists[0]
            dist2 = dists[1]
            if dist1 <= dist2:
                dist = dist1
            else:
                dist = dist2
            if dist <= 50000:
                prox_list.append(mi.id)
                prox_d.append(mi.mesh_size)
            elif dist >= 100000:
                dist_list.append(mi.id)
                dist_d.append(mi.mesh_size)
        dist_id_array = np.array(dist_list)
        prox_id_array = np.array(prox_list)
        prox_d_array = np.array(prox_d)
        dist_d_array = np.array(dist_d)

        return prox_d_array, dist_d_array, prox_id_array, dist_d_array

    def map_axoness2skel(sso):
        from syconn.reps.segmentation_helper import majorityvote_skeleton_property_kimimaro
        try:
            sso.cnn_axoness2skel_kimimaro()
        except ValueError:
            return
        if not sso.knx_skeleton_dict is None or len(sso.knx_skeleton_dict["nodes"]) >= 2:
            # vertex predictions
            node_preds = sso.knx_skeleton_dict["axoness"]
            # perform average only on axon dendrite and soma predictions
            nodes_ax_den_so = np.array(node_preds, dtype=np.int)
            # set en-passant and terminal boutons to axon class.
            nodes_ax_den_so[nodes_ax_den_so == 3] = 1
            nodes_ax_den_so[nodes_ax_den_so == 4] = 1
            sso.knx_skeleton_dict[pred_key] = node_preds

            # average along skeleton, stored as: "{}_avg{}".format(pred_key, max_dist)
            majorityvote_skeleton_property_kimimaro(sso, prop_key=pred_key,
                                                    max_dist=max_dist)
            # suffix '_avg{}' is added by `_average_node_axoness_views`
            nodes_ax_den_so = sso.knx_skeleton_dict["{}_avg{}".format(pred_key, max_dist)]
            # recover bouton predictions within axons and store smoothed result
            nodes_ax_den_so[(node_preds == 3) & (nodes_ax_den_so == 1)] = 3
            nodes_ax_den_so[(node_preds == 4) & (nodes_ax_den_so == 1)] = 4
            sso.knx_skeleton_dict["{}_avg{}".format(pred_key, max_dist)] = nodes_ax_den_so

            # will create a compartment majority voting after removing all soma nodes
            # the restul will be written to: ``ax_pred_key + "_comp_maj"``
            majority_vote_compartments_kimimaro(sso, "{}_avg{}".format(pred_key, max_dist))
            nodes_ax_den_so = sso.knx_skeleton_dict["{}_avg{}_comp_maj".format(pred_key, max_dist)]
            # recover bouton predictions within axons and store majority result
            nodes_ax_den_so[(node_preds == 3) & (nodes_ax_den_so == 1)] = 3
            nodes_ax_den_so[(node_preds == 4) & (nodes_ax_den_so == 1)] = 4
            sso.knx_skeleton_dict["{}_avg{}_comp_maj".format(pred_key, max_dist)] = nodes_ax_den_so
            sso.save_skeleton_kimimaro(skel="knx_skeleton_dict")


    def ct_mito_analysis(ssd, celltype, compartment=0):
        """
        analysis of mitochondria length in proximal and distal dendrites
        :param ssd: SuperSegmentationDataset
        :param celltype: 0:"STN", 1:"DA", 2:"MSN", 3:"LMAN", 4:"HVC", 5:"GP", 6:"FS", 7:"TAN", 8:"INT"
                r = length of intersections
        :return: sholl_dict, bt_dict, avg_b, med_b = mean/median amount of branching points per celltype, avg_t, med_t = mean/median amount of terminating points per celltype
        """
        # per celltype function
        # for cell == celltype:
        # sholl_analysis(cell)
        cellids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == celltype]
        prox_mito_dict = dict()
        dist_mito_dict = dict()
        mean_prox_array = np.array((len(cellids)))
        med_prox_array = np.array((len(cellids)))
        mean_dist_array = np.array((len(cellids)))
        med_dist_array = np.array((len(cellids)))
        comp_dict = {1: 'axons', 0: 'dendrites'}
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "GP", 6: "FS", 7: "TAN", 8: "INT"}
        if not os.path.exists("mito"):
            os.mkdir("mito")
        for i, cell in enumerate(ssd.get_super_segmentation_object(cellids)):
            cell.load_skeleton_kimimaro(skel="knx_skeleton_dict")
            try:
                unique_preds = np.unique(cell.knx_skeleton_dict['axoness_avg10000'])
            except KeyError:
                continue
            if not (0 in unique_preds and (1 in unique_preds or 3 in unique_preds or 4 in unique_preds) and 2 in unique_preds):
                continue
            prox_mito_dict[cell.id]["diameter"], dist_mito_dict[cell.id]["diameter"], prox_mito_dict[cell.id]["mi_id"], dist_mito_dict[cell.id]["mi_id"] = mito_prox_dis(cell, compartment)
            prox_ds = prox_mito_dict[cell.id]["diameter"]
            dist_ds = dist_mito_dict[cell.id]["diameter"]
            # plot mitos_per cell
            columns = ['pathlength in µm', ('count of %.9s' % comp_dict[compartment])]
            foldername = 'mito/mito_percell_%.4s_n' % ct_dict[celltype]
            if not os.path.exists(foldername):
                os.mkdir(foldername)
            sns.distplot(prox_ds)
            plt.title('mitochondrium diameter %.9s in %.4s' % (comp_dict[compartment], ct_dict[celltype]))
            plt.xlabel('diameter in nm')
            plt.ylabel('count of cells')
            plt.savefig('%.s/prox_mito_%.i_%.4s_%.4s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
            plt.close()
            sns.distplot(dist_ds)
            plt.title('mitochondrium diameter %.9s in %.4s' % (comp_dict[compartment], ct_dict[celltype]))
            plt.xlabel('diameter in nm')
            plt.ylabel('count of cells')
            plt.savefig(
                '%.s/dist_mito_%.i_%.4s_%.4s.png' % (foldername, cell.id, ct_dict[celltype], comp_dict[compartment]))
            plt.close()
            med_prox = np.median(prox_ds)
            mean_prox = np.mean(prox_ds)
            med_dist = np.median(dist_ds)
            mean_dist = np.mean(dist_ds)
            med_prox_array[i] = med_prox
            mean_prox_array[i] = mean_prox
            mean_dist_array[i] = mean_dist
            med_dist_array[i] = med_dist
            prox_mito_dict[cell.id]["mean"] = mean_prox
            prox_mito_dict[cell.id]["median"] = med_prox
            dist_mito_dict[cell.id]["mean"] = mean_dist
            dist_mito_dict[cell.id]["median"] = med_dist
        # graphs per celltype
        raise ValueError
        sns.countplot(x = med_dist_array, color = "blue")
        med_dist_filename = ('mito/median_dist_%.4s_%.4s_n.png' % (ct_dict[celltype], comp_dict[compartment]))
        plt.title('medium mitochondrium diameter in distal %s in %.4s' % (comp_dict[compartment],ct_dict[celltype]))
        plt.xlabel('length in nm')
        plt.ylabel('count of cells')
        plt.savefig(med_dist_filename)
        plt.close()
        sns.countplot(x = mean_dist_array, color = "blue")
        mean_dist_filename = ('mito/mean_dist_%.4s_%.4s_n.png' % (ct_dict[celltype], comp_dict[compartment]))
        plt.title('mean mitochondrium diameter in distal %s in %.4s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('length in nm')
        plt.ylabel('count of cells')
        plt.savefig(mean_dist_filename)
        plt.close()
        sns.countplot(x = mean_prox_array, color = "blue")
        med_prox_filename = ('mito/median_prox_%.4s_%.4s_n.png' % (ct_dict[celltype], comp_dict[compartment]))
        plt.title('medium mitochondrium diameter in proximal %s in %.4s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('length in nm')
        plt.ylabel('count of cells')
        plt.savefig(med_prox_filename)
        plt.close()
        sns.countplot(x=mean_prox_array, color="blue")
        mean_prox_filename = ('mito/mean_prox_%.4s_%.4s_n.png' % (ct_dict[celltype], comp_dict[compartment]))
        plt.title('mean mitochondrium diameter in proximal %s in %.4s' % (comp_dict[compartment], ct_dict[celltype]))
        plt.xlabel('length in nm')
        plt.ylabel('count of cells')
        plt.savefig(mean_dist_filename)
        plt.close()
        return prox_mito_dict, dist_mito_dict

    gp_den_prox_dict, gp_den_dist_dict = ct_mito_analysis(ssd, celltype=5)
    gp_ax_prox_dict, gp_ax_dist_dict = ct_mito_analysis(ssd, celltype=5, compartment = 1)
    msn_den_prox_dict, msn_den_dist_dict = ct_mito_analysis(ssd, celltype=2)
    msn_ax_prox_dict, msn_ax_dist_dict = ct_mito_analysis(ssd, celltype=2, compartment = 1)
    raise ValueError
