if __name__ == '__main__':
    from syconn import global_params
    from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
    from syconn.reps.segmentation import SegmentationDataset
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import networkx as nx
    import os as os
    import time
    from syconn.handler.config import initialize_logging
    from syconn.handler.basics import load_pkl2obj
    from tqdm import tqdm
    #global_params.wd = "/ssdscratch/pschuber/songbird/j0126/assembled_core_relabeled_base_merges_relabeled_to_v4b_base_20180214_full_agglo_cbsplit_with_reconnects_no_soma_merger_manual_edges_removed"
    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv",working_dir=global_params.config.working_dir)


    def ct_fsholl_analysis(ssd, celltype):
        """
        computes similar dsata than sholl analysis but not with bins. Computes overall dendritic or axonic length, number of primary dendrites, spine density, branching point density.
        :param ssd: SuperSegmentationDataset
        :param j0126: STN=0, modulatory=1, MSN=2, LMAN=3, HVC=4, GP=5, INT=6
                r = length of intersections
                j0251:ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7:"GPi", 8:"FS", 9:"LTS", 10:"NGF"}
                partner_axoness: 0 = dendrite, 1 = axon, 2 = soma, 3 = en-passant bouton, 4 = terminal bouton
                partner spiness: 0: dendritic shaft, 1: spine head, 2: spine neck
        :return: spine_density, branching_point_density, primary dendrites, overall_dendritic length
        """
        start = time.time()
        f_name = "u/arother/test_folder/210531_fast_sholl_j0251v3_GPhp_completedendrites_newbranchpoints"
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('sholl_analysis', log_dir=f_name + '/logs/')
        time_stamps = [time.time()]
        step_idents = ['t-0']
        comp_dict = {1: 'axons', 0: 'dendrites'}
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7:"GPi", 8:"FS", 9:"LTS", 10:"NGF"}
        ct_dict_GP = {6: "GP"}
        #cellids = load_pkl2obj("/wholebrain/scratch/arother/j0256_prep/full_%3s_arr.pkl" % ct_dict[celltype])
        #gpeids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%3s_arr.pkl" % ct_dict[celltype[0]])
        #gpiids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%3s_arr.pkl" % ct_dict[celltype[1]])
        gpeids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/handpicked_%3s_arr.pkl" % ct_dict[celltype[0]])
        gpiids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/handpicked_%3s_arr.pkl" % ct_dict[celltype[1]])
        cellids = np.hstack(np.array([gpeids, gpiids]))
        celltype = celltype[0]
        dataset_voxelsize = np.array([27119, 27350, 15494])
        dataset_nm = dataset_voxelsize * ssd.scaling
        log.info('Step 1/2 generating sholl per cell')
        average_dendrite_length = np.zeros(len(cellids))
        average_dendrite_length_nocutoffs = np.zeros(len(cellids))
        primary_dendrite_amount = np.zeros(len(cellids))
        branching_point_density = np.zeros(len(cellids))
        terminating_point_density = np.zeros(len(cellids))
        spine_density = np.zeros(len(cellids))
        overall_dendrite_length = np.zeros(len(cellids))
        for pi, cell in enumerate(tqdm(ssd.get_super_segmentation_object(cellids))):
            cell.load_skeleton()
            #nonaxon_inds = np.nonzero(cell.skeleton["axoness_avg10000"] != 1)[0]
            nondendrite_inds = np.nonzero(cell.skeleton["axoness_avg10000"] != 0)[0]
            spine_shaftinds = np.nonzero(cell.skeleton["spiness"] == 0)[0]
            spine_otherinds = np.nonzero(cell.skeleton["spiness"] == 3)[0]
            spine_headinds = np.nonzero(cell.skeleton["spiness"] == 1)[0]
            spine_neckinds = np.nonzero(cell.skeleton["spiness"] == 2)[0]
            nonspine_inds = np.hstack([spine_shaftinds, spine_otherinds])
            spine_inds = np.hstack([spine_headinds, spine_neckinds])
            g = cell.weighted_graph(add_node_attr=('axoness_avg10000', "spiness"))
            dendrite_graph = g.copy()
            dendrite_graph.remove_nodes_from(nondendrite_inds)
            den_shaft_graph = dendrite_graph.copy()
            #den_shaft_graph.remove_nodes_from(spine_inds)
            spine_graph = dendrite_graph.copy()
            spine_graph.remove_nodes_from(nonspine_inds)
            spine_amount_skeleton = len(list(nx.connected_component_subgraphs(spine_graph)))
            primary_den_cell = len(list(nx.connected_component_subgraphs(dendrite_graph)))
            den_length = den_shaft_graph.size(weight="weight") / 1000  # µm
            average_den_length_cell = den_length/primary_den_cell
            dendrite_graphs = list(nx.connected_component_subgraphs(dendrite_graph))
            den_length_nocutoffs = 0
            no_cutoffs = []
            for_bp = 0
            for di, dendrite in enumerate(dendrite_graphs.copy()):
                #dendrite.remove_nodes_from(spine_inds)
                den_size = dendrite.size(weight = "weight")/ 1000
                if den_size < 1:
                    continue
                for_bp += 1
                if den_size/average_den_length_cell >= 0.25:
                    den_length_nocutoffs += den_size
                    no_cutoffs.append(di)
            average_den_length_nocutoffs_cell = den_length_nocutoffs/len(no_cutoffs)
            degrees = np.zeros(len(den_shaft_graph.nodes()))
            for ix, node_id in enumerate(den_shaft_graph.nodes()):
                degrees[ix] = g.degree[node_id]
            # remove all branching points and count amount of connected components with length of 10 µm
            branching_point_inds = np.nonzero(degrees >= 3)[0]
            branching_graph = den_shaft_graph.copy()
            branching_point_amount = 0
            prev_branches = for_bp
            for bi in branching_point_inds:
                branching_graph.remove_nodes_from([bi])
                connected_branch_pieces = list(nx.connected_component_subgraphs(branching_graph))
                branches = 0
                for branch_piece in connected_branch_pieces:
                    if branch_piece.size(weight="weights") / 1000 >= 1:
                        branches += 1
                if branches == 0:
                    break
                new_branches = branches - prev_branches
                branching_point_amount += new_branches
                prev_branches += new_branches
            terminating_point_amout = len(degrees[degrees == 1])
            branching_point_density_cell = branching_point_amount/den_length
            terminating_point_density_cell = terminating_point_amout/ den_length
            spine_density_cell = spine_amount_skeleton/ den_length
            average_dendrite_length[pi] = average_den_length_cell
            average_dendrite_length_nocutoffs[pi] = average_den_length_nocutoffs_cell
            spine_density[pi] = spine_density_cell
            branching_point_density[pi] = branching_point_density_cell
            terminating_point_density[pi] = terminating_point_density_cell
            primary_dendrite_amount[pi] = primary_den_cell
            overall_dendrite_length[pi] = den_length
            #in the future also add axon length but atm axon either fragmented or false positives
            #find out how to get rid of potential false positives first
            '''
            axon_graph = g.copy()
            axon_graph.remove_nodes_from(nonaxon_inds)
            if len(list(nx.connected_component_subgraphs(axon_graph))) > 1:
                possible_axon_list = list(list(nx.connected_component_subgraphs(axon_graph)))
                possible_axon_length = np.zeros(len(possible_axon_list))
                for ia, possible_axon in enumerate(possible_axon_list):
                    possible_axon_length[ia] = len(possible_axon.nodes)
                longest_axon = possible_axon_list[int(np.argmax(possible_axon_length)
                axon_graph = longest_axon
            axon_length = np.sum(axon_graph.edges[i]["weight"] for i in axon_graph.edges)
            '''

        gpi_inds = np.in1d(cellids, gpiids)
        gpe_inds = np.in1d(cellids, gpeids)
        gpi_spine_density = spine_density[gpi_inds]
        gpi_terminating_point_density = terminating_point_density[gpi_inds]
        gpi_branching_point_density = branching_point_density[gpi_inds]
        gpi_average_den_length = average_dendrite_length[gpi_inds]
        gpi_average_den_length_nocutoffs = average_dendrite_length_nocutoffs[gpi_inds]
        gpi_primary_den_amount = primary_dendrite_amount[gpi_inds]
        gpi_dendrite_length = overall_dendrite_length[gpi_inds]
        gpe_spine_density = spine_density[gpe_inds]
        gpe_terminating_point_density = terminating_point_density[gpe_inds]
        gpe_branching_point_density = branching_point_density[gpe_inds]
        gpe_average_den_length = average_dendrite_length[gpe_inds]
        gpe_average_den_length_nocutoffs = average_dendrite_length_nocutoffs[gpe_inds]
        gpe_primary_den_amount = primary_dendrite_amount[gpe_inds]
        gpe_dendrite_length = overall_dendrite_length[gpe_inds]

        time_stamps = [time.time()]
        step_idents = ['t-2']
        celltime = time.time()
        print("%.2f min, %.2f sec for iterating over GPs" % (celltime // 60, celltime % 60))

        log.info('Step 2/2 plot parameters of GP cells')
        sns.distplot(spine_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"}, kde=False,
                     label="GP")
        sns.distplot(gpe_spine_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_spine_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/spineden_GP.png' % f_name)
        plt.title('Spine density in GP')
        plt.xlabel('spines/ µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(overall_dendrite_length, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,
                     label="GP")
        sns.distplot(gpe_dendrite_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_dendrite_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/denlength_GP.png' % f_name)
        plt.title('Overall length of dendrite in GP')
        plt.xlabel('pathlength in µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(average_dendrite_length_nocutoffs, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,
                     label="GP")
        sns.distplot(gpe_average_den_length_nocutoffs,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_average_den_length_nocutoffs,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/avgdenlength_nocutoffs_GP.png' % f_name)
        plt.title('Average dendritic length without potentially cut off dendrites in GP')
        plt.xlabel('pathlength in µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(average_dendrite_length, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,
                     label="GP")
        sns.distplot(gpe_average_den_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_average_den_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/avgdenlength_GP.png' % f_name)
        plt.title('Average dendritic length in GP')
        plt.xlabel('pathlength in µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(primary_dendrite_amount, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,
                     label="GP")
        sns.distplot(gpe_primary_den_amount,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_primary_den_amount,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/primaryden_GP.png' % f_name)
        plt.title('Amount of primary dendrites in GP')
        plt.xlabel('amount of primary dendrites')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(branching_point_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,
                     label="GP")
        sns.distplot(gpe_branching_point_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_branching_point_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/branch_point_den_GP.png' % f_name)
        plt.title('Branching point density in GP dendrites')
        plt.xlabel('branching points/ µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(terminating_point_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,
                     label="GP")
        sns.distplot(gpe_terminating_point_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_terminating_point_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/termden_GP.png' % f_name)
        plt.title('Termination point density in GP dendrites')
        plt.xlabel('termination points/ µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        #normalised histograms
        sns.distplot(spine_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist= True,label="GP")
        sns.distplot(gpe_spine_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_spine_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/spineden_GP_norm.png' % f_name)
        plt.title('Spine density in GP')
        plt.xlabel('spines/ µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(overall_dendrite_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, label="GP", norm_hist= True)
        sns.distplot(gpe_dendrite_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist= True)
        sns.distplot(gpi_dendrite_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist= True)
        avg_filename = ('%s/denlength_GP_norm.png' % f_name)
        plt.title('Overall length of dendrite in GP')
        plt.xlabel('pathlength in µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(average_dendrite_length_nocutoffs,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, label="GP", norm_hist=True)
        sns.distplot(gpe_average_den_length_nocutoffs,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_average_den_length_nocutoffs,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/avgdenlength_nocutoffs_GP_norm.png' % f_name)
        plt.title('Average dendritic length without potentially cut off dendrites in GP')
        plt.xlabel('pathlength in µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(average_dendrite_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, label="GP", norm_hist=True)
        sns.distplot(gpe_average_den_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_average_den_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/avgdenlength_GP_norm.png' % f_name)
        plt.title('Average dendritic length in GP')
        plt.xlabel('pathlength in µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(primary_dendrite_amount,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, label="GP", norm_hist=True)
        sns.distplot(gpe_primary_den_amount,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_primary_den_amount,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/primaryden_GP_norm.png' % f_name)
        plt.title('Amount of primary dendrites in GP')
        plt.xlabel('amount of primary dendrites')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(branching_point_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,label="GP", norm_hist=True)
        sns.distplot(gpe_branching_point_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_branching_point_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/branch_point_den_GP_norm.png' % f_name)
        plt.title('Branching point density in GP dendrites')
        plt.xlabel('branching points/ µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(terminating_point_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False,label="GP", norm_hist=True)
        sns.distplot(gpe_terminating_point_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_terminating_point_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/termden_GP_norm.png' % f_name)
        plt.title('Termination point density in GP dendrites')
        plt.xlabel('termination points/ µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        #plot again without GP, only subpopulations
        sns.distplot(gpe_spine_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_spine_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/spineden_GP_sp.png' % f_name)
        plt.title('Spine density in GP')
        plt.xlabel('spines/ µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_dendrite_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_dendrite_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/denlength_GP_sp.png' % f_name)
        plt.title('Overall dendritic length in GP')
        plt.xlabel('pathlength in µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_average_den_length_nocutoffs,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_average_den_length_nocutoffs,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/avgdenlength_nocutoffs_GP_sp.png' % f_name)
        plt.title('Average dendritic length without potentially cut off dendrites in GP')
        plt.xlabel('pathlength in µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_average_den_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_average_den_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/avgdenlength_GP_sp.png' % f_name)
        plt.title('Average dendritic length in GP')
        plt.xlabel('pathlength in µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_primary_den_amount,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_primary_den_amount,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/primaryden_GP_sp.png' % f_name)
        plt.title('Amount of primary dendrites in GP')
        plt.xlabel('amount of primary dendrites')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_branching_point_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_branching_point_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/branch_point_den_GP_sp.png' % f_name)
        plt.title('Branching point density in GP dendrites')
        plt.xlabel('branching points/ µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_terminating_point_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe")
        sns.distplot(gpi_terminating_point_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi")
        avg_filename = ('%s/termden_GP_sp.png' % f_name)
        plt.title('Termination point density in GP dendrites')
        plt.xlabel('termination points/ µm')
        plt.ylabel('count of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        # normalised histograms
        sns.distplot(gpe_spine_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_spine_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/spineden_GP_norm_sp.png' % f_name)
        plt.title('Spine density in GP')
        plt.xlabel('spines/ µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_dendrite_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_dendrite_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/denlength_GP_norm_sp.png' % f_name)
        plt.title('Overall dendritic length in GP')
        plt.xlabel('pathlength in µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_average_den_length_nocutoffs,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_average_den_length_nocutoffs,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/avgdenlength_nocutoffs_GP_norm_sp.png' % f_name)
        plt.title('Average dendritic length without potentially cut off dendrites in GP')
        plt.xlabel('pathlength in µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_average_den_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_average_den_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/avgdenlength_GP_norm_sp.png' % f_name)
        plt.title('Average dendritic length in GP')
        plt.xlabel('pathlength in µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_primary_den_amount,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_primary_den_amount,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/primaryden_GP_norm_sp.png' % f_name)
        plt.title('Amount of primary dendrites in GP')
        plt.xlabel('amount of primary dendrites')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_branching_point_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_branching_point_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/branch_point_den_GP_norm_sp.png' % f_name)
        plt.title('Branching point density in GP dendrites')
        plt.xlabel('branching points/ µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(gpe_terminating_point_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "mediumorchid"},
                     kde=False, label="GPe", norm_hist=True)
        sns.distplot(gpi_terminating_point_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "springgreen"},
                     kde=False, label="GPi", norm_hist=True)
        avg_filename = ('%s/termden_GP_norm_sp.png' % f_name)
        plt.title('Termination point density in GP dendrites')
        plt.xlabel('termination points/ µm')
        plt.ylabel('fraction of cells')
        plt.legend()
        plt.savefig(avg_filename)
        plt.close()

        time_stamps = [time.time()]
        step_idents = ['t-3']
        plottime = time.time() - celltime
        print("%.2f min, %.2f sec for plotting" % (plottime // 60, plottime % 60))

        log.info('fast sholl for GP finished')


    ct_fsholl_analysis(ssd, celltype=[6,7])