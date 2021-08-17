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
    from tqdm import tqdm
    from syconn.handler.config import initialize_logging
    from syconn.handler.basics import load_pkl2obj
    #global_params.wd = "/ssdscratch/pschuber/songbird/j0126/assembled_core_relabeled_base_merges_relabeled_to_v4b_base_20180214_full_agglo_cbsplit_with_reconnects_no_soma_merger_manual_edges_removed"
    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv",working_dir=global_params.config.working_dir)


    def ct_fsholl_analysis(ssd, celltype, min_comp_length = 100):
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
        f_name = "u/arother/test_folder/210614_j0251v3_fastsholl_MSN_mcl%i" % min_comp_length
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('sholl_analysis', log_dir=f_name + '/logs/')
        time_stamps = [time.time()]
        step_idents = ['t-0']
        comp_dict = {1: 'axons', 0: 'dendrites'}
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7:"GPi", 8:"FS", 9:"LTS", 10:"NGF"}
        msnids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%3s_arr.pkl" % ct_dict[celltype])
        log.info('Step 1/2 generating sholl per cell')
        average_dendrite_length = np.zeros(len(msnids))
        average_dendrite_length_nocutoffs = np.zeros(len(msnids))
        primary_dendrite_amount = np.zeros(len(msnids))
        branching_point_density = np.zeros(len(msnids))
        terminating_point_density = np.zeros(len(msnids))
        spine_density = np.zeros(len(msnids))
        overall_dendrite_length = np.zeros(len(msnids))
        longest_axons = np.zeros(len(msnids))
        longest_dendrites = np.zeros(len(msnids))
        for pi, cell in enumerate(tqdm(ssd.get_super_segmentation_object(msnids))):
            cell.load_skeleton()
            raise ValueError
            nonaxon_inds = np.nonzero(cell.skeleton["axoness_avg10000"] != 1)[0]
            nondendrite_inds = np.nonzero(cell.skeleton["axoness_avg10000"] != 0)[0]
            spine_shaftinds = np.nonzero(cell.skeleton["spiness"] == 0)[0]
            spine_otherinds = np.nonzero(cell.skeleton["spiness"] == 3)[0]
            spine_headinds = np.nonzero(cell.skeleton["spiness"] == 1)[0]
            spine_neckinds = np.nonzero(cell.skeleton["spiness"] == 2)[0]
            nonspine_inds = np.hstack([spine_shaftinds, spine_otherinds])
            spine_inds = np.hstack([spine_headinds, spine_neckinds])
            g = cell.weighted_graph(add_node_attr=('axoness_avg10000', "spiness"))
            #get longest axon or size of axon if doesn give more than 1
            axon_graph = g.copy()
            axon_graph.remove_nodes_from(nonaxon_inds)
            if len(list(nx.connected_component_subgraphs(axon_graph))) > 1:
                possible_axon_list = list(nx.connected_component_subgraphs(axon_graph))
                possible_axon_length = np.zeros(len(possible_axon_list))
                for ia, possible_axon in enumerate(possible_axon_list):
                    possible_axon_length[ia] = possible_axon.size(weight="weight") / 1000
                longest_axon_cell = np.max(possible_axon_length)
            else:
                longest_axon_cell = axon_graph.size(weight="weight") / 1000
            if longest_axon_cell < min_comp_length:
                continue
            dendrite_graph = g.copy()
            if dendrite_graph.size(weight = "weight") / 1000 < min_comp_length:
                continue
            dendrite_graph.remove_nodes_from(nondendrite_inds)
            den_shaft_graph = dendrite_graph.copy()
            den_shaft_graph.remove_nodes_from(spine_inds)
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
            den_length_branches = np.zeros(len(dendrite_graphs))
            for di, dendrite in enumerate(dendrite_graphs.copy()):
                dendrite.remove_nodes_from(spine_inds)
                den_size = dendrite.size(weight = "weight")/ 1000
                den_length_branches[di] = den_size
                if den_size < 1:
                    continue
                for_bp += 1
                if den_size/average_den_length_cell >= 0.25:
                    den_length_nocutoffs += den_size
                    no_cutoffs.append(di)
            if len(no_cutoffs) == 0:
                continue
            longest_dendrite_cell = np.max(den_length_branches)
            average_den_length_nocutoffs_cell = den_length_nocutoffs/len(no_cutoffs)
            degrees = np.zeros(len(den_shaft_graph.nodes()))
            for ix, node_id in enumerate(den_shaft_graph.nodes()):
                degrees[ix] = g.degree[node_id]
            # remove all branching points and count amount of connected components with length of 10 µm
            # problem with following code: might break into lots of small pieces and then none of them counts
            '''
            branching_point_inds = np.nonzero(degrees >= 3)[0]
            branching_graph = den_shaft_graph.copy()
            branching_graph.remove_nodes_from(branching_point_inds)
            connected_branch_pieces = list(nx.connected_component_subgraphs(branching_graph))
            connectec_den_pieces = np.zeros(len(connected_branch_pieces))
            for bi, branch_piece in enumerate(connected_branch_pieces):
                if branch_piece.size(weight = "weight") / 1000 >= 1:
                    connectec_den_pieces[bi] = 1
            branching_point_amount = np.sum(connectec_den_pieces)
                        #alternative: iterative branching point analysis (takes longer)
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
            '''
            terminating_point_amout = len(degrees[degrees == 1])
            #branching_point_density_cell = branching_point_amount/den_length
            terminating_point_density_cell = terminating_point_amout/ den_length
            spine_density_cell = spine_amount_skeleton/ den_length
            average_dendrite_length[pi] = average_den_length_cell
            average_dendrite_length_nocutoffs[pi] = average_den_length_nocutoffs_cell
            spine_density[pi] = spine_density_cell
            #branching_point_density[pi] = branching_point_density_cell
            terminating_point_density[pi] = terminating_point_density_cell
            primary_dendrite_amount[pi] = primary_den_cell
            overall_dendrite_length[pi] = den_length
            longest_axons[pi] = longest_axon_cell
            longest_dendrites[pi] = longest_dendrite_cell

        nonzero_inds = average_dendrite_length > 0
        average_dendrite_length = average_dendrite_length[nonzero_inds]
        average_dendrite_length_nocutoffs = average_dendrite_length_nocutoffs[nonzero_inds]
        primary_dendrite_amount = primary_dendrite_amount[nonzero_inds]
        #branching_point_density = branching_point_density[nonzero_inds]
        terminating_point_density = terminating_point_density[nonzero_inds]
        spine_density = spine_density[nonzero_inds]
        overall_dendrite_length = overall_dendrite_length[nonzero_inds]
        longest_dendrites = longest_dendrites[nonzero_inds]
        longest_axons = longest_axons[nonzero_inds]


        time_stamps = [time.time()]
        step_idents = ['t-2']
        celltime = time.time()
        print("%.2f min, %.2f sec for iterating over MSNs" % (celltime // 60, celltime % 60))

        log.info('Step 2/2 plot parameters of MSN cells')
        plt.scatter(x = spine_density, y=longest_dendrites, c="black", alpha = 0.7)
        avg_filename = ('%s/spinedenvsmaxdenlen_MSN.png' % f_name)
        plt.title('Spine density vs maximal dendritic length in MSN')
        plt.xlabel('spines/ µm')
        plt.ylabel('pathlength [µm]')
        plt.savefig(avg_filename)
        plt.close()

        plt.scatter(x=spine_density, y=longest_axons, c="black", alpha=0.7)
        avg_filename = ('%s/spinedenvsmaxdaxolen_MSN.png' % f_name)
        plt.title('Spine density vs maximal axonic length in MSN')
        plt.xlabel('spines/ µm')
        plt.ylabel('pathlength [µm]')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(spine_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"}, kde=False)
        avg_filename = ('%s/spineden_MSN.png' % f_name)
        plt.title('Spine density in MSN')
        plt.xlabel('spines/ µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(longest_axons, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/maxax_MSN.png' % f_name)
        plt.title('Maximal axon length in MSN')
        plt.xlabel('pathlength [µm]')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(longest_dendrites, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/maxden_MSN.png' % f_name)
        plt.title('Maximal dendrite length in MSN')
        plt.xlabel('pathlength [µm]')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(overall_dendrite_length, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/denlen_MSN.png' % f_name)
        plt.title('Dendrite length in MSN')
        plt.xlabel('pathlength µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(average_dendrite_length_nocutoffs, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/avgdenlength_nocutoffs_MSN.png' % f_name)
        plt.title('Average dendritic length without potentially cut off dendrites in MSN')
        plt.xlabel('pathlength in µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(average_dendrite_length, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/avgdenlength_MSN.png' % f_name)
        plt.title('Average dendritic length in MSN')
        plt.xlabel('pathlength in µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(primary_dendrite_amount, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/primaryden_MSN.png' % f_name)
        plt.title('Amount of primary dendrites in MSN')
        plt.xlabel('amount of primary dendrites')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        '''
        sns.distplot(branching_point_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/branch_point_den_MSN.png' % f_name)
        plt.title('Branching point density in MSN dendrites')
        plt.xlabel('branching points/ µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()
        '''

        sns.distplot(terminating_point_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/termden_MSN.png' % f_name)
        plt.title('Termination point density in MSN dendrites')
        plt.xlabel('termination points/ µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        #normalised histograms

        sns.distplot(longest_axons, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/maxax_MSN_norm.png' % f_name)
        plt.title('Maximal axon length in MSN')
        plt.xlabel('pathlength [µm]')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(longest_dendrites, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/maxden_MSN_norm.png' % f_name)
        plt.title('Maximal dendrite length in MSN')
        plt.xlabel('pathlength [µm]')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(spine_density, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist= True)
        avg_filename = ('%s/spineden_MSN_norm.png' % f_name)
        plt.title('Spine density in MSN')
        plt.xlabel('spines/ µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(overall_dendrite_length, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/denlen_MSN_norm.png' % f_name)
        plt.title('Dendrite length in MSN')
        plt.xlabel('pathlength µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(average_dendrite_length_nocutoffs,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/avgdenlength_nocutoffs_MSN_norm.png' % f_name)
        plt.title('Average dendritic length without potentially cut off dendrites in MSN')
        plt.xlabel('pathlength in µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(average_dendrite_length,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/avgdenlength_MSN_norm.png' % f_name)
        plt.title('Average dendritic length in MSN')
        plt.xlabel('pathlength in µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        sns.distplot(primary_dendrite_amount,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/primaryden_MSN_norm.png' % f_name)
        plt.title('Amount of primary dendrites in MSN')
        plt.xlabel('amount of primary dendrites')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        ''''
        sns.distplot(branching_point_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        avg_filename = ('%s/branch_point_den_MSN_norm.png' % f_name)
        plt.title('Branching point density in MSN dendrites')
        plt.xlabel('branching points/ µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()
        `'''

        sns.distplot(terminating_point_density,
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False, norm_hist=True)
        plt.title('Termination point density in GP dendrites')
        plt.xlabel('termination points/ µm')
        plt.ylabel('fraction of cells')
        plt.savefig(avg_filename)
        plt.close()

        #log plots
        spine_density = spine_density[spine_density > 0]
        sns.distplot(np.log10(spine_density), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/spineden_MSN_log.png' % f_name)
        plt.title('log Spine density in MSN')
        plt.xlabel('log spines/ µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        longest_axons = longest_axons[longest_axons > 0]
        sns.distplot(np.log10(longest_axons), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/maxax_MSN_log.png' % f_name)
        plt.title('Log maximal axon length in MSN')
        plt.xlabel('log pathlength [µm]')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        longest_dendrites = longest_dendrites[longest_dendrites > 0]
        sns.distplot(np.log10(longest_dendrites), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/maxden_MSN_log.png' % f_name)
        plt.title('Log maximal dendrite length in MSN')
        plt.xlabel('log pathlength [µm]')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        overall_dendrite_length = overall_dendrite_length[overall_dendrite_length > 0]
        sns.distplot(np.log10(overall_dendrite_length), hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/denlen_MSN_log.png' % f_name)
        plt.title('log Dendrite length in MSN')
        plt.xlabel('log pathlength µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        average_dendrite_length_nocutoffs = average_dendrite_length_nocutoffs[average_dendrite_length_nocutoffs > 0]
        sns.distplot(np.log10(average_dendrite_length_nocutoffs),
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/avgdenlength_nocutoffs_MSN_log.png' % f_name)
        plt.title('log Average dendritic length without potentially cut off dendrites in MSN')
        plt.xlabel('log pathlength in µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        average_dendrite_length = average_dendrite_length[average_dendrite_length > 0]
        sns.distplot(np.log10(average_dendrite_length),
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/avgdenlength_MSN_log.png' % f_name)
        plt.title('log Average dendritic length in MSN')
        plt.xlabel('log pathlength in µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        primary_dendrite_amount = primary_dendrite_amount[primary_dendrite_amount > 0]
        sns.distplot(np.log10(primary_dendrite_amount),
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/primaryden_MSN_log.png' % f_name)
        plt.title('log Amount of primary dendrites in MSN')
        plt.xlabel('log amount of primary dendrites')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        '''
        branching_point_density = branching_point_density[branching_point_density > 0]
        sns.distplot(np.log10(branching_point_density),
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/branch_point_den_MSN_log.png' % f_name)
        plt.title('Log Branching point density in MSN dendrites')
        plt.xlabel('log branching points/ µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()
        '''

        terminating_point_density = terminating_point_density[terminating_point_density > 0]
        sns.distplot(np.log10(terminating_point_density),
                     hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "black"},
                     kde=False)
        avg_filename = ('%s/termden_MSN_log.png' % f_name)
        plt.title('log Termination point density in MSN dendrites')
        plt.xlabel('log termination points/ µm')
        plt.ylabel('count of cells')
        plt.savefig(avg_filename)
        plt.close()

        time_stamps = [time.time()]
        step_idents = ['t-3']
        plottime = time.time() - celltime
        print("%.2f min, %.2f sec for plotting" % (plottime // 60, plottime % 60))

        log.info('fast sholl for MSN finished')


    ct_fsholl_analysis(ssd, celltype=2, min_comp_length=500)