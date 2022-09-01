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
    global_params.wd = "/ssdscratch/pschuber/songbird/j0251/rag_flat_Jan2019_v3"

# see if amount of spines classified as synapses
#check branches between 3 or 4 nodes until 20 nodes with spine prediction in dendrites
#check density & ratio in synapses

#use sso.weighted graph to delete all nodes that are not dendrites to get the amount of dendrites
#this is needed to calculate the spine amount per dendrite
#then also delete all the ones that are not spines
#count the number of spines
#in next steps also calculate the shortest path for the nodes in the spines, their maximum distance to get the length of the spines
#but also to get spine density

    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    sd_synssv = SegmentationDataset("syn_ssv", working_dir=global_params.config.working_dir)

    def spine_per_cell(sso):
        """
        gets all the spine nodes per cell, counts them, togehter with the number of dendrites and calculates the shortest path of each of them
        compartments: 0 = dendrite, 1 = axon, 2 = soma
        spiness values: 0 = dendritic shaft, 1 = spine head, 2 = spine neck, 3 = other
        :param sso: cell
        :return:
        """
        sso.load_skeleton()
        nondendrite_inds = np.nonzero(sso.skeleton["axoness_avg10000"] != 0)[0]
        spine_shaftinds = np.nonzero(sso.skeleton["spiness"] == 0)[0]
        spine_otherinds = np.nonzero(sso.skeleton["spiness"] == 3)[0]
        spine_headinds = np.nonzero(sso.skeleton["spiness"] == 1)[0]
        spine_neckinds = np.nonzero(sso.skeleton["spiness"] == 2)[0]
        nonspine_inds = np.hstack([spine_shaftinds, spine_otherinds])
        spine_inds = np.hstack([spine_headinds, spine_neckinds])

        g = sso.weighted_graph(add_node_attr=('axoness_avg10000', "spiness"))
        dendrite_graph = g.copy()
        '''
        for n, d in g.nodes(data=True):
            axoness = d["axoness_avg10000"]
            if axoness == 3 or axoness == 4:
                axoness = 1
            if d['axoness_avg10000'] != 0:
                dendrite_graph.remove_node(n)
                spine_graph.remove_node(n)
            if d["spiness"] == 0 or d["spiness"] == 3:
                if n not in spine_graph.nodes():
                    continue
                spine_graph.remove_node(n)
        '''
        dendrite_graph.remove_nodes_from(nondendrite_inds)
        spine_graph = dendrite_graph.copy()
        spine_graph.remove_nodes_from(nonspine_inds)
        dendrite_subgraphs = list(nx.connected_component_subgraphs(dendrite_graph))
        spine_subgraphs = list(nx.connected_component_subgraphs(spine_graph))
        den_amount = len(dendrite_subgraphs)
        spine_amount = len(spine_subgraphs)
        spines_per_dendrite = spine_amount/den_amount

        if den_amount == 0:
            return 0, 0, 0, 0

        if spine_amount == 0:
            return 0, 0, 0, 0

        #reshape dendrite subgraphs to array in order to search trough nodes
        den_nodes_array = np.zeros((den_amount, len(dendrite_graph))) - 1
        den_length = np.zeros(den_amount)
        for di, dendrite in enumerate(dendrite_subgraphs):
            #if just one node, can convert to array directly but has to be list first
            if len(dendrite.nodes()) > 1:
                den_nodes_array[di][:len(dendrite.nodes())] = dendrite.nodes()
                for dedge in dendrite.edges():
                    if np.any(np.in1d(dedge, spine_inds)):
                        continue
                    den_length[di] += dendrite.edges[dedge]["weight"]
            else:
                den_nodes_array[di][:len(dendrite.nodes())] = list(dendrite.nodes())

        den_length = den_length / 1000 #µm

        #calculate shortest path for spine_nodes to calculate distance to soma and length of spine
        spine_length_arr = np.zeros(spine_amount)
        spines_on_dendrite = np.zeros((den_amount, spine_amount))
        spineam_on_dendrite = np.zeros(den_amount)
        for si, spine in enumerate(list(nx.connected_component_subgraphs(spine_graph))):
            if len(spine.nodes()) > 1:
                if len(spine.nodes()) > 60:
                    node_inds = np.array(spine.nodes())
                    coords = sso.skeleton["nodes"][node_inds[0]]
                    print(coords[0], sso.id)
                    spine_amount -= 1
                    continue
                node_inds = np.array(spine.nodes())
                #remove nodes with degree = 2, since they are neither end nor beginning of spine
                deg2nodes = np.zeros(len(node_inds)) - 1
                for ni, node in enumerate(node_inds):
                    if spine_graph.degree[node] == 2:
                        deg2nodes[ni] = ni
                deg2nodes = deg2nodes[deg2nodes >=0 ]
                if len(deg2nodes) > 0:
                    node_inds = np.delete(node_inds, deg2nodes)
            else:
                node_inds = np.array(list(spine.nodes()))
            coords = sso.skeleton["nodes"][node_inds]
            #shortest_path_perspine = sso.shortestpath2soma(coords)
            #minpath = np.min(shortest_path_perspine)
            #maxpath = np.max(shortest_path_perspine)
            #spine_length = maxpath - minpath
            #spine_length_arr[si] = spine_length
            spine_den_ind = int(np.where(den_nodes_array == node_inds[0])[0])
            #spines_on_dendrite[spine_den_ind][si] = minpath
            spineam_on_dendrite[spine_den_ind] += 1


        #calculate average spine length, spine density
        #spine_length_arr = spine_length_arr[spine_length_arr > 0]
        #avg_spine_length = np.mean(spine_length_arr)/1000 #in µm
        #max_path_dendrite = np.max(spines_on_dendrite, axis = 1)/ 1000 #µm
        # average pathlength between two spines
        #inter_spine_pathlength = max_path_dendrite/spineam_on_dendrite
        #avg_inter_spine = np.mean(inter_spine_pathlength)
        # amount of spines per 10 µm
        den_inds = den_length > 0
        den_length = den_length[den_inds]
        spineam_on_dendrite = spineam_on_dendrite[den_inds]
        spine_density = (10/(den_length[spineam_on_dendrite > 0]/spineam_on_dendrite[spineam_on_dendrite > 0]))
        avg_spine_density = np.mean(spine_density)

        return spine_amount, spines_per_dendrite, avg_spine_density, den_amount #avg_spine_length, avg_inter_spine


    def spine_analysis(ssd, celltype):
        """
        analyse spine amount, spine amount per dendrite, spine density, spine length in full cells
        :param ssd: super segmentation dataset
        :param celltype: j0251: STN=0, DA=1, MSN=2, LMAN=3, HVC=4, TAN=5, GPe=6, GPi=7,FS=8, LTS=9, NGF=10
        :return: spine amount, spine amount per dendrite, spine density spine length
        """
        start = time.time()
        f_name = ("u/arother/test_folder/201216_j0251v3_MSN_spineanalysismax60_nopathlength")
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        log = initialize_logging('spine_analysis', log_dir=f_name + '/logs/')
        time_stamps = [time.time()]
        step_idents = ['t-0']
        ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
                   10: "NGF"}
        # cellids = ssd.ssv_ids[ssd.load_cached_data("celltype_cnn_e3") == celltype]
        cellids = load_pkl2obj("/wholebrain/scratch/arother/j0251v3_prep/full_%.3s_arr.pkl" % ct_dict[celltype])
        spine_amount = np.zeros(len(cellids))
        spine_per_dendrite = np.zeros(len(cellids))
        spine_length = np.zeros(len(cellids))
        inter_spine_pathlength = np.zeros(len(cellids))
        spine_density = np.zeros(len(cellids))
        primary_dendrite = np.zeros(len(cellids))
        for i, cell in enumerate(ssd.get_super_segmentation_object(cellids)):
            #spine_am_cell, spine_per_den_cell, spine_density_cell, den_amount_cell, spine_length_cell, inter_spine_cell = spine_per_cell(sso= cell)
            spine_am_cell, spine_per_den_cell, spine_density_cell, den_amount_cell = spine_per_cell(sso=cell)
            if spine_am_cell == 0:
                continue
            spine_amount[i] = spine_am_cell
            spine_per_dendrite[i] = spine_per_den_cell
            #spine_length[i] = spine_length_cell
            #inter_spine_pathlength[i] = inter_spine_cell
            spine_density[i] = spine_density_cell
            primary_dendrite[i] = den_amount_cell
            percentage = 100 * (i / len(cellids))
            print("%.2f percent, cell %.4i done" % (percentage, cell.id))

        spine_amount = spine_amount[spine_amount > 0]
        spine_per_dendrite = spine_per_dendrite[spine_per_dendrite > 0]
        #spine_length = spine_length[spine_length > 0]
        #inter_spine_pathlength = inter_spine_pathlength[inter_spine_pathlength > 0]
        spine_density = spine_density[spine_density > 0]
        primary_dendrite = primary_dendrite[primary_dendrite > 0]
        celltime = time.time() - start
        print("%.2f sec for processing cells" % celltime)

        #plot all of these per cell
        sns.distplot(spine_amount, kde=False, color="skyblue")
        plt.title('average amount of spines in %s' % ct_dict[celltype])
        plt.xlabel('amount of spines')
        plt.ylabel('count of cells')
        plt.savefig('%s/spine_amount_%s.png' % (f_name, ct_dict[celltype]))
        plt.close()

        sns.distplot(spine_per_dendrite, kde=False, color="skyblue")
        plt.title('average amount of spines per dendrite in %s' % ct_dict[celltype])
        plt.xlabel('amount of spines')
        plt.ylabel('count of cells')
        plt.savefig('%s/spine_perden_%s.png' % (f_name, ct_dict[celltype]))
        plt.close()

        #sns.distplot(spine_length, kde=False, color="skyblue")
        #plt.title('average spine length in %s' % ct_dict[celltype])
        #plt.xlabel('spine length in µm')
        #plt.ylabel('count of cells')
        #plt.savefig('%s/spine_perden_%s.png' % (f_name, ct_dict[celltype]))
        #plt.close()

        #sns.distplot(inter_spine_pathlength, kde=False, color="skyblue")
        #plt.title('average pathlength between spines in %s' % ct_dict[celltype])
        #plt.xlabel('pathlength in µm')
        #plt.ylabel('count of cells')
        #plt.savefig('%s/spine_interpath_%s.png' % (f_name, ct_dict[celltype]))
        #plt.close()

        sns.distplot(spine_density, kde=False, color="skyblue")
        plt.title('spine density in %s' % ct_dict[celltype])
        plt.xlabel('amount of spines in 10 µm')
        plt.ylabel('count of cells')
        plt.savefig('%s/spine_density_%s.png' % (f_name, ct_dict[celltype]))
        plt.close()

        sns.distplot(primary_dendrite, kde=False, color="skyblue")
        plt.title('primary dendrite in %s' % ct_dict[celltype])
        plt.xlabel('amount of primary dendrites')
        plt.ylabel('count of cells')
        plt.savefig('%s/prime_den_%s.png' % (f_name, ct_dict[celltype]))
        plt.close()

        sns.distplot(np.log10(spine_amount), kde=False, color="skyblue")
        plt.title('log average amount of spines in %s' % ct_dict[celltype])
        plt.xlabel('log amount of spines')
        plt.ylabel('count of cells')
        plt.savefig('%s/logspine_amount_%s.png' % (f_name, ct_dict[celltype]))
        plt.close()

        sns.distplot(np.log10(spine_per_dendrite), kde=False, color="skyblue")
        plt.title('log average amount of spines per dendrite in %s' % ct_dict[celltype])
        plt.xlabel('log amount of spines')
        plt.ylabel('count of cells')
        plt.savefig('%s/logspine_perden_%s.png' % (f_name, ct_dict[celltype]))
        plt.close()

        #sns.distplot(np.log10(spine_length), kde=False, color="skyblue")
        #plt.title('log average spine length in %s' % ct_dict[celltype])
        #plt.xlabel('log spine length in µm')
        #plt.ylabel('count of cells')
        #plt.savefig('%s/logspine_perden_%s.png' % (f_name, ct_dict[celltype]))
        #plt.close()

        #sns.distplot(np.log10(inter_spine_pathlength), kde=False, color="skyblue")
        #plt.title('average pathlength between spines in %s' % ct_dict[celltype])
        #plt.xlabel('log pathlength in µm')
        #plt.ylabel('count of cells')
        #plt.savefig('%s/logspine_interpath_%s.png' % (f_name, ct_dict[celltype]))
        #plt.close()

        sns.distplot(np.log10(spine_density), kde=False, color="skyblue")
        plt.title('log spine density in %s' % ct_dict[celltype])
        plt.xlabel('log amount of spines in 10 µm')
        plt.ylabel('count of cells')
        plt.savefig('%s/logspine_density_%s.png' % (f_name, ct_dict[celltype]))
        plt.close()

        sns.distplot(np.log10(primary_dendrite), kde=False, color="skyblue")
        plt.title('log of primary dendrite in %s' % ct_dict[celltype])
        plt.xlabel('log amount of primary dendrites')
        plt.ylabel('count of cells')
        plt.savefig('%s/logprime_den_%s.png' % (f_name, ct_dict[celltype]))
        plt.close()

        plottime = time.time() - celltime
        print("%.2f sec for plotting" % plottime)
        time_stamps.append(time.time())
        step_idents.append('finished analysis')

        return spine_amount, spine_per_dendrite, spine_density, #spine_length, inter_spine_pathlength,

    #spine_amount, spine_per_dendrite, inter_spine_pathlength, spine_density, spine_length = spine_analysis(ssd, celltype=2)
    spine_amount, spine_per_dendrite, spine_density = spine_analysis(ssd, celltype=2)

    raise ValueError