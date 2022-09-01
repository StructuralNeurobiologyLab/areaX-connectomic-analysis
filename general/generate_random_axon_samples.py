#generate 20 µm long random axon samples from all celltypes
#get 10 per celltype from different cells
#write into one kzip
#exclude samples that are closer than 50 µm? from soma


if __name__ == '__main__':
    from wholebrain.scratch.arother.bio_analysis.general.analysis_morph_helper import check_comp_lengths_ct
    from wholebrain.scratch.arother.bio_analysis.general.analysis_conn_helper import filter_synapse_caches_for_ct
    from wholebrain.scratch.arother.bio_analysis.general.result_helper import ResultsForPlotting
    import time
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    from syconn.reps.segmentation import SegmentationDataset
    from syconn.reps.super_segmentation import SuperSegmentationObject
    import os as os
    import pandas as pd
    from syconn.handler.basics import write_obj2pkl, load_pkl2obj
    import numpy as np
    from tqdm import tqdm
    import scipy
    import networkx as nx
    import knossos_utils as ku

    global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
    start = time.time()
    ct_dict = {0: "STN", 1: "DA", 2: "MSN", 3: "LMAN", 4: "HVC", 5: "TAN", 6: "GPe", 7: "GPi", 8: "FS", 9: "LTS",
               10: "NGF"}
    min_comp_len = 500
    min_ax_len = 1000
    max_MSN_path_len = 7500
    cells_per_celltype = 10
    skel_length = 20 #µm
    dist2soma = 50 #µm
    f_name = "cajal/nvmescratch/users/arother/rm_vesicle_project/220831_j0251v4_rndm_ax_samples_noseed2_ctn_%i_skel_%i" % (
        cells_per_celltype, skel_length)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('generate random samples for vesicle annotation', log_dir=f_name + '/logs/')
    log.info("min_comp_len = %i, max_MSN_path_len = %i, min_ax_len = %i, cell number per ct = %i, skeleton length = %i, min distance to soma = %i" % (
    min_comp_len, max_MSN_path_len, min_ax_len, cells_per_celltype, skel_length, dist2soma))
    time_stamps = [time.time()]
    step_idents = ['t-0']
    known_mergers = load_pkl2obj("/wholebrain/scratch/arother/j0251v4_prep/merger_arr.pkl")

    log.info("Iterate over celltypes to randomly select %i cells per celltype" % cells_per_celltype)
    ax_ct = [1, 3, 4]
    random_axon_skels = []
    cts = list(ct_dict.keys())
    selected_cellids_perct = {i: np.zeros(cells_per_celltype) for i in cts}
    skel_num = 0
    for i, ct in enumerate(tqdm(cts)):
        log.info("Start getting random samples from celltype %s, %i/%i" % (ct_dict[ct], i, len(cts)))
        #only get cells with min_comp_len, MSN with max_comp_len or axons with min ax_len
        if ct in ax_ct:
            cell_dict = load_pkl2obj(
                "/cajal/nvmescratch/users/arother/j0251v4_prep/ax_%.3s_dict.pkl" % (ct_dict[ct]))
            cellids = np.array(list(cell_dict.keys()))
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            cellids = check_comp_lengths_ct(cellids = cellids, fullcelldict=cell_dict, min_comp_len=min_ax_len, axon_only=True, max_path_len=None)
        else:
            cell_dict = load_pkl2obj(
                "/cajal/nvmescratch/users/j0251v4_prep/full_%.3s_dict.pkl" % (ct_dict[ct]))
            cellids = load_pkl2obj(
                "/cajal/nvmescratch/users/j0251v4_prep/full_%.3s_arr.pkl" % ct_dict[ct])
            merger_inds = np.in1d(cellids, known_mergers) == False
            cellids = cellids[merger_inds]
            if ct == 2:
                cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                axon_only=False, max_path_len=max_MSN_path_len)
            else:
                cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=cell_dict, min_comp_len=min_comp_len,
                                                axon_only=False, max_path_len=None)
        log.info("%i cells of celltype %s match criteria" % (len(cellids), ct_dict[ct]))
        #randomly select fraction of cells, number = cells_per_celltye
        rndm_cellids = np.random.choice(cellids, size=10, replace=False)
        selected_cellids_perct[ct] = rndm_cellids
        #iterate over example cells to select a random sample of their axon
        for ic, cellid in enumerate(rndm_cellids):
            cell = SuperSegmentationObject(cellid)
            cell.load_skeleton()
            g = cell.weighted_graph()
            if ct not in ax_ct:
                axoness = cell.skeleton["axoness_avg10000"]
                axoness[axoness == 3] = 1
                axoness[axoness == 4] = 1
                axon_inds = np.nonzero(axoness == 1)[0]
                non_axon_inds = np.nonzero(axoness != 1)[0]
                g.remove_nodes_from(non_axon_inds)
                axon_coords = cell.skeleton["nodes"][axon_inds]
                #remove nodes that are too clode to soma
                #1st get nodes that surround soma and then see which of these is really connected to soma
                kdtree = scipy.spatial.cKDTree(axon_coords * cell.scaling)
                potential_close2soma_inds = kdtree.query_ball_point(cell_dict[cellid]["soma centre"], dist2soma * 1000)
                potential_close_coords = axon_coords[potential_close2soma_inds] * cell.scaling
                potential_close_nodes = np.array(list(g.nodes()))[potential_close2soma_inds]
                distances2soma = cell.shortestpath2soma(potential_close_coords)
                distances2soma = np.array(distances2soma) / 1000 # in µm
                close2soma_inds = np.nonzero(distances2soma <= dist2soma)[0]
                graph_close_inds = potential_close_nodes[close2soma_inds]
                g.remove_nodes_from(graph_close_inds)
            #get random node and cut out 20 µm skeleton sample
            rndm_node = np.random.choice(g.nodes())
            #generate appr. 20 µm large pieces
            rndm_sample_graph = nx.ego_graph(g, rndm_node, radius = skel_length/2* 1000, distance = "weight")
            #write graph to knossus_utils.skeleton, code partly from function of skeleton_utils nx_graph_to_annotation but does not work with normal nx_graph
            #and supersegmentation_object save skeleton to kzip
            skel = ku.skeleton.SkeletonAnnotation()
            skel.scaling = cell.scaling
            skel.comment = "skeleton %i" % skel_num
            skel_nodes = []
            skel_node_dict = {}
            for i, node in enumerate(rndm_sample_graph.nodes()):
                c = cell.skeleton["nodes"][node]
                r = cell.skeleton["diameters"][node] / 2
                skel_nodes.append(ku.skeleton.SkeletonNode().
                                  from_scratch(skel, c[0], c[1], c[2], radius=r))
                skel.addNode(skel_nodes[-1])
                skel_node_dict[node] = i
            for edge in rndm_sample_graph.edges():
                skel.addEdge(skel_nodes[skel_node_dict[edge[0]]], skel_nodes[skel_node_dict[edge[1]]])
            random_axon_skels.append(skel)
            skel_num += 1

        log.info("Wrote skeletons to kzip from %s" % ct_dict[ct])


    log.info("Save results")
    ku.skeleton_utils.write_skeleton(path="%s/skels.k.zip" % f_name, new_annos=random_axon_skels)
    random_ids_df = pd.DataFrame(selected_cellids_perct)
    random_ids_df.to_csv("%s/randomly_selected_cellids.csv" % f_name)

    #save all sekeltons in one kzip file