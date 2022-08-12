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
    f_name = "wholebrain/scratch/arother/rm_vesicle_project/220812_j0251v4_rndm_ax_samples_ctn_%i_skel_%i" % (
        cells_per_celltype, skel_length)
    if not os.path.exists(f_name):
        os.mkdir(f_name)
    log = initialize_logging('LMAN MSN connectivity estimate', log_dir=f_name + '/logs/')
    log.info("min_comp_len = %i, max_MSN_path_len = %i, min_ax_len = %i, cell number per ct = %i, skeleton length = %i, min distance to soma = %i" % (
    min_comp_len, max_MSN_path_len, min_ax_len, cells_per_celltype, skel_length, dist2soma))
    time_stamps = [time.time()]
    step_idents = ['t-0']

    log.info("Iterate over celltypes to randomly select %i cells per celltype" % cells_per_celltype)
    ax_ct = [1, 3, 4]
    example_axon_skels = []
    cts = list(ct_dict.keys())
    selected_cellids_perct = {i: np.zeros(cells_per_celltype) for i in cts}
    for i, ct in enumerate(tqdm(cts)):
        log.info("Start getting random samples from celltype %s, %i/%i" % (ct_dict[ct], i, len(cts)))
        #only get cells with min_comp_len, MSN with max_comp_len or axons with min ax_len
        if ct in ax_ct:
            ct_dict = load_pkl2obj(
                "/wholebrain/scratch/arother/j0251v4_prep/ax_%.3s_dict.pkl" % (ct_dict[ct]))
            cellids = list(ct_dict.keys())
            cellids = check_comp_lengths_ct(cellids = cellids, fullcelldict=ct_dict, min_comp_len=min_ax_len, axon_only=True, max_path_len=None)
        else:
            ct_dict = load_pkl2obj(
                "/wholebrain/scratch/arother/j0251v4_prep/ax_%.3s_dict.pkl" % (ct_dict[ct]))
            cellids = load_pkl2obj(
                "/wholebrain/scratch/arother/j0251v4_prep/full_%.3s_arr.pkl" % ct_dict[ct])
            if ct == 2:
                cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=ct_dict, min_comp_len=min_comp_len,
                                                axon_only=False, max_path_len=max_MSN_path_len)
            else:
                cellids = check_comp_lengths_ct(cellids=cellids, fullcelldict=ct_dict, min_comp_len=min_comp_len,
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
            raise ValueError
            if ct in ax_ct:
                coords = cell.skeleton["nodes"]
            else:
                axon_inds = np.nonzero(cell.skeleton["axoness_avg10000"] == 1)[0]
                axon_coords = cell.skeleton["nodes"][axon_inds]
                #remove nodes that are too clode to soma
                distances2soma = cell.shortest_path2soma(axon_graph.nodes())


    #steps per celltype
    #get only complete cells with mcl
    #get only long axon parts to increase celltype probability
    #get randomly 10 cells from each (seed?)
    #get cell ids to check manually that no merger in this celltype
    #randomly select point on skeleton and get 20 µm skeleton piece on this axon

    #save all sekeltons in one kzip file