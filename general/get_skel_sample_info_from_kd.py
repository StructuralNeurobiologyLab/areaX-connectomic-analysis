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

    #get length of skeleton samples from knossos file
    #get bounding box min and max

    #load knossos file to skeleton
    cells_per_celltype = 10
    skel_length = 30
    f_name = "cajal/nvmescratch/users/arother/rm_vesicle_project/220831_j0251v4_rndm_ax_samples_gt_ctn_%i_skel_%i" % (
        cells_per_celltype, skel_length)
    kn_skels = ku.load_dataset(f'{f_name}/skels.k.zip')
    #add logging

    #iterate over skeleton ids to get pathlength and write into dataframe
    columns = ['skeleton id', 'pathlength in µm', 'min bb', 'max bb']
    skel_info = pd.DataFrame(columns= columns, index = range(len(kn_skels)))
    for i, skel in enumerate(tqdm(kn_skels)):
        skel_info.loc[i, 'skeleton id'] = i
        #multiply with cell.scaling to get physical sizes
        #get pathlength e.g. via graph
        g = nx.weighted_graph(skel)
        pathlength = g.size(weight = 'weight') / 1000
        #get min and max bb from coordinates
        min_bb = np.min(skel.nodes())
        max_bb = np.max(skel.nodes())
        skel_info.loc[i, 'pathlength in µm'] = pathlength
        skel_info.loc[i, 'min bb'] = min_bb
        skel_info.loc[i, 'max bb'] = max_bb

    skel_info.save_csv(f'{f_name}/221215_skel_info.csv')

    #logging finished