if __name__ == '__main__':
    from syconn.handler.config import initialize_logging
    from syconn import global_params
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from knossos_utils import skeleton_utils as su
    from knossos_utils.skeleton import SkeletonNode
    from syconn.reps.super_segmentation import SuperSegmentationDataset

    #get length of skeleton samples from knossos file
    #get bounding box min and max

    cells_per_celltype = 10
    skel_length = 30
    global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
    f_name = "cajal/nvmescratch/users/arother/rm_vesicle_project/220901_j0251v4_rndm_ax_samples_gt_ctn_%i_skel_%i" % (
        cells_per_celltype, skel_length)
    log = initialize_logging('get info about skels random samples for vesicle annotation', log_dir=f_name + '/logs/')
    log.info("cell number per ct = %i, skeleton length = %i" % (cells_per_celltype, skel_length))
    log.info('Load skeleton knossos dataset')
    ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)
    #loads dictionary with skeletons and ids as keys
    kn_skels = su.load_skeleton(f'{f_name}/skels.k.zip', scaling=ssd.scaling)
    num_skels = len(kn_skels)

    #iterate over skeleton ids to get pathlength and write into dataframe
    log.info('Iterate over skeleton fragments')
    columns = ['skeleton id', 'pathlength in µm', 'min bb', 'max bb']
    skel_info = pd.DataFrame(columns= columns, index = range(1, num_skels + 1))
    for i in tqdm(range(1, num_skels + 1)):
        skel_info.loc[i, 'skeleton id'] = i
        skel = kn_skels[i]
        # get total skeleton length
        pathlength = skel.physical_length() / 1000  # µm
        skel_info.loc[i, 'pathlength in µm'] = pathlength
        # get min and max bb from coordinates in physical space
        skel_node_positions = np.array([SkeletonNode.getCoordinate_scaled(node) for node in skel.getNodes()])
        min_bb = np.min(skel_node_positions, axis = 0)
        max_bb = np.max(skel_node_positions, axis = 0)
        skel_info.loc[i, 'min bb'] = min_bb
        skel_info.loc[i, 'max bb'] = max_bb

    skel_info.to_csv(f'{f_name}/221215_skel_info.csv')
    log.info('Information about skeleton fragments saved')