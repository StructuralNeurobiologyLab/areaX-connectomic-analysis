#generate hybrid cloud pkl for cnn_training to increase speed

from syconn.handler.prediction_pts import _load_ssv_hc
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.handler.basics import write_obj2pkl, load_pkl2obj
from syconn import global_params
import pandas as pd
import numpy as np
from functools import partial
from multiprocessing import pool
from tqdm import tqdm
from scipy.spatial import cKDTree


v6_gt = pd.read_csv("wholebrain/songbird/j0251/groundtruth/celltypes/j0251_celltype_gt_v6_j0251_72_seg_20210127_agglo2_IDs.csv", names = ["cellids", "celltype"])
cellids = np.array(v6_gt["cellids"])
filename = "cajal/nvmescratch/users/arother/cnn_training/hybrid_clouds/"

global_params.wd = "/ssdscratch/songbird/j0251/j0251_72_seg_20210127_agglo2"
ssd = SuperSegmentationDataset(working_dir=global_params.config.working_dir)

ssd_kwargs = dict(working_dir=global_params.wd, version='ctgt_v4')

#parameters from default parameters of script cnn_celltype_ptcnv_j0251.py
use_myelin = False
use_syntype = True
cellshape_only = False
pts_feat_dict = dict(sv=0, mi=1, syn_ssv=3, syn_ssv_sym=3, syn_ssv_asym=4, vc=2, sv_myelin=5)

#for this function use parts of syconn.handler.prediction_pts.pts_loader_scalar
def get_hybrid_clouds(cellid, use_myelin, use_syntype, cellshape_only, filename):
    ssv = ssd.get_super_segmentation_object(cellid)
    feat_dc = dict(pts_feat_dict)
    if cellshape_only:
        feat_dc = dict(sv=feat_dc['sv'])
    else:
        if use_syntype:
            if 'syn_ssv' in feat_dc:
                del feat_dc['syn_ssv']
        else:
            del feat_dc['syn_ssv_sym']
            del feat_dc['syn_ssv_asym']
            assert 'syn_ssv' in feat_dc
        if not use_myelin:
            del feat_dc['sv_myelin']
    args = (ssv, tuple(feat_dc.keys()), tuple(feat_dc.values()), 'celltype', None, use_myelin)
    hc = _load_ssv_hc(args)
    ssv.clear_cache()
    write_obj2pkl("%s/%i_hc.pkl" % (filename, cellid), hc)

def get_cKDtrees_pkl(cellid):
    '''
    function to write cKTree from hc cloud, which is precomputed and
    saved as pkl.
    :param cellid: cellid for hc cloud
    :return:
    saves pkl of cKdTree of hc
    '''
    hc = load_pkl2obj("%s/%i_hc.pkl" % (filename, cellid))
    ckdtree = cKDTree(hc.nodes)
    write_obj2pkl("%s/%i_kdtree.pkl" % (filename, cellid), ckdtree)

p = pool.Pool()
#p.map(partial(get_hybrid_clouds, use_myelin = use_myelin, use_syntype = use_syntype,
 #                    cellshape_only = cellshape_only, filename = filename), tqdm(cellids))


p.map(get_cKDtrees_pkl, tqdm(cellids))


