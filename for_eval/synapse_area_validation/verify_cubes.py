"""python verify_cubes.py <csv>: check each cube against the server (origin, raw bytes, pre/post labels, sj voxels at centre)."""
import sys, numpy as np, pandas as pd
from fetch_cubes import open_volume
from common import RAW_LAYER, SEG_LAYER
from gt_sj_pipeline import WD
from syconn import global_params; global_params.wd = WD
from syconn.reps.super_segmentation import SuperSegmentationDataset
ssd = SuperSegmentationDataset(working_dir=WD); raw, seg = open_volume(RAW_LAYER), open_volume(SEG_LAYER)
lo, hi = [180, 180, 72], [220, 220, 88]
for _, r in pd.read_csv(sys.argv[1]).iterrows():
    o = np.load('cubes/%s_origin.npy' % r.syn_id).astype(int); box = tuple(slice(o[i] + lo[i], o[i] + hi[i]) for i in range(3))
    R = np.load('cubes/%s_raw.npy' % r.syn_id)[180:220, 180:220, 72:88]; S = np.load('cubes/%s_seg.npy' % r.syn_id)[180:220, 180:220, 72:88]
    P = np.load('cubes/%s_pred.npy' % r.syn_id)[180:220, 180:220, 72:88]
    ss = np.asarray(seg[box]).squeeze(); ids = np.unique(ss); ids = ids[ids > 0]; lut = dict(zip(ids, ssd.sv2ssv_ids(ids)))
    m = np.vectorize(lambda v: lut.get(v, 0))(ss)
    print(r.syn_id, 'origin', bool((o == np.array([r.x, r.y, r.z]) - [200, 200, 80]).all()), '| raw==server', np.array_equal(R, np.asarray(raw[box]).squeeze()),
          '| pre', np.array_equal(S == 1, m == r.pre_id), '| post', np.array_equal(S == 2, m == r.post_id), '| sj vx at centre', int(P.sum()))
