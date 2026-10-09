#!/usr/bin/env python
"""cubes/<sid>_segssv.npy: 1 = every SV of the pre SSV, 2 = every SV of the post SSV.
(cubes/<sid>_seg.npy only marks the single SV whose id equals the SSV id.)"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gt_sj_pipeline import fetch_seg, WD
from syconn import global_params
global_params.wd = WD
from syconn.reps.super_segmentation import SuperSegmentationDataset
ssd = SuperSegmentationDataset(working_dir=WD)
df = pd.read_csv(os.environ.get('SAMPLE_CSV', 'sample/synapses.csv'))
if len(sys.argv) > 1: df = df[df.syn_id.isin(sys.argv[1:])]
for _, r in df.iterrows():
    out = 'cubes/%s_segssv.npy' % r.syn_id
    if os.path.exists(out): print(r.syn_id, 'cached'); continue
    origin = np.load('cubes/%s_origin.npy' % r.syn_id)
    S = np.array(np.load('cubes/%s_seg.npy' % r.syn_id, mmap_mode='r').shape)
    sv = fetch_seg(origin, S)
    ids = np.unique(sv); ids = ids[ids != 0]
    lut = ssd.sv2ssv_ids(ids)
    m = np.zeros(S, np.uint8)
    for sid_, tag in ((int(r.pre_id), 1), (int(r.post_id), 2)):
        svs = [i for i in ids if lut.get(int(i), 0) == sid_]
        m[np.isin(sv, svs)] = tag
        print(r.syn_id, tag, len(svs), 'SVs', int((m == tag).sum()), 'vx')
    np.save(out, m)
