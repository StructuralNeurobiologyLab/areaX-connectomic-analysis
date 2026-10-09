#!/usr/bin/env python
"""Download a raw EM cutout around each sampled synapse and cache it locally.

Annotating from a local cache rather than streaming keeps the GUI instant and
lets the annotator work offline.  A 4 x 4 x 4 um cube is 400 x 400 x 160 mag1
voxels, about 25 MB as uint8.

    python fetch_cubes.py                     # raw only
    python fetch_cubes.py --seg               # also the neuron segmentation

The segmentation is optional context for telling which of several nearby clefts
is the target one.  It never reveals the automated size, so it does not break
blinding.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATASET_SHAPE, RAW_LAYER, SEG_LAYER, VOXEL_NM


def open_volume(path, mip=0):
    from cloudvolume import CloudVolume
    # The server's provenance file is malformed JSON and CloudVolume refuses to
    # construct without this override.
    return CloudVolume(path, mip=mip, use_https=True, progress=False,
                       fill_missing=True, provenance={})


def cutout(cv, centre, half):
    lo = np.maximum(centre - half, 0)
    hi = np.minimum(centre + half, DATASET_SHAPE)
    block = cv[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return np.asarray(block)[..., 0], lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='sample/synapses.csv')
    ap.add_argument('--out-dir', default='cubes')
    ap.add_argument('--size-um', type=float, nargs=3, default=[4.0, 4.0, 4.0],
                    help='cube edge length in um (x y z)')
    ap.add_argument('--seg', action='store_true',
                    help='also fetch the neuron segmentation')
    ap.add_argument('--pred', action='store_true',
                    help="also fetch the network's sj prediction (control arm). "
                         "It must not be shown while painting: painting toward "
                         "the prediction makes the comparison circular.")
    ap.add_argument('--only', default=None,
                    help='comma-separated syn_ids, e.g. syn01')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    half = np.ceil(np.array(args.size_um) * 1000.0 / VOXEL_NM / 2).astype(int)
    print('cube half-extent: %s voxels (%s um)'
          % (half.tolist(), np.array(args.size_um).tolist()))

    df = pd.read_csv(args.csv)
    if args.only:
        want = set(args.only.split(','))
        df = df[df.syn_id.isin(want)]
        if df.empty:
            sys.exit('no syn_id matched %s' % args.only)
    os.makedirs(args.out_dir, exist_ok=True)

    raw = open_volume(RAW_LAYER)
    seg = open_volume(SEG_LAYER) if args.seg else None
    pred = None
    if args.pred:
        from gt_sj_pipeline import PRED_SJ, PRED_SJ_LABEL
        pred = open_volume('precomputed://file://' + PRED_SJ)

    for _, row in df.iterrows():
        centre = np.array([row.x, row.y, row.z], dtype=int)
        for name, cv in (('raw', raw), ('seg', seg), ('pred', pred)):
            if cv is None:
                continue
            path = os.path.join(args.out_dir, '%s_%s.npy' % (row.syn_id, name))
            # A cached file is only valid if it was cut at this coordinate.
            # Ids are reused when the sample is redrawn, so a name match alone
            # would silently hand back a cube from a different location.
            opath = os.path.join(args.out_dir, '%s_origin.npy' % row.syn_id)
            fresh = (os.path.exists(path) and os.path.exists(opath)
                     and np.array_equal(np.load(opath), centre - half))
            if fresh and not args.overwrite:
                print('%s %-4s cached' % (row.syn_id, name))
                continue
            block, lo = cutout(cv, centre, half)
            if name == 'pred':
                from gt_sj_pipeline import PRED_SJ_LABEL
                block = (block == PRED_SJ_LABEL).astype(np.uint8)
            if name == 'seg':
                # Only membership in the two partners matters, and uint64
                # labels would cost 205 MB per cube against 26 MB for this.
                mask = np.zeros(block.shape, dtype=np.uint8)
                mask[block == int(row.pre_id)] = 1
                mask[block == int(row.post_id)] = 2
                if not mask.any():
                    print('  warning: neither partner id found in the cube')
                block = mask
            np.save(path, block)
            # The cube's origin in dataset voxels; the GUI needs it to map a
            # click back to a global coordinate.
            np.save(os.path.join(args.out_dir, '%s_origin.npy' % row.syn_id), lo)
            print('%s %-4s %s  origin=%s  -> %s (%.1f MB)'
                  % (row.syn_id, name, block.shape, lo.tolist(), path,
                     block.nbytes / 1e6))


if __name__ == '__main__':
    main()
