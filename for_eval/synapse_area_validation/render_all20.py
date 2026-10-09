#!/usr/bin/env python
"""Multipanel 3D figure: pre/post surfaces, predicted sj mesh, painted sj mesh.

One panel per synapse. Meshes for both sj variants come from SyConn's own
`calc_contact_syn_mesh`, run on the voxels the pipeline assigned to the
object on the CSV's named (pre, post) pair -- so the areas printed are the
same quantity the published `size` is.

    python render_all20.py --csv <sample csv> --out figures/final40.png   (SyConn environment; writes final40_areas.csv)
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import VOXEL_NM

GOLD, GREEN, RED, BLUE = '#c8901a', '#1baf7a', '#d1495b', '#3a7ca5'


def mesher():
    from syconn import global_params
    from syconn.proc.meshes import calc_contact_syn_mesh, mesh_area_calc
    from gt_sj_pipeline import WD
    global_params.wd = WD
    cfg = global_params.config
    scaling = np.array(cfg['scaling'])
    kws = cfg['meshes']['meshing_props_points']['syn_ssv']

    class _S(object):
        type = 'syn_ssv'
        def __init__(s, vv, sc): s._voxel_list, s.scaling = vv, sc

    def run(vox_global):
        m = calc_contact_syn_mesh(_S(vox_global, scaling), **kws)[0]
        verts = np.array(m[1]).reshape(-1, 3)
        faces = np.array(m[0]).reshape(-1, 3)
        return verts, faces, mesh_area_calc(m) / 2.0
    return run


def pick_object(d, pre, post, coord_global):
    """Dumped objects on the CSV's named pair: (nearest to the synapse
    coordinate, [the other clusters of the same pair])."""
    want = {int(pre), int(post)}
    found = []
    for f in glob.glob(os.path.join(d, 'obj_*.npz')):
        a, b, x, y, z = re.match(r'obj_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)',
                                 os.path.basename(f)).groups()
        if {int(a), int(b)} == want:
            dist = np.linalg.norm((np.array([x, y, z], float) - coord_global) * VOXEL_NM)
            found.append((dist, np.load(f)['voxels']))
    if not found:
        return None, []
    found.sort(key=lambda t: t[0])
    return found[0][1], [v for _, v in found[1:]]


def surface(mask, lo, step=2):
    """Down-sampled isosurface of a binary cube region, in nm."""
    sub = mask[lo[0]:lo[0]+SPAN[0], lo[1]:lo[1]+SPAN[1], lo[2]:lo[2]+SPAN[2]]
    sub = sub[::step, ::step, :]
    if sub.sum() < 20:
        return None
    try:
        v, f, _, _ = measure.marching_cubes(sub.astype(np.float32), 0.5,
                                            step_size=2)
    except (RuntimeError, ValueError):
        return None
    v = v * np.array([step, step, 1]) + lo
    return v * VOXEL_NM, f


SPAN = np.array([110, 110, 44])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='sample/synapses.csv')
    ap.add_argument('--obj-dir', default='objs')
    ap.add_argument('--cube-dir', default='cubes')
    ap.add_argument('--out', default='figures/final20.png')
    ap.add_argument('--only', default=None)
    args = ap.parse_args()

    mesh = mesher()
    df = pd.read_csv(args.csv)
    if args.only:
        df = df[df.syn_id.isin(args.only.split(','))]

    ncol = 5 if len(df) > 4 else len(df)
    nrow = int(np.ceil(len(df) / float(ncol)))
    fig = plt.figure(figsize=(4.2 * ncol, 4.6 * nrow))
    recs = []
    for k, (_, r) in enumerate(df.iterrows()):
        ax = fig.add_subplot(nrow, ncol, k + 1, projection='3d')
        origin = np.load(os.path.join(args.cube_dir, '%s_origin.npy' % r.syn_id))
        segp = os.path.join(args.cube_dir, '%s_segssv.npy' % r.syn_id)
        if not os.path.exists(segp):       # fall back to the single-SV mask
            segp = os.path.join(args.cube_dir, '%s_seg.npy' % r.syn_id)
        seg = np.load(segp)
        coord_global = np.array([r.x, r.y, r.z], float)
        rec = dict(syn=r.syn_id, x=int(r.x), y=int(r.y), z=int(r.z),
                   pre_id=int(r.pre_id), post_id=int(r.post_id),
                   api=r['size'], pred=np.nan, manual=np.nan)

        drawn = []
        for tag, col, key in (('pred', GREEN, 'pred'), ('manual', GOLD, 'manual')):
            v, others = pick_object(
                os.path.join(args.obj_dir, '%s_%s' % (r.syn_id, tag)),
                r.pre_id, r.post_id, coord_global)
            if v is None:
                continue
            verts, faces, area = mesh(v + origin)
            # the mesher returns global nm; everything else here is cube-local
            verts = verts - origin * np.array(VOXEL_NM, dtype=float)
            rec[key] = area
            ax.add_collection3d(Poly3DCollection(
                verts[faces], facecolor=col, edgecolor='none', alpha=1.0, zorder=10))
            drawn.append(verts)
            # other clusters of the same pair in the cube, drawn faint: these
            # are what a 250 nm gap would or would not merge
            for ov in others:
                if len(ov) <= 100:
                    continue
                overts, ofaces, _ = mesh(ov + origin)
                overts = overts - origin * np.array(VOXEL_NM, dtype=float)
                ax.add_collection3d(Poly3DCollection(
                    overts[ofaces], facecolor=col, edgecolor='none', alpha=0.25, zorder=5))

        if drawn:
            c = np.concatenate(drawn).mean(0) / VOXEL_NM
        else:
            c = (np.array([r.x, r.y, r.z]) - origin).astype(float)
        lo = np.clip(np.round(c - SPAN / 2).astype(int), 0,
                     np.array(seg.shape) - SPAN)

        for lab, col in ((1, RED), (2, BLUE)):
            s = surface(seg == lab, lo)
            if s is None:
                continue
            v, f = s
            ax.add_collection3d(Poly3DCollection(
                v[f], facecolor=col, edgecolor='none', alpha=0.10, zorder=1))

        ctr = (lo + SPAN / 2.0) * VOXEL_NM
        half = (SPAN * VOXEL_NM).max() / 2.0
        for setlim, i in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
            setlim(ctr[i] - half, ctr[i] + half)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=18, azim=-60)
        ax.set_axis_off()
        fmt = lambda v: ('%.3f' % v) if v == v else '--'
        ax.set_title('%s   %s\u2192%s (%s)\n'
                     'xyz %d %d %d   pre %d   post %d\n'
                     'API %.3f    predicted %s    painted %s' %
                     (r.syn_id, r.pre_type, r.post_type, r.post_morph_name,
                      int(r.x), int(r.y), int(r.z), int(r.pre_id), int(r.post_id),
                      r['size'], fmt(rec['pred']), fmt(rec['manual'])),
                     fontsize=8.5, linespacing=1.35)
        recs.append(rec)
        print('%s  api %.3f  pred %.3f  manual %.3f'
              % (r.syn_id, r['size'], rec['pred'], rec['manual']))

    fig.suptitle('Synapse area validation, n=%d   |   green = sj predicted by the '
                 'network, gold = manually painted sj, red/blue = pre/post cell '
                 '(all areas in um2)' % len(df), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=105)
    pd.DataFrame(recs).to_csv(
        os.path.splitext(args.out)[0] + '_areas.csv', index=False)
    print('wrote %s' % args.out)


if __name__ == '__main__':
    main()
