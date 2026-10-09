#!/usr/bin/env python
"""Run SyConn's own synapse pipeline on MANUALLY PAINTED sj voxels.

This is the clean validation of the published synaptic-area measure.  Instead
of tracing lines and re-implementing an area estimator -- which drags in its
own meshing and obliquity artefacts -- feed the manual annotation in where the
*predicted* sj goes, and let the identical machinery produce the number:

    segmentation -> detect_cs -> closing/dilation -> intersect with sj
    -> cluster at cs_gap_nm -> Poisson mesh -> mesh_area -> /2

Everything downstream of the sj mask is byte-for-byte SyConn's, so any
difference in the output is a difference in the *annotation*, not in the
measurement.

Run in a SyConn environment (python with syconn, cloudvolume)

    python gt_sj_pipeline.py --gt <cube.npz> --out results/gt_areas.csv

Deviation to be aware of: contacts are detected on the *agglomerated* (SSV)
segmentation rather than on supervoxels followed by an SV->SSV mapping.  The
two agree on which voxels lie between two different neurons -- intra-SSV
self-contacts are dropped either way -- but the per-object closing acts on
whole-neuron contacts here rather than on SV fragments.
"""

import argparse
import os
import re
import sys

import numpy as np
from scipy import ndimage

import os
# SyConn working directory of j0251 (the sj prediction volume is read from it); override with SYN_AREA_WD
WD = os.environ.get('SYN_AREA_WD', '/cajal/nvmescratch/projects/data/songbird_tmp/j0251/j0251_72_seg_20210127_agglo2_syn_20220811')
# The network's own sj prediction, for the control arm (the pipeline itself
# needs only the manual mask). Labels of this volume: {1: mito, 2: sj, 3: vc}.
PRED_SJ = WD + '/j0251_72_mito_sj_vc.20220811.var0.full.76152679'
PRED_SJ_LABEL = 2
SEG_LAYER = ('precomputed://https://syconn.esc.mpcdf.mpg.de/'
             'j0251_72_seg_20210127_agglo2/segmentation')


def load_gt(path):
    """Manual sj labels plus the cube's corner offset, taken from the name."""
    m = re.search(r'offset_(\d+)_(\d+)_(\d+)', os.path.basename(path))
    if not m:
        raise SystemExit('cannot parse offset from %s' % path)
    corner = np.array([int(m.group(i)) for i in (1, 2, 3)])
    if path.endswith('.npz'):
        lab = np.load(path)['labels']
    else:
        import h5py
        with h5py.File(path, 'r') as f:
            lab = np.asarray(f['labels'])
    return (lab > 0).astype('u1'), corner


def fetch_seg(offset, size):
    from cloudvolume import CloudVolume
    cv = CloudVolume(SEG_LAYER, mip=0, use_https=True, progress=False,
                     fill_missing=True, provenance={})
    lo = np.maximum(offset, 0)
    hi = np.minimum(offset + size, np.array(cv.shape[:3]))
    block = np.asarray(cv[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])[..., 0]
    out = np.zeros(size, dtype=block.dtype)
    s = lo - offset
    out[s[0]:s[0] + block.shape[0], s[1]:s[1] + block.shape[1],
        s[2]:s[2] + block.shape[2]] = block
    return out


def contacts_from_seg(seg, cs_filtersize, cs_dilation):
    """detect_cs + the per-object closing/dilation, as extract_contact_sites does."""
    from syconn.extraction.find_object_properties import (detect_cs,
                                                          find_object_properties)
    if seg.max() > np.iinfo(np.uint32).max:
        print('  WARNING: ids exceed uint32; the pipeline casts to uint32 too, '
              'so this matches production but ids may collide')
    contacts = np.asarray(detect_cs(seg.astype(np.uint32, copy=False)))

    _, bb_dc, _ = find_object_properties(contacts)
    n_closings = int(max(np.array(cs_filtersize) // 2))
    for ix in bb_dc.keys():
        st, en = np.array(bb_dc[ix])
        st = np.maximum(st - n_closings, 0)
        en = en + n_closings
        sl = tuple(slice(st[i], en[i]) for i in range(3))
        sub = contacts[sl]
        binary = (sub == ix).astype(np.int8, copy=False)
        res = (ndimage.binary_closing(binary, iterations=n_closings)
               if n_closings > 0 else binary)
        if cs_dilation > 0:
            res = ndimage.binary_dilation(res, iterations=cs_dilation)
        proc = ((binary == 1) | (sub == 0)) & (res == 1)
        contacts[sl][proc] = res[proc] * ix
    return contacts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True, help='ground-truth cube (.npz/.h5)')
    ap.add_argument('--predicted', action='store_true',
                    help='CONTROL: use the network sj prediction over the same '
                         'cube instead of the manual mask. Feeding the real '
                         'input back through this script must reproduce the '
                         'published areas; if it does not, the reimplementation '
                         'is wrong and no manual comparison means anything.')
    ap.add_argument('--no-morph', action='store_true',
                    help='skip the binary_opening/closing the pipeline applies '
                         'to the predicted sj mask (default: apply it, so the '
                         'manual mask goes through the identical path)')
    ap.add_argument('--pad', type=int, default=0,
                    help='CONTROL only: enlarge the analysed region by N voxels '
                         'on every side. Production saw the whole dataset, so a '
                         'synapse crossing the cube face is clipped here but not '
                         'there; padding tests how much of the disagreement that '
                         'explains. Meaningless for manual sj, which exists only '
                         'inside the cube.')
    ap.add_argument('--out', default=None, help='CSV of the resulting synapses')
    ap.add_argument('--save-meshes', default=None, help='directory for .pkl meshes')
    ap.add_argument('--save-voxels', default=None,
                    help='directory for per-object voxel masks (.npz, cube '
                         'local coords), so an object can be overlaid on EM')
    args = ap.parse_args()

    from syconn import global_params
    global_params.wd = WD
    from syconn.proc.meshes import calc_contact_syn_mesh, mesh_area_calc
    from syconn.extraction.cs_processing_steps import connected_cluster_kdtree
    from syconn.reps import connectivity_helper as ch
    from syconn.proc import sd_proc as _  # noqa: keeps import side effects
    from syconn.handler.basics import majority_element_1d  # noqa

    cfg = global_params.config
    co = cfg['cell_objects']
    scaling = np.array(cfg['scaling'])
    mesh_kws = cfg['meshes']['meshing_props_points']['syn_ssv']
    min_vx = co['min_obj_vx']['syn_ssv']
    mesh_min = cfg['meshes']['mesh_min_obj_vx']
    cs_filtersize = np.array(co['cs_filtersize'])
    stencil = cs_filtersize // 2
    overlap = int(max(stencil))

    sj, corner = load_gt(args.gt)
    S = np.array(sj.shape)
    print('GT cube %s at corner %s, %d painted voxels'
          % (S.tolist(), corner.tolist(), int(sj.sum())))
    if args.predicted and args.pad:
        corner = corner - args.pad
        S = S + 2 * args.pad
        print('  padded region to %s at corner %s' % (S.tolist(), corner.tolist()))
    if args.predicted:
        from cloudvolume import CloudVolume
        pcv = CloudVolume('precomputed://file://' + PRED_SJ, mip=0,
                          progress=False, fill_missing=True, provenance={})
        sj = (np.asarray(pcv[corner[0]:corner[0] + S[0],
                             corner[1]:corner[1] + S[1],
                             corner[2]:corner[2] + S[2]])[..., 0]
              == PRED_SJ_LABEL).astype('u1')
        print('  CONTROL: replaced by the predicted sj, %d voxels'
              % int(sj.sum()))

    if not args.no_morph:
        from syconn.proc.image import apply_morphological_operations
        struct = ndimage.generate_binary_structure(3, 1)
        sj = apply_morphological_operations(
            sj.copy(), co['extract_morph_op']['sj'],
            mop_kwargs=dict(structure=struct)).astype('u1', copy=False)
        print('  after %s: %d voxels'
              % (co['extract_morph_op']['sj'], int(sj.sum())))

    # detect_cs does a valid convolution, so load stencil margin on top of the
    # closing margin; the result is then cropped back to the GT cube.
    seg_off = corner - overlap - stencil
    seg_size = S + 2 * overlap + 2 * stencil
    seg = fetch_seg(seg_off, seg_size)
    print('  segmentation %s, %d distinct ids'
          % (seg.shape, len(np.unique(seg))))

    contacts = contacts_from_seg(seg, cs_filtersize, co['cs_dilation'])
    print('  contacts %s, %d contact objects'
          % (contacts.shape, len(np.unique(contacts)) - 1))

    # sj is only known inside the GT cube; pad the closing margin with zeros
    sj_pad = np.zeros(S + 2 * overlap, dtype='u1')
    sj_pad[overlap:-overlap, overlap:-overlap, overlap:-overlap] = sj
    contacts[sj_pad == 0] = 0                      # <- syn = cs AND sj
    syn = contacts[overlap:-overlap, overlap:-overlap, overlap:-overlap]
    print('  syn voxels after intersection: %d' % int((syn > 0).sum()))

    # Production maps each cs object's SV pair to SSVs, drops pairs where a
    # partner belongs to no SSV or where both map to the same SSV (a neuron
    # touching itself), and pools the surviving fragments per SSV pair.  Also
    # drops syn objects below min_obj_vx['syn'].
    from syconn.reps.super_segmentation import SuperSegmentationDataset
    ssd = SuperSegmentationDataset(working_dir=WD)
    cs_ids = np.unique(syn)
    cs_ids = cs_ids[cs_ids != 0]
    pairs_sv = ch.cs_id_to_partner_ids_vec(cs_ids)
    lut = ssd.sv2ssv_ids(np.unique(pairs_sv.flatten()))
    by_ssv_pair = {}
    n_small = n_unmapped = n_self = 0
    for cs_id, (sv_a, sv_b) in zip(cs_ids, pairs_sv):
        vx_local = np.argwhere(syn == cs_id)
        if len(vx_local) < co['min_obj_vx']['syn']:
            n_small += 1
            continue
        a, b = lut.get(int(sv_a), 0), lut.get(int(sv_b), 0)
        if a == 0 or b == 0:
            n_unmapped += 1
            continue
        if a == b:
            n_self += 1
            continue
        by_ssv_pair.setdefault((min(a, b), max(a, b)), []).append(vx_local)
    print('  cs objects: %d -> %d SSV pairs (dropped %d < %d vx, %d unmapped SV, '
          '%d intra-SSV)' % (len(cs_ids), len(by_ssv_pair), n_small,
                             co['min_obj_vx']['syn'], n_unmapped, n_self))

    rows = []
    for partners, frags in by_ssv_pair.items():
        vx = np.concatenate(frags) + corner
        # pass the fragments separately, exactly as combine_and_split_syn does
        ccs = connected_cluster_kdtree([f + corner for f in frags],
                                       dist_intra_object=co['cs_gap_nm'],
                                       dist_inter_object=20000, scale=scaling)
        for cc in ccs:
            v = vx[np.array(list(cc))]
            if len(v) < min_vx:
                continue
            loc = v - corner
            touches = bool((loc.min(0) == 0).any() or (loc.max(0) == S - 1).any())
            rec = dict(pre_or_post_a=int(partners[0]), b=int(partners[1]),
                       n_voxels=len(v),
                       x=int(np.round(v[:, 0].mean())),
                       y=int(np.round(v[:, 1].mean())),
                       z=int(np.round(v[:, 2].mean())),
                       touches_border=touches)
            if len(v) > mesh_min:
                class _S(object):
                    type = 'syn_ssv'
                    def __init__(s, vv, sc): s._voxel_list, s.scaling = vv, sc
                mesh = calc_contact_syn_mesh(_S(v, scaling), **mesh_kws)[0]
                area = mesh_area_calc(mesh)
                rec['mesh_area'] = area
                rec['size_um2'] = area / 2
                if args.save_meshes:
                    import pickle
                    os.makedirs(args.save_meshes, exist_ok=True)
                    pickle.dump((np.array(mesh[1]).reshape(-1, 3),
                                 np.array(mesh[0]).reshape(-1, 3)),
                                open(os.path.join(args.save_meshes,
                                     'gt_%d_%d_%d.pkl' % (rec['x'], rec['y'],
                                                          rec['z'])), 'wb'))
            else:
                rec['mesh_area'] = 0.0
                rec['size_um2'] = 0.0
            if args.save_voxels:
                os.makedirs(args.save_voxels, exist_ok=True)
                np.savez_compressed(
                    os.path.join(args.save_voxels, 'obj_%d_%d_%d_%d_%d.npz'
                                 % (partners[0], partners[1], rec['x'],
                                    rec['y'], rec['z'])),
                    voxels=loc.astype('i4'), shape=np.array(S))
            rows.append(rec)

    import pandas as pd
    df = pd.DataFrame(rows).sort_values('size_um2', ascending=False)
    print('\n%d manual synapses (%d away from the cube border)'
          % (len(df), int((~df.touches_border).sum()) if len(df) else 0))
    if len(df):
        print(df.to_string(index=False))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        df.to_csv(args.out, index=False)
        print('wrote %s' % args.out)


if __name__ == '__main__':
    main()
