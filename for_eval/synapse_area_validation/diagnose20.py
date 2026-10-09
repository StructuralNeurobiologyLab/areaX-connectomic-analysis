#!/usr/bin/env python
"""Per-synapse diagnosis of published vs painted object, from the --save-voxels dumps.

Never touches the raw sj class.  For each synapse:
  * manual object = dump on the named (pre,post) pair nearest the CSV coordinate
  * published object = predicted dump on the named pair nearest the CSV coordinate
  * splits the published object into face-connected patches (26-conn), then
    reports how many patches, their sizes, gaps, and which one(s) the paint hits
  * overlap of the RAW paint with every predicted object of ANY pair, to see
    where the paint went if not onto the named pair
"""
import glob, os, re, sys, json
import numpy as np, pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import VOXEL_NM

def objs(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, 'obj_*.npz'))):
        a, b, x, y, z = map(int, re.match(r'obj_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)', os.path.basename(f)).groups())
        out.append(dict(a=a, b=b, c=np.array([x, y, z]), v=np.load(f)['voxels'], f=os.path.basename(f)))
    return out

def mask_of(v, S):
    m = np.zeros(S, bool); m[tuple(v.T)] = True; return m

def patches(v, S):
    m = mask_of(v, S)
    lab, n = ndimage.label(m, structure=np.ones((3, 3, 3)))
    sizes = ndimage.sum(m, lab, range(1, n + 1)).astype(int)
    return lab, sizes

def extent_nm(v):
    return ((v.max(0) - v.min(0) + 1) * VOXEL_NM)

def main():
    df = pd.read_csv(os.environ.get('SAMPLE_CSV', 'sample/synapses.csv'))
    ft = pd.read_csv('results/final_table.csv').set_index('syn') if 'SAMPLE_CSV' not in os.environ else None
    rows = []
    for _, r in df.iterrows():
        sid = r.syn_id
        dm, dp = 'objs/%s_manual' % sid, 'objs/%s_pred' % sid
        if not (os.path.isdir(dm) and os.path.isdir(dp)):
            print(sid, 'dumps missing'); continue
        origin = np.load('cubes/%s_origin.npy' % sid)
        gtf = glob.glob(os.path.join(os.environ.get('GT_DIR', 'gt_manual/final'), '%s_*_sj.npz' % sid))[0]
        raw = np.load(gtf)['labels'].astype(bool)
        S = np.array(raw.shape)
        coord = np.array([r.x, r.y, r.z]) - origin
        want = {int(r.pre_id), int(r.post_id)}
        M, P = objs(dm), objs(dp)
        Mp = [o for o in M if {o['a'], o['b']} == want]
        Pp = [o for o in P if {o['a'], o['b']} == want]
        dist = lambda o: np.linalg.norm((o['c'] - origin - coord) * VOXEL_NM)
        if not Pp:
            print(sid, 'NO predicted object on named pair'); continue
        pub = min(Pp, key=dist)
        man = min(Mp, key=dist) if Mp else None
        rec = dict(syn=sid, api=r['size'], manual_um2=(ft.loc[sid, 'manual_um2'] if ft is not None else np.nan),
                   pub_nvx=len(pub['v']), pub_dist_nm=round(dist(pub)),
                   n_pred_objs_pair=len(Pp), pred_pair_sizes=' '.join(str(len(o['v'])) for o in sorted(Pp, key=dist)),
                   raw_paint_vx=int(raw.sum()), man_nvx=len(man['v']) if man else 0,
                   n_man_objs_pair=len(Mp))
        ex = extent_nm(pub['v']); rec['pub_extent_um'] = 'x'.join('%.2f' % (e / 1000) for e in ex)
        rx = extent_nm(np.argwhere(raw)); rec['paint_extent_um'] = 'x'.join('%.2f' % (e / 1000) for e in rx)
        # patches of the published object
        lab, sizes = patches(pub['v'], S)
        order = np.argsort(-sizes)
        rec['pub_n_patches'] = len(sizes)
        rec['pub_patch_sizes'] = ' '.join(str(sizes[i]) for i in order[:6])
        rec['pub_n_patches_ge100'] = int((sizes >= 100).sum())
        # which patches does the raw paint hit (paint dilated by 1 voxel)
        pd1 = ndimage.binary_dilation(raw, iterations=1)
        hit = np.unique(lab[pd1 & (lab > 0)])
        rec['paint_hits_patches'] = ' '.join('%d(%d)' % (h, sizes[h - 1]) for h in hit)
        # fraction of raw paint inside the published object (dilated 1) and vice versa
        pubm = mask_of(pub['v'], S)
        pub1 = ndimage.binary_dilation(pubm, iterations=1)
        rec['paint_in_pub'] = round(float((raw & pub1).sum() / raw.sum()), 2)
        # fraction of published voxels within 100 nm of any raw paint voxel
        t = cKDTree(np.argwhere(raw) * VOXEL_NM)
        dd, _ = t.query(pub['v'] * VOXEL_NM, distance_upper_bound=100)
        rec['pub_within100nm_of_paint'] = round(float(np.isfinite(dd).mean()), 2)
        dd2, _ = t.query(pub['v'] * VOXEL_NM)
        rec['pub_max_dist_to_paint_um'] = round(float(dd2.max()) / 1000, 2)
        # where else did the paint go?  overlap of paint(+1 vx) with every predicted object of any pair
        other = []
        anym = np.zeros(S, bool)
        for o in P:
            idx = tuple(o['v'].T)
            anym[idx] = True
            n = int(pd1[idx].sum())
            if n > 0.05 * raw.sum():
                other.append((n, o))
        other.sort(key=lambda t: -t[0])
        rec['paint_overlaps'] = '; '.join('%s-%s:%d' % (o['a'], o['b'], n) + ('*' if {o['a'], o['b']} == want else '') for n, o in other[:4])
        anym = ndimage.binary_dilation(anym, iterations=1)
        rec['paint_in_no_pred_obj'] = round(float((raw & ~anym).sum() / raw.sum()), 2)
        # registration test: shift the paint by (dx,dy,dz) and find the shift that
        # maximises overlap with the published object.  A shared non-zero shift
        # across synapses = raw/prediction offset; scattered = not registration.
        best = (-1, (0, 0, 0)); base = None
        pv = np.argwhere(raw)
        for dx in range(-8, 9, 2):
            for dy in range(-8, 9, 2):
                for dz in range(-3, 4):
                    q = pv + [dx, dy, dz]
                    ok = (q >= 0).all(1) & (q < S).all(1)
                    n = int(pub1[tuple(q[ok].T)].sum())
                    if (dx, dy, dz) == (0, 0, 0): base = n
                    if n > best[0]: best = (n, (dx, dy, dz))
        rec['best_shift_vx'] = '%d %d %d' % best[1]
        rec['overlap_gain_at_best_shift'] = round(best[0] / max(base, 1), 2)
        # manual object vs pub
        if man is not None:
            mm = mask_of(man['v'], S)
            rec['manobj_in_pub'] = round(float((mm & pub1).sum() / mm.sum()), 2)
            rec['man_dist_nm'] = round(dist(man))
        # manual arm without the morph ops, if dumped
        dnm = 'objs/%s_manual_nomorph' % sid
        if os.path.isdir(dnm):
            Mn = [o for o in objs(dnm) if {o['a'], o['b']} == want]
            rec['man_nomorph_nvx'] = len(min(Mn, key=dist)['v']) if Mn else 0
            nmcsv = 'results/final/%s_manual_nomorph_objs.csv' % sid
            if os.path.exists(nmcsv) and Mn:
                t2 = pd.read_csv(nmcsv); o = min(Mn, key=dist)
                m2 = t2[(t2.x == o['c'][0]) & (t2.y == o['c'][1]) & (t2.z == o['c'][2])]
                rec['man_nomorph_um2'] = round(float(m2.size_um2.iloc[0]), 4) if len(m2) else np.nan
        rows.append(rec)
        print(sid, json.dumps({k: (v if isinstance(v, str) else (float(v) if not isinstance(v, (int, np.integer)) else int(v))) for k, v in rec.items()}, indent=None))
    out = pd.DataFrame(rows)
    out.to_csv(os.environ.get('DIAG_OUT', 'results/diagnose20.csv'), index=False)
    print('wrote results/diagnose20.csv')

if __name__ == '__main__':
    main()
