#!/usr/bin/env python
"""Draw a size-stratified sample of synapses from the j0251 REST API.

The sample is taken from exactly the pool the paper analyses (prob >= 0.6,
size >= 0.1 um^2) so that the validation transfers to the published numbers.
Sizes are stratified into log-spaced bins because the distribution is heavily
skewed -- a uniform random draw would put almost nothing above 1 um^2, and the
regression against the automated measure needs the whole range.

    python sample_synapses.py --n-per-bin 4 --out sample/synapses.csv
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (API, CELLTYPES, DATASET_SHAPE, MORPH, PROB_MIN, SIZE_MIN,
                    VERSION, VOXEL_NM)

FIELDS = ['x', 'y', 'z', 'size', 'prob', 'pre_id', 'post_id', 'post_morph',
          'pre_celltype', 'post_celltype']


def query(pool_size, size_min=SIZE_MIN):
    params = dict(max_results=pool_size, prob_min=PROB_MIN, size_min=size_min,
                  return_fields=','.join(FIELDS), version=VERSION)
    url = '%s?%s' % (API, urllib.parse.urlencode(params))
    print('querying %d synapses ...' % pool_size)
    with urllib.request.urlopen(url, timeout=300) as fh:
        d = json.loads(fh.read().decode())
    df = pd.DataFrame({k: d[k] for k in FIELDS})
    print('  got %d' % len(df))
    return df


def drop_border(df, margin_um):
    """Reject synapses too close to the volume edge to hold a full cutout."""
    margin_vx = np.ceil(margin_um * 1000.0 / VOXEL_NM).astype(int)
    ok = np.ones(len(df), dtype=bool)
    for i, ax in enumerate('xyz'):
        ok &= (df[ax] >= margin_vx[i]) & (df[ax] < DATASET_SHAPE[i] - margin_vx[i])
    print('  %d of %d survive the %.1f um border margin'
          % (ok.sum(), len(df), margin_um))
    return df[ok].reset_index(drop=True)


NON_NEURONAL = ['ASTRO', 'FRAG', 'MICRO', 'MIGR', 'OLIGO',
                'unclassified']


def drop_non_neuronal(df):
    """Keep only synapses between cell types the connectome analyses use.

    Glia and FRAG appearing as a synaptic partner are almost always false
    detections.  Including them would mix detection errors into what is meant
    to be a measurement of *size* accuracy.
    """
    bad = {CELLTYPES.index(n) for n in NON_NEURONAL}
    ok = ~(df.pre_celltype.isin(bad) | df.post_celltype.isin(bad))
    print('  %d of %d have neuronal partners on both sides'
          % (ok.sum(), len(df)))
    return df[ok].reset_index(drop=True)


def stratify(df, n_bins, n_per_bin, rng):
    """Uniform random pick from each of n_bins log-spaced size bins.

    Log spacing because the size distribution is strongly right-skewed: a flat
    random draw would give almost nothing above 1 um^2, and the regression
    against the automated measure is only informative if it spans the range.
    The top bin is left open-ended so it is not starved by the few very large
    outliers that set the maximum.
    """
    top = float(np.percentile(df['size'], 99.5))
    edges = list(np.geomspace(df['size'].min(), top, n_bins))
    edges.append(np.inf)
    picked = []
    for b in range(n_bins):
        cand = df[(df['size'] >= edges[b]) & (df['size'] < edges[b + 1])]
        take = min(n_per_bin, len(cand))
        if take < n_per_bin:
            print('  bin %d: only %d candidates' % (b, len(cand)))
        sel = cand.sample(n=take, random_state=int(rng.integers(1 << 31)))
        sel = sel.copy()
        sel['size_bin'] = b
        picked.append(sel)
        hi = '%.3f' % edges[b + 1] if np.isfinite(edges[b + 1]) else 'inf'
        print('  bin %d [%.3f, %s): %d candidates -> %d picked'
              % (b, edges[b], hi, len(cand), take))
    out = pd.concat(picked).reset_index(drop=True)
    for c in ['x', 'y', 'z', 'pre_id', 'post_id', 'post_morph', 'pre_celltype',
              'post_celltype', 'size_bin']:
        out[c] = out[c].astype('int64')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool-size', type=int, default=50000,
                    help='random synapses to draw from before stratifying')
    ap.add_argument('--n-bins', type=int, default=5)
    ap.add_argument('--n-per-bin', type=int, default=4,
                    help='4 x 5 bins = the 20-synapse pilot')
    ap.add_argument('--margin-um', type=float, default=2.5,
                    help='keep synapses this far from the volume border')
    ap.add_argument('--size-min', type=float, default=SIZE_MIN,
                    help='lower size bound of the pool in um^2; the paper uses '
                         '0.1, a lower value extends the validated range below '
                         'the analysed pool')
    ap.add_argument('--exclude', nargs='*', default=[],
                    help='CSVs whose synapses must not be drawn again')
    ap.add_argument('--shuffle', action='store_true',
                    help='write the rows in random order and number them in '
                         'that order, so annotation order is uncorrelated '
                         'with size')
    ap.add_argument('--seed', type=int, default=20260831)
    ap.add_argument('--out', default='sample/synapses.csv')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = query(args.pool_size, args.size_min)
    df = drop_border(df, args.margin_um)
    df = drop_non_neuronal(df)
    if args.exclude:
        seen = set()
        for f in args.exclude:
            e = pd.read_csv(f)
            seen |= set(zip(e.x, e.y, e.z))
        keep = ~pd.Series(list(zip(df.x, df.y, df.z))).isin(seen)
        print('  excluding %d already-chosen synapses' % (~keep).sum())
        df = df[keep.values].reset_index(drop=True)
    sel = stratify(df, args.n_bins, args.n_per_bin, rng)

    if args.shuffle:
        sel = sel.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        sel = sel.reset_index(drop=True)
    sel.insert(0, 'syn_id', ['syn%02d' % (i + 1) for i in range(len(sel))])
    sel['pre_type'] = [CELLTYPES[c] for c in sel.pre_celltype]
    sel['post_type'] = [CELLTYPES[c] for c in sel.post_celltype]
    sel['post_morph_name'] = [MORPH.get(m, str(m)) for m in sel.post_morph]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sel.to_csv(args.out, index=False)
    print('\nwrote %s (%d synapses)' % (args.out, len(sel)))
    cols = ['syn_id', 'x', 'y', 'z', 'size', 'prob', 'pre_type', 'post_type',
            'post_morph_name', 'size_bin']
    print(sel[cols].to_string(index=False))
    print('\npostsynaptic morphology mix:')
    print(sel.post_morph_name.value_counts().to_string())


if __name__ == '__main__':
    main()
