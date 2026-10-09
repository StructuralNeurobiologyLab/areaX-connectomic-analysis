"""Per-cell morphology table for all full-cell classes in ONE pass (replaces general/analysis_prep.py and the
repeated per-script skeleton passes), written in the prep-cache format of analysis_prep.py plus one flat csv.

Stage 1 (skeleton only, every SSV of the full-cell classes, ~190 k): is the cell "full" (axoness_avg10000 with
    3,4 -> 1 contains dendrite 0, axon 1 and soma 2)? analysis_prep.find_full_cells loaded skeleton + mesh + vertex labels
    for every SSV; here only the skeleton is loaded for the pre-check.
Stage 2 (full cells only, ~10 k): analysis_prep_func.get_per_cell_morphology_params (axon/dendrite length,
    complete pathlength, compartment mesh areas via the SyConn-fork compartmentalize_mesh_fromskel(k=1), soma
    centre + radius from the vertex labels) plus the per-cell workers other scripts recompute on their own skeleton
    pass: get_spine_density, get_median_comp_radii_cell, get_myelin_fraction, get_dendrite_info_cell,
    check_cutoff_dendrites (7 um), cell volume from cell.size.
Stage 3: synapse counts/summed sizes per compartment from the synapse cache (synapse_amount_percell), as in
    analysis_prep.py.

Outputs in $BIO_ANALYSIS_PREP: full_<CT>_dict.pkl / full_<CT>_arr.pkl (the keys of analysis_prep.py, so every script works
unchanged), cell_numbers.csv, percell_morph.csv (all cells x all columns), stage1_fullness.npz.

    rerun/run.sh -m bio_analysis.rerun.build_percell_table [--nb_cpus 64] [--celltypes MSN,STN,...] [--limit N]
"""
import argparse
import os
import time
import traceback

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--version', default='v6')
parser.add_argument('--nb_cpus', type=int, default=None)
parser.add_argument('--celltypes', default=None, help='comma separated subset, e.g. GPe,GPi (default: all full-cell classes)')
parser.add_argument('--limit', type=int, default=None, help='debug: first N SSVs per class')
parser.add_argument('--syn_prob', type=float, default=0.6)
parser.add_argument('--min_syn_size', type=float, default=0.1)
parser.add_argument('--cutoff_dist', type=float, default=7000., help='nm, dendrite-to-border rule (Methods: 7 um)')
parser.add_argument('--skip_stage1', action='store_true', help='reuse stage1_fullness.npz')
parser.add_argument('--skip_stage2', action='store_true', help='reuse stage2_results.pkl')
args = parser.parse_args()

assert 'BIO_ANALYSIS_PREP' in os.environ, 'set BIO_ANALYSIS_PREP (rerun/run.sh does this)'
from syconn import global_params
from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
from syconn.reps.segmentation import SegmentationDataset
from bio_analysis.rerun.mp import pmap
from syconn.handler.basics import write_obj2pkl
from syconn.handler.config import initialize_logging
from bio_analysis.general.analysis_params import Analysis_Params
from bio_analysis.general.analysis_prep_func import get_per_cell_morphology_params, synapse_amount_percell
from bio_analysis.general.analysis_morph_helper import (get_spine_density, get_median_comp_radii_cell, get_myelin_fraction,
                                                        get_dendrite_info_cell, check_cutoff_dendrites)
from bio_analysis.general.analysis_conn_helper import filter_synapse_caches_general

ap = Analysis_Params(version=args.version)
global_params.wd = ap.working_dir()
prep = ap.file_locations
os.makedirs(prep + 'logs', exist_ok=True)
log = initialize_logging('build_percell_table', log_dir=prep + 'logs/')
ct_dict = ap.ct_dict(with_glia=False)
full_cts = [ct for ct in ct_dict if ct not in ap.axon_cts()]
if args.celltypes:
    wanted = args.celltypes.split(',')
    full_cts = [ct for ct in full_cts if ct_dict[ct] in wanted]
ssd = SuperSegmentationDataset(working_dir=ap.working_dir())
ct_key = ap.celltype_key()
ssv_cts = ssd.load_numpy_data(ct_key)
log.info(f'wd {ap.working_dir()}, celltype key {ct_key}, prep {prep}, classes {[ct_dict[c] for c in full_cts]}')


def stage1(cellid):
    """skeleton-only fullness check (the criterion of get_per_cell_morphology_params)."""
    try:
        cell = SuperSegmentationObject(cellid)
        cell.load_skeleton()
        ax = cell.skeleton['axoness_avg10000']
        ax[ax == 3] = 1
        ax[ax == 4] = 1
        u = np.unique(ax)
        return cellid, bool(0 in u and 1 in u and 2 in u), len(ax)
    except Exception as e:  # no logging inside forked workers
        return cellid, False, -1


def stage2(cell_input):
    """everything the scripts derive per full cell; returns the analysis_prep dict + extra columns."""
    cellid, borders, cutoff = cell_input
    out = {'cellid': cellid}
    try:
        cid, d = get_per_cell_morphology_params(cellid)
        if cid == 0:
            out['full'] = False
            return out
        out['full'] = True
        out['params'] = d
        try:
            sd = get_spine_density([cellid, 1, d])
            out['spine density'], out['dendrite length no spines'] = (sd if isinstance(sd, list) else (np.nan, np.nan))
        except Exception:
            out['spine density'] = out['dendrite length no spines'] = np.nan
        try:
            out['axon median radius'], out['dendrite median radius'] = get_median_comp_radii_cell([cellid, False, False])
            out['dendrite median radius no spines'] = get_median_comp_radii_cell([cellid, False, True])[1]  # ct_morph_analyses uses no_spine=True
        except Exception:
            out['axon median radius'] = out['dendrite median radius'] = out['dendrite median radius no spines'] = np.nan
        try:
            out['axon myelin length'], out['axon myelin fraction'] = get_myelin_fraction([cellid, 0, True, False])
        except Exception:
            out['axon myelin length'] = out['axon myelin fraction'] = np.nan
        try:
            out['dendrite length no spines (info)'], out['primary dendrites'], out['branching points'] = get_dendrite_info_cell([cellid, 0])
        except Exception:
            out['dendrite length no spines (info)'] = out['primary dendrites'] = out['branching points'] = np.nan
        try:
            out['dendrite not cut off'] = bool(check_cutoff_dendrites([cellid, cutoff, borders])[1])
        except Exception:
            out['dendrite not cut off'] = None
        try:
            cell = SuperSegmentationObject(cellid)
            size = cell.size
            if size < 0:
                cell.calculate_size()
                size = cell.size
            out['cell volume'] = size * np.prod(cell.scaling) * 1e-9
        except Exception:
            out['cell volume'] = np.nan
    except Exception:
        out['full'] = False
        out['error'] = traceback.format_exc().splitlines()[-1]
    return out


t0 = time.time()
# ---------- stage 1
s1_file = prep + 'stage1_fullness.npz'
if args.skip_stage1 and os.path.exists(s1_file):
    z = np.load(s1_file)
    s1_ids, s1_full, s1_ct, s1_nodes = z['ids'], z['full'], z['ct'], z['n_nodes']
    log.info(f'stage 1 reused from {s1_file}: {len(s1_ids)} SSVs')
else:
    cand, cand_ct = [], []
    for ct in full_cts:
        ids = ssd.ssv_ids[ssv_cts == ct]
        if args.limit:
            ids = ids[:args.limit]
        cand.append(ids)
        cand_ct.append(np.full(len(ids), ct))
    cand = np.concatenate(cand)
    cand_ct = np.concatenate(cand_ct)
    log.info(f'stage 1: fullness check on {len(cand)} SSVs')
    res = pmap(stage1, [int(c) for c in cand], nb_cpus=args.nb_cpus)
    s1_ids = np.array([r[0] for r in res], dtype=np.uint64)
    s1_full = np.array([r[1] for r in res])
    ct_of = dict(zip(cand.tolist(), cand_ct.tolist()))
    s1_ct = np.array([ct_of[int(i)] for i in s1_ids])
    s1_nodes = np.array([r[2] for r in res])
    np.savez(s1_file, ids=s1_ids, full=s1_full, ct=s1_ct, n_nodes=s1_nodes)
    log.info(f'stage 1 done in {time.time() - t0:.0f} s: {int(s1_full.sum())} full cells, '
             f'{int((s1_nodes < 0).sum())} SSVs without loadable skeleton')

# ---------- stage 2
borders = np.array(global_params.config.entries['cube_of_interest_bb']) * [10, 10, 25]
full_ids = s1_ids[s1_full]
full_ct = s1_ct[s1_full]
order = np.argsort(-s1_nodes[s1_full])  # largest skeletons first: shortens the long tail of the pool
full_ids, full_ct = full_ids[order], full_ct[order]
t1 = time.time()
s2_file = prep + 'stage2_results.pkl'
if args.skip_stage2 and os.path.exists(s2_file):
    import pickle
    res = pickle.load(open(s2_file, 'rb'))
    log.info(f'stage 2 reused from {s2_file}: {len(res)} cells')
else:
    log.info(f'stage 2: per-cell parameters for {len(full_ids)} full cells')
    res = pmap(stage2, [[int(i), borders, args.cutoff_dist] for i in full_ids], nb_cpus=args.nb_cpus, chunksize=1)
    import pickle
    pickle.dump(res, open(s2_file, 'wb'))
errs = [r for r in res if r.get('error')]
log.info(f'stage 2 done in {time.time() - t1:.0f} s; {len(errs)} cells failed' + (f', e.g. {errs[0]}' if errs else ''))

# ---------- stage 3 + outputs
sd_synssv = SegmentationDataset('syn_ssv', working_dir=ap.working_dir())
m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob = filter_synapse_caches_general(
    sd_synssv, syn_prob_thresh=args.syn_prob, min_syn_size=args.min_syn_size)
m_axs = m_axs.copy()
m_axs[m_axs == 3] = 1
m_axs[m_axs == 4] = 1
res_by_id = {r['cellid']: r for r in res}
lengths_to_test = [100, 200, 500, 1000]
cols = np.hstack(['total', 'full cells', np.concatenate(
    [[f'axon length >= {i} µm', f'dendrite length >= {i} µm', f'axon and dendrite length >= {i} µm'] for i in lengths_to_test])])
cell_number_info = pd.DataFrame(columns=cols, index=[ct_dict[c] for c in full_cts])
rows = []
for ct in full_cts:
    name = ct_dict[ct]
    ids = [int(i) for i, c in zip(full_ids, full_ct) if c == ct and res_by_id[int(i)].get('full')]
    axon_syns, den_syns, soma_syns = synapse_amount_percell(celltype=ct, syn_cts=m_cts, syn_sizes=m_sizes,
                                                            syn_ssv_partners=m_ssv_partners, syn_axs=m_axs,
                                                            axo_denso=True, all_comps=True)
    cell_dict = {}
    for cid in ids:
        d = dict(res_by_id[cid]['params'])
        for comp, src in (('axon', axon_syns), ('dendrite', den_syns), ('soma', soma_syns)):
            d[f'{comp} synapse amount'] = src[cid]['amount'] if cid in src else 0
            d[f'{comp} summed synapse size'] = src[cid]['summed size'] if cid in src else 0
        cell_dict[cid] = d
        row = {'cellid': cid, 'celltype': name}
        row.update(d)
        row.update({k: v for k, v in res_by_id[cid].items() if k not in ('params', 'cellid', 'full')})
        rows.append(row)
    cell_array = np.array(ids)
    write_obj2pkl(f'{prep}/full_{name}_dict.pkl', cell_dict)
    write_obj2pkl(f'{prep}/full_{name}_arr.pkl', cell_array)
    cell_number_info.loc[name, 'total'] = int((ssv_cts == ct).sum())
    cell_number_info.loc[name, 'full cells'] = len(cell_array)
    al = np.array([cell_dict[c]['axon length'] for c in ids])
    dl = np.array([cell_dict[c]['dendrite length'] for c in ids])
    for L in lengths_to_test:
        cell_number_info.loc[name, f'axon length >= {L} µm'] = int((al >= L).sum())
        cell_number_info.loc[name, f'dendrite length >= {L} µm'] = int((dl >= L).sum())
        cell_number_info.loc[name, f'axon and dendrite length >= {L} µm'] = int(((al >= L) & (dl >= L)).sum())
    log.info(f'{name}: {len(cell_array)} full cells, {int(((al >= 200) & (dl >= 200)).sum())} with axon and dendrite >= 200 um')
cell_number_info.to_csv(f'{prep}/cell_numbers.csv')
df = pd.DataFrame(rows)
df['soma centre'] = df['soma centre'].apply(lambda c: None if c is None else ','.join(f'{x:.1f}' for x in np.atleast_1d(c)))
df.to_csv(f'{prep}/percell_morph.csv', index=False)
log.info(f'written {len(df)} rows to percell_morph.csv; total {time.time() - t0:.0f} s')
print('PERCELL_DONE')
