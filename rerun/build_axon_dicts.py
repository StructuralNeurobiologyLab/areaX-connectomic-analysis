"""Axon dictionaries ax_<CT>_dict.pkl (DA, LMAN, HVC) for a version, in one pass (replaces the commented-out axon block
of general/analysis_prep.py, which loaded skeleton AND mesh of every axon SSV).

Per axon SSV: 'axon length' (skeleton, analysis_prep_func.get_per_cellfrag_morph_params: weighted_graph().size(weight)/1000),
'axon mesh surface area' (mesh_area_calc(mesh); computed only for axons with length >= --mesh_min_len um, NaN below:
every script filters axons at >= 50 um before using areas), 'axon synapse amount' / 'axon summed synapse size' from the
synapse cache (synapse_amount_percell, all_comps=False) as in analysis_prep.py.

    rerun/run.sh -m bio_analysis.rerun.build_axon_dicts --version v8 [--nb_cpus 64] [--mesh_min_len 50] [--limit N]
"""
import argparse
import os
import time
import traceback

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--version', default='v6')
parser.add_argument('--nb_cpus', type=int, default=None)
parser.add_argument('--mesh_min_len', type=float, default=50.)
parser.add_argument('--celltypes', default=None)
parser.add_argument('--limit', type=int, default=None)
parser.add_argument('--syn_prob', type=float, default=0.6)
parser.add_argument('--min_syn_size', type=float, default=0.1)
args = parser.parse_args()

assert 'BIO_ANALYSIS_PREP' in os.environ
from syconn import global_params
from syconn.reps.super_segmentation import SuperSegmentationDataset, SuperSegmentationObject
from syconn.reps.segmentation import SegmentationDataset
from bio_analysis.rerun.mp import pmap
from syconn.handler.basics import write_obj2pkl
from syconn.handler.config import initialize_logging
from syconn.proc.meshes import mesh_area_calc
from bio_analysis.general.analysis_params import Analysis_Params
from bio_analysis.general.analysis_prep_func import synapse_amount_percell
from bio_analysis.general.analysis_conn_helper import filter_synapse_caches_general

ap = Analysis_Params(version=args.version)
global_params.wd = ap.working_dir()
prep = ap.file_locations
os.makedirs(prep + 'logs', exist_ok=True)
log = initialize_logging('build_axon_dicts', log_dir=prep + 'logs/')
ct_dict = ap.ct_dict(with_glia=False)
ax_cts = [ct for ct in ap.axon_cts() if args.celltypes is None or ct_dict[ct] in args.celltypes.split(',')]
ssd = SuperSegmentationDataset(working_dir=ap.working_dir())
ssv_cts = ssd.load_numpy_data(ap.celltype_key())


def skel_length(cellid):
    try:
        c = SuperSegmentationObject(cellid)
        c.load_skeleton()
        return c.weighted_graph().size(weight='weight') / 1000
    except Exception:  # no logging inside forked workers
        return np.nan


def mesh_area(cellid):
    try:
        return mesh_area_calc(SuperSegmentationObject(cellid).mesh)
    except Exception:
        return np.nan


sd_synssv = SegmentationDataset('syn_ssv', working_dir=ap.working_dir())
m_cts, m_ids, m_axs, m_ssv_partners, m_sizes, m_spiness, m_rep_coord, syn_prob = filter_synapse_caches_general(
    sd_synssv, syn_prob_thresh=args.syn_prob, min_syn_size=args.min_syn_size)
m_axs = m_axs.copy()
m_axs[m_axs == 3] = 1
m_axs[m_axs == 4] = 1
t0 = time.time()
for ct in ax_cts:
    name = ct_dict[ct]
    ids = ssd.ssv_ids[ssv_cts == ct]
    if args.limit:
        ids = ids[:args.limit]
    log.info(f'{name}: {len(ids)} axon SSVs, skeleton lengths')
    lengths = np.array(pmap(skel_length, [int(i) for i in ids], nb_cpus=args.nb_cpus))
    log.info(f'{name}: {int(np.isnan(lengths).sum())} SSVs without loadable skeleton')
    sel = np.where(lengths >= args.mesh_min_len)[0]
    log.info(f'{name}: {len(sel)} axons >= {args.mesh_min_len} um, mesh areas ({time.time() - t0:.0f} s)')
    areas = np.full(len(ids), np.nan)
    areas[sel] = pmap(mesh_area, [int(ids[i]) for i in sel], nb_cpus=args.nb_cpus, chunksize=4)
    axon_syns = synapse_amount_percell(celltype=ct, syn_cts=m_cts, syn_sizes=m_sizes, syn_ssv_partners=m_ssv_partners,
                                       syn_axs=m_axs, axo_denso=True, all_comps=False)
    d = {}
    for i, cid in enumerate(ids):
        cid = int(cid)
        d[cid] = {'axon length': float(lengths[i]), 'axon mesh surface area': float(areas[i]),
                  'axon synapse amount': axon_syns[cid]['amount'] if cid in axon_syns else 0,
                  'axon summed synapse size': axon_syns[cid]['summed size'] if cid in axon_syns else 0}
    write_obj2pkl(f'{prep}/ax_{name}_dict.pkl', d)
    log.info(f'{name}: ax_{name}_dict.pkl written ({time.time() - t0:.0f} s)')
print('AXON_DICTS_DONE')
