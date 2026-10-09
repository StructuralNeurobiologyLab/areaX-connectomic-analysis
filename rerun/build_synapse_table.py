"""Build (or verify) the filtered synapse cache that every connectivity script starts from.

The cache is exactly the return value of ``analysis_conn_helper.filter_synapse_caches_general`` (synapse
probability > prob, mesh_area/2 > min size) and is written to ``$BIO_ANALYSIS_PREP/syn_cache_sp<p>_ms<s>[_<ct key>].npz``.
With the env variable set, the unchanged scripts pick the cache up through that function.

    python -m bio_analysis.rerun.build_synapse_table [--syn_prob 0.6] [--min_syn_size 0.1] [--verify]

``--verify`` recomputes from the dataset arrays and compares with the cached arrays.
"""
import argparse
import os
import time

import numpy as np

from bio_analysis.general.analysis_params import Analysis_Params

parser = argparse.ArgumentParser()
parser.add_argument('--version', default='v6')
parser.add_argument('--syn_prob', type=float, default=0.6)
parser.add_argument('--min_syn_size', type=float, default=0.1)
parser.add_argument('--verify', action='store_true')
args = parser.parse_args()

assert 'BIO_ANALYSIS_PREP' in os.environ, 'set BIO_ANALYSIS_PREP to the prep directory (rerun/run.sh does this)'
from syconn import global_params
from syconn.reps.segmentation import SegmentationDataset
from bio_analysis.general.analysis_conn_helper import filter_synapse_caches_general

ap = Analysis_Params(version=args.version)
global_params.wd = ap.working_dir()
os.makedirs(ap.file_locations, exist_ok=True)
sd = SegmentationDataset('syn_ssv', working_dir=ap.working_dir())
t0 = time.time()
cached = filter_synapse_caches_general(sd, syn_prob_thresh=args.syn_prob, min_syn_size=args.min_syn_size)
names = ['m_cts', 'm_ids', 'm_axs', 'm_ssv_partners', 'm_sizes', 'm_spiness', 'm_rep_coord', 'syn_prob']
print(f'{len(cached[1])} synapses with prob > {args.syn_prob} and size > {args.min_syn_size} um2 in {time.time() - t0:.0f} s')
for n, a in zip(names, cached):
    print(f'  {n:15s} {a.shape} {a.dtype}')
if args.verify:
    os.environ['BIO_ANALYSIS_SYN_CACHE'] = '0'
    t0 = time.time()
    fresh = filter_synapse_caches_general(sd, syn_prob_thresh=args.syn_prob, min_syn_size=args.min_syn_size)
    ok = all(np.array_equal(a, b) for a, b in zip(cached, fresh))
    print(f'verify: recomputed in {time.time() - t0:.0f} s, identical = {ok}')
print('SYN_CACHE_DONE')
