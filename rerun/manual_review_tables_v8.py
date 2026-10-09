"""Tables S3 (mergers of manually inspected cells) and S12 (autapses) for the v8 cell types.

Same definitions as for_eval/merger_eval_pathlength.py (v6 table), applied per v8 type to the joined manual review
table curation/manual_review_v8.xlsx:
  number total          = inspected cells of that v8 type (all v8 full cells for GPe, GPi, LTS, STN, TAN and INT4;
                          the random 500 for MSN; the reviewed cells for INT1-3)
  erroneous             = cells marked include = n (earlier or v8 review)
  incomplete            = erroneous because a compartment is missing ('not all compartments', 'too little axon present')
  cells with mergers    = erroneous cells with merger = y
  mergers per mm        = summed 'number of mergers' of the erroneous cells / summed complete skeleton pathlength (mm)
                          of all inspected cells of that type (v8 per-cell cache, same skeletons as v6)
  mergers with neurons / glia = the same merger count split by 'merger with' (one glia merger per cell that lists glia, the rest neurons)
Table S12: autapse review restricted to the v8 analysis set after exclusion.

    BIO_ANALYSIS_PREP=<v8 prep dir> python rerun/manual_review_tables_v8.py [out dir]   # writes table_S3_v8.csv, table_S12_v8.csv
"""
import os, pickle, sys
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREP = os.environ['BIO_ANALYSIS_PREP'].rstrip('/') + '/'
XLS = REPO + '/curation/manual_review_v8.xlsx'
OUT = sys.argv[1] if len(sys.argv) > 1 else '.'
ORDER = ['MSN', 'GPe', 'GPi', 'LTS', 'STN', 'TAN', 'INT1', 'INT2', 'INT3', 'INT4']
INCOMPLETE = ('not all compartments', 'too little axon present')

F = pd.read_excel(XLS, 'non-MSN full cells'); M = pd.read_excel(XLS, 'MSN merger review'); A = pd.read_excel(XLS, 'MSN autapse review')
excl = set(int(i) for i in pickle.load(open(PREP + 'merger_arr.pkl', 'rb')))
assert len(excl) == 30, len(excl)
plen = {}
for ct in ORDER:
    d = pickle.load(open(PREP + f'full_{ct}_dict.pkl', 'rb'))
    plen.update({int(k): v['complete pathlength'] for k, v in d.items()})

# ---- Table S3
F = F[F.merger_review != 'not reviewed'].copy()
F['erroneous'] = (F.include_earlier == 'n') | (F.include_v8 == 'n')
F['incomplete'] = F.erroneous & F.exclude_because.isin(INCOMPLETE)
F['merger_cell'] = F.erroneous & (F.merger == 'y')
F['n_merg'] = np.where(F.merger_cell, F.n_mergers.fillna(1), 0)       # v8 review rows record y/n only -> one merger
F['glia'] = np.where(F.merger_cell & F.merger_with.astype(str).str.contains('glia'), 1, 0)   # one glia merger per cell noted
M['erroneous'] = M.merger == 'y'; M['incomplete'] = False; M['merger_cell'] = M.erroneous
M['n_merg'] = np.where(M.merger_cell, M.n_mergers.fillna(1), 0); M['glia'] = np.where(M.merger_cell & M.merger_with.astype(str).str.contains('glia'), 1, 0)
rows = []
for ct in ORDER:
    sub = M if ct == 'MSN' else F[F.v8_type == ct]
    missing = [i for i in sub.cellid if int(i) not in plen]
    assert not missing, (ct, missing)
    mm = sum(plen[int(i)] for i in sub.cellid) / 1000
    nm = sub.n_merg.sum()
    rows.append({'cell type': ct, 'number total': len(sub), 'number erroneous cells': int(sub.erroneous.sum()),
                 'number incomplete cells': int(sub.incomplete.sum()), 'number cells with mergers': int(sub.merger_cell.sum()),
                 'mergers per mm pathlength': round(nm / mm, 3), 'number mergers with neurons': int(nm - sub.glia.sum()),
                 'number mergers with glia': int(sub.glia.sum()), 'summed pathlength mm': round(mm, 1),
                 'number of mergers': int(nm), 'excluded from analysis': int(sub.cellid.isin(excl).sum()),
                 'review source': ', '.join(f'{k} {v}' for k, v in sub.merger_review.value_counts().items())})
S3 = pd.DataFrame(rows)

# ---- Table S12 (analysis set after exclusion; MSN = the 20 cells with a recorded soma review)
rows = []
for ct in ORDER:
    if ct == 'MSN':
        n, y = len(A), int(A.autapse_soma_RM.astype(str).str.lower().str.startswith('y').sum())
    else:
        sub = F[(F.v8_type == ct) & F.autapse.isin(['y', 'n']) & ~F.cellid.isin(excl) & (F.v8_full_cell_in_analysis_set == 'y')]
        n, y = len(sub), int((sub.autapse == 'y').sum())
    rows.append({'cell type': ct, 'number of manually inspected cells': n, 'cells with min. 1 autapse': y,
                 'percentage of cells with autapses': f'{round(100 * y / n):d}%' if n else ''})
S12 = pd.DataFrame(rows)
S3.to_csv(OUT + '/table_S3_v8.csv', index=False); S12.to_csv(OUT + '/table_S12_v8.csv', index=False)
pd.set_option('display.width', 250)
print(S3.to_string(index=False)); print(); print(S12.to_string(index=False))
tot_m, tot_mm = S3['number of mergers'].sum(), S3['summed pathlength mm'].sum()
print(f'\nall inspected cells of analysed v8 types: {S3["number total"].sum()} cells, {tot_m} mergers, {tot_mm:.1f} mm, '
      f'{tot_m / tot_mm:.4f} mergers per mm; excluded cells among them: {S3["excluded from analysis"].sum()}')
