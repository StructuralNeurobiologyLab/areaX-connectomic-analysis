"""Every number of the Rother et al. manuscript / Document S1 that comes out of the analysis csvs, looked up in one
results root (v8 run names); the manuscript location of each number is given in the key. Values are rounded as
printed in the paper where the paper rounds.

    python rerun/paper_numbers.py <results_root> [--json out.json]
    python rerun/paper_numbers.py <root_a> --diff <root_b> [--md out.md]      # two runs of the same version
"""
import argparse, json, os
import numpy as np, pandas as pd

F6 = "general/251023_j0251v8_cts_percentages_mcl_200_ax50_synprob_0.60_TePkBrNGF_newmergers_bw_fs_20_STNnoGP"
AX0 = "general/250828_j0251v8_cts_percentages_mcl_200_ax0_synprob_0.60_TePkBrNGF_newmergers_bw_fs_20"
MCL50 = "general/241024_j0251v8_cts_percentages_mcl_50_ax50_synprob_0.60_TePkBrNGF_newmergers_bw_fs_20"
MERG = "general/260916_j0251v8_cts_percentages_mcl_200_ax50_synprob_0.60_TePkBrNGF_newmergers_bw_fs_20_mergers_in"
SPINE = "dir_indir_pathway_analysis/241025_j0251v8_MSN_3_GPe_GPiratio_spine_density_mcl_200_synprob_0.60_kde1_f20"
SHUF = "dir_indir_pathway_analysis/250311_j0251v8_MSN_GP_ratio_shuffle2_it100_fs20_hb30"
SYN_MSN = "dir_indir_pathway_analysis/241025_j0251v8_MSN_GPe_GPi_syn_multisyn_mcl_200_synprob_0.60_kde1_f20_stn_gt"
SYN_STN = "dir_indir_pathway_analysis/241025_j0251v8_STN_GPe_GPi_syn_multisyn_mcl_200_synprob_0.60_kde1_f20_stn_gt"
SYN_GP = "dir_indir_pathway_analysis/241025_j0251v8_GPe_GPi_syn_multisyn_mcl_200_synprob_0.60_kde1_f20"
REC = "dir_indir_pathway_analysis/241025_j0251v8_{k}_recurr_conn_mcl_200_synprob_0.60_STNGPINTv6_fs20"
STNSUB = "dir_indir_pathway_analysis/241025_j0251v8_STN_4_GPe_GPiratio_spine_density_mcl_200_synprob_0.60_kde1_f20"
COMP = "dir_indir_pathway_analysis/241025_j0251v8_{ct}_input_comps_mcl_200_ax50_synprob_0.60_STNGPINTv6_TeGy_fs20"
CELLNUM = "dir_indir_pathway_analysis/250429_j0251v8_cellnumber_pathway_analysis_{t}_STNGPINTv6"
MSNCMP = "dir_indir_pathway_analysis/241029_j0251v8_MSN_GPe_GPi_comparison_mcl_200_ax50_synprob_0.6"
MORPH = "general/250711_j0251v8_ct_morph_analyses_newmergers_mcl_200_ax200_TeBKv6MSNyw_fs20npca1_umap5_a0.5"
INTPCA = "dir_indir_pathway_analysis/241025_j0251v8_['INT1', 'INT2', 'INT3']_morph_comp_radius_spiness_examplecells_mcl200_fs20_med1_RdTeINTv6_nc1"
TABS8 = "general/261001_j0251v8_ct_comp_pathlength_fraction_mcl_200"
CTS = ['MSN', 'STN', 'TAN', 'GPe', 'GPi', 'LTS', 'INT1', 'INT2', 'INT3', 'INT4']


def numbers(root):
    rd = lambda sub, f: pd.read_csv(os.path.join(root, sub, f), index_col=0)
    ex = lambda sub, f: os.path.isfile(os.path.join(root, sub, f))
    N = {}
    def put(k, v, nd=None):
        if isinstance(v, (np.floating, float)):
            v = float(v) if nd is None else round(float(v), nd)
        elif isinstance(v, (np.integer,)):
            v = int(v)
        N[k] = v
    if ex(F6, 'numbers_perct.csv'):
        num = rd(F6, 'numbers_perct.csv')['number of suitable cells']
        for ct in num.index: put(f'Results p1 / Fig 4,5 legends: n {ct}', int(num[ct]))
        put('Results p1 / Methods: total neurons (sum MSN..INT4)', int(num[CTS].sum()))
        put('Results p1: INT1-4 total', int(num[['INT1', 'INT2', 'INT3', 'INT4']].sum()))
        put('Results p1: projecting axons >= 50 um (DA+LMAN+HVC)', int(num[['DA', 'LMAN', 'HVC']].sum()))
        out, inc = rd(F6, 'outgoing_syn_sizes_matrix_rel.csv'), rd(F6, 'incoming_syn_sizes_matrix_rel.csv')
        glob = rd(F6, 'outgoing_syn_sizes_matrix_abs_sum_datatset_norm.csv')
        for pre in out.index:
            for post in out.columns:
                put(f'Fig 6B out % {pre} -> {post}', out.loc[pre, post], 2)
                put(f'Fig 6C in % {pre} -> {post}', inc.loc[pre, post], 2)
                put(f'Fig 6A global % {pre} -> {post}', glob.loc[pre, post], 3)
        put('Results: STN % of total neurons', 100 * num['STN'] / num[CTS].sum(), 2)
        put('Results: GPe -> INT2+INT3+INT4 (% GPe output)', out.loc['GPe', ['INT2', 'INT3', 'INT4']].sum(), 2)
        put('Results: GPe -> INT2+INT3 (% GPe output)', out.loc['GPe', ['INT2', 'INT3']].sum(), 2)
        mo = rd(F6, 'MSN/outgoing_syns_overview_pd_MSN.csv'); gi = rd(F6, 'GPi/incoming_syns_overview_pd_GPi.csv')
        for t in ('GPi', 'GPe'):
            put(f'Results: MSN -> {t} mean syn number per MSN', mo.loc[t, 'mean syn number'], 2); put(f'Results: MSN -> {t} std syn number per MSN', mo.loc[t, 'std syn number'], 2)
        put('Results: GPi from MSN mean syn number per GPi', gi.loc['MSN', 'mean syn number'], 1); put('Results: GPi from MSN std syn number per GPi', gi.loc['MSN', 'std syn number'], 1)
        # Fig S2D: synapse number vs area matrices
        for kind, fa, fn in (('global', 'outgoing_syn_sizes_matrix_abs_sum_datatset_norm.csv', 'outgoing_syn_number_matrix_abs_sum.csv'),
                             ('output', 'outgoing_syn_sizes_matrix_rel.csv', 'outgoing_syn_number_matrix_rel.csv'), ('input', 'incoming_syn_sizes_matrix_rel.csv', 'incoming_syn_number_matrix_rel.csv')):
            A, Bn = rd(F6, fa), rd(F6, fn)
            if kind == 'global':
                Bn = 100 * Bn / Bn.values.sum()
            a, b = A.values.ravel().astype(float), Bn.loc[A.index, A.columns].values.ravel().astype(float)
            m = ~(np.isnan(a) | np.isnan(b)); put(f'Fig S2D legend: Pearson r area vs number ({kind})', np.corrcoef(a[m], b[m])[0, 1], 3)
    for name, sub in (('S2A mergers in', MERG), ('S2B mcl50', MCL50), ('S2C ax0', AX0)):
        if ex(sub, 'incoming_syn_sizes_matrix_rel.csv'):
            inc2 = rd(sub, 'incoming_syn_sizes_matrix_rel.csv'); out2 = rd(sub, 'outgoing_syn_sizes_matrix_rel.csv')
            for ct in ('GPe', 'GPi', 'STN', 'MSN'):
                put(f'Fig {name}: HVC -> {ct} in %', inc2.loc['HVC', ct], 2); put(f'Fig {name}: LMAN -> {ct} in %', inc2.loc['LMAN', ct], 2)
            put(f'Fig {name}: STN -> GPi out %', out2.loc['STN', 'GPi'], 2); put(f'Fig {name}: GPe -> STN out %', out2.loc['GPe', 'STN'], 2)
            if ex(sub, 'numbers_perct.csv'):
                n2 = rd(sub, 'numbers_perct.csv')['number of suitable cells']
                for ct in CTS: put(f'Fig {name}: n {ct}', int(n2[ct]))
    if ex(SPINE, 'conn_subgroup_numbers.csv'):
        for tag, sub in (('all MSN', SPINE), ('complete dendrites', SPINE + '_fullden')):
            g = rd(sub, 'conn_subgroup_numbers.csv').set_index('celltype')['number of MSN cells']; tot = g.sum()
            for k in g.index: put(f'Fig 2D/S5F legend ({tag}): {k} n', int(g[k])); put(f'Fig 2D/S5F legend ({tag}): {k} %', 100 * g[k] / tot, 2)
            put(f'Results ({tag}): MSN total in groups', int(tot)); put(f'Results ({tag}): % MSN projecting to GPi (both + only GPi)', 100 * (g['MSN both'] + g['MSN only GPi']) / tot, 2)
            sp = rd(sub, 'spearman_corr_results.csv')
            for i in sp.index: put(f'Table S9 ({tag}): Spearman {i} r', sp.loc[i, 'stats'], 3); put(f'Table S9 ({tag}): Spearman {i} p', float(sp.loc[i, 'p-value']))
            df = rd(sub, 'MSN_morph_GPe_GPiratio.csv') if ex(sub, 'MSN_morph_GPe_GPiratio.csv') else None
            if df is not None:
                nsyn_cols = [c for c in df.columns if 'syn number' in c.lower() or 'number' in c.lower()]
                put(f'S1 S7 check ({tag}): MSN rows', len(df))
    if ex(SHUF, 'GP_ratio_area_hist_counts.csv'):
        d = rd(SHUF, 'GP_ratio_area_hist_counts.csv'); tot = d.sum(); frac = d.iloc[27:].sum() / tot * 100
        put('Results: % MSN with GP ratio >= 0.9 observed', frac['observed'], 2); put('Results: % MSN GP ratio >= 0.9 shuffled mean', frac.iloc[1:].mean(), 2); put('Results: % MSN GP ratio >= 0.9 shuffled std', frac.iloc[1:].std(), 3)
        o = d['observed'].iloc[1:29]; s = d.iloc[1:29, 1:].mean(axis=1)
        put('Results: GP ratio peak bin observed (x30)', int(o.idxmax())); put('Results: GP ratio peak bin shuffled (x30)', int(s.idxmax()))
    for tag, sub, f, npf, pairs in (('MSN', SYN_MSN, 'ranksums_results_MSN_GPi_GPi.csv', 'syns_cell_pair_numbers.csv', ('to GPe vs to GPi',)),
                                    ('STN', SYN_STN, 'ranksums_results_STN_GPi_GPi.csv', 'syns_cell_pair_numbers.csv', ('to GPe vs to GPi', 'from GPe vs from GPi')),
                                    ('GP', SYN_GP, 'ranksums_results_GPe_GPi.csv', 'syn_cell_pair_numbers.csv', ('to GPe vs to GPi inter', 'to GPe vs to GPi intra', 'GPe inter vs intra', 'GPi inter vs intra'))):
        if ex(sub, f):
            rk, npn = rd(sub, f), rd(sub, npf)
            for row in rk.index:
                for col in pairs:
                    if col in rk.columns: put(f'Table S9 ({tag}): {row} {col}', float(rk.loc[row, col]))
            for i in npn.index:
                for c in npn.columns: put(f'Table S9 n ({tag}): {i} {c}', int(npn.loc[i, c]) if not pd.isna(npn.loc[i, c]) else None)
            if ex(sub, 'median_syn_sizes.csv') or True:
                for extra in ('syn_sizes_overview.csv', 'multi_syn_overview.csv'):
                    if ex(sub, extra):
                        t = rd(sub, extra)
                        for i in t.index:
                            for c in t.columns: put(f'Fig 2F-H/3C-E ({tag}) {extra}: {i} {c}', float(t.loc[i, c]) if pd.api.types.is_number(t.loc[i, c]) else str(t.loc[i, c]))
    for k, a, b in (('STN_GPi', 'STN', 'GPi'), ('STN_GPe', 'STN', 'GPe'), ('MSN_TAN', 'MSN', 'TAN')):
        sub = REC.format(k=k)
        if ex(sub, f'recurr_conn_overview_{k}.csv'):
            t = rd(sub, f'recurr_conn_overview_{k}.csv')
            for i in t.index:
                for c in t.columns:
                    v = t.loc[i, c]
                    if pd.api.types.is_number(v) and not pd.isna(v): put(f'Table S11 ({a}-{b}): {i} / {c}', float(v), 4)
    if ex(STNSUB, 'conn_subgroup_numbers.csv'):
        g = rd(STNSUB, 'conn_subgroup_numbers.csv').set_index('celltype')['number of STN cells']; tot = g.sum()
        for k in g.index: put(f'Table S10: {k} n', int(g[k])); put(f'Table S10: {k} %', 100 * g[k] / tot, 2)
    if ex(COMP.format(ct='GPi'), 'all_comps_results_per_GPi_cell.csv'):
        d = rd(COMP.format(ct='GPi'), 'all_comps_results_per_GPi_cell.csv'); sub = d[(d.celltype == 'GPe') & (d.compartment == 'soma')]
        put('Results: GPi with somatic GPe input n', int((sub['Number of synapses '] > 0).sum())); put('Results: GPi cells (comp run)', int(d.cellid.nunique()))
        for ct in CTS:
            s2 = COMP.format(ct=ct)
            if ex(s2, f'all_comps_results_per_{ct}_cell.csv'):
                d2 = rd(s2, f'all_comps_results_per_{ct}_cell.csv')
                for col in [c for c in d2.columns if 'percent' in c.lower() or 'fraction' in c.lower()][:1]:
                    med = d2.groupby(['celltype', 'compartment'])[col].median()
                    for (pre, comp), v in med.items(): put(f'Fig S9 ({ct} inputs): median {col} {pre} {comp}', float(v), 2)
                tan = d2[d2.celltype == 'TAN']
                put(f'Fig S9 legend: {ct} cells with TAN input', int((tan['Number of synapses '] > 0).groupby(tan.cellid).any().sum()) if len(tan) else 0)
    for t in ('MSN_GPi_GPe', 'GPe_GPi_STN', 'STN_GPe_GPi', 'STN_GPi_GPe', 'MSN_GPe_GPi', 'INT2_GPe_GPi', 'INT3_GPe_GPi', 'HVC_MSN_GPi', 'HVC_STN_GPi', 'LMAN_MSN_GPi', 'LMAN_STN_GPi'):
        for ln in ('200', '200_50'):
            sub = CELLNUM.format(t=f'{t}_{ln}')
            if ex(sub, f'summary_cell_numbers_{t}.csv'):
                c = rd(sub, f'summary_cell_numbers_{t}.csv')
                for i in c.index:
                    for col in c.columns:
                        v = c.loc[i, col]
                        if pd.api.types.is_number(v) and not pd.isna(v): put(f'cell-number pathway {t}: {i} / {col}', float(v), 3)
    if ex(MSNCMP, 'comp_compartment_MSN only GPe_MSN only GPi_comp_volume_mcl200/ranksum_MSN only GPe_MSN only GPi.csv'):
        c = rd(MSNCMP, 'comp_compartment_MSN only GPe_MSN only GPi_comp_volume_mcl200/ranksum_MSN only GPe_MSN only GPi.csv')
        for i in c.index:
            for col in c.columns: put(f'Table S9 / Fig S5H-J: MSN groups {i} {col}', float(c.loc[i, col]))
    if ex(MORPH, 'summary_params_ct.csv'):
        m = rd(MORPH, 'summary_params_ct.csv')
        for i in m.index:
            for col in m.columns:
                v = m.loc[i, col]
                if pd.api.types.is_number(v) and not pd.isna(v): put(f'Table S7-ish / S1 soma diameters: {i} / {col}', float(v), 4)
    if ex(TABS8, 'table_S8_comp_pathlength_fraction.csv'):
        t = rd(TABS8, 'table_S8_comp_pathlength_fraction.csv')
        for i in t.index:
            for col in t.columns: put(f'Table S8: {i} {col}', float(t.loc[i, col]), 2)
    return N


def main():
    p = argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('--diff'); p.add_argument('--json'); p.add_argument('--md')
    a = p.parse_args()
    A = numbers(a.root)
    if a.json: json.dump(A, open(a.json, 'w'), indent=1)
    if not a.diff:
        for k, v in A.items(): print(f'{k}: {v}')
        return
    B = numbers(a.diff)
    lines = ['| number (manuscript location) | before | after |', '|---|---|---|']
    same = 0
    for k in A:
        if k not in B:
            lines.append(f'| {k} | {A[k]} | MISSING |'); continue
        x, y = A[k], B[k]
        eq = (x == y) or (isinstance(x, float) and isinstance(y, float) and np.isclose(x, y, rtol=1e-9, atol=0))
        if eq: same += 1
        else: lines.append(f'| {k} | {x} | {y} |')
    for k in B:
        if k not in A: lines.append(f'| {k} | MISSING | {B[k]} |')
    head = [f'{same} of {len(A)} numbers unchanged, {len(lines) - 2} changed/missing', '']
    txt = '\n'.join(head + lines)
    if a.md: open(a.md, 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
