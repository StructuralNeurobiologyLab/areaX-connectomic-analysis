#!/usr/bin/env python
"""Scatter of predicted vs manually painted synapse size with log-log fit,
bootstrap confidence band and slope CI; plus a linear-axis version.
    python plot_fit.py figures/final40_areas.csv figures/final40"""
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
a = pd.read_csv(sys.argv[1]); out = sys.argv[2]; n = len(a)
CI = float(sys.argv[3]) if len(sys.argv) > 3 else 95.0; q_lo, q_hi = (100 - CI) / 2, 100 - (100 - CI) / 2; CIs = ('%g' % CI) + '%%'
lx, ly = np.log10(a.manual.values), np.log10(a.api.values); s, i = np.polyfit(lx, ly, 1); r2 = np.corrcoef(lx, ly)[0, 1]**2
rng = np.random.default_rng(0); B = 5000; xx = np.linspace(np.log10(0.03), np.log10(2), 100)
fits = np.empty((B, 2)); lines = np.empty((B, len(xx)))
for b in range(B):
    k = rng.integers(0, n, n); fits[b] = np.polyfit(lx[k], ly[k], 1); lines[b] = fits[b, 0] * xx + fits[b, 1]
lo, hi = np.percentile(lines, [q_lo, q_hi], axis=0); s_lo, s_hi = np.percentile(fits[:, 0], [q_lo, q_hi])
ratio = np.log(a.api / a.manual); gm, sdl = np.exp(ratio.mean()), ratio.std()
sl, il = np.polyfit(a.manual, a.api, 1); r2l = np.corrcoef(a.manual, a.api)[0, 1]**2; s0 = (a.manual * a.api).sum() / (a.manual**2).sum()
print('n=%d | log-log slope %.2f (%s CI %.2f-%.2f) r2 %.2f | geomean pred/manual %.2f sd-log %.2f | linear slope %.2f intercept %.3f r2 %.2f | through origin %.2f | spearman %.2f'
      % (n, s, CIs.replace('%%','%'), s_lo, s_hi, r2, gm, sdl, sl, il, r2l, s0, a.api.corr(a.manual, method='spearman')))
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0.03, 2], [0.03, 2], color='#999999', lw=1, ls='--', label='identity')
ax.fill_between(10**xx, 10**lo, 10**hi, color='#3a7ca5', alpha=0.18, lw=0, label=CIs.replace('%%','%') + ' bootstrap confidence band')
ax.plot(10**xx, 10**(s * xx + i), color='#3a7ca5', lw=1.4, label='log-log fit, n = %d: slope %.2f (%s CI %.2f–%.2f), r$^2$ %.2f' % (n, s, CIs.replace('%%','%'), s_lo, s_hi, r2))
ax.scatter(a.manual, a.api, s=52, color='#3a7ca5', zorder=3)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.03, 2); ax.set_ylim(0.03, 2); ax.set_aspect('equal')
ax.set_xlabel('manually painted synapse size (µm²)'); ax.set_ylabel('predicted synapse size (µm²)')
ax.legend(fontsize=8.5, loc='upper left', frameon=False); ax.grid(alpha=0.25, which='both')
fig.tight_layout(); fig.savefig(out + '_scatter.png', dpi=150)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1.5], [0, 1.5], color='#999999', lw=1, ls='--', label='identity')
x2 = np.array([0, 1.5]); ax.plot(x2, sl * x2 + il, color='#3a7ca5', lw=1.4, label='linear fit, n = %d: slope %.2f, intercept %.2f, r$^2$ %.2f' % (n, sl, il, r2l))
ax.plot(x2, s0 * x2, color='#c8901a', lw=1.4, label='fit through origin: slope %.2f' % s0)
ax.scatter(a.manual, a.api, s=52, color='#3a7ca5', zorder=3)
ax.set_xlim(0, 1.5); ax.set_ylim(0, 1.5); ax.set_aspect('equal')
ax.set_xlabel('manually painted synapse size (µm²)'); ax.set_ylabel('predicted synapse size (µm²)')
ax.legend(fontsize=9, loc='upper left', frameon=False); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(out + '_scatter_linear.png', dpi=150)
