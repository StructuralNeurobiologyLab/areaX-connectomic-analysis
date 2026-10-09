"""Temperature scaling of the PTv3 cell type classifier: one temperature T is fitted on the held-out ground truth contexts of
the 8-fold cross-validation (logits saved by percell_eval.py --save_logits) by minimising the per-context negative
log-likelihood. The certainty of a cell is 1 - H(mean softmax over contexts) / ln C (as in Schubert et al. 2022).

    python calibrate_temperature.py <folder with fold*.npz> <out npy>
"""
import sys, glob
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

folder, out = sys.argv[1], sys.argv[2]
L, Y = [], []
for f in sorted(glob.glob(f'{folder}/fold*.npz')):
    d = np.load(f)
    L.append(d['logits'].astype(np.float64))
    Y.append(d['labels'])
L = np.concatenate(L)
Y = np.concatenate(Y)  # (n_cells, contexts, classes)
n, R, C = L.shape
print(f'held-out ground truth: {n} cells x {R} contexts x {C} classes')


def nll(logT):
    lp = log_softmax(L / np.exp(logT), -1)
    return -lp[np.arange(n)[:, None], np.arange(R)[None, :], Y[:, None]].mean()


res = minimize_scalar(nll, bounds=(-3, 3), method='bounded')
T = float(np.exp(res.x))
print(f'temperature T = {T:.3f}: per-context NLL {nll(0):.3f} -> {res.fun:.3f}')


def certainty(logits, T=1.0, r=None):
    p = softmax(logits / T, -1)
    p = p[:, :r] if r else p
    pm = p.mean(1)
    return 1 + (pm * np.log(np.clip(pm, 1e-30, None))).sum(-1) / np.log(logits.shape[-1])


for tag, t in (('T = 1', 1.0), (f'T = {T:.2f}', T)):
    c = certainty(L, t, r=20)
    pred = softmax(L / t, -1)[:, :20].mean(1).argmax(-1)
    ok = pred == Y
    print(f'  {tag}: certainty at redundancy 20, median {np.median(c):.3f}, correct cells {np.median(c[ok]):.3f}, '
          f'wrong cells (n = {(~ok).sum()}) {np.median(c[~ok]) if (~ok).any() else float("nan"):.3f}, fraction > 0.9: {(c > .9).mean():.2f}')
np.save(out, np.array([T]))
