"""Per-cell redundancy evaluation (the v6 paper metric) for a PTv3 fold checkpoint.

For every held-out cell of the fold, R contexts are drawn (same sampler as the v6 eval: valid transform,
npoints 50k, ctx 20 um), the softmax is averaged over the first r contexts for r in REDUN, and one prediction per
cell is made. Reports accuracy / macro F1 per redundancy and writes per-cell predictions to csv.
"""
import argparse, os, time, csv
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--cval', type=int, required=True)
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--out', type=str, required=True)
parser.add_argument('--R', type=int, default=50)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--width', type=int, default=32)
parser.add_argument('--num_classes', type=int, default=17)
parser.add_argument('--infer_mode', action='store_true')
parser.add_argument('--save_logits', type=str, default='')
args = parser.parse_args()
REDUN = [1, 10, 20, 50]
dev = 'cuda'
os.makedirs(os.path.dirname(args.out), exist_ok=True)

import morphx.processing.clouds as clouds
from syconn.cnn.TrainData import CellCloudDataJ0251
from syconn.handler.prediction_pts import init_hc_cache_gt
from sklearn.metrics import accuracy_score, f1_score, classification_report
from syconn.cnn.ptv3.celltype import PTv3CellType

t0 = time.time(); init_hc_cache_gt(); print(f'cache {time.time() - t0:.0f}s', flush=True)
tf = clouds.Compose([clouds.Center(), clouds.Normalization(2000)])
ds = CellCloudDataJ0251(npoints=50000, transform=tf, train=False, cv_val=args.cval, cellshape_only=False,
                        use_syntype=True, onehot=True, batch_size=args.R, ctx_size=20000, map_myelin=False)
cells = np.array(ds.sso_ids)
labels = {c: int(ds.label_dc[c]) for c in cells}
print(f'fold {args.cval}: {len(cells)} held-out cells', flush=True)


class PerCell(torch.utils.data.Dataset):
    def __len__(self):
        return len(cells)

    def __getitem__(self, i):
        np.random.seed(1000 + i)
        ds.sso_ids = np.array([cells[i]])  # force the cell
        b = ds[0]
        return int(cells[i]), torch.as_tensor(b['features']).float(), torch.as_tensor(b['pts']).float()


loader = torch.utils.data.DataLoader(PerCell(), batch_size=None, num_workers=args.workers, persistent_workers=False)
w = args.width
m = PTv3CellType(in_channels=5, num_classes=args.num_classes, grid_size=0.05,
                 enc_channels=(w, 2 * w, 4 * w, 8 * w, 16 * w), enc_num_head=(max(1, w // 16), max(2, w // 8), max(4, w // 4), max(8, w // 2), max(16, w)),
                 drop_path=0.0).to(dev).eval()
m.load_state_dict(torch.load(args.ckpt, map_location=dev))
if args.infer_mode:
    from syconn.cnn.ptv3.celltype import set_inference_mode
    set_inference_mode(m); print('inference mode: fixed patch, no shuffle, exact per-cloud grid', flush=True)
rows, t_inf, n_samples, all_logits = [], 0.0, 0, []
with torch.no_grad():
    for cid, x, p in loader:
        logits = []
        torch.cuda.synchronize(); t = time.time()
        for s in range(0, x.shape[0], 25):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits.append(m(x[s:s + 25].to(dev), p[s:s + 25].to(dev)).float().cpu())
        torch.cuda.synchronize(); t_inf += time.time() - t; n_samples += x.shape[0]
        logits = torch.cat(logits); all_logits.append(logits.numpy())
        probs = torch.softmax(logits, -1).numpy()  # (R, C)
        rows.append([int(cid), labels[int(cid)]] + [int(probs[:r].mean(0).argmax()) for r in REDUN])
ys = np.array([r[1] for r in rows])
CT = ['DA', 'LMAN', 'HVC', 'MSN', 'STN', 'TAN', 'GPe', 'GPi', 'LTS', 'INT1', 'INT2', 'INT3', 'ASTRO', 'OLIGO', 'MICRO', 'MIGR', 'FRAG', 'INT4', 'OPC'][:args.num_classes]
with open(args.out, 'w') as f:
    wr = csv.writer(f); wr.writerow(['cell', 'label'] + [f'pred_r{r}' for r in REDUN]); wr.writerows(rows)
for j, r in enumerate(REDUN):
    ps = np.array([row[2 + j] for row in rows])
    print(f'RESULT fold={args.cval} redun={r} cells={len(ys)} acc={accuracy_score(ys, ps):.4f} '
          f'macroF1={f1_score(ys, ps, labels=np.unique(ys), average="macro"):.4f}', flush=True)
ps = np.array([row[-1] for row in rows])
print(classification_report(ys, ps, labels=list(range(args.num_classes)), target_names=CT, zero_division=0))
print(f'INFERENCE {n_samples} samples in {t_inf:.1f} s GPU = {t_inf / n_samples * 1000:.1f} ms/sample; wall {time.time() - t0:.0f} s')
if args.save_logits:
    np.savez(args.save_logits, cells=np.array([r[0] for r in rows]), labels=ys, logits=np.stack(all_logits))
print('PERCELL_DONE')
