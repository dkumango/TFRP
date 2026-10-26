###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- tensorflow 2.20, keras 3.14, scipy
###############################################################################
"""E4. Faithfulness, robustness and complexity metrics, one implementation for all methods.

Every method's maps (exp_code/e4_maps_*.py) are evaluated on the same keras ResNet50, the same
images and the same random draws. Metrics (per image, so that E5 can test paired differences):

  deletion_auc   Petsiuk et al. 2018. Pixels removed in descending relevance (to 0 = mean colour),
                 20 steps; AUC of the target probability (lower is better)
  insertion_auc  Petsiuk et al. 2018. Pixels inserted into a blurred image (Gaussian, sigma 5)
                 in descending relevance, 20 steps; AUC (higher is better)
  aopc_morf      Samek et al. 2017. 16x16 patches replaced by uniform noise in descending patch
                 relevance, 50 steps; mean drop of the target probability (higher is better)
  mufidelity     Bhatt et al. 2020. 128 random subsets of a 9x9 grid (20% removed); Pearson
                 correlation between removed relevance and probability drop (higher is better)
  infidelity     Yeh et al. 2019. 50 Gaussian perturbations (sigma = 0.05 of the pixel range) of the
                 target logit; with the optimal attribution scale beta*, normalised by E[dF^2],
                 so that it is scale-free (lower is better)
  road_morf      Rong et al. 2022 (100 images). Most relevant 10/30/50/70/90% of pixels replaced by noisy
                 linear imputation; mean target probability over the levels (lower is better)
  max_sens, avg_sens  Yeh et al. 2019, normalised: max / mean over draws of
                 ||phi(x + d) - phi(x)|| / ||phi(x)||, d ~ U(-r, r), r = 0.02 of the range (lower is better)
  gini, entropy  complexity of |phi| (Chalasani et al. 2020; Bhatt et al. 2020)

Output: exp_result/E4/metrics_per_image.csv (resumable per method)
Usage : python exp_code/e4_metrics.py [--methods ...] [--threads 0]
"""

import argparse
import os
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
# Model calls use one fixed batch shape (Scorer._batched): with oneDNN, varying batch sizes made
# the resident memory grow to >10 GB within 40 images.
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import tensorflow as tf
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('exp_code', 'exp_code/common'):
    sys.path.insert(0, os.path.join(ROOT, p))
import preprocess as P
import e4_common as E

OUT_CSV = os.path.join(ROOT, 'exp_result', 'E4', 'metrics_per_image.csv')
STEPS = 20
AOPC_PATCH, AOPC_STEPS = 16, 50
MUF_GRID, MUF_FRAC, MUF_SAMPLES, N_MUF = 9, 0.2, 128, 250
INF_SAMPLES, INF_SIGMA = 50, 0.05 * 255.0
ROAD_LEVELS, N_ROAD = (0.1, 0.3, 0.5, 0.7, 0.9), 100
ROAD_NOISE = 0.2 * 58.0     # 0.2 of the ImageNet pixel std, as in the ROAD reference implementation
PIX_LO, PIX_HI = -124.0, 152.0   # caffe-preprocessed pixel range


class Scorer:
    def __init__(self):
        self.prob = tf.keras.applications.ResNet50(weights='imagenet')
        self.logit = tf.keras.applications.ResNet50(weights='imagenet', classifier_activation=None)

    @staticmethod
    def _batched(model, xs, target, batch=16):
        """Fixed batch shape (the last batch is zero-padded) so that no new kernels are built per size."""
        out = []
        for b in range(0, len(xs), batch):
            chunk = xs[b:b + batch]
            n = len(chunk)
            if n < batch:
                chunk = np.concatenate([chunk, np.zeros((batch - n,) + chunk.shape[1:], chunk.dtype)])
            out.append(model(chunk, training=False).numpy()[:n, target])
        return np.concatenate(out)

    def p(self, xs, target):
        return self._batched(self.prob, xs, target)

    def z(self, xs, target):
        return self._batched(self.logit, xs, target)


def auc(curve):
    curve = np.asarray(curve)
    return float((curve.sum() - 0.5 * (curve[0] + curve[-1])) / (len(curve) - 1))


def deletion_insertion(S, x, phi, t, blurred):
    order = np.argsort(-phi.ravel(), kind='stable')
    n = phi.size
    dels, ins = [], []
    for k in range(STEPS + 1):
        mask = np.zeros(n, bool)
        mask[order[:int(round(k / STEPS * n))]] = True
        mask = mask.reshape(phi.shape)[..., None]
        dels.append(np.where(mask, 0.0, x))
        ins.append(np.where(mask, x, blurred))
    p = S.p(np.stack(dels + ins).astype(np.float32), t)
    return auc(p[:STEPS + 1]), auc(p[STEPS + 1:])


def aopc(S, x, phi, t, rng):
    g = phi.shape[0] // AOPC_PATCH
    patch = phi[:g * AOPC_PATCH, :g * AOPC_PATCH].reshape(g, AOPC_PATCH, g, AOPC_PATCH).sum((1, 3)).ravel()
    order = np.argsort(-patch, kind='stable')
    xs, cur = [x.copy()], x.copy()
    for k in range(AOPC_STEPS):
        r, c = divmod(order[k], g)
        cur = cur.copy()
        cur[r * AOPC_PATCH:(r + 1) * AOPC_PATCH, c * AOPC_PATCH:(c + 1) * AOPC_PATCH] = \
            rng.uniform(PIX_LO, PIX_HI, (AOPC_PATCH, AOPC_PATCH, 3))
        xs.append(cur)
    p = S.p(np.stack(xs).astype(np.float32), t)
    return float(np.mean(p[0] - p))


def mufidelity(S, x, phi, t, rng):
    H, W = phi.shape
    cells = MUF_GRID * MUF_GRID
    k = int(round(MUF_FRAC * cells))
    ys = np.minimum(np.arange(H) * MUF_GRID // H, MUF_GRID - 1)
    xs_ = np.minimum(np.arange(W) * MUF_GRID // W, MUF_GRID - 1)
    cell_of = ys[:, None] * MUF_GRID + xs_[None, :]
    cell_rel = np.bincount(cell_of.ravel(), weights=phi.ravel(), minlength=cells)
    subsets = [rng.choice(cells, k, replace=False) for _ in range(MUF_SAMPLES)]
    batch = [np.where(np.isin(cell_of, s)[..., None], 0.0, x) for s in subsets]
    p0 = S.p(x[None].astype(np.float32), t)[0]
    drop = p0 - S.p(np.stack(batch).astype(np.float32), t)
    removed = np.array([cell_rel[s].sum() for s in subsets])
    if drop.std() == 0 or removed.std() == 0:
        return np.nan
    return float(np.corrcoef(removed, drop)[0, 1])


def infidelity(S, x, phi, t, rng):
    I = rng.normal(0, INF_SIGMA, (INF_SAMPLES,) + x.shape).astype(np.float32)
    f0 = S.z(x[None].astype(np.float32), t)[0]
    df = f0 - S.z((x[None] - I).astype(np.float32), t)
    ip = (I * phi[None, ..., None]).reshape(INF_SAMPLES, -1).sum(1)
    beta = (ip * df).sum() / ((ip * ip).sum() + 1e-12)
    return float(np.mean((beta * ip - df) ** 2) / (np.mean(df ** 2) + 1e-12))


NEIGHBOURS = [(-1, 0, 1 / 6), (1, 0, 1 / 6), (0, -1, 1 / 6), (0, 1, 1 / 6),
              (-1, -1, 1 / 12), (-1, 1, 1 / 12), (1, -1, 1 / 12), (1, 1, 1 / 12)]


def road_impute(x, removed, rng):
    """Noisy linear imputation (ROAD): every removed pixel equals the weighted mean of its in-image
    8-neighbours (weights 1/6 direct, 1/12 diagonal), solved jointly; Gaussian noise is then added.

    For removed pixel p:  sum_q w_q * x_p - sum_{q removed} w_q * x_q = sum_{q known} w_q * x_q
    """
    H, W = removed.shape
    rr, cc = np.nonzero(removed)
    n = len(rr)
    idx = -np.ones((H, W), int)
    idx[rr, cc] = np.arange(n)
    diag = np.zeros(n)
    b = np.zeros((n, 3))
    rows, cols, vals = [], [], []
    for dr, dc, w in NEIGHBOURS:
        r2, c2 = rr + dr, cc + dc
        inside = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
        r2c, c2c = np.clip(r2, 0, H - 1), np.clip(c2, 0, W - 1)
        diag += w * inside
        unknown = inside & removed[r2c, c2c]
        known = inside & ~removed[r2c, c2c]
        rows.append(np.nonzero(unknown)[0])
        cols.append(idx[r2c[unknown], c2c[unknown]])
        vals.append(np.full(unknown.sum(), -w))
        b[known] += w * x[r2c[known], c2c[known]]
    A = sp.csc_matrix((np.concatenate(vals + [diag]),
                       (np.concatenate(rows + [np.arange(n)]), np.concatenate(cols + [np.arange(n)]))),
                      shape=(n, n))
    sol = spla.splu(A).solve(b)
    out = x.copy()
    out[rr, cc] = sol + rng.normal(0, ROAD_NOISE, sol.shape)
    return out


def road(S, x, phi, t, rng):
    order = np.argsort(-phi.ravel(), kind='stable')
    xs = []
    for lv in ROAD_LEVELS:
        removed = np.zeros(phi.size, bool)
        removed[order[:int(round(lv * phi.size))]] = True
        xs.append(road_impute(x, removed.reshape(phi.shape), rng))
    return float(np.mean(S.p(np.stack(xs).astype(np.float32), t)))


def complexity(phi):
    a = np.sort(np.abs(phi).ravel())
    n = a.size
    gini = float(1 - 2 * np.sum((n - np.arange(1, n + 1) + 0.5) * a) / (n * a.sum() + 1e-12))
    p = a / (a.sum() + 1e-12)
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return gini, ent


def sensitivity(method):
    if not (E.exists(method, 'sens') and E.exists(method, 'sensbase')):
        return {}
    base = np.load(E.map_path(method, 'sensbase')).astype(np.float32)
    noisy = np.load(E.map_path(method, 'sens')).astype(np.float32)
    out = {}
    for j, i in enumerate(E.sens_indices()):
        nb = np.linalg.norm(base[j]) + 1e-12
        d = [np.linalg.norm(noisy[j, k] - base[j]) / nb for k in range(noisy.shape[1])]
        out[i] = (float(np.max(d)), float(np.mean(d)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', default='')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--out', default='', help='result file name in exp_result/E4 for a parallel run '
                                               '(default metrics_per_image.csv)')
    ap.add_argument('--max_images', type=int, default=40,
                    help='images processed per call; memory grows over long runs, so run_e4_metrics.sh '
                         'restarts the process until ALL METRICS DONE')
    args = ap.parse_args()
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    available = sorted({f.split('__')[0] for f in os.listdir(E.maps_dir()) if f.endswith('__eval.npy')})
    methods = args.methods.split(',') if args.methods else available
    out_csv = os.path.join(os.path.dirname(OUT_CSV), args.out) if args.out else OUT_CSV
    done_rows = {}
    # results of all parallel runs count as done (metrics_per_image*.csv)
    parts = [os.path.join(os.path.dirname(OUT_CSV), f) for f in os.listdir(os.path.dirname(OUT_CSV))
             if f.startswith('metrics_per_image') and f.endswith('.csv')]
    if parts:
        prev = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True)
        for mf, g in prev.groupby('method_file'):
            done_rows[mf] = set(g.loc[g.part == 'eval', 'index'].astype(int))
            if (g.part == 'sens_done').any():
                done_rows[('sens', mf)] = True

    S = Scorer()
    ev = E.eval_indices()
    x_ev = np.stack([P.model_input(E.MODEL, i)[0] for i in ev])
    with np.load(os.path.join(E.maps_dir(), 'targets.npz')) as z:   # written by e4_maps_tf.py
        t_ev = z['eval']
    # random sub-subsets of the evaluation images for the two expensive metrics
    muf_set = set(np.random.RandomState(11).choice(ev, N_MUF, replace=False).tolist())
    road_set = set(np.random.RandomState(13).choice(ev, N_ROAD, replace=False).tolist())

    budget = args.max_images
    skipped = []
    for mf in methods:
        mf = E.safe(mf)
        prev = done_rows.get(mf, set())
        todo = [j for j, i in enumerate(ev) if i not in prev]
        if not todo and ('sens', mf) in done_rows:
            continue
        # parallel runs (--out) must not score the same method: a method touched by another run within
        # the last 30 min is left to that run
        lock = os.path.join(os.path.dirname(OUT_CSV), 'locks', f'{mf}.owner')
        os.makedirs(os.path.dirname(lock), exist_ok=True)
        if os.path.exists(lock) and time.time() - os.path.getmtime(lock) < 1800:
            if open(lock).read().strip() != os.path.basename(out_csv):
                skipped.append(mf)
                continue
        with open(lock, 'w') as fh:
            fh.write(os.path.basename(out_csv))
        maps = np.load(os.path.join(E.maps_dir(), f'{mf}__eval.npy'), mmap_mode='r')
        sens = sensitivity(mf)
        rows, t_start = [], time.time()
        for j in todo[:budget]:
            i = ev[j]
            x, phi, t = x_ev[j], np.nan_to_num(np.asarray(maps[j], np.float32)), int(t_ev[j])
            rng = np.random.RandomState(500000 + i)
            blurred = np.stack([gaussian_filter(x[..., c], 5) for c in range(3)], -1)
            row = dict(method_file=mf, index=i, target=t)
            row['deletion_auc'], row['insertion_auc'] = deletion_insertion(S, x, phi, t, blurred)
            row['aopc_morf'] = aopc(S, x, phi, t, rng)
            row['infidelity'] = infidelity(S, x, phi, t, rng)
            row['mufidelity'] = mufidelity(S, x, phi, t, rng) if i in muf_set else np.nan
            row['road_morf'] = road(S, x, phi, t, rng) if i in road_set else np.nan
            row['gini'], row['entropy'] = complexity(phi)
            row['max_sens'], row['avg_sens'] = sens.get(i, (np.nan, np.nan))
            row['part'] = 'eval'
            rows.append(row)
        budget -= len(rows)
        if len(rows) == len(todo):
            # method complete: sensitivity images that are not in the evaluation subset
            for i, (mx, av) in sens.items():
                if i not in set(ev):
                    rows.append(dict(method_file=mf, index=i, max_sens=mx, avg_sens=av, part='sens'))
            rows.append(dict(method_file=mf, part='sens_done'))
        if rows:
            pd.DataFrame(rows).to_csv(out_csv, mode='a', header=not os.path.exists(out_csv), index=False)
        n_now = len(prev) + len([r for r in rows if r.get('part') == 'eval'])
        print(f'{mf:28s} {n_now}/{len(ev)} images ({(time.time() - t_start) / max(1, len(todo[:args.max_images])):.1f}s/img)',
              flush=True)
        if budget <= 0:
            return
    if skipped:
        print(f'methods left to parallel runs: {skipped}', flush=True)
    print('ALL METRICS DONE', flush=True)


if __name__ == '__main__':
    main()
