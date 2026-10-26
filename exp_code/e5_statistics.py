###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- pandas, scipy, statsmodels not required
###############################################################################
"""E5. Statistics of the E4 metrics (ResNet50).

Per method and metric: median with a bootstrap 95 % CI of the median (2,000 resamples, seed 0).
Across methods: Friedman test on the images scored by every method (per metric).
TFRP z+ (default) against each other method: Wilcoxon signed-rank test on paired images, Holm correction
within each metric, and the matched-pairs rank-biserial correlation as effect size.

Input : exp_result/E4/metrics_per_image*.csv (written by e4_metrics.py)
Output: exp_result/E5/summary_metrics.csv, friedman.csv, wilcoxon_vs_tfrp.csv, table_metrics.md
Usage : python exp_code/e5_statistics.py [--reference TFRP_zplus]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, 'exp_result', 'E4')
OUT = os.path.join(ROOT, 'exp_result', 'E5')
# metric: True if higher is better
METRICS = {'deletion_auc': False, 'insertion_auc': True, 'aopc_morf': True, 'road_morf': False,
           'mufidelity': True, 'infidelity': False, 'max_sens': False, 'avg_sens': False,
           'gini': True, 'entropy': False}


def boot_ci(v, n=2000, seed=0):
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    rng = np.random.RandomState(seed)
    med = np.median(v[rng.randint(0, len(v), (n, len(v)))], axis=1)
    return np.percentile(med, 2.5), np.percentile(med, 97.5)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', default='TFRP_zplus')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(IN, 'metrics_per_image*.csv'))], ignore_index=True)
    d = d[d.part.isin(['eval', 'sens'])].drop_duplicates(['method_file', 'index', 'part'])
    # one row per (method, image): sensitivity-only images contribute only to the sensitivity metrics
    wide = d.groupby(['method_file', 'index'])[list(METRICS)].first()
    methods = sorted(wide.index.get_level_values(0).unique())

    summ = []
    for m in methods:
        for k in METRICS:
            v = wide.loc[m, k].to_numpy(float)
            lo, hi = boot_ci(v)
            summ.append(dict(method=m, metric=k, n=int(np.isfinite(v).sum()), median=np.nanmedian(v) if np.isfinite(v).any() else np.nan, ci_lo=lo, ci_hi=hi))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(OUT, 'summary_metrics.csv'), index=False)

    fr, wil = [], []
    for k, higher in METRICS.items():
        t = wide[k].unstack(0).dropna(axis=1, how='all').dropna()   # methods without the metric (RISE: no
        # sensitivity) are dropped, then images x methods complete cases only
        if t.shape[1] >= 3 and len(t) >= 5:
            stat, p = stats.friedmanchisquare(*[t[c] for c in t.columns])
            ranks = t.rank(axis=1, ascending=not higher).mean()
            fr.append(dict(metric=k, n_images=len(t), n_methods=t.shape[1], chi2=stat, p=p,
                           best_mean_rank=ranks.idxmin(), reference_mean_rank=ranks.get(args.reference, np.nan)))
        if args.reference not in t.columns:
            continue
        rows = []
        for m in t.columns.drop(args.reference):
            diff = (t[args.reference] - t[m]).to_numpy()
            diff = diff[diff != 0]
            if len(diff) < 5:
                continue
            res = stats.wilcoxon(diff)
            r_pos = stats.rankdata(np.abs(diff))[diff > 0].sum()
            rbc = (2 * r_pos / (len(diff) * (len(diff) + 1)) - 0.5) * 2   # matched-pairs rank-biserial
            rows.append(dict(metric=k, other=m, n=len(t), p=res.pvalue, rank_biserial=rbc,
                             better=(rbc > 0) == higher))
        if rows:
            ps = holm([r['p'] for r in rows])
            for r, pa in zip(rows, ps):
                r['p_holm'] = pa
                better = r.pop('better')
                r['reference_better'] = better if pa < 0.05 else None
            wil += rows
    pd.DataFrame(fr).to_csv(os.path.join(OUT, 'friedman.csv'), index=False)
    pd.DataFrame(wil).to_csv(os.path.join(OUT, 'wilcoxon_vs_tfrp.csv'), index=False)

    # markdown table: median [95 % CI]
    lines = ['# E5 metric medians [bootstrap 95 % CI of the median] (ResNet50)', '',
             '| method | ' + ' | '.join(f'{k} {"↑" if h else "↓"}' for k, h in METRICS.items()) + ' |',
             '|---' * (len(METRICS) + 1) + '|']
    for m in methods:
        cells = []
        for k in METRICS:
            r = summ[(summ.method == m) & (summ.metric == k)].iloc[0]
            cells.append(f'{r["median"]:.3g} [{r.ci_lo:.3g}, {r.ci_hi:.3g}] (n={r.n})' if r.n else '–')
        lines.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    open(os.path.join(OUT, 'table_metrics.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print(pd.DataFrame(fr).to_string())
    print(pd.DataFrame(wil).round(4).to_string())


if __name__ == '__main__':
    main()
