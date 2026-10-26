###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- numpy, pandas, scipy
###############################################################################
"""E1 analysis: conservation, paired statistics and map agreement across implementations.

Outputs (exp_result/E1/):
  summary_rce.csv     median RCE, bootstrap 95% CI of the median, IQR, status per (impl, model, rule)
  wilcoxon.csv        paired Wilcoxon signed-rank TFRP vs each implementation (Holm-corrected)
  map_agreement.csv   per (impl, model, rule): median Pearson / Spearman of TFRP vs the implementation
                      over the first 50 images (channel-summed maps)
  table_rce.md        compact markdown table for the manuscript draft
Usage: python exp_code/e1_analyze.py
"""

import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('exp_code', 'exp_code/common'):
    sys.path.insert(0, os.path.join(ROOT, p))
import e1_common as C
import preprocess as P

REF = 'TFRP v2'
IMPL_ORDER = ['TFRP v2', 'Zennit (canonized)', 'Zennit (not canonized)', 'Captum', 'iNNvestigate']


def boot_median_ci(v, n_boot=2000, seed=0):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    med = np.median(rng.choice(v, size=(n_boot, v.size), replace=True), axis=1)
    return np.percentile(med, 2.5), np.percentile(med, 97.5)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def load():
    frames = [pd.read_csv(os.path.join(C.OUT_DIR, f)) for f in ('tfrp.csv', 'torch.csv', 'innvestigate.csv')
              if os.path.exists(os.path.join(C.OUT_DIR, f))]
    return pd.concat(frames, ignore_index=True)


def summary(df):
    rows = []
    for (impl, model, rule), g in df.groupby(['implementation', 'model', 'rule']):
        ok = g[g.status == 'OK']
        status = 'OK' if len(ok) == len(g) else g.status[g.status != 'OK'].iloc[0]
        lo, hi = boot_median_ci(ok.RCE_pct)
        rows.append(dict(implementation=impl, model=model, rule=rule, status=status, n_ok=len(ok),
                         RCE_median=ok.RCE_pct.median(), RCE_ci_lo=lo, RCE_ci_hi=hi,
                         RCE_q25=ok.RCE_pct.quantile(.25), RCE_q75=ok.RCE_pct.quantile(.75),
                         n_nonfinite=int((~np.isfinite(ok.RCE_pct)).sum()),
                         time_median_s=ok.time_s.median()))
    return pd.DataFrame(rows)


def wilcoxon(df):
    rows = []
    ok = df[df.status == 'OK']
    for (model, rule), g in ok.groupby(['model', 'rule']):
        piv = g.pivot_table(index='image', columns='implementation', values='RCE_pct')
        if REF not in piv:
            continue
        for impl in piv.columns:
            if impl == REF:
                continue
            pair = piv[[REF, impl]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 10 or np.allclose(pair[REF], pair[impl]):
                p = np.nan
            else:
                p = stats.wilcoxon(pair[REF], pair[impl]).pvalue
            rows.append(dict(model=model, rule=rule, comparison=f'{REF} vs {impl}', n=len(pair),
                             median_TFRP=pair[REF].median(), median_other=pair[impl].median(), p=p))
    out = pd.DataFrame(rows)
    if len(out):
        mask = out.p.notna()
        out.loc[mask, 'p_holm'] = holm(out.loc[mask, 'p'])
    return out


def map_agreement():
    mdir = os.path.join(P.CACHE_DIR, 'E1_maps')
    if not os.path.isdir(mdir):
        return pd.DataFrame()
    rows = []
    safe = lambda s: s.replace(' ', '_').replace('(', '').replace(')', '')
    for model in C.MODELS:
        for rule in C.RULES:
            ref_path = os.path.join(mdir, f'{safe(REF)}__{model}__{rule}.npy')
            if not os.path.exists(ref_path):
                continue
            ref = np.load(ref_path).astype(np.float64)
            for impl in IMPL_ORDER[1:]:
                path = os.path.join(mdir, f'{safe(impl)}__{model}__{rule}.npy')
                if not os.path.exists(path):
                    continue
                other = np.load(path).astype(np.float64)
                n = min(len(ref), len(other))
                pr, sr = [], []
                for a, b in zip(ref[:n], other[:n]):
                    a, b = a.ravel(), b.ravel()
                    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))) or a.std() == 0 or b.std() == 0:
                        continue
                    pr.append(stats.pearsonr(a, b)[0])
                    sr.append(stats.spearmanr(a, b)[0])
                rows.append(dict(model=model, rule=rule, implementation=impl, n=len(pr),
                                 pearson_median=np.median(pr) if pr else np.nan,
                                 spearman_median=np.median(sr) if sr else np.nan))
    return pd.DataFrame(rows)


def md_table(s):
    lines = []
    for rule in C.RULES:
        sub = s[s.rule == rule]
        lines.append(f'\n### {rule}\n')
        lines.append('| implementation | ' + ' | '.join(C.MODELS) + ' |')
        lines.append('|---|' + '---|' * len(C.MODELS))
        for impl in IMPL_ORDER:
            cells = []
            for m in C.MODELS:
                r = sub[(sub.implementation == impl) & (sub.model == m)]
                if r.empty:
                    cells.append('–')
                elif r.status.iloc[0] != 'OK':
                    cells.append(r.status.iloc[0])
                else:
                    cells.append(f'{r.RCE_median.iloc[0]:.2g} [{r.RCE_ci_lo.iloc[0]:.2g}, {r.RCE_ci_hi.iloc[0]:.2g}]')
            lines.append(f'| {impl} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main():
    df = load()
    s = summary(df)
    s.to_csv(os.path.join(C.OUT_DIR, 'summary_rce.csv'), index=False)
    w = wilcoxon(df)
    w.to_csv(os.path.join(C.OUT_DIR, 'wilcoxon.csv'), index=False)
    a = map_agreement()
    a.to_csv(os.path.join(C.OUT_DIR, 'map_agreement.csv'), index=False)
    md = '# E1 median RCE (%) [bootstrap 95% CI of the median]\n' + md_table(s)
    with open(os.path.join(C.OUT_DIR, 'table_rce.md'), 'w', encoding='utf-8') as f:
        f.write(md + '\n')
    print(md)
    if len(a):
        print('\n# Map agreement with TFRP (median over images)\n')
        print(a.to_string(index=False))


if __name__ == '__main__':
    main()
