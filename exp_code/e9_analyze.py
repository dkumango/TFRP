###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- pandas
###############################################################################
"""E9 analysis: epsilon sweep, TFRP vs canonized Zennit (identical weights).

Per (implementation, model, epsilon): median RCE over finite values, the share of images whose
relevance overflowed (NaN/Inf, float32) and the share with RCE > 100% (relevance created).
Output: exp_result/E9/summary.csv and summary.md
Usage : python exp_code/e9_analyze.py
"""

import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'exp_result', 'E9')


def main():
    df = pd.concat([pd.read_csv(os.path.join(OUT, f)) for f in ('tfrp.csv', 'zennit.csv')
                    if os.path.exists(os.path.join(OUT, f))], ignore_index=True)
    rows = []
    for (impl, model, eps), g in df.groupby(['implementation', 'model', 'epsilon']):
        v = g.RCE_pct.to_numpy(dtype=float)
        fin = np.isfinite(v)
        rows.append(dict(implementation=impl, model=model, epsilon=eps, n=len(v),
                         overflow_pct=100 * (~fin).mean(),
                         exploded_pct=100 * (v[fin] > 100).sum() / len(v),
                         RCE_median_finite=np.median(v[fin]) if fin.any() else np.nan))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    lines = ['# E9 epsilon sweep: median RCE % (finite) / overflow % / RCE>100 %\n']
    for model, g in s.groupby('model', sort=False):
        lines.append(f'\n## {model}\n')
        piv = g.pivot(index='epsilon', columns='implementation',
                      values=['RCE_median_finite', 'overflow_pct', 'exploded_pct'])
        lines.append('```\n' + piv.to_string(float_format=lambda v: f'{v:.3g}') + '\n```')
    md = '\n'.join(lines)
    with open(os.path.join(OUT, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write(md + '\n')
    print(md)


if __name__ == '__main__':
    main()
