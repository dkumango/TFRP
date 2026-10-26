###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- pandas
###############################################################################
"""E13. Summarise the out-of-the-box conservation study across libraries.

Reads exp_result/E13/{tfrp,torch,innvestigate}_per_image.csv and writes
  exp_result/E13/summary_long.csv   one row per (library, config, rule, model)
  exp_result/E13/summary_wide.csv   models as columns, cell = status or median RCE (%)
Usage: python exp_code/e13_analyze.py
"""

import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E13 = os.path.join(ROOT, 'exp_result', 'E13')
MODEL_ORDER = ['vgg16', 'resnet50', 'inception_v3', 'mobilenet_v2', 'densenet121', 'efficientnet_b0', 'convnext_tiny']


def cell(g):
    ok = g[g.status == 'OK']
    if len(ok) == 0:
        return g.status.iloc[0]
    return f'{np.median(ok.RCE_pct):.2g}'


def main():
    files = sorted(f for f in os.listdir(E13) if 'per_image' in f and f.endswith('.csv'))
    frames = [pd.read_csv(os.path.join(E13, f)) for f in files]
    df = pd.concat(frames, ignore_index=True)
    keys = ['library', 'config', 'rule', 'model']

    long = (df.groupby(keys, sort=False)
              .apply(lambda g: pd.Series({
                  'status': 'OK' if (g.status == 'OK').all() else g.status[g.status != 'OK'].iloc[0],
                  'n_ok': int((g.status == 'OK').sum()),
                  'RCE_median_pct': g.RCE_pct.median(),
                  'RCE_q25_pct': g.RCE_pct.quantile(0.25),
                  'RCE_q75_pct': g.RCE_pct.quantile(0.75),
                  'time_median_s': g.time_s.median() if 'time_s' in g else np.nan,
                  'error': g.error.dropna().iloc[0] if 'error' in g and g.error.notna().any() else '',
              }), include_groups=False)
              .reset_index())
    long.to_csv(os.path.join(E13, 'summary_long.csv'), index=False)

    wide = (df.groupby(keys, sort=False).apply(cell, include_groups=False)
              .unstack('model').reindex(columns=MODEL_ORDER))
    wide.to_csv(os.path.join(E13, 'summary_wide.csv'))
    pd.set_option('display.width', 250)
    print('Cell = median RCE (%) over images, or the failure status\n')
    print(wide.to_string())


if __name__ == '__main__':
    main()
