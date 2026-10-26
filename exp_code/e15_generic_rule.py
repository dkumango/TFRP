###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- tensorflow 2.20, keras 3.14, xplique 2.0.1
###############################################################################
"""E15 (exploratory). Generic AD-derived rule vs. the heuristic rules of TFRP.

On EfficientNet-B0 (SiLU, sigmoid gates of squeeze-and-excitation) and ConvNeXt-T (GELU, layer
normalisation) the default TFRP rules for gates (signal-take-all) and non-ReLU unary operations
(identity) are heuristics. Here they are replaced by the node-level path-integrated rule
(generic_rule='ig', 32 midpoint steps) and compared with the defaults:
  RCE (%), Pearson correlation between the two maps, Deletion / Insertion AUC (xplique), time per map.

Output: exp_result/E15/generic_rule_per_image.csv, generic_rule_summary.csv
Usage : python exp_code/e15_generic_rule.py [--n 50] [--threads 1]
"""

import argparse
import os
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import tensorflow as tf
from xplique.metrics import Deletion, Insertion

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('exp_code/common', 'exp_code/tfrp'):
    sys.path.insert(0, os.path.join(ROOT, p))
import preprocess as P
import general_TFRP_v2 as V2

A = tf.keras.applications
MODELS = {'efficientnet_b0': (A.EfficientNetB0, A.efficientnet.preprocess_input),
          'convnext_tiny': (A.ConvNeXtTiny, A.convnext.preprocess_input)}
OUT = os.path.join(ROOT, 'exp_result', 'E15')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    os.makedirs(OUT, exist_ok=True)
    idx = P.sample_indices(args.n, seed=42)
    names = P.image_list()
    rows, summ = [], []
    for mname in args.models.split(','):
        ctor, prep = MODELS[mname]
        model = ctor(weights='imagenet')
        xs = np.stack([prep(np.asarray(P.rgb_cache(224)[i], np.float32).copy()) for i in idx])
        targets = model.predict(xs, batch_size=16, verbose=0).argmax(1)   # softmax argmax = logit argmax
        maps = {}
        for rule in (None, 'ig'):
            ms = []
            for k, i in enumerate(idx):
                t0 = time.time()
                R, seed, total, _ = V2.get_relevance_map_generalized(model, xs[k:k + 1], target_class_idx=int(targets[k]),
                                                                     generic_rule=rule)
                ms.append(R[0].sum(-1))
                rows.append(dict(model=mname, rule=rule or 'heuristic', image=names[i], seed=seed, sum_R_in=total,
                                 RCE_pct=abs(total - seed) / abs(seed) * 100, time_s=time.time() - t0))
            maps[rule or 'heuristic'] = np.stack(ms)
            print(f'{mname} {rule or "heuristic"} done', flush=True)
        y = tf.keras.utils.to_categorical(targets, 1000)
        corr = [np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b in zip(maps['heuristic'], maps['ig'])]
        for rule, ms in maps.items():
            expl = np.nan_to_num(ms[..., None].astype(np.float32))
            r = pd.DataFrame([x for x in rows if x['model'] == mname and x['rule'] == rule])
            summ.append(dict(model=mname, rule=rule, RCE_median=r.RCE_pct.median(), RCE_q75=r.RCE_pct.quantile(.75),
                             time_median_s=r.time_s.median(),
                             deletion_auc=float(Deletion(model, xs, y, batch_size=16)(expl)),
                             insertion_auc=float(Insertion(model, xs, y, batch_size=16)(expl)),
                             pearson_heuristic_vs_ig=float(np.median(corr))))
            print(summ[-1], flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f'generic_rule_per_image{args.tag}.csv'), index=False)
        pd.DataFrame(summ).to_csv(os.path.join(OUT, f'generic_rule_summary{args.tag}.csv'), index=False)
        np.savez_compressed(os.path.join(OUT, f'maps_{mname}.npz'), **maps, indices=np.array(idx))
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    main()
