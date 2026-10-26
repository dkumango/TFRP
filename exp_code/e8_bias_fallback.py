###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- tensorflow 2.20, keras 3.14, xplique 2.0.1
###############################################################################
"""E8. Bias-aware variants of z+ on the architectures where z+ loses relevance to biases.

Under z+ a unit with z+ = 0 but non-zero relevance is active only through its (folded batch-norm)
bias, and the bias absorbs its relevance (Proposition 1; E2: MobileNetV2 99.9 %, EfficientNet-B0 39.4 %).
Variants (all other units keep z+):
  zplus        standard z+ (TFRP default)
  floor1e-9    z+ with max(w, 1e-9) -- the hidden behaviour of the submitted code (v1)
  fallback_flat  relevance of z+ = 0 units sent uniformly to their receptive field (ZPLUS_FALLBACK='flat')
  fallback_w2    ... in proportion to w^2 (ZPLUS_FALLBACK='w2')
Metrics: RCE (%), where the remaining loss occurs (layer log), Deletion / Insertion AUC (xplique),
Pearson correlation with the standard z+ map, time per map. ResNet50 is a control (no z+ = 0 units).

Output: exp_result/E8/per_image.csv, summary.csv, loss_by_type.csv, maps_<model>.npz
Usage : python exp_code/e8_bias_fallback.py [--n 100] [--threads 4] [--models efficientnet_b0]
"""

import argparse
import collections
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
MODELS = {'mobilenet_v2': (A.MobileNetV2, A.mobilenet_v2.preprocess_input),
          'efficientnet_b0': (A.EfficientNetB0, A.efficientnet.preprocess_input),
          'resnet50': (A.ResNet50, A.resnet50.preprocess_input)}
VARIANTS = {'zplus': dict(floor=0.0, fallback=None), 'floor1e-9': dict(floor=1e-9, fallback=None),
            'fallback_flat': dict(floor=0.0, fallback='flat'), 'fallback_w2': dict(floor=0.0, fallback='w2')}
OUT = os.path.join(ROOT, 'exp_result', 'E8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--n_control', type=int, default=20, help='images for the ResNet50 control')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--models', default=','.join(MODELS), help='re-run a subset; other models are kept in the CSVs')
    args = ap.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    os.makedirs(OUT, exist_ok=True)
    names = P.image_list()
    rows, summ, loss_rows = [], [], []
    run = args.models.split(',')
    old = {k: pd.read_csv(os.path.join(OUT, f'{k}.csv')) for k in ('per_image', 'summary', 'loss_by_type')
           if os.path.exists(os.path.join(OUT, f'{k}.csv'))}
    old = {k: d[~d.model.isin(run)] for k, d in old.items()}
    for mname in run:
        ctor, prep = MODELS[mname]
        idx = P.sample_indices(args.n_control if mname == 'resnet50' else args.n, seed=42)
        model = ctor(weights='imagenet')
        xs = np.stack([prep(np.asarray(P.rgb_cache(224)[i], np.float32).copy()) for i in idx])
        targets = model.predict(xs, batch_size=16, verbose=0).argmax(1)
        maps = {}
        for vname, cfg in VARIANTS.items():
            V2.ZPLUS_WEIGHT_FLOOR, V2.ZPLUS_FALLBACK = cfg['floor'], cfg['fallback']
            ms, by_type = [], collections.defaultdict(list)
            for k, i in enumerate(idx):
                t0 = time.time()
                R, seed, total, _, log = V2.get_relevance_map_generalized(
                    model, xs[k:k + 1], target_class_idx=int(targets[k]), return_layer_log=True)
                dt = time.time() - t0
                ms.append(R[0].sum(-1))
                per = collections.defaultdict(float)
                for _, otype, up, lo in log:
                    per[otype] += (up - lo) / seed * 100
                for otype, v in per.items():
                    by_type[otype].append(v)
                rows.append(dict(model=mname, variant=vname, image=names[i], seed=seed, sum_R_in=total,
                                 RCE_pct=abs(total - seed) / abs(seed) * 100, time_s=dt))
            maps[vname] = np.stack(ms)
            for otype, v in by_type.items():
                if abs(np.median(v)) > 1e-6:
                    loss_rows.append(dict(model=mname, variant=vname, op_type=otype, loss_pct_median=float(np.median(v))))
            print(f'{mname} {vname} done', flush=True)
        V2.ZPLUS_WEIGHT_FLOOR, V2.ZPLUS_FALLBACK = 0.0, None
        y = tf.keras.utils.to_categorical(targets, 1000)
        for vname, ms in maps.items():
            expl = np.nan_to_num(ms[..., None].astype(np.float32))
            r = pd.DataFrame([x for x in rows if x['model'] == mname and x['variant'] == vname])
            corr = [np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b in zip(maps['zplus'], ms)]
            summ.append(dict(model=mname, variant=vname, n=len(r), RCE_median=r.RCE_pct.median(),
                             RCE_q25=r.RCE_pct.quantile(.25), RCE_q75=r.RCE_pct.quantile(.75),
                             deletion_auc=float(Deletion(model, xs, y, batch_size=16)(expl)),
                             insertion_auc=float(Insertion(model, xs, y, batch_size=16)(expl)),
                             pearson_vs_zplus=float(np.nanmedian(corr)), time_median_s=r.time_s.median()))
            print(summ[-1], flush=True)
        for key, new in (('per_image', rows), ('summary', summ), ('loss_by_type', loss_rows)):
            pd.concat([old.get(key, pd.DataFrame()), pd.DataFrame(new)], ignore_index=True).to_csv(
                os.path.join(OUT, f'{key}.csv'), index=False)
        np.savez_compressed(os.path.join(OUT, f'maps_{mname}.npz'), **maps, indices=np.array(idx))
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    main()
