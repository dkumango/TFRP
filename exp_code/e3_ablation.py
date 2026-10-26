###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- tensorflow 2.20, keras 3.14, xplique 2.0.1
###############################################################################
"""E3. Component ablation of TFRP (z+ rule, four CNNs).

Variants (one component changed at a time with respect to V0):
  V0 default        uniform split at Add, gradient (uniform) average pooling, sign-aware first layer
                    (the TFRP default when E3 was run; the default is V3 since 2026-10-23)
  V1 add-prop       proportional split at Add (relevance proportional to each branch's activation)
  V2 pool-prop      proportional redistribution in AveragePooling2D / GAP
  V3 add+pool-prop  V1 and V2 together (the redistribution used by canonized Zennit)
  V4 legacy-input   first layer w+x on signed input (the submitted version's rule)
  V5 v1-published   the submitted implementation (layer-list traversal, original defects)
  V6 weight-floor   z+ with the v1 positive part max(w, 1e-9) instead of max(w, 0)
Metrics
  RCE (%)                    500 images (seed-42 random subset)
  agreement with Zennit      median Pearson / Spearman vs. canonized Zennit z+ maps from E1 (same 50 images)
  Deletion / Insertion AUC   xplique, 100 images (seed-42 random subset)

Execution is chunked: every call processes at most --chunk images of the first unfinished
(model, variant) block and exits (TensorFlow's CPU allocator grows by several GB per 1000
propagations). A block is finalised -- agreement and fidelity computed, rows moved to the result
files -- by the call that completes it. Run in a loop until 'ALL BLOCKS DONE' (run_e3.sh).

Output: exp_result/E3/ablation_per_image.csv, ablation_fidelity.csv, summary.csv
Usage : bash exp_code/run_e3.sh
"""

import argparse
import contextlib
import io
import os
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from xplique.metrics import Deletion, Insertion

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('exp_code', 'exp_code/common', 'exp_code/tfrp', 'exp_code/tfrp'):
    sys.path.insert(0, os.path.join(ROOT, p))
import preprocess as P
import general_TFRP_v2 as V2
import general_TFRP_v1 as V1

A = tf.keras.applications
CTORS = {'vgg16': A.VGG16, 'resnet50': A.ResNet50, 'inception_v3': A.InceptionV3, 'mobilenet_v2': A.MobileNetV2}
OUT_DIR = os.path.join(ROOT, 'exp_result', 'E3')
PART_DIR = os.path.join(OUT_DIR, 'partial')
# V0 is the v1 redistribution (uniform at Add and average pooling); since 2026-10-23 the TFRP default
# is V3. All options are explicit so that the ablation is independent of the library defaults.
UNIFORM = dict(add_rule='uniform', avgpool_rule='gradient')
VARIANTS = {
    'V0 default': dict(UNIFORM),
    'V1 add-prop': dict(add_rule='proportional', avgpool_rule='gradient'),
    'V2 pool-prop': dict(add_rule='uniform', avgpool_rule='proportional'),
    'V3 add+pool-prop': dict(add_rule='proportional', avgpool_rule='proportional'),
    'V4 legacy-input': dict(UNIFORM, legacy_input=True),
    'V5 v1-published': dict(v1=True),
    'V6 weight-floor': dict(UNIFORM, weight_floor=1e-9),
}
N_RCE, N_MAP, N_FID = 500, 50, 100


def relevance(model, x, opts):
    """Raw input relevance (1, H, W, C), seed and propagated total for one variant."""
    opts = dict(opts)
    if opts.pop('v1', False):
        V1.SIGNED_INPUT_ZPLUS = False
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                R, seed, total, _ = V1.get_relevance_map_generalized(model, x, use_logit=True, global_rule='z_plus')
        finally:
            V1.SIGNED_INPUT_ZPLUS = True
        return R * (total / seed), seed, total
    V2.SIGNED_INPUT_ZPLUS = not opts.pop('legacy_input', False)
    V2.ZPLUS_WEIGHT_FLOOR = opts.pop('weight_floor', 0.0)
    try:
        R, seed, total, _ = V2.get_relevance_map_generalized(model, x, use_logit=True, global_rule='z_plus', **opts)
    finally:
        V2.SIGNED_INPUT_ZPLUS = True
        V2.ZPLUS_WEIGHT_FLOOR = 0.0
    return R, seed, total


def subsets():
    rce_idx, map_idx, fid_idx = P.sample_indices(N_RCE), P.sample_indices(N_MAP), P.sample_indices(N_FID)
    return sorted(set(rce_idx) | set(map_idx) | set(fid_idx)), set(rce_idx), map_idx, fid_idx


def agreement(maps, model_name):
    path = os.path.join(P.CACHE_DIR, 'E1_maps', f'Zennit_canonized__{model_name}__z+.npy')
    if not os.path.exists(path):
        return np.nan, np.nan
    ref = np.load(path).astype(np.float64)
    pr, sr = [], []
    for a, b in zip(maps, ref):
        a, b = np.asarray(a, np.float64).ravel(), b.ravel()
        if np.all(np.isfinite(a)) and a.std() > 0:
            pr.append(stats.pearsonr(a, b)[0])
            sr.append(stats.spearmanr(a, b)[0])
    return (np.median(pr), np.median(sr)) if pr else (np.nan, np.nan)


def finalise(model, model_name, vname, tag, rce_set, map_idx, fid_idx):
    rows = pd.read_csv(os.path.join(PART_DIR, f'{tag}.csv'))
    maps = {}
    for f in sorted(os.listdir(PART_DIR)):
        if f.startswith(f'{tag}__maps_'):
            with np.load(os.path.join(PART_DIR, f)) as z:
                maps.update({int(k[1:]): z[k] for k in z.files})
    pear, spear = agreement([maps[i] for i in map_idx], model_name)
    fid = dict(variant=vname, model=model_name, pearson_vs_zennit=pear, spearman_vs_zennit=spear)
    x_fid = np.concatenate([P.model_input(model_name, i) for i in fid_idx])
    logits = tf.keras.Model(model.inputs, model.get_layer('predictions').output)(x_fid).numpy()
    y_fid = tf.keras.utils.to_categorical(logits.argmax(1), 1000)
    expl = np.nan_to_num(np.stack([maps[i][..., None] for i in fid_idx]).astype(np.float32))
    fid['deletion_auc'] = float(Deletion(model, x_fid, y_fid, batch_size=16)(expl))
    fid['insertion_auc'] = float(Insertion(model, x_fid, y_fid, batch_size=16)(expl))

    rows = rows[rows['index'].isin(rce_set)]
    per_image = os.path.join(OUT_DIR, 'ablation_per_image.csv')
    rows.to_csv(per_image, mode='a', header=not os.path.exists(per_image), index=False)
    fid_csv = os.path.join(OUT_DIR, 'ablation_fidelity.csv')
    pd.DataFrame([fid]).to_csv(fid_csv, mode='a', header=not os.path.exists(fid_csv), index=False)
    for f in os.listdir(PART_DIR):
        if f.startswith(tag):
            os.remove(os.path.join(PART_DIR, f))
    ok = rows[rows.status == 'OK'].RCE_pct
    print(f"{model_name:13s} {vname:18s} median RCE={ok.median():.3e}%  r_vs_Zennit={pear:.3f}  "
          f"del={fid['deletion_auc']:.4f} ins={fid['insertion_auc']:.4f}", flush=True)


def summarise():
    df = pd.read_csv(os.path.join(OUT_DIR, 'ablation_per_image.csv'))
    fid = pd.read_csv(os.path.join(OUT_DIR, 'ablation_fidelity.csv'))
    summ = (df[df.status == 'OK'].groupby(['model', 'variant']).RCE_pct
            .agg(RCE_median='median', RCE_q25=lambda s: s.quantile(.25), RCE_q75=lambda s: s.quantile(.75))
            .reset_index().merge(fid, on=['model', 'variant']))
    summ.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False)
    print(summ.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunk', type=int, default=150)
    ap.add_argument('--threads', type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    os.makedirs(PART_DIR, exist_ok=True)
    names = P.image_list()
    order, rce_set, map_idx, fid_idx = subsets()
    fid_path = os.path.join(OUT_DIR, 'ablation_fidelity.csv')
    done = set(zip(*[pd.read_csv(fid_path)[c] for c in ('model', 'variant')])) if os.path.exists(fid_path) else set()

    for model_name, ctor in CTORS.items():
        pending = [v for v in VARIANTS if (model_name, v) not in done]
        if not pending:
            continue
        model = ctor(weights='imagenet')
        vname = pending[0]
        tag = f"{model_name}__{vname.split()[0]}"
        part_csv = os.path.join(PART_DIR, f'{tag}.csv')
        start = len(pd.read_csv(part_csv)) if os.path.exists(part_csv) else 0
        rows, maps = [], {}
        keep = set(map_idx) | set(fid_idx)
        for i in order[start:start + args.chunk]:
            t0 = time.time()
            try:
                R, seed, total = relevance(model, P.model_input(model_name, i), VARIANTS[vname])
                row = dict(status='OK', seed=seed, sum_R_in=total, RCE_pct=abs(total - seed) / abs(seed) * 100)
                if i in keep:
                    maps[f'i{i:05d}'] = R[0].sum(-1)
            except Exception as e:  # noqa: BLE001 - recorded
                row = dict(status='CRASH', error=f'{type(e).__name__}: {str(e)[:200]}')
            rows.append(dict(row, variant=vname, model=model_name, index=i, image=names[i], time_s=time.time() - t0))
        pd.DataFrame(rows).to_csv(part_csv, mode='a', header=not os.path.exists(part_csv), index=False)
        if maps:
            np.savez(os.path.join(PART_DIR, f'{tag}__maps_{start:05d}.npz'), **maps)
        n_done = start + len(rows)
        print(f'{model_name:13s} {vname:18s} chunk {start}-{n_done - 1} ({n_done}/{len(order)})', flush=True)
        if n_done >= len(order):
            finalise(model, model_name, vname, tag, rce_set, map_idx, fid_idx)
        return
    summarise()
    print('ALL BLOCKS DONE', flush=True)


if __name__ == '__main__':
    main()
