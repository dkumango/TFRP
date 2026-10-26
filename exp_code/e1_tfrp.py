###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- tensorflow 2.20, keras 3.14
###############################################################################
"""E1. Identical-weight comparison -- TFRP v2 part.

keras.applications models (the weights exported for the PyTorch rebuild), the shared image
cache, four rules (all ignoring biases), seed = bias-free target logit.

Output: exp_result/E1/tfrp.csv (resumable), relevance maps of the first 50 images in the cache
Usage : python exp_code/e1_tfrp.py [--n 1000] [--threads 3]
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('exp_code', 'exp_code/common', 'exp_code/tfrp'):
    sys.path.insert(0, os.path.join(ROOT, p))
import e1_common as C
import preprocess as P
import general_TFRP_v2 as V2

A = tf.keras.applications
CTORS = {'vgg16': A.VGG16, 'resnet50': A.ResNet50, 'inception_v3': A.InceptionV3, 'mobilenet_v2': A.MobileNetV2}
RULE_ARGS = {
    'z+': dict(global_rule='z_plus'),
    'epsilon': dict(global_rule='epsilon', epsilon=C.EPSILON),
    'alpha_beta': dict(global_rule='alpha_beta', alpha=C.ALPHA, beta=C.BETA),
    'composite': dict(global_rule='z_plus', alpha=C.ALPHA, beta=C.BETA, epsilon=C.EPSILON,
                      composite_preset={'Conv2D': 'alpha_beta', 'DepthwiseConv2D': 'alpha_beta',
                                        'Dense': 'epsilon'}),
}
IMPL = 'TFRP v2'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--one_block', action='store_true',
                    help='process at most --chunk images of the first unfinished (model, rule) block and exit. '
                         "TensorFlow's CPU allocator keeps growing over thousands of propagations (several GB per "
                         '1000 images on ResNet50), so run_e1.sh restarts the process after every chunk')
    ap.add_argument('--chunk', type=int, default=200)
    args = ap.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    os.makedirs(C.OUT_DIR, exist_ok=True)
    csv = os.path.join(C.OUT_DIR, 'tfrp.csv')
    done = C.done_keys(csv)
    images = P.image_list()[:args.n]
    part_dir = os.path.join(C.OUT_DIR, 'partial')

    for model_name in C.MODELS:
        pending = [r for r in C.RULES if (IMPL, model_name, r) not in done]
        if not pending:
            continue
        model = CTORS[model_name](weights='imagenet')
        for rule in pending:
            if args.one_block:
                run_chunk(model, model_name, rule, images, csv, part_dir, args.chunk)
                return
            rows, maps = [], []
            for i, fname in enumerate(images):
                x = P.model_input(model_name, i)
                t0 = time.time()
                try:
                    R, seed, total, _ = V2.get_relevance_map_generalized(model, x, use_logit=True, **RULE_ARGS[rule])
                    row = dict(status='OK', seed=seed, sum_R_in=total, RCE_pct=C.rce(total, seed))
                    if i in C.map_set():
                        maps.append(R[0].sum(-1))
                except Exception as e:  # noqa: BLE001 - recorded, run continues
                    row = dict(status='CRASH', error=f'{type(e).__name__}: {str(e)[:200]}')
                row.update(implementation=IMPL, model=model_name, rule=rule, image=fname,
                           seed_kind='bias_free', time_s=time.time() - t0)
                rows.append(row)
            C.append_rows(csv, rows)
            if maps:
                C.save_maps(IMPL, model_name, rule, maps)
            ok = [r['RCE_pct'] for r in rows if r['status'] == 'OK']
            print(f'{model_name:13s} {rule:10s} OK {len(ok)}/{len(rows)} median RCE={np.median(ok) if ok else float("nan"):.3e}%',
                  flush=True)
        tf.keras.backend.clear_session()
    print('ALL BLOCKS DONE', flush=True)


def run_chunk(model, model_name, rule, images, csv, part_dir, chunk):
    """Process the next `chunk` images of one block; finalise the block when it is complete."""
    os.makedirs(part_dir, exist_ok=True)
    tag = f'{model_name}__{rule}'
    part_csv = os.path.join(part_dir, f'tfrp__{tag}.csv')
    start = len(pd.read_csv(part_csv)) if os.path.exists(part_csv) else 0
    rows, maps = [], {}
    for i in range(start, min(start + chunk, len(images))):
        x = P.model_input(model_name, i)
        t0 = time.time()
        try:
            R, seed, total, _ = V2.get_relevance_map_generalized(model, x, use_logit=True, **RULE_ARGS[rule])
            row = dict(status='OK', seed=seed, sum_R_in=total, RCE_pct=C.rce(total, seed))
            if i in C.map_set():
                maps[f'i{i:05d}'] = R[0].sum(-1)
        except Exception as e:  # noqa: BLE001 - recorded, run continues
            row = dict(status='CRASH', error=f'{type(e).__name__}: {str(e)[:200]}')
        row.update(implementation=IMPL, model=model_name, rule=rule, image=images[i],
                   seed_kind='bias_free', time_s=time.time() - t0)
        rows.append(row)
    C.append_rows(part_csv, rows)
    if maps:
        np.savez(os.path.join(part_dir, f'tfrp__{tag}__maps_{start:05d}.npz'), **maps)
    n_done = start + len(rows)
    print(f'{model_name:13s} {rule:10s} chunk {start}-{n_done - 1} written ({n_done}/{len(images)})', flush=True)
    if n_done < len(images):
        return
    # block complete: move rows to the main CSV and merge the maps in index order
    block = pd.read_csv(part_csv)
    C.append_rows(csv, block.to_dict('records'))
    collected = {}
    for f in sorted(os.listdir(part_dir)):
        if f.startswith(f'tfrp__{tag}__maps_'):
            with np.load(os.path.join(part_dir, f)) as z:
                collected.update({k: z[k] for k in z.files})
            os.remove(os.path.join(part_dir, f))
    if collected:
        C.save_maps(IMPL, model_name, rule, [collected[k] for k in sorted(collected)])
    os.remove(part_csv)
    ok = block[block.status == 'OK'].RCE_pct
    print(f'{model_name:13s} {rule:10s} OK {len(ok)}/{len(block)} median RCE={ok.median():.3e}%', flush=True)


if __name__ == '__main__':
    main()
