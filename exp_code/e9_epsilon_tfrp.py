###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- tensorflow 2.20, keras 3.14
###############################################################################
"""E9. epsilon-rule sweep -- TFRP part (identical weights; Zennit part: e9_epsilon_zennit.py).

For each epsilon the LRP-epsilon rule (biases ignored) is applied to every Dense/Conv layer.
Reports the RCE distribution, so that non-conservation and divergence can be attributed to the
rule and its parameter rather than to the implementation.

Output: exp_result/E9/tfrp.csv
Usage : python exp_code/e9_epsilon_tfrp.py [--n 100]
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
import preprocess as P
import general_TFRP_v2 as V2

A = tf.keras.applications
CTORS = {'vgg16': A.VGG16, 'resnet50': A.ResNet50, 'inception_v3': A.InceptionV3, 'mobilenet_v2': A.MobileNetV2}
EPSILONS = [1e-9, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]
OUT_DIR = os.path.join(ROOT, 'exp_result', 'E9')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--threads', type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    os.makedirs(OUT_DIR, exist_ok=True)
    names = P.image_list()
    indices = P.sample_indices(args.n)
    rows = []
    for model_name, ctor in CTORS.items():
        model = ctor(weights='imagenet')
        for eps in EPSILONS:
            for i in indices:
                fname = names[i]
                t0 = time.time()
                _, seed, total, _ = V2.get_relevance_map_generalized(
                    model, P.model_input(model_name, i), use_logit=True, global_rule='epsilon', epsilon=eps)
                rows.append(dict(implementation='TFRP v2', model=model_name, epsilon=eps, image=fname, seed=seed,
                                 sum_R_in=total, RCE_pct=abs(total - seed) / abs(seed) * 100, time_s=time.time() - t0))
            r = [x['RCE_pct'] for x in rows if x['model'] == model_name and x['epsilon'] == eps]
            print(f'{model_name:13s} eps={eps:7.0e} median RCE={np.nanmedian(r):.3e}%  '
                  f'>100%: {np.mean(np.array(r) > 100) * 100:.0f}%', flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'tfrp.csv'), index=False)
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    main()
