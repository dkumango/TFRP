###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- tensorflow 2.20, keras 3.14
###############################################################################
"""E2 + E8. Where TFRP loses relevance, and how much of it is bias absorption.

For every keras.applications model of E13 and the z+ rule, each propagation step is logged.
At every weighted layer the relevance that cannot be passed down is split into
  bias_absorbed : relevance sitting on output units whose positive input contribution z+ is 0
                  (the unit is active only through its bias / folded BatchNorm shift)
  other         : the remainder (stabiliser effects, numerical error)
Losses at Add (uniform split onto zero-activation branches) and at the other operation types
are recorded separately. The decomposition is checked against the measured total leakage.

Output: exp_result/E2/layer_profile.csv (per image x layer), leak_decomposition.csv (per model)
Usage : python exp_code/e2_layer_profile.py [--n 100]
"""

import argparse
import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('exp_code', 'exp_code/common', 'exp_code/tfrp'):
    sys.path.insert(0, os.path.join(ROOT, p))
import preprocess as P
import general_TFRP_v2 as V2

A = tf.keras.applications
MODELS = {
    'vgg16': (A.VGG16, A.vgg16.preprocess_input, 224),
    'resnet50': (A.ResNet50, A.resnet50.preprocess_input, 224),
    'inception_v3': (A.InceptionV3, A.inception_v3.preprocess_input, 299),
    'mobilenet_v2': (A.MobileNetV2, A.mobilenet_v2.preprocess_input, 224),
    'densenet121': (A.DenseNet121, A.densenet.preprocess_input, 224),
    'efficientnet_b0': (A.EfficientNetB0, A.efficientnet.preprocess_input, 224),
    'convnext_tiny': (A.ConvNeXtTiny, A.convnext.preprocess_input, 224),
}
OUT_DIR = os.path.join(ROOT, 'exp_result', 'E2')


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

    absorbed = {}
    original = V2.distribute_relevance_gen

    def spy(layer, p_i, R_j, rule='z_plus', **kw):
        out = original(layer, p_i, R_j, rule=rule, **kw)
        f, w = V2._forward_op(layer), V2._kernel(layer)
        z = f(tf.maximum(p_i, 0.), tf.maximum(w, 0.)) + f(tf.minimum(p_i, 0.), tf.minimum(w, 0.))
        # exactly zero positive contribution: R_j / (0 + stabiliser) multiplies only zero terms -> lost
        absorbed[layer.name] = float(tf.reduce_sum(R_j * tf.cast(z <= 0.0, tf.float32)))
        return out

    V2.distribute_relevance_gen = spy
    rows, summary = [], []
    for mname, (ctor, prep, size) in MODELS.items():
        model = ctor(weights='imagenet')
        cache = P.rgb_cache(size)
        for i in indices:
            x = prep(np.asarray(cache[i], dtype=np.float32).copy())[None]
            absorbed.clear()
            _, seed, total, _, log = V2.get_relevance_map_generalized(model, x, use_logit=True,
                                                                      global_rule='z_plus', return_layer_log=True)
            for lname, ltype, upper, lower in log:
                loss = upper - lower
                rows.append(dict(model=mname, image=names[i], layer=lname, type=ltype,
                                 loss_pct=loss / seed * 100,
                                 bias_absorbed_pct=absorbed.get(lname, 0.0) / seed * 100))
            img = [r for r in rows if r['model'] == mname and r['image'] == names[i]]
            summary.append(dict(model=mname, image=names[i],
                                leak_pct=(seed - total) / seed * 100,
                                sum_layer_loss_pct=sum(r['loss_pct'] for r in img),
                                weighted_loss_pct=sum(r['loss_pct'] for r in img if r['type'] in V2.WEIGHTED_LAYERS),
                                bias_absorbed_pct=sum(r['bias_absorbed_pct'] for r in img),
                                add_loss_pct=sum(r['loss_pct'] for r in img if r['type'] == 'Add')))
        s = pd.DataFrame([r for r in summary if r['model'] == mname])
        print(f'{mname:16s} leak={s.leak_pct.median():.3e}%  weighted-layer loss={s.weighted_loss_pct.median():.3e}%  '
              f'on z+=0 units={s.bias_absorbed_pct.median():.3e}%  Add={s.add_loss_pct.median():.3e}%', flush=True)
        tf.keras.backend.clear_session()
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'layer_profile.csv'), index=False)
        pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, 'leak_decomposition.csv'), index=False)


if __name__ == '__main__':
    main()
