###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- tensorflow 2.20, keras 3.14, xplique 2.0.1, scikit-image
###############################################################################
"""E7. Quantitative model-parameter randomisation test (Adebayo et al., 2018), ResNet50.

Cascading randomisation from the top: the kernels of all Conv2D/Dense layers of a stage are
re-drawn (Glorot normal, fixed seed) and their biases set to zero, cumulatively
  predictions -> conv5 -> conv4 -> conv3 -> conv2 -> conv1.
After each stage the maps of every method are recomputed for the same images and the same target
(the original top-1 class) and compared with the maps of the trained model:
  spearman_abs  Spearman rank correlation of |map|
  spearman      Spearman rank correlation of the signed map
  ssim          SSIM of |map| normalised to [0, 1]
A method that passes the test shows similarity that drops quickly as more layers are randomised.

Output: exp_result/E7/sanity_per_image.csv
Usage : python exp_code/e7_sanity.py [--n 50]
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
from scipy import stats
from skimage.metrics import structural_similarity
from xplique import attributions as XA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('exp_code', 'exp_code/common', 'exp_code/tfrp'):
    sys.path.insert(0, os.path.join(ROOT, p))
import preprocess as P
import general_TFRP_v2 as V2
from e4_common import to_2d

STAGES = ['predictions', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1']
COMPOSITE = {'Conv2D': 'alpha_beta', 'Dense': 'epsilon'}
OUT_DIR = os.path.join(ROOT, 'exp_result', 'E7')


def tfrp(model, x, t, **kw):
    R, _, _, _ = V2.get_relevance_map_generalized(model, x[None], target_class_idx=int(t), use_logit=True, **kw)
    return to_2d(R[0])


def maps_for(model, xs, ts):
    y = tf.keras.utils.to_categorical(ts, 1000)
    xpl = {
        'Saliency': XA.Saliency(model, batch_size=16),
        'GuidedBackprop': XA.GuidedBackprop(model, batch_size=16),
        'IntegratedGradients': XA.IntegratedGradients(model, batch_size=16, steps=50),
        'SmoothGrad': XA.SmoothGrad(model, batch_size=16, nb_samples=20, noise=0.2),
        'GradCAM': XA.GradCAM(model, batch_size=16),
    }
    # 'TFRP z+' / 'TFRP composite' use the uniform split at Add and average pooling (the TFRP default
    # when E7 was run); 'TFRP proportional (V3)' is the default since 2026-10-23.
    uniform = dict(add_rule='uniform', avgpool_rule='gradient')
    out = {
        'TFRP z+': np.stack([tfrp(model, x, t, **uniform) for x, t in zip(xs, ts)]),
        'TFRP composite': np.stack([tfrp(model, x, t, global_rule='z_plus', composite_preset=COMPOSITE, epsilon=0.01,
                                         **uniform) for x, t in zip(xs, ts)]),
        'TFRP proportional (V3)': np.stack([tfrp(model, x, t, add_rule='proportional', avgpool_rule='proportional')
                                            for x, t in zip(xs, ts)]),
    }
    for name, ex in xpl.items():
        out[name] = np.stack([to_2d(m) for m in np.asarray(ex(xs, y))])
    return out


def randomise(model, stage, rng):
    for layer in model.layers:
        if layer.name.startswith(stage) and layer.__class__.__name__ in ('Conv2D', 'Dense'):
            k = layer.kernel
            fan_in, fan_out = np.prod(k.shape[:-1]), k.shape[-1]
            layer.kernel.assign(rng.normal(0, np.sqrt(2.0 / (fan_in + fan_out)), k.shape).astype(np.float32))
            if layer.use_bias:
                layer.bias.assign(np.zeros(layer.bias.shape, np.float32))


def similarity(a, b):
    a, b = np.nan_to_num(a).ravel(), np.nan_to_num(b).ravel()
    aa, bb = np.abs(a), np.abs(b)
    na, nb = aa / (aa.max() + 1e-12), bb / (bb.max() + 1e-12)
    side = int(np.sqrt(a.size))
    return dict(spearman_abs=stats.spearmanr(aa, bb)[0], spearman=stats.spearmanr(a, b)[0],
                ssim=structural_similarity(na.reshape(side, side), nb.reshape(side, side), data_range=1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--threads', type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    os.makedirs(OUT_DIR, exist_ok=True)
    idx = P.sample_indices(args.n, seed=21)
    names = P.image_list()
    xs = np.concatenate([P.model_input('resnet50', i) for i in idx])
    model = tf.keras.applications.ResNet50(weights='imagenet', classifier_activation=None)
    ts = model.predict(xs, batch_size=16, verbose=0).argmax(1)

    original = maps_for(model, xs, ts)
    rows = []
    rng = np.random.RandomState(2018)
    for depth, stage in enumerate(STAGES, start=1):
        randomise(model, stage, rng)
        current = maps_for(model, xs, ts)
        for method, maps in current.items():
            for j, i in enumerate(idx):
                rows.append(dict(method=method, stage=stage, depth=depth, image=names[i],
                                 **similarity(original[method][j], maps[j])))
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT_DIR, 'sanity_per_image.csv'), index=False)
        print(f'--- randomised through {stage}')
        print(df[df.stage == stage].groupby('method')[['spearman_abs', 'ssim']].median().round(3).to_string(), flush=True)


if __name__ == '__main__':
    main()
