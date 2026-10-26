###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- tensorflow 2.20, keras 3.14, matplotlib
###############################################################################
"""E12a. Raw relevance maps of TFRP (revised default), canonized Zennit, earlier TFRP versions and
iNNvestigate on identical networks, for a visual comparison (z+ rule).

Columns: input | TFRP (proportional, default) | Zennit (canonized) | TFRP (uniform split, earlier
default) | TFRP v1 (submitted) | iNNvestigate. Maps are channel-summed raw relevance, each normalised
symmetrically by its own 99.5th percentile of |R| (red positive, blue negative); no masking, smoothing
or dilation. The number in each panel is the Pearson correlation with the Zennit map.
Zennit, iNNvestigate and uniform-TFRP maps come from E1 (same images, identical weights); the
current TFRP and v1 maps are computed here (and the current ones checked against E1).

Output: exp_result/E12/compare_tfrp_zennit_<model>.png
Usage : python exp_code/e12_compare_tfrp_zennit.py [--positions 0,3,13,36]
"""

import argparse
import contextlib
import io
import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('exp_code/common', 'exp_code/tfrp', 'exp_code/tfrp'):
    sys.path.insert(0, os.path.join(ROOT, p))
import preprocess as P
import general_TFRP_v2 as V2
import general_TFRP_v1 as V1

A = tf.keras.applications
CTORS = {'vgg16': A.VGG16, 'resnet50': A.ResNet50, 'inception_v3': A.InceptionV3, 'mobilenet_v2': A.MobileNetV2}
MAPS = os.path.join(P.CACHE_DIR, 'E1_maps')
OUT = os.path.join(ROOT, 'exp_result', 'E12')


def norm(m):
    m = np.nan_to_num(np.asarray(m, np.float64))
    s = np.percentile(np.abs(m), 99.5)
    return np.clip(m / (s + 1e-12), -1, 1)


def pearson(a, b):
    a, b = np.nan_to_num(np.ravel(a).astype(np.float64)), np.nan_to_num(np.ravel(b).astype(np.float64))
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--positions', default='0,3,13,36', help='positions in the E1 map subset')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    idx_all = np.load(os.path.join(MAPS, 'map_indices.npy'))
    pos = [int(p) for p in args.positions.split(',')]
    names = P.image_list()

    for model_name, ctor in CTORS.items():
        model = ctor(weights='imagenet')
        size, _ = P.MODEL_INFO[model_name]
        zen = np.load(os.path.join(MAPS, f'Zennit_canonized__{model_name}__z+.npy'))
        inn = np.load(os.path.join(MAPS, f'iNNvestigate__{model_name}__z+.npy'))
        uni = np.load(os.path.join(MAPS, 'archive_uniform', f'TFRP_v2__{model_name}__z+.npy'))
        cur_e1 = os.path.join(MAPS, f'TFRP_v2__{model_name}__z+.npy')
        cur_e1 = np.load(cur_e1) if os.path.exists(cur_e1) else None
        cols = ['Input', 'TFRP (revised)', 'Zennit (canonized)', 'TFRP (uniform split)', 'TFRP v1 (submitted)',
                'iNNvestigate']
        fig, axes = plt.subplots(len(pos), len(cols), figsize=(2.3 * len(cols), 2.35 * len(pos)))
        for r, p in enumerate(pos):
            i = int(idx_all[p])
            x = P.model_input(model_name, i)
            cur = V2.get_relevance_map_generalized(model, x)[0][0].sum(-1)
            if cur_e1 is not None:
                print(f'{model_name} {names[i]}: on-the-fly vs E1 map r = {pearson(cur, cur_e1[p]):.6f}')
            V1.SIGNED_INPUT_ZPLUS = False
            with contextlib.redirect_stdout(io.StringIO()):
                v1 = V1.get_relevance_map_generalized(model, x, use_logit=True, global_rule='z_plus')[0][0].sum(-1)
            V1.SIGNED_INPUT_ZPLUS = True
            panels = [None, cur, zen[p], uni[p], v1, inn[p]]
            for c, (title, m) in enumerate(zip(cols, panels)):
                ax = axes[r, c]
                ax.set_xticks([]); ax.set_yticks([])
                if m is None:
                    ax.imshow(np.asarray(P.rgb_cache(size)[i]))
                    ax.set_ylabel(names[i].split('_', 2)[-1][:-4].replace('_', ' '), fontsize=9)
                else:
                    ax.imshow(norm(m), cmap='bwr', vmin=-1, vmax=1)
                    if c != 2:
                        ax.text(0.03, 0.04, f'r={pearson(m, zen[p]):.2f}', transform=ax.transAxes, fontsize=8,
                                color='black', bbox=dict(facecolor='white', alpha=0.7, pad=1, lw=0))
                if r == 0:
                    ax.set_title(title, fontsize=9)
        fig.suptitle(f'{model_name}: raw z+ relevance, identical weights (r = Pearson correlation with Zennit)',
                     fontsize=10)
        fig.tight_layout()
        path = os.path.join(OUT, f'compare_tfrp_zennit_{model_name}.png')
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print('saved', path, flush=True)
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    main()
