###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- tensorflow 2.20, keras 3.14, matplotlib
###############################################################################
"""E12. Manuscript figures with raw relevance maps (identical normalisation, no post-processing).

fig_validation  (Section 5.1): ResNet50 and InceptionV3, two images each; Input | TFRP | Zennit
                (canonized) | iNNvestigate; Pearson r with the Zennit map in each panel. The maps
                come from E1 (identical weights, z+).
fig_methods     (Section 5.9): ResNet50, four images; Input | TFRP z+ | TFRP composite | Integrated
                Gradients | SmoothGrad | Guided Backprop | Grad-CAM | RISE. Reference-method maps come
                from E4; TFRP maps are computed here with the current defaults.
All maps: channel-summed and scaled by their own 99.5th percentile of |R|. Signed maps use a symmetric
blue-white-red scale around 0; non-negative maps (z+ LRP, Grad-CAM, RISE) are shifted by their
minimum and shown white to red. No masking, log scaling, smoothing or dilation.

Output: exp_result/E12/fig_validation.png|.pdf, fig_methods.png|.pdf
Usage : python exp_code/e12_figures.py
"""

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
for p in ('exp_code', 'exp_code/common', 'exp_code/tfrp'):
    sys.path.insert(0, os.path.join(ROOT, p))
import preprocess as P
import e4_common as E
import general_TFRP_v2 as V2

OUT = os.path.join(ROOT, 'exp_result', 'E12')
E1_MAPS = os.path.join(P.CACHE_DIR, 'E1_maps')
COMPOSITE = {'Conv2D': 'alpha_beta', 'DepthwiseConv2D': 'alpha_beta', 'Dense': 'epsilon'}
LABEL = lambda name: name.split('_', 2)[-1][:-4].replace('_', ' ')


def norm(m):
    """Non-negative maps (z+ LRP, Grad-CAM, RISE): shift by the minimum (removes constant offsets such as
    RISE's mean confidence) and scale to [0, 1] -> white to red. Signed maps: scale symmetrically
    around 0 -> blue-white-red. Both use the map's own 99.5th percentile, so no colour means "negative"
    unless the map has negative values."""
    m = np.nan_to_num(np.asarray(m, np.float64))
    if m.min() >= 0:
        m = m - m.min()
    s = np.percentile(np.abs(m), 99.5)
    return np.clip(m / (s + 1e-12), -1, 1)


def pearson(a, b):
    a, b = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def panel(ax, m, title=None, note=None):
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(norm(m), cmap='bwr', vmin=-1, vmax=1)
    if title:
        ax.set_title(title, fontsize=9)
    if note:
        ax.text(0.03, 0.04, note, transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.75, pad=1, lw=0))


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=200 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)
    print('saved', name, flush=True)


def fig_validation(images):
    idx = list(np.load(os.path.join(E1_MAPS, 'map_indices.npy')))
    names = P.image_list()
    rows = [(m, i) for m in ('resnet50', 'inception_v3') for i in images]
    fig, axes = plt.subplots(len(rows), 4, figsize=(9.2, 2.35 * len(rows)))
    for r, (model, i) in enumerate(rows):
        pos = idx.index(i)
        maps = {k: np.load(os.path.join(E1_MAPS, f'{k}__{model}__z+.npy'))[pos]
                for k in ('TFRP_v2', 'Zennit_canonized', 'iNNvestigate')}
        size, _ = P.MODEL_INFO[model]
        ax = axes[r, 0]
        ax.imshow(np.asarray(P.rgb_cache(size)[i])); ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"{'ResNet50' if model == 'resnet50' else 'InceptionV3'}\n{LABEL(names[i])}", fontsize=9)
        if r == 0:
            ax.set_title('Input', fontsize=9)
        ref = maps['Zennit_canonized']
        panel(axes[r, 1], maps['TFRP_v2'], 'TFRP' if r == 0 else None, f"r = {pearson(maps['TFRP_v2'], ref):.3f}")
        panel(axes[r, 2], ref, 'Zennit (canonized)' if r == 0 else None)
        panel(axes[r, 3], maps['iNNvestigate'], 'iNNvestigate' if r == 0 else None,
              f"r = {pearson(maps['iNNvestigate'], ref):.2f}")
    fig.tight_layout()
    save(fig, 'fig_validation')


def fig_methods(images):
    ev = list(E.eval_indices())
    names = P.image_list()
    model = tf.keras.applications.ResNet50(weights='imagenet')
    with np.load(os.path.join(E.maps_dir(), 'targets.npz')) as z:
        targets = z['eval']
    cols = [('TFRP z⁺', None), ('TFRP composite', None), ('Integrated Gradients', 'IntegratedGradients'),
            ('SmoothGrad', 'SmoothGrad'), ('Guided Backprop', 'GuidedBackprop'), ('Grad-CAM', 'GradCAM'),
            ('RISE', 'RISE')]
    loaded = {f: np.load(E.map_path(f, 'eval'), mmap_mode='r') for _, f in cols if f}
    fig, axes = plt.subplots(len(images), len(cols) + 1, figsize=(2.05 * (len(cols) + 1), 2.15 * len(images)))
    for r, i in enumerate(images):
        j = ev.index(i)
        x, t = P.model_input('resnet50', i), int(targets[j])
        ax = axes[r, 0]
        ax.imshow(np.asarray(P.rgb_cache(224)[i])); ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(LABEL(names[i]), fontsize=9)
        if r == 0:
            ax.set_title('Input', fontsize=9)
        for c, (title, f) in enumerate(cols, start=1):
            if title == 'TFRP z⁺':
                m = V2.get_relevance_map_generalized(model, x, target_class_idx=t)[0][0].sum(-1)
            elif title == 'TFRP composite':
                m = V2.get_relevance_map_generalized(model, x, target_class_idx=t, global_rule='z_plus',
                                                     composite_preset=COMPOSITE, epsilon=0.01)[0][0].sum(-1)
            else:
                m = np.asarray(loaded[f][j], np.float32)
            panel(axes[r, c], m, title if r == 0 else None)
    fig.tight_layout()
    save(fig, 'fig_methods')


def main():
    os.makedirs(OUT, exist_ok=True)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    fig_validation(images=[59, 312])            # koala bear, bicycle
    fig_methods(images=[59, 101, 312, 761])     # koala bear, domestic cat, bicycle, soccer ball


if __name__ == '__main__':
    main()
