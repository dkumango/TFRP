###############################################################################
# Execution Environment: any (Python 3.7 / 3.12) -- numpy
###############################################################################
"""E4 shared settings: image subsets, perturbations for sensitivity and map storage.

All attribution maps are channel-summed (H, W) float16 arrays computed on ResNet50 with the
keras.applications ImageNet weights (PyTorch implementations use the identical-weight rebuild).
Perturbations are generated from (image index, draw) seeds with numpy's legacy RandomState, so
every environment produces bit-identical noisy inputs.
"""

import os

import numpy as np

MODEL = 'resnet50'
N_EVAL = 500          # Deletion, Insertion, AOPC, MuFidelity, Infidelity
N_SENS = 100          # Max-sensitivity
SENS_DRAWS = 5
SENS_RADIUS = 0.02 * 255.0   # L_inf radius in the caffe input scale (0.02 of the pixel range)


def eval_indices():
    from preprocess import sample_indices
    return sample_indices(N_EVAL, seed=42)


def sens_indices():
    from preprocess import sample_indices
    return sample_indices(N_SENS, seed=7)


def perturbation(index, draw, shape):
    rng = np.random.RandomState(100000 + 10 * int(index) + int(draw))
    return rng.uniform(-SENS_RADIUS, SENS_RADIUS, size=shape).astype(np.float32)


def maps_dir():
    from preprocess import CACHE_DIR
    d = os.path.join(CACHE_DIR, 'E4_maps')
    if not os.path.isdir(d):
        os.makedirs(d)
    return d


def safe(name):
    return name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus').replace('/', '_')


def map_path(method, kind):
    """kind: 'eval' (N_EVAL, H, W), 'sensbase' (N_SENS, H, W), 'sens' (N_SENS, DRAWS, H, W)."""
    return os.path.join(maps_dir(), '%s__%s.npy' % (safe(method), kind))


def to_2d(m):
    """Channel-sum an attribution to (H, W)."""
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 3:
        m = m.sum(-1)
    return m


def save(method, kind, maps):
    np.save(map_path(method, kind), np.asarray(maps, dtype=np.float16))


def exists(method, kind):
    return os.path.exists(map_path(method, kind))
