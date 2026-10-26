###############################################################################
# Execution Environment: any (Python 3.7 / 3.12) -- numpy, pandas
###############################################################################
"""E1 shared settings: rules, sample sizes, output paths and the resumable CSV writer."""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'exp_result', 'E1')
MODELS = ['vgg16', 'resnet50', 'inception_v3', 'mobilenet_v2']
RULES = ['z+', 'epsilon', 'alpha_beta', 'composite']   # all ignore biases
EPSILON = 0.01
ALPHA, BETA = 2.0, 1.0
N_MAPS = 50          # relevance maps kept for cross-implementation similarity
_MAP_SET = None


def map_set():
    """Cache indices whose maps are stored: a seed-42 random subset (the cache is class-ordered)."""
    global _MAP_SET
    if _MAP_SET is None:
        from preprocess import sample_indices
        idx = sample_indices(N_MAPS)
        np.save(os.path.join(maps_dir(), 'map_indices.npy'), np.array(idx))
        _MAP_SET = set(idx)
    return _MAP_SET


def maps_dir():
    from preprocess import CACHE_DIR
    d = os.path.join(CACHE_DIR, 'E1_maps')
    if not os.path.isdir(d):
        os.makedirs(d)
    return d


def done_keys(csv_path):
    """(implementation, model, rule) combinations already completed in csv_path."""
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    return set(zip(df.implementation, df.model, df.rule))


def append_rows(csv_path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)


def save_maps(impl, model, rule, maps):
    safe = impl.replace(' ', '_').replace('(', '').replace(')', '')
    np.save(os.path.join(maps_dir(), '%s__%s__%s.npy' % (safe, model, rule)),
            np.asarray(maps, dtype=np.float16))


def rce(total, seed):
    return abs(total - seed) / (abs(seed) + 1e-12) * 100
