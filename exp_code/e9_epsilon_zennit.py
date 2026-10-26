###############################################################################
# Execution Environment: Python 3.12 (environment B, see README)  -- torch (CPU), zennit 1.0.0
###############################################################################
"""E9. epsilon-rule sweep -- Zennit part (canonized; identical weights via keras_graph_torch).

Same epsilons, images and seed (bias-free logit) as e9_epsilon_tfrp.py.
Output: exp_result/E9/zennit.csv
Usage : python exp_code/e9_epsilon_zennit.py [--n 100]
"""

import argparse
import os
import sys
import time
import warnings

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from zennit.attribution import Gradient
from zennit.composites import LayerMapComposite, layer_map_base
from zennit.rules import Epsilon
from zennit.types import Convolution, Linear

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('exp_code', 'exp_code/common'):
    sys.path.insert(0, os.path.join(ROOT, p))
import preprocess as P
import keras_graph_torch as K

# same values as e9_epsilon_tfrp.py (not imported: that module needs TensorFlow)
EPSILONS = [1e-9, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]
OUT_DIR = os.path.join(ROOT, 'exp_result', 'E9')

MODELS = ['vgg16', 'resnet50', 'inception_v3', 'mobilenet_v2']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--threads', type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs(OUT_DIR, exist_ok=True)
    names = P.image_list()
    indices = P.sample_indices(args.n)
    rows = []
    for model_name in MODELS:
        model = K.load(os.path.join(P.CACHE_DIR, 'models'), model_name, fold_bn=True, add_mode='sum')
        fc = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
        for eps in EPSILONS:
            comp = LayerMapComposite(layer_map=layer_map_base() + [
                (Convolution, Epsilon(eps, zero_params='bias')), (Linear, Epsilon(eps, zero_params='bias'))])
            for i in indices:
                fname = names[i]
                x = torch.tensor(P.model_input(model_name, i).transpose(0, 3, 1, 2).copy())
                with torch.no_grad():
                    z = model(x)[0]
                t = int(z.argmax())
                seed = float(z[t] - fc.bias[t])
                out = torch.zeros(1, 1000)
                out[0, t] = seed
                t0 = time.time()
                with Gradient(model=model, composite=comp) as attributor:
                    _, rel = attributor(x.requires_grad_(True), out)
                total = float(rel.sum())
                rows.append(dict(implementation='Zennit (canonized)', model=model_name, epsilon=eps, image=fname,
                                 seed=seed, sum_R_in=total, RCE_pct=abs(total - seed) / abs(seed) * 100,
                                 time_s=time.time() - t0))
            r = [x['RCE_pct'] for x in rows if x['model'] == model_name and x['epsilon'] == eps]
            print(f'{model_name:13s} eps={eps:7.0e} median RCE={np.nanmedian(r):.3e}%  '
                  f'>100%: {np.mean(np.array(r) > 100) * 100:.0f}%', flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'zennit.csv'), index=False)


if __name__ == '__main__':
    main()
