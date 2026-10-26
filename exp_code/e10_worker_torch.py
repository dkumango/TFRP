###############################################################################
# Execution Environment: Python 3.12 (environment B, see README)  -- torch (CPU), zennit 1.0.0, captum
###############################################################################
"""E10 worker: time Zennit (canonized) or Captum LRP-z+ on one identical-weight model.

Same protocol as e10_worker_tf.py ('LOADED', JSON times, 'DONE').
Usage: python exp_code/e10_worker_torch.py --impl zennit --model resnet50 --n 50 --threads 4
"""

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('exp_code', 'exp_code/common'):
    sys.path.insert(0, os.path.join(ROOT, p))
import preprocess as P
import keras_graph_torch as K
from zennit.attribution import Gradient
from zennit.composites import LayerMapComposite, layer_map_base
from zennit.rules import ZPlus
from zennit.types import Convolution, Linear
from captum.attr import LRP
from captum.attr._utils.lrp_rules import Alpha1_Beta0_Rule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--impl', choices=['zennit', 'captum'], required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    model_dir = os.path.join(P.CACHE_DIR, 'models')

    if args.impl == 'zennit':
        model = K.load(model_dir, args.model, fold_bn=True, add_mode='sum')
        comp = LayerMapComposite(layer_map=layer_map_base() + [
            (Convolution, ZPlus(zero_params='bias')), (Linear, ZPlus(zero_params='bias'))])

        def explain(x):
            with torch.no_grad():
                t = int(model(x).argmax())
            out = torch.zeros(1, 1000)
            out[0, t] = 1.0
            with Gradient(model=model, composite=comp) as attributor:
                attributor(x.requires_grad_(True), out)
    else:
        model = K.load(model_dir, args.model, fold_bn=True, add_mode='captum')
        for mod in model.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                mod.rule = Alpha1_Beta0_Rule(set_bias_to_zero=True)
        lrp = LRP(model)

        def explain(x):
            with torch.no_grad():
                t = int(model(x).argmax())
            lrp.attribute(x.requires_grad_(True), target=t)

    idx = P.sample_indices(args.n + 3, seed=5)
    to_t = lambda i: torch.tensor(P.model_input(args.model, i).transpose(0, 3, 1, 2).copy())
    for i in idx[:3]:
        explain(to_t(i))
    print('LOADED', flush=True)
    times = []
    for i in idx[3:]:
        x = to_t(i)
        t0 = time.perf_counter()
        explain(x)
        times.append(time.perf_counter() - t0)
    print(json.dumps({'times': times}), flush=True)
    print('DONE', flush=True)
    time.sleep(3)


if __name__ == '__main__':
    main()
