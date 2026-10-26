###############################################################################
# Execution Environment: Python 3.12 (environment B, see README)  -- torch (CPU), zennit 1.0.0, captum
###############################################################################
"""E4. Attribution maps of the PyTorch LRP implementations on ResNet50 (identical weights).

Methods
  Zennit z+               canonized (BN folded, Sum -> Norm), ZPlus on conv/dense, biases ignored
  Zennit EpsilonPlusFlat  Zennit's recommended composite, canonized
  Captum z+               BN folded, Addition_Module, Alpha1_Beta0_Rule(set_bias_to_zero=True)
Seeds: Zennit = bias-free target logit; Captum = full logit (library default).
Same subsets and perturbations as e4_maps_tf.py.

Usage: python exp_code/e4_maps_torch.py [--threads 0]
"""

import argparse
import copy
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
from zennit.composites import LayerMapComposite, EpsilonPlusFlat, layer_map_base
from zennit.rules import ZPlus
from zennit.types import Convolution, Linear
from captum.attr import LRP
from captum.attr._utils.lrp_rules import Alpha1_Beta0_Rule

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('exp_code', 'exp_code/common'):
    sys.path.insert(0, os.path.join(ROOT, p))
import preprocess as P
import e4_common as E
import keras_graph_torch as K

MODEL_DIR = os.path.join(P.CACHE_DIR, 'models')


def zennit_map(model, composite, x, target, seed):
    out = torch.zeros(1, 1000)
    out[0, target] = seed
    with Gradient(model=model, composite=composite) as attributor:
        _, rel = attributor(x.clone().requires_grad_(True), out)
    return rel[0].detach().sum(0).numpy()


def captum_map(model, x, target):
    return LRP(model).attribute(x.clone().requires_grad_(True), target=target)[0].detach().sum(0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    zen = K.load(MODEL_DIR, E.MODEL, fold_bn=True, add_mode='sum')
    cap = K.load(MODEL_DIR, E.MODEL, fold_bn=True, add_mode='captum')
    for mod in cap.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            mod.rule = Alpha1_Beta0_Rule(set_bias_to_zero=True)
    fc = [m for m in zen.modules() if isinstance(m, nn.Linear)][-1]
    zplus = LayerMapComposite(layer_map=layer_map_base() + [
        (Convolution, ZPlus(zero_params='bias')), (Linear, ZPlus(zero_params='bias'))])
    methods = {
        'Zennit z+': lambda x, t, s: zennit_map(zen, zplus, x, t, s),
        'Zennit EpsilonPlusFlat': lambda x, t, s: zennit_map(zen, EpsilonPlusFlat(), x, t, s),
        'Captum z+': lambda x, t, s: captum_map(cap, x, t),
    }

    def run(fn, x_nhwc):
        x = torch.tensor(x_nhwc.transpose(0, 3, 1, 2).copy())
        with torch.no_grad():
            z = zen(x)[0]
        t = int(z.argmax())
        return fn(x, t, float(z[t] - fc.bias[t]))

    time_csv = os.path.join(ROOT, 'exp_result', 'E4', 'maps_time_torch.csv')
    os.makedirs(os.path.dirname(time_csv), exist_ok=True)
    ev, sv = E.eval_indices(), E.sens_indices()
    for name, fn in methods.items():
        timing = {'method': name}
        if not E.exists(name, 'eval'):
            t0 = time.time()
            E.save(name, 'eval', [run(fn, P.model_input(E.MODEL, i)) for i in ev])
            timing['sec_per_map'] = (time.time() - t0) / len(ev)
        if not E.exists(name, 'sens'):
            xs = [P.model_input(E.MODEL, i) for i in sv]
            E.save(name, 'sensbase', [run(fn, x) for x in xs])
            E.save(name, 'sens', [[run(fn, x + E.perturbation(i, d, x.shape)) for d in range(E.SENS_DRAWS)]
                                  for i, x in zip(sv, xs)])
        pd.DataFrame([timing]).to_csv(time_csv, mode='a', header=not os.path.exists(time_csv), index=False)
        print(f'{name:24s} done sec/map={timing.get("sec_per_map", float("nan")):.3f}', flush=True)


if __name__ == '__main__':
    main()
