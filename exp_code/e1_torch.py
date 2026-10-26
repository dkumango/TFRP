###############################################################################
# Execution Environment: Python 3.12 (environment B, see README)  -- torch (CPU), zennit 1.0.0, captum
###############################################################################
"""E1. Identical-weight comparison -- Zennit and Captum part.

The Keras networks are rebuilt in PyTorch with identical weights (common/keras_graph_torch.py,
logits verified to <3e-5). Implementations:
  Zennit (canonized)     : BN folded into conv, residual Sum -> Norm, AvgPool -> Norm, ReLU -> Pass
                           (Zennit's recommended practice)
  Zennit (not canonized) : BN kept (layer_map_base: BatchNorm -> Pass), residual `+` without a rule
  Captum                 : BN folded, residual additions as Captum Addition_Module (documented way)
Rules (biases ignored where the library allows it):
  z+         Zennit ZPlus(zero_params='bias')        | Captum Alpha1_Beta0_Rule(set_bias_to_zero=True)
  epsilon    Zennit Epsilon(0.01, zero_params='bias') | Captum EpsilonRule(0.01) (no bias option)
  alpha_beta Zennit AlphaBeta(2, 1, zero_params='bias') | Captum: no such rule
  composite  conv AlphaBeta(2,1), dense Epsilon(0.01)   | Captum: not available
Seeds: Zennit = bias-free target logit (same as TFRP); Captum = full logit (Captum always
starts from the model output).

Output: exp_result/E1/torch.csv (resumable), maps of the first 50 images
Usage : python exp_code/e1_torch.py [--n 1000] [--threads 3]
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
import torch
import torch.nn as nn

from zennit.attribution import Gradient
from zennit.composites import LayerMapComposite, layer_map_base
from zennit.rules import ZPlus, Epsilon, AlphaBeta
from zennit.types import Convolution, Linear

from captum.attr import LRP
from captum.attr._utils.lrp_rules import Alpha1_Beta0_Rule, EpsilonRule

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('exp_code', 'exp_code/common'):
    sys.path.insert(0, os.path.join(ROOT, p))
import e1_common as C
import preprocess as P
import keras_graph_torch as K

MODEL_DIR = os.path.join(P.CACHE_DIR, 'models')


def zennit_composite(rule):
    zb = 'bias'
    if rule == 'z+':
        conv, lin = ZPlus(zero_params=zb), ZPlus(zero_params=zb)
    elif rule == 'epsilon':
        conv, lin = Epsilon(C.EPSILON, zero_params=zb), Epsilon(C.EPSILON, zero_params=zb)
    elif rule == 'alpha_beta':
        conv, lin = AlphaBeta(C.ALPHA, C.BETA, zero_params=zb), AlphaBeta(C.ALPHA, C.BETA, zero_params=zb)
    else:  # composite
        conv, lin = AlphaBeta(C.ALPHA, C.BETA, zero_params=zb), Epsilon(C.EPSILON, zero_params=zb)
    return LayerMapComposite(layer_map=layer_map_base() + [(Convolution, conv), (Linear, lin)])


def zennit_relevance(model, rule, x, target, seed):
    x = x.clone().requires_grad_(True)
    out = torch.zeros(1, 1000)
    out[0, target] = seed
    with Gradient(model=model, composite=zennit_composite(rule)) as attributor:
        _, rel = attributor(x, out)
    return rel


def captum_relevance(model, rule, x, target):
    m = copy.deepcopy(model)
    for mod in m.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            mod.rule = Alpha1_Beta0_Rule(set_bias_to_zero=True) if rule == 'z+' else EpsilonRule(C.EPSILON)
    return LRP(m).attribute(x.clone().requires_grad_(True), target=target)


IMPLS = {  # name: (fold_bn, add_mode, rules run)
    'Zennit (canonized)': (True, 'sum', C.RULES),
    # the un-canonized configuration only illustrates the configuration pitfall; z+ suffices
    'Zennit (not canonized)': (False, 'plain', ['z+']),
    'Captum': (True, 'captum', ['z+', 'epsilon']),
}
NOT_PROVIDED = {'Captum'}   # libraries that lack the remaining rules (vs. rules not run by design)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--threads', type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    os.makedirs(C.OUT_DIR, exist_ok=True)
    csv = os.path.join(C.OUT_DIR, 'torch.csv')
    done = C.done_keys(csv)
    images = P.image_list()[:args.n]

    for model_name in C.MODELS:
        reference = K.load(MODEL_DIR, model_name, fold_bn=True, add_mode='plain')
        fc = [m for m in reference.modules() if isinstance(m, nn.Linear)][-1]
        inputs = []
        for i in range(len(images)):
            x = torch.tensor(P.model_input(model_name, i).transpose(0, 3, 1, 2).copy())
            with torch.no_grad():
                z = reference(x)[0]
            t = int(z.argmax())
            inputs.append((x, t, float(z[t]), float(z[t] - fc.bias[t])))

        for impl, (fold, add_mode, rules) in IMPLS.items():
            model = K.load(MODEL_DIR, model_name, fold_bn=fold, add_mode=add_mode)
            for rule in C.RULES:
                if (impl, model_name, rule) in done:
                    print(f'skip {impl} {model_name} {rule} (done)', flush=True)
                    continue
                rows, maps = [], []
                for i, fname in enumerate(images):
                    x, target, full, bias_free = inputs[i]
                    row = dict(implementation=impl, model=model_name, rule=rule, image=fname)
                    if rule not in rules:
                        if impl in NOT_PROVIDED:
                            rows.append(dict(row, status='NOT_AVAILABLE', error='rule not provided by the library'))
                        else:
                            rows.append(dict(row, status='NOT_RUN', error='configuration run for z+ only'))
                        break
                    t0 = time.time()
                    try:
                        if impl.startswith('Zennit'):
                            rel, seed, kind = zennit_relevance(model, rule, x, target, bias_free), bias_free, 'bias_free'
                        else:
                            rel, seed, kind = captum_relevance(model, rule, x, target), full, 'full'
                        total = float(rel.sum())
                        row.update(status='OK', seed=seed, seed_kind=kind, sum_R_in=total, RCE_pct=C.rce(total, seed),
                                   RCE_vs_biasfree_pct=C.rce(total, bias_free))
                        if i in C.map_set():
                            maps.append(rel[0].detach().sum(0).numpy())
                    except Exception as e:  # noqa: BLE001 - recorded
                        row.update(status='CRASH', error=f'{type(e).__name__}: {str(e)[:200]}')
                    row['time_s'] = time.time() - t0
                    rows.append(row)
                    if row['status'] != 'OK':
                        break  # structural failure is image-independent
                C.append_rows(csv, rows)
                if maps:
                    C.save_maps(impl, model_name, rule, maps)
                ok = [r['RCE_pct'] for r in rows if r['status'] == 'OK']
                print(f'{impl:24s} {model_name:13s} {rule:10s} status={rows[-1]["status"]:13s} '
                      f'median RCE={np.median(ok) if ok else float("nan"):.3e}%  {rows[-1].get("error", "")[:70]}',
                      flush=True)


if __name__ == '__main__':
    main()
