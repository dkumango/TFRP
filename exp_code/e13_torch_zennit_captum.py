###############################################################################
# Execution Environment: Python 3.12 (environment B, see README)  -- torch (CPU), torchvision, zennit 1.0.0, captum
###############################################################################
"""E13. Out-of-the-box relevance conservation: Zennit and Captum on torchvision models.

Each library is used the way its documentation shows, on its own model zoo and
preprocessing (weights.transforms()):

  Zennit z+ (plain)       : LayerMapComposite(layer_map_base + ZPlus on Conv/Linear, bias ignored),
                            no canonizer  -> what a user gets without model-specific work
  Zennit z+ (documented)  : same rules + the canonizer Zennit ships for the model
                            (VGGCanonizer / ResNetCanonizer, otherwise SequentialMergeBatchNorm)
  Zennit EpsilonPlusFlat  : Zennit's recommended composite + the same canonizer
  Captum LRP (default)    : LRP(model).attribute(x, target)   (Captum's default per-layer rules)
  Captum LRP z+           : Alpha1_Beta0_Rule set on every Conv2d/Linear

RCE is measured against the seed score each configuration starts from:
  Zennit z+ configs : bias-free target logit (z+ ignores biases; same seed as TFRP)
  EpsilonPlusFlat   : full target logit (its rules keep biases)
  Captum            : full target logit (Captum always seeds with the model output)
so it quantifies leakage inside the propagation, not differences in the seed.

Input : exp_result/E13/images.txt (written by e13_tfrp_keras.py)
Output: exp_result/E13/torch_per_image.csv
Usage : python exp_code/e13_torch_zennit_captum.py --image_dir <image_dir>
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
import torchvision.models as tvm
from PIL import Image

from zennit.attribution import Gradient
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import LayerMapComposite, EpsilonPlusFlat, layer_map_base
from zennit.rules import ZPlus
from zennit.torchvision import VGGCanonizer, ResNetCanonizer
from zennit.types import Convolution, Linear

from captum.attr import LRP
from captum.attr._utils.lrp_rules import Alpha1_Beta0_Rule

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'exp_result', 'E13')

MODELS = {
    'vgg16': (tvm.vgg16, tvm.VGG16_Weights.IMAGENET1K_V1),
    'resnet50': (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V1),
    'inception_v3': (tvm.inception_v3, tvm.Inception_V3_Weights.IMAGENET1K_V1),
    'mobilenet_v2': (tvm.mobilenet_v2, tvm.MobileNet_V2_Weights.IMAGENET1K_V1),
    'densenet121': (tvm.densenet121, tvm.DenseNet121_Weights.IMAGENET1K_V1),
    'efficientnet_b0': (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.IMAGENET1K_V1),
    'convnext_tiny': (tvm.convnext_tiny, tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
}


def documented_canonizers(name):
    if name == 'vgg16':
        return [VGGCanonizer()], 'VGGCanonizer'
    if name == 'resnet50':
        return [ResNetCanonizer()], 'ResNetCanonizer'
    return [SequentialMergeBatchNorm()], 'SequentialMergeBatchNorm'


def zplus_composite(canonizers=None):
    return LayerMapComposite(layer_map=layer_map_base() + [
        (Convolution, ZPlus(zero_params='bias')), (Linear, ZPlus(zero_params='bias'))],
        canonizers=canonizers)


def last_linear(model):
    return [m for m in model.modules() if isinstance(m, nn.Linear)][-1]


def zennit_run(model, composite, x, target, seed_value):
    x = x.clone().requires_grad_(True)
    attr_out = torch.zeros(1, 1000)
    attr_out[0, target] = seed_value
    with Gradient(model=model, composite=composite) as attributor:
        _, rel = attributor(x, attr_out)
    return float(rel.sum())


def captum_run(model, x, target, zplus):
    m = copy.deepcopy(model)
    if zplus:
        for mod in m.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                mod.rule = Alpha1_Beta0_Rule()
    attr = LRP(m).attribute(x.clone().requires_grad_(True), target=target)
    return float(attr.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image_dir', required=True)
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--out', default='torch_per_image.csv')
    args = ap.parse_args()

    torch.manual_seed(42)
    images = open(os.path.join(OUT_DIR, 'images.txt'), encoding='utf-8').read().split()
    rows = []
    for mname in args.models.split(','):
        ctor, weights = MODELS[mname]
        model = ctor(weights=weights).eval()
        prep = weights.transforms()
        canon, canon_name = documented_canonizers(mname)
        fc = last_linear(model)

        configs = [
            ('Zennit', 'plain (no canonizer)', 'z+', 'bias_free',
             lambda m, x, t, s: zennit_run(m, zplus_composite(), x, t, s)),
            ('Zennit', f'documented ({canon_name})', 'z+', 'bias_free',
             lambda m, x, t, s: zennit_run(m, zplus_composite(documented_canonizers(mname)[0]), x, t, s)),
            ('Zennit', f'documented ({canon_name})', 'EpsilonPlusFlat', 'full',
             lambda m, x, t, s: zennit_run(m, EpsilonPlusFlat(canonizers=documented_canonizers(mname)[0]), x, t, s)),
            ('Captum', 'plain (default rules)', 'default', 'full',
             lambda m, x, t, s: captum_run(m, x, t, zplus=False)),
            ('Captum', 'plain', 'z+ (Alpha1_Beta0)', 'full',
             lambda m, x, t, s: captum_run(m, x, t, zplus=True)),
        ]
        inputs = {}
        for fname in images:  # targets and seeds from the untouched model
            x = prep(Image.open(os.path.join(args.image_dir, fname)).convert('RGB'))[None]
            with torch.no_grad():
                logits = model(x)
            target = int(logits.argmax())
            full = float(logits[0, target])
            inputs[fname] = (x, target, full, full - float(fc.bias[target]))

        for lib, config, rule, seed_kind, fn in configs:
            m = copy.deepcopy(model)  # a canonizer that fails to restore must not affect other configs
            for fname in images:
                x, target, full, bias_free = inputs[fname]
                seed = bias_free if seed_kind == 'bias_free' else full
                row = {'library': lib, 'config': config, 'rule': rule, 'model': mname, 'image': fname,
                       'seed': seed_kind}
                t0 = time.time()
                try:
                    s = fn(m, x, target, seed)
                    row.update(status='OK', R_ref=seed, sum_R_in=s,
                               RCE_pct=abs(s - seed) / (abs(seed) + 1e-12) * 100)
                except Exception as e:  # noqa: BLE001 - recorded as crash
                    row.update(status='CRASH', error=f'{type(e).__name__}: {str(e)[:200]}')
                row['time_s'] = time.time() - t0
                rows.append(row)
                if row['status'] != 'OK':
                    break  # structural failure is image-independent
            done = [r for r in rows if (r['library'], r['config'], r['rule'], r['model']) == (lib, config, rule, mname)]
            ok = [r['RCE_pct'] for r in done if r['status'] == 'OK']
            print(f"{mname:16s} {lib:7s} {config:40s} {rule:18s} status={done[-1]['status']:6s} "
                  f"median RCE={np.median(ok) if ok else float('nan'):.3e}%  {done[-1].get('error', '')[:80]}",
                  flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, args.out), index=False)


if __name__ == '__main__':
    main()
