###############################################################################
# Execution Environment: Python 3.12 (environment B, see README)  -- torch (CPU), torchvision, zennit 1.0.0, captum
###############################################################################
"""E14. Configuration burden of module-hook LRP libraries (Zennit, Captum) on torchvision models.

Module-hook libraries attach a rule to nn.Module instances; operations written as functions in a
model's forward (x + y, x * g, F.avg_pool2d, ...) are invisible to them and receive the plain
gradient, which is not a conservative LRP rule for additions and products. Each such site must be
rewritten into a module by a model-specific canonizer before LRP is correct.

The model graphs are traced with torch.fx and the sites are counted per category:
  residual_add        tensor + tensor (relevance duplicated by the plain gradient)
  gate_mul            tensor * tensor (relevance scaled by the gate)
  functional_avgpool  functional average pooling (gradient instead of a proportional rule)
  bn_unmergeable      BatchNorm whose input is not the output of a conv used only by that BN
                      (SequentialMergeBatchNorm cannot fold it)
  module_no_rule_zennit / module_no_rule_captum  module types without a rule in the library
and the availability of a dedicated canonizer in Zennit is recorded.

Output: exp_result/E14/torch_sites.csv
Usage : python exp_code/e14_torch_sites.py
"""

import operator
import os
import sys
import warnings

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import pandas as pd
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from captum.attr._core.lrp import SUPPORTED_LAYERS_WITH_RULES, SUPPORTED_NON_LINEAR_LAYERS
from zennit import types as ztypes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'exp_result', 'E14')

MODELS = {
    'vgg16': tvm.vgg16, 'resnet50': tvm.resnet50, 'inception_v3': tvm.inception_v3,
    'mobilenet_v2': tvm.mobilenet_v2, 'densenet121': tvm.densenet121,
    'efficientnet_b0': tvm.efficientnet_b0, 'convnext_tiny': tvm.convnext_tiny,
}
ZENNIT_CANONIZER = {'vgg16': 'VGGCanonizer', 'resnet50': 'ResNetCanonizer'}
# modules Zennit handles (rule or pass-through by design)
ZENNIT_OK = (ztypes.Convolution, ztypes.Linear, ztypes.BatchNorm, ztypes.Activation, ztypes.AvgPool,
             nn.MaxPool2d, nn.AdaptiveMaxPool2d, nn.Dropout, nn.Flatten, nn.Identity)
CAPTUM_OK = tuple(SUPPORTED_LAYERS_WITH_RULES) + tuple(SUPPORTED_NON_LINEAR_LAYERS) + (nn.Identity, nn.Flatten)
ADD_FUNCS = {operator.add, operator.iadd, torch.add}
MUL_FUNCS = {operator.mul, operator.imul, torch.mul}
AVG_FUNCS = {F.avg_pool2d, F.adaptive_avg_pool2d}


class Tracer(fx.Tracer):
    """Keep torchvision building blocks traceable; only leaf modules are recorded as call_module."""


def is_tensor_node(a):
    return isinstance(a, fx.Node)


def count_sites(name, model):
    gm = fx.GraphModule(model, Tracer().trace(model))
    modules = dict(gm.named_modules())
    users = {n: len(n.users) for n in gm.graph.nodes}
    row = dict(model=name, residual_add=0, gate_mul=0, functional_avgpool=0, bn_unmergeable=0,
               bn_total=0, module_no_rule_zennit=0, module_no_rule_captum=0)
    no_rule_z, no_rule_c = set(), set()
    for n in gm.graph.nodes:
        if n.op == 'call_function' or n.op == 'call_method':
            tgt = n.target
            tensor_args = [a for a in n.args if is_tensor_node(a)]
            if (tgt in ADD_FUNCS or tgt in ('add', 'add_')) and len(tensor_args) + (n.op == 'call_method') >= 2:
                row['residual_add'] += 1
            elif (tgt in MUL_FUNCS or tgt in ('mul', 'mul_')) and len(tensor_args) + (n.op == 'call_method') >= 2:
                row['gate_mul'] += 1
            elif tgt in AVG_FUNCS:
                row['functional_avgpool'] += 1
        elif n.op == 'call_module':
            m = modules[n.target]
            if isinstance(m, nn.BatchNorm2d):
                row['bn_total'] += 1
                src = n.args[0]
                ok = (isinstance(src, fx.Node) and src.op == 'call_module'
                      and isinstance(modules[src.target], nn.Conv2d) and users[src] == 1)
                row['bn_unmergeable'] += int(not ok)
            if not isinstance(m, ZENNIT_OK):
                row['module_no_rule_zennit'] += 1
                no_rule_z.add(type(m).__name__)
            if not isinstance(m, CAPTUM_OK):
                row['module_no_rule_captum'] += 1
                no_rule_c.add(type(m).__name__)
    row['zennit_no_rule_types'] = ','.join(sorted(no_rule_z))
    row['captum_no_rule_types'] = ','.join(sorted(no_rule_c))
    row['zennit_dedicated_canonizer'] = ZENNIT_CANONIZER.get(name, 'none')
    row['sites_requiring_canonization'] = (row['residual_add'] + row['gate_mul'] + row['functional_avgpool']
                                           + row['bn_unmergeable'])
    return row


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for name, ctor in MODELS.items():
        model = ctor(weights=None, **({'aux_logits': True, 'init_weights': False} if name == 'inception_v3' else {})).eval()
        try:
            rows.append(count_sites(name, model))
        except Exception as e:  # noqa: BLE001 - recorded (untraceable models need manual rewriting too)
            rows.append(dict(model=name, error=f'{type(e).__name__}: {str(e)[:150]}'))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'torch_sites.csv'), index=False)


if __name__ == '__main__':
    main()
