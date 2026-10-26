###############################################################################
# Execution Environment: Python 3.12 (environment B, see README)  -- torch (CPU), zennit 1.0.0, captum
###############################################################################
"""Rebuild an exported Keras functional graph as a PyTorch module with identical weights.

The spec/weights come from exp_code/e1_prepare.py. Every operation that an LRP library
attaches a rule to (Conv2d, Linear, BatchNorm2d, activations, pooling, residual sums) is an
nn.Module; purely structural operations (padding, concatenation, flatten) are functional
calls, which LRP libraries pass through with the plain gradient.

Options
  fold_bn  : fold BatchNormalization into the preceding convolution (Zennit/Captum
             "canonization"); otherwise keep nn.BatchNorm2d modules
  add_mode : 'sum'    -> zennit.layer.Sum module (Zennit rule target for residual additions)
             'captum' -> captum Addition_Module (Captum's documented residual support)
             'plain'  -> python `+` (no module; relevance passes with the plain gradient)
Keras 'same' padding is reproduced exactly (asymmetric, bottom/right heavy) with explicit
padding; the input is NCHW.
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_pad(size, k, s, d=1):
    out = math.ceil(size / s)
    total = max((out - 1) * s + (k - 1) * d + 1 - size, 0)
    return total // 2, total - total // 2


class KerasGraph(nn.Module):
    def __init__(self, spec, weights, fold_bn=True, add_mode='sum'):
        super().__init__()
        self.mods = nn.ModuleDict()
        self.plan = []          # (kind, module key or None, input names, output name, extra)
        self.input_name = spec['input']
        alias = {}
        consumers = {}
        for node in spec['nodes']:
            for t in node['inputs']:
                consumers[t] = consumers.get(t, 0) + 1
        producer = {}

        def key(name):
            return name.replace('.', '_').replace('/', '_')

        def resolve(names):
            return [alias.get(n, n) for n in names]

        for node in spec['nodes']:
            t, c, name = node['type'], node['config'], node['name']
            ins, out = resolve(node['inputs']), node['outputs'][0]
            w = [weights[f'{name}__{i}'] for i in range(len([k for k in weights.files if k.startswith(name + '__')]))]
            k = key(name)

            if t in ('Conv2D', 'DepthwiseConv2D'):
                H, W, C = node['input_shapes'][0]
                kh, kw = c['kernel_size']
                sh, sw = c['strides']
                dh, dw = c['dilation_rate']
                if c['padding'] == 'same':
                    pt, pb = _same_pad(H, kh, sh, dh)
                    pl, pr = _same_pad(W, kw, sw, dw)
                    if pt or pb or pl or pr:
                        self.plan.append(('pad', None, ins, out + ':pad', (pl, pr, pt, pb, 0.0)))
                        ins = [out + ':pad']
                if t == 'Conv2D':
                    groups = c.get('groups', 1)
                    conv = nn.Conv2d(C, c['filters'], (kh, kw), (sh, sw), dilation=(dh, dw), groups=groups, bias=True)
                    conv.weight.data = torch.tensor(w[0].transpose(3, 2, 0, 1).copy())
                else:
                    m = c['depth_multiplier']
                    conv = nn.Conv2d(C, C * m, (kh, kw), (sh, sw), dilation=(dh, dw), groups=C, bias=True)
                    conv.weight.data = torch.tensor(w[0].transpose(2, 3, 0, 1).reshape(C * m, 1, kh, kw).copy())
                conv.bias.data = torch.tensor(w[1]) if c['use_bias'] else torch.zeros(conv.out_channels)
                self.mods[k] = conv
                act = c['activation']
                pre = out if act == 'linear' else out + ':pre'
                self.plan.append(('mod', k, ins, pre, None))
                producer[pre] = ('conv', k)
                if act == 'relu':
                    self.mods[k + '_relu'] = nn.ReLU()
                    self.plan.append(('mod', k + '_relu', [pre], out, None))
                elif act != 'linear':
                    raise NotImplementedError(f'{name}: activation {act}')

            elif t == 'BatchNormalization':
                C = node['input_shapes'][0][-1]
                idx = 0
                gamma = w[idx] if c['scale'] else np.ones(C, np.float32)
                idx += int(c['scale'])
                beta = w[idx] if c['center'] else np.zeros(C, np.float32)
                idx += int(c['center'])
                mean, var = w[idx], w[idx + 1]
                src = ins[0]
                if fold_bn and producer.get(src, ('', ''))[0] == 'conv' and consumers.get(src, 0) == 1:
                    conv = self.mods[producer[src][1]]
                    scale = gamma / np.sqrt(var + c['epsilon'])
                    conv.weight.data = conv.weight.data * torch.tensor(scale)[:, None, None, None]
                    conv.bias.data = (conv.bias.data - torch.tensor(mean)) * torch.tensor(scale) + torch.tensor(beta)
                    alias[out] = src
                    producer[src] = ('conv', producer[src][1])
                else:
                    bn = nn.BatchNorm2d(C, eps=c['epsilon'])
                    bn.weight.data, bn.bias.data = torch.tensor(gamma), torch.tensor(beta)
                    bn.running_mean.data, bn.running_var.data = torch.tensor(mean), torch.tensor(var)
                    self.mods[k] = bn
                    self.plan.append(('mod', k, ins, out, None))

            elif t == 'Activation':
                act = c['activation']
                if act in ('linear', 'softmax'):   # softmax: keep logits
                    alias[out] = ins[0]
                elif act == 'relu':
                    self.mods[k] = nn.ReLU()
                    self.plan.append(('mod', k, ins, out, None))
                else:
                    raise NotImplementedError(f'{name}: activation {act}')

            elif t == 'ReLU':
                if c.get('negative_slope', 0) or c.get('threshold', 0):
                    raise NotImplementedError(f'{name}: leaky/thresholded ReLU')
                self.mods[k] = nn.ReLU6() if c.get('max_value') == 6.0 else nn.ReLU()
                self.plan.append(('mod', k, ins, out, None))

            elif t == 'ZeroPadding2D':
                (pt, pb), (pl, pr) = c['padding']
                self.plan.append(('pad', None, ins, out, (pl, pr, pt, pb, 0.0)))

            elif t in ('MaxPooling2D', 'AveragePooling2D'):
                H, W, _ = node['input_shapes'][0]
                kh, kw = c['pool_size']
                sh, sw = c['strides']
                if t == 'MaxPooling2D':
                    if c['padding'] == 'same':
                        pt, pb = _same_pad(H, kh, sh)
                        pl, pr = _same_pad(W, kw, sw)
                        self.plan.append(('pad', None, ins, out + ':pad', (pl, pr, pt, pb, float('-inf'))))
                        ins = [out + ':pad']
                    self.mods[k] = nn.MaxPool2d((kh, kw), (sh, sw))
                else:
                    pad = (0, 0)
                    if c['padding'] == 'same':
                        pt, pb = _same_pad(H, kh, sh)
                        pl, pr = _same_pad(W, kw, sw)
                        if pt != pb or pl != pr:
                            raise NotImplementedError(f'{name}: asymmetric same average pooling')
                        pad = (pt, pl)
                    self.mods[k] = nn.AvgPool2d((kh, kw), (sh, sw), padding=pad, count_include_pad=False)
                self.plan.append(('mod', k, ins, out, None))

            elif t == 'GlobalAveragePooling2D':
                self.mods[k] = nn.AdaptiveAvgPool2d(1)
                self.plan.append(('mod', k, ins, out + ':gap', None))
                self.plan.append(('flatten2d', None, [out + ':gap'], out, None))

            elif t == 'Flatten':
                # functional (no module): LRP libraries pass it with the plain gradient
                self.plan.append(('flatten_nhwc', None, ins, out, None))

            elif t == 'Dense':
                lin = nn.Linear(w[0].shape[0], w[0].shape[1], bias=True)
                lin.weight.data = torch.tensor(w[0].T.copy())
                lin.bias.data = torch.tensor(w[1]) if c['use_bias'] else torch.zeros(w[0].shape[1])
                self.mods[k] = lin
                act = c['activation']
                pre = out if act in ('linear', 'softmax') else out + ':pre'
                self.plan.append(('mod', k, ins, pre, None))
                if act == 'relu':
                    self.mods[k + '_relu'] = nn.ReLU()
                    self.plan.append(('mod', k + '_relu', [pre], out, None))
                elif act not in ('linear', 'softmax'):
                    raise NotImplementedError(f'{name}: activation {act}')

            elif t == 'Add':
                if add_mode == 'sum':
                    from zennit.layer import Sum
                    self.mods[k] = Sum()
                    self.plan.append(('stack_mod', k, ins, out, None))
                elif add_mode == 'captum':
                    from captum.attr._utils.custom_modules import Addition_Module
                    prev = ins[0]
                    for j, other in enumerate(ins[1:]):
                        kk = f'{k}_{j}'
                        self.mods[kk] = Addition_Module()
                        tgt = out if j == len(ins) - 2 else f'{out}:{j}'
                        self.plan.append(('mod2', kk, [prev, other], tgt, None))
                        prev = tgt
                else:
                    self.plan.append(('add', None, ins, out, None))

            elif t == 'Concatenate':
                if c['axis'] not in (-1, 3):
                    raise NotImplementedError(f'{name}: concat axis {c["axis"]}')
                self.plan.append(('cat', None, ins, out, None))

            elif t == 'Dropout':
                alias[out] = ins[0]

            else:
                raise NotImplementedError(f'{name}: {t}')

        self.output_name = alias.get(spec['output'], spec['output'])
        self.eval()

    def forward(self, x):
        env = {self.input_name: x}
        for kind, k, ins, out, extra in self.plan:
            args = [env[n] for n in ins]
            if kind == 'mod':
                env[out] = self.mods[k](args[0])
            elif kind == 'mod2':
                env[out] = self.mods[k](args[0], args[1])
            elif kind == 'stack_mod':
                env[out] = self.mods[k](torch.stack(args, dim=-1))
            elif kind == 'pad':
                pl, pr, pt, pb, value = extra
                env[out] = F.pad(args[0], (pl, pr, pt, pb), value=value)
            elif kind == 'add':
                env[out] = sum(args)
            elif kind == 'cat':
                env[out] = torch.cat(args, dim=1)
            elif kind == 'flatten2d':
                env[out] = args[0].flatten(1)
            elif kind == 'flatten_nhwc':   # Keras flattens NHWC; keeps Dense weights unchanged
                env[out] = args[0].permute(0, 2, 3, 1).flatten(1)
        return env[self.output_name]


def load(model_dir, name, **kwargs):
    with open(os.path.join(model_dir, f'{name}.json'), encoding='utf-8') as f:
        spec = json.load(f)
    weights = np.load(os.path.join(model_dir, f'{name}.npz'))
    return KerasGraph(spec, weights, **kwargs)
