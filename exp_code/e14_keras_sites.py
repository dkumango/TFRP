###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- tensorflow 2.20, keras 3.14
###############################################################################
"""E14. The same site categories in the keras.applications graphs, and how TFRP handles them.

TFRP propagates over the Keras node graph, so additions, products, pooling and normalisation are
graph nodes with a generic rule regardless of whether they were written as layers or as keras.ops
calls. For every model the sites of each category are counted and every operation type is checked
against TFRP's dispatch: 'rule' (handled), 'pass' (shape-preserving unary op), or 'rejected'
(UnsupportedLayerError). Manual sites = 0 means a single call works without model-specific code.

Output: exp_result/E14/keras_sites.csv
Usage : python exp_code/e14_keras_sites.py
"""

import collections
import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import keras
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'exp_code', 'tfrp'))
import general_TFRP_v2 as V2

A = keras.applications
MODELS = {'vgg16': A.VGG16, 'resnet50': A.ResNet50, 'inception_v3': A.InceptionV3, 'mobilenet_v2': A.MobileNetV2,
          'densenet121': A.DenseNet121, 'efficientnet_b0': A.EfficientNetB0, 'convnext_tiny': A.ConvNeXtTiny}
RULE_TYPES = set(V2.WEIGHTED_LAYERS) | set(V2.GRADIENT_LAYERS) | {'Add', 'Multiply', 'Concatenate', 'ZeroPadding2D',
                                                                   'Flatten', 'Reshape'}


def walk(functional, counts, kinds):
    for depth in sorted(functional._nodes_by_depth, reverse=True):
        for node in functional._nodes_by_depth[depth]:
            op = node.operation
            if node.is_input or op is None:
                continue
            if isinstance(op, keras.Model):
                counts['nested_model'] += 1
                walk(V2._as_functional(op), counts, kinds)
                continue
            t = type(op).__name__
            n_in = len(node.input_tensors)
            if t == 'Add' and n_in > 1:
                counts['residual_add'] += 1
            elif t == 'Multiply' and n_in > 1:
                counts['gate_mul'] += 1
            elif t in ('AveragePooling2D', 'GlobalAveragePooling2D'):
                counts['avgpool'] += 1
            elif t in ('BatchNormalization', 'LayerNormalization'):
                counts['normalisation'] += 1
            if t in RULE_TYPES:
                kinds[t] = 'rule'
            elif n_in == 1 and tuple(node.input_tensors[0].shape[1:]) == tuple(node.outputs[0].shape[1:]):
                kinds.setdefault(t, 'pass')
            else:
                kinds[t] = 'rejected'


def main():
    os.makedirs(os.path.join(ROOT, 'exp_result', 'E14'), exist_ok=True)
    rows = []
    for name, ctor in MODELS.items():
        model = ctor(weights=None)
        counts, kinds = collections.Counter(), {}
        walk(V2._as_functional(model), counts, kinds)
        rows.append(dict(model=name, **{k: counts[k] for k in ('residual_add', 'gate_mul', 'avgpool',
                                                                  'normalisation', 'nested_model')},
                         op_types_rule=','.join(sorted(k for k, v in kinds.items() if v == 'rule')),
                         op_types_pass=','.join(sorted(k for k, v in kinds.items() if v == 'pass')),
                         op_types_rejected=','.join(sorted(k for k, v in kinds.items() if v == 'rejected')),
                         manual_sites=sum(v == 'rejected' for v in kinds.values())))
        print(rows[-1], flush=True)
        keras.backend.clear_session()
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'exp_result', 'E14', 'keras_sites.csv'), index=False)


if __name__ == '__main__':
    main()
