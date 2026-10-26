###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- tensorflow 2.20, keras 3.14
###############################################################################
"""E0. Layer-level unit tests for TFRP v1 (published) and v2 (revision).

For each layer type a tiny Keras graph is built with fixed random weights and checked for
  (a) forward response : the linear map TFRP uses reproduces the Keras layer (pre-activation)
  (b) conservation     : z+ relevance sum at the input equals the bias-free target logit
  (c) support          : unsupported layers must raise instead of returning a wrong map

Output: exp_result/E0/unit_tests.csv
Usage : python exp_code/e0_unit_tests.py   (run from the project root)
"""

import contextlib
import io
import os
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers as L

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'exp_code', 'tfrp'))
sys.path.insert(0, os.path.join(ROOT, 'exp_code/tfrp'))
import general_TFRP_v2 as V2
import general_TFRP_v1 as V1

OUT_DIR = os.path.join(ROOT, 'exp_result', 'E0')
SHAPE = (12, 12, 4)
TOL_PCT = 1e-3   # conservation tolerance in percent (float32 accumulation)


def head(x):
    """Common classifier head: positive bias-free Dense so the target logit is positive."""
    x = L.Flatten()(x)
    return L.Dense(3, use_bias=False, kernel_initializer=keras.initializers.RandomUniform(0.01, 0.1))(x)


def relu_input():
    inp = keras.Input(SHAPE)
    return inp, L.ReLU()(inp)


def build(name):
    """Return (model, signed_input) for a named test graph."""
    keras.utils.set_random_seed(0)
    inp, x = relu_input()
    signed = False
    if name == 'Dense(2D)':
        out = L.Dense(3, use_bias=False, kernel_initializer=keras.initializers.RandomUniform(0.01, 0.1))(
            L.Dense(16, activation='relu')(L.Flatten()(x)))
        return keras.Model(inp, out), signed
    if name == 'Dense(4D)':
        y = L.Dense(8, activation='relu')(x)
    elif name == 'Conv2D(s1,same)':
        y = L.Conv2D(6, 3, padding='same', activation='relu')(x)
    elif name == 'Conv2D(s2,valid)':
        y = L.Conv2D(6, 3, strides=2, padding='valid', activation='relu')(x)
    elif name == 'Conv2D(dilated)':
        y = L.Conv2D(6, 3, dilation_rate=2, padding='same', activation='relu')(x)
    elif name == 'Conv2D(signed input)':
        signed = True
        inp = keras.Input(SHAPE)
        y = L.Conv2D(6, 3, padding='same', activation='relu')(inp)
    elif name == 'DepthwiseConv2D(s1)':
        y = L.DepthwiseConv2D(3, padding='same', activation='relu')(x)
    elif name == 'DepthwiseConv2D(s2)':
        y = L.DepthwiseConv2D(3, strides=2, padding='same', activation='relu')(x)
    elif name == 'BatchNormalization':
        y = L.ReLU()(L.BatchNormalization()(L.Conv2D(6, 3, padding='same')(x)))
    elif name == 'Add':
        a = L.Conv2D(4, 1, activation='relu')(x)
        y = L.ReLU()(L.Add()([a, x]))
    elif name == 'Concatenate':
        a = L.Conv2D(3, 1, activation='relu')(x)
        b = L.Conv2D(5, 3, padding='same', activation='relu')(x)
        y = L.Concatenate()([a, b])
    elif name == 'ZeroPadding2D(asym)':
        y = L.Conv2D(4, 3, padding='valid', activation='relu')(L.ZeroPadding2D(((0, 1), (0, 1)))(x))
    elif name == 'MaxPooling2D':
        y = L.MaxPooling2D(2)(L.Conv2D(4, 3, padding='same', activation='relu')(x))
    elif name == 'GlobalAveragePooling2D':
        g = L.GlobalAveragePooling2D()(L.Conv2D(8, 3, padding='same', activation='relu')(x))
        out = L.Dense(3, use_bias=False, kernel_initializer=keras.initializers.RandomUniform(0.01, 0.1))(g)
        return keras.Model(inp, out), signed
    elif name == 'AveragePooling2D':
        y = L.AveragePooling2D(2)(L.Conv2D(4, 3, padding='same', activation='relu')(x))
    elif name == 'Multiply(SE gate)':
        c = L.Conv2D(4, 3, padding='same', activation='relu')(x)
        s = L.Dense(4, activation='sigmoid')(L.GlobalAveragePooling2D(keepdims=True)(c))
        y = L.Multiply()([c, s])
    elif name == 'LayerNormalization':
        y = L.ReLU()(L.LayerNormalization()(L.Conv2D(4, 3, padding='same')(x)))
    elif name == 'Reshape':
        y = L.Reshape((SHAPE[0] * SHAPE[1], SHAPE[2]))(x)
    elif name == 'keras.ops Add (residual)':
        a = L.Conv2D(4, 3, padding='same', activation='relu')(x)
        y = a + x                     # raw tensor op, no Add layer
    elif name == 'Nested Sequential':
        block = keras.Sequential([keras.Input(SHAPE), L.Conv2D(6, 3, padding='same'),
                                  L.BatchNormalization(), L.ReLU()])
        y = block(x)
    elif name == 'Conv2D(groups=C, depthwise)':
        y = L.Conv2D(8, 3, padding='same', groups=4, activation='relu')(x)
    elif name == 'Conv2D(groups=2)':
        y = L.Conv2D(6, 3, padding='same', groups=2, activation='relu')(x)
    else:
        raise KeyError(name)
    return keras.Model(inp, head(y)), signed


TESTS = ['Dense(2D)', 'Dense(4D)', 'Conv2D(s1,same)', 'Conv2D(s2,valid)', 'Conv2D(dilated)',
         'Conv2D(signed input)', 'DepthwiseConv2D(s1)', 'DepthwiseConv2D(s2)', 'BatchNormalization',
         'Add', 'Concatenate', 'ZeroPadding2D(asym)', 'MaxPooling2D', 'GlobalAveragePooling2D',
         'AveragePooling2D', 'Multiply(SE gate)', 'LayerNormalization', 'Reshape',
         'keras.ops Add (residual)', 'Nested Sequential', 'Conv2D(groups=C, depthwise)', 'Conv2D(groups=2)']


def forward_check(model):
    """Max |TFRP v2 linear map + bias - Keras pre-activation| over weighted layers."""
    worst = 0.0
    flat = []
    for layer in model.layers:  # expand nested models
        flat.extend(layer.layers if isinstance(layer, keras.Model) else [layer])
    for layer in flat:
        if layer.__class__.__name__ not in V2.WEIGHTED_LAYERS:
            continue
        x = tf.random.normal((1,) + tuple(layer.input.shape[1:]), seed=1)
        act = layer.activation
        layer.activation = keras.activations.linear
        ref = layer(x)
        layer.activation = act
        z = V2._forward_op(layer)(x, V2._kernel(layer))
        if layer.use_bias:
            z = z + layer.bias
        worst = max(worst, float(tf.reduce_max(tf.abs(z - ref))))
    return worst


def run(module, model, x):
    """Return (raw input relevance, RCE %)."""
    with contextlib.redirect_stdout(io.StringIO()):
        out = module.get_relevance_map_generalized(model, x, target_class_idx=0, use_logit=True,
                                                   global_rule='z_plus')
    R, init, final = out[0], out[1], out[2]
    if module is V1:  # v1 rescales the map to the output score; undo it
        R = R * (final / init)
    return R, abs(final - init) / (abs(init) + 1e-12) * 100


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for name in TESTS:
        model, signed = build(name)
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, (1,) + SHAPE).astype(np.float32)
        if not signed:
            x = np.abs(x)
        row = {'test': name, 'forward_max_abs_err': forward_check(model)}
        maps = {}
        for tag, module in (('v1', V1), ('v2', V2)):
            try:
                maps[tag], rce = run(module, model, x)
                row[f'{tag}_RCE_pct'] = rce
                row[f'{tag}_status'] = 'PASS' if rce < TOL_PCT else 'FAIL (leak)'
            except V2.UnsupportedLayerError as e:
                row[f'{tag}_RCE_pct'] = np.nan
                row[f'{tag}_status'] = 'UNSUPPORTED (raised)'
            except Exception as e:  # noqa: BLE001 - recorded as a crash for the report
                row[f'{tag}_RCE_pct'] = np.nan
                row[f'{tag}_status'] = f'CRASH: {type(e).__name__}: {str(e)[:80]}'
        # Relative L2 difference of the v1 map from the v2 map (v2 forward response is verified)
        if 'v1' in maps and 'v2' in maps and maps['v1'].shape == maps['v2'].shape:
            row['v1_vs_v2_rel_L2'] = float(np.linalg.norm(maps['v1'] - maps['v2']) /
                                           (np.linalg.norm(maps['v2']) + 1e-12))
        else:
            row['v1_vs_v2_rel_L2'] = np.nan
        rows.append(row)
        print(f"{name:24s} fwd_err={row['forward_max_abs_err']:.1e} | "
              f"v1 {row['v1_status']:<22s} RCE={row['v1_RCE_pct']:.2e}% | "
              f"v2 {row['v2_status']:<22s} RCE={row['v2_RCE_pct']:.2e}% | "
              f"map diff v1-v2={row['v1_vs_v2_rel_L2']:.2e}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'unit_tests.csv'), index=False)
    print(f"\nSaved {os.path.join(OUT_DIR, 'unit_tests.csv')}")


if __name__ == '__main__':
    main()
