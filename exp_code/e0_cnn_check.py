###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- tensorflow 2.20, keras 3.14
###############################################################################
"""E0. TFRP v1 vs v2 on the four paper CNNs, with per-layer leakage localisation.

For each model and image: z+ RCE of v1 (published, legacy input rule) and v2, the
relative L2 difference between their maps, and v2's per-layer conservation log, so that
any remaining leakage can be attributed to a layer type.

Outputs:
  exp_result/E0/cnn_check_per_image.csv
  exp_result/E0/cnn_check_layer_leak.csv   (v2, summed over images, per layer type)
Usage:
  python exp_code/e0_cnn_check.py --image_dir data/imagen --n 20
"""

import argparse
import contextlib
import io
import os
import random
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'exp_code', 'tfrp'))
sys.path.insert(0, os.path.join(ROOT, 'exp_code/tfrp'))
import general_TFRP_v2 as V2
import general_TFRP_v1 as V1

APPS = tf.keras.applications
MODELS = {
    'vgg16': (APPS.VGG16, APPS.vgg16.preprocess_input, 224),
    'resnet50': (APPS.ResNet50, APPS.resnet50.preprocess_input, 224),
    'inception_v3': (APPS.InceptionV3, APPS.inception_v3.preprocess_input, 299),
    'mobilenet_v2': (APPS.MobileNetV2, APPS.mobilenet_v2.preprocess_input, 224),
}


def sample_images(image_dir, n, seed):
    files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    return sorted(random.Random(seed).sample(files, min(n, len(files))))


def load(path, size, prep):
    img = Image.open(path).convert('RGB').resize((size, size), Image.Resampling.BILINEAR)
    return prep(np.asarray(img, dtype=np.float32).copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image_dir', required=True)
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, 'exp_result', 'E0')
    os.makedirs(out_dir, exist_ok=True)
    images = sample_images(args.image_dir, args.n, args.seed)
    rows, layer_rows = [], []

    for mname, (ctor, prep, size) in MODELS.items():
        model = ctor(weights='imagenet')
        for fname in images:
            x = load(os.path.join(args.image_dir, fname), size, prep)[None]

            V1.SIGNED_INPUT_ZPLUS = False  # published v1 behaviour
            with contextlib.redirect_stdout(io.StringIO()):
                R1, i1, f1, _ = V1.get_relevance_map_generalized(model, x, use_logit=True, global_rule='z_plus')
            R1 = R1 * (f1 / i1)
            V1.SIGNED_INPUT_ZPLUS = True

            R2, i2, f2, _, log = V2.get_relevance_map_generalized(model, x, use_logit=True, global_rule='z_plus',
                                                                  return_layer_log=True)
            rows.append({
                'model': mname, 'image': fname, 'R_out': i2,
                'v1_RCE_pct': abs(f1 - i1) / abs(i1) * 100,
                'v2_RCE_pct': abs(f2 - i2) / abs(i2) * 100,
                'v1_vs_v2_rel_L2': float(np.linalg.norm(R1 - R2) / (np.linalg.norm(R2) + 1e-12)),
            })
            for lname, ltype, upper, lower in log:
                layer_rows.append({'model': mname, 'image': fname, 'layer': lname, 'type': ltype,
                                   'leak_abs': upper - lower, 'leak_pct_of_Rout': (upper - lower) / i2 * 100})
            r = rows[-1]
            print(f"{mname:13s} {fname[:32]:32s} v1 RCE={r['v1_RCE_pct']:.2e}% v2 RCE={r['v2_RCE_pct']:.2e}% "
                  f"map diff={r['v1_vs_v2_rel_L2']:.3f}")
        tf.keras.backend.clear_session()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'cnn_check_per_image.csv'), index=False)
    ldf = pd.DataFrame(layer_rows)
    by_type = (ldf.groupby(['model', 'type'])['leak_pct_of_Rout']
               .agg(lambda s: s.sum() / ldf['image'].nunique()).rename('mean_leak_pct_per_image').reset_index())
    by_type.to_csv(os.path.join(out_dir, 'cnn_check_layer_leak.csv'), index=False)

    print('\n=== Mean over images ===')
    print(df.groupby('model')[['v1_RCE_pct', 'v2_RCE_pct', 'v1_vs_v2_rel_L2']].mean().to_string())
    print('\n=== v2 leakage by layer type (% of R_out, mean per image; nonzero only) ===')
    print(by_type[by_type.mean_leak_pct_per_image.abs() > 1e-4].to_string(index=False))


if __name__ == '__main__':
    main()
