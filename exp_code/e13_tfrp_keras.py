###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- tensorflow 2.20, keras 3.14
###############################################################################
"""E13. Out-of-the-box relevance conservation: TFRP on keras.applications models.

TFRP is called exactly as a user would call it -- model + rule, nothing else:
    get_relevance_map_generalized(model, x, use_logit=True, global_rule='z_plus')
Both the published v1 and the revised v2 are run. v2 raises UnsupportedLayerError on
layers it has no rule for; that is recorded as status UNSUPPORTED (v1 never raises, it
silently copies relevance).

This script also writes the shared image list (exp_result/E13/images.txt) used by the
PyTorch (Zennit/Captum) and Python 3.7 (iNNvestigate) scripts.

RCE is measured against the seed score each library starts from (here: the bias-free
target logit), i.e. it quantifies leakage *inside* the propagation.

Output: exp_result/E13/tfrp_per_image.csv
Usage : python exp_code/e13_tfrp_keras.py --image_dir <image_dir> --n 50
"""

import argparse
import contextlib
import io
import os
import random
import sys
import time

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

A = tf.keras.applications
MODELS = {  # name: (constructor, preprocess, input size)
    'vgg16': (A.VGG16, A.vgg16.preprocess_input, 224),
    'resnet50': (A.ResNet50, A.resnet50.preprocess_input, 224),
    'inception_v3': (A.InceptionV3, A.inception_v3.preprocess_input, 299),
    'mobilenet_v2': (A.MobileNetV2, A.mobilenet_v2.preprocess_input, 224),
    'densenet121': (A.DenseNet121, A.densenet.preprocess_input, 224),
    'efficientnet_b0': (A.EfficientNetB0, A.efficientnet.preprocess_input, 224),
    'convnext_tiny': (A.ConvNeXtTiny, A.convnext.preprocess_input, 224),
}
OUT_DIR = os.path.join(ROOT, 'exp_result', 'E13')


def write_image_list(image_dir, n, seed):
    files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    chosen = sorted(random.Random(seed).sample(files, min(n, len(files))))
    with open(os.path.join(OUT_DIR, 'images.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(chosen))
    return chosen


def load(path, size, prep):
    img = Image.open(path).convert('RGB').resize((size, size), Image.Resampling.BILINEAR)
    return prep(np.asarray(img, dtype=np.float32).copy())


def run_v1(model, x):
    V1.SIGNED_INPUT_ZPLUS = False  # published behaviour
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _, init, final, _ = V1.get_relevance_map_generalized(model, x, use_logit=True, global_rule='z_plus')
    finally:
        V1.SIGNED_INPUT_ZPLUS = True
    return init, final


def run_v2(model, x):
    _, init, final, _ = V2.get_relevance_map_generalized(model, x, use_logit=True, global_rule='z_plus')
    return init, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image_dir', required=True)
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--out', default='tfrp_per_image.csv')
    ap.add_argument('--threads', type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    os.makedirs(OUT_DIR, exist_ok=True)
    images = write_image_list(args.image_dir, args.n, args.seed)
    rows = []
    for mname in args.models.split(','):
        ctor, prep, size = MODELS[mname]
        model = ctor(weights='imagenet')
        for lib, fn in (('TFRP-v1 (published)', run_v1), ('TFRP-v2 (revised)', run_v2)):
            stop_after_first = False
            for fname in images:
                x = load(os.path.join(args.image_dir, fname), size, prep)[None]
                row = {'library': lib, 'config': 'plain', 'rule': 'z+', 'model': mname, 'image': fname}
                t0 = time.time()
                try:
                    init, final = fn(model, x)
                    row.update(status='OK', R_ref=init, sum_R_in=final,
                               RCE_pct=abs(final - init) / (abs(init) + 1e-12) * 100)
                except V2.UnsupportedLayerError as e:
                    row.update(status='UNSUPPORTED', error=str(e)[:200])
                    stop_after_first = True
                except Exception as e:  # noqa: BLE001 - recorded as crash
                    row.update(status='CRASH', error=f'{type(e).__name__}: {str(e)[:200]}')
                    stop_after_first = True
                row['time_s'] = time.time() - t0
                rows.append(row)
                if stop_after_first:  # structural failure is image-independent
                    break
            done = [r for r in rows if r['library'] == lib and r['model'] == mname]
            ok = [r['RCE_pct'] for r in done if r['status'] == 'OK']
            print(f"{mname:16s} {lib:22s} status={done[-1]['status']:12s} "
                  f"median RCE={np.median(ok) if ok else float('nan'):.3e}%  "
                  f"{done[-1].get('error', '')[:90]}", flush=True)
        tf.keras.backend.clear_session()
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, args.out), index=False)


if __name__ == '__main__':
    main()
