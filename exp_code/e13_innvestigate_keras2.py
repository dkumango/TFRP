###############################################################################
# Execution Environment: Python 3.7 (environment C, see README)  -- tensorflow 1.15, keras 2.2.4, innvestigate 1.0.9
###############################################################################
"""E13. Out-of-the-box relevance conservation: iNNvestigate on keras 2.2.4 applications.

iNNvestigate is used as its README shows:
    model = innvestigate.utils.model_wo_softmax(model)
    analyzer = innvestigate.create_analyzer(name, model, neuron_selection_mode='max_activation')
with name = 'lrp.z_plus' (z+) and 'lrp.sequential_preset_a' (its recommended composite).
iNNvestigate seeds the backward pass with the full target logit, so RCE is measured against
that value (leakage inside the propagation).

keras 2.2.4 ships no EfficientNet / ConvNeXt; those rows are recorded as NOT_AVAILABLE.

Input : exp_result/E13/images.txt (written by e13_tfrp_keras.py)
Output: exp_result/E13/innvestigate_per_image.csv
Usage : python exp_code/e13_innvestigate_keras2.py --image_dir <image_dir>
"""

import argparse
import os
import sys
import time
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import keras
import keras.applications as A
from keras.preprocessing import image
import innvestigate
import innvestigate.utils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'exp_result', 'E13')

MODELS = {
    'vgg16': (A.VGG16, A.vgg16.preprocess_input, 224),
    'resnet50': (A.ResNet50, A.resnet50.preprocess_input, 224),
    'inception_v3': (A.InceptionV3, A.inception_v3.preprocess_input, 299),
    'mobilenet_v2': (A.MobileNetV2, A.mobilenet_v2.preprocess_input, 224),
    'densenet121': (A.DenseNet121, A.densenet.preprocess_input, 224),
    'efficientnet_b0': None,
    'convnext_tiny': None,
}
ANALYZERS = [('lrp.z_plus', 'z+'), ('lrp.sequential_preset_a', 'SequentialPresetA')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image_dir', required=True)
    ap.add_argument('--models', default=','.join(MODELS))
    args = ap.parse_args()

    np.random.seed(42)
    images = open(os.path.join(OUT_DIR, 'images.txt'), encoding='utf-8').read().split()
    rows = []
    for mname in args.models.split(','):
        spec = MODELS[mname]
        if spec is None:
            for _, rule in ANALYZERS:
                rows.append({'library': 'iNNvestigate', 'config': 'documented', 'rule': rule, 'model': mname,
                             'status': 'NOT_AVAILABLE', 'error': 'model not in keras 2.2.4 applications'})
            print('%-16s NOT_AVAILABLE in keras 2.2.4' % mname)
            continue
        ctor, prep, size = spec
        keras.backend.clear_session()
        model = innvestigate.utils.model_wo_softmax(ctor(weights='imagenet'))
        for aname, rule in ANALYZERS:
            row0 = {'library': 'iNNvestigate', 'config': 'documented', 'rule': rule, 'model': mname, 'seed': 'full'}
            try:
                analyzer = innvestigate.create_analyzer(aname, model, neuron_selection_mode='max_activation')
            except Exception as e:  # noqa: BLE001 - recorded as crash
                rows.append(dict(row0, status='CRASH', error='%s: %s' % (type(e).__name__, str(e)[:200])))
                print('%-16s %-18s CRASH %s' % (mname, rule, str(e)[:80]))
                continue
            rces = []
            for fname in images:
                img = image.load_img(os.path.join(args.image_dir, fname), target_size=(size, size))
                x = prep(image.img_to_array(img)[None])
                logits = model.predict(x)
                seed = float(np.max(logits[0]))
                row = dict(row0, image=fname)
                t0 = time.time()
                try:
                    R = analyzer.analyze(x)
                    s = float(np.sum(R))
                    rce = abs(s - seed) / (abs(seed) + 1e-12) * 100
                    row.update(status='OK', R_ref=seed, sum_R_in=s, RCE_pct=rce)
                    rces.append(rce)
                except Exception as e:  # noqa: BLE001 - recorded as crash
                    row.update(status='CRASH', error='%s: %s' % (type(e).__name__, str(e)[:200]))
                row['time_s'] = time.time() - t0
                rows.append(row)
                if row['status'] != 'OK':
                    break
            print('%-16s %-18s status=%-6s median RCE=%.3e%%' % (
                mname, rule, rows[-1]['status'], np.median(rces) if rces else float('nan')))
            sys.stdout.flush()
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'innvestigate_per_image.csv'), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'innvestigate_per_image.csv'), index=False)


if __name__ == '__main__':
    main()
