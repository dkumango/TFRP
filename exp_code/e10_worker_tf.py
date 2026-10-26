###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- tensorflow 2.20, keras 3.14
###############################################################################
"""E10 worker: time TFRP (z+) on one model; memory is measured by exp_code/e10_driver.py.

Protocol (stdout): 'LOADED' after model load and warm-up, one JSON line with per-image times,
'DONE', then a short pause so that the driver can read the peak working set.
Usage: python exp_code/e10_worker_tf.py --model resnet50 --n 50 --threads 4
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('exp_code/common', 'exp_code/tfrp'):
    sys.path.insert(0, os.path.join(ROOT, p))
import preprocess as P
import general_TFRP_v2 as V2

A = tf.keras.applications
CTORS = {'vgg16': A.VGG16, 'resnet50': A.ResNet50, 'inception_v3': A.InceptionV3, 'mobilenet_v2': A.MobileNetV2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    model = CTORS[args.model](weights='imagenet')
    idx = P.sample_indices(args.n + 3, seed=5)
    for i in idx[:3]:
        V2.get_relevance_map_generalized(model, P.model_input(args.model, i))
    print('LOADED', flush=True)
    times = []
    for i in idx[3:]:
        x = P.model_input(args.model, i)
        t0 = time.perf_counter()
        V2.get_relevance_map_generalized(model, x)
        times.append(time.perf_counter() - t0)
    print(json.dumps({'times': times}), flush=True)
    print('DONE', flush=True)
    time.sleep(3)


if __name__ == '__main__':
    main()
