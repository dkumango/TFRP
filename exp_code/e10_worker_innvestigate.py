###############################################################################
# Execution Environment: Python 3.7 (environment C, see README)  -- tensorflow 1.15, keras 2.2.4, innvestigate 1.0.9
# Run with USERPROFILE=${KERAS2_HOME:-$HOME/.keras2_home} HOME=${KERAS2_HOME:-$HOME/.keras2_home} (separate keras weight cache).
###############################################################################
"""E10 worker: time iNNvestigate lrp.z_plus on one model (same protocol as e10_worker_tf.py).

The analyzer is built (graph construction) before 'LOADED', so the timed loop measures analysis only.
Usage: python exp_code/e10_worker_innvestigate.py --model resnet50 --n 50 --threads 4
"""

import argparse
import json
import os
import sys
import time
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
warnings.filterwarnings('ignore')
import tensorflow as tf
import keras

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('exp_code', 'exp_code/common'):
    sys.path.insert(0, os.path.join(ROOT, p))
import preprocess as P
from e1_innvestigate import load_model, make_analyzer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    keras.backend.set_session(tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=args.threads, inter_op_parallelism_threads=1)))
    model = load_model(args.model)
    analyzer = make_analyzer('z+', model)
    idx = P.sample_indices(args.n + 3, seed=5)
    for i in idx[:3]:
        analyzer.analyze(P.model_input(args.model, i))
    print('LOADED')
    sys.stdout.flush()
    times = []
    for i in idx[3:]:
        x = P.model_input(args.model, i)
        t0 = time.perf_counter()
        analyzer.analyze(x)
        times.append(time.perf_counter() - t0)
    print(json.dumps({'times': times}))
    print('DONE')
    sys.stdout.flush()
    time.sleep(3)


if __name__ == '__main__':
    main()
