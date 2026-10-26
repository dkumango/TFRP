###############################################################################
# Execution Environment: Python 3.7 (environment C, see README)  -- tensorflow 1.15, keras 2.2.4, innvestigate 1.0.9
# Run with USERPROFILE=${KERAS2_HOME:-$HOME/.keras2_home} HOME=${KERAS2_HOME:-$HOME/.keras2_home} (separate keras weight cache).
###############################################################################
"""E4. Attribution maps of iNNvestigate on ResNet50 (keras 2.2.4, keras 3 BN epsilon, same weights).

Methods: iNNvestigate z+ (lrp.z_plus) and iNNvestigate composite (conv alpha-beta(2,1),
dense epsilon 0.01, biases ignored -- the same composite as in E1). Same subsets and perturbations
as e4_maps_tf.py.

Usage: python exp_code/e4_maps_innvestigate.py [--threads 0]
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
import tensorflow as tf
import keras

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('exp_code', 'exp_code/common'):
    sys.path.insert(0, os.path.join(ROOT, p))
import preprocess as P
import e4_common as E
from e1_innvestigate import load_model, make_analyzer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        keras.backend.set_session(tf.Session(config=tf.ConfigProto(
            intra_op_parallelism_threads=args.threads, inter_op_parallelism_threads=1)))

    model = load_model(E.MODEL)
    time_csv = os.path.join(ROOT, 'exp_result', 'E4', 'maps_time_innvestigate.csv')
    if not os.path.isdir(os.path.dirname(time_csv)):
        os.makedirs(os.path.dirname(time_csv))
    ev, sv = E.eval_indices(), E.sens_indices()
    for name, rule in (('iNNvestigate z+', 'z+'), ('iNNvestigate composite', 'composite')):
        analyzer = make_analyzer(rule, model)
        fn = lambda x: E.to_2d(analyzer.analyze(x)[0])
        timing = {'method': name}
        if not E.exists(name, 'eval'):
            t0 = time.time()
            E.save(name, 'eval', [fn(P.model_input(E.MODEL, i)) for i in ev])
            timing['sec_per_map'] = (time.time() - t0) / len(ev)
        if not E.exists(name, 'sens'):
            xs = [P.model_input(E.MODEL, i) for i in sv]
            E.save(name, 'sensbase', [fn(x) for x in xs])
            E.save(name, 'sens', [[fn(x + E.perturbation(i, d, x.shape)) for d in range(E.SENS_DRAWS)]
                                  for i, x in zip(sv, xs)])
        pd.DataFrame([timing]).to_csv(time_csv, mode='a', header=not os.path.exists(time_csv), index=False)
        print('%-24s done sec/map=%.3f' % (name, timing.get('sec_per_map', float('nan'))))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
