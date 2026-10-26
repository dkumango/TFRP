###############################################################################
# Execution Environment: Python 3.7 (environment C, see README)  -- tensorflow 1.15, keras 2.2.4, innvestigate 1.0.9
###############################################################################
"""E1. Identical-weight comparison -- iNNvestigate part.

keras 2.2.4 applications load the same ImageNet weight files as keras 3; this is verified by
comparing logits with those exported by e1_prepare.py. Inputs come from the shared uint8 cache
with identical numpy preprocessing.
Rules (biases ignored):
  z+         lrp.z_plus
  epsilon    lrp.epsilon_IB, epsilon = 0.01
  alpha_beta lrp.alpha_2_beta_1_IB
  composite  LRP with conv -> Alpha2Beta1IgnoreBiasRule, dense -> EpsilonRule(0.01, bias=False)
Seed: full target logit (iNNvestigate starts from the model output).

Output: exp_result/E1/innvestigate.csv (resumable), maps of the first 50 images
Usage : python exp_code/e1_innvestigate.py [--n 1000] [--threads 3]
"""

import argparse
import os
import sys
import time
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
import keras
import keras.applications as A
import innvestigate
import innvestigate.utils
import innvestigate.utils.keras.checks as kchecks
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRP
from innvestigate.analyzer.relevance_based import relevance_rule as rrule

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('exp_code', 'exp_code/common'):
    sys.path.insert(0, os.path.join(ROOT, p))
import e1_common as C
import preprocess as P

CTORS = {'vgg16': A.VGG16, 'resnet50': A.ResNet50, 'inception_v3': A.InceptionV3, 'mobilenet_v2': A.MobileNetV2}
IMPL = 'iNNvestigate'


class EpsilonIBRule(rrule.EpsilonRule):
    def __init__(self, *args, **kwargs):
        kwargs.update(epsilon=C.EPSILON, bias=False)
        super(EpsilonIBRule, self).__init__(*args, **kwargs)


def load_model(model_name):
    """keras 2.2.4 application with the keras 3 architecture constants.

    keras 2.2.4's ResNet50 uses BatchNormalization epsilon 1e-3, keras 3 uses 1.001e-5 with the
    same weight file; the model is rebuilt with the keras 3 epsilon so the networks are identical.
    """
    model = CTORS[model_name](weights='imagenet')
    if model_name == 'resnet50':
        cfg = model.get_config()
        for layer in cfg['layers']:
            if layer['class_name'] == 'BatchNormalization':
                layer['config']['epsilon'] = 1.001e-5
        rebuilt = keras.Model.from_config(cfg)
        rebuilt.set_weights(model.get_weights())
        model = rebuilt
    return innvestigate.utils.model_wo_softmax(model)


def make_analyzer(rule, model):
    sel = dict(neuron_selection_mode='max_activation')
    if rule == 'z+':
        return innvestigate.create_analyzer('lrp.z_plus', model, **sel)
    if rule == 'epsilon':
        return innvestigate.create_analyzer('lrp.epsilon_IB', model, epsilon=C.EPSILON, **sel)
    if rule == 'alpha_beta':
        return innvestigate.create_analyzer('lrp.alpha_2_beta_1_IB', model, **sel)
    return LRP(model, rule=[(kchecks.is_dense_layer, EpsilonIBRule),
                            (kchecks.is_conv_layer, rrule.Alpha2Beta1IgnoreBiasRule)], **sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--threads', type=int, default=3)
    args = ap.parse_args()
    keras.backend.set_session(tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=args.threads, inter_op_parallelism_threads=1)))

    if not os.path.isdir(C.OUT_DIR):
        os.makedirs(C.OUT_DIR)
    csv = os.path.join(C.OUT_DIR, 'innvestigate.csv')
    done = C.done_keys(csv)
    images = P.image_list()[:args.n]
    check = []

    for model_name in C.MODELS:
        model = load_model(model_name)
        ref = np.load(os.path.join(P.CACHE_DIR, 'models', model_name + '_logits.npy'))
        x10 = np.concatenate([P.model_input(model_name, i) for i in range(10)])
        diff = float(np.abs(model.predict(x10) - ref).max())
        check.append('%s: max |logit diff| vs keras 3 = %.2e' % (model_name, diff))
        print(check[-1])
        sys.stdout.flush()

        for rule in C.RULES:
            if (IMPL, model_name, rule) in done:
                print('skip %s %s (done)' % (model_name, rule))
                continue
            rows, maps = [], []
            try:
                analyzer = make_analyzer(rule, model)
            except Exception as e:  # noqa: BLE001 - recorded
                C.append_rows(csv, [dict(implementation=IMPL, model=model_name, rule=rule, status='CRASH',
                                         error='%s: %s' % (type(e).__name__, str(e)[:200]))])
                continue
            for i, fname in enumerate(images):
                x = P.model_input(model_name, i)
                seed = float(np.max(model.predict(x)[0]))
                row = dict(implementation=IMPL, model=model_name, rule=rule, image=fname, seed_kind='full')
                t0 = time.time()
                try:
                    R = analyzer.analyze(x)
                    total = float(np.sum(R))
                    row.update(status='OK', seed=seed, sum_R_in=total, RCE_pct=C.rce(total, seed))
                    if i in C.map_set():
                        maps.append(R[0].sum(-1))
                except Exception as e:  # noqa: BLE001 - recorded
                    row.update(status='CRASH', error='%s: %s' % (type(e).__name__, str(e)[:200]))
                row['time_s'] = time.time() - t0
                rows.append(row)
                if row['status'] != 'OK':
                    break
            C.append_rows(csv, rows)
            if maps:
                C.save_maps(IMPL, model_name, rule, maps)
            ok = [r['RCE_pct'] for r in rows if r['status'] == 'OK']
            print('%-13s %-10s OK %d/%d median RCE=%.3e%%' % (model_name, rule, len(ok), len(rows),
                                                              np.median(ok) if ok else float('nan')))
            sys.stdout.flush()
        keras.backend.clear_session()
        keras.backend.set_session(tf.Session(config=tf.ConfigProto(
            intra_op_parallelism_threads=args.threads, inter_op_parallelism_threads=1)))

    with open(os.path.join(C.OUT_DIR, 'innvestigate_weight_check.txt'), 'w') as f:
        f.write('\n'.join(check) + '\n')


if __name__ == '__main__':
    main()
