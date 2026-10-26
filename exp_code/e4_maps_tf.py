###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- tensorflow 2.20, keras 3.14, xplique 2.0.1
###############################################################################
"""E4. Attribution maps of the TensorFlow methods on ResNet50.

Methods: TFRP z+, TFRP composite (conv alpha-beta(2,1), dense epsilon 0.01), Saliency,
Guided Backprop, Integrated Gradients (50 steps, zero baseline), SmoothGrad (50 samples,
noise 0.2), Grad-CAM, Grad-CAM++, RISE (1000 masks, 7x7 grid). Gradient-based methods use the
pre-softmax logit model; the target is the top-1 class.

For each method the job list is: the evaluation subset, then (except RISE) the sensitivity
subset and its SENS_DRAWS perturbed copies. Execution is chunked -- each call computes the next
--chunk maps of one method and exits (TensorFlow's CPU allocator grows over long runs); the call
that completes a method merges the chunks into <method>__eval/sensbase/sens.npy.
Run through exp_code/run_e4_maps_tf.sh.

Usage: python exp_code/e4_maps_tf.py --method "TFRP z+" [--chunk 100] [--threads 0]
"""

import argparse
import os
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import tensorflow as tf
from xplique import attributions as XA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('exp_code', 'exp_code/common', 'exp_code/tfrp'):
    sys.path.insert(0, os.path.join(ROOT, p))
import preprocess as P
import e4_common as E
import general_TFRP_v2 as V2

COMPOSITE = {'Conv2D': 'alpha_beta', 'DepthwiseConv2D': 'alpha_beta', 'Dense': 'epsilon'}
METHODS = ['TFRP z+', 'TFRP composite', 'Saliency', 'GuidedBackprop', 'GradCAM', 'GradCAMPP', 'SmoothGrad',
           'IntegratedGradients', 'RISE']
NO_SENSITIVITY = {'RISE'}   # stochastic masks dominate its sensitivity; excluded and stated in the paper


def build_explainer(name, logit_model):
    if name == 'Saliency':
        return XA.Saliency(logit_model, batch_size=16)
    if name == 'GuidedBackprop':
        return XA.GuidedBackprop(logit_model, batch_size=16)
    if name == 'IntegratedGradients':
        return XA.IntegratedGradients(logit_model, batch_size=16, steps=50, baseline_value=0.0)
    if name == 'SmoothGrad':
        return XA.SmoothGrad(logit_model, batch_size=16, nb_samples=50, noise=0.2)
    if name == 'GradCAM':
        return XA.GradCAM(logit_model, batch_size=16)
    if name == 'GradCAMPP':
        return XA.GradCAMPP(logit_model, batch_size=16)
    if name == 'RISE':
        return XA.Rise(logit_model, batch_size=32, nb_samples=1000, grid_size=7)
    raise KeyError(name)


def tfrp_map(model, x, target, composite):
    kw = dict(global_rule='z_plus', composite_preset=COMPOSITE, alpha=2.0, beta=1.0, epsilon=0.01) if composite \
        else dict(global_rule='z_plus')
    R, _, _, _ = V2.get_relevance_map_generalized(model, x, target_class_idx=target, use_logit=True, **kw)
    return E.to_2d(R[0])


def compute(name, model, explainer, xs, targets):
    if name.startswith('TFRP'):
        return np.stack([tfrp_map(model, x[None], int(t), name.endswith('composite')) for x, t in zip(xs, targets)])
    y = tf.keras.utils.to_categorical(targets, 1000)
    maps = []
    for b in range(0, len(xs), 16):
        maps.extend(E.to_2d(m) for m in np.asarray(explainer(xs[b:b + 16], y[b:b + 16])))
    return np.stack(maps)


def job_list(name, ev, sv):
    """(kind, subset position, draw) in the order the merged arrays are assembled."""
    jobs = [('eval', j, -1) for j in range(len(ev))]
    if name not in NO_SENSITIVITY:
        jobs += [('sensbase', j, -1) for j in range(len(sv))]
        jobs += [('sens', j, d) for j in range(len(sv)) for d in range(E.SENS_DRAWS)]
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True)
    ap.add_argument('--chunk', type=int, default=100)
    ap.add_argument('--threads', type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    name = args.method
    if E.exists(name, 'eval') and (name in NO_SENSITIVITY or E.exists(name, 'sens')):
        print(f'{name}: METHOD DONE', flush=True)
        return

    ev, sv = E.eval_indices(), E.sens_indices()
    jobs = job_list(name, ev, sv)
    part_dir = os.path.join(E.maps_dir(), 'partial')
    os.makedirs(part_dir, exist_ok=True)
    prefix = E.safe(name) + '__part_'
    parts = sorted(f for f in os.listdir(part_dir) if f.startswith(prefix))
    start = sum(np.load(os.path.join(part_dir, f), mmap_mode='r').shape[0] for f in parts)

    model = tf.keras.applications.ResNet50(weights='imagenet')
    logit_model = tf.keras.applications.ResNet50(weights='imagenet', classifier_activation=None)
    t_path = os.path.join(E.maps_dir(), 'targets.npz')
    if os.path.exists(t_path):
        with np.load(t_path) as z:
            t_ev, t_sv = z['eval'], z['sens']
    else:
        t_ev = logit_model.predict(np.concatenate([P.model_input(E.MODEL, i) for i in ev]), batch_size=32, verbose=0).argmax(1)
        t_sv = logit_model.predict(np.concatenate([P.model_input(E.MODEL, i) for i in sv]), batch_size=32, verbose=0).argmax(1)
        np.savez(t_path, eval=t_ev, sens=t_sv)

    batch = jobs[start:start + args.chunk]
    xs, ts = [], []
    for kind, j, d in batch:
        if kind == 'eval':
            xs.append(P.model_input(E.MODEL, ev[j])[0]); ts.append(t_ev[j])
        else:
            x = P.model_input(E.MODEL, sv[j])[0]
            xs.append(x + E.perturbation(sv[j], d, x.shape) if kind == 'sens' else x); ts.append(t_sv[j])
    explainer = None if name.startswith('TFRP') else build_explainer(name, logit_model)
    t0 = time.time()
    maps = compute(name, model, explainer, np.stack(xs).astype(np.float32), np.array(ts))
    sec = (time.time() - t0) / len(batch)
    np.save(os.path.join(part_dir, f'{prefix}{start:05d}.npy'), maps.astype(np.float16))
    time_csv = os.path.join(ROOT, 'exp_result', 'E4', 'maps_time_tf.csv')
    os.makedirs(os.path.dirname(time_csv), exist_ok=True)
    pd.DataFrame([dict(method=name, start=start, n=len(batch), sec_per_map=sec, threads=args.threads)]).to_csv(
        time_csv, mode='a', header=not os.path.exists(time_csv), index=False)
    done = start + len(batch)
    print(f'{name:20s} maps {start}-{done - 1} ({done}/{len(jobs)})  {sec:.2f}s/map', flush=True)
    if done < len(jobs):
        return

    parts = sorted(f for f in os.listdir(part_dir) if f.startswith(prefix))
    allmaps = np.concatenate([np.load(os.path.join(part_dir, f)) for f in parts])
    n_ev, n_sv = len(ev), len(sv)
    E.save(name, 'eval', allmaps[:n_ev])
    if name not in NO_SENSITIVITY:
        E.save(name, 'sensbase', allmaps[n_ev:n_ev + n_sv])
        E.save(name, 'sens', allmaps[n_ev + n_sv:].reshape((n_sv, E.SENS_DRAWS) + allmaps.shape[1:]))
    for f in parts:
        os.remove(os.path.join(part_dir, f))
    print(f'{name}: METHOD DONE', flush=True)


if __name__ == '__main__':
    main()
