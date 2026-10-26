###############################################################################
# Execution Environment: Python 3.12 (environment A, see README)  -- psutil, pandas
###############################################################################
"""E10. Execution time and memory of the LRP implementations (z+, identical networks).

Each (implementation, model) pair runs in its own worker process (exp_code/e10_worker_*.py) with the
same number of threads. The driver records the worker's resident memory when it reports 'LOADED'
(model loaded, analyzer built, 3 warm-up images) and its peak working set when it reports 'DONE'.
Reported: median and IQR of the per-image time over 50 images, peak memory, and the increase of the
peak over the loaded state (memory needed for the explanations themselves). Run on an otherwise idle
machine.

Output: exp_result/E10/efficiency.csv
Usage : python exp_code/e10_driver.py [--threads 4] [--n 50]
"""

import argparse
import json
import os
import subprocess
import sys
import time

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import psutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'exp_result', 'E10')
MODELS = ['vgg16', 'resnet50', 'inception_v3', 'mobilenet_v2']
KERAS2_HOME = os.environ.get('KERAS2_HOME', os.path.join(os.path.expanduser('~'), '.keras2_home'))
PY_TF = os.environ.get('PY_TF', 'python')          # interpreters of environments A, B, C
PY_TORCH = os.environ.get('PY_TORCH', 'python')
PY_LEGACY = os.environ.get('PY_LEGACY', 'python')


def commands(model, n, threads):
    common = ['--model', model, '--n', str(n), '--threads', str(threads)]
    return {
        'TFRP': ([PY_TF, 'exp_code/e10_worker_tf.py'] + common, {}),
        'Zennit (canonized)': ([PY_TORCH, 'exp_code/e10_worker_torch.py',
                                '--impl', 'zennit'] + common, {}),
        'Captum': ([PY_TORCH, 'exp_code/e10_worker_torch.py',
                    '--impl', 'captum'] + common, {}),
        'iNNvestigate': ([PY_LEGACY, 'exp_code/e10_worker_innvestigate.py'] + common,
                         {'USERPROFILE': KERAS2_HOME, 'HOME': KERAS2_HOME}),
    }


def descendants_peak(proc):
    """Peak working set of the worker (on Windows the venv launcher spawns the real interpreter)."""
    procs = [proc] + proc.children(recursive=True)
    vals = []
    for p in procs:
        try:
            mi = p.memory_info()
            vals.append((getattr(mi, 'peak_wset', mi.rss), mi.rss))
        except psutil.Error:
            pass
    return max(v[0] for v in vals) if vals else np.nan, max(v[1] for v in vals) if vals else np.nan


def run(impl, cmd, env_extra):
    env = dict(os.environ, PYTHONIOENCODING='utf-8', **env_extra)
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         encoding='utf-8', errors='replace')
    ps = psutil.Process(p.pid)
    loaded_rss, peak, times = np.nan, np.nan, None
    for line in p.stdout:
        line = line.strip()
        if line == 'LOADED':
            loaded_rss = descendants_peak(ps)[1]
        elif line.startswith('{'):
            times = json.loads(line)['times']
        elif line == 'DONE':
            peak = descendants_peak(ps)[0]
    err = p.stderr.read()
    p.wait()
    status = 'OK' if times else 'CRASH'
    return dict(status=status, loaded_MB=loaded_rss / 2**20, peak_MB=peak / 2**20,
                peak_increase_MB=(peak - loaded_rss) / 2**20,
                time_median_s=float(np.median(times)) if times else np.nan,
                time_q25_s=float(np.percentile(times, 25)) if times else np.nan,
                time_q75_s=float(np.percentile(times, 75)) if times else np.nan,
                error='' if times else err.strip().splitlines()[-1][:200] if err.strip() else 'no output')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--n', type=int, default=50)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for model in MODELS:
        for impl, (cmd, env_extra) in commands(model, args.n, args.threads).items():
            t0 = time.time()
            r = dict(implementation=impl, model=model, threads=args.threads, **run(impl, cmd, env_extra))
            rows.append(r)
            print(f"{model:13s} {impl:20s} {r['status']:5s} time={r['time_median_s']:.3f}s "
                  f"peak={r['peak_MB']:.0f}MB (+{r['peak_increase_MB']:.0f})  [{time.time() - t0:.0f}s] {r['error'][:60]}",
                  flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, 'efficiency.csv'), index=False)


if __name__ == '__main__':
    main()
