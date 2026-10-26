#!/usr/bin/env bash
# E1 identical-weight comparison: three environments in parallel (3 threads each).
# Prerequisite: python exp_code/e1_prepare.py (image cache + exported models).
# Each script is resumable: completed (implementation, model, rule) blocks are skipped.
# Usage: bash exp_code/run_e1.sh [n_images]
set -u
cd "$(dirname "$0")/.." || exit 1
N=${1:-1000}
LOG=exp_result/E1/logs
mkdir -p "$LOG"
export PYTHONIOENCODING=utf-8 TF_CPP_MIN_LOG_LEVEL=2

run() {  # run <name> <python> <script>
  echo "[$(date '+%F %T')] START $1" >> "$LOG/status.txt"
  "$2" "$3" --n "$N" --threads 3 >> "$LOG/$1.log" 2>&1
  echo "[$(date '+%F %T')] END   $1 rc=$?" >> "$LOG/status.txt"
}

# TFRP: a fresh process every 200 images -- TensorFlow's CPU allocator grows by several GB
# per 1000 propagations on ResNet50, so long-lived processes exhaust the 16 GB of RAM
tfrp_loop() {
  echo "[$(date '+%F %T')] START tfrp" >> "$LOG/status.txt"
  for _ in $(seq 1 200); do
    ${PY_TF:-python} exp_code/e1_tfrp.py --n "$N" --threads 3 --one_block --chunk 200 >> "$LOG/tfrp.log" 2>&1
    tail -n 3 "$LOG/tfrp.log" | grep -q "ALL BLOCKS DONE" && break
  done
  echo "[$(date '+%F %T')] END   tfrp" >> "$LOG/status.txt"
}
tfrp_loop &
run torch        ${PY_TORCH:-python} exp_code/e1_torch.py &
# keras 2.2.4 and keras 3 cache different ResNet50 weight files under the same name in ~/.keras;
# the Python 3.7 run gets its own home so the two never overwrite each other.
USERPROFILE=${KERAS2_HOME:-$HOME/.keras2_home} HOME=${KERAS2_HOME:-$HOME/.keras2_home} run innvestigate ${PY_LEGACY:-python}        exp_code/e1_innvestigate.py &
wait
echo "[$(date '+%F %T')] ALL DONE" >> "$LOG/status.txt"
