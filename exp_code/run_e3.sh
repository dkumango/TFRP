#!/usr/bin/env bash
# E3 ablation: a fresh process for every chunk of 150 images (bounded memory), until all
# (model, variant) blocks are finalised. Resumable at any point.
# Usage: bash exp_code/run_e3.sh [threads]
set -u
cd "$(dirname "$0")/.." || exit 1
T=${1:-3}
LOG=exp_result/E3/logs
mkdir -p "$LOG"
export PYTHONIOENCODING=utf-8 TF_CPP_MIN_LOG_LEVEL=2
echo "[$(date '+%F %T')] START e3" >> "$LOG/status.txt"
for _ in $(seq 1 400); do
  ${PY_TF:-python} exp_code/e3_ablation.py --chunk 150 --threads "$T" >> "$LOG/e3.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "[$(date '+%F %T')] python rc=$rc" >> "$LOG/status.txt"; fi
  tail -n 3 "$LOG/e3.log" | grep -q "ALL BLOCKS DONE" && break
done
echo "[$(date '+%F %T')] END   e3" >> "$LOG/status.txt"
