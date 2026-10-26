#!/usr/bin/env bash
# E4 metrics: a fresh process every 40 images (memory grows over long runs), until every method
# with maps in the cache is complete. Resumable. Re-run after new maps are added.
# Usage: bash exp_code/run_e4_metrics.sh [threads]
set -u
cd "$(dirname "$0")/.." || exit 1
T=${1:-3}
LOG=exp_result/E4/logs
mkdir -p "$LOG"
export PYTHONIOENCODING=utf-8 TF_CPP_MIN_LOG_LEVEL=2
echo "[$(date '+%F %T')] START metrics" >> "$LOG/status.txt"
for _ in $(seq 1 1000); do
  ${PY_TF:-python} exp_code/e4_metrics.py --threads "$T" --max_images 40 >> "$LOG/metrics.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "[$(date '+%F %T')] metrics python rc=$rc" >> "$LOG/status.txt"; break; fi
  tail -n 2 "$LOG/metrics.log" | grep -q "ALL METRICS DONE" && break
done
echo "[$(date '+%F %T')] END   metrics" >> "$LOG/status.txt"
