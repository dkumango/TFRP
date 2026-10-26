#!/usr/bin/env bash
# E4 TensorFlow attribution maps: one process per chunk of maps (bounded memory), method by method.
# Resumable. Usage: bash exp_code/run_e4_maps_tf.sh [threads] [chunk] ["method,method,..."]
set -u
cd "$(dirname "$0")/.." || exit 1
T=${1:-3}
CHUNK=${2:-100}
METHODS=${3:-"TFRP z+,TFRP composite,Saliency,GuidedBackprop,GradCAM,GradCAMPP,SmoothGrad,IntegratedGradients,RISE"}
LOG=exp_result/E4/logs
mkdir -p "$LOG"
export PYTHONIOENCODING=utf-8 TF_CPP_MIN_LOG_LEVEL=2
IFS=',' read -ra LIST <<< "$METHODS"
for m in "${LIST[@]}"; do
  echo "[$(date '+%F %T')] START maps $m" >> "$LOG/status.txt"
  for _ in $(seq 1 100); do
    ${PY_TF:-python} exp_code/e4_maps_tf.py --method "$m" --chunk "$CHUNK" --threads "$T" >> "$LOG/maps_tf.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then echo "[$(date '+%F %T')] $m python rc=$rc" >> "$LOG/status.txt"; break; fi
    tail -n 2 "$LOG/maps_tf.log" | grep -q "$m: METHOD DONE" && break
  done
  echo "[$(date '+%F %T')] END   maps $m" >> "$LOG/status.txt"
done
