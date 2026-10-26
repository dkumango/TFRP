#!/usr/bin/env bash
# E13 out-of-the-box conservation study. Runs the three environments sequentially.
#   TFRP (Python 3.12, environment A) -> writes exp_result/E13/images.txt
#   Zennit + Captum (Python 3.12, environment B)
#   iNNvestigate (Python 3.7, environment C)
# Usage: bash exp_code/run_e13.sh <image_dir> [n_images]
set -u
cd "$(dirname "$0")/.." || exit 1
IMG_DIR=${1:?image_dir}
N=${2:-50}
LOG=exp_result/E13/logs
mkdir -p "$LOG"
export PYTHONIOENCODING=utf-8 TF_CPP_MIN_LOG_LEVEL=2

step() {  # step <name> <cmd...>
  local name=$1; shift
  echo "[$(date '+%F %T')] START $name" | tee -a "$LOG/status.txt"
  "$@" > "$LOG/$name.log" 2>&1
  echo "[$(date '+%F %T')] END   $name rc=$?" | tee -a "$LOG/status.txt"
}

step tfrp_keras      ${PY_TF:-python}       exp_code/e13_tfrp_keras.py --image_dir "$IMG_DIR" --n "$N"
step zennit_captum   ${PY_TORCH:-python} exp_code/e13_torch_zennit_captum.py --image_dir "$IMG_DIR"
USERPROFILE=${KERAS2_HOME:-$HOME/.keras2_home} HOME=${KERAS2_HOME:-$HOME/.keras2_home} step innvestigate    ${PY_LEGACY:-python}        exp_code/e13_innvestigate_keras2.py --image_dir "$IMG_DIR"
echo "[$(date '+%F %T')] ALL DONE" | tee -a "$LOG/status.txt"
