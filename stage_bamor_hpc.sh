#!/usr/bin/env bash
# Stage the minimal BAMOR synthetic-diagnostic code on zhengliang-hpc.
#
# Scheduler-launched entries expect /home/zhengliang01/scheduleurm_work/BAMOR
# to already exist on whichever node receives the task.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSPORT=(python3 "$ROOT_DIR/../scheduleurm/skill/transport_exec.py")
REMOTE_HOST="${REMOTE_HOST:-zhengliang01@202.197.46.16}"
PROXY_JUMP="${PROXY_JUMP:-jtl110gpu2}"
REMOTE_DIR="${REMOTE_DIR:-/home/zhengliang01/scheduleurm_work/BAMOR}"
REMOTE_PY="${REMOTE_PY:-/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python}"
WHEEL_DIR_LOCAL="${WHEEL_DIR_LOCAL:-/tmp/bamor_wheels}"
WHEEL_DIR_REMOTE="${WHEEL_DIR_REMOTE:-/home/zhengliang01/scheduleurm_work/env_packs/bamor_wheels}"
INSTALL_GYMNASIUM="${INSTALL_GYMNASIUM:-1}"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -J "$PROXY_JUMP")
SSH=("${TRANSPORT[@]}" -- ssh "${SSH_OPTS[@]}" "$REMOTE_HOST")
RSYNC_RSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -J $PROXY_JUMP"

echo "Staging BAMOR to $REMOTE_HOST:$REMOTE_DIR"

"${SSH[@]}" "mkdir -p '$REMOTE_DIR' '$WHEEL_DIR_REMOTE'"

if [[ "$INSTALL_GYMNASIUM" == "1" ]]; then
  if ! "${SSH[@]}" "$REMOTE_PY - <<'PY'
import gymnasium
print(gymnasium.__version__)
PY" >/dev/null 2>&1; then
    echo "Preparing local gymnasium wheelhouse..."
    python3 -m pip download --only-binary=:all: --dest "$WHEEL_DIR_LOCAL" gymnasium==1.2.0
    "${TRANSPORT[@]}" --bulk -- rsync -az --delete -e "$RSYNC_RSH" "$WHEEL_DIR_LOCAL"/ "$REMOTE_HOST:$WHEEL_DIR_REMOTE"/
    "${SSH[@]}" "$REMOTE_PY -m pip install --no-index --find-links '$WHEEL_DIR_REMOTE' gymnasium==1.2.0"
  fi
fi

"${TRANSPORT[@]}" --bulk -- rsync -az -e "$RSYNC_RSH" \
  "$ROOT_DIR/train_compare_baselines.py" \
  "$ROOT_DIR/train_synthetic_bamor.py" \
  "$ROOT_DIR/synthetic_mo_env.py" \
  "$ROOT_DIR/bamor_core.py" \
  "$ROOT_DIR/aggregate_bamor_diagnostics.py" \
  "$ROOT_DIR/run_bamor_diagnostic_shard.py" \
  "$ROOT_DIR/run_bamor_diagnostic_matrix.sh" \
  "$ROOT_DIR/submit_bamor_diagnostic_scheduler.sh" \
  "$ROOT_DIR/submit_bamor_diagnostic_shards.sh" \
  "$ROOT_DIR/pull_bamor_hpc_results.sh" \
  "$REMOTE_HOST:$REMOTE_DIR"/

"${SSH[@]}" "cd '$REMOTE_DIR' && $REMOTE_PY - <<'PY'
import numpy, torch, gymnasium
import train_compare_baselines
print('remote preflight ok', numpy.__version__, torch.__version__, gymnasium.__version__)
PY"

echo "Done."
