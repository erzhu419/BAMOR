#!/usr/bin/env bash
# Python wrapper for scheduler CPU-pool nodes whose local envs may miss MuJoCo deps.
set -euo pipefail

PY_BASE="${BAMOR_REMOTE_PY:-/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python}"
WHEEL_DIR="${BAMOR_WHEEL_DIR:-/home/zhengliang01/scheduleurm_work/env_packs/bamor_mujoco_wheels}"

if ! "$PY_BASE" - <<'PY' >/dev/null 2>&1
import gymnasium, mo_gymnasium, mujoco
PY
then
  echo "[run_bamor_remote_python] installing missing MuJoCo deps from $WHEEL_DIR" >&2
  "$PY_BASE" -m pip install --no-index --no-deps --find-links "$WHEEL_DIR" \
    gymnasium mo-gymnasium mujoco \
    absl-py etils glfw pyopengl fsspec imageio importlib_resources zipp \
    typing-extensions cloudpickle farama-notifications pygame pillow scipy
fi

exec "$PY_BASE" "$@"
