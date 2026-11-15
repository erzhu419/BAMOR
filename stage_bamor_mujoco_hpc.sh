#!/usr/bin/env bash
# Stage BAMOR MuJoCo-transfer code and dependencies on zhengliang-hpc.
#
# The scheduler nodes expect the working tree to exist at REMOTE_DIR.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSPORT=(python3 "$ROOT_DIR/../scheduleurm/skill/transport_exec.py")
REMOTE_HOST="${REMOTE_HOST:-zhengliang01@202.197.46.16}"
PROXY_JUMP="${PROXY_JUMP:-jtl110gpu2}"
REMOTE_DIR="${REMOTE_DIR:-/home/zhengliang01/scheduleurm_work/BAMOR}"
REMOTE_PY="${REMOTE_PY:-/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python}"
MO_GYMNASIUM_VERSION="${MO_GYMNASIUM_VERSION:-1.3.2}"
MUJOCO_VERSION="${MUJOCO_VERSION:-3.3.0}"
PYMOO_VERSION="${PYMOO_VERSION:-0.6.1.5}"
WHEEL_DIR_LOCAL="${WHEEL_DIR_LOCAL:-/tmp/bamor_mujoco_wheels}"
WHEEL_DIR_REMOTE="${WHEEL_DIR_REMOTE:-/home/zhengliang01/scheduleurm_work/env_packs/bamor_mujoco_wheels}"
MORL_BASELINES_LOCAL="${MORL_BASELINES_LOCAL:-$ROOT_DIR/../morl-baselines}"
MORL_BASELINES_REMOTE="${MORL_BASELINES_REMOTE:-$REMOTE_DIR/morl-baselines}"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -J "$PROXY_JUMP")
SSH=("${TRANSPORT[@]}" -- ssh "${SSH_OPTS[@]}" "$REMOTE_HOST")
RSYNC_RSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -J $PROXY_JUMP"

echo "Staging BAMOR MuJoCo code to $REMOTE_HOST:$REMOTE_DIR"

"${SSH[@]}" "mkdir -p '$REMOTE_DIR' '$WHEEL_DIR_REMOTE'"

if ! "${SSH[@]}" "$REMOTE_PY - <<'PY'
import imageio, mo_gymnasium, mujoco
print(mo_gymnasium.__version__, mujoco.__version__, imageio.__version__)
PY" >/dev/null 2>&1; then
  echo "Preparing local MuJoCo wheelhouse..."
  rm -rf "$WHEEL_DIR_LOCAL"
  mkdir -p "$WHEEL_DIR_LOCAL"
  python3 -m pip download --only-binary=:all: \
    --platform manylinux2014_x86_64 \
    --python-version 310 \
    --implementation cp \
    --abi cp310 \
    --dest "$WHEEL_DIR_LOCAL" \
    "mo-gymnasium==$MO_GYMNASIUM_VERSION" \
    "mujoco==$MUJOCO_VERSION" \
    "absl-py" \
    "etils[epath]" \
    "glfw" \
    "pyopengl" \
    "fsspec" \
    "imageio" \
    "importlib_resources" \
    "zipp" \
    "typing-extensions"
  "${TRANSPORT[@]}" --bulk -- rsync -az --delete -e "$RSYNC_RSH" "$WHEEL_DIR_LOCAL"/ "$REMOTE_HOST:$WHEEL_DIR_REMOTE"/
  echo "Installing remote MuJoCo dependencies from wheelhouse..."
  "${SSH[@]}" "$REMOTE_PY -m pip install --no-index --no-deps --find-links '$WHEEL_DIR_REMOTE' 'mo-gymnasium==$MO_GYMNASIUM_VERSION' 'mujoco==$MUJOCO_VERSION' absl-py etils glfw pyopengl fsspec imageio importlib_resources zipp typing-extensions"
fi

if ! "${SSH[@]}" "$REMOTE_PY - <<'PY'
import pymoo, autograd, cma, alive_progress, dill, deprecated
print(pymoo.__version__)
PY" >/dev/null 2>&1; then
  echo "Preparing local pymoo wheelhouse..."
  mkdir -p "$WHEEL_DIR_LOCAL"
  python3 -m pip download --only-binary=:all: \
    --no-deps \
    --platform manylinux2014_x86_64 \
    --python-version 310 \
    --implementation cp \
    --abi cp310 \
    --dest "$WHEEL_DIR_LOCAL" \
    "pymoo==$PYMOO_VERSION" \
    "autograd" \
    "cma" \
    "alive-progress" \
    "about-time" \
    "graphemeu" \
    "dill" \
    "Deprecated" \
    "wrapt"
  "${TRANSPORT[@]}" --bulk -- rsync -az -e "$RSYNC_RSH" "$WHEEL_DIR_LOCAL"/ "$REMOTE_HOST:$WHEEL_DIR_REMOTE"/
  echo "Installing remote pymoo dependencies from wheelhouse..."
  "${SSH[@]}" "$REMOTE_PY -m pip install --no-index --no-deps --find-links '$WHEEL_DIR_REMOTE' 'pymoo==$PYMOO_VERSION' autograd cma alive-progress about-time graphemeu dill Deprecated wrapt"
fi

"${TRANSPORT[@]}" --bulk -- rsync -az -e "$RSYNC_RSH" \
  "$ROOT_DIR/train_bamor_mujoco.py" \
  "$ROOT_DIR/piecewise_mo_env.py" \
  "$ROOT_DIR/train_compare_baselines.py" \
  "$ROOT_DIR/train_synthetic_bamor.py" \
  "$ROOT_DIR/synthetic_mo_env.py" \
  "$ROOT_DIR/bamor_core.py" \
  "$ROOT_DIR/run_bamor_mujoco_seed_shard.py" \
  "$ROOT_DIR/aggregate_bamor_diagnostics.py" \
  "$ROOT_DIR/submit_bamor_mujoco_scheduler.sh" \
  "$REMOTE_HOST:$REMOTE_DIR"/

if [[ -d "$MORL_BASELINES_LOCAL/morl_baselines" ]]; then
  echo "Staging morl-baselines package to $REMOTE_HOST:$MORL_BASELINES_REMOTE"
  "${SSH[@]}" "mkdir -p '$MORL_BASELINES_REMOTE'"
  "${TRANSPORT[@]}" --bulk -- rsync -az --delete \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    -e "$RSYNC_RSH" \
    "$MORL_BASELINES_LOCAL/morl_baselines" \
    "$MORL_BASELINES_LOCAL/setup.py" \
    "$MORL_BASELINES_LOCAL/pyproject.toml" \
    "$REMOTE_HOST:$MORL_BASELINES_REMOTE"/
else
  echo "Warning: morl-baselines source not found at $MORL_BASELINES_LOCAL; CAPQL tasks will be skipped." >&2
fi

"${SSH[@]}" "cd '$REMOTE_DIR' && $REMOTE_PY - <<'PY'
import numpy, torch, gymnasium, mo_gymnasium, mujoco
from piecewise_mo_env import make_piecewise_mo_env
from train_bamor_mujoco import ensure_wandb_stub

env = make_piecewise_mo_env('mo-halfcheetah-v5', switch_interval=1000, seed=0)
obs, info = env.reset()
action = env.action_space.sample()
next_obs, reward, terminated, truncated, info = env.step(action)
env.close()
ensure_wandb_stub()
from morl_baselines.multi_policy.capql.capql import CAPQL
print(
    'remote mujoco preflight ok',
    'numpy', numpy.__version__,
    'torch', torch.__version__,
    'gymnasium', gymnasium.__version__,
    'mo_gymnasium', mo_gymnasium.__version__,
    'mujoco', mujoco.__version__,
    'capql', CAPQL.__name__,
    'obs_dim', len(obs),
    'reward_dim', len(reward),
)
PY"

echo "Done."
