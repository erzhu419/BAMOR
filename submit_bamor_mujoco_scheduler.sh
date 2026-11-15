#!/usr/bin/env bash
# Submit a BAMOR MuJoCo failed-environment transfer matrix through scheduler.
#
# Each (env, method, seed) tuple is an independent CPU task.  Use this after
# running stage_bamor_mujoco_hpc.sh so the remote cwd has code and dependencies.
#
# Example:
#   ENV=mo-halfcheetah-v5 TOTAL_STEPS=50000 SEEDS=3 bash submit_bamor_mujoco_scheduler.sh
#   ENVS="mo-halfcheetah-v5 mo-hopper-v5" SEEDS=5 DRY_RUN=1 bash submit_bamor_mujoco_scheduler.sh
set -euo pipefail

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_ROOT="${REMOTE_ROOT:-/home/zhengliang01/scheduleurm_work/BAMOR}"
SCHEDULER="${SCHEDULER:-/home/erzhu419/.claude/skills/scheduler/scheduler.py}"
SCHEDULER_PY="${SCHEDULER_PY:-python3}"
PY="${PY:-/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python}"
SCHEDULER_CWD="${SCHEDULER_CWD:-$LOCAL_ROOT}"

TOTAL_STEPS="${TOTAL_STEPS:-50000}"
SWITCH_INTERVAL="${SWITCH_INTERVAL:-25000}"
SEED_START="${SEED_START:-0}"
SEEDS="${SEEDS:-3}"
SEED_LIST="${SEED_LIST:-}"
DEVICE="${DEVICE:-cpu}"
HIDDEN="${HIDDEN:-64}"
EVAL_FREQ="${EVAL_FREQ:-$TOTAL_STEPS}"
CPU_PER_TASK="${CPU_PER_TASK:-8}"
RAM_MB="${RAM_MB:-8192}"
SAVE_DIR="${SAVE_DIR:-results/mujoco_failed_transfer_$(date +%Y%m%d_%H%M%S)}"
METHODS="${METHODS:-uniform bamor bamor_v3_contrast_mix bamor_v4_contrast_expert_mix bamor_v5_contrast_moe_template_mix bamor_v6_contrast_moe_head_template_mix}"
ENVS="${ENVS:-${ENV:-mo-halfcheetah-v5}}"
NODE_MODE="${NODE_MODE:-hpc}"
HPC_NODES="${HPC_NODES:-node001 node002 node003 node004 node005 node006 node007}"
NODES="${NODES:-}"
PROJECT="${PROJECT:-BAMOR}"
PRIORITY="${PRIORITY:-normal}"
DRY_RUN="${DRY_RUN:-0}"
DISPATCH="${DISPATCH:-1}"
ALLOW_DUPLICATE="${ALLOW_DUPLICATE:-0}"

if [[ ! -f "$SCHEDULER" ]]; then
  echo "scheduler not found: $SCHEDULER" >&2
  exit 2
fi

case "$NODE_MODE" in
  auto|hpc|preferred|require) ;;
  *)
    echo "NODE_MODE must be one of: auto, hpc, preferred, require" >&2
    exit 2
    ;;
esac

NODE_ARR=()
if [[ "$NODE_MODE" == "preferred" || "$NODE_MODE" == "require" ]]; then
  read -r -a NODE_ARR <<< "$NODES"
  if [[ "${#NODE_ARR[@]}" -eq 0 ]]; then
    echo "NODES must contain at least one scheduler node when NODE_MODE=$NODE_MODE" >&2
    exit 2
  fi
fi

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$LOCAL_ROOT/$SAVE_DIR/scheduler_logs"
fi

echo "BAMOR MuJoCo failed-env scheduler matrix"
echo "  local_root     : $LOCAL_ROOT"
echo "  remote_root    : $REMOTE_ROOT"
echo "  scheduler_cwd  : $SCHEDULER_CWD"
echo "  save_dir       : $SAVE_DIR"
echo "  envs           : $ENVS"
echo "  methods        : $METHODS"
echo "  total_steps    : $TOTAL_STEPS"
echo "  switch_interval: $SWITCH_INTERVAL"
if [[ -n "$SEED_LIST" ]]; then
  SEED_SPEC="$SEED_LIST"
else
  SEED_SPEC="$(seq "$SEED_START" "$((SEED_START + SEEDS - 1))")"
fi
echo "  seeds          : $SEED_SPEC"
echo "  device/hidden  : $DEVICE / $HIDDEN"
echo "  eval_freq      : $EVAL_FREQ"
echo "  cpu/ram        : ${CPU_PER_TASK} cores / ${RAM_MB} MB"
echo "  placement      : $NODE_MODE${NODES:+ ($NODES)}"
if [[ "$NODE_MODE" == "hpc" ]]; then
  echo "  hpc pool       : $HPC_NODES (scheduler selects placement)"
fi
echo "  dispatch       : $DISPATCH"
echo

submitted=0
node_idx=0
for env_name in $ENVS; do
  env_safe="${env_name//-/_}"
  for seed in $SEED_SPEC; do
    for method in $METHODS; do
      placement="scheduler-auto"
      node=""
      if [[ "$NODE_MODE" == "hpc" ]]; then
        placement="hpc-pool:scheduler-auto"
      elif [[ "$NODE_MODE" != "auto" ]]; then
        node="${NODE_ARR[$((node_idx % ${#NODE_ARR[@]}))]}"
        node_idx=$((node_idx + 1))
        placement="${NODE_MODE}:${node}"
      fi

      log_dir="$SAVE_DIR/scheduler_logs/${env_safe}_${method}_seed${seed}"
      abs_log_dir="$LOCAL_ROOT/$log_dir"
      if [[ "$DRY_RUN" != "1" ]]; then
        mkdir -p "$abs_log_dir"
      fi

      cmd="$PY -u train_bamor_mujoco.py --env $env_name --method $method --total_steps $TOTAL_STEPS --switch_interval $SWITCH_INTERVAL --seed $seed --num_seeds 1 --save_dir $SAVE_DIR --eval_freq $EVAL_FREQ --device $DEVICE --hidden $HIDDEN && echo DONE"
      signature="BAMOR/mujoco/${env_name}/${method}/seed${seed}/steps${TOTAL_STEPS}/switch${SWITCH_INTERVAL}/hidden${HIDDEN}/eval-locked-full-simplex-v1"
      desc="BAMOR mujoco ${env_name} ${method} seed ${seed}"

      echo "[$((submitted + 1))] $desc -> $placement"
      echo "    $cmd"

      submitted=$((submitted + 1))
      if [[ "$DRY_RUN" == "1" ]]; then
        continue
      fi

      args=(
        "$SCHEDULER" submit
        --project "$PROJECT"
        --description "$desc"
        --cmd "$cmd"
        --cwd "$SCHEDULER_CWD"
        --signature "$signature"
        --vram 0
        --cpu "$CPU_PER_TASK"
        --ram-mb "$RAM_MB"
        --priority "$PRIORITY"
        --result-dir "$abs_log_dir"
        --local-result-dir "$abs_log_dir"
        --allow-no-ckpt
        --allow-no-resume
        --allow-cpu-training
        --cpu-training-justification "BAMOR MuJoCo failed-environment transfer sweep on CPU scheduler nodes."
        --env "OMP_NUM_THREADS=$CPU_PER_TASK" "MKL_NUM_THREADS=$CPU_PER_TASK" "OPENBLAS_NUM_THREADS=$CPU_PER_TASK"
      )
      if [[ "$NODE_MODE" == "hpc" ]]; then
        args+=(--skip-launch-staging)
        for allowed_node in $HPC_NODES; do
          args+=(--allowed-node "$allowed_node")
        done
      elif [[ "$NODE_MODE" == "preferred" ]]; then
        args+=(--preferred-node "$node")
      elif [[ "$NODE_MODE" == "require" ]]; then
        args+=(--require-node "$node")
      fi
      if [[ "$ALLOW_DUPLICATE" == "1" ]]; then
        args+=(--allow-duplicate)
      fi

      "$SCHEDULER_PY" "${args[@]}"
    done
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "Dry run only; no scheduler tasks submitted."
  exit 0
fi

echo
echo "Submitted $submitted tasks."
echo "Remote results will accumulate under: $REMOTE_ROOT/$SAVE_DIR"
echo "Local scheduler logs under: $LOCAL_ROOT/$SAVE_DIR/scheduler_logs"

if [[ "$DISPATCH" == "1" ]]; then
  echo
  echo "Dispatching scheduler queue..."
  "$SCHEDULER_PY" "$SCHEDULER" dispatch
fi
