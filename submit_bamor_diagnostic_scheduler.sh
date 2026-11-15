#!/usr/bin/env bash
# Submit the BAMOR diagnostic matrix as independent scheduler tasks.
#
# Each method/seed pair is one CPU-only task placed by scheduler by default.
# Results are written to per-task subdirectories to avoid concurrent writes.
#
# Example:
#   TOTAL_STEPS=100000 SEEDS=20 bash submit_bamor_diagnostic_scheduler.sh
#   METHODS="uniform bamor_ns bamor_ns_eps" SEEDS=5 DRY_RUN=1 bash submit_bamor_diagnostic_scheduler.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEDULER="${SCHEDULER:-/home/erzhu419/.claude/skills/scheduler/scheduler.py}"
SCHEDULER_PY="${SCHEDULER_PY:-python3}"
PY="${PY:-/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python}"
SCHEDULER_CWD="${SCHEDULER_CWD:-$ROOT_DIR}"

TOTAL_STEPS="${TOTAL_STEPS:-100000}"
SWITCH_INTERVAL="${SWITCH_INTERVAL:-300}"
SEED_START="${SEED_START:-0}"
SEEDS="${SEEDS:-20}"
DEVICE="${DEVICE:-cpu}"
EVAL_FREQ="${EVAL_FREQ:-0}"
CPU_PER_TASK="${CPU_PER_TASK:-8}"
RAM_MB="${RAM_MB:-8192}"
SAVE_DIR="${SAVE_DIR:-results/diagnostic_hpc_$(date +%Y%m%d_%H%M%S)}"
METHODS="${METHODS:-uniform bamor_oracle_cone bamor_oracle bamor_fixed_cone bamor bamor_ns bamor_ns_eps bamor_nc fixed}"
NODE_MODE="${NODE_MODE:-hpc}"
HPC_NODE="${HPC_NODE:-zhengliang-hpc}"
SLURM_PARTITION="${SLURM_PARTITION:-cpu}"
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
  mkdir -p "$ROOT_DIR/$SAVE_DIR"
fi

echo "BAMOR scheduler diagnostic matrix"
echo "  root           : $ROOT_DIR"
echo "  scheduler_cwd  : $SCHEDULER_CWD"
echo "  save_dir       : $SAVE_DIR"
echo "  methods        : $METHODS"
echo "  total_steps    : $TOTAL_STEPS"
echo "  switch_interval: $SWITCH_INTERVAL"
echo "  seeds          : $SEED_START..$((SEED_START + SEEDS - 1))"
echo "  device         : $DEVICE"
echo "  eval_freq      : $EVAL_FREQ"
echo "  cpu/ram        : ${CPU_PER_TASK} cores / ${RAM_MB} MB"
echo "  placement      : $NODE_MODE${NODES:+ ($NODES)}"
if [[ "$NODE_MODE" == "hpc" ]]; then
  echo "  hpc/slurm      : $HPC_NODE / ${SLURM_PARTITION:-default}"
fi
echo "  dispatch       : $DISPATCH"
echo

submitted=0
node_idx=0
for seed in $(seq "$SEED_START" "$((SEED_START + SEEDS - 1))"); do
  for method in $METHODS; do
    placement="scheduler-auto"
    node=""
    if [[ "$NODE_MODE" == "hpc" ]]; then
      node="$HPC_NODE"
      placement="hpc-slurm:${HPC_NODE}${SLURM_PARTITION:+/${SLURM_PARTITION}}"
    elif [[ "$NODE_MODE" != "auto" ]]; then
      node="${NODE_ARR[$((node_idx % ${#NODE_ARR[@]}))]}"
      node_idx=$((node_idx + 1))
      placement="${NODE_MODE}:${node}"
    fi

    task_dir="$SAVE_DIR/${method}_seed${seed}"
    abs_task_dir="$ROOT_DIR/$task_dir"
    if [[ "$DRY_RUN" != "1" ]]; then
      mkdir -p "$abs_task_dir"
    fi

    cmd="$PY train_compare_baselines.py --method $method --total_steps $TOTAL_STEPS --switch_interval $SWITCH_INTERVAL --seed $seed --seeds 1 --save_dir $task_dir --device $DEVICE --eval_freq $EVAL_FREQ && echo DONE"
    signature="BAMOR/diagnostic/${method}/steps${TOTAL_STEPS}/switch${SWITCH_INTERVAL}/eval${EVAL_FREQ}"
    desc="BAMOR diagnostic ${method} seed ${seed}"

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
      --result-dir "$abs_task_dir"
      --local-result-dir "$abs_task_dir"
      --allow-no-ckpt
      --allow-no-resume
      --allow-cpu-training
      --cpu-training-justification "BAMOR diagnostic matrix is an independent CPU-only synthetic seed sweep on zhengliang-hpc CPU nodes."
      --env "OMP_NUM_THREADS=$CPU_PER_TASK" "MKL_NUM_THREADS=$CPU_PER_TASK" "OPENBLAS_NUM_THREADS=$CPU_PER_TASK"
    )
    if [[ "$NODE_MODE" == "hpc" ]]; then
      args+=(--require-node "$node")
      if [[ -n "$SLURM_PARTITION" ]]; then
        args+=(--slurm-partition "$SLURM_PARTITION")
      fi
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

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "Dry run only; no scheduler tasks submitted."
  exit 0
fi

echo
echo "Submitted $submitted tasks."
echo "Results will accumulate under: $ROOT_DIR/$SAVE_DIR"
echo "Summarize with:"
echo "  $PY aggregate_bamor_diagnostics.py $SAVE_DIR"

if [[ "$DISPATCH" == "1" ]]; then
  echo
  echo "Dispatching scheduler queue..."
  "$SCHEDULER_PY" "$SCHEDULER" dispatch
fi
