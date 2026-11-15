#!/usr/bin/env bash
# Submit BAMOR diagnostic matrix as scheduler-planned CPU-parallel shards.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEDULER="${SCHEDULER:-/home/erzhu419/.claude/skills/scheduler/scheduler.py}"
SCHEDULER_PY="${SCHEDULER_PY:-python3}"
PY="${PY:-/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python}"

TOTAL_STEPS="${TOTAL_STEPS:-100000}"
SWITCH_INTERVAL="${SWITCH_INTERVAL:-300}"
SEED_START="${SEED_START:-0}"
SEEDS="${SEEDS:-20}"
DEVICE="${DEVICE:-cpu}"
EVAL_FREQ="${EVAL_FREQ:-0}"
SAVE_DIR="${SAVE_DIR:-results/diagnostic_hpc_sharded_$(date +%Y%m%d_%H%M%S)}"
METHODS="${METHODS:-uniform bamor_oracle_cone bamor_oracle bamor_fixed_cone bamor bamor_ns bamor_ns_eps bamor_nc fixed}"
NODES="${NODES:-}"
ITEM_OFFSET="${ITEM_OFFSET:-0}"
MAX_WORKERS_PER_SHARD="${MAX_WORKERS_PER_SHARD:-12}"
THREADS_PER_RUN="${THREADS_PER_RUN:-2}"
RAM_MB="${RAM_MB:-65536}"
PROJECT="${PROJECT:-BAMOR}"
PRIORITY="${PRIORITY:-normal}"
DRY_RUN="${DRY_RUN:-0}"
DISPATCH="${DISPATCH:-1}"
ALLOW_DUPLICATE="${ALLOW_DUPLICATE:-0}"

method_count=$(wc -w <<< "$METHODS")
total_items=$((method_count * SEEDS))
items=$((total_items - ITEM_OFFSET))
if (( items <= 0 )); then
  echo "No remaining items: total=$total_items offset=$ITEM_OFFSET" >&2
  exit 2
fi

run_name="$(basename "$SAVE_DIR")"
cmd_template="$PY run_bamor_diagnostic_shard.py --start {start} --end {end} --item-offset $ITEM_OFFSET --methods '$METHODS' --seed-start $SEED_START --seeds $SEEDS --total-steps $TOTAL_STEPS --switch-interval $SWITCH_INTERVAL --save-dir '$SAVE_DIR' --device $DEVICE --eval-freq $EVAL_FREQ --workers {workers} --max-workers $MAX_WORKERS_PER_SHARD --threads-per-run $THREADS_PER_RUN && echo DONE"

echo "BAMOR scheduler sharded diagnostic matrix"
echo "  save_dir       : $SAVE_DIR"
echo "  methods        : $METHODS"
echo "  total_items    : $total_items"
echo "  item_offset    : $ITEM_OFFSET"
echo "  submitted items: $items"
echo "  total_steps    : $TOTAL_STEPS"
echo "  nodes          : ${NODES:-scheduler default}"
echo "  max workers    : $MAX_WORKERS_PER_SHARD"
echo "  threads/run    : $THREADS_PER_RUN"
echo

args=(
  "$SCHEDULER" submit-cpu-batch
  --items "$items"
  --item-multiplier 1
  --cmd-template "$cmd_template"
  --cwd "$ROOT_DIR"
  --signature "BAMOR/diagnostic-shard/${run_name}/offset${ITEM_OFFSET}/{node}"
  --description "BAMOR diagnostic shard ${run_name} {node} [{start},{end})"
  --ram-mb "$RAM_MB"
  --priority "$PRIORITY"
  --project "$PROJECT"
  --env "OMP_NUM_THREADS=$THREADS_PER_RUN" "MKL_NUM_THREADS=$THREADS_PER_RUN" "OPENBLAS_NUM_THREADS=$THREADS_PER_RUN"
  --allow-cpu-training
  --cpu-training-justification "BAMOR diagnostic shard runs independent CPU-only synthetic method/seed tasks on zhengliang-hpc CPU nodes."
  --allow-no-ckpt
  --allow-no-resume
)
if [[ -n "$NODES" ]]; then
  args+=(--nodes "$NODES")
fi
if [[ "$ALLOW_DUPLICATE" == "1" ]]; then
  args+=(--allow-duplicate)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi

"$SCHEDULER_PY" "${args[@]}"

if [[ "$DRY_RUN" != "1" && "$DISPATCH" == "1" ]]; then
  "$SCHEDULER_PY" "$SCHEDULER" dispatch
fi
