#!/bin/bash
# Mujoco BAPR-style run: 5 methods × 5 seeds on mo-halfcheetah-v5
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/server_package/code"

TOTAL_STEPS=${TOTAL_STEPS:-300000}
SWITCH_INTERVAL=${SWITCH_INTERVAL:-30000}
ENV=${ENV:-mo-halfcheetah-v5}
MEM_PER_JOB_MB=${MEM_PER_JOB_MB:-1500}
SAFETY_MARGIN_MB=${SAFETY_MARGIN_MB:-1024}
POLL_SEC=${POLL_SEC:-30}
STAGGER_SEC=${STAGGER_SEC:-15}
PY="${PY:-python}"
MAX_CPU_PARALLEL=${MAX_CPU_PARALLEL:-8}

read -ra METHODS <<< "${METHODS:-bamor_ns bamor bamor_nc uniform fixed}"
read -ra SEEDS   <<< "${SEEDS:-0 1 2 3 4}"

SAVE_DIR="${SAVE_DIR:-../results/mujoco_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$SAVE_DIR/logs"

echo "════════════════════════════════════════════════════════════════════════"
echo "  BAMOR Mujoco Run (BAPR-style env)"
echo "════════════════════════════════════════════════════════════════════════"
echo "  env             : $ENV"
echo "  save_dir        : $SAVE_DIR"
echo "  methods         : ${METHODS[*]}"
echo "  seeds           : ${SEEDS[*]}"
echo "  total_steps     : $TOTAL_STEPS"
echo "  switch_interval : $SWITCH_INTERVAL"
echo "  max_cpu_parallel: $MAX_CPU_PARALLEL"
echo "────────────────────────────────────────────────────────────────────────"

pick_gpu() {
    local need=$(( MEM_PER_JOB_MB + SAFETY_MARGIN_MB ))
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | tr -d ' ' | awk -F, -v need="$need" '$2 >= need {print $2, $1}' \
        | sort -rn | awk 'NR==1 {print $2}'
}

running_count() { jobs -rp | wc -l; }

launch_one() {
    local method="$1" seed="$2" gpu="$3"
    local log="$SAVE_DIR/logs/${method}_seed${seed}.log"
    local done_marker="$SAVE_DIR/$(echo $ENV | tr '-' '_')/${method}_seed${seed}.json"
    if [[ -f "$done_marker" ]]; then
        echo "[skip ] $method seed=$seed (result exists)"
        return 0
    fi
    echo "[start] $method seed=$seed → gpu=$gpu"
    (
        CUDA_VISIBLE_DEVICES="$gpu" \
        $PY train_bamor_mujoco.py \
            --env "$ENV" --method "$method" \
            --total_steps "$TOTAL_STEPS" \
            --switch_interval "$SWITCH_INTERVAL" \
            --seed "$seed" --num_seeds 1 \
            --save_dir "$SAVE_DIR" --device cuda \
            > "$log" 2>&1
        echo "[done ] $method seed=$seed" >> "$SAVE_DIR/logs/_progress.log"
    ) &
}

JOBS=()
for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
        JOBS+=("$m $s")
    done
done
echo "[launcher] total jobs: ${#JOBS[@]}"
echo "────────────────────────────────────────────────────────────────────────"

for job in "${JOBS[@]}"; do
    read -r m s <<< "$job"
    while true; do
        if [[ $(running_count) -ge $MAX_CPU_PARALLEL ]]; then
            sleep "$POLL_SEC"; continue
        fi
        gpu=$(pick_gpu || true)
        if [[ -z "$gpu" ]]; then
            echo "[wait ] no GPU slot, retry in ${POLL_SEC}s"
            sleep "$POLL_SEC"; continue
        fi
        break
    done
    launch_one "$m" "$s" "$gpu"
    sleep "$STAGGER_SEC"
done

echo "[launcher] all jobs dispatched, waiting..."
wait
echo "[launcher] all jobs finished. Save dir: $SAVE_DIR"
