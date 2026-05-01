#!/bin/bash
# Local multi-env BAMOR run.  Single RTX 4060, no GPU contention.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

TOTAL_STEPS_HC=${TOTAL_STEPS_HC:-1000000}
TOTAL_STEPS_HOP=${TOTAL_STEPS_HOP:-1000000}
TOTAL_STEPS_MC=${TOTAL_STEPS_MC:-200000}
SWITCH_INTERVAL=${SWITCH_INTERVAL:-500000}
MAX_PAR=${MAX_PAR:-4}             # 4 jobs × ~1GB on 8GB GPU
STAGGER_SEC=${STAGGER_SEC:-15}
PY="${PY:-/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python}"

read -ra METHODS <<< "${METHODS:-bamor_ns uniform fixed}"
read -ra SEEDS   <<< "${SEEDS:-0 1 2}"
read -ra ENVS    <<< "${ENVS:-mo-mountaincarcontinuous-v0 mo-halfcheetah-v5 mo-hopper-v5}"
# MC first so we get fast feedback on the simplest env

SAVE_DIR="${SAVE_DIR:-results/local_multi_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$SAVE_DIR/logs"

echo "════════════════════════════════════════════════════════════════════════"
echo "  BAMOR Local Multi-Env Run (RTX 4060)"
echo "════════════════════════════════════════════════════════════════════════"
echo "  envs            : ${ENVS[*]}"
echo "  methods         : ${METHODS[*]}"
echo "  seeds           : ${SEEDS[*]}"
echo "  steps (MC/HC/H) : ${TOTAL_STEPS_MC} / ${TOTAL_STEPS_HC} / ${TOTAL_STEPS_HOP}"
echo "  switch_interval : ${SWITCH_INTERVAL}"
echo "  max_parallel    : ${MAX_PAR}"
echo "  save_dir        : $SAVE_DIR"
echo "────────────────────────────────────────────────────────────────────────"

steps_for() {
    case "$1" in
        mo-halfcheetah-*)              echo $TOTAL_STEPS_HC ;;
        mo-hopper-*)                   echo $TOTAL_STEPS_HOP ;;
        mo-mountaincarcontinuous-*)    echo $TOTAL_STEPS_MC ;;
        *)                             echo 500000 ;;
    esac
}

running_count() { jobs -rp | wc -l; }

launch_one() {
    local env="$1" method="$2" seed="$3"
    local env_safe=$(echo $env | tr '-' '_')
    local log="$SAVE_DIR/logs/${env_safe}_${method}_seed${seed}.log"
    local done_marker="$SAVE_DIR/${env_safe}/${method}_seed${seed}.json"
    if [[ -f "$done_marker" ]]; then
        echo "[skip ] $env $method seed=$seed (result exists)"
        return 0
    fi
    local steps=$(steps_for "$env")
    echo "[start] $env $method seed=$seed steps=$steps"
    (
        CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
        $PY -u train_bamor_mujoco.py \
            --env "$env" --method "$method" \
            --total_steps "$steps" \
            --switch_interval "$SWITCH_INTERVAL" \
            --seed "$seed" --num_seeds 1 \
            --save_dir "$SAVE_DIR" --device cuda \
            > "$log" 2>&1
        echo "[done ] $env $method seed=$seed" >> "$SAVE_DIR/logs/_progress.log"
    ) &
}

JOBS=()
for e in "${ENVS[@]}"; do
    for m in "${METHODS[@]}"; do
        for s in "${SEEDS[@]}"; do
            JOBS+=("$e $m $s")
        done
    done
done
echo "[launcher] total jobs: ${#JOBS[@]}"
echo "────────────────────────────────────────────────────────────────────────"

for job in "${JOBS[@]}"; do
    read -r env m s <<< "$job"
    while [[ $(running_count) -ge $MAX_PAR ]]; do
        sleep 30
    done
    launch_one "$env" "$m" "$s"
    sleep "$STAGGER_SEC"
done

echo "[launcher] all jobs dispatched, waiting..."
wait
echo "[launcher] all jobs finished. Save dir: $SAVE_DIR"
