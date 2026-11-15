#!/bin/bash
# Multi-env mujoco run — 3 envs × 3 methods × 3 seeds = 27 jobs in a single
# queue at 6 parallel.  Each (env, method, seed) triple gets its own log.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/server_package/code"

TOTAL_STEPS_HC=${TOTAL_STEPS_HC:-1000000}        # halfcheetah
TOTAL_STEPS_HOP=${TOTAL_STEPS_HOP:-1000000}      # hopper
TOTAL_STEPS_MC=${TOTAL_STEPS_MC:-200000}         # mountaincar (much simpler)
SWITCH_INTERVAL=${SWITCH_INTERVAL:-500000}       # slow switching
MEM_PER_JOB_MB=${MEM_PER_JOB_MB:-1500}
SAFETY_MARGIN_MB=${SAFETY_MARGIN_MB:-1024}
POLL_SEC=${POLL_SEC:-30}
STAGGER_SEC=${STAGGER_SEC:-15}
MAX_CPU_PARALLEL=${MAX_CPU_PARALLEL:-6}
PY="${PY:-python}"

read -ra METHODS <<< "${METHODS:-bamor_ns uniform fixed}"
read -ra SEEDS   <<< "${SEEDS:-0 1 2}"
read -ra ENVS    <<< "${ENVS:-mo-halfcheetah-v5 mo-hopper-v5 mo-mountaincarcontinuous-v0}"

SAVE_DIR="${SAVE_DIR:-../results/multi_env_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$SAVE_DIR/logs"

echo "════════════════════════════════════════════════════════════════════════"
echo "  BAMOR Multi-Env Mujoco Run"
echo "════════════════════════════════════════════════════════════════════════"
echo "  envs            : ${ENVS[*]}"
echo "  methods         : ${METHODS[*]}"
echo "  seeds           : ${SEEDS[*]}"
echo "  steps (HC/Hop/MC): ${TOTAL_STEPS_HC} / ${TOTAL_STEPS_HOP} / ${TOTAL_STEPS_MC}"
echo "  switch_interval : ${SWITCH_INTERVAL}"
echo "  max_cpu_parallel: ${MAX_CPU_PARALLEL}"
echo "  save_dir        : $SAVE_DIR"
echo "────────────────────────────────────────────────────────────────────────"

_RR_COUNTER=0
pick_gpu() {
    local need=$(( MEM_PER_JOB_MB + SAFETY_MARGIN_MB ))
    local gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    local gpu_idx=$(( _RR_COUNTER % gpu_count ))
    local free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=$gpu_idx | tr -d ' ')
    if [[ -z "$free_mb" ]] || [[ "$free_mb" -lt "$need" ]]; then
        for i in $(seq 0 $((gpu_count-1))); do
            free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=$i | tr -d ' ')
            if [[ -n "$free_mb" && "$free_mb" -ge "$need" ]]; then gpu_idx=$i; break; fi
        done
    fi
    echo "$gpu_idx"
}
running_count() { jobs -rp | wc -l; }

steps_for() {
    case "$1" in
        mo-halfcheetah-*)              echo $TOTAL_STEPS_HC ;;
        mo-hopper-*)                   echo $TOTAL_STEPS_HOP ;;
        mo-mountaincarcontinuous-*)    echo $TOTAL_STEPS_MC ;;
        *)                             echo 500000 ;;
    esac
}

launch_one() {
    local env="$1" method="$2" seed="$3" gpu="$4"
    local env_safe=$(echo $env | tr '-' '_')
    local log="$SAVE_DIR/logs/${env_safe}_${method}_seed${seed}.log"
    local done_marker="$SAVE_DIR/${env_safe}/${method}_seed${seed}.json"
    if [[ -f "$done_marker" ]]; then
        echo "[skip ] $env $method seed=$seed (result exists)"
        return 0
    fi
    local steps=$(steps_for "$env")
    echo "[start] $env $method seed=$seed steps=$steps → gpu=$gpu"
    (
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
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
    launch_one "$env" "$m" "$s" "$gpu"
    _RR_COUNTER=$(( _RR_COUNTER + 1 ))
    sleep "$STAGGER_SEC"
done

echo "[launcher] all 27 jobs dispatched, waiting..."
wait
echo "[launcher] all jobs finished. Save dir: $SAVE_DIR"
