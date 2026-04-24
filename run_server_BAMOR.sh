#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# BAMOR server launcher — run from BAMOR/ root
# Usage: bash run_server_BAMOR.sh
# Results: bamor_experiments/server_<timestamp>/
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"   # always run from BAMOR/ root

TOTAL_STEPS=${TOTAL_STEPS:-100000}
SWITCH_INTERVAL=${SWITCH_INTERVAL:-300}
MEM_PER_JOB_MB=${MEM_PER_JOB_MB:-1500}
SAFETY_MARGIN_MB=${SAFETY_MARGIN_MB:-1024}
POLL_SEC=${POLL_SEC:-20}
STAGGER_SEC=${STAGGER_SEC:-8}
PY="${PY:-python}"

NPROC=$(nproc)
DEFAULT_CPU_PAR=$(( NPROC / 2 ))
[[ $DEFAULT_CPU_PAR -gt 8 ]] && DEFAULT_CPU_PAR=8
[[ $DEFAULT_CPU_PAR -lt 1 ]] && DEFAULT_CPU_PAR=1
MAX_CPU_PARALLEL=${MAX_CPU_PARALLEL:-$DEFAULT_CPU_PAR}

read -ra METHODS <<< "${METHODS:-bamor bamor_nc bamor_ns uniform fixed}"
read -ra SEEDS   <<< "${SEEDS:-0 1 2 3 4}"

SAVE_DIR="${SAVE_DIR:-bamor_experiments/server_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$SAVE_DIR/logs"

if $PY -c "import pymoo, morl_baselines" 2>/dev/null; then
    METHODS+=(capql)
    echo "[launcher] pymoo+morl-baselines detected → including CAPQL"
fi

echo "════════════════════════════════════════════════════════════════════════"
echo "  BAMOR Server Run"
echo "════════════════════════════════════════════════════════════════════════"
echo "  save_dir        : $SAVE_DIR"
echo "  methods         : ${METHODS[*]}"
echo "  seeds           : ${SEEDS[*]}"
echo "  total_steps     : $TOTAL_STEPS"
echo "  mem_per_job_mb  : $MEM_PER_JOB_MB  (safety margin $SAFETY_MARGIN_MB)"
echo "  max_cpu_parallel: $MAX_CPU_PARALLEL  (cpu cores = $NPROC)"
echo "────────────────────────────────────────────────────────────────────────"

pick_gpu() {
    # Pick GPU with MOST free memory (≥ budget) — naturally load-balances across GPUs.
    local need=$(( MEM_PER_JOB_MB + SAFETY_MARGIN_MB ))
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | tr -d ' ' | awk -F, -v need="$need" '$2 >= need {print $2, $1}' \
        | sort -rn | awk 'NR==1 {print $2}'
}

running_count() { jobs -rp | wc -l; }

launch_one() {
    local method="$1" seed="$2" gpu="$3"
    local log="$SAVE_DIR/logs/${method}_seed${seed}.log"
    local done_marker="$SAVE_DIR/${method}_seed${seed}.json"
    if [[ -f "$done_marker" ]]; then
        echo "[skip ] $method seed=$seed (result exists)"
        return 0
    fi
    echo "[start] $method seed=$seed → gpu=$gpu"
    (
        CUDA_VISIBLE_DEVICES="$gpu" \
        $PY train_compare_baselines.py \
            --method "$method" \
            --total_steps "$TOTAL_STEPS" \
            --switch_interval "$SWITCH_INTERVAL" \
            --seed "$seed" --seeds 1 \
            --save_dir "$SAVE_DIR" \
            --device cuda \
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
            echo "[wait ] no GPU slot with ≥$((MEM_PER_JOB_MB+SAFETY_MARGIN_MB))MB free, retry in ${POLL_SEC}s"
            sleep "$POLL_SEC"; continue
        fi
        break
    done
    launch_one "$m" "$s" "$gpu"
    sleep "$STAGGER_SEC"
done

echo "[launcher] all jobs dispatched, waiting for completion..."
wait
echo "[launcher] all jobs finished."
echo "────────────────────────────────────────────────────────────────────────"

$PY - <<PY_AGG
import json, glob
from pathlib import Path
import numpy as np

save_dir = "$SAVE_DIR"
files = sorted(glob.glob(f"{save_dir}/*_seed*.json"))
if not files:
    print("  [aggregate] no result files found")
    raise SystemExit(0)

data = {}
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    key = d.get('method', Path(f).stem.rsplit('_seed', 1)[0])
    data.setdefault(key, []).append(d)

def fmt(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    return f"{a.mean():10.1f}±{a.std():.1f}" if a.size else "-"

print()
print("═" * 92)
print(f"  Final Summary (over {len(files)} runs)")
print("═" * 92)
print(f"  {'Method':<14}{'HV overall':<22}{'HV Mode0':<20}{'HV Mode1':<20}{'HV Mode2':<20}")
print("  " + "-" * 88)
for method in sorted(data):
    runs = data[method]
    overall = [r.get('hv_overall') for r in runs]
    per_mode = [[], [], []]
    for r in runs:
        hvs = r.get('hv_per_mode', []) or []
        for i in range(min(len(hvs), 3)):
            per_mode[i].append(hvs[i])
    cols = [fmt(overall)] + [fmt(x) for x in per_mode]
    print(f"  {method:<14}{cols[0]:<22}{cols[1]:<20}{cols[2]:<20}{cols[3]:<20}")
print("═" * 92)
print(f"  Results dir: {save_dir}")
PY_AGG
