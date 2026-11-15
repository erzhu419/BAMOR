#!/usr/bin/env bash
# Run the BAMOR diagnostic matrix on the synthetic benchmark.
#
# Defaults are intentionally conservative and sequential. Override from the
# shell for larger runs, for example:
#   TOTAL_STEPS=100000 SEEDS=20 DEVICE=cuda bash run_bamor_diagnostic_matrix.sh
set -euo pipefail

PY="${PY:-python3}"
TOTAL_STEPS="${TOTAL_STEPS:-100000}"
SWITCH_INTERVAL="${SWITCH_INTERVAL:-300}"
SEED_START="${SEED_START:-0}"
SEEDS="${SEEDS:-5}"
DEVICE="${DEVICE:-cpu}"
SAVE_DIR="${SAVE_DIR:-results/diagnostic_$(date +%Y%m%d_%H%M%S)}"
EVAL_FREQ="${EVAL_FREQ:-0}"

METHODS="${METHODS:-uniform bamor_oracle_cone bamor_oracle bamor_fixed_cone bamor bamor_ns bamor_ns_eps bamor_nc fixed}"

mkdir -p "$SAVE_DIR"

echo "BAMOR diagnostic matrix"
echo "  save_dir       : $SAVE_DIR"
echo "  methods        : $METHODS"
echo "  total_steps    : $TOTAL_STEPS"
echo "  switch_interval: $SWITCH_INTERVAL"
echo "  seeds          : $SEED_START..$((SEED_START + SEEDS - 1))"
echo "  device         : $DEVICE"
echo "  eval_freq      : $EVAL_FREQ"
echo

for method in $METHODS; do
  "$PY" train_compare_baselines.py \
    --method "$method" \
    --total_steps "$TOTAL_STEPS" \
    --switch_interval "$SWITCH_INTERVAL" \
    --seed "$SEED_START" \
    --seeds "$SEEDS" \
    --save_dir "$SAVE_DIR" \
    --device "$DEVICE" \
    --eval_freq "$EVAL_FREQ"
done

"$PY" - "$SAVE_DIR" <<'PY'
import glob
import json
import math
import statistics as stats
import sys
from collections import defaultdict

save_dir = sys.argv[1]
rows = defaultdict(list)
for path in glob.glob(f"{save_dir}/*_seed*.json"):
    with open(path) as fh:
        run = json.load(fh)
    rows[run["method"]].append(run)

def fmt(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return "-"
    mean = stats.mean(values) / 1000.0
    sd = (stats.pstdev(values) if len(values) > 1 else 0.0) / 1000.0
    return f"{mean:.1f}+/-{sd:.1f}"

print()
print("Summary, HV x 1e3")
print(f"{'method':<20} {'n':>3} {'overall':>16} {'mode0':>16} {'mode1':>16} {'mode2':>16}")
for method in sorted(rows):
    runs = rows[method]
    overall = [r.get("hv_overall") for r in runs]
    per_mode = [[], [], []]
    for r in runs:
        hvs = r.get("hv_per_mode") or []
        for i in range(min(3, len(hvs))):
            per_mode[i].append(hvs[i])
    print(
        f"{method:<20} {len(runs):>3} {fmt(overall):>16} "
        f"{fmt(per_mode[0]):>16} {fmt(per_mode[1]):>16} {fmt(per_mode[2]):>16}"
    )
PY
