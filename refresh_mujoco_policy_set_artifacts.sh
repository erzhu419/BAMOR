#!/usr/bin/env bash
# Refresh the 3-D MuJoCo policy-set summary, seed diagnostics, figure, and paper.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
BUILD_PAPER="${BUILD_PAPER:-1}"
PRESETS="${PRESETS:-ant10 hopper10}"
TAG="${TAG:-3d10}"

preset_args=()
for preset in $PRESETS; do
  preset_args+=(--preset "$preset")
done

"$PY" "$ROOT_DIR/aggregate_mujoco_policy_set_union.py" \
  "${preset_args[@]}" \
  --write-json "$ROOT_DIR/results/mujoco_policy_set_union_${TAG}_summary.json" \
  --write-md "$ROOT_DIR/results/mujoco_policy_set_union_${TAG}_summary.md" \
  --write-seed-csv "$ROOT_DIR/results/mujoco_policy_set_union_${TAG}_seed_diagnostics.csv"

"$PY" "$ROOT_DIR/paper/make_mujoco_policy_set_figure.py"

if [[ "$BUILD_PAPER" == "1" ]]; then
  (
    cd "$ROOT_DIR/paper"
    pdflatex -interaction=nonstopmode paper.tex
    pdflatex -interaction=nonstopmode paper.tex
  )
fi

echo "Refreshed MuJoCo policy-set artifacts."
echo "  presets: $PRESETS"
echo "  results/mujoco_policy_set_union_${TAG}_summary.json"
echo "  results/mujoco_policy_set_union_${TAG}_summary.md"
echo "  results/mujoco_policy_set_union_${TAG}_seed_diagnostics.csv"
echo "  paper/fig_mujoco_policy_set_gain.pdf"
if [[ "$BUILD_PAPER" == "1" ]]; then
  echo "  paper/paper.pdf"
fi
