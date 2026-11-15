#!/usr/bin/env bash
# Pull BAMOR scheduler results from zhengliang-hpc back to the local workspace.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSPORT=(python3 "$ROOT_DIR/../scheduleurm/skill/transport_exec.py")
REMOTE_HOST="${REMOTE_HOST:-zhengliang01@202.197.46.16}"
PROXY_JUMP="${PROXY_JUMP:-jtl110gpu2}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/zhengliang01/scheduleurm_work/BAMOR}"

SAVE_DIR="${1:-${SAVE_DIR:-}}"
if [[ -z "$SAVE_DIR" ]]; then
  echo "usage: $0 <results/diagnostic_dir>" >&2
  exit 2
fi

mkdir -p "$ROOT_DIR/$SAVE_DIR"
"${TRANSPORT[@]}" --bulk -- rsync -az -e "ssh -o BatchMode=yes -o ConnectTimeout=10 -J $PROXY_JUMP" \
  "$REMOTE_HOST:$REMOTE_ROOT/$SAVE_DIR"/ \
  "$ROOT_DIR/$SAVE_DIR"/

echo "Pulled: $ROOT_DIR/$SAVE_DIR"
