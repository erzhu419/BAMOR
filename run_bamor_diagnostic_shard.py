#!/usr/bin/env python3
"""Run a shard of the BAMOR method/seed diagnostic matrix on one CPU node."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_METHODS = (
    "uniform bamor_oracle_cone bamor_oracle bamor_fixed_cone "
    "bamor bamor_ns bamor_ns_eps bamor_nc fixed"
)


def build_items(methods: list[str], seed_start: int, seeds: int) -> list[tuple[str, int]]:
    items = []
    for seed in range(seed_start, seed_start + seeds):
        for method in methods:
            items.append((method, seed))
    return items


def run_one(args, item_idx: int, method: str, seed: int) -> tuple[int, int]:
    save_dir = Path(args.save_dir) / f"{method}_seed{seed}"
    output_json = save_dir / f"{method}_seed{seed}.json"
    log_dir = Path(args.save_dir) / "_shard_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and output_json.exists() and output_json.stat().st_size > 0:
        print(f"[skip] item={item_idx} {method} seed={seed} -> {output_json}", flush=True)
        return item_idx, 0

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads_per_run)
    env["MKL_NUM_THREADS"] = str(args.threads_per_run)
    env["OPENBLAS_NUM_THREADS"] = str(args.threads_per_run)
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        "train_compare_baselines.py",
        "--method", method,
        "--total_steps", str(args.total_steps),
        "--switch_interval", str(args.switch_interval),
        "--seed", str(seed),
        "--seeds", "1",
        "--save_dir", str(save_dir),
        "--device", args.device,
        "--eval_freq", str(args.eval_freq),
    ]
    log_path = log_dir / f"item{item_idx:04d}_{method}_seed{seed}.log"
    print(f"[run] item={item_idx} {method} seed={seed} log={log_path}", flush=True)
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f"[done] item={item_idx} {method} seed={seed} rc={proc.returncode}", flush=True)
    return item_idx, proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True,
                        help="Shard-local start index, inclusive")
    parser.add_argument("--end", type=int, required=True,
                        help="Shard-local end index, exclusive")
    parser.add_argument("--item-offset", type=int, default=0,
                        help="Global item offset added to --start/--end")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--switch-interval", type=int, default=300)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-freq", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--threads-per-run", type=int, default=2)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction,
                        default=True)
    args = parser.parse_args()

    methods = args.methods.split()
    items = build_items(methods, args.seed_start, args.seeds)
    global_start = args.item_offset + args.start
    global_end = args.item_offset + args.end
    if global_start < 0 or global_end > len(items) or global_start > global_end:
        raise ValueError(
            f"Invalid shard [{global_start}, {global_end}) for {len(items)} total items"
        )

    shard = list(enumerate(items[global_start:global_end], start=global_start))
    workers = max(1, min(args.workers, args.max_workers, len(shard) or 1))

    print(
        f"BAMOR shard global=[{global_start},{global_end}) "
        f"items={len(shard)} workers={workers} threads_per_run={args.threads_per_run}",
        flush=True,
    )
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [
            pool.submit(run_one, args, item_idx, method, seed)
            for item_idx, (method, seed) in shard
        ]
        for fut in as_completed(futs):
            item_idx, rc = fut.result()
            if rc != 0:
                failures.append((item_idx, rc))

    if failures:
        print(f"FAILED items: {failures}", flush=True)
        return 1
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
