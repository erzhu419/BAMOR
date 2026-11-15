#!/usr/bin/env python3
"""Run a scheduler CPU shard of BAMOR MuJoCo env/method/seed items."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_words(raw: str) -> list[str]:
    return [x for x in raw.split() if x]


def parse_seeds(raw: str, seed_start: int, seeds: int) -> list[int]:
    if raw.strip():
        return [int(x) for x in raw.replace(",", " ").split()]
    return list(range(seed_start, seed_start + seeds))


def build_items(envs: list[str], methods: list[str], seeds: list[int]) -> list[tuple[str, str, int]]:
    return [(env_name, method, seed) for env_name in envs for seed in seeds for method in methods]


def run_one(args: argparse.Namespace, item_idx: int, env_name: str, method: str, seed: int) -> tuple[int, int]:
    env_safe = env_name.replace("-", "_")
    output_json = Path(args.save_dir) / env_safe / f"{method}_seed{seed}.json"
    log_dir = Path(args.save_dir) / "_shard_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and output_json.exists() and output_json.stat().st_size > 0:
        print(f"[skip] item={item_idx} {env_name} {method} seed={seed} -> {output_json}", flush=True)
        return item_idx, 0

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads_per_run)
    env["MKL_NUM_THREADS"] = str(args.threads_per_run)
    env["OPENBLAS_NUM_THREADS"] = str(args.threads_per_run)
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        "-u",
        "train_bamor_mujoco.py",
        "--env", env_name,
        "--method", method,
        "--total_steps", str(args.total_steps),
        "--switch_interval", str(args.switch_interval),
        "--seed", str(seed),
        "--num_seeds", "1",
        "--save_dir", args.save_dir,
        "--eval_freq", str(args.eval_freq),
        "--device", args.device,
        "--hidden", str(args.hidden),
    ]
    log_path = log_dir / f"item{item_idx:04d}_{env_safe}_{method}_seed{seed}.log"
    print(f"[run] item={item_idx} {env_name} {method} seed={seed} log={log_path}", flush=True)
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f"[done] item={item_idx} {env_name} {method} seed={seed} rc={proc.returncode}", flush=True)
    return item_idx, proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--item-offset", type=int, default=0)
    parser.add_argument("--envs", default="mo-halfcheetah-v5")
    parser.add_argument("--methods", required=True)
    parser.add_argument("--seed-list", default="")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--switch-interval", type=int, default=25_000)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--threads-per-run", type=int, default=2)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    items = build_items(parse_words(args.envs), parse_words(args.methods), parse_seeds(args.seed_list, args.seed_start, args.seeds))
    global_start = args.item_offset + args.start
    global_end = args.item_offset + args.end
    if global_start < 0 or global_end > len(items) or global_start > global_end:
        raise ValueError(f"Invalid shard [{global_start}, {global_end}) for {len(items)} total items")

    shard = list(enumerate(items[global_start:global_end], start=global_start))
    workers = max(1, min(args.workers, args.max_workers, len(shard) or 1))
    print(
        f"BAMOR MuJoCo shard global=[{global_start},{global_end}) "
        f"items={len(shard)} workers={workers} threads_per_run={args.threads_per_run}",
        flush=True,
    )

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(run_one, args, item_idx, env_name, method, seed)
                for item_idx, (env_name, method, seed) in shard]
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
