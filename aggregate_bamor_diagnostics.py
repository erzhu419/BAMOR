#!/usr/bin/env python3
"""Aggregate BAMOR diagnostic JSON files from flat or scheduler-sharded runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path
from typing import Iterable


METHOD_ORDER = [
    "uniform",
    "uniform_ctx_oracle",
    "bamor_oracle_cone",
    "bamor_ctx_oracle_cone",
    "bamor_oracle",
    "bamor_ctx_oracle",
    "bamor_fixed_cone",
    "bamor_ctx_fixed_cone",
    "bamor",
    "bamor_ctx",
    "bamor_ns",
    "bamor_ns_eps",
    "bamor_ctx_ns_eps",
    "bamor_nc",
    "bamor_ctx_nc",
    "bamor_fastdet",
    "bamor_fastdet_ctx",
    "bamor_fastdet_ctx_nc",
    "uniform_trans_ctx",
    "bamor_fastdet_trans_ctx",
    "bamor_fastdet_trans_ctx_nc",
    "bamor_v2",
    "bamor_v2_belief",
    "bamor_v2_cone",
    "bamor_v2_hybrid",
    "bamor_v2_hybrid_belief",
    "bamor_v2_hybrid_cone",
    "bamor_v2_contrast",
    "bamor_v2_contrast_hybrid",
    "bamor_v2_contrast_hybrid_cone",
    "bamor_v2_generic",
    "bamor_v2_contrast_generic",
    "bamor_v2_contrast_generic_cone",
    "bamor_v2_boundary_hybrid",
    "bamor_v2_boundary_hybrid_cone",
    "bamor_v2_boundary_generic",
    "bamor_v2_proto_hybrid",
    "bamor_v2_proto_hybrid_cone",
    "bamor_v2_proto_generic",
    "bamor_v2_seq",
    "bamor_v2_seq_contrast",
    "bamor_v2_seq_contrast_cone",
    "fixed",
    "capql",
    "gpi_pd",
]

TRUE_ANCHORS_2D = (
    (0.85, 0.15),
    (0.15, 0.85),
    (0.50, 0.50),
)
GLOBAL_REF = (-5.0, -5.0)


def iter_run_json(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*_seed*.json")):
        if path.name == "all_results.json":
            continue
        yield path


def fmt(values, scale: float = 1000.0) -> str:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return "-"
    mean = stats.mean(xs) / scale
    sd = (stats.pstdev(xs) if len(xs) > 1 else 0.0) / scale
    return f"{mean:.1f}+/-{sd:.1f}"


def compute_hv_2d(points, ref_point=GLOBAL_REF) -> float:
    pts = [
        (float(p[0]), float(p[1]))
        for p in points
        if p is not None and len(p) >= 2
        and float(p[0]) > ref_point[0] and float(p[1]) > ref_point[1]
    ]
    if not pts:
        return 0.0

    nd = []
    for p in pts:
        dominated = False
        for q in pts:
            if (q[0] >= p[0] and q[1] >= p[1]) and (q[0] > p[0] or q[1] > p[1]):
                dominated = True
                break
        if not dominated:
            nd.append(p)
    if not nd:
        return 0.0

    nd.sort(key=lambda p: p[0], reverse=True)
    hv = 0.0
    max_obj1 = ref_point[1]
    for p in nd:
        if p[1] > max_obj1:
            hv += (p[0] - ref_point[0]) * (p[1] - max_obj1)
            max_obj1 = p[1]
    return hv


def l2(a, b) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def cone_metrics(run: dict, radius: float) -> dict:
    per_mode = run.get("pareto_per_mode") or []
    cone_hv = []
    cone_count = []
    outside_count = []
    full_hv_recomputed = []
    for mode_idx, front in enumerate(per_mode[: len(TRUE_ANCHORS_2D)]):
        anchor = TRUE_ANCHORS_2D[mode_idx]
        inside_returns = []
        all_returns = []
        n_inside = 0
        n_outside = 0
        for point in front or []:
            w = point.get("weight")
            ret = point.get("return")
            if not w or not ret:
                continue
            all_returns.append(ret)
            if l2(w, anchor) <= radius:
                inside_returns.append(ret)
                n_inside += 1
            else:
                n_outside += 1
        cone_hv.append(compute_hv_2d(inside_returns))
        full_hv_recomputed.append(compute_hv_2d(all_returns))
        cone_count.append(n_inside)
        outside_count.append(n_outside)
    return {
        "cone_hv_per_mode": cone_hv,
        "cone_hv_sum": sum(cone_hv),
        "cone_hv_mean": stats.mean(cone_hv) if cone_hv else None,
        "cone_count_per_mode": cone_count,
        "outside_count_per_mode": outside_count,
        "full_hv_recomputed_per_mode": full_hv_recomputed,
        "outside_hv_gap_sum": sum(full_hv_recomputed) - sum(cone_hv),
    }


def method_sort_key(method: str):
    try:
        return (0, METHOD_ORDER.index(method))
    except ValueError:
        return (1, method)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Result directory to scan recursively")
    parser.add_argument("--write-json", default=None,
                        help="Optional path for aggregated JSON summary")
    parser.add_argument("--scale", type=float, default=1000.0,
                        help="Scale HV values by this number for printing")
    parser.add_argument("--cone-radius", type=float, default=0.25,
                        help="L2 radius around true anchors for cone-local HV")
    parser.add_argument("--no-cone-report", action="store_true",
                        help="Disable cone-local metric table")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rows = defaultdict(list)
    parse_errors = []

    for path in iter_run_json(root):
        try:
            with path.open() as fh:
                run = json.load(fh)
        except Exception as exc:  # keep aggregation useful while jobs finish
            parse_errors.append((str(path), str(exc)))
            continue
        method = run.get("method")
        if method:
            run["_path"] = str(path)
            rows[method].append(run)

    print(f"Scanned: {root}")
    if parse_errors:
        print(f"Skipped unreadable JSON files: {len(parse_errors)}")
    print()
    print("Summary, HV x 1e3")
    print(f"{'method':<20} {'n':>3} {'overall':>16} {'mode0':>16} {'mode1':>16} {'mode2':>16} {'det/true':>14}")

    summary = {}
    for method in sorted(rows, key=method_sort_key):
        runs = sorted(rows[method], key=lambda r: int(r.get("seed", 0)))
        overall = [r.get("hv_overall") for r in runs]
        per_mode = [[], [], []]
        detected = []
        true_switches = []
        for run in runs:
            cm = cone_metrics(run, args.cone_radius)
            run["_cone_metrics"] = cm
            hvs = run.get("hv_per_mode") or []
            for idx in range(min(3, len(hvs))):
                per_mode[idx].append(hvs[idx])
            if "bamor_switches_detected" in run:
                detected.append(run.get("bamor_switches_detected"))
            if "env_switches_true" in run:
                true_switches.append(run.get("env_switches_true"))

        det_true = "-"
        if detected or true_switches:
            det = fmt(detected, scale=1.0) if detected else "-"
            tru = fmt(true_switches, scale=1.0) if true_switches else "-"
            det_true = f"{det}/{tru}"

        print(
            f"{method:<20} {len(runs):>3} {fmt(overall, args.scale):>16} "
            f"{fmt(per_mode[0], args.scale):>16} "
            f"{fmt(per_mode[1], args.scale):>16} "
            f"{fmt(per_mode[2], args.scale):>16} "
            f"{det_true:>14}"
        )
        summary[method] = {
            "n": len(runs),
            "seeds": [r.get("seed") for r in runs],
            "hv_overall": overall,
            "hv_per_mode": per_mode,
            "cone_radius": args.cone_radius,
            "cone_hv_sum": [r["_cone_metrics"]["cone_hv_sum"] for r in runs],
            "cone_hv_mean": [r["_cone_metrics"]["cone_hv_mean"] for r in runs],
            "cone_hv_per_mode": [
                [r["_cone_metrics"]["cone_hv_per_mode"][m]
                 for r in runs
                 if len(r["_cone_metrics"]["cone_hv_per_mode"]) > m]
                for m in range(3)
            ],
            "outside_hv_gap_sum": [r["_cone_metrics"]["outside_hv_gap_sum"] for r in runs],
            "bamor_switches_detected": detected,
            "env_switches_true": true_switches,
            "paths": [r["_path"] for r in runs],
        }

    if not rows:
        print("(no *_seed*.json files found)")
    elif not args.no_cone_report:
        print()
        print(f"True-cone summary, radius={args.cone_radius}, HV x 1e3")
        print(f"{'method':<20} {'n':>3} {'cone_sum':>16} {'cone0':>16} {'cone1':>16} {'cone2':>16} {'outside_gap':>16}")
        for method in sorted(rows, key=method_sort_key):
            runs = sorted(rows[method], key=lambda r: int(r.get("seed", 0)))
            cone_sum = [r["_cone_metrics"]["cone_hv_sum"] for r in runs]
            outside_gap = [r["_cone_metrics"]["outside_hv_gap_sum"] for r in runs]
            per_mode_cone = [[], [], []]
            for run in runs:
                vals = run["_cone_metrics"]["cone_hv_per_mode"]
                for idx in range(min(3, len(vals))):
                    per_mode_cone[idx].append(vals[idx])
            print(
                f"{method:<20} {len(runs):>3} {fmt(cone_sum, args.scale):>16} "
                f"{fmt(per_mode_cone[0], args.scale):>16} "
                f"{fmt(per_mode_cone[1], args.scale):>16} "
                f"{fmt(per_mode_cone[2], args.scale):>16} "
                f"{fmt(outside_gap, args.scale):>16}"
            )

    if args.write_json:
        out = Path(args.write_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            json.dump(summary, fh, indent=2)
        print()
        print(f"Wrote: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
