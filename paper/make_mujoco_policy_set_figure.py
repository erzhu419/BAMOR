"""Plot seed-level policy-set gains for the 3-D MuJoCo stress test."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
TAG = os.environ.get("TAG", "3d10")
CSV_PATH = HERE.parent / "results" / f"mujoco_policy_set_union_{TAG}_seed_diagnostics.csv"
OUT = HERE / "fig_mujoco_policy_set_gain"

V5 = "bamor_v5_contrast_moe_template_mix"
V6 = "bamor_v6_contrast_moe_head_template_mix"
UNION = "v5_v6_union"


def load_rows() -> dict[str, list[dict]]:
    by_env: dict[str, list[dict]] = {}
    with CSV_PATH.open(newline="") as fh:
        for row in csv.DictReader(fh):
            env = row["env"]
            by_env.setdefault(env, []).append(row)
    return by_env


def as_b(row: dict, method: str) -> float:
    return float(row[f"hv:{method}"]) / 1e9


def main() -> None:
    by_env = load_rows()
    env_order = ["mo-ant-v5", "mo-hopper-v5"]
    env_titles = {
        "mo-ant-v5": "Ant",
        "mo-hopper-v5": "Hopper",
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    for ax, env in zip(axes, env_order):
        rows = sorted(
            by_env[env],
            key=lambda row: (float(row["best_constituent_hv"]), int(row["seed"])),
        )
        x = np.arange(len(rows))
        seeds = [int(row["seed"]) for row in rows]
        v5 = np.array([as_b(row, V5) for row in rows])
        v6 = np.array([as_b(row, V6) for row in rows])
        best = np.maximum(v5, v6)
        union = np.array([as_b(row, UNION) for row in rows])
        gains = union - best

        for xi, y0, y1 in zip(x, best, union):
            ax.plot([xi, xi], [y0, y1], color="#7a7a7a", lw=1.0, alpha=0.65)
        ax.scatter(x - 0.07, v5, s=18, color="#386cb0", alpha=0.70, label="v5 MoE-template")
        ax.scatter(x + 0.07, v6, s=18, color="#fdb462", alpha=0.80, label="v6 shared-head")
        ax.scatter(x, union, s=30, color="#1b9e77", edgecolor="black", linewidth=0.35,
                   zorder=3, label="policy set")

        ax.set_title(
            f"{env_titles[env]}: mean gain {gains.mean():.3f}B",
            fontsize=9,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"s{s}" for s in seeds], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("HV (billions)")
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT.with_suffix('.pdf')} and .png")


if __name__ == "__main__":
    main()
