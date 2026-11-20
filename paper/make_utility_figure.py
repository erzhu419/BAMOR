"""Generate the main-text 20-seed direct utility comparison figure without training.

Reads completed compact readouts:
  - results/iclr_utility_confirmation_20260927/readout.json
  - results/iclr_capql_full_simplex_20260930/compact_readout.json
Writes:
  - paper/Fig2.pdf
  - paper/Fig2.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .make_evidence_tables import full_simplex_table, utility_confirmation_tables
else:
    from make_evidence_tables import full_simplex_table, utility_confirmation_tables

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
CONFIRM_PATH = RESULTS / "iclr_utility_confirmation_20260927" / "readout.json"
FULL_SIMPLEX_PATH = RESULTS / "iclr_capql_full_simplex_20260930" / "compact_readout.json"

MODE_NAMES = {
    "mo-halfcheetah-v5": ["Normal", "Low\nspeed", "Low\ncost", "Locked"],
    "mo-hopper-v5": ["Normal", "Fragile", "Cheap\nenergy", "Grounded"],
}

COLORS = {
    "v25": "#1f5aa6",
    "capql": "#d95f02",
    "capql_full_simplex": "#1b9e77",
}

LABELS = {
    "v25": "BAMOR",
    "capql": "Native CAPQL ($22.5^\\circ$)",
    "capql_full_simplex": "Full-simplex CAPQL",
}


def load_verified_data():
    confirm = json.loads(CONFIRM_PATH.read_text())
    full = json.loads(FULL_SIMPLEX_PATH.read_text())
    utility_confirmation_tables(confirm)
    full_simplex_table(full, confirm)
    return confirm, full


def build_figure(out_stem: Path | None = None):
    confirm, full = load_verified_data()
    if out_stem is None:
        out_stem = HERE / "Fig2"

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "pdf.fonttype": 42,
    })

    fig = plt.figure(figsize=(6.0, 4.9))
    ax_ci = fig.add_axes((0.29, 0.62, 0.695, 0.34))
    ax_hc = fig.add_axes((0.13, 0.145, 0.38, 0.335))
    ax_hop = fig.add_axes((0.60, 0.145, 0.38, 0.335))

    # Panel (a) uses the interval specified for each contrast in the tables.
    contrasts = [
        ("mo-halfcheetah-v5", "native", "HC: BAMOR $-$ Native", "#1f5aa6"),
        ("mo-halfcheetah-v5", "v25_minus_full_simplex", "HC: BAMOR $-$ Full-Simp.", "#2b7bba"),
        ("mo-halfcheetah-v5", "full_simplex_minus_native", "HC: Full-Simp. $-$ Native", "#1b9e77"),
        ("mo-hopper-v5", "native", "Hopper: BAMOR $-$ Native", "#1f5aa6"),
        ("mo-hopper-v5", "v25_minus_full_simplex", "Hopper: BAMOR $-$ Full-Simp.", "#1f5aa6"),
        ("mo-hopper-v5", "full_simplex_minus_native", "Hopper: Full-Simp. $-$ Native", "#666666"),
    ]
    y_positions = [5.3, 4.3, 3.3, 2.0, 1.0, 0.0]
    rng = np.random.default_rng(42)

    ax_ci.axvline(0.0, color="#222222", linestyle="--", linewidth=0.9, zorder=1)
    ax_ci.axhline(2.65, color="#cccccc", linestyle="-", linewidth=0.7, zorder=0)

    for y, (env, key, label, color) in zip(y_positions, contrasts):
        if key == "native":
            entry = confirm["environments"][env]["paired"]["direct_utility"]
        else:
            entry = full["environments"][env]["paired"][key]["direct_utility"]
        deltas = np.asarray(entry["paired_deltas"], dtype=float)
        mean_d = float(entry["mean_delta"])
        interval = "paired_mean_ci95" if key == "full_simplex_minus_native" else "paired_mean_ci97_5"
        ci_low, ci_high = entry[interval]

        jitter = rng.uniform(-0.16, 0.16, size=len(deltas))
        ax_ci.scatter(
            deltas, np.full_like(deltas, y) + jitter,
            s=13, color=color, alpha=0.32, edgecolors="none", zorder=2,
        )
        ax_ci.errorbar(
            mean_d, y,
            xerr=[[mean_d - ci_low], [ci_high - mean_d]],
            fmt="o", color=color, markersize=5.0, elinewidth=1.9,
            capsize=3.2, capthick=1.3, zorder=4,
        )

    ax_ci.set_yticks(y_positions)
    ax_ci.set_yticklabels([c[2] for c in contrasts])
    ax_ci.set_xlabel("Paired utility difference $\\Delta U$ (20 seeds)")
    ax_ci.set_title("(a)", loc="left")
    ax_ci.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax_ci.spines["top"].set_visible(False)
    ax_ci.spines["right"].set_visible(False)

    # Panels (b) & (c): Per-mode mean direct utility with +/- 1 SEM across 20 seeds
    methods = ("v25", "capql", "capql_full_simplex")
    bar_width = 0.25
    offsets = (-bar_width, 0.0, bar_width)

    for ax, env, panel_title in (
        (ax_hc, "mo-halfcheetah-v5", "(b) HalfCheetah"),
        (ax_hop, "mo-hopper-v5", "(c) Hopper"),
    ):
        rows = full["environments"][env]["seed_metrics"]
        x = np.arange(len(MODE_NAMES[env]), dtype=float)
        ax.axhline(0.0, color="#444444", linewidth=0.7, zorder=1)
        for offset, method in zip(offsets, methods):
            mat = np.asarray(
                [r["methods"][method]["metrics"]["direct_utility"]["per_mode"] for r in rows],
                dtype=float,
            )
            means = mat.mean(axis=0)
            sems = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
            hatch = {"v25": "", "capql": "////", "capql_full_simplex": ".."}[method]
            ax.bar(
                x + offset, means, width=bar_width * 0.92,
                color=COLORS[method], edgecolor="#333333", linewidth=0.45,
                hatch=hatch, label=LABELS[method], zorder=2,
            )
            ax.errorbar(
                x + offset, means, yerr=sems,
                fmt="none", ecolor="#222222", elinewidth=1.0, capsize=2.2, zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(MODE_NAMES[env])
        ax.set_title(panel_title)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_hc.set_ylabel("Mean direct utility $U_h$ ($\\pm$ SEM)")
    handles, labels = ax_hc.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.53, 0.012),
        ncol=3, frameon=False,
    )
    pdf_path = out_stem.with_suffix(".pdf")
    png_path = out_stem.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    print(f"Wrote {pdf_path} and {png_path}")
    return pdf_path, png_path


if __name__ == "__main__":
    build_figure()
