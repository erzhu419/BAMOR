"""Generate a 'pretty' piecewise sub-manifold figure from current 20-seed data.

Single panel: x-axis = obj 0 (safety), y-axis = obj 1 (progress).
Three modes coloured (red/green/blue) showing their distinct Pareto arcs.
Overlay BAMOR-TC trained policy points + Uniform points to show how
BAMOR-TC concentrates onto the valid arc per mode while Uniform scatters.
"""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / 'results' / 'server_tuned20'

MODE_COLORS = ['#d62728', '#2ca02c', '#1f77b4']  # red, green, blue
MODE_NAMES  = ['Mode 0: safety-zone (anchor [0.85, 0.15])',
               'Mode 1: progress-zone (anchor [0.15, 0.85])',
               'Mode 2: balanced (anchor [0.5, 0.5])']

def collect_per_mode(method):
    """For each mode, collect all (obj0, obj1) points from this method's seeds."""
    by_mode = [[], [], []]
    for f in sorted(RESULTS_DIR.glob(f'{method}_seed*.json')):
        j = json.load(open(f))
        for mi, mode_pareto in enumerate(j.get('pareto_per_mode', [])):
            if mi >= 3 or not mode_pareto:
                continue
            for p in mode_pareto:
                ret = p['return'] if isinstance(p, dict) else p
                if len(ret) >= 2:
                    by_mode[mi].append(ret[:2])
    return [np.array(m) if m else np.empty((0, 2)) for m in by_mode]


def main():
    bamor_tc = collect_per_mode('bamor_ns')
    uniform  = collect_per_mode('uniform')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # ── Panel 1: BAMOR-TC ────────────────────────────────────────
    ax = axes[0]
    for mi in range(3):
        pts = bamor_tc[mi]
        if pts.size == 0: continue
        ax.scatter(pts[:, 0], pts[:, 1], s=8, color=MODE_COLORS[mi],
                   alpha=0.45, edgecolors='none', label=MODE_NAMES[mi])
    ax.set_title('BAMOR-TC (ours)', fontsize=12)
    ax.set_xlabel('Objective 0 (safety return)')
    ax.set_ylabel('Objective 1 (progress return)')
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc='upper right', fontsize=8.5, frameon=True, framealpha=0.95)

    # ── Panel 2: Uniform ─────────────────────────────────────────
    ax = axes[1]
    for mi in range(3):
        pts = uniform[mi]
        if pts.size == 0: continue
        ax.scatter(pts[:, 0], pts[:, 1], s=8, color=MODE_COLORS[mi],
                   alpha=0.45, edgecolors='none', label=MODE_NAMES[mi])
    ax.set_title('Uniform-sampling', fontsize=12)
    ax.set_xlabel('Objective 0 (safety return)')
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc='upper right', fontsize=8.5, frameon=True, framealpha=0.95)

    fig.suptitle('Piecewise Local Pareto Sub-manifolds: each mode lives on a distinct arc',
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = HERE / 'fig_submfld_trained_v2'
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.png'), dpi=160)
    plt.close(fig)
    print(f'wrote {out.with_suffix(".pdf")} and .png')

    # ── Bonus: cone-vs-uniform single panel for §1 hero figure ──
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for mi in range(3):
        pts = bamor_tc[mi]
        if pts.size == 0: continue
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color=MODE_COLORS[mi],
                   alpha=0.55, edgecolors='none', label=f'Mode {mi}')
    ax.set_title('Piecewise Pareto sub-manifolds recovered by BAMOR-TC\n(20 seeds, synthetic non-stationary MO benchmark)',
                 fontsize=11)
    ax.set_xlabel('Objective 0 (safety return)')
    ax.set_ylabel('Objective 1 (progress return)')
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.95)
    fig.tight_layout()
    out = HERE / 'fig_submfld_hero'
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.png'), dpi=160)
    plt.close(fig)
    print(f'wrote {out.with_suffix(".pdf")} and .png')


if __name__ == '__main__':
    main()
