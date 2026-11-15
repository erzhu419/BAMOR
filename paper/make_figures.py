"""Generate main-result table + training/pareto figures for the BAMOR paper.

Reads per-seed JSONs from results/server_curved20/ and writes:
    - paper/table_main_results.tex
    - paper/fig_training_curves_v2.pdf  (also .png for quick view)
    - paper/fig_pareto_permode.pdf
"""
import json
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / 'results' / 'server_curved20'

PRETTY = {
    'bamor_ns': ('BAMOR-TC (ours)',     'tab:blue',   '-'),
    'bamor':    ('BAMOR $+$ surprise',  'tab:orange', '--'),
    'bamor_nc': ('BAMOR $-$ Cone',      'tab:green',  ':'),
    'uniform':  ('Uniform-sampling',    'tab:red',    '-.'),
    'fixed':    ('Fixed-weight',        'tab:gray',   (0, (3, 1, 1, 1))),
}
ORDER = ['bamor_ns', 'bamor', 'bamor_nc', 'uniform', 'fixed']


def load_all():
    data = defaultdict(list)
    for f in sorted(RESULTS_DIR.glob('*_seed*.json')):
        with open(f) as fh:
            d = json.load(fh)
        data[d['method']].append(d)
    return data


def write_main_table(data, out_path):
    lines = [
        r'\begin{table}[t]',
        r'\centering\small',
        r'\caption{Hypervolume (HV) on the synthetic non-stationary MO environment at 100K steps (20 seeds).  Mode anchors are $w_0{=}[0.85,0.15]$, $w_1{=}[0.15,0.85]$, $w_2{=}[0.5,0.5]$.  Reference point $(-5,-5)$.  Bold = best per column.}',
        r'\label{tab:main_results}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'Method & HV overall & HV Mode 0 & HV Mode 1 & HV Mode 2 \\',
        r'\midrule',
    ]
    stats = {}
    for m in ORDER:
        runs = data.get(m, [])
        if not runs:
            continue
        overall = np.array([r['hv_overall'] for r in runs])
        per_mode = np.array([r['hv_per_mode'] for r in runs])  # (seeds, 3)
        stats[m] = {
            'overall': (overall.mean(), overall.std()),
            'm0':      (per_mode[:, 0].mean(), per_mode[:, 0].std()),
            'm1':      (per_mode[:, 1].mean(), per_mode[:, 1].std()),
            'm2':      (per_mode[:, 2].mean(), per_mode[:, 2].std()),
        }

    best = {k: max(stats, key=lambda m: stats[m][k][0]) for k in ('overall', 'm0', 'm1', 'm2')}

    def cell(m, key):
        mu, sd = stats[m][key]
        s = f'${mu/1000:.1f}{{\\pm}}{sd/1000:.1f}$'
        return r'\textbf{' + s + '}' if best[key] == m else s

    for m in ORDER:
        if m not in stats:
            continue
        name = PRETTY[m][0]
        lines.append(f'{name} & {cell(m, "overall")} & {cell(m, "m0")} & {cell(m, "m1")} & {cell(m, "m2")} \\\\')
    lines += [r'\bottomrule', r'\multicolumn{5}{l}{\footnotesize HV values in units of $10^3$; $\pm$ = one standard deviation across 20 seeds.}',
              r'\end{tabular}', r'\end{table}']

    out_path.write_text('\n'.join(lines))
    print(f'[table] wrote {out_path}')


def make_training_curves(data, out_path):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))

    for m in ORDER:
        if m not in data or not data[m]:
            continue
        label, color, style = PRETTY[m]
        # Stack curves across seeds
        curves = []
        for r in data[m]:
            curve = r.get('hv_curve', [])
            if not curve:
                continue
            xs = np.array([p['step'] for p in curve])
            ys = np.array([p['hv'] for p in curve])
            curves.append(ys)
        if not curves:
            continue
        arr = np.vstack(curves) / 1000.0
        mean = arr.mean(0)
        std = arr.std(0)
        ax.plot(xs / 1000, mean, color=color, ls=style, lw=1.8, label=label)
        ax.fill_between(xs / 1000, mean - std, mean + std, color=color, alpha=0.12)

    ax.set_xlabel('Training steps ($\\times 10^3$)')
    ax.set_ylabel('Hypervolume ($\\times 10^3$)')
    ax.set_title('Training dynamics under piecewise-stationary modes')
    ax.legend(loc='lower right', fontsize=8, frameon=True)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix('.pdf'))
    fig.savefig(out_path.with_suffix('.png'), dpi=150)
    plt.close(fig)
    print(f'[fig] wrote {out_path.with_suffix(".pdf")}')


def make_pareto_arcs(data, out_path):
    """For each mode 0/1/2, plot the per-mode Pareto fronts of BAMOR vs Uniform."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True, sharey=True)
    mode_names = ['Mode 0 (safety-dom, anchor $[0.85,0.15]$)',
                  'Mode 1 (progress-dom, anchor $[0.15,0.85]$)',
                  'Mode 2 (balanced, anchor $[0.5,0.5]$)']

    for idx, ax in enumerate(axes):
        for m in ['bamor_ns', 'uniform']:
            pts = []
            for r in data.get(m, []):
                pareto = r.get('pareto_per_mode', [])
                if idx < len(pareto) and pareto[idx]:
                    for p in pareto[idx]:
                        ret = p.get('return') if isinstance(p, dict) else p
                        if ret is not None and len(ret) >= 2:
                            pts.append(ret[:2])
            if not pts:
                continue
            pts = np.array(pts)
            label, color, _ = PRETTY[m]
            ax.scatter(pts[:, 0], pts[:, 1], s=9, color=color, alpha=0.55, label=label, edgecolors='none')
        ax.set_title(mode_names[idx], fontsize=8.5)
        ax.set_xlabel('Objective 0 (safety)')
        if idx == 0:
            ax.set_ylabel('Objective 1 (progress)')
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(loc='lower left', fontsize=7.5, frameon=True)

    fig.suptitle('Recovered per-mode Pareto fronts on local arcs', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path.with_suffix('.pdf'))
    fig.savefig(out_path.with_suffix('.png'), dpi=150)
    plt.close(fig)
    print(f'[fig] wrote {out_path.with_suffix(".pdf")}')


def main():
    data = load_all()
    print('Loaded methods:', {k: len(v) for k, v in data.items()})

    # Table is generated by gen_final_table.py (server-side) — do not overwrite here.
    make_training_curves(data, HERE / 'fig_training_curves_v2')
    make_pareto_arcs(data, HERE / 'fig_pareto_permode')


if __name__ == '__main__':
    main()
