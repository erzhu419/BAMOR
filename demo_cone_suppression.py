#!/usr/bin/env python3
"""
demo_cone_suppression.py

Demonstrates BAMOR's core claim:
  "Target Cone restricts preference sampling to the current mode's
   valid subspace, reducing wasted exploration during mode switches."

Uses the DEFAULT SyntheticMOEnv modes (reward_scale mechanism):
  Mode 0 (safe_zone):  scale=[3.0, 0.1]  → safety responsive, progress collapsed → true anchor ≈ [0.97, 0.03]
  Mode 1 (fast_lane):  scale=[0.1, 3.0]  → progress responsive, safety collapsed → true anchor ≈ [0.03, 0.97]
  Mode 2 (balanced):   scale=[1.0, 1.0]  → both responsive                        → true anchor ≈ [0.50, 0.50]

Key design: reward_scale makes per-objective variance clearly separable,
so the TargetConeEstimator can LEARN the correct anchor from data.

Key metric: CONE HIT RATE
  Hit if |w[0] - true_anchor[mode][0]| < CONE_RADIUS.
  BAMOR should concentrate samples near the correct anchor; Uniform stays flat.

Output:
  experiments/figures/fig_cone_suppression.png
  experiments/cone_suppression_stats.json
"""

import sys, os, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from bamor_core import BAMORPipeline
from synthetic_mo_env import SyntheticMOEnv   # uses default modes (reward_scale)


# ─── True anchors (from default SyntheticMOEnv mode configs) ──────────────────
TRUE_ANCHORS = np.array([
    [0.98, 0.02],  # mode 0: safe_zone  (safety dominant)
    [0.02, 0.98],  # mode 1: fast_lane  (progress dominant)
])
CONE_RADIUS = 0.20   # tolerance for "cone hit"
NUM_MODES   = 2
MODE_NAMES  = ['safe_zone\nanchor=0.98', 'fast_lane\nanchor=0.02']
MODE_COLORS = ['#2ca02c', '#d62728']


def make_env(seed=0, switch_interval=300, max_steps=2100):
    """2 clearly-separated modes: safety dominant vs. progress dominant.

    reward_scale ensures per-objective variance is separable:
      Mode 0: scale=[5.0, 0.1] → obj0 variance >> obj1 → anchor ≈ [0.98, 0.02]
      Mode 1: scale=[0.1, 5.0] → obj1 variance >> obj0 → anchor ≈ [0.02, 0.98]

    These extreme scales give the TargetConeEstimator a clear signal,
    so the learned anchor converges quickly to the true value.
    """
    from synthetic_mo_env import ModeConfig
    modes = [
        ModeConfig(
            name='safe_zone',
            reward_scale=np.array([5.0, 0.1]),  # safety dominates
            speed_multiplier=0.5,               # slow → position stays near 0
            noise_std=0.05,
            true_anchor=np.array([0.98, 0.02]),
        ),
        ModeConfig(
            name='fast_lane',
            reward_scale=np.array([0.1, 5.0]),  # progress dominates
            speed_multiplier=2.0,               # fast → large position swings
            noise_std=0.10,
            true_anchor=np.array([0.02, 0.98]),
        ),
    ]
    return SyntheticMOEnv(modes=modes, switch_interval=switch_interval,
                          max_steps=max_steps, seed=seed)


def run_method(method, num_episodes=10, switch_interval=100, max_steps=600, seed=0):
    """Run one method for num_episodes long episodes, recording per-step data."""
    np.random.seed(seed)
    env = make_env(seed=seed, switch_interval=switch_interval, max_steps=max_steps)

    bamor = None
    if method == 'bamor':
        bamor = BAMORPipeline(
            num_modes=NUM_MODES, reward_dim=2, gamma=0.99,
            surprise_threshold=8.0,   # high enough to prevent false positives
            concentration=25.0,       # tight cone → strong suppression signal
            ema_alpha=0.05,           # slow EMA → stable statistics
            switch_cooldown=200,      # env switches every 300; cooldown prevents double-fire
            warmup=150,               # EMA must stabilize before any detection fires
        )

    records = []    # list of dicts per step
    global_step = 0

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            # Sample weight
            if bamor:
                w, w_anchor = bamor.sample_weight()
            else:
                w = np.random.dirichlet(np.ones(2)).astype(np.float32)
                w_anchor = np.array([0.5, 0.5], np.float32)

            action = env.action_space.sample()
            obs, rew, term, trunc, info = env.step(action)
            done = term or trunc
            mode_idx = info['mode_idx']

            records.append({
                'step':     global_step,
                'ep':       ep,
                'mode':     mode_idx,
                'w0':       float(w[0]),
                'anchor0':  float(w_anchor[0]),
            })

            if bamor:
                bamor.observe_reward(rew)

            global_step += 1

        # Print BAMOR state after each episode
        if bamor and (ep + 1) % 1 == 0:
            s = bamor.get_state()
            anchors = s['anchors']
            anchor_str = '  '.join(
                f'M{m}=[{anchors[m,0]:.2f},{anchors[m,1]:.2f}]'
                for m in range(len(anchors)))
            print(f'  ep={ep+1:3d}  step={global_step:6d}  '
                  f'switches={s["mode_switch_count"]:3d}  '
                  f'anchors: {anchor_str}')

    env.close()
    return records


def cone_hit(w0, mode_idx):
    return abs(w0 - TRUE_ANCHORS[mode_idx, 0]) < CONE_RADIUS


def rolling_hit_rate(records, window=80):
    hits = [1 if cone_hit(r['w0'], r['mode']) else 0 for r in records]
    steps = [r['step'] for r in records]
    rates = []
    for i in range(len(hits)):
        lo = max(0, i - window // 2)
        hi = min(len(hits), i + window // 2)
        rates.append(np.mean(hits[lo:hi]))
    return steps, rates, hits


def find_switches(records):
    switches = []
    prev = records[0]['mode']
    for r in records[1:]:
        if r['mode'] != prev:
            switches.append(r['step'])
            prev = r['mode']
    return switches


def print_stats(bamor_recs, uniform_recs):
    hit_b = np.mean([cone_hit(r['w0'], r['mode']) for r in bamor_recs])
    hit_u = np.mean([cone_hit(r['w0'], r['mode']) for r in uniform_recs])
    baseline = 1.0 / NUM_MODES

    print(f'\n{"="*55}')
    print(f'  CONE HIT RATE  (|w[0] - true_anchor| < {CONE_RADIUS})')
    print(f'{"="*55}')
    print(f'  BAMOR  : {hit_b:.3f}')
    print(f'  Uniform: {hit_u:.3f}  (random baseline: {baseline:.3f})')
    print(f'  Lift   : {hit_b - hit_u:+.3f}  ({hit_b/max(hit_u,1e-9):.1f}x)')

    print(f'\n  Per-mode breakdown:')
    for m in range(NUM_MODES):
        rb = [r for r in bamor_recs   if r['mode'] == m]
        ru = [r for r in uniform_recs if r['mode'] == m]
        hb = np.mean([cone_hit(r['w0'], m) for r in rb]) if rb else float('nan')
        hu = np.mean([cone_hit(r['w0'], m) for r in ru]) if ru else float('nan')
        sb = np.std([r['w0'] for r in rb]) if rb else float('nan')
        su = np.std([r['w0'] for r in ru]) if ru else float('nan')
        print(f'  Mode {m}: hit BAMOR={hb:.3f} Uniform={hu:.3f}  '
              f'std BAMOR={sb:.3f} Uniform={su:.3f}  '
              f'(n_bamor={len(rb)}, n_uniform={len(ru)})')

    return {
        'hit_bamor': float(hit_b), 'hit_uniform': float(hit_u),
        'baseline': float(baseline), 'lift': float(hit_b - hit_u),
    }


def plot_results(bamor_recs, uniform_recs, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(
        'BAMOR Target Cone Suppression — Core Claim Demo\n'
        'Focused weight sampling concentrates exploration in the current mode\'s valid subspace',
        fontsize=13, fontweight='bold')

    steps_b = [r['step'] for r in bamor_recs]
    w0_b    = [r['w0']   for r in bamor_recs]
    mode_b  = [r['mode'] for r in bamor_recs]
    anc_b   = [r['anchor0'] for r in bamor_recs]

    steps_u = [r['step'] for r in uniform_recs]
    w0_u    = [r['w0']   for r in uniform_recs]

    switches = find_switches(bamor_recs)
    xlim = (0, max(steps_b[-1], steps_u[-1]))

    # ── Panel 1: Weight trajectory ──────────────────────────────────────────
    ax = axes[0]
    # Mode background shading
    prev_m, prev_s = mode_b[0], steps_b[0]
    for i in range(1, len(mode_b)):
        if mode_b[i] != prev_m:
            ax.axvspan(prev_s, steps_b[i], alpha=0.1, color=MODE_COLORS[prev_m])
            prev_m, prev_s = mode_b[i], steps_b[i]
    ax.axvspan(prev_s, steps_b[-1], alpha=0.1, color=MODE_COLORS[prev_m])

    # True anchor as dashed lines per mode region
    for i, (s, m) in enumerate(zip(steps_b, mode_b)):
        if i == 0 or mode_b[i-1] != m:
            # find end of this mode region
            end = steps_b[-1]
            for j in range(i+1, len(mode_b)):
                if mode_b[j] != m:
                    end = steps_b[j]
                    break
            ax.hlines(TRUE_ANCHORS[m, 0], s, end,
                      colors=MODE_COLORS[m], lw=1.5, ls='--', alpha=0.7,
                      label=f'True anchor mode {m}' if i == 0 else None)

    ax.scatter(steps_b, w0_b,  s=3, alpha=0.5, color='#1f77b4', label='BAMOR w[0]',  zorder=3)
    ax.scatter(steps_u, w0_u,  s=3, alpha=0.25, color='gray',   label='Uniform w[0]', zorder=2)
    ax.plot(steps_b, anc_b, color='orange', lw=1.0, alpha=0.8, label='BAMOR anchor (estimated)')

    for sw in switches:
        ax.axvline(sw, color='black', lw=0.7, ls=':', alpha=0.7)

    ax.set_ylabel('w[0]  (safety weight)', fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(*xlim)
    ax.legend(fontsize=7, loc='upper right', ncol=3)
    ax.set_title('Sampled preference weight trajectory  (vertical dots = mode switches; dashed = true anchor)')

    # ── Panel 2: Rolling cone hit rate ───────────────────────────────────────
    ax = axes[1]
    steps_b2, rate_b, hits_b = rolling_hit_rate(bamor_recs, window=80)
    steps_u2, rate_u, hits_u = rolling_hit_rate(uniform_recs, window=80)

    ax.fill_between(steps_b2, rate_b, alpha=0.25, color='#1f77b4')
    ax.plot(steps_b2, rate_b, color='#1f77b4', lw=2.0,
            label=f'BAMOR  (mean={np.mean(hits_b):.3f})')
    ax.fill_between(steps_u2, rate_u, alpha=0.15, color='gray')
    ax.plot(steps_u2, rate_u, color='gray', lw=1.5, ls='--',
            label=f'Uniform (mean={np.mean(hits_u):.3f})')
    ax.axhline(1/NUM_MODES, color='red', ls=':', lw=1.2,
               label=f'Random baseline = 1/{NUM_MODES} = {1/NUM_MODES:.2f}')

    for sw in switches:
        ax.axvline(sw, color='black', lw=0.7, ls=':', alpha=0.7)

    ax.set_ylabel(f'Cone hit rate\n(rolling 80 steps)', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(*xlim)
    ax.legend(fontsize=9)
    ax.set_title('Fraction of samples in current mode\'s true Target Cone\n'
                 f'(cone = |w[0] − true_anchor| < {CONE_RADIUS})')

    # ── Panel 3: Per-mode weight histograms ──────────────────────────────────
    ax = axes[2]
    n_cols = NUM_MODES * 2
    col_w  = 0.8
    positions_b = [i * 2.5 for i in range(NUM_MODES)]
    positions_u = [p + 1.0 for p in positions_b]

    for m in range(NUM_MODES):
        w_b = [r['w0'] for r in bamor_recs   if r['mode'] == m]
        w_u = [r['w0'] for r in uniform_recs if r['mode'] == m]

        if w_b:
            bp = ax.boxplot([w_b], positions=[positions_b[m]], widths=col_w * 0.8,
                            patch_artist=True, manage_ticks=False,
                            boxprops=dict(facecolor='#1f77b4', alpha=0.6),
                            medianprops=dict(color='darkblue', lw=2),
                            whiskerprops=dict(lw=1), capprops=dict(lw=1),
                            flierprops=dict(marker='.', ms=2, alpha=0.3, color='#1f77b4'))
        if w_u:
            bp = ax.boxplot([w_u], positions=[positions_u[m]], widths=col_w * 0.8,
                            patch_artist=True, manage_ticks=False,
                            boxprops=dict(facecolor='gray', alpha=0.5),
                            medianprops=dict(color='black', lw=2),
                            whiskerprops=dict(lw=1), capprops=dict(lw=1),
                            flierprops=dict(marker='.', ms=2, alpha=0.2, color='gray'))

        # True anchor and cone
        mid = (positions_b[m] + positions_u[m]) / 2
        anchor = TRUE_ANCHORS[m, 0]
        ax.hlines(anchor, positions_b[m] - 0.5, positions_u[m] + 0.5,
                  colors=MODE_COLORS[m], lw=2.5, ls='-', alpha=0.9)
        ax.axhspan(max(anchor - CONE_RADIUS, 0), min(anchor + CONE_RADIUS, 1),
                   xmin=(positions_b[m] - 0.5) / (max(positions_u) + 1),
                   xmax=(positions_u[m] + 0.5) / (max(positions_u) + 1),
                   alpha=0.1, color=MODE_COLORS[m])

    # X-axis labels
    xtick_pos = [p + 0.5 for p in positions_b]
    xtick_lbl = [f'Mode {m}\n{MODE_NAMES[m]}' for m in range(NUM_MODES)]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lbl, fontsize=9)
    ax.set_ylabel('w[0]  (sampled)', fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Sampled weight distribution per mode\n'
                 '(colored horizontal line = true anchor; shaded band = cone; blue=BAMOR, gray=Uniform)')

    legend_patches = [
        mpatches.Patch(color='#1f77b4', alpha=0.6, label='BAMOR'),
        mpatches.Patch(color='gray',    alpha=0.5, label='Uniform'),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc='upper right')

    for i in range(2):
        axes[i].set_xlabel('Environment step', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved → {save_path}')
    plt.close()


if __name__ == '__main__':
    SWITCH_INTERVAL = 300   # env steps between mode switches
    MAX_STEPS       = 15000 # single long episode: 50 mode switches, no cross-episode confusion
    NUM_EPISODES    = 1     # one long run keeps mode phase continuous

    print('=== BAMOR (running...) ===')
    bamor_recs = run_method('bamor', NUM_EPISODES, SWITCH_INTERVAL, MAX_STEPS, seed=42)

    print('\n=== Uniform (running...) ===')
    uniform_recs = run_method('uniform', NUM_EPISODES, SWITCH_INTERVAL, MAX_STEPS, seed=42)

    stats = print_stats(bamor_recs, uniform_recs)

    fig_path  = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'experiments', 'figures', 'fig_cone_suppression.png'))
    json_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'experiments', 'cone_suppression_stats.json'))

    plot_results(bamor_recs, uniform_recs, fig_path)

    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'Stats  → {json_path}')
    print('\nDone.')
