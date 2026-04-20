#!/usr/bin/env python3
"""
BAMOR vs Baselines on SyntheticMOEnv (fast convergence, clean demonstration)

This is the PRIMARY experiment for the paper — SyntheticMOEnv is designed to
cleanly exhibit substrate constraints and piecewise Pareto sub-manifolds.
MuJoCo experiments serve as scalability validation.

Usage:
  python train_synthetic_bamor.py --method all --max_episodes 500 --seed 0
"""

import os, sys, json, time, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_mo_env import SyntheticMOEnv, ModeConfig as SynModeConfig
from bamor_core import BAMORPipeline


# ================================================================
# Networks (small — synthetic env is 2D state, 1D action)
# ================================================================

class SmallQNet(nn.Module):
    def __init__(self, state_dim, action_dim, weight_dim, reward_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + weight_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, reward_dim),
        )
    def forward(self, s, a, w):
        return self.net(torch.cat([s, a, w], dim=-1))


class SmallPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, weight_dim, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + weight_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, s, w):
        h = self.trunk(torch.cat([s, w], dim=-1))
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-20, 2)
        return mean, log_std

    def get_action(self, s, w, deterministic=False):
        mean, log_std = self.forward(s, w)
        if deterministic:
            return torch.tanh(mean)
        dist = Normal(mean, log_std.exp())
        x = dist.rsample()
        action = torch.tanh(x)
        log_prob = (dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return action, log_prob


# ================================================================
# Simple buffer
# ================================================================

class Buffer:
    def __init__(self, cap, s_dim, a_dim, w_dim, r_dim):
        self.cap, self.ptr, self.size = cap, 0, 0
        self.s  = np.zeros((cap, s_dim), np.float32)
        self.a  = np.zeros((cap, a_dim), np.float32)
        self.w  = np.zeros((cap, w_dim), np.float32)
        self.r  = np.zeros((cap, r_dim), np.float32)
        self.sn = np.zeros((cap, s_dim), np.float32)
        self.d  = np.zeros((cap, 1), np.float32)

    def push(self, s, a, w, r, sn, d):
        i = self.ptr
        self.s[i], self.a[i], self.w[i] = s, a, w
        self.r[i], self.sn[i], self.d[i] = r, sn, float(d)
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n, device):
        idx = np.random.randint(0, self.size, n)
        to = lambda x: torch.FloatTensor(x[idx]).to(device)
        return to(self.s), to(self.a), to(self.w), to(self.r), to(self.sn), to(self.d)


# ================================================================
# Training
# ================================================================

def create_env(switch_interval=300, max_steps=200, seed=0):
    """3 modes with structurally different reward functions.

    Each mode's Pareto front lives in a DIFFERENT region of objective space:
      Mode A (safety): high safety baseline, progress capped → top-left
      Mode B (speed):  low safety, high progress ceiling    → bottom-right
      Mode C (balanced): moderate both                      → middle

    This creates non-overlapping Piecewise Local Pareto Sub-manifolds.
    """
    import numpy as np

    import math

    # Key design: each mode has DIFFERENT reward functions that create
    # non-overlapping Pareto arcs. The speed_multiplier controls the
    # REACHABLE position range, which combined with the reward shape
    # determines where each mode's arc sits in objective space.
    #
    # Critical: speed must be moderate (not too fast!) so the agent
    # can control its position and the w→position→reward mapping
    # produces a smooth arc, not just a single saturated point.
    # Key design: each mode's "responsive" objective depends on position,
    # while its "collapsed" objective depends on action (independent of pos).
    # This creates clear per-objective variance asymmetry for TargetConeEstimator:
    #   responsive obj → high std (varies with position),
    #   collapsed obj  → low std  (varies only with action, small scale).
    # Non-overlapping arcs are preserved by different pos→reward mappings per mode.
    modes = [
        SynModeConfig(
            name='safety_zone',
            reward_scale=np.array([1.0, 1.0]),
            speed_multiplier=0.8,           # moderate: can reach pos≈3-4
            noise_std=0.02,
            true_anchor=np.array([0.85, 0.15]),
            reward_fns=[
                # Safety: RESPONSIVE — high baseline near origin, varies with pos
                lambda pos, act: 3.0 * math.exp(-0.04 * pos**2),
                # Progress: COLLAPSED — depends on action, tiny scale
                lambda pos, act: 0.3 * (act + 1.0) / 2.0,
            ],
        ),
        SynModeConfig(
            name='speed_zone',
            reward_scale=np.array([1.0, 1.0]),
            speed_multiplier=1.2,           # moderate-fast: can reach pos≈5-7
            noise_std=0.04,
            true_anchor=np.array([0.15, 0.85]),
            reward_fns=[
                # Safety: COLLAPSED — depends on action, tiny scale
                lambda pos, act: 0.5 * (1.0 - (act + 1.0) / 2.0),
                # Progress: RESPONSIVE — strong log, varies with pos
                lambda pos, act: 2.0 * math.log(max(pos, 0.01) + 1),
            ],
        ),
        SynModeConfig(
            name='balanced',
            reward_scale=np.array([1.0, 1.0]),
            speed_multiplier=1.0,           # normal speed
            noise_std=0.03,
            true_anchor=np.array([0.50, 0.50]),
            reward_fns=[
                # Safety: MODERATE — varies with pos
                lambda pos, act: 1.5 * math.exp(-0.05 * (pos - 1.5)**2),
                # Progress: MODERATE — varies with pos (both responsive)
                lambda pos, act: 1.2 * math.sqrt(max(pos, 0) + 0.01),
            ],
        ),
    ]
    return SyntheticMOEnv(modes=modes, switch_interval=switch_interval,
                          max_steps=max_steps, seed=seed)


def get_canonical_modes():
    """Return the canonical 3-mode config used in all experiments.

    Exported so external scripts (e.g. train_compare_baselines.py) can
    create environments with identical modes.
    """
    return create_env().modes


def train(method, max_episodes=500, switch_interval=300, seed=0,
          save_dir='results/synthetic', device='cpu'):
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = torch.device(device)

    env = create_env(switch_interval=switch_interval, seed=seed)
    eval_env = create_env(switch_interval=switch_interval, seed=seed+1000)

    s_dim = env.observation_space.shape[0]  # 2
    a_dim = env.action_space.shape[0]       # 1
    r_dim = 2
    gamma, tau, alpha, lr, bs = 0.99, 0.005, 0.2, 3e-4, 128
    anchor_eta = 0.02

    # Networks
    q1 = SmallQNet(s_dim, a_dim, r_dim, r_dim).to(dev)
    q2 = SmallQNet(s_dim, a_dim, r_dim, r_dim).to(dev)
    q1t = SmallQNet(s_dim, a_dim, r_dim, r_dim).to(dev)
    q2t = SmallQNet(s_dim, a_dim, r_dim, r_dim).to(dev)
    q1t.load_state_dict(q1.state_dict())
    q2t.load_state_dict(q2.state_dict())
    pi = SmallPolicy(s_dim, a_dim, r_dim).to(dev)

    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    pi_opt = torch.optim.Adam(pi.parameters(), lr=lr)
    buf = Buffer(100_000, s_dim, a_dim, r_dim, r_dim)

    # BAMOR pipeline
    bamor = None
    if method.startswith('bamor'):
        bamor = BAMORPipeline(
            num_modes=3, reward_dim=r_dim, gamma=gamma,
            surprise_threshold=5.0, concentration=20.0,
            disable_surprise=(method == 'bamor_ns'),
            disable_cone=(method == 'bamor_nc'),
        )

    # Logging
    log = {'method': method, 'seed': seed, 'episodes': [], 'pareto': [],
           'bamor_metrics': [], 'mode_switches': [], 'hv': []}
    # Shared reference point estimated from first eval, reused for consistency
    _ref_point = None

    total_steps = 0
    last_mode = 0

    for ep in range(max_episodes):
        obs, info = env.reset()
        ep_reward = np.zeros(r_dim)
        ep_steps = 0

        done = False
        while not done:
            # Sample weight
            if bamor:
                w_np, w_anchor_np = bamor.sample_weight()
            elif method == 'uniform':
                w_np = np.random.dirichlet(np.ones(r_dim)).astype(np.float32)
                w_anchor_np = None
            elif method == 'fixed':
                w_np = np.array([0.5, 0.5], dtype=np.float32)
                w_anchor_np = None
            else:
                w_np = np.random.dirichlet(np.ones(r_dim)).astype(np.float32)
                w_anchor_np = None

            # Act
            if total_steps < 200:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    st = torch.FloatTensor(obs).unsqueeze(0).to(dev)
                    wt = torch.FloatTensor(w_np).unsqueeze(0).to(dev)
                    action, _ = pi.get_action(st, wt)
                    action = action.cpu().numpy().flatten()

            nobs, rew, term, trunc, info = env.step(action)
            done = term or trunc
            buf.push(obs, action, w_np, rew, nobs, done)
            if bamor:
                bamor.observe_reward(rew)

            # Mode switch tracking
            cur_mode = info.get('mode_idx', 0)
            if cur_mode != last_mode:
                log['mode_switches'].append({'step': total_steps, 'from': last_mode, 'to': cur_mode})
                last_mode = cur_mode

            ep_reward += rew
            ep_steps += 1
            total_steps += 1
            obs = nobs

            # Train (every 2 steps to reduce overhead)
            if buf.size >= bs and total_steps >= 200 and total_steps % 4 == 0:
                s, a, w, r, sn, d = buf.sample(bs, dev)

                with torch.no_grad():
                    na, nlp = pi.get_action(sn, w)
                    q1n = q1t(sn, na, w)
                    q2n = q2t(sn, na, w)
                    qn = torch.min(q1n, q2n)
                    qn_s = (w * qn).sum(-1, keepdim=True)
                    tgt_s = (w * r).sum(-1, keepdim=True) + gamma * (1-d) * (qn_s - alpha * nlp)
                    # Feed per-objective target Q to diagnostics for Δl/Δg computation
                    if bamor:
                        bamor.record_target_q(qn.mean(0).cpu().numpy())

                q1v = (w * q1(s, a, w)).sum(-1, keepdim=True)
                q2v = (w * q2(s, a, w)).sum(-1, keepdim=True)
                qloss = F.mse_loss(q1v, tgt_s) + F.mse_loss(q2v, tgt_s)
                q_opt.zero_grad(); qloss.backward(); q_opt.step()

                na2, lp2 = pi.get_action(s, w)
                q_new = torch.min(q1(s, na2, w), q2(s, na2, w))
                q_s = (w * q_new).sum(-1, keepdim=True)
                ploss = (alpha * lp2 - q_s).mean()
                if w_anchor_np is not None and method == 'bamor':
                    wa = torch.FloatTensor(w_anchor_np).unsqueeze(0).expand(bs, -1).to(dev)
                    ploss = ploss + anchor_eta * ((w - wa)**2).sum(-1).mean()
                pi_opt.zero_grad(); ploss.backward(); pi_opt.step()

                for p, pt in zip(q1.parameters(), q1t.parameters()):
                    pt.data.lerp_(p.data, tau)
                for p, pt in zip(q2.parameters(), q2t.parameters()):
                    pt.data.lerp_(p.data, tau)

        log['episodes'].append({
            'ep': ep, 'step': total_steps,
            'return': ep_reward.tolist(),
            'mode': info.get('mode_name', '?'),
            'length': ep_steps,
        })

        # BAMOR metrics
        if bamor and total_steps > 400:
            m = bamor.get_metrics(w_np)
            log['bamor_metrics'].append({
                'ep': ep,
                **{k: v.tolist() if hasattr(v, 'tolist') else float(v) for k, v in m.items()}
            })

        # Periodic eval
        if (ep + 1) % 20 == 0:
            pareto, hv, _ref_point = eval_pareto(eval_env, pi, r_dim, dev,
                                                  ref_point=_ref_point)
            log['pareto'].append({'ep': ep, 'step': total_steps, 'front': pareto})
            log['hv'].append({'ep': ep, 'step': total_steps, 'hv': hv})
            returns = np.array([p['return'] for p in pareto])
            corr = np.corrcoef(returns[:,0], returns[:,1])[0,1] if len(returns) > 2 else 0
            recent = [e['return'] for e in log['episodes'][-10:]]
            avg_r = np.mean(recent, axis=0) if recent else [0,0]

            dl = log['bamor_metrics'][-1].get('delta_local', 0) if log['bamor_metrics'] else 0
            dg = log['bamor_metrics'][-1].get('delta_global', 0) if log['bamor_metrics'] else 0
            if isinstance(dl, list): dl = sum(dl)
            if isinstance(dg, list): dg = sum(dg)
            ratio = dl / (dg + 1e-8)

            print(f'  [{method}] ep={ep+1:>4d} step={total_steps:>6d} '
                  f'avg_r=[{avg_r[0]:+.1f},{avg_r[1]:+.1f}] '
                  f'corr={corr:+.2f} HV={hv:.1f} '
                  f'Δl/Δg={dl:.3f}/{dg:.3f}({ratio:.2f}) '
                  f'switches={len(log["mode_switches"])}')

    # Final eval with per-mode breakdown (lock mode during eval)
    print(f'\n  [{method}] Final per-mode Pareto:')
    for mode_idx in range(env.num_modes):
        eval_env._lock_mode = True
        eval_env.current_mode_idx = mode_idx
        pareto_mode, hv_mode, _ = eval_pareto(eval_env, pi, r_dim, dev,
                                               num_w=11, ref_point=_ref_point)
        eval_env._lock_mode = False
        returns = np.array([p['return'] for p in pareto_mode])
        corr = np.corrcoef(returns[:,0], returns[:,1])[0,1] if len(returns) > 2 else 0
        log['pareto'].append({'ep': max_episodes, 'mode': mode_idx,
                              'front': pareto_mode, 'hv': hv_mode})
        print(f'    Mode {mode_idx}: safety=[{returns[:,0].min():+.2f},{returns[:,0].max():+.2f}] '
              f'progress=[{returns[:,1].min():+.2f},{returns[:,1].max():+.2f}] '
              f'corr={corr:+.2f} HV={hv_mode:.2f}')

    # Save
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{method}_seed{seed}.json')
    with open(path, 'w') as f:
        json.dump(log, f, indent=2, default=str)
    model_path = os.path.join(save_dir, f'{method}_seed{seed}_model.pt')
    torch.save({'pi': pi.state_dict(), 'q1': q1.state_dict(), 'q2': q2.state_dict()}, model_path)
    print(f'  [{method}] Saved to {path}')
    env.close(); eval_env.close()
    return log


def compute_hv_2d(returns, ref_point):
    """Hypervolume indicator for a 2D reward set relative to ref_point.

    Works on raw reward points (not necessarily Pareto-optimal) by first
    filtering to non-dominated points above the reference, then sweeping.

    Args:
        returns:    np.ndarray of shape (N, 2)
        ref_point:  (r0, r1) — worst-case reference (all Pareto points dominate it)
    Returns:
        hv: float ≥ 0
    """
    pts = np.asarray(returns, dtype=np.float64)
    ref = np.asarray(ref_point, dtype=np.float64)

    # Keep points that dominate the reference
    mask = (pts[:, 0] > ref[0]) & (pts[:, 1] > ref[1])
    pts = pts[mask]
    if len(pts) == 0:
        return 0.0

    # Filter dominated points (non-dominated front)
    nd = []
    for p in pts:
        if not any(np.all(q >= p) and np.any(q > p) for q in pts):
            nd.append(p)
    if not nd:
        return 0.0

    # Sort by obj0 descending, sweep in obj1 direction
    nd = np.array(sorted(nd, key=lambda p: p[0], reverse=True))
    hv = 0.0
    max_obj1 = ref[1]
    for p in nd:
        if p[1] > max_obj1:
            hv += (p[0] - ref[0]) * (p[1] - max_obj1)
            max_obj1 = p[1]
    return float(hv)


def eval_pareto(env, pi, r_dim, dev, num_w=21, num_ep=3, ref_point=None):
    wg = np.linspace(0.01, 0.99, num_w)
    weights = np.stack([1-wg, wg], axis=-1).astype(np.float32)
    results = []
    for w in weights:
        ep_rets = []
        for _ in range(num_ep):
            obs, _ = env.reset()
            tr = np.zeros(r_dim)
            for _ in range(500):
                with torch.no_grad():
                    st = torch.FloatTensor(obs).unsqueeze(0).to(dev)
                    wt = torch.FloatTensor(w).unsqueeze(0).to(dev)
                    a = pi.get_action(st, wt, deterministic=True)
                    if isinstance(a, tuple): a = a[0]
                obs, rew, term, trunc, _ = env.step(a.cpu().numpy().flatten())
                tr += rew
                if term or trunc: break
            ep_rets.append(tr)
        results.append({'weight': w.tolist(), 'return': np.mean(ep_rets, axis=0).tolist()})

    # Compute hypervolume
    all_returns = np.array([p['return'] for p in results])
    if ref_point is None:
        # Conservative reference: min across all points minus 10% margin
        ref_point = (all_returns[:, 0].min() - abs(all_returns[:, 0].min()) * 0.1 - 1.0,
                     all_returns[:, 1].min() - abs(all_returns[:, 1].min()) * 0.1 - 1.0)
    hv = compute_hv_2d(all_returns, ref_point)
    return results, hv, ref_point


METHODS = ['bamor', 'bamor_nc', 'bamor_ns', 'uniform', 'fixed']

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', default='all', choices=METHODS+['all'])
    p.add_argument('--max_episodes', type=int, default=500)
    p.add_argument('--switch_interval', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num_seeds', type=int, default=3)
    p.add_argument('--save_dir', default='results/synthetic')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    methods = METHODS if args.method == 'all' else [args.method]
    for seed in range(args.seed, args.seed + args.num_seeds):
        for method in methods:
            print(f'\n{"="*60}')
            print(f'  {method}  seed={seed}')
            print(f'{"="*60}')
            train(method, args.max_episodes, args.switch_interval, seed, args.save_dir, args.device)

    print(f'\n✅ All done. Results in {args.save_dir}/')


if __name__ == '__main__':
    main()
