#!/usr/bin/env python3
"""
train_compare_baselines.py — Unified BAMOR vs. Baselines Comparison

Methods:
  Single-policy weight-conditioned (fair comparison):
    bamor       : BAMOR (Target Cone + belief update)
    bamor_nc    : BAMOR ablation — no Target Cone (uniform sampling)
    bamor_ns    : BAMOR ablation — no surprise detection
    capql       : CAPQL (ICLR 2023), uniform weight sampling
    uniform     : uniform Dirichlet weight sampling
    fixed       : fixed weight [0.5, 0.5]

  Multi-policy weight-conditioned (extended comparison):
    gpi_pd      : GPI-PD continuous (NeurIPS 2023), maintains weight support set

All methods share:
  - Same SyntheticMOEnv with identical mode-switch schedule
  - Same total_timesteps budget
  - Same eval protocol: HV over 21 weight vectors, repeated 3 episodes
  - Same ref_point for HV (estimated from first run, shared across methods)

Usage:
  python train_compare_baselines.py --total_steps 50000 --seed 0
  python train_compare_baselines.py --total_steps 50000 --seeds 5 --method all
"""

import os, sys, json, argparse, math
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Auto-add morl-baselines to path — search up to 5 parent levels
def _find_morl_baselines():
    p = Path(__file__).resolve().parent
    for _ in range(5):
        candidate = p / 'morl-baselines'
        if candidate.exists():
            return candidate
        p = p.parent
    return None
_morl_path = _find_morl_baselines()
if _morl_path and str(_morl_path) not in sys.path:
    sys.path.insert(0, str(_morl_path))

from synthetic_mo_env import SyntheticMOEnv, ModeConfig
from bamor_core import BAMORPipeline
from train_synthetic_bamor import (
    SmallQNet, SmallPolicy, Buffer,
    compute_hv_2d, eval_pareto,
    create_env, get_canonical_modes,
)

# ── morl-baselines import guard ──────────────────────────────────────────────
MORL_BASELINES_PATH = _morl_path  # already resolved above


# ════════════════════════════════════════════════════════════════════════════
# mo-gymnasium wrapper  (required by CAPQL / GPI-PD)
# ════════════════════════════════════════════════════════════════════════════

import gymnasium as gym
from gymnasium import spaces

# Fixed conservative reference point for cross-method overall HV comparison.
# Synthetic env rewards are non-negative (Gaussian fns, ≥0 per step).
# GLOBAL_REF is set well below any policy's per-episode return so that
# even a random policy gets nonzero HV and comparisons are on a common scale.
GLOBAL_REF = (-5.0, -5.0)


class MOGymWrapper(gym.Env):
    """Adapts SyntheticMOEnv to the mo-gymnasium API (reward_space attribute).

    CAPQL and GPI-PD require:
        env.reward_space : Box(reward_dim,)
        env.reward_dim   : int
        step()           → 5-tuple with np.ndarray reward
    """
    metadata = {"render_modes": []}

    def __init__(self, switch_interval=300, max_steps=200, seed=None, modes=None):
        super().__init__()
        kwargs = dict(switch_interval=switch_interval, max_steps=max_steps, seed=seed)
        if modes is not None:
            kwargs['modes'] = modes
        self._env = SyntheticMOEnv(**kwargs)
        self.observation_space = self._env.observation_space
        self.action_space      = self._env.action_space
        self.reward_space      = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self._env.reward_dim,), dtype=np.float64)
        self.reward_dim        = self._env.reward_dim
        self.num_modes         = self._env.num_modes

    # Pass-through mode locking for per-mode eval
    @property
    def _lock_mode(self):
        return getattr(self._env, '_lock_mode', False)
    @_lock_mode.setter
    def _lock_mode(self, v):
        self._env._lock_mode = v

    @property
    def current_mode_idx(self):
        return self._env.current_mode_idx
    @current_mode_idx.setter
    def current_mode_idx(self, v):
        self._env.current_mode_idx = v

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        obs, rew, term, trunc, info = self._env.step(action)
        return obs, np.asarray(rew, dtype=np.float64), term, trunc, info

    def close(self):
        self._env.close()


# ════════════════════════════════════════════════════════════════════════════
# Unified eval: works for BAMOR (policy net) and morl-baselines agents
# ════════════════════════════════════════════════════════════════════════════

def eval_front_bamor(env, pi, r_dim, dev, num_w=21, num_ep=3, ref_point=None):
    """Evaluate BAMOR policy (SmallPolicy) → returns (results, hv, ref_point)."""
    return eval_pareto(env, pi, r_dim, dev, num_w=num_w, num_ep=num_ep,
                       ref_point=ref_point)


def eval_front_agent(env, agent, r_dim, num_w=21, num_ep=3, ref_point=None):
    """Evaluate a morl-baselines agent (has .eval(obs, w) → action)."""
    wg = np.linspace(0.01, 0.99, num_w)
    weights = np.stack([1 - wg, wg], axis=-1).astype(np.float32)
    results = []
    for w in weights:
        ep_rets = []
        for _ in range(num_ep):
            obs, _ = env.reset()
            tr = np.zeros(r_dim)
            done = False
            while not done:
                action = agent.eval(obs, w)
                obs, rew, term, trunc, _ = env.step(action)
                tr += np.asarray(rew)
                done = term or trunc
            ep_rets.append(tr)
        results.append({'weight': w.tolist(),
                        'return': np.mean(ep_rets, axis=0).tolist()})

    all_returns = np.array([p['return'] for p in results])
    if ref_point is None:
        ref_point = (all_returns[:, 0].min() - abs(all_returns[:, 0].min()) * 0.1 - 1.0,
                     all_returns[:, 1].min() - abs(all_returns[:, 1].min()) * 0.1 - 1.0)
    hv = compute_hv_2d(all_returns, ref_point)
    return results, hv, ref_point


def eval_per_mode(env_or_wrapper, eval_fn, num_modes, r_dim, ref_point=None):
    """Run eval locked per mode, return list of (results, hv) per mode."""
    per_mode = []
    for m in range(num_modes):
        env_or_wrapper._lock_mode = True
        env_or_wrapper.current_mode_idx = m
        out = eval_fn(ref_point=ref_point)
        env_or_wrapper._lock_mode = False
        per_mode.append(out)
        if ref_point is None:
            ref_point = out[2]   # reuse ref from first mode
    return per_mode


# ════════════════════════════════════════════════════════════════════════════
# BAMOR / Uniform / Fixed training  (reuses train_synthetic_bamor logic)
# ════════════════════════════════════════════════════════════════════════════

def train_bamor_family(method, total_steps, switch_interval, seed, save_dir,
                       device='cpu', ref_point=None):
    """Train one of: bamor, bamor_nc, bamor_ns, uniform, fixed."""
    np.random.seed(seed); torch.manual_seed(seed)
    dev = torch.device(device)

    env      = create_env(switch_interval=switch_interval, seed=seed)
    eval_env = create_env(switch_interval=switch_interval, seed=seed + 1000)

    s_dim, a_dim, r_dim = 2, 1, 2
    gamma, tau, alpha, lr, bs = 0.99, 0.005, 0.2, 3e-4, 128
    anchor_eta = 0.02

    q1  = SmallQNet(s_dim, a_dim, r_dim, r_dim).to(dev)
    q2  = SmallQNet(s_dim, a_dim, r_dim, r_dim).to(dev)
    q1t = SmallQNet(s_dim, a_dim, r_dim, r_dim).to(dev); q1t.load_state_dict(q1.state_dict())
    q2t = SmallQNet(s_dim, a_dim, r_dim, r_dim).to(dev); q2t.load_state_dict(q2.state_dict())
    pi  = SmallPolicy(s_dim, a_dim, r_dim).to(dev)
    q_opt  = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    pi_opt = torch.optim.Adam(pi.parameters(), lr=lr)
    buf    = Buffer(100_000, s_dim, a_dim, r_dim, r_dim)

    bamor = None
    if method.startswith('bamor'):
        bamor = BAMORPipeline(num_modes=3, reward_dim=r_dim, gamma=gamma,
                              surprise_threshold=7.0, concentration=20.0,
                              switch_cooldown=290, warmup=150,
                              disable_surprise=(method == 'bamor_ns'),
                              disable_cone=(method == 'bamor_nc'))

    log = {'method': method, 'seed': seed, 'hv_curve': [], 'mode_switches': []}
    obs, info = env.reset()
    ep_r = np.zeros(r_dim)
    last_mode = 0
    _ref = ref_point

    for step in range(total_steps):
        # Weight sampling
        if bamor:
            w_np, w_anchor_np = bamor.sample_weight()
        elif method == 'uniform':
            w_np = np.random.dirichlet(np.ones(r_dim)).astype(np.float32)
            w_anchor_np = None
        else:
            w_np = np.array([0.5, 0.5], np.float32)
            w_anchor_np = None

        # Action
        if step < 200:
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

        cur_mode = info.get('mode_idx', 0)
        if cur_mode != last_mode:
            log['mode_switches'].append(step)
            last_mode = cur_mode

        ep_r += rew
        if done:
            obs, info = env.reset()
            ep_r[:] = 0
        else:
            obs = nobs

        # Train
        if buf.size >= bs and step >= 200 and step % 4 == 0:
            s, a, w, r, sn, d = buf.sample(bs, dev)
            with torch.no_grad():
                na, nlp = pi.get_action(sn, w)
                q1n = q1t(sn, na, w); q2n = q2t(sn, na, w)
                qn  = torch.min(q1n, q2n)
                if bamor:
                    bamor.record_target_q(qn.mean(0).cpu().numpy())
                qn_s  = (w * qn).sum(-1, keepdim=True)
                tgt_s = (w * r).sum(-1, keepdim=True) + gamma * (1-d) * (qn_s - alpha * nlp)
            q1v = (w * q1(s, a, w)).sum(-1, keepdim=True)
            q2v = (w * q2(s, a, w)).sum(-1, keepdim=True)
            qloss = F.mse_loss(q1v, tgt_s) + F.mse_loss(q2v, tgt_s)
            q_opt.zero_grad(); qloss.backward(); q_opt.step()

            na2, lp2 = pi.get_action(s, w)
            q_s   = (w * torch.min(q1(s, na2, w), q2(s, na2, w))).sum(-1, keepdim=True)
            ploss = (alpha * lp2 - q_s).mean()
            if w_anchor_np is not None and method == 'bamor':
                wa    = torch.FloatTensor(w_anchor_np).unsqueeze(0).expand(bs, -1).to(dev)
                ploss = ploss + anchor_eta * ((w - wa)**2).sum(-1).mean()
            pi_opt.zero_grad(); ploss.backward(); pi_opt.step()

            for p, pt in zip(q1.parameters(), q1t.parameters()): pt.data.lerp_(p.data, tau)
            for p, pt in zip(q2.parameters(), q2t.parameters()): pt.data.lerp_(p.data, tau)

        # Periodic eval
        if (step + 1) % (total_steps // 10) == 0:
            pareto, hv, _ref = eval_pareto(eval_env, pi, r_dim, dev, ref_point=_ref)
            log['hv_curve'].append({'step': step + 1, 'hv': hv})
            if bamor:
                m = bamor.get_metrics(w_np)
                dl, dg = float(m.get('delta_local', 0)), float(m.get('delta_global', 0))
                ratio  = dl / (dg + 1e-8)
            else:
                dl, dg, ratio = 0., 0., 0.
            print(f'  [{method}|s={seed}] step={step+1:>6d} HV={hv:.1f} '
                  f'Δl/Δg={dl:.3f}/{dg:.3f}({ratio:.2f})')

    # Final eval (overall + per-mode)
    overall, hv_overall, _ref = eval_pareto(eval_env, pi, r_dim, dev,
                                             num_w=21, ref_point=_ref)
    log['hv_overall'] = hv_overall
    log['pareto_overall'] = overall
    log['pareto_per_mode'] = []
    log['hv_per_mode'] = []

    for m in range(env.num_modes):
        eval_env._lock_mode = True
        eval_env.current_mode_idx = m
        # Per-mode: use GLOBAL_REF so HV values are comparable across methods
        pm, hv_m, _ = eval_pareto(eval_env, pi, r_dim, dev, num_w=11, ref_point=GLOBAL_REF)
        eval_env._lock_mode = False
        returns = np.array([p['return'] for p in pm])
        corr = np.corrcoef(returns[:, 0], returns[:, 1])[0, 1] if len(returns) > 2 else 0
        log['pareto_per_mode'].append(pm)
        log['hv_per_mode'].append(hv_m)
        print(f'    Mode {m}: HV={hv_m:.2f}  corr={corr:+.2f}')

    # Diagnostic: BAMOR's own detected-switch count (for comparison with env's true switches)
    if bamor is not None:
        log['bamor_switches_detected'] = int(getattr(bamor.belief_tracker, 'mode_switch_count', -1))
        log['env_switches_true'] = len(log['mode_switches'])
        print(f'    BAMOR detected {log["bamor_switches_detected"]} switches (env truth: {log["env_switches_true"]})')

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{method}_seed{seed}.json')
    with open(path, 'w') as f:
        json.dump(log, f, indent=2, default=str)
    print(f'  Saved → {path}')
    env.close(); eval_env.close()
    return log, _ref


# ════════════════════════════════════════════════════════════════════════════
# CAPQL wrapper
# ════════════════════════════════════════════════════════════════════════════

def train_capql(total_steps, switch_interval, seed, save_dir,
                ref_point=None, device='cpu'):
    try:
        from morl_baselines.multi_policy.capql.capql import CAPQL
    except ImportError as _e:
        print(f'  [capql] import failed ({_e}) — skipping')
        return None, ref_point

    _modes = get_canonical_modes()
    env      = MOGymWrapper(switch_interval=switch_interval, seed=seed,      modes=_modes)
    eval_env = MOGymWrapper(switch_interval=switch_interval, seed=seed+1000, modes=_modes)
    r_dim    = env.reward_dim

    _ref_np = np.array(ref_point) if ref_point is not None else np.array(GLOBAL_REF)

    agent = CAPQL(env, learning_rate=3e-4, gamma=0.99, batch_size=128,
                  net_arch=[64, 64], log=False, seed=seed)

    print(f'  [capql|s={seed}] training {total_steps} steps...')
    agent.train(total_timesteps=total_steps,
                eval_env=eval_env,
                ref_point=_ref_np,
                num_eval_weights_for_front=21,
                num_eval_episodes_for_front=3,
                eval_freq=max(total_steps // 10, 1000),
                checkpoints=False)

    # Eval
    overall, hv_overall, _ref = eval_front_agent(eval_env, agent, r_dim,
                                                  ref_point=ref_point)
    log = {'method': 'capql', 'seed': seed,
           'hv_overall': hv_overall, 'pareto_overall': overall,
           'pareto_per_mode': [], 'hv_per_mode': []}

    print(f'  [capql|s={seed}] HV={hv_overall:.1f}')
    for m in range(eval_env.num_modes):
        eval_env._lock_mode = True
        eval_env.current_mode_idx = m
        # Per-mode: use GLOBAL_REF so HV is comparable across methods
        pm, hv_m, _ = eval_front_agent(eval_env, agent, r_dim, ref_point=GLOBAL_REF)
        eval_env._lock_mode = False
        log['pareto_per_mode'].append(pm)
        log['hv_per_mode'].append(hv_m)
        print(f'    Mode {m}: HV={hv_m:.2f}')

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'capql_seed{seed}.json')
    with open(path, 'w') as f:
        json.dump(log, f, indent=2, default=str)
    print(f'  Saved → {path}')
    env.close(); eval_env.close()
    return log, _ref


# ════════════════════════════════════════════════════════════════════════════
# GPI-PD wrapper
# ════════════════════════════════════════════════════════════════════════════

def train_gpi_pd(total_steps, switch_interval, seed, save_dir,
                 ref_point=None, device='cpu'):
    try:
        from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import GPIPDContinuousAction
    except ImportError as _e:
        print(f'  [gpi_pd] import failed ({_e}) — skipping')
        print('  [gpi_pd] hint: install pycddlib==2.1.6 (requires libcdd-dev system library)')
        return None, ref_point

    _modes = get_canonical_modes()
    env      = MOGymWrapper(switch_interval=switch_interval, seed=seed,      modes=_modes)
    eval_env = MOGymWrapper(switch_interval=switch_interval, seed=seed+1000, modes=_modes)
    r_dim    = env.reward_dim

    _ref_np = np.array(ref_point) if ref_point is not None else np.array(GLOBAL_REF)

    agent = GPIPDContinuousAction(
        env,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=128,
        net_arch=[64, 64],
        buffer_size=100_000,
        learning_starts=200,
        gradient_updates=1,
        use_gpi=True,         # enable GPI weight selection
        log=False,
        seed=seed,
        device=device,
    )

    print(f'  [gpi_pd|s={seed}] training {total_steps} steps...')
    agent.train(total_timesteps=total_steps,
                eval_env=eval_env,
                ref_point=_ref_np,
                num_eval_weights_for_front=21,
                num_eval_episodes_for_front=3,
                timesteps_per_iter=max(total_steps // 10, 1000),
                eval_freq=max(total_steps // 10, 1000),
                checkpoints=False)

    # Eval
    overall, hv_overall, _ref = eval_front_agent(eval_env, agent, r_dim,
                                                  ref_point=ref_point)
    log = {'method': 'gpi_pd', 'seed': seed,
           'hv_overall': hv_overall, 'pareto_overall': overall,
           'pareto_per_mode': [], 'hv_per_mode': []}

    print(f'  [gpi_pd|s={seed}] HV={hv_overall:.1f}')
    for m in range(eval_env.num_modes):
        eval_env._lock_mode = True
        eval_env.current_mode_idx = m
        pm, hv_m, _ = eval_front_agent(eval_env, agent, r_dim, ref_point=_ref)
        eval_env._lock_mode = False
        log['pareto_per_mode'].append(pm)
        log['hv_per_mode'].append(hv_m)
        print(f'    Mode {m}: HV={hv_m:.2f}')

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'gpi_pd_seed{seed}.json')
    with open(path, 'w') as f:
        json.dump(log, f, indent=2, default=str)
    print(f'  Saved → {path}')
    env.close(); eval_env.close()
    return log, _ref


# ════════════════════════════════════════════════════════════════════════════
# Summary table
# ════════════════════════════════════════════════════════════════════════════

def print_summary(all_logs, num_modes):
    methods = list(all_logs.keys())
    print(f'\n{"="*65}')
    print(f'  Results summary  (mean ± std across seeds)')
    print(f'{"="*65}')
    print(f'  {"Method":<12} {"HV overall":>12}  ' +
          '  '.join(f'HV Mode{m}' for m in range(num_modes)))
    print(f'  {"-"*63}')
    for method, logs in all_logs.items():
        if not logs:
            continue
        hvs_overall = [l['hv_overall'] for l in logs if l and 'hv_overall' in l]
        hvs_mode    = [[l['hv_per_mode'][m] for l in logs if l and 'hv_per_mode' in l]
                       for m in range(num_modes)]
        if not hvs_overall:
            continue
        overall_str = f'{np.mean(hvs_overall):>8.1f}±{np.std(hvs_overall):<5.1f}'
        mode_strs   = [f'{np.mean(hm):>6.1f}±{np.std(hm):<4.1f}'
                       for hm in hvs_mode]
        print(f'  {method:<12} {overall_str}  ' + '  '.join(mode_strs))
    print(f'{"="*65}')


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

SINGLE_POLICY = ['bamor', 'bamor_nc', 'bamor_ns', 'capql', 'uniform', 'fixed']
MULTI_POLICY  = ['gpi_pd']
ALL_METHODS   = SINGLE_POLICY + MULTI_POLICY


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', default='all',
                   choices=ALL_METHODS + ['all', 'single_policy', 'multi_policy'])
    p.add_argument('--total_steps',    type=int, default=50_000)
    p.add_argument('--switch_interval',type=int, default=300)
    p.add_argument('--seed',           type=int, default=0)
    p.add_argument('--seeds',          type=int, default=3,
                   help='Number of seeds starting from --seed')
    p.add_argument('--save_dir',       default='results/compare')
    p.add_argument('--device',         default='cpu')
    args = p.parse_args()

    if args.method == 'all':
        methods = ALL_METHODS
    elif args.method == 'single_policy':
        methods = SINGLE_POLICY
    elif args.method == 'multi_policy':
        methods = MULTI_POLICY
    else:
        methods = [args.method]

    seeds = list(range(args.seed, args.seed + args.seeds))

    print(f'Methods : {methods}')
    print(f'Seeds   : {seeds}')
    print(f'Steps   : {args.total_steps}')
    print(f'Switch  : every {args.switch_interval} steps')
    print(f'Save    : {args.save_dir}')

    all_logs = {m: [] for m in methods}
    shared_ref = GLOBAL_REF   # fixed conservative ref for fair cross-method HV

    for seed in seeds:
        for method in methods:
            print(f'\n{"="*55}\n  {method}  seed={seed}\n{"="*55}')

            if method in ('bamor', 'bamor_nc', 'bamor_ns', 'uniform', 'fixed'):
                log, shared_ref = train_bamor_family(
                    method, args.total_steps, args.switch_interval,
                    seed, args.save_dir, args.device, ref_point=shared_ref)

            elif method == 'capql':
                log, shared_ref = train_capql(
                    args.total_steps, args.switch_interval,
                    seed, args.save_dir, ref_point=shared_ref, device=args.device)

            elif method == 'gpi_pd':
                log, shared_ref = train_gpi_pd(
                    args.total_steps, args.switch_interval,
                    seed, args.save_dir, ref_point=shared_ref, device=args.device)

            if log is not None:
                all_logs[method].append(log)

    # Save combined
    combined_path = os.path.join(args.save_dir, 'all_results.json')
    os.makedirs(args.save_dir, exist_ok=True)
    with open(combined_path, 'w') as f:
        json.dump(all_logs, f, indent=2, default=str)
    print(f'\nAll results → {combined_path}')

    # Print summary table
    num_modes = 3   # SyntheticMOEnv default
    print_summary(all_logs, num_modes)


if __name__ == '__main__':
    main()
