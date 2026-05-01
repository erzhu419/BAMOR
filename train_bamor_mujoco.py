#!/usr/bin/env python3
"""
BAMOR vs Baselines on Piecewise-Stationary Multi-Objective MuJoCo

Trains and evaluates:
  1. BAMOR      — full: surprise + Target Cone + anchor regularization
  2. BAMOR-NC   — ablation: no cone (uniform sampling, surprise still active)
  3. BAMOR-NS   — ablation: no surprise (cone active, no mode detection)
  4. Uniform    — MO-SAC with uniform Dirichlet weight sampling
  5. CAPQL      — from morl-baselines (uniform simplex, no mode awareness)

Usage:
  python train_bamor_mujoco.py --env mo-halfcheetah-v5 --method all --total_steps 500000
  python train_bamor_mujoco.py --env mo-hopper-v5 --method bamor --total_steps 1000000
  python train_bamor_mujoco.py --env mo-halfcheetah-v5 --method capql --total_steps 500000
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# --- Project imports ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from piecewise_mo_env import PiecewiseStationaryMOEnv, make_piecewise_mo_env
from bamor_core import BAMORPipeline
from train_synthetic_bamor import compute_hv_2d


# ================================================================
# Networks (weight-conditioned MO-SAC)
# ================================================================

class SoftQNet(nn.Module):
    """Q(s, a, w) → R^reward_dim.  Weight-conditioned multi-objective critic."""

    def __init__(self, state_dim, action_dim, weight_dim, reward_dim, hidden=256):
        super().__init__()
        in_dim = state_dim + action_dim + weight_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, reward_dim),
        )

    def forward(self, state, action, weight):
        x = torch.cat([state, action, weight], dim=-1)
        return self.net(x)


class PolicyNet(nn.Module):
    """π(a|s, w).  Weight-conditioned Gaussian policy with tanh squashing."""

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, state_dim, action_dim, weight_dim, hidden=256):
        super().__init__()
        in_dim = state_dim + weight_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, state, weight):
        h = self.trunk(torch.cat([state, weight], dim=-1))
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def get_action(self, state, weight, deterministic=False):
        mean, log_std = self.forward(state, weight)
        if deterministic:
            return torch.tanh(mean)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


# ================================================================
# Replay Buffer
# ================================================================

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, weight_dim, reward_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.weights = np.zeros((capacity, weight_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, reward_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, s, a, w, r, s_next, done):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.weights[self.ptr] = w
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s_next
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idx]).to(self.device),
            torch.FloatTensor(self.actions[idx]).to(self.device),
            torch.FloatTensor(self.weights[idx]).to(self.device),
            torch.FloatTensor(self.rewards[idx]).to(self.device),
            torch.FloatTensor(self.next_states[idx]).to(self.device),
            torch.FloatTensor(self.dones[idx]).to(self.device),
        )


# ================================================================
# Trainer
# ================================================================

class MOSACTrainer:
    """Weight-conditioned multi-objective SAC trainer."""

    def __init__(self, state_dim, action_dim, reward_dim, hidden=256,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 buffer_capacity=500_000, batch_size=256, device="auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.reward_dim = reward_dim

        # Networks
        self.q1 = SoftQNet(state_dim, action_dim, reward_dim, reward_dim, hidden).to(self.device)
        self.q2 = SoftQNet(state_dim, action_dim, reward_dim, reward_dim, hidden).to(self.device)
        self.q1_target = SoftQNet(state_dim, action_dim, reward_dim, reward_dim, hidden).to(self.device)
        self.q2_target = SoftQNet(state_dim, action_dim, reward_dim, reward_dim, hidden).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.policy = PolicyNet(state_dim, action_dim, reward_dim, hidden).to(self.device)

        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Buffer
        self.buffer = ReplayBuffer(
            buffer_capacity, state_dim, action_dim, reward_dim, reward_dim, self.device
        )

    def update(self, w_anchor=None, anchor_eta=0.01):
        """One gradient step. Returns (mean_q, target_q_vec) for diagnostics."""
        if self.buffer.size < self.batch_size:
            return None, None

        s, a, w, r, s_next, done = self.buffer.sample(self.batch_size)

        # --- Critic update ---
        with torch.no_grad():
            next_a, next_log_prob = self.policy.get_action(s_next, w)
            q1_next = self.q1_target(s_next, next_a, w)
            q2_next = self.q2_target(s_next, next_a, w)
            q_next = torch.min(q1_next, q2_next)
            # Scalarize for entropy: ⟨w, Q_next⟩ - α log π
            q_next_scalar = (w * q_next).sum(dim=-1, keepdim=True)
            target_scalar = (w * r).sum(dim=-1, keepdim=True) + \
                            self.gamma * (1 - done) * (q_next_scalar - self.alpha * next_log_prob)
            # Per-objective target (for diagnostics)
            target_vec = r + self.gamma * (1 - done) * q_next

        q1_val = (w * self.q1(s, a, w)).sum(dim=-1, keepdim=True)
        q2_val = (w * self.q2(s, a, w)).sum(dim=-1, keepdim=True)
        q_loss = F.mse_loss(q1_val, target_scalar) + F.mse_loss(q2_val, target_scalar)

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # --- Policy update ---
        new_a, log_prob = self.policy.get_action(s, w)
        q1_new = self.q1(s, new_a, w)
        q2_new = self.q2(s, new_a, w)
        q_new = torch.min(q1_new, q2_new)
        q_scalar = (w * q_new).sum(dim=-1, keepdim=True)
        policy_loss = (self.alpha * log_prob - q_scalar).mean()

        # NOTE on anchor regularisation: the original ploss term was
        #   eta * ||w - w_anchor||^2
        # but neither w nor w_anchor depends on policy parameters, so it had
        # zero gradient and was a no-op (gpt diagnosis #5).  The paper's
        # "anchor-shifted regularisation" is a property of the *regularised
        # scalarisation* (Pareto-front geometry, §3.5 Thm), not a policy-loss
        # term.  Removing the no-op here for clarity.

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        # --- Target update ---
        for p, p_tgt in zip(self.q1.parameters(), self.q1_target.parameters()):
            p_tgt.data.lerp_(p.data, self.tau)
        for p, p_tgt in zip(self.q2.parameters(), self.q2_target.parameters()):
            p_tgt.data.lerp_(p.data, self.tau)

        mean_q = q_scalar.mean().item()
        target_q_mean = target_vec.mean(dim=0).detach().cpu().numpy()
        return mean_q, target_q_mean


# ================================================================
# Weight samplers
# ================================================================

def sample_uniform_weight(reward_dim):
    """Dirichlet(1,...,1) — uniform over simplex."""
    w = np.random.dirichlet(np.ones(reward_dim))
    return w.astype(np.float32), None  # no anchor


def sample_fixed_weight(reward_dim):
    """Fixed balanced weight."""
    w = np.ones(reward_dim, dtype=np.float32) / reward_dim
    return w, None


# ================================================================
# Evaluation
# ================================================================

def evaluate_pareto_with_weights(env, trainer, weights, num_episodes=3, device="cpu"):
    """Evaluate policy on the *given* weight grid → approximate Pareto front."""
    weights = np.asarray(weights, dtype=np.float32)
    results = []
    for w in weights:
        ep_returns = []
        for _ in range(num_episodes):
            obs, _ = env.reset()
            total_r = np.zeros(trainer.reward_dim)
            done = False
            steps = 0
            while not done and steps < 1000:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    w_t = torch.FloatTensor(w).unsqueeze(0).to(device)
                    action = trainer.policy.get_action(obs_t, w_t, deterministic=True)
                    if isinstance(action, tuple):
                        action = action[0]
                obs, rew, term, trunc, _ = env.step(action.cpu().numpy().flatten())
                total_r += rew
                done = term or trunc
                steps += 1
            ep_returns.append(total_r)
        mean_return = np.mean(ep_returns, axis=0)
        results.append({"weight": w.tolist(), "return": mean_return.tolist()})
    return results


def evaluate_pareto(env, trainer, num_weights=11, num_episodes=3, device="cpu"):
    """Evaluate policy on a uniform-simplex grid of weights → Pareto front."""
    if trainer.reward_dim == 2:
        weight_grid = np.linspace(0, 1, num_weights)
        weights = np.stack([weight_grid, 1 - weight_grid], axis=-1).astype(np.float32)
    else:
        weights = np.random.dirichlet(np.ones(trainer.reward_dim),
                                       size=num_weights).astype(np.float32)
    return evaluate_pareto_with_weights(env, trainer, weights,
                                         num_episodes=num_episodes, device=device)


# ================================================================
# Main training loop
# ================================================================

def train_method(
    env_name: str,
    method: str,
    total_steps: int,
    switch_interval: int,
    seed: int,
    save_dir: str,
    eval_freq: int = 20_000,
    log_freq: int = 1000,
    hidden: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    batch_size: int = 256,
    learning_starts: int = 5000,
    anchor_eta: float = 0.01,
    num_modes: int = 3,
    device: str = "auto",
):
    """Train a single method and save results."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- Environment ---
    env = make_piecewise_mo_env(env_name, switch_interval=switch_interval, seed=seed)
    eval_env = make_piecewise_mo_env(env_name, switch_interval=switch_interval, seed=seed + 1000)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    reward_dim = env.reward_dim
    print(f"[{method}] env={env_name}, obs={obs_dim}, act={act_dim}, rew={reward_dim}, modes={num_modes}")

    # --- Trainer ---
    trainer = MOSACTrainer(
        obs_dim, act_dim, reward_dim, hidden=hidden,
        lr=lr, gamma=gamma, batch_size=batch_size, device=device,
    )
    dev = trainer.device

    # --- BAMOR Pipeline ---
    # gpt diagnosis #1+#2: in MuJoCo, threshold=6 caused 134 spurious switches
    # per 20K steps. We now use threshold=18 + CUSUM-style required_consecutive=5
    # (defaults inside BeliefTracker), which empirically gave 0 spurious /
    # correct detection at the real switch point.  switch_cooldown is also
    # bumped from default 80 → 2000 because mujoco episodes are ~1000 steps
    # and the policy reward drift is far slower than 1D-synthetic dynamics.
    bamor = None
    if method.startswith("bamor"):
        bamor = BAMORPipeline(
            num_modes=num_modes,
            reward_dim=reward_dim,
            gamma=gamma,
            surprise_threshold=18.0,
            concentration=15.0,
            switch_cooldown=2000,
            warmup=2000,
            disable_surprise=(method == "bamor_ns"),
            disable_cone=(method == "bamor_nc"),
        )

    # --- Logging ---
    log_data = {
        "method": method, "env": env_name, "seed": seed,
        "total_steps": total_steps, "switch_interval": switch_interval,
        "steps": [], "episode_rewards": [], "mode_switches": [],
        "pareto_evals": [], "bamor_metrics": [],
        "ground_truth": env.get_ground_truth(),
    }

    # --- Training ---
    obs, info = env.reset()
    episode_reward = np.zeros(reward_dim)
    episode_steps = 0
    episode_count = 0
    mode_switch_count = 0
    last_mode = info["mode_idx"]

    start_time = time.time()

    for step in range(1, total_steps + 1):
        # Sample weight
        if bamor is not None:
            w_np, w_anchor_np = bamor.sample_weight()
        elif method == "uniform":
            w_np, w_anchor_np = sample_uniform_weight(reward_dim)
        elif method == "fixed":
            w_np, w_anchor_np = sample_fixed_weight(reward_dim)
        else:
            w_np, w_anchor_np = sample_uniform_weight(reward_dim)

        # Act
        if step < learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(dev)
                w_t = torch.FloatTensor(w_np).unsqueeze(0).to(dev)
                action, _ = trainer.policy.get_action(obs_t, w_t)
                action = action.cpu().numpy().flatten()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store
        trainer.buffer.push(obs, action, w_np, reward, next_obs, done)

        # BAMOR observe — capture belief BEFORE / AFTER to detect its own switch
        prev_bamor_mode = (bamor.belief_tracker.current_mode
                           if bamor is not None else None)
        if bamor is not None:
            bamor.observe_reward(reward)
        cur_bamor_mode = (bamor.belief_tracker.current_mode
                          if bamor is not None else None)

        # Track ENV mode switches (ground truth)
        cur_mode = info["mode_idx"]
        if cur_mode != last_mode:
            mode_switch_count += 1
            log_data["mode_switches"].append({
                "step": step, "from": last_mode, "to": cur_mode,
                "kind": "env_truth",
            })
            last_mode = cur_mode

        # Track BAMOR's OWN belief switches (gpt diagnosis #3 — these were
        # invisible before; spurious switches in the belief tracker cause
        # the cone estimator to be trained on wrongly-labelled samples).
        if (bamor is not None and prev_bamor_mode is not None
                and cur_bamor_mode != prev_bamor_mode):
            log_data["mode_switches"].append({
                "step": step, "from": int(prev_bamor_mode),
                "to": int(cur_bamor_mode), "kind": "bamor_belief",
            })

        episode_reward += reward
        episode_steps += 1

        # Update
        if step >= learning_starts:
            w_anchor_t = None
            if w_anchor_np is not None and method == "bamor":
                w_anchor_t = torch.FloatTensor(w_anchor_np).unsqueeze(0).to(dev)
                w_anchor_t = w_anchor_t.expand(batch_size, -1)

            mean_q, target_q_vec = trainer.update(
                w_anchor=w_anchor_t,
                anchor_eta=anchor_eta if method == "bamor" else 0.0,
            )

            if bamor is not None and target_q_vec is not None:
                bamor.record_target_q(target_q_vec)

        # Episode done
        if done:
            episode_count += 1
            log_data["episode_rewards"].append({
                "step": step, "return": episode_reward.tolist(),
                "mode": info.get("mode_name", "?"), "length": episode_steps,
            })
            obs, info = env.reset()
            episode_reward = np.zeros(reward_dim)
            episode_steps = 0
            last_mode = info["mode_idx"]
        else:
            obs = next_obs

        # Logging
        if step % log_freq == 0:
            elapsed = time.time() - start_time
            sps = step / max(elapsed, 1)
            metrics_str = ""
            if bamor is not None:
                m = bamor.get_metrics(w_np)
                log_data["bamor_metrics"].append({
                    "step": step,
                    **{k: v.tolist() if hasattr(v, 'tolist') else float(v) for k, v in m.items()}
                })
                metrics_str = f" Δl/Δg={m.get('delta_local',0):.2f}/{m.get('delta_global',0):.2f}"
            recent = log_data["episode_rewards"][-5:] if log_data["episode_rewards"] else []
            if recent:
                avg_r = np.mean([e["return"] for e in recent], axis=0)
                print(f"  [{method}] step={step:>7d} ep={episode_count:>4d} "
                      f"avg_r={avg_r.round(1)} mode_sw={mode_switch_count} "
                      f"sps={sps:.0f}{metrics_str}")

        # Evaluation
        if step % eval_freq == 0:
            pareto = evaluate_pareto(eval_env, trainer, num_weights=11,
                                     num_episodes=3, device=dev)
            log_data["pareto_evals"].append({"step": step, "front": pareto})
            returns = np.array([p["return"] for p in pareto])
            print(f"  [{method}] EVAL step={step}: Pareto range "
                  f"obj0=[{returns[:,0].min():.1f}, {returns[:,0].max():.1f}] "
                  f"obj1=[{returns[:,1].min():.1f}, {returns[:,1].max():.1f}]")

    # --- Final HV eval ----------------------------------------------------
    # Overall HV: with mode switches active.
    # Per-mode HV: lock env to each mode, eval Pareto, compute HV.
    # Reference point must be dominated by ALL Pareto points (worse on every obj).
    # For mo-halfcheetah-v5: obj0 = forward_vel (typically 0–1500),
    # obj1 = -control_cost (typically -3000 to -50).  Pick conservative floor.
    GLOBAL_REF = np.array([-50.0, -3000.0])
    final_overall = evaluate_pareto(eval_env, trainer, num_weights=21,
                                    num_episodes=3, device=dev)
    log_data["pareto_overall"] = final_overall
    log_data["hv_overall"] = compute_hv_2d(
        np.array([p["return"] for p in final_overall]), GLOBAL_REF)

    # Per-mode evaluation uses CONE-RESTRICTED weights aligned with each
    # mode's substrate-constrained anchor.  This is the in-scope evaluation
    # — preferences outside the cone are physically unreachable under that
    # mode's reward caps, so measuring HV on them would be meaningless.
    log_data["pareto_per_mode"] = []
    log_data["hv_per_mode"] = []
    log_data["per_mode_cone"] = []
    for m in range(env.num_modes):
        mc = env.mode_configs[m]
        anchor = (mc.true_anchor if mc.true_anchor is not None
                  else np.full(reward_dim, 1.0 / reward_dim))
        radius = mc.cone_radius
        # Build a cone-restricted weight grid.  For 2-D reward, the simplex
        # is parameterised by w[0]; clip to [anchor[0]±radius] ∩ [0.01, 0.99]
        # and sweep linearly.
        if reward_dim == 2:
            w_lo = max(0.01, float(anchor[0]) - radius)
            w_hi = min(0.99, float(anchor[0]) + radius)
            wgrid = np.linspace(w_lo, w_hi, 11, dtype=np.float32)
            cone_weights = np.stack([wgrid, 1.0 - wgrid], axis=-1)
        else:
            # Higher dim: sample Dirichlet around anchor with concentration
            # commensurate with the cone radius (smaller radius → tighter).
            kappa = 1.0 / max(radius, 0.01)
            cone_weights = np.random.dirichlet(kappa * np.maximum(anchor, 1e-3),
                                               size=11).astype(np.float32)

        eval_env._lock_mode = True
        eval_env.current_mode_idx = m
        eval_env._apply_mode(m)
        pm = evaluate_pareto_with_weights(eval_env, trainer, cone_weights,
                                           num_episodes=3, device=dev)
        eval_env._lock_mode = False
        hv_m = compute_hv_2d(np.array([p["return"] for p in pm]), GLOBAL_REF)
        log_data["pareto_per_mode"].append(pm)
        log_data["hv_per_mode"].append(hv_m)
        log_data["per_mode_cone"].append({
            "name": mc.name,
            "anchor": anchor.tolist(),
            "radius": radius,
            "weights": cone_weights.tolist(),
        })
        print(f"  [{method}] Mode {m} ({mc.name}): cone HV={hv_m:.1f}")
    print(f"  [{method}] HV overall (full simplex) = {log_data['hv_overall']:.1f}")

    # --- Save ---
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, f"{method}_seed{seed}.json")
    with open(result_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    print(f"[{method}] Saved to {result_path}")

    # Save model
    model_path = os.path.join(save_dir, f"{method}_seed{seed}_model.pt")
    torch.save({
        "policy": trainer.policy.state_dict(),
        "q1": trainer.q1.state_dict(),
        "q2": trainer.q2.state_dict(),
    }, model_path)

    env.close()
    eval_env.close()
    return log_data


# ================================================================
# CAPQL baseline (uses morl-baselines directly)
# ================================================================

def train_capql(env_name, total_steps, switch_interval, seed, save_dir,
                eval_freq=20_000, device="auto"):
    """Train CAPQL from morl-baselines on piecewise-stationary env."""
    try:
        from morl_baselines.multi_policy.capql.capql import CAPQL
    except ImportError:
        print("CAPQL not available in morl-baselines. Skipping.")
        return None

    env = make_piecewise_mo_env(env_name, switch_interval=switch_interval, seed=seed)
    eval_env = make_piecewise_mo_env(env_name, switch_interval=switch_interval, seed=seed + 1000)

    ref_point = np.zeros(env.reward_dim)  # conservative reference point

    agent = CAPQL(
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_size=500_000,
        net_arch=[256, 256],
        batch_size=256,
        alpha=0.2,
        learning_starts=5000,
        seed=seed,
        device=device,
        log=False,  # disable wandb
    )

    print(f"[capql] Training on {env_name} for {total_steps} steps...")
    agent.train(
        total_timesteps=total_steps,
        eval_env=eval_env,
        ref_point=ref_point,
        eval_freq=eval_freq,
        num_eval_weights_for_front=50,
        num_eval_episodes_for_front=3,
    )

    # Save CAPQL results
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, f"capql_seed{seed}.json")
    # CAPQL logs via wandb; save basic info
    with open(result_path, "w") as f:
        json.dump({
            "method": "capql", "env": env_name, "seed": seed,
            "total_steps": total_steps,
            "ground_truth": env.get_ground_truth(),
        }, f, indent=2, default=str)

    env.close()
    eval_env.close()
    print(f"[capql] Done. Saved to {result_path}")


# ================================================================
# Entry point
# ================================================================

METHODS = ["bamor", "bamor_nc", "bamor_ns", "uniform", "fixed", "capql"]

def main():
    parser = argparse.ArgumentParser(description="BAMOR MuJoCo Experiments")
    parser.add_argument("--env", type=str, default="mo-halfcheetah-v5",
                        help="mo-gymnasium env ID")
    parser.add_argument("--method", type=str, default="all",
                        choices=METHODS + ["all"],
                        help="Method to train (or 'all')")
    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--switch_interval", type=int, default=100_000,
                        help="Steps between mode switches")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=1,
                        help="Number of random seeds to run")
    parser.add_argument("--save_dir", type=str, default="results/bamor_mujoco")
    parser.add_argument("--eval_freq", type=int, default=20_000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--anchor_eta", type=float, default=0.01)
    args = parser.parse_args()

    methods = METHODS if args.method == "all" else [args.method]
    save_dir = os.path.join(args.save_dir, args.env.replace("-", "_"))

    for seed in range(args.seed, args.seed + args.num_seeds):
        for method in methods:
            print(f"\n{'='*60}")
            print(f"  Method: {method}  Env: {args.env}  Seed: {seed}")
            print(f"{'='*60}")

            if method == "capql":
                train_capql(
                    args.env, args.total_steps, args.switch_interval,
                    seed, save_dir, args.eval_freq, args.device,
                )
            else:
                train_method(
                    env_name=args.env,
                    method=method,
                    total_steps=args.total_steps,
                    switch_interval=args.switch_interval,
                    seed=seed,
                    save_dir=save_dir,
                    eval_freq=args.eval_freq,
                    hidden=args.hidden,
                    anchor_eta=args.anchor_eta,
                    device=args.device,
                )

    print(f"\n✅ All experiments complete. Results in {save_dir}/")


if __name__ == "__main__":
    main()
