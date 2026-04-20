"""
train_synthetic.py — SAC training on SyntheticMOEnv with BAMOR vs baselines

This is the PRIMARY experiment script for the paper. It trains weight-conditioned
SAC on the synthetic MO environment with controllable mode switches, allowing
direct comparison between BAMOR and baselines.

Output data (saved to --save_dir):
  - Training curves per objective
  - Pareto front evaluation scatter (per-mode colored = the "killer plot")
  - Δ_global vs Δ_local time series
  - Anchor estimation accuracy vs ground truth
  - Mode switch detection timeline

Usage:
  # Full BAMOR
  python train_synthetic.py --method bamor --max_episodes 200

  # Baseline: uniform sampling
  python train_synthetic.py --method uniform --max_episodes 200

  # Ablation: BAMOR without Target Cone
  python train_synthetic.py --method bamor_no_cone --max_episodes 200

  # Run all methods
  python train_synthetic.py --method all --max_episodes 200
"""

import os
import json
import time
import argparse
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
from typing import Tuple, Optional

# Force unbuffered print for conda/subprocess compatibility
print = functools.partial(print, flush=True)

from synthetic_mo_env import SyntheticMOEnv
from bamor_core import BAMORPipeline


# ════════════════════════════════════════════════════════════════
# Networks: lightweight MO-SAC for 2D state, 1D action
# ════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SoftQNet(nn.Module):
    """Weight-conditioned Q-network: Q(s, a, w) → ℝ^reward_dim"""
    def __init__(self, state_dim=2, action_dim=1, weight_dim=2, hidden=128, reward_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + weight_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, reward_dim),
        )

    def forward(self, state, action, weight):
        x = torch.cat([state, action, weight], dim=-1)
        return self.net(x)


class PolicyNet(nn.Module):
    """Weight-conditioned Gaussian policy: π(a|s,w)"""
    def __init__(self, state_dim=2, action_dim=1, weight_dim=2, hidden=128,
                 action_range=1.0, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + weight_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)
        self.action_range = action_range
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state, weight):
        x = self.net(torch.cat([state, weight], dim=-1))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state, weight):
        mean, log_std = self.forward(state, weight)
        std = log_std.exp()
        z = torch.randn_like(mean)
        action_raw = mean + std * z
        action = torch.tanh(action_raw) * self.action_range
        log_prob = Normal(mean, std).log_prob(action_raw) - \
                   torch.log(1 - action.pow(2) / (self.action_range ** 2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def get_action(self, state, weight, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        weight = torch.FloatTensor(weight).unsqueeze(0).to(device)
        mean, log_std = self.forward(state, weight)
        if deterministic:
            action = torch.tanh(mean) * self.action_range
        else:
            action, _ = self.sample(state, weight)
        return action.detach().cpu().numpy()[0]


# ════════════════════════════════════════════════════════════════
# Replay Buffer
# ════════════════════════════════════════════════════════════════

class SimpleReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, weight, reward, next_state, done):
        self.buffer.append((state, action, weight, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, weights, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(weights),
                np.array(rewards), np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ════════════════════════════════════════════════════════════════
# Trainer
# ════════════════════════════════════════════════════════════════

class SyntheticMOSACTrainer:
    """Lightweight MO-SAC trainer for the synthetic env."""

    def __init__(self, state_dim=2, action_dim=1, reward_dim=2, hidden=128,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 buffer_capacity=100000, batch_size=256):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.reward_dim = reward_dim

        self.q1 = SoftQNet(state_dim, action_dim, reward_dim, hidden, reward_dim).to(device)
        self.q2 = SoftQNet(state_dim, action_dim, reward_dim, hidden, reward_dim).to(device)
        self.q1_target = SoftQNet(state_dim, action_dim, reward_dim, hidden, reward_dim).to(device)
        self.q2_target = SoftQNet(state_dim, action_dim, reward_dim, hidden, reward_dim).to(device)
        self.policy = PolicyNet(state_dim, action_dim, reward_dim, hidden).to(device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.q1_opt = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=lr)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=lr)

        self.buffer = SimpleReplayBuffer(buffer_capacity)

    def update(self, w_anchor=None, anchor_eta=0.0):
        if len(self.buffer) < self.batch_size:
            return None, None

        s, a, w, r, ns, d = self.buffer.sample(self.batch_size)
        s = torch.FloatTensor(s).to(device)
        a = torch.FloatTensor(a).to(device)
        w = torch.FloatTensor(w).to(device)
        r = torch.FloatTensor(r).to(device)
        ns = torch.FloatTensor(ns).to(device)
        d = torch.FloatTensor(d).unsqueeze(1).to(device)

        # Target Q
        with torch.no_grad():
            na, next_log_prob = self.policy.sample(ns, w)
            tq1 = self.q1_target(ns, na, w)
            tq2 = self.q2_target(ns, na, w)
            tq = torch.min(tq1, tq2) - self.alpha * next_log_prob
            target = r + (1 - d) * self.gamma * tq

        # Q losses
        q1_pred = self.q1(s, a, w)
        q2_pred = self.q2(s, a, w)
        q1_loss = F.mse_loss(q1_pred, target)
        q2_loss = F.mse_loss(q2_pred, target)

        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        # Policy loss
        new_a, log_prob = self.policy.sample(s, w)
        q_new = torch.min(self.q1(s, new_a, w), self.q2(s, new_a, w))
        # Scalarize: ⟨w, Q⟩
        scalar_q = (q_new * w).sum(dim=-1, keepdim=True)
        policy_loss = (self.alpha * log_prob - scalar_q).mean()

        # Anchor regularization
        if w_anchor is not None and anchor_eta > 0:
            anchor_reg = BAMORPipeline.anchor_regularization_loss(w, w_anchor, eta=anchor_eta)
            policy_loss = policy_loss + anchor_reg

        self.policy_opt.zero_grad(); policy_loss.backward(); self.policy_opt.step()

        # Soft update targets
        for tp, p in zip(self.q1_target.parameters(), self.q1.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for tp, p in zip(self.q2_target.parameters(), self.q2.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        # Return target Q mean per objective for drift tracking
        target_q_vec = target.detach().mean(dim=0).cpu().numpy()
        return scalar_q.mean().item(), target_q_vec


# ════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════

def train_method(method: str, args):
    """Train one method variant and return results dict."""
    print(f"\n{'='*60}")
    print(f"Training: {method}")
    print(f"{'='*60}")

    env = SyntheticMOEnv(
        switch_interval=args.switch_interval,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    gt = env.get_ground_truth()

    trainer = SyntheticMOSACTrainer(
        state_dim=2, action_dim=1, reward_dim=2,
        hidden=args.hidden, lr=args.lr, gamma=args.gamma,
        batch_size=args.batch_size,
    )

    # BAMOR pipeline
    bamor = None
    anchor_eta = 0.0
    if method in ("bamor", "bamor_no_cone", "bamor_no_surprise"):
        bamor = BAMORPipeline(
            num_modes=args.bamor_modes, reward_dim=2, gamma=args.gamma,
            surprise_threshold=args.bamor_threshold,
            concentration=args.bamor_concentration,
            disable_surprise=(method == "bamor_no_surprise"),
            disable_cone=(method == "bamor_no_cone"),
        )
        anchor_eta = args.bamor_eta

    # Logging
    episode_rewards = []
    episode_delta_globals = []
    episode_delta_locals = []
    episode_improvements = []
    episode_cone_radii = []
    episode_anchor_errors = []
    episode_mode_switches = []
    pareto_evals = []

    start_time = time.time()

    for ep in range(args.max_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ep_reward = np.zeros(2, dtype=np.float64)

        for step in range(args.max_steps):
            # Sample weight
            if bamor is not None:
                w, w_anchor = bamor.sample_weight()
            elif method == "biased":
                w = np.random.dirichlet([3.0, 1.0]).astype(np.float32)
                w_anchor = None
            elif method == "fixed":
                w = np.array([0.5, 0.5], dtype=np.float32)
                w_anchor = None
            else:  # uniform
                w = np.random.dirichlet([1.0, 1.0]).astype(np.float32)
                w_anchor = None

            # Act
            action = trainer.policy.get_action(obs, w, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Store
            trainer.buffer.push(obs, action, w, reward, next_obs,
                                float(terminated or truncated))

            # BAMOR: observe reward
            if bamor is not None:
                bamor.observe_reward(reward)

            ep_reward += reward
            obs = next_obs

            # Train
            if len(trainer.buffer) >= args.batch_size:
                w_anchor_tensor = None
                if bamor is not None and w_anchor is not None:
                    w_anchor_tensor = torch.FloatTensor(
                        np.tile(w_anchor, (args.batch_size, 1))
                    ).to(device)
                scalar_q, target_q_vec = trainer.update(
                    w_anchor=w_anchor_tensor, anchor_eta=anchor_eta)
                if bamor is not None and target_q_vec is not None:
                    bamor.record_target_q(target_q_vec)

            if terminated or truncated:
                break

        episode_rewards.append(ep_reward.tolist())

        # BAMOR metrics
        if bamor is not None:
            w_cur, _ = bamor.sample_weight()
            metrics = bamor.get_metrics(w_cur)
            episode_delta_globals.append(metrics['delta_global'])
            episode_delta_locals.append(metrics['delta_local'])
            episode_improvements.append(metrics['improvement'])
            radius = bamor.cone_estimator.belief_weighted_radius(bamor.belief_tracker.belief)
            episode_cone_radii.append(radius)
            episode_mode_switches.append(bamor.belief_tracker.mode_switch_count)

            # Anchor accuracy vs ground truth
            errors = []
            for h_idx, mc in enumerate(gt['mode_configs']):
                if h_idx < bamor.num_modes:
                    true_a = np.array(mc['true_anchor'])
                    est_a = bamor.cone_estimator.constraint.w_anchor[h_idx]
                    errors.append(float(np.linalg.norm(true_a - est_a)))
            episode_anchor_errors.append(np.mean(errors) if errors else 0.0)
        else:
            episode_delta_globals.append(0.0)
            episode_delta_locals.append(0.0)
            episode_improvements.append(0.0)
            episode_cone_radii.append(0.0)
            episode_anchor_errors.append(0.0)
            episode_mode_switches.append(0)

        if (ep + 1) % max(1, args.max_episodes // 10) == 0 or ep == 0:
            r = np.array(episode_rewards[max(0,len(episode_rewards)-20):])
            print(f"  Ep {ep+1:4d}: reward=[{r[:,0].mean():.1f}, {r[:,1].mean():.1f}], "
                  f"Δ_g={episode_delta_globals[-1]:.4f}, Δ_l={episode_delta_locals[-1]:.4f}")

    # Pareto front evaluation
    print(f"  Evaluating Pareto front...")
    for w0 in np.linspace(0.0, 1.0, 11):
        w_eval = np.array([w0, 1 - w0], dtype=np.float32)
        eval_rewards = []
        for eval_ep in range(1):
            obs, _ = env.reset(seed=10000 + eval_ep)
            ep_r = np.zeros(2)
            for _ in range(args.max_steps):
                action = trainer.policy.get_action(obs, w_eval, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                ep_r += r
                if term or trunc:
                    break
            eval_rewards.append(ep_r.tolist())
        mean_r = np.mean(eval_rewards, axis=0).tolist()
        pareto_evals.append({
            "weight": [float(w0), float(1 - w0)],
            "mean_reward": mean_r,
            "mode": info.get("mode", "unknown"),
        })

    wall_time = time.time() - start_time
    print(f"  Done in {wall_time:.1f}s")

    return {
        "method": method,
        "episode_rewards": episode_rewards,
        "delta_globals": episode_delta_globals,
        "delta_locals": episode_delta_locals,
        "improvements": episode_improvements,
        "cone_radii": episode_cone_radii,
        "anchor_errors": episode_anchor_errors,
        "mode_switches": episode_mode_switches,
        "pareto_front": pareto_evals,
        "wall_time": wall_time,
        "ground_truth": gt,
        "bamor_summary": bamor.summary() if bamor else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Train MO-SAC on synthetic env")
    parser.add_argument('--method', type=str, default='all',
                        choices=['bamor', 'bamor_no_cone', 'bamor_no_surprise',
                                 'uniform', 'biased', 'fixed', 'all'])
    parser.add_argument('--max_episodes', type=int, default=200)
    parser.add_argument('--max_steps', type=int, default=600)
    parser.add_argument('--switch_interval', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='synthetic_results')
    # BAMOR hyperparameters
    parser.add_argument('--bamor_modes', type=int, default=3)
    parser.add_argument('--bamor_threshold', type=float, default=8.0)
    parser.add_argument('--bamor_concentration', type=float, default=10.0)
    parser.add_argument('--bamor_eta', type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.method == 'all':
        methods = ['bamor', 'bamor_no_cone', 'bamor_no_surprise', 'uniform', 'biased', 'fixed']
    else:
        methods = [args.method]

    all_results = {}
    for method in methods:
        results = train_method(method, args)
        all_results[method] = results

        # Save individual result
        with open(os.path.join(args.save_dir, f'{method}.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Save combined results
    with open(os.path.join(args.save_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print comparison
    print("\n" + "="*80)
    print("TRAINING RESULTS COMPARISON")
    print("="*80)
    print(f"{'Method':<22} {'R[0] last50':>12} {'R[1] last50':>12} "
          f"{'Δ_g':>8} {'Δ_l':>8} {'AncErr':>8} {'Time':>8}")
    print("-"*80)
    for method, res in all_results.items():
        r = np.array(res['episode_rewards'][-50:])
        dg = np.mean(res['delta_globals'][-50:])
        dl = np.mean(res['delta_locals'][-50:])
        ae = np.mean(res['anchor_errors'][-50:])
        print(f"{method:<22} {r[:,0].mean():>12.1f} {r[:,1].mean():>12.1f} "
              f"{dg:>8.4f} {dl:>8.4f} {ae:>8.4f} {res['wall_time']:>7.1f}s")

    print(f"\nResults saved to {args.save_dir}/")
    print(f"Run: python plot_results.py --input_dir {args.save_dir}")


if __name__ == "__main__":
    main()
