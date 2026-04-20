"""
capql_synthetic.py — Run CAPQL (morl-baselines) on the synthetic MO environment

Adapts the SyntheticMOEnv to the mo-gymnasium API required by CAPQL,
then trains CAPQL as a baseline for comparison with BAMOR.

This provides the strongest baseline: CAPQL is the ICLR 2023 SOTA for
continuous-action MORL. Comparing against it directly demonstrates
BAMOR's advantage in non-stationary multi-mode settings.

Usage:
  python capql_synthetic.py --total_timesteps 50000 --save_dir synthetic_results

Requirements:
  pip install mo-gymnasium  (for CAPQL's internal dependencies)
"""

import os
import json
import time
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from synthetic_mo_env import SyntheticMOEnv


class MOGymSyntheticWrapper(gym.Env):
    """Wraps SyntheticMOEnv to satisfy the mo-gymnasium API for CAPQL.

    CAPQL requires:
      - env.unwrapped.reward_space: Box with shape (reward_dim,)
      - env.unwrapped.reward_dim: int
      - step() returns vector reward as np.ndarray
      - gymnasium 5-tuple: (obs, reward, terminated, truncated, info)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, switch_interval=200, max_steps=600, seed=None, **kwargs):
        super().__init__()
        self._env = SyntheticMOEnv(
            switch_interval=switch_interval,
            max_steps=max_steps,
            seed=seed,
            **kwargs,
        )

        # Spaces from inner env
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        # mo-gymnasium required attributes
        self.reward_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._env.reward_dim,),
            dtype=np.float64,
        )
        self.reward_dim = self._env.reward_dim

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        # Ensure reward is np.ndarray
        reward = np.asarray(reward, dtype=np.float64)
        return obs, reward, terminated, truncated, info

    def get_ground_truth(self):
        return self._env.get_ground_truth()


def run_capql(args):
    """Train CAPQL on the synthetic environment."""
    try:
        from morl_baselines.multi_policy.capql.capql import CAPQL
    except ImportError:
        print("ERROR: morl-baselines CAPQL not found.")
        print("Make sure morl-baselines is installed: pip install morl-baselines")
        print("Or add to PYTHONPATH: export PYTHONPATH=/path/to/morl-baselines:$PYTHONPATH")
        return None

    print("="*60)
    print("Training CAPQL (ICLR 2023 baseline)")
    print("="*60)

    env = MOGymSyntheticWrapper(
        switch_interval=args.switch_interval,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    eval_env = MOGymSyntheticWrapper(
        switch_interval=args.switch_interval,
        max_steps=args.max_steps,
        seed=args.seed + 10000,
    )

    start_time = time.time()

    agent = CAPQL(
        env,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        net_arch=[args.hidden, args.hidden],
        log=False,  # disable wandb
        seed=args.seed,
    )

    # Train
    agent.train(
        total_timesteps=args.total_timesteps,
        eval_env=eval_env,
        ref_point=np.array([-3000.0, -3000.0]),  # reference point for hypervolume
        num_eval_weights_for_front=21,
        num_eval_episodes_for_front=3,
        eval_freq=args.eval_freq,
    )

    wall_time = time.time() - start_time

    # Evaluate Pareto front
    print("Evaluating CAPQL Pareto front...")
    pareto_evals = []
    for w0 in np.linspace(0.0, 1.0, 21):
        w_eval = np.array([w0, 1 - w0], dtype=np.float32)
        eval_rewards = []
        for eval_ep in range(3):
            obs, _ = eval_env.reset(seed=10000 + eval_ep)
            ep_r = np.zeros(2)
            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.eval(obs, w_eval)
                obs, r, terminated, truncated, info = eval_env.step(action)
                ep_r += r
            eval_rewards.append(ep_r.tolist())
        mean_r = np.mean(eval_rewards, axis=0).tolist()
        pareto_evals.append({
            "weight": [float(w0), float(1 - w0)],
            "mean_reward": mean_r,
            "mode": info.get("mode", "unknown"),
        })

    results = {
        "method": "capql",
        "episode_rewards": [],  # CAPQL doesn't easily expose per-episode rewards
        "delta_globals": [],
        "delta_locals": [],
        "improvements": [],
        "cone_radii": [],
        "anchor_errors": [],
        "mode_switches": [],
        "pareto_front": pareto_evals,
        "wall_time": wall_time,
        "ground_truth": env.get_ground_truth(),
        "bamor_summary": "",
    }

    print(f"CAPQL training done in {wall_time:.1f}s")
    print("Pareto front sample:")
    for p in pareto_evals[::5]:
        print(f"  w={p['weight']}, reward={p['mean_reward']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run CAPQL on synthetic env")
    parser.add_argument('--total_timesteps', type=int, default=50000)
    parser.add_argument('--max_steps', type=int, default=600)
    parser.add_argument('--switch_interval', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_freq', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='synthetic_results')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    results = run_capql(args)
    if results is None:
        return

    # Save
    output_path = os.path.join(args.save_dir, 'capql.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")

    # Also merge into all_results.json if it exists
    combined_path = os.path.join(args.save_dir, 'all_results.json')
    if os.path.exists(combined_path):
        with open(combined_path) as f:
            all_results = json.load(f)
        all_results['capql'] = results
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Merged into {combined_path}")


if __name__ == "__main__":
    main()
