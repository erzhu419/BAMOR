"""
Synthetic Multi-Objective Environment with Controllable Mode Switching

Purpose: Validate BAMOR's theoretical claims (Lean Parts XIII-XV) in a
controlled setting where:
  - Mode switch timing is known → can verify surprise detection accuracy
  - Per-mode substrate constraints are known → can verify cone estimation
  - Δ_global and Δ_local are analytically computable → can verify bounds

Design:
  A 1D continuous navigation task with 2 objectives:
    Objective 0: "Safety"  — penalty for being far from origin
    Objective 1: "Progress" — reward for moving to the right

  The environment has K discrete modes that switch at fixed intervals.
  Each mode changes the dynamics (speed multiplier) and reward landscape:
    - Mode "safe":   high safety sensitivity, low progress potential
    - Mode "fast":   low safety sensitivity, high progress potential
    - Mode "mixed":  both objectives moderately responsive

  The mode determines which objectives the agent can actually influence
  (the "substrate constraint"). This directly creates the Target Cone effect.

Gymnasium-compatible: inherits from gymnasium.Env for standardization.

Lean correspondence:
    - K modes → H in ModeConeConstraint
    - Per-mode reward scaling → w_anchor(h) that the estimator should discover
    - Known switch schedule → ground truth for surprise detection evaluation
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class ModeConfig:
    """Configuration for a single environment mode.

    Lean correspondence: one element of H in ModeConeConstraint.

    Each mode can define EITHER:
    - reward_scale (legacy): multiplies a shared base reward function
    - reward_fns (new): per-objective callables f(position, action) → float
      This enables different modes to have structurally different rewards,
      creating non-overlapping Pareto sub-manifolds.
    """
    name: str
    # Reward scaling per objective — determines substrate constraint
    # High scale = objective is responsive = high anchor weight
    # Near-zero scale = objective is collapsed = near-zero anchor weight
    reward_scale: np.ndarray    # shape (num_objectives,)
    # Dynamics modification
    speed_multiplier: float     # how fast the agent can move
    noise_std: float            # transition noise
    # Ground truth anchor (for evaluation only — algorithm must discover this)
    true_anchor: np.ndarray     # shape (num_objectives,) — on simplex
    # Per-objective reward functions: list of callables (pos, action) → float
    # If provided, these REPLACE the base reward * reward_scale mechanism
    reward_fns: list = None


class SyntheticMOEnv(gym.Env):
    """Synthetic MO environment with controllable mode switches.

    State: position ∈ [-10, 10]
    Action: velocity ∈ [-1, 1]
    Reward: 2D vector [safety_reward, progress_reward]

    Modes switch every `switch_interval` steps.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        modes: Optional[List[ModeConfig]] = None,
        switch_interval: int = 200,
        max_steps: int = 1000,
        reward_dim: int = 2,
        seed: Optional[int] = None,
    ):
        """
        Args:
            modes:           list of ModeConfig (default: 3 standard modes)
            switch_interval: steps between mode switches
            max_steps:       episode length
            reward_dim:      number of objectives
            seed:            random seed
        """
        super().__init__()

        self.reward_dim = reward_dim
        self.switch_interval = switch_interval
        self.max_steps = max_steps

        # Default modes that create clear substrate constraints
        if modes is None:
            self.modes = [
                ModeConfig(
                    name="safe_zone",
                    reward_scale=np.array([3.0, 0.1]),   # safety responsive, progress collapsed
                    speed_multiplier=0.3,
                    noise_std=0.05,
                    true_anchor=np.array([0.97, 0.03]),
                ),
                ModeConfig(
                    name="fast_lane",
                    reward_scale=np.array([0.1, 3.0]),   # safety collapsed, progress responsive
                    speed_multiplier=2.0,
                    noise_std=0.1,
                    true_anchor=np.array([0.03, 0.97]),
                ),
                ModeConfig(
                    name="balanced",
                    reward_scale=np.array([1.0, 1.0]),   # both responsive
                    speed_multiplier=1.0,
                    noise_std=0.08,
                    true_anchor=np.array([0.50, 0.50]),
                ),
            ]
        else:
            self.modes = modes

        self.num_modes = len(self.modes)

        # Spaces
        # State: [position, current_speed, mode_indicator (not visible to agent)]
        # We give the agent [position, speed] — mode must be inferred from rewards
        self.observation_space = spaces.Box(
            low=np.array([-10.0, -3.0], dtype=np.float32),
            high=np.array([10.0, 3.0], dtype=np.float32),
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )

        # Internal state
        self.position = 0.0
        self.velocity = 0.0
        self.step_count = 0          # within-episode counter (reset on reset()) — for truncation
        self.global_step = 0         # across-episode counter (never reset) — for mode switching
        self.current_mode_idx = 0
        self.mode_switch_log: deque = deque(maxlen=1000)  # (global_step, mode_idx); capped

        self._np_random = np.random.RandomState(seed)

    @property
    def current_mode(self) -> ModeConfig:
        return self.modes[self.current_mode_idx]

    def _get_obs(self) -> np.ndarray:
        return np.array([self.position, self.velocity], dtype=np.float32)

    def _compute_reward(self, action: float) -> np.ndarray:
        """Compute multi-objective reward vector.

        Two modes of operation:

        1. If mode has reward_fns: each objective uses its own function
           reward[i] = mode.reward_fns[i](position, action)
           This creates structurally different Pareto fronts per mode
           (non-overlapping sub-manifolds).

        2. Otherwise (legacy): shared base reward × mode.reward_scale
           Objective 0 (Safety):   -|position| × scale[0]
           Objective 1 (Progress): action[0]   × scale[1]
        """
        mode = self.current_mode
        pos = self.position
        act = action[0] if hasattr(action, '__len__') else action

        if mode.reward_fns is not None:
            # NEW: per-mode reward functions → non-overlapping sub-manifolds
            rewards = np.array([fn(pos, act) for fn in mode.reward_fns], dtype=np.float64)
            # Light noise proportional to reward magnitude for cone estimation
            noise = self._np_random.normal(0, 0.1, size=self.reward_dim)
            rewards += noise
        else:
            # LEGACY: shared base reward × scale
            raw_rewards = np.array([
                -abs(pos),    # safety: penalize distance from origin
                act,          # progress: reward positive velocity
            ], dtype=np.float64)
            rewards = raw_rewards * mode.reward_scale
            noise = self._np_random.normal(0, 1, size=self.reward_dim) * mode.reward_scale * 0.3
            rewards += noise

        return rewards

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random = np.random.RandomState(seed)

        self.position = self._np_random.uniform(-1, 1)
        self.velocity = 0.0
        self.step_count = 0           # within-episode counter resets (for truncation)
        # global_step and current_mode_idx are NOT reset — mode switches are
        # a property of the global environment evolution, not individual episodes.
        # This ensures BAMOR's belief tracker actually sees non-stationary switches.
        if not getattr(self, '_lock_mode', False):
            # Recompute mode from persistent global counter
            self.current_mode_idx = (self.global_step // self.switch_interval) % self.num_modes

        return self._get_obs(), {"mode": self.current_mode.name, "mode_idx": self.current_mode_idx}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        mode = self.current_mode

        # Dynamics: position update with mode-dependent speed
        self.velocity = action[0] * mode.speed_multiplier
        noise = self._np_random.normal(0, mode.noise_std)
        self.position += self.velocity + noise
        self.position = np.clip(self.position, -10.0, 10.0)

        # Reward
        reward = self._compute_reward(action)

        # Step counters: both advance each step
        self.step_count += 1    # within-episode (for truncation)
        self.global_step += 1   # global (for mode switching)

        # Mode switching (deterministic schedule on global_step), skipped if mode is locked
        old_mode = self.current_mode_idx
        if getattr(self, '_lock_mode', False):
            new_mode = self.current_mode_idx
        else:
            new_mode = (self.global_step // self.switch_interval) % self.num_modes
        if new_mode != self.current_mode_idx:
            self.current_mode_idx = new_mode
            self.mode_switch_log.append((self.global_step, new_mode))

        # Termination
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "mode": self.current_mode.name,
            "mode_idx": self.current_mode_idx,
            "mode_switched": new_mode != old_mode,
            "true_anchor": self.current_mode.true_anchor,
            "reward_scale": self.current_mode.reward_scale,
            "step": self.step_count,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def get_ground_truth(self) -> Dict:
        """Return ground truth for evaluation against BAMOR estimates.

        This is the key advantage of the synthetic env — we KNOW the true
        substrate constraints and can verify the algorithm discovers them.
        """
        return {
            "num_modes": self.num_modes,
            "switch_interval": self.switch_interval,
            "mode_configs": [
                {
                    "name": m.name,
                    "true_anchor": m.true_anchor.tolist(),
                    "reward_scale": m.reward_scale.tolist(),
                    "speed_multiplier": m.speed_multiplier,
                }
                for m in self.modes
            ],
            "switch_log": self.mode_switch_log,
        }


class SyntheticMOEnvStochastic(SyntheticMOEnv):
    """Variant with stochastic (random) mode switches.

    Instead of a fixed schedule, modes switch via a Markov chain with
    controllable transition probabilities. This is more realistic but
    harder to detect — tests BAMOR's surprise detection robustness.
    """

    def __init__(
        self,
        modes: Optional[List[ModeConfig]] = None,
        min_dwell: int = 50,
        switch_prob: float = 0.02,
        **kwargs,
    ):
        """
        Args:
            min_dwell:   minimum steps before a mode switch can occur
            switch_prob: probability of switching at each step (after min_dwell)
        """
        # Remove switch_interval from kwargs since we override it
        kwargs.pop('switch_interval', None)
        super().__init__(modes=modes, switch_interval=999999, **kwargs)
        self.min_dwell = min_dwell
        self.switch_prob = switch_prob
        self.steps_in_mode = 0

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = super().step(action)

        self.steps_in_mode += 1

        # Stochastic switching
        if self.steps_in_mode >= self.min_dwell:
            if self._np_random.random() < self.switch_prob:
                old_mode = self.current_mode_idx
                # Switch to a different mode (uniform over others)
                candidates = [i for i in range(self.num_modes) if i != old_mode]
                self.current_mode_idx = self._np_random.choice(candidates)
                self.mode_switch_log.append((self.global_step, self.current_mode_idx))
                self.steps_in_mode = 0
                info["mode_switched"] = True
                info["mode"] = self.current_mode.name
                info["mode_idx"] = self.current_mode_idx
                info["true_anchor"] = self.current_mode.true_anchor
                info["reward_scale"] = self.current_mode.reward_scale

        return obs, reward, terminated, truncated, info


# ════════════════════════════════════════════════════════════════
# Convenience: run a quick diagnostic to verify env behavior
# ════════════════════════════════════════════════════════════════

def run_diagnostic(env_class=SyntheticMOEnv, num_steps=600, seed=42):
    """Run the environment and print mode/reward statistics."""
    env = env_class(switch_interval=200, max_steps=num_steps, seed=seed)
    obs, info = env.reset()

    mode_rewards = {m.name: [] for m in env.modes}
    mode_history = []

    for t in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        mode_rewards[info["mode"]].append(reward)
        mode_history.append(info["mode_idx"])

        if terminated or truncated:
            break

    print("=== Synthetic MO Environment Diagnostic ===")
    print(f"Total steps: {t+1}")
    print(f"Mode switches: {env.mode_switch_log}")
    print()

    for name, rewards in mode_rewards.items():
        if len(rewards) == 0:
            continue
        r = np.array(rewards)
        print(f"Mode '{name}' ({len(rewards)} steps):")
        for obj in range(env.reward_dim):
            print(f"  Obj {obj}: mean={r[:, obj].mean():.3f}, "
                  f"std={r[:, obj].std():.3f}, "
                  f"CV={r[:, obj].std() / (abs(r[:, obj].mean()) + 1e-8):.3f}")

        # Verify: per-mode CV should match the expected anchor weights
        stds = np.array([r[:, i].std() for i in range(env.reward_dim)])
        means = np.abs(np.array([r[:, i].mean() for i in range(env.reward_dim)])) + 1e-8
        cv = stds / means
        estimated_anchor = cv / cv.sum()
        mode_cfg = [m for m in env.modes if m.name == name][0]
        print(f"  Estimated anchor (from CV): {estimated_anchor.round(3)}")
        print(f"  True anchor:                {mode_cfg.true_anchor.round(3)}")
        print()

    print("Ground truth:", env.get_ground_truth())
    return env


if __name__ == "__main__":
    print("=== Deterministic Mode Switching ===")
    run_diagnostic(SyntheticMOEnv, num_steps=600)
    print("\n" + "="*50 + "\n")
    print("=== Stochastic Mode Switching ===")
    run_diagnostic(SyntheticMOEnvStochastic, num_steps=600)
