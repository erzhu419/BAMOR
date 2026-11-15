"""
Piecewise-Stationary Multi-Objective Environment Wrapper

Embodies the BAMOR insight: each mode's substrate constraints restrict
the achievable Pareto front to a narrow Target Cone. Multiple modes
tile different slices of the full Pareto front.

Design Principle:
  - Mode A (e.g., high gravity): speed objective physically capped →
    valid preferences near w_energy ≈ 1, narrow cone around energy axis
  - Mode B (e.g., low friction): energy penalty low, speed achievable →
    valid preferences near w_speed ≈ 1, narrow cone around speed axis
  - Mode C (e.g., normal): balanced trade-off → moderate cone

Together, modes A+B+C tile different segments of the Pareto front.
Traditional MORL wastes capacity searching the full simplex per mode;
BAMOR restricts to the current Target Cone.

Usage:
    env = PiecewiseStationaryMOEnv(
        base_env_id="mo-halfcheetah-v5",
        mode_configs=[...],  # or use presets
        switch_interval=200_000,
    )
"""

import gymnasium as gym
import mo_gymnasium  # noqa: F401 — registers MO envs
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence
import copy


@dataclass
class ModeConfig:
    """Configuration for a single environment mode.

    Each mode defines TWO layers of substrate constraints:
    1. Physics perturbations (gravity/friction/mass) — changes dynamics
    2. Reward scaling — directly changes which objectives are achievable

    The reward_scale is the PRIMARY mechanism for creating Target Cones:
    - reward_scale=[3.0, 0.3] means objective 0 is 10x more responsive than obj 1
    - This forces the optimal preference toward obj 0 (high anchor weight)
    - An agent optimizing obj 1 with w=[0.1, 0.9] gets almost nothing in this mode

    Physics perturbations are SECONDARY: they create realistic dynamics shifts
    that also affect the reward landscape indirectly.
    """
    name: str
    # === Reward scaling (LEGACY — now optional) ===
    reward_scale: np.ndarray = None  # multiply each reward component; shape=(reward_dim,)
    reward_shift: np.ndarray = None  # additive shift per component (optional)
    # === Physics perturbations (PRIMARY in BAPR-style modes) ===
    gravity_scale: float = 1.0           # multiply gravity magnitude
    friction_scale: float = 1.0          # multiply all friction coefficients
    mass_scale: float = 1.0              # multiply all body masses
    dof_damping_scale: float = 1.0       # multiply joint damping (BAPR addition)
    # === Hard substrate constraints (NEW) ===
    # reward_cap[i] = per-step UPPER cap on objective i (np.inf = no cap).
    # This is the *substrate-collapse* mechanism BAMOR's theory rests on:
    # under this mode, no policy can earn more than reward_cap[i] on obj i,
    # regardless of action — the cone of physically-feasible preferences shrinks.
    reward_cap: np.ndarray = None
    # === Ground truth (for evaluation only; None = let estimator discover) ===
    true_anchor: np.ndarray = None
    cone_radius: float = 0.15            # how narrow the valid preference space is
    description: str = ""

    def __post_init__(self):
        if self.true_anchor is not None:
            self.true_anchor = np.asarray(self.true_anchor, dtype=np.float64)
            assert np.all(self.true_anchor >= 0), "Anchor must be non-negative"
            assert np.isclose(self.true_anchor.sum(), 1.0), "Anchor must sum to 1"
        if self.reward_scale is not None:
            self.reward_scale = np.asarray(self.reward_scale, dtype=np.float64)
            assert np.all(self.reward_scale >= 0), "Reward scale must be non-negative"
        if self.reward_shift is not None:
            self.reward_shift = np.asarray(self.reward_shift, dtype=np.float64)
        if self.reward_cap is not None:
            self.reward_cap = np.asarray(self.reward_cap, dtype=np.float64)


# ================================================================
# Preset mode configurations for standard MO-MuJoCo environments
# ================================================================

def halfcheetah_modes() -> "list[ModeConfig]":
    """HalfCheetah-v5: 2 objectives [x_velocity, -energy_cost].

    LEGACY preset (reward_scale-driven).  Kept for backwards compatibility
    with the original BAMOR experiments — see `halfcheetah_bapr_modes` for
    the BAPR-style (physics-only) variant we now use as default.
    """
    return [
        ModeConfig(
            name="efficiency",
            reward_scale=np.array([0.1, 5.0]),
            mass_scale=2.5,
            gravity_scale=1.5,
            true_anchor=np.array([0.10, 0.90]),
            cone_radius=0.08,
            description="Energy-dominated: speed reward suppressed 50x, energy amplified",
        ),
        ModeConfig(
            name="sprint",
            reward_scale=np.array([5.0, 0.1]),
            friction_scale=0.3,
            true_anchor=np.array([0.90, 0.10]),
            cone_radius=0.08,
            description="Speed-dominated: speed reward amplified 50x, energy suppressed",
        ),
        ModeConfig(
            name="balanced",
            reward_scale=np.array([1.0, 1.0]),
            true_anchor=np.array([0.5, 0.5]),
            cone_radius=0.20,
            description="Balanced: both objectives equally responsive",
        ),
    ]


def halfcheetah_bapr_modes() -> "list[ModeConfig]":
    """HalfCheetah-v5 BAPR-style + hard substrate caps.

    Each mode combines BAPR-style physics perturbations with a *per-step
    hard reward cap* on one or both objectives.  The caps are the substrate
    constraints BAMOR's Target Cone is designed to exploit: under a cap,
    every preference vector that asks for more than the cap on the capped
    objective is *physically unreachable*.

    For mo-halfcheetah-v5 (per-step rewards):
      obj 0 = forward_velocity   (typical 0–15 / step, episode 200–600 total)
      obj 1 = -control_cost      (typical -1.5 to -0.05 / step, episode -1500 to -50)

    Modes:
      - normal       : no cap, full simplex valid          → anchor [0.5, 0.5]
      - low_speed    : speed capped at 0.3/step            → anchor [0.15, 0.85]
                       (hard substrate: progress collapsed; energy regime)
      - low_cost     : -control_cost capped at -0.05/step  → anchor [0.85, 0.15]
                       (hard substrate: cost-budget infinite; speed regime)
      - locked       : both capped (small)                 → anchor [0.5, 0.5]
                       (hard substrate: rigid joints, narrow operating band)
    """
    return [
        ModeConfig(
            name="normal",
            true_anchor=np.array([0.5, 0.5]),
            cone_radius=0.30,
            description="Baseline physics, no caps — full simplex valid.",
        ),
        ModeConfig(
            name="low_speed",
            mass_scale=2.0,
            gravity_scale=0.6,
            reward_cap=np.array([0.3, np.inf]),
            true_anchor=np.array([0.15, 0.85]),
            cone_radius=0.15,
            description=(
                "Heavy + weak gravity + speed hard-capped at 0.3/step → "
                "energy-dominated cone."
            ),
        ),
        ModeConfig(
            name="low_cost",
            mass_scale=0.5,
            gravity_scale=1.5,
            reward_cap=np.array([np.inf, -0.05]),
            true_anchor=np.array([0.85, 0.15]),
            cone_radius=0.15,
            description=(
                "Light + strong gravity + control-cost hard-capped at -0.05/step → "
                "speed-dominated cone."
            ),
        ),
        ModeConfig(
            name="locked",
            dof_damping_scale=3.0,
            reward_cap=np.array([1.0, -0.2]),
            true_anchor=np.array([0.5, 0.5]),
            cone_radius=0.15,
            description=(
                "Rigid joints + both objectives hard-capped → narrow operating band."
            ),
        ),
    ]


def hopper_modes() -> "list[ModeConfig]":
    """Hopper-v5: [forward velocity, jumping height, -unweighted control cost].

    MO-Gym adds the survival reward to every component before mode scaling.
    Mode A (Reckless): jumping component amplified, forward/cost suppressed.
    Mode B (Survival): forward component amplified, jumping/cost suppressed.
    Mode C (Efficient): cost component amplified, forward/jumping suppressed.
    """
    return [
        ModeConfig(
            name="reckless",
            reward_scale=np.array([0.1, 5.0, 0.1]),
            gravity_scale=0.6,
            true_anchor=np.array([0.05, 0.90, 0.05]),
            cone_radius=0.06,
            description="Jumping-dominated 50x: forward and cost components suppressed"
        ),
        ModeConfig(
            name="survival",
            reward_scale=np.array([5.0, 0.1, 0.2]),
            gravity_scale=1.8,
            true_anchor=np.array([0.85, 0.05, 0.10]),
            cone_radius=0.06,
            description="Forward-dominated 50x: jumping and cost components suppressed"
        ),
        ModeConfig(
            name="efficient",
            reward_scale=np.array([0.2, 0.2, 5.0]),
            true_anchor=np.array([0.05, 0.05, 0.90]),
            cone_radius=0.06,
            description="Energy-dominated 50x: minimize cost"
        ),
    ]


def walker2d_modes() -> "list[ModeConfig]":
    """Walker2d-v5: 2 objectives [speed, -energy_cost].

    Same 2-obj structure as HalfCheetah, different physics.
    """
    return [
        ModeConfig(
            name="uphill",
            reward_scale=np.array([0.1, 5.0]),
            gravity_scale=1.8,
            mass_scale=1.5,
            true_anchor=np.array([0.10, 0.90]),
            cone_radius=0.08,
            description="Uphill: speed suppressed 50x, energy amplified"
        ),
        ModeConfig(
            name="downhill",
            reward_scale=np.array([5.0, 0.1]),
            gravity_scale=0.6,
            true_anchor=np.array([0.90, 0.10]),
            cone_radius=0.08,
            description="Downhill: speed amplified 50x, energy suppressed"
        ),
        ModeConfig(
            name="flat",
            reward_scale=np.array([1.0, 1.0]),
            friction_scale=1.5,
            true_anchor=np.array([0.50, 0.50]),
            cone_radius=0.20,
            description="Flat road: balanced trade-off"
        ),
    ]


def mountaincar_bapr_modes() -> "list[ModeConfig]":
    """mo-mountaincarcontinuous-v0 BAPR-style: 2 objectives, 3 modes.

    obj 0 = -1 per step (time penalty, reach goal fast)
    obj 1 = -squared_action (energy cost)

    Substrate constraints via soft caps + physics scale (gravity proxy via reward_scale).
    Modes:
      - normal      : default, no caps, full simplex valid
      - slow_only   : time-penalty soft-capped tight; only energy matters
      - energy_only : energy-penalty soft-capped tight; only time matters
    """
    return [
        ModeConfig(
            name="normal",
            true_anchor=np.array([0.5, 0.5]),
            cone_radius=0.30,
            description="Baseline — full speed/energy trade-off available.",
        ),
        ModeConfig(
            name="slow_only",
            reward_cap=np.array([-2.0, np.inf]),  # time-penalty floor: episode max -2000 (vs -1000 typical)
            true_anchor=np.array([0.15, 0.85]),
            cone_radius=0.15,
            description="Time-penalty heavily capped → energy-dominated cone.",
        ),
        ModeConfig(
            name="energy_only",
            reward_cap=np.array([np.inf, -0.005]),  # energy-cost cap small: still penalised but bounded
            true_anchor=np.array([0.85, 0.15]),
            cone_radius=0.15,
            description="Energy-cost soft-capped → time/speed-dominated cone.",
        ),
    ]


def hopper_bapr_modes() -> "list[ModeConfig]":
    """mo-hopper-v5: [forward velocity, jumping height, -control cost], 4 modes.

    MO-Gym adds the survival reward to every component before mode caps.
    Substrate caps in different objectives create disjoint Target Cones.
    Same soft-cap mechanism via tanh applied in step().
    """
    return [
        ModeConfig(
            name="normal",
            true_anchor=np.array([1/3, 1/3, 1/3]),
            cone_radius=0.30,
            description="Baseline — all 3 objectives reachable.",
        ),
        ModeConfig(
            name="fragile",
            reward_cap=np.array([np.inf, 0.5, np.inf]),  # jumping component capped
            mass_scale=1.5,
            true_anchor=np.array([0.85, 0.05, 0.10]),
            cone_radius=0.15,
            description="Forward-dominated: jumping component soft-capped, increased mass.",
        ),
        ModeConfig(
            name="cheap_energy",
            reward_cap=np.array([np.inf, np.inf, -0.05]),  # energy cost cap small
            true_anchor=np.array([0.10, 0.85, 0.05]),
            cone_radius=0.15,
            description="Jumping-dominated: cost component soft-capped.",
        ),
        ModeConfig(
            name="grounded",      # cannot jump high, must conserve energy
            reward_cap=np.array([0.5, 0.3, np.inf]),  # forward and jumping components capped
            gravity_scale=1.5,
            true_anchor=np.array([0.05, 0.05, 0.90]),
            cone_radius=0.15,
            description="Cost-dominated: forward and jumping components capped, increased gravity.",
        ),
    ]


def ant_modes() -> "list[ModeConfig]":
    """Ant-v5: [x_velocity, y_velocity, -unweighted_control_cost].

    MO-Gym adds survival and contact rewards to every component before these
    mode scales apply. Mode names are historical; there is no health-only axis.
    """
    return [
        ModeConfig(
            name="energy_save",
            reward_scale=np.array([0.1, 0.1, 5.0]),
            mass_scale=2.0,
            friction_scale=1.5,
            true_anchor=np.array([0.05, 0.05, 0.90]),
            cone_radius=0.06,
            description="Control-cost component amplified 50x, including shared bonuses; heavy + high friction"
        ),
        ModeConfig(
            name="race",
            reward_scale=np.array([0.1, 5.0, 0.1]),
            gravity_scale=0.7,
            friction_scale=0.5,
            true_anchor=np.array([0.05, 0.90, 0.05]),
            cone_radius=0.06,
            description="Y-velocity component amplified 50x, including shared bonuses; low gravity + friction"
        ),
        ModeConfig(
            name="cautious",
            reward_scale=np.array([5.0, 0.1, 0.2]),
            gravity_scale=1.5,
            friction_scale=0.3,
            true_anchor=np.array([0.85, 0.05, 0.10]),
            cone_radius=0.06,
            description="X-velocity component emphasized, including shared bonuses; high gravity + low friction"
        ),
    ]


ENV_PRESETS = {
    "mo-halfcheetah-v5": halfcheetah_bapr_modes,   # default: BAPR-style + soft caps
    "mo-hopper-v5": hopper_bapr_modes,             # default: BAPR-style + soft caps
    "mo-mountaincarcontinuous-v0": mountaincar_bapr_modes,  # simplest 2D continuous
    "mo-walker2d-v5": walker2d_modes,
    "mo-ant-v5": ant_modes,
    # v4 aliases
    "mo-halfcheetah-v4": halfcheetah_bapr_modes,
    "mo-hopper-v4": hopper_bapr_modes,
    "mo-walker2d-v4": walker2d_modes,
    "mo-ant-v4": ant_modes,
}

# Legacy presets (reward-scale-driven) accessible by explicit lookup
ENV_PRESETS_LEGACY = {
    "mo-halfcheetah-v5": halfcheetah_modes,
    "mo-halfcheetah-v4": halfcheetah_modes,
}


class PiecewiseStationaryMOEnv(gym.Wrapper):
    """Wraps a mo-gymnasium environment with piecewise-stationary mode switching.

    Each mode applies different physics parameters, creating substrate
    constraints that restrict the achievable Pareto front to a narrow
    Target Cone. Modes switch on a schedule (deterministic or stochastic).

    Key properties:
    - `current_mode_idx`: which mode is currently active
    - `current_mode`: the active ModeConfig
    - `mode_history`: list of (timestep, mode_idx) recording all switches
    - `get_ground_truth()`: returns anchors, radii, switch schedule

    Compatible with morl-baselines training loops.
    """

    def __init__(
        self,
        base_env_id: str = "mo-halfcheetah-v5",
        mode_configs: Optional[Sequence[ModeConfig]] = None,
        switch_interval: int = 200_000,
        switch_jitter: float = 0.0,
        mode_order: str = "sequential",   # "sequential", "random", or "exponential"
        seed: int = 0,
        **env_kwargs,
    ):
        """
        Args:
            base_env_id: mo-gymnasium environment ID
            mode_configs: list of ModeConfig. If None, uses preset for base_env_id
            switch_interval: steps between mode switches (mean if jitter > 0)
            switch_jitter: fraction of switch_interval to jitter (0 = deterministic)
            mode_order: "sequential" cycles A→B→C→A...; "random" picks uniformly
            seed: random seed
            **env_kwargs: passed to gym.make
        """
        env = gym.make(base_env_id, **env_kwargs)
        super().__init__(env)

        # Mode configuration
        if mode_configs is None:
            if base_env_id in ENV_PRESETS:
                mode_configs = ENV_PRESETS[base_env_id]()
            else:
                raise ValueError(
                    f"No preset modes for '{base_env_id}'. "
                    f"Available: {list(ENV_PRESETS.keys())}. "
                    f"Provide mode_configs explicitly."
                )
        self.mode_configs = list(mode_configs)
        self.num_modes = len(self.mode_configs)
        assert self.num_modes >= 2, "Need at least 2 modes for piecewise stationarity"

        # Verify reward dimension matches anchors
        obs, _ = env.reset(seed=seed)
        act = env.action_space.sample()
        _, rew, _, _, _ = env.step(act)
        self.reward_dim = len(rew)
        for mc in self.mode_configs:
            if mc.true_anchor is not None:
                assert len(mc.true_anchor) == self.reward_dim, \
                    f"Mode '{mc.name}': anchor dim {len(mc.true_anchor)} != reward dim {self.reward_dim}"

        # Store original physics for restoration
        if hasattr(self.unwrapped, 'model'):
            model = self.unwrapped.model
            self._orig_gravity = model.opt.gravity.copy()
            self._orig_body_mass = model.body_mass.copy()
            self._orig_geom_friction = model.geom_friction.copy()
            # dof_damping (BAPR addition) — shape (nv,)
            if hasattr(model, 'dof_damping'):
                self._orig_dof_damping = model.dof_damping.copy()
            else:
                self._orig_dof_damping = None
        else:
            self._orig_gravity = None
            self._orig_dof_damping = None

        # Switching schedule
        self.switch_interval = switch_interval
        self.switch_jitter = switch_jitter
        self.mode_order = mode_order

        # State
        self._rng = np.random.RandomState(seed)
        self._total_steps = 0
        self._steps_in_mode = 0
        self._next_switch = self._sample_next_switch()
        self.current_mode_idx = 0
        self.mode_history: "list[tuple[int, int]]" = [(0, 0)]  # (step, mode_idx)
        self._lock_mode = False   # if True, _maybe_switch_mode is a no-op (eval)

        # Apply initial mode
        self._apply_mode(0)

    @property
    def current_mode(self) -> ModeConfig:
        return self.mode_configs[self.current_mode_idx]

    def _sample_next_switch(self) -> int:
        """Sample next switch interval (deterministic / jitter / exponential)."""
        if self.mode_order == "exponential":
            # BAPR-style: dwell ~ Exponential(mean=switch_interval).  Floor at
            # 1000 steps to avoid degenerate near-zero dwells.
            d = int(self._rng.exponential(self.switch_interval))
            return max(d, 1000)
        if self.switch_jitter > 0:
            lo = int(self.switch_interval * (1 - self.switch_jitter))
            hi = int(self.switch_interval * (1 + self.switch_jitter))
            return self._rng.randint(lo, max(lo + 1, hi))
        return self.switch_interval

    def _apply_mode(self, mode_idx: int):
        """Apply physics parameters for the given mode."""
        mc = self.mode_configs[mode_idx]
        if self._orig_gravity is not None:
            model = self.unwrapped.model
            model.opt.gravity[:] = self._orig_gravity * mc.gravity_scale
            model.body_mass[:] = self._orig_body_mass * mc.mass_scale
            model.geom_friction[:] = self._orig_geom_friction * mc.friction_scale
            if self._orig_dof_damping is not None and hasattr(model, 'dof_damping'):
                model.dof_damping[:] = self._orig_dof_damping * mc.dof_damping_scale
        self.current_mode_idx = mode_idx

    def _maybe_switch_mode(self):
        """Check if it's time to switch modes."""
        if self._lock_mode:
            return
        if self._steps_in_mode >= self._next_switch:
            # Determine next mode
            if self.mode_order == "sequential":
                new_mode = (self.current_mode_idx + 1) % self.num_modes
            else:  # random or exponential — both pick uniformly from non-current
                candidates = [i for i in range(self.num_modes) if i != self.current_mode_idx]
                new_mode = int(self._rng.choice(candidates))

            self._apply_mode(new_mode)
            self.mode_history.append((self._total_steps, new_mode))
            self._steps_in_mode = 0
            self._next_switch = self._sample_next_switch()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._total_steps += 1
        self._steps_in_mode += 1

        # Apply mode-dependent reward scaling (LEGACY substrate constraint)
        mc = self.current_mode
        if mc.reward_scale is not None:
            reward = reward * mc.reward_scale
        if mc.reward_shift is not None:
            reward = reward + mc.reward_shift
        # Hard substrate cap: per-step upper ceiling (BAMOR-TC's home turf).
        # An objective whose cap is reached cannot be exceeded by any policy →
        # preferences asking for more than the cap are physically unreachable.
        if mc.reward_cap is not None:
            # Smooth cap (gpt-fix v3c): replace hard `min` with `tanh`-style
            # ceiling so the gradient stays continuous at the cap boundary.
            # For finite cap c: r_capped = c · tanh(r / c) (matches r when |r| ≪ c,
            # asymptotes to c as r → ∞).  np.inf entries (no cap) pass through.
            cap = mc.reward_cap
            for i in range(len(reward)):
                if np.isfinite(cap[i]):
                    if cap[i] > 0:
                        reward[i] = cap[i] * np.tanh(reward[i] / cap[i])
                    elif cap[i] < 0:
                        # For negative cap (upper bound on -loss type rewards),
                        # mirror via abs and re-negate.
                        a = abs(cap[i])
                        reward[i] = -a * np.tanh(-reward[i] / a)

        # Inject mode info (not visible to agent, for logging/eval only)
        info["mode_idx"] = self.current_mode_idx
        info["mode_name"] = mc.name
        info["true_anchor"] = mc.true_anchor
        info["steps_in_mode"] = self._steps_in_mode
        info["reward_unscaled"] = info.get("reward_unscaled", reward)  # original

        # Check for mode switch
        self._maybe_switch_mode()

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        # Reset episode but keep mode state (piecewise-stationary across episodes)
        obs, info = self.env.reset(seed=seed, options=options)
        # Re-apply current mode physics (in case env.reset restored defaults)
        self._apply_mode(self.current_mode_idx)
        info["mode_idx"] = self.current_mode_idx
        info["mode_name"] = self.current_mode.name
        info["true_anchor"] = self.current_mode.true_anchor
        return obs, info

    def get_ground_truth(self) -> dict:
        """Return ground truth for evaluation.

        Returns:
            dict with:
                - mode_configs: list of mode configs with anchors and radii
                - mode_history: list of (step, mode_idx) switches
                - num_modes: number of modes
                - reward_dim: dimensionality of reward
        """
        return {
            "mode_configs": [
                {
                    "name": mc.name,
                    "true_anchor": mc.true_anchor.tolist() if mc.true_anchor is not None else None,
                    "cone_radius": mc.cone_radius,
                    "gravity_scale": mc.gravity_scale,
                    "friction_scale": mc.friction_scale,
                    "mass_scale": mc.mass_scale,
                    "description": mc.description,
                }
                for mc in self.mode_configs
            ],
            "mode_history": list(self.mode_history),
            "num_modes": self.num_modes,
            "reward_dim": self.reward_dim,
        }

    def get_all_anchors(self) -> np.ndarray:
        """Return (num_modes, reward_dim) array of all mode anchors."""
        return np.array([mc.true_anchor for mc in self.mode_configs])

    def get_all_radii(self) -> np.ndarray:
        """Return (num_modes,) array of all cone radii."""
        return np.array([mc.cone_radius for mc in self.mode_configs])


def make_piecewise_mo_env(
    env_name: str = "mo-halfcheetah-v5",
    switch_interval: int = 50_000,
    switch_jitter: float = 0.0,
    mode_order: str = "exponential",   # BAPR-style: random dwell + random next-mode
    seed: int = 0,
) -> PiecewiseStationaryMOEnv:
    """Convenience factory for creating piecewise-stationary MO environments.

    Defaults match the BAPR-style sweet-spot env (BAPR_ENV_DESIGN.md):
    - 4 discrete physics-only modes (see `halfcheetah_bapr_modes`)
    - dwell time ~ Exponential(50K env steps), random next-mode
    - no reward shaping; mo-gym's native MO reward unmodified
    """
    return PiecewiseStationaryMOEnv(
        base_env_id=env_name,
        switch_interval=switch_interval,
        switch_jitter=switch_jitter,
        mode_order=mode_order,
        seed=seed,
    )


# ================================================================
# Quick test
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing PiecewiseStationaryMOEnv")
    print("=" * 60)

    for env_id in ["mo-halfcheetah-v5", "mo-hopper-v5"]:
        print(f"\n--- {env_id} ---")
        env = make_piecewise_mo_env(env_id, switch_interval=100, seed=42)

        gt = env.get_ground_truth()
        print(f"  Modes: {gt['num_modes']}, Reward dim: {gt['reward_dim']}")
        for mc in gt["mode_configs"]:
            print(f"    {mc['name']:15s}: anchor={mc['true_anchor']}, "
                  f"radius={mc['cone_radius']:.2f}, {mc['description'][:50]}")

        # Run a few steps to trigger mode switches
        obs, info = env.reset()
        mode_counts = {}
        total_rewards = []
        for step in range(350):
            action = env.action_space.sample()
            obs, rew, term, trunc, info = env.step(action)
            mode_name = info["mode_name"]
            mode_counts[mode_name] = mode_counts.get(mode_name, 0) + 1
            total_rewards.append(rew)
            if term or trunc:
                obs, info = env.reset()

        print(f"  Steps per mode: {mode_counts}")
        print(f"  Mode switches: {env.mode_history}")
        rewards = np.array(total_rewards)
        print(f"  Reward stats: mean={rewards.mean(axis=0).round(3)}, "
              f"std={rewards.std(axis=0).round(3)}")

        # Verify anchors tile the simplex
        anchors = env.get_all_anchors()
        print(f"  Anchor coverage:")
        for i, mc in enumerate(env.mode_configs):
            print(f"    Mode {i} ({mc.name}): anchor={anchors[i].round(3)}, "
                  f"covers w₁∈[{max(0, anchors[i][0]-mc.cone_radius):.2f}, "
                  f"{min(1, anchors[i][0]+mc.cone_radius):.2f}]")

        env.close()

    print("\n✅ All tests passed!")
