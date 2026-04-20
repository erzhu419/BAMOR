"""
BAMOR Core Components — Code counterpart of BAMOR.lean Parts XIII/XIV/XV

This module implements the key BAMOR mechanisms that have no counterpart
in morl-baselines:

  Part XIII (ModeConeConstraint)  →  TargetConeEstimator
  Part XIV  (Multi-step tracking) →  BeliefTracker + SurpriseComputer
  Part XV   (Improvement quant.)  →  Diagnostics / logging

The core insight: each environment mode has a narrow valid preference
subspace (Target Cone). The algorithm must LEARN these boundaries from
data, not assume them a priori.

Typical pipeline:
    surprise = SurpriseComputer.compute(reward_vec)
    belief   = BeliefTracker.update(surprise)
    w, anchor = TargetConeEstimator.sample(belief)
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict


# ════════════════════════════════════════════════════════════════
# Part XIV correspondence: Bayesian Belief Tracker (BAPR import)
# ════════════════════════════════════════════════════════════════

class SurpriseComputer:
    """Mahalanobis-based surprise detector (Lean Part IV correspondence).

    Computes a multi-objective surprise signal by comparing observed
    rewards against a running prediction, normalized by per-objective
    aleatoric variance.

    Lean correspondence:
        mahalanobis_sq(surp, prec) = Σ_i prec_i · surp_i²
        mahalanobis_likelihood(surp, prec) = exp(-½ · mahalanobis_sq)
    """

    def __init__(self, reward_dim: int, ema_alpha: float = 0.05,
                 var_floor: float = 1e-4, warmup: int = 20):
        """
        Args:
            reward_dim:  number of objectives (m in Lean)
            ema_alpha:   exponential moving average rate for reward prediction
            var_floor:   minimum variance to prevent division by zero
            warmup:      steps before surprise is reliable
        """
        self.reward_dim = reward_dim
        self.ema_alpha = ema_alpha
        self.var_floor = var_floor
        self.warmup = warmup

        # Running statistics per objective
        self.reward_mean = np.zeros(reward_dim, dtype=np.float64)
        self.reward_var = np.ones(reward_dim, dtype=np.float64)
        self.step_count = 0

    def update_stats(self, reward_vec: np.ndarray):
        """Update running mean and variance (Welford's algorithm)."""
        self.step_count += 1
        delta = reward_vec - self.reward_mean
        self.reward_mean += self.ema_alpha * delta
        delta2 = reward_vec - self.reward_mean
        self.reward_var = (1 - self.ema_alpha) * self.reward_var + \
                          self.ema_alpha * delta * delta2

    def compute(self, reward_vec: np.ndarray) -> dict:
        """Compute Mahalanobis surprise from a new reward observation.

        Returns:
            dict with keys:
                'surprise_vec':  per-objective surprise  (Fin m → ℝ)
                'precision':     per-objective precision  (1/σ²)
                'mahalanobis_sq': scalar Mahalanobis distance²
                'likelihood':    exp(-½ · mahalanobis_sq)
                'is_warm':       whether we have enough data
        """
        reward_vec = np.asarray(reward_vec, dtype=np.float64)
        self.update_stats(reward_vec)

        is_warm = self.step_count > self.warmup

        # Per-objective surprise (residual)
        surprise_vec = reward_vec - self.reward_mean

        # Precision = 1 / max(variance, floor)
        precision = 1.0 / np.maximum(self.reward_var, self.var_floor)

        # Mahalanobis distance² = Σ prec_i · surp_i²
        mahalanobis_sq = np.sum(precision * surprise_vec ** 2)

        # Likelihood = exp(-½ · d²)  — always > 0 (Lean: mahalanobis_likelihood_pos)
        likelihood = np.exp(-0.5 * mahalanobis_sq)

        return {
            'surprise_vec': surprise_vec,
            'precision': precision,
            'mahalanobis_sq': mahalanobis_sq,
            'likelihood': likelihood,
            'is_warm': is_warm,
        }

    def reset(self):
        """Reset statistics (called on hard mode switch)."""
        self.reward_mean[:] = 0.0
        self.reward_var[:] = 1.0
        self.step_count = 0


class BeliefTracker:
    """Bayesian belief over environment modes / run-lengths.

    Lean correspondence:
        BAPR.update_belief(ρ, L, Z) = (ρ(h) · L(h)) / Z
        BAPR.normalization_const(ρ, L) = Σ_h ρ(h) · L(h)
        BAPR.update_belief_sum_one — guaranteed by construction

    The belief ρ is a probability distribution over H modes.
    After each surprise observation, we update via Bayes' rule.
    A high surprise triggers belief redistribution ("amnesic reset").
    """

    def __init__(self, num_modes: int, surprise_threshold: float = 8.0,
                 reset_prior: Optional[np.ndarray] = None,
                 switch_cooldown: int = 30):
        """
        Args:
            num_modes:           |H| — number of environment modes
            surprise_threshold:  Mahalanobis² threshold for mode switch
            reset_prior:         belief to reset to on mode switch (default: uniform)
            switch_cooldown:     minimum steps between mode switches (prevents
                                 rapid-fire re-triggers during transients)
        """
        self.num_modes = num_modes
        self.surprise_threshold = surprise_threshold
        self.switch_cooldown = switch_cooldown
        self.reset_prior = reset_prior if reset_prior is not None else \
            np.ones(num_modes, dtype=np.float64) / num_modes

        # Current belief ρ : H → ℝ (Lean: ρ_nn, ρ_sum)
        self.belief = self.reset_prior.copy()
        self.current_mode = np.argmax(self.belief)
        self.mode_switch_count = 0
        self.steps_since_switch = 0

    def update(self, surprise_info: dict,
               mode_likelihoods: Optional[np.ndarray] = None) -> dict:
        """Bayesian belief update.

        Lean correspondence:
            ρ' = update_belief(ρ, L, Z)
            Z  = normalization_const(ρ, L)  > 0 (guaranteed by mahalanobis_normalization_pos)

        Args:
            surprise_info:     output from SurpriseComputer.compute()
            mode_likelihoods:  L(h) per mode; if None, uses internal heuristic

        Returns:
            dict with 'belief', 'mode_switched', 'current_mode'
        """
        mahal_sq = surprise_info['mahalanobis_sq']
        self.steps_since_switch += 1

        # Detect mode switch via surprise threshold
        # Cooldown prevents rapid-fire re-triggers during transients:
        # when a regime changes, surprise stays high for several steps
        # while the EMA adapts. Without cooldown, each step would trigger
        # another round-robin increment, corrupting mode assignments.
        mode_switched = (surprise_info['is_warm'] and
                         mahal_sq > self.surprise_threshold and
                         self.steps_since_switch >= self.switch_cooldown)

        if mode_switched:
            # Amnesic reset — assign to next mode slot (round-robin)
            # This ensures different regimes get different mode indices,
            # enabling per-mode cone estimation to separate them
            old_mode = self.current_mode
            new_mode = (old_mode + 1) % self.num_modes

            self.belief = np.zeros(self.num_modes, dtype=np.float64)
            self.belief[new_mode] = 1.0  # concentrate on new mode
            self.current_mode = new_mode
            self.mode_switch_count += 1
            self.steps_since_switch = 0
        elif mode_likelihoods is not None:
            # Standard Bayes update: ρ'(h) = ρ(h)·L(h) / Z
            unnormalized = self.belief * mode_likelihoods
            Z = np.sum(unnormalized)
            if Z > 0:  # Lean: normalization_const > 0
                self.belief = unnormalized / Z
            # else: keep current belief (degenerate case)
            self.current_mode = np.argmax(self.belief)
        else:
            # No external likelihoods and no switch: slowly diffuse belief
            # towards current mode (reinforcement)
            decay = 0.99
            self.belief *= decay
            self.belief[self.current_mode] += (1 - decay)
            self.belief /= self.belief.sum()

        return {
            'belief': self.belief.copy(),
            'mode_switched': mode_switched,
            'current_mode': self.current_mode,
            'mode_switch_count': self.mode_switch_count,
            'steps_since_switch': self.steps_since_switch,
        }

    def reset(self):
        self.belief = self.reset_prior.copy()
        self.current_mode = np.argmax(self.belief)
        self.mode_switch_count = 0
        self.steps_since_switch = 0


# ════════════════════════════════════════════════════════════════
# Part XIII correspondence: Mode-Specific Preference Constraints
# ════════════════════════════════════════════════════════════════

@dataclass
class ModeConeConstraint:
    """Code counterpart of Lean's ModeConeConstraint structure.

    Lean correspondence:
        structure ModeConeConstraint (H : Type) (m : ℕ) where
            w_anchor     : H → Fin m → ℝ
            cone_radius  : H → ℝ
            h_anchor_nn  : ∀ h i, 0 ≤ w_anchor h i
            h_anchor_sum : ∀ h, Σ w_anchor h i = 1
            h_radius_pos : ∀ h, 0 < cone_radius h

    Each mode h has:
        - w_anchor[h]:    the physical preference center (on the simplex)
        - cone_radius[h]: allowed deviation from anchor
    """
    w_anchor: np.ndarray      # shape (num_modes, reward_dim) — simplex rows
    cone_radius: np.ndarray   # shape (num_modes,) — positive radii

    def __post_init__(self):
        """Verify Lean-level invariants (h_anchor_nn, h_anchor_sum, h_radius_pos)."""
        assert np.all(self.w_anchor >= 0), "h_anchor_nn violated: anchors must be non-negative"
        sums = self.w_anchor.sum(axis=1)
        assert np.allclose(sums, 1.0), f"h_anchor_sum violated: anchor rows must sum to 1, got {sums}"
        assert np.all(self.cone_radius > 0), "h_radius_pos violated: radii must be positive"

    @property
    def num_modes(self) -> int:
        return self.w_anchor.shape[0]

    @property
    def reward_dim(self) -> int:
        return self.w_anchor.shape[1]


class TargetConeEstimator:
    """Learns the ModeConeConstraint from data — the key innovation.

    Unlike the Lean proof (which assumes constraints are given), the
    algorithm must DISCOVER each mode's valid preference subspace by
    observing which objectives are physically responsive.

    Strategy: track per-objective reward variance per mode.
        - High variance → objective is responsive → high anchor weight
        - Near-zero variance → objective is collapsed → low anchor weight
        - This directly estimates the "substrate constraint" from data

    Lean correspondence:
        belief_weighted_anchor(mcc, ρ) = Σ_h ρ(h) · w_anchor(h)
        belief_weighted_anchor_nn  — guaranteed by construction
        belief_weighted_anchor_sum — guaranteed by construction
    """

    def __init__(self, num_modes: int, reward_dim: int,
                 window_size: int = 200, min_radius: float = 0.05,
                 max_radius: float = 0.45, sensitivity: float = 2.0,
                 normalize_rewards: bool = True):
        """
        Args:
            num_modes:          |H|
            reward_dim:         m — number of objectives
            window_size:        sliding window for reward variance estimation
            min_radius:         minimum cone radius (never collapse to a point)
            max_radius:         maximum cone radius (cap exploration)
            sensitivity:        how aggressively to shrink cone when variance drops
            normalize_rewards:  if True, z-score normalize rewards before std
                                estimation to remove cross-objective scale bias
        """
        self.num_modes = num_modes
        self.reward_dim = reward_dim
        self.window_size = window_size
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.sensitivity = sensitivity
        self.normalize_rewards = normalize_rewards

        # Per-mode reward history for variance estimation
        self.reward_history: List[deque] = [
            deque(maxlen=window_size) for _ in range(num_modes)
        ]

        # Global reward statistics for z-score normalization
        # Tracks overall mean/std across ALL modes, so that per-mode std
        # comparisons are relative to a shared baseline
        self._global_reward_mean = np.zeros(reward_dim, dtype=np.float64)
        self._global_reward_var = np.ones(reward_dim, dtype=np.float64)
        self._global_count = 0

        # Initialize with uniform anchors and moderate radii
        initial_anchor = np.ones((num_modes, reward_dim)) / reward_dim
        initial_radius = np.full(num_modes, (min_radius + max_radius) / 2)
        self.constraint = ModeConeConstraint(
            w_anchor=initial_anchor,
            cone_radius=initial_radius,
        )

        # Track which mode is being observed
        self._current_mode = 0

    def observe(self, reward_vec: np.ndarray, mode: int):
        """Record a reward observation for a specific mode.

        This is the data-driven learning of substrate constraints.
        """
        self.reward_history[mode].append(reward_vec.copy())
        self._current_mode = mode

        # Update global normalization statistics (Welford's online algorithm)
        self._global_count += 1
        delta = reward_vec - self._global_reward_mean
        self._global_reward_mean += delta / self._global_count
        delta2 = reward_vec - self._global_reward_mean
        self._global_reward_var += (delta * delta2 - self._global_reward_var) / self._global_count

        self._update_constraint(mode)

    def _update_constraint(self, mode: int):
        """Re-estimate the anchor and radius for a mode from its reward history.

        Core logic: objectives with higher reward standard deviation get higher
        anchor weight, because variance indicates the objective is physically
        responsive (the agent's actions / environment dynamics actually affect it).
        Collapsed objectives have near-zero variance → near-zero anchor weight.

        When normalize_rewards=True, rewards are z-score normalized against the
        global statistics BEFORE computing per-mode std. This removes the effect
        of cross-objective scale differences (e.g., when safety reward is in
        [-10, 0] but progress reward is in [-1, 1]).
        """
        history = self.reward_history[mode]
        if len(history) < 10:
            return  # not enough data

        rewards = np.array(history)  # (N, reward_dim)

        if self.normalize_rewards and self._global_count > 20:
            # Z-score normalize: remove global mean/std bias so that
            # per-mode std comparison is fair across objectives
            global_std = np.sqrt(np.maximum(self._global_reward_var, 1e-8))
            rewards = (rewards - self._global_reward_mean) / global_std

        # Per-objective standard deviation — direct responsiveness measure
        stds = np.std(rewards, axis=0) + 1e-8  # (reward_dim,)

        # Responsiveness score: direct std comparison with sensitivity exponent
        responsiveness = stds ** self.sensitivity

        # Normalize to simplex → anchor (Lean: h_anchor_sum)
        anchor = responsiveness / responsiveness.sum()
        anchor = np.clip(anchor, 0.01, None)  # Lean: h_anchor_nn (floor)
        anchor /= anchor.sum()  # re-normalize

        # Cone radius: inversely proportional to how "peaked" the anchor is
        # If one objective dominates (peaked anchor), radius is small
        # If all objectives are balanced, radius is large
        entropy = -np.sum(anchor * np.log(anchor + 1e-10))
        max_entropy = np.log(self.reward_dim)
        normalized_entropy = entropy / max_entropy  # ∈ [0, 1]

        radius = self.min_radius + normalized_entropy * (self.max_radius - self.min_radius)

        self.constraint.w_anchor[mode] = anchor
        self.constraint.cone_radius[mode] = radius

    def belief_weighted_anchor(self, belief: np.ndarray) -> np.ndarray:
        """Compute belief-weighted anchor: w̃ = Σ_h ρ(h) · w_anchor(h).

        Lean correspondence:
            belief_weighted_anchor(mcc, ρ) i = Σ_h ρ(h) · mcc.w_anchor(h, i)

        Guarantees (from Lean):
            belief_weighted_anchor_nn:  ∀ i, 0 ≤ w̃(i)
            belief_weighted_anchor_sum: Σ_i w̃(i) = 1
        """
        anchor = belief @ self.constraint.w_anchor  # (num_modes,) @ (num_modes, reward_dim)
        # Numerical safety: guarantee simplex constraints
        anchor = np.clip(anchor, 0.0, None)
        s = anchor.sum()
        if s > 0:
            anchor /= s
        else:
            anchor = np.ones(self.reward_dim) / self.reward_dim
        return anchor

    def belief_weighted_radius(self, belief: np.ndarray) -> float:
        """Compute belief-weighted cone radius."""
        return float(belief @ self.constraint.cone_radius)

    def get_diagnostics(self) -> dict:
        """Return current estimated constraints for logging."""
        return {
            'anchors': self.constraint.w_anchor.copy(),
            'radii': self.constraint.cone_radius.copy(),
            'history_sizes': [len(h) for h in self.reward_history],
        }


class ContextAwareWeightSampler:
    """Belief-driven Target Cone weight sampler.

    This is the code-level realization of the theoretical insight:
    "each environment mode has a narrow valid preference subspace."

    Instead of sampling w uniformly from the full simplex (as all
    morl-baselines algorithms do), we sample from a Gaussian centered
    on the belief-weighted anchor, truncated to the Target Cone.

    Lean correspondence:
        Part XI:  anchor_regularized_scalarize uses w_anchor
        Part XIII: belief_anchor_valid_for_regularization — strong concavity preserved
        Part XV:  target_cone_improvement_nonneg — never worse than uniform
    """

    def __init__(self, cone_estimator: TargetConeEstimator,
                 belief_tracker: BeliefTracker,
                 concentration = 10.0):
        """
        Args:
            cone_estimator:  estimates per-mode anchors and radii
            belief_tracker:  provides current belief ρ
            concentration:   Dirichlet concentration around anchor (higher = tighter).
                             Can be a scalar (same for all modes) or an array of shape
                             (num_modes,) for per-mode concentration.  The effective
                             value at sampling time is belief-weighted across modes.
        """
        self.cone_estimator = cone_estimator
        self.belief_tracker = belief_tracker
        num_modes = belief_tracker.num_modes
        conc = np.asarray(concentration, dtype=float)
        if conc.ndim == 0:
            conc = np.full(num_modes, float(concentration))
        if conc.shape != (num_modes,):
            raise ValueError(
                f"concentration must be scalar or array of length {num_modes}, "
                f"got shape {conc.shape}")
        self.concentration = conc  # shape (num_modes,)

    def _effective_concentration(self, belief: np.ndarray) -> float:
        """Belief-weighted concentration scalar."""
        return float(self.concentration @ belief)

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a weight vector and return (w, w_anchor).

        Returns:
            w:        sampled preference weight (on simplex, within Target Cone)
            w_anchor: belief-weighted anchor (for regularization loss)

        The weight is sampled from a Dirichlet distribution centered on the
        belief-weighted anchor:
            w ~ Dirichlet(concentration · w_anchor)

        This guarantees:
            1. w is on the simplex (Lean: hw_nn, hw_sum)
            2. w is concentrated around the anchor (Target Cone)
            3. Larger concentration → tighter cone → more suppression
        """
        belief = self.belief_tracker.belief
        w_anchor = self.cone_estimator.belief_weighted_anchor(belief)
        radius = self.cone_estimator.belief_weighted_radius(belief)

        # Adjust concentration based on cone radius
        # Smaller radius → higher concentration → tighter sampling
        base_conc = self._effective_concentration(belief)
        effective_concentration = base_conc / max(radius, 0.01)

        # Dirichlet parameter α = concentration · anchor
        # This centers the distribution on the anchor
        alpha = effective_concentration * w_anchor + 1e-3  # floor to prevent α=0

        # Sample from Dirichlet
        w = np.random.dirichlet(alpha).astype(np.float32)

        return w, w_anchor.astype(np.float32)

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a batch of weight vectors.

        Returns:
            ws:       (batch_size, reward_dim) — sampled weights
            anchors:  (batch_size, reward_dim) — repeated anchor
        """
        belief = self.belief_tracker.belief
        w_anchor = self.cone_estimator.belief_weighted_anchor(belief)
        radius = self.cone_estimator.belief_weighted_radius(belief)

        base_conc = self._effective_concentration(belief)
        effective_concentration = base_conc / max(radius, 0.01)
        alpha = effective_concentration * w_anchor + 1e-3

        ws = np.random.dirichlet(alpha, size=batch_size).astype(np.float32)
        anchors = np.tile(w_anchor.astype(np.float32), (batch_size, 1))

        return ws, anchors


# ════════════════════════════════════════════════════════════════
# Part XV correspondence: Tracking Diagnostics & Improvement Metrics
# ════════════════════════════════════════════════════════════════

class BAMORDiagnostics:
    """Tracks the quantities from Part XV for logging / paper figures.

    Lean correspondence:
        target_cone_improvement:      Δ_global − Δ_local
        target_cone_improvement_nonneg: Δ_local + γ·Δ_global/(1−γ) ≤ Δ_global/(1−γ)
        full_suppression_ratio:       when Δ_local=0, bound = γ·Δ_global/(1−γ)
    """

    def __init__(self, reward_dim: int, gamma: float = 0.99,
                 window_size: int = 500):
        self.reward_dim = reward_dim
        self.gamma = gamma
        self.window_size = window_size

        # Per-objective drift history (for estimating Δ_vec)
        self.drift_history: deque = deque(maxlen=window_size)
        self.prev_target_q: Optional[np.ndarray] = None

    def record_target_drift(self, target_q_vec: np.ndarray):
        """Record per-objective target Q drift.

        This estimates Δ_vec(i) = |Q*_{n+1}(i) − Q*_n(i)| from the
        target network's Q-values.
        """
        if self.prev_target_q is not None:
            drift = np.abs(target_q_vec - self.prev_target_q)
            self.drift_history.append(drift)
        self.prev_target_q = target_q_vec.copy()

    def compute_metrics(self, w: np.ndarray) -> dict:
        """Compute Part XV improvement metrics.

        Args:
            w: current preference weight (Target-Cone-aligned)

        Returns:
            dict with:
                'delta_global':  max_i Δ_vec(i) — worst-case per-objective drift
                'delta_local':   ⟨w, Δ_vec⟩ — Target-Cone-suppressed drift
                'naive_bound':   Δ_global / (1 − γ)
                'bamor_bound':   Δ_local + γ · Δ_global / (1 − γ)
                'improvement':   Δ_global − Δ_local
                'ratio':         bamor_bound / naive_bound
        """
        if len(self.drift_history) < 10:
            return {'delta_global': 0.0, 'delta_local': 0.0,
                    'naive_bound': 0.0, 'bamor_bound': 0.0,
                    'improvement': 0.0, 'ratio': 1.0}

        drift_array = np.array(self.drift_history)
        delta_vec = np.mean(drift_array, axis=0)  # average Δ_vec per objective

        delta_global = float(np.max(delta_vec))
        delta_local = float(np.dot(w, delta_vec))

        naive_bound = delta_global / (1 - self.gamma) if self.gamma < 1 else float('inf')
        bamor_bound = delta_local + self.gamma * delta_global / (1 - self.gamma) \
            if self.gamma < 1 else float('inf')

        improvement = delta_global - delta_local
        ratio = bamor_bound / naive_bound if naive_bound > 1e-10 else 1.0

        return {
            'delta_vec': delta_vec,
            'delta_global': delta_global,
            'delta_local': delta_local,
            'naive_bound': naive_bound,
            'bamor_bound': bamor_bound,
            'improvement': improvement,
            'ratio': ratio,
        }


# ════════════════════════════════════════════════════════════════
# Integration: BAMOR Pipeline (composes all components)
# ════════════════════════════════════════════════════════════════

class BAMORPipeline:
    """Complete BAMOR pipeline integrating all components.

    This is the top-level object that a SAC trainer interacts with.
    It replaces the simple `sample_weight()` function with a full
    belief-driven Target Cone system.

    Ablation support:
        disable_surprise=True  → no change-point detection (threshold=∞)
        disable_cone=True      → uniform weight sampling (concentration=1, no anchor)

    Usage in training loop:
        bamor = BAMORPipeline(num_modes=3, reward_dim=2)

        # Each step:
        surprise = bamor.observe_reward(reward_vec)
        w, w_anchor = bamor.sample_weight()
        metrics = bamor.get_metrics(w)

        # In loss computation:
        anchor_reg = bamor.anchor_regularization(w_tensor, anchor_tensor, eta=0.01)
    """

    def __init__(self, num_modes: int = 3, reward_dim: int = 2,
                 gamma: float = 0.99,
                 surprise_threshold: float = 8.0,
                 concentration = 10.0,
                 cone_window: int = 200,
                 ema_alpha: float = 0.05,
                 switch_cooldown: int = 80,
                 warmup: int = 50,
                 disable_surprise: bool = False,
                 disable_cone: bool = False):

        self.disable_surprise = disable_surprise
        self.disable_cone = disable_cone

        self.surprise_computer = SurpriseComputer(
            reward_dim=reward_dim, ema_alpha=ema_alpha, warmup=warmup)

        # Ablation: disable surprise → set threshold impossibly high
        effective_threshold = 1e10 if disable_surprise else surprise_threshold
        self.belief_tracker = BeliefTracker(
            num_modes=num_modes,
            surprise_threshold=effective_threshold,
            switch_cooldown=switch_cooldown)

        self.cone_estimator = TargetConeEstimator(
            num_modes=num_modes, reward_dim=reward_dim,
            window_size=cone_window, normalize_rewards=False)

        # Ablation: disable cone → very low concentration ≈ uniform sampling
        # concentration may be scalar or per-mode array; 0.1 scalar overrides all modes
        effective_concentration = 0.1 if disable_cone else concentration
        self.weight_sampler = ContextAwareWeightSampler(
            cone_estimator=self.cone_estimator,
            belief_tracker=self.belief_tracker,
            concentration=effective_concentration)

        self.diagnostics = BAMORDiagnostics(
            reward_dim=reward_dim, gamma=gamma)

        self.reward_dim = reward_dim
        self.num_modes = num_modes
        self._cone_grace_remaining = 0  # steps to skip cone estimation after switch

    def observe_reward(self, reward_vec: np.ndarray) -> dict:
        """Process a new reward observation through the full pipeline.

        Steps:
            1. Compute surprise (Part IV: Mahalanobis)
            2. Update belief (Part II/XIV: Bayesian update)
            3. If mode switched, reset surprise stats and start grace period
            4. Update cone estimate (skip during grace period to avoid contamination)

        Returns:
            Combined info dict from all components.
        """
        reward_vec = np.asarray(reward_vec, dtype=np.float64)

        # Step 1: Surprise
        surprise_info = self.surprise_computer.compute(reward_vec)

        # Step 2: Belief update
        belief_info = self.belief_tracker.update(surprise_info)

        # Step 3: Reset surprise computer on mode switch
        # so it quickly adapts to the new regime's statistics.
        # Also start a grace period for cone estimation: the first ~30 steps
        # after detection may still carry data from the old mode (detection lag).
        if belief_info['mode_switched']:
            self.surprise_computer.reset()
            self._cone_grace_remaining = 30

        # Step 4: Cone estimation (skip during grace period)
        if self._cone_grace_remaining > 0:
            self._cone_grace_remaining -= 1
        else:
            current_mode = belief_info['current_mode']
            self.cone_estimator.observe(reward_vec, current_mode)

        return {**surprise_info, **belief_info}

    def sample_weight(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a Target-Cone-restricted weight vector.

        Returns:
            (w, w_anchor) — both on the simplex
        """
        return self.weight_sampler.sample()

    def get_metrics(self, w: np.ndarray) -> dict:
        """Get Part XV diagnostic metrics."""
        return self.diagnostics.compute_metrics(w)

    def record_target_q(self, target_q_vec: np.ndarray):
        """Record target Q for drift estimation."""
        self.diagnostics.record_target_drift(target_q_vec)

    @staticmethod
    def anchor_regularization_loss(w, w_anchor, eta: float = 0.01):
        """Compute anchor-shifted regularization penalty.

        Lean correspondence:
            Part XI: anchor_regularized_scalarize = ⟨w, Q⟩ − η·‖w − w_anchor‖²
            Part XIII: belief_anchor_valid_for_regularization — preserves strong concavity

        This returns the PENALTY term: η · ‖w − w_anchor‖²
        which should be SUBTRACTED from the actor loss (or ADDED to the
        loss if we're minimizing).

        Args:
            w:        current weight (torch tensor, (batch, reward_dim) or (reward_dim,))
            w_anchor: target anchor (torch tensor, same shape)
            eta:      regularization strength

        Returns:
            Scalar penalty: η · mean(‖w − w_anchor‖²)
        """
        import torch
        diff = w - w_anchor
        penalty = eta * (diff ** 2).sum(dim=-1).mean()
        return penalty

    def get_state(self) -> dict:
        """Serialize for checkpointing."""
        return {
            'belief': self.belief_tracker.belief.copy(),
            'anchors': self.cone_estimator.constraint.w_anchor.copy(),
            'radii': self.cone_estimator.constraint.cone_radius.copy(),
            'reward_mean': self.surprise_computer.reward_mean.copy(),
            'reward_var': self.surprise_computer.reward_var.copy(),
            'step_count': self.surprise_computer.step_count,
            'mode_switch_count': self.belief_tracker.mode_switch_count,
        }

    def load_state(self, state: dict):
        """Restore from checkpoint."""
        self.belief_tracker.belief = state['belief']
        self.cone_estimator.constraint.w_anchor = state['anchors']
        self.cone_estimator.constraint.cone_radius = state['radii']
        self.surprise_computer.reward_mean = state['reward_mean']
        self.surprise_computer.reward_var = state['reward_var']
        self.surprise_computer.step_count = state['step_count']
        self.belief_tracker.mode_switch_count = state['mode_switch_count']

    def summary(self) -> str:
        """Human-readable summary of BAMOR state."""
        belief = self.belief_tracker.belief
        mode = self.belief_tracker.current_mode
        anchor = self.cone_estimator.belief_weighted_anchor(belief)
        radius = self.cone_estimator.belief_weighted_radius(belief)

        ablation_str = ""
        if self.disable_surprise:
            ablation_str += " [ABLATION: surprise OFF]"
        if self.disable_cone:
            ablation_str += " [ABLATION: cone OFF]"

        lines = [
            f"═══ BAMOR Pipeline Status{ablation_str} ═══",
            f"  Current mode:      {mode} (belief: {belief})",
            f"  Mode switches:     {self.belief_tracker.mode_switch_count}",
            f"  Steps since switch:{self.belief_tracker.steps_since_switch}",
            f"  Belief anchor:     {anchor}",
            f"  Cone radius:       {radius:.4f}",
            f"  Per-mode anchors:",
        ]
        for h in range(self.num_modes):
            a = self.cone_estimator.constraint.w_anchor[h]
            r = self.cone_estimator.constraint.cone_radius[h]
            n = len(self.cone_estimator.reward_history[h])
            lines.append(f"    Mode {h}: anchor={a}, radius={r:.4f}, samples={n}")
        return "\n".join(lines)
