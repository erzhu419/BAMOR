"""BAMOR-specific components: Mahalanobis Surprise + Target Cone Sampler.

These are numpy-based (called once per multi_update, not inside lax.scan).

Corresponds to:
  - BAMOR.lean Part IV (Mahalanobis likelihood)
  - BAMOR.lean Part XI/XII (Target Cone + Anchor-Shifted Regularization)
"""
import numpy as np


class MahalanobisSurpriseComputer:
    """Multi-objective surprise via Mahalanobis distance.

    Unlike BAPR's scalar SurpriseComputer, this uses per-objective Q-std
    divergence weighted by precision (inverse aleatoric variance).

    Corresponds to BAMOR.lean L467-468: mahalanobis_sq = Σ prec_i * surp_i²
    """

    def __init__(self, num_objectives: int, ema_alpha: float = 0.3):
        self.m = num_objectives
        self.ema_alpha = ema_alpha
        self.ema_surprise = np.zeros(num_objectives)

    def compute(self, per_obj_q_std: np.ndarray,
                per_obj_q_std_prev: np.ndarray = None) -> tuple:
        """Compute Mahalanobis surprise from per-objective Q-std.

        Args:
            per_obj_q_std: [m] current per-objective ensemble Q-std mean
            per_obj_q_std_prev: [m] previous (or target) per-objective Q-std

        Returns:
            mahalanobis_surprise: float — scalar for belief_tracker.update()
            per_obj_surprise: [m] — per-objective surprise for logging
        """
        if per_obj_q_std_prev is None:
            per_obj_q_std_prev = self.ema_surprise

        raw_surp = np.abs(per_obj_q_std - per_obj_q_std_prev)

        # EMA smoothing
        self.ema_surprise = (self.ema_alpha * raw_surp +
                             (1 - self.ema_alpha) * self.ema_surprise)

        # Precision = 1 / variance (high Q-std → low precision → downweight)
        aleatoric_var = per_obj_q_std + 1e-6
        precision = 1.0 / aleatoric_var

        # Mahalanobis distance
        mahalanobis_sq = (precision * self.ema_surprise ** 2).sum()
        mahalanobis_surprise = np.sqrt(mahalanobis_sq)

        return float(mahalanobis_surprise), self.ema_surprise.copy()

    def reset(self):
        self.ema_surprise = np.zeros(self.m)


# ─── Target Cone Profiles ───
# Maps run-length (stability) to physically valid Target Cone boundaries.
# Short run-length (crisis) → narrow cone biased towards objective 0.
# Long run-length (stable)  → wide cone allowing balanced exploration.

MODE_CONE_PROFILES = [
    # Crisis (h=0~4): just changed, very unstable → focus on primary objective
    {"mode": "crisis",   "cone_lower": 0.8, "cone_upper": 1.0, "w_anchor": [0.9, 0.1]},
    # Recovery (h=5~14): recovering → gradually open cone
    {"mode": "recovery", "cone_lower": 0.5, "cone_upper": 0.8, "w_anchor": [0.65, 0.35]},
    # Stable (h=15~19): long stable → wide cone
    {"mode": "stable",   "cone_lower": 0.2, "cone_upper": 0.6, "w_anchor": [0.4, 0.6]},
]


def get_cone_mixing_weights(belief_array: np.ndarray) -> np.ndarray:
    """Project 20-dim run-length belief to 3 Target Cone mixing weights.

    Args:
        belief_array: [max_run_length] (e.g. 20)
    Returns:
        [3] mixing probabilities summing to 1
    """
    belief = np.asarray(belief_array)
    w_crisis = belief[0:5].sum()
    w_recovery = belief[5:15].sum()
    w_stable = belief[15:].sum()
    return np.array([w_crisis, w_recovery, w_stable])


class ContextAwareWeightSampler:
    """BAMOR Target Cone sampler.

    Mixes Target Cone boundaries based on Bayesian belief,
    then samples a physically valid preference weight w_restricted
    and computes the corresponding anchor w_anchor.

    Corresponds to BAMOR.lean Part XI & XII.
    """

    def __init__(self, mode_profiles=None, get_mixing_weights_fn=None):
        self.mode_profiles = mode_profiles or MODE_CONE_PROFILES
        self.get_mixing_weights = get_mixing_weights_fn or get_cone_mixing_weights

    def sample_and_get_anchor(self, belief_array: np.ndarray):
        """Sample w_restricted and w_anchor from the current Target Cone.

        Args:
            belief_array: [max_run_length] Bayesian belief vector

        Returns:
            w_restricted: [m] numpy array (sampled preference weight)
            w_anchor: [m] numpy array (anchor for regularization)
        """
        mixing_weights = self.get_mixing_weights(belief_array)

        cone_lower = 0.0
        cone_upper = 0.0
        anchor_0 = 0.0

        for w_mix, profile in zip(mixing_weights, self.mode_profiles):
            cone_lower += w_mix * profile['cone_lower']
            cone_upper += w_mix * profile['cone_upper']
            anchor_0 += w_mix * profile['w_anchor'][0]

        # Safety: prevent boundary inversion (Pitfall #10)
        cone_lower = min(cone_lower, cone_upper)

        # Sample w_0 (objective 0 weight) uniformly within cone
        w_0 = np.random.uniform(cone_lower, cone_upper)

        # Build full weight vector (m=2)
        w_restricted = np.array([w_0, 1.0 - w_0], dtype=np.float32)
        w_anchor = np.array([anchor_0, 1.0 - anchor_0], dtype=np.float32)

        return w_restricted, w_anchor
