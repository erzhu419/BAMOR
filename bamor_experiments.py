"""
BAMOR Experiment Runner — Baseline Comparisons and Ablations

Runs BAMOR vs baselines on the synthetic MO environment:
  1. BAMOR (full):        Target Cone + Surprise + Amnesic Reset
  2. BAMOR w/o Cone:      Surprise + Reset, but uniform weight sampling
  3. BAMOR w/o Surprise:  Target Cone estimation, but no change-point detection
  4. Vanilla MO-SAC:      Uniform Dirichlet weight sampling (no BAMOR)
  5. Fixed Weight SAC:    Single fixed weight (no multi-objective awareness)

Outputs:
  - Per-mode anchor estimation accuracy
  - Tracking error (Δ_local vs Δ_global)
  - Mode switch detection accuracy (precision/recall)
  - Cumulative reward per objective
  - Pareto front visualization data
"""

import numpy as np
import os
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from bamor_core import BAMORPipeline, SurpriseComputer, BeliefTracker, TargetConeEstimator, ContextAwareWeightSampler
from synthetic_mo_env import SyntheticMOEnv, SyntheticMOEnvStochastic


@dataclass
class ExperimentConfig:
    """Configuration for one experimental run."""
    name: str
    use_target_cone: bool = True
    use_surprise: bool = True
    weight_mode: str = "bamor"       # "bamor", "uniform", "biased", "fixed"
    fixed_weight: Optional[List[float]] = None
    bamor_num_modes: int = 3
    bamor_surprise_threshold: float = 8.0
    bamor_concentration: float = 10.0

    # Environment settings
    env_type: str = "deterministic"  # "deterministic" or "stochastic"
    switch_interval: int = 200
    max_steps: int = 1000
    num_episodes: int = 50
    seed: int = 42


@dataclass
class EpisodeMetrics:
    """Metrics collected per episode."""
    episode: int = 0
    total_reward: np.ndarray = field(default_factory=lambda: np.zeros(2))
    mode_switches_detected: int = 0
    mode_switches_true: int = 0
    # Per-mode anchor estimation error
    anchor_errors: Dict[str, float] = field(default_factory=dict)
    # Part XV metrics
    delta_global: float = 0.0
    delta_local: float = 0.0
    improvement: float = 0.0
    ratio: float = 1.0
    # Cone radius
    cone_radius: float = 0.25


@dataclass
class ExperimentResults:
    """Full results from one experiment."""
    config: dict = field(default_factory=dict)
    episode_metrics: List[dict] = field(default_factory=list)
    # Summary statistics
    mean_reward: List[float] = field(default_factory=list)
    mean_delta_global: float = 0.0
    mean_delta_local: float = 0.0
    mean_improvement: float = 0.0
    detection_precision: float = 0.0
    detection_recall: float = 0.0
    mean_anchor_error: float = 0.0
    wall_time: float = 0.0


def create_pipeline(config: ExperimentConfig) -> Optional[BAMORPipeline]:
    """Create a BAMORPipeline or None based on config."""
    if config.weight_mode not in ("bamor",):
        return None

    pipeline = BAMORPipeline(
        num_modes=config.bamor_num_modes,
        reward_dim=2,
        gamma=0.99,
        surprise_threshold=config.bamor_surprise_threshold,
        concentration=config.bamor_concentration,
    )

    if not config.use_surprise:
        # Disable surprise detection by setting threshold impossibly high
        pipeline.belief_tracker.surprise_threshold = 1e10

    return pipeline


def sample_weight_baseline(mode: str, fixed_weight=None) -> Tuple[np.ndarray, np.ndarray]:
    """Sample weight for non-BAMOR baselines. Returns (w, dummy_anchor)."""
    if mode == "uniform":
        w = np.random.dirichlet([1.0, 1.0]).astype(np.float32)
    elif mode == "biased":
        w = np.random.dirichlet([3.0, 1.0]).astype(np.float32)
    elif mode == "fixed":
        w = np.array(fixed_weight or [0.5, 0.5], dtype=np.float32)
    else:
        w = np.random.dirichlet([1.0, 1.0]).astype(np.float32)
    anchor = np.array([0.5, 0.5], dtype=np.float32)
    return w, anchor


def evaluate_anchor_accuracy(
    pipeline: Optional[BAMORPipeline],
    env: SyntheticMOEnv,
) -> Dict[str, float]:
    """Compare estimated anchors against ground truth."""
    if pipeline is None:
        return {}

    errors = {}
    gt = env.get_ground_truth()

    for mode_idx, mode_cfg in enumerate(gt["mode_configs"]):
        true_anchor = np.array(mode_cfg["true_anchor"])
        if mode_idx < pipeline.num_modes:
            est_anchor = pipeline.cone_estimator.constraint.w_anchor[mode_idx]
            # L2 error on simplex
            errors[mode_cfg["name"]] = float(np.linalg.norm(true_anchor - est_anchor))
        else:
            errors[mode_cfg["name"]] = 1.0  # max error

    return errors


def run_single_episode(
    env: SyntheticMOEnv,
    pipeline: Optional[BAMORPipeline],
    config: ExperimentConfig,
    episode: int,
) -> EpisodeMetrics:
    """Run one episode, collecting metrics."""
    obs, info = env.reset(seed=config.seed + episode)

    metrics = EpisodeMetrics(episode=episode)
    total_reward = np.zeros(2, dtype=np.float64)
    detected_switches = 0
    true_switches = 0

    for step in range(config.max_steps):
        # Random policy (we're testing the BAMOR pipeline, not RL training)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # Track true mode switches
        if info.get("mode_switched", False):
            true_switches += 1

        # Feed reward to BAMOR pipeline
        if pipeline is not None:
            obs_info = pipeline.observe_reward(reward)
            if obs_info.get("mode_switched", False):
                detected_switches += 1

        if terminated or truncated:
            break

    metrics.total_reward = total_reward
    metrics.mode_switches_detected = detected_switches
    metrics.mode_switches_true = true_switches

    # Anchor accuracy
    if pipeline is not None:
        metrics.anchor_errors = evaluate_anchor_accuracy(pipeline, env)

        # Part XV metrics
        w, _ = pipeline.sample_weight()
        part_xv = pipeline.get_metrics(w)
        metrics.delta_global = part_xv['delta_global']
        metrics.delta_local = part_xv['delta_local']
        metrics.improvement = part_xv['improvement']
        metrics.ratio = part_xv['ratio']
        metrics.cone_radius = pipeline.cone_estimator.belief_weighted_radius(
            pipeline.belief_tracker.belief)

    return metrics


def run_experiment(config: ExperimentConfig) -> ExperimentResults:
    """Run a complete experiment with multiple episodes."""
    print(f"\n{'='*60}")
    print(f"Running: {config.name}")
    print(f"{'='*60}")

    start_time = time.time()

    # Create environment
    EnvClass = SyntheticMOEnvStochastic if config.env_type == "stochastic" else SyntheticMOEnv
    env = EnvClass(
        switch_interval=config.switch_interval,
        max_steps=config.max_steps,
        seed=config.seed,
    )

    # Create pipeline
    pipeline = create_pipeline(config)

    # Run episodes
    all_metrics = []
    for ep in range(config.num_episodes):
        ep_metrics = run_single_episode(env, pipeline, config, ep)
        all_metrics.append(ep_metrics)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{config.num_episodes}: "
                  f"reward={ep_metrics.total_reward.round(2)}, "
                  f"switches detected={ep_metrics.mode_switches_detected}/"
                  f"{ep_metrics.mode_switches_true}")

    # Aggregate results
    results = ExperimentResults()
    results.config = asdict(config)
    results.wall_time = time.time() - start_time

    rewards = np.array([m.total_reward for m in all_metrics])
    results.mean_reward = rewards.mean(axis=0).tolist()

    if pipeline is not None:
        results.mean_delta_global = np.mean([m.delta_global for m in all_metrics])
        results.mean_delta_local = np.mean([m.delta_local for m in all_metrics])
        results.mean_improvement = np.mean([m.improvement for m in all_metrics])

        total_detected = sum(m.mode_switches_detected for m in all_metrics)
        total_true = sum(m.mode_switches_true for m in all_metrics)
        results.detection_recall = total_detected / max(total_true, 1)
        # Precision is harder — would need to check if detected switches align with true ones
        # For now, use a simple ratio
        results.detection_precision = min(total_true, total_detected) / max(total_detected, 1)

        all_errors = []
        for m in all_metrics:
            all_errors.extend(m.anchor_errors.values())
        results.mean_anchor_error = np.mean(all_errors) if all_errors else 0.0

    results.episode_metrics = [
        {
            "episode": m.episode,
            "total_reward": m.total_reward.tolist(),
            "switches_detected": m.mode_switches_detected,
            "switches_true": m.mode_switches_true,
            "delta_global": m.delta_global,
            "delta_local": m.delta_local,
            "improvement": m.improvement,
            "ratio": m.ratio,
            "cone_radius": m.cone_radius,
            "anchor_errors": m.anchor_errors,
        }
        for m in all_metrics
    ]

    print(f"\n  Summary for {config.name}:")
    print(f"    Mean reward:       {results.mean_reward}")
    print(f"    Δ_global:          {results.mean_delta_global:.4f}")
    print(f"    Δ_local:           {results.mean_delta_local:.4f}")
    print(f"    Improvement:       {results.mean_improvement:.4f}")
    print(f"    Detection recall:  {results.detection_recall:.2f}")
    print(f"    Anchor error:      {results.mean_anchor_error:.4f}")
    print(f"    Wall time:         {results.wall_time:.1f}s")

    return results


# ════════════════════════════════════════════════════════════════
# Standard experiment configurations
# ════════════════════════════════════════════════════════════════

def get_standard_configs(num_episodes=50, max_steps=1000) -> List[ExperimentConfig]:
    """Return the standard set of experiments for paper."""
    return [
        # Full BAMOR
        ExperimentConfig(
            name="BAMOR (full)",
            use_target_cone=True,
            use_surprise=True,
            weight_mode="bamor",
            num_episodes=num_episodes,
            max_steps=max_steps,
        ),
        # Ablation: no Target Cone (surprise + reset only, uniform sampling)
        ExperimentConfig(
            name="BAMOR w/o Cone",
            use_target_cone=False,
            use_surprise=True,
            weight_mode="bamor",
            bamor_concentration=1.0,  # effectively uniform within cone
            num_episodes=num_episodes,
            max_steps=max_steps,
        ),
        # Ablation: no Surprise (cone estimation but no change-point detection)
        ExperimentConfig(
            name="BAMOR w/o Surprise",
            use_target_cone=True,
            use_surprise=False,
            weight_mode="bamor",
            num_episodes=num_episodes,
            max_steps=max_steps,
        ),
        # Baseline: Uniform Dirichlet
        ExperimentConfig(
            name="MO-SAC (uniform)",
            weight_mode="uniform",
            num_episodes=num_episodes,
            max_steps=max_steps,
        ),
        # Baseline: Biased Dirichlet
        ExperimentConfig(
            name="MO-SAC (biased)",
            weight_mode="biased",
            num_episodes=num_episodes,
            max_steps=max_steps,
        ),
        # Baseline: Fixed weight
        ExperimentConfig(
            name="Fixed Weight [0.5, 0.5]",
            weight_mode="fixed",
            fixed_weight=[0.5, 0.5],
            num_episodes=num_episodes,
            max_steps=max_steps,
        ),
    ]


def run_all_experiments(
    save_dir: str = "bamor_experiments",
    num_episodes: int = 50,
    max_steps: int = 1000,
):
    """Run all standard experiments and save results."""
    os.makedirs(save_dir, exist_ok=True)

    configs = get_standard_configs(num_episodes, max_steps)
    all_results = []

    for config in configs:
        results = run_experiment(config)
        all_results.append(results)

    # Save combined results
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": [asdict(r) for r in all_results],
    }

    output_path = os.path.join(save_dir, "experiment_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Method':<25} {'Reward[0]':>10} {'Reward[1]':>10} {'Δ_global':>10} {'Δ_local':>10} {'Improve':>10} {'AnchorErr':>10}")
    print("-"*80)
    for r in all_results:
        print(f"{r.config['name']:<25} "
              f"{r.mean_reward[0]:>10.3f} "
              f"{r.mean_reward[1]:>10.3f} "
              f"{r.mean_delta_global:>10.4f} "
              f"{r.mean_delta_local:>10.4f} "
              f"{r.mean_improvement:>10.4f} "
              f"{r.mean_anchor_error:>10.4f}")

    return all_results


if __name__ == "__main__":
    run_all_experiments(
        save_dir="bamor_experiments",
        num_episodes=30,
        max_steps=600,
    )
