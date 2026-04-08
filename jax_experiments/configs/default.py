"""Hyperparameter configuration for JAX-based RL — BAMOR extended."""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # Environment
    env_name: str = "Hopper-v2"
    brax_backend: str = "spring"
    varying_params: List[str] = field(default_factory=lambda: ["gravity"])
    log_scale_limit: float = 3.0
    ood_change_range: float = 4.0
    task_num: int = 40
    test_task_num: int = 40
    changing_period: int = 20000
    changing_interval: int = 4000

    # Algorithm
    algo: str = "resac"  # resac | escp | bapr | bamor
    seed: int = 8
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    lr: float = 3e-4
    batch_size: int = 256
    replay_size: int = 1_000_000
    hidden_dim: int = 256
    max_iters: int = 2000
    samples_per_iter: int = 4000
    updates_per_iter: int = 250
    start_train_steps: int = 10_000
    max_episode_steps: int = 1000

    # Ensemble (RE-SAC / ESCP / BAPR / BAMOR)
    ensemble_size: int = 10
    beta: float = -2.0
    beta_ood: float = 0.01
    beta_bc: float = 0.001
    weight_reg: float = 0.01

    # Context / ESCP
    ep_dim: int = 2
    repr_loss_weight: float = 1.0
    rbf_radius: float = 2.0
    consistency_loss_weight: float = 50.0
    diversity_loss_weight: float = 0.025
    context_warmup_iters: int = 50
    rnn_fix_length: int = 16

    # BOCD (BAPR / BAMOR)
    max_run_length: int = 20
    hazard_rate: float = 0.05
    base_variance: float = 0.1
    variance_growth: float = 0.05
    surprise_ema_alpha: float = 0.3
    penalty_scale: float = 5.0
    belief_warmup_steps: int = 50

    # ─── BAMOR-specific ───
    num_objectives: int = 2                # m: number of objectives
    preference_weights: List[float] = field(default_factory=lambda: [0.5, 0.5])
    eta_anchor: float = 0.01              # Anchor-shifted regularization strength
    enable_target_cone: bool = True       # Enable Target Cone sampling

    # Logging
    save_root: str = "jax_experiments/results"
    run_name: str = ""
    log_interval: int = 5
    save_interval: int = 50
    eval_episodes: int = 5

    # Supported environments
    ENVS: List[str] = field(default_factory=lambda: [
        "HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Ant-v2"
    ])
