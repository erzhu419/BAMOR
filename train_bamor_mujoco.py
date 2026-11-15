#!/usr/bin/env python3
"""
BAMOR vs Baselines on Piecewise-Stationary Multi-Objective MuJoCo

Trains and evaluates:
  1. BAMOR      — full: surprise + Target Cone + anchor regularization
  2. BAMOR-NC   — ablation: no cone (uniform sampling, surprise still active)
  3. BAMOR-NS   — ablation: no surprise (cone active, no mode detection)
  4. Uniform    — MO-SAC with uniform Dirichlet weight sampling
  5. CAPQL      — from morl-baselines (22.5-degree angle sampler, no mode awareness)

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
import random
import types
from pathlib import Path
from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# --- Project imports ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from piecewise_mo_env import PiecewiseStationaryMOEnv, make_piecewise_mo_env
from bamor_core import BAMORPipeline, BeliefTracker, SurpriseComputer
from episode_credit_baseline import INITIAL_EPISODES
from episode_credit_blend import EpisodeCreditBlend
from episode_credit_trend import TREND_WINDOW
from train_synthetic_bamor import compute_hv_2d
from train_compare_baselines import (
    GenericTransitionContextTracker,
    LatentTransitionEncoder,
    LearnedTransitionContextTracker,
    RegimePrototypeBeliefTracker,
    SequenceTransitionContextTracker,
    SequenceTransitionEncoder,
    TransitionContextTracker,
    temporal_contrastive_loss,
    pseudo_regime_contrastive_loss,
)


LATENT_CONTEXT_DIM = 8
V3_SEQ_LEN = 16
V2_MUJOCO_METHODS = {
    "bamor_v2_contrast_hybrid",
    "bamor_v2_proto_hybrid",
    "bamor_v2_proto_generic",
}
V3_MUJOCO_METHODS = {
    "bamor_v3_contrast_norm_mix",
    "bamor_v3_contrast_lattice_mix",
    "bamor_v3_contrast_mix",
    "bamor_v3_contrast_stratified",
    "bamor_v3_seq_belief",
}
V4_MUJOCO_METHODS = {
    "bamor_v4_contrast_expert_mix",
    "bamor_v4_contrast_expert_norm_mix",
    "bamor_v4_contrast_soft_expert_mix",
}
V5_MUJOCO_METHODS = {
    "bamor_v5_contrast_expert_template_mix",
    "bamor_v5_contrast_moe_sharp_template_mix",
    "bamor_v5_contrast_moe_template_mix",
}
V6_MUJOCO_METHODS = {
    "bamor_v6_contrast_moe_head_template_mix",
}
V7_MUJOCO_METHODS = {
    "bamor_v7_contrast_critic_dual_template_mix",
}
V8_MUJOCO_METHODS = {
    "bamor_v8_contrast_dual_update_template_mix",
}
V11_MUJOCO_METHODS = {
    "bamor_v11_contrast_relabel_mix",
}
V12_MUJOCO_METHODS = {
    "bamor_v12_adaptive_operator_mix",
}
V13_MUJOCO_METHODS = {
    "bamor_v13_block_operator_mix",
}
V14_MUJOCO_METHODS = {
    "bamor_v14_adaptive_envelope_mix",
}
V15_MUJOCO_METHODS = {
    "bamor_v15_adaptive_policy_envelope_mix",
}
V16_MUJOCO_METHODS = {
    "bamor_v16_risk_adaptive_mix",
}
V17_MUJOCO_METHODS = {
    "bamor_v17_archive_risk_mix",
}
V18_MUJOCO_METHODS = {
    "bamor_v18_coverage_archive_mix",
}
V19_MUJOCO_METHODS = {
    "bamor_v19_hybrid_archive_mix",
}
V20_MUJOCO_METHODS = {
    "bamor_v20_hier_archive_mix",
}
V21_MUJOCO_METHODS = {
    "bamor_v21_covbias_hier_archive_mix",
}
V22_MUJOCO_METHODS = {
    "bamor_v22_dense_hier_archive_mix",
}
V23_MUJOCO_METHODS = {
    "bamor_v23_extreme_hier_archive_mix",
    "bamor_v23_fixed_selector_archive_mix",
}
V24_MUJOCO_METHODS = {
    "bamor_v24_target_consistent_archive_mix",
    "bamor_v24_target_consistent_fixed_archive_mix",
    "bamor_v24_ownpref_actor_fixed_archive_mix",
    "bamor_v24_ownpref_critic_fixed_archive_mix",
    "bamor_v24_ownpref_both_fixed_archive_mix",
}
V25_MUJOCO_METHODS = {
    "bamor_v25_progress_archive_mix",
    "bamor_v25_progress_fixed_archive_mix",
    "bamor_v25_ownpref_actor_progress_fixed_archive_mix",
    "bamor_v25_ownpref_actor_progress_archive_mix",
    "bamor_v25_ownpref_both_progress_fixed_archive_mix",
    "bamor_v25_ownpref_both_progress_archive_mix",
    "bamor_v25_pairwise_progress_archive_mix",
    "bamor_v25_pairwise_progress_fixed_archive_mix",
}
V26_MUJOCO_METHODS = {"bamor_v26_aligned_actor_fixed_archive_mix"}
V27_MUJOCO_METHODS = {
    "bamor_v27_aligned_target_fixed_archive_mix",
    "bamor_v27_aligned_target_ownpref_actor_fixed_archive_mix",
    "bamor_v27_aligned_target_progress_archive_mix",
    "bamor_v27_aligned_target_progress_fixed_archive_mix",
}
V28_MUJOCO_METHODS = {"bamor_v28_mean_value_fixed_archive_mix"}
FORECAST_EPISODIC_METHODS = {
    "bamor_common_target_episode_forecast_fixed": ("ema", False),
    "bamor_common_target_episode_forecast_ema": ("ema", True),
    "bamor_common_target_episode_forecast_equal_mix": ("equal_mixture", True),
    "bamor_common_target_episode_forecast_past_mix": ("blend", True),
}
EPISODIC_ALLOCATION_METHODS = set(FORECAST_EPISODIC_METHODS) | {
    "bamor_common_target_episode_adaptive", "bamor_common_target_episode_fixed"}
COMMON_TARGET_METHODS = EPISODIC_ALLOCATION_METHODS | {
    "bamor_common_target_adaptive", "bamor_common_target_fixed",
    "bamor_common_target_projected_fixed", "bamor_common_target_projected_adaptive"}
EXTREME_PORTFOLIO_METHODS = (
    V23_MUJOCO_METHODS | V24_MUJOCO_METHODS | V25_MUJOCO_METHODS
    | V26_MUJOCO_METHODS | V27_MUJOCO_METHODS | V28_MUJOCO_METHODS
)
SEQUENCE_CONTEXT_METHODS = {
    "bamor_v3_seq_belief",
}
SINGLE_LATENT_METHODS = COMMON_TARGET_METHODS | V2_MUJOCO_METHODS | {
    "bamor_v3_contrast_norm_mix",
    "bamor_v3_contrast_lattice_mix",
    "bamor_v3_contrast_mix",
    "bamor_v3_contrast_stratified",
} | V4_MUJOCO_METHODS | V5_MUJOCO_METHODS | V6_MUJOCO_METHODS | V7_MUJOCO_METHODS | V8_MUJOCO_METHODS | V11_MUJOCO_METHODS | V12_MUJOCO_METHODS | V13_MUJOCO_METHODS | V14_MUJOCO_METHODS | V15_MUJOCO_METHODS | V16_MUJOCO_METHODS | V17_MUJOCO_METHODS | V18_MUJOCO_METHODS | V19_MUJOCO_METHODS | V20_MUJOCO_METHODS | V21_MUJOCO_METHODS | V22_MUJOCO_METHODS | EXTREME_PORTFOLIO_METHODS
CONTEXT_MUJOCO_METHODS = (
    COMMON_TARGET_METHODS | V2_MUJOCO_METHODS | V3_MUJOCO_METHODS
    | V4_MUJOCO_METHODS | V5_MUJOCO_METHODS | V6_MUJOCO_METHODS | V7_MUJOCO_METHODS | V8_MUJOCO_METHODS
    | V11_MUJOCO_METHODS | V12_MUJOCO_METHODS | V13_MUJOCO_METHODS | V14_MUJOCO_METHODS | V15_MUJOCO_METHODS | V16_MUJOCO_METHODS | V17_MUJOCO_METHODS | V18_MUJOCO_METHODS | V19_MUJOCO_METHODS | V20_MUJOCO_METHODS | V21_MUJOCO_METHODS | V22_MUJOCO_METHODS | EXTREME_PORTFOLIO_METHODS
)
V2_DETECTOR_METHODS = {
    "bamor_v2_proto_hybrid",
    "bamor_v2_proto_generic",
}
V3_DETECTOR_METHODS = {
    "bamor_v3_seq_belief",
}
DETECTOR_CONTEXT_METHODS = V2_DETECTOR_METHODS | V3_DETECTOR_METHODS
OLD_BAMOR_METHODS = {"bamor", "bamor_nc", "bamor_ns"}


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


class PairwiseResidualPolicyNet(PolicyNet):
    """Add mixed-preference mean-action capacity without a vertex residual."""

    def __init__(self, state_dim, action_dim, context_dim, reward_dim, hidden=256):
        super().__init__(state_dim, action_dim, context_dim, hidden)
        self.reward_dim = int(reward_dim)
        self.pairs = tuple((i, j) for i in range(reward_dim)
                           for j in range(i + 1, reward_dim))
        self.pair_mean_weight = nn.Parameter(
            torch.zeros(len(self.pairs), action_dim, hidden))
        self.pair_mean_bias = nn.Parameter(
            torch.zeros(len(self.pairs), action_dim))

    def forward(self, state, context):
        h = self.trunk(torch.cat([state, context], dim=-1))
        pref = context[:, :self.reward_dim].clamp_min(0.0)
        pref = pref / pref.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        gates = torch.stack([2.0 * pref[:, i] * pref[:, j]
                             for i, j in self.pairs], dim=-1)
        residuals = torch.einsum("bh,pah->bpa", h, self.pair_mean_weight)
        residuals = residuals + self.pair_mean_bias.unsqueeze(0)
        mean = self.mean_head(h) + (gates.unsqueeze(-1) * residuals).sum(dim=1)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std


def preference_expert_templates(reward_dim):
    """Preference-simplex templates used by hard/soft expert actors."""
    reward_dim = int(reward_dim)
    if reward_dim <= 5:
        templates = []
        for bitmask in range(1, 1 << reward_dim):
            mask = np.array(
                [(bitmask >> i) & 1 for i in range(reward_dim)],
                dtype=np.float32,
            )
            templates.append(mask / mask.sum())
        return np.asarray(templates, dtype=np.float32)

    corners = np.eye(reward_dim, dtype=np.float32)
    center = np.full((1, reward_dim), 1.0 / reward_dim, dtype=np.float32)
    return np.concatenate([corners, center], axis=0)


class PreferenceExpertPolicyNet(nn.Module):
    """Hard-routed actor heads for different preference-simplex regions."""

    def __init__(self, state_dim, action_dim, context_dim, reward_dim,
                 hidden=256, expert_threshold=0.2):
        super().__init__()
        self.action_dim = action_dim
        self.reward_dim = int(reward_dim)
        self.expert_threshold = float(expert_threshold)
        if self.reward_dim <= 5:
            self.num_experts = (1 << self.reward_dim) - 1
        else:
            self.num_experts = self.reward_dim + 1
        self.experts = nn.ModuleList([
            PolicyNet(state_dim, action_dim, context_dim, hidden)
            for _ in range(self.num_experts)
        ])

    def expert_indices(self, context):
        weights = context[:, :self.reward_dim].clamp_min(0.0)
        max_idx = weights.argmax(dim=-1)
        if self.reward_dim <= 5:
            active = weights >= self.expert_threshold
            active.scatter_(1, max_idx.unsqueeze(1), True)
            powers = (2 ** torch.arange(
                self.reward_dim, device=context.device, dtype=torch.long)).view(1, -1)
            bitmask = (active.long() * powers).sum(dim=-1)
            return bitmask - 1

        center_idx = torch.full_like(max_idx, self.reward_dim)
        return torch.where(weights.max(dim=-1).values >= 0.55, max_idx, center_idx)

    def get_action(self, state, context, deterministic=False):
        expert_idx = self.expert_indices(context)
        actions = state.new_zeros((state.shape[0], self.action_dim))
        log_probs = state.new_zeros((state.shape[0], 1))

        for idx_t in expert_idx.unique(sorted=True):
            idx = int(idx_t.item())
            mask = expert_idx == idx_t
            if deterministic:
                actions[mask] = self.experts[idx].get_action(
                    state[mask], context[mask], deterministic=True)
            else:
                a_i, lp_i = self.experts[idx].get_action(
                    state[mask], context[mask], deterministic=False)
                actions[mask] = a_i
                log_probs[mask] = lp_i
        if deterministic:
            return actions
        return actions, log_probs


class PreferenceSoftExpertPolicyNet(nn.Module):
    """Soft-gated expert actor heads over preference-simplex templates."""

    def __init__(self, state_dim, action_dim, context_dim, reward_dim,
                 hidden=256, gate_temperature=0.04):
        super().__init__()
        self.action_dim = action_dim
        self.reward_dim = int(reward_dim)
        self.gate_temperature = float(gate_temperature)
        templates = preference_expert_templates(self.reward_dim)
        self.register_buffer("templates", torch.from_numpy(templates))
        self.experts = nn.ModuleList([
            PolicyNet(state_dim, action_dim, context_dim, hidden)
            for _ in range(len(templates))
        ])

    def gate_weights(self, context):
        pref = context[:, :self.reward_dim].clamp_min(0.0)
        pref = pref / pref.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        d2 = (pref.unsqueeze(1) - self.templates.unsqueeze(0)).pow(2).sum(dim=-1)
        return torch.softmax(-d2 / max(self.gate_temperature, 1e-6), dim=-1)

    def forward(self, state, context):
        means = []
        log_stds = []
        for expert in self.experts:
            mean_i, log_std_i = expert(state, context)
            means.append(mean_i)
            log_stds.append(log_std_i)
        means = torch.stack(means, dim=1)
        log_stds = torch.stack(log_stds, dim=1)
        gates = self.gate_weights(context).unsqueeze(-1)
        mean = (gates * means).sum(dim=1)
        log_std = (gates * log_stds).sum(dim=1).clamp(
            PolicyNet.LOG_STD_MIN, PolicyNet.LOG_STD_MAX)
        return mean, log_std

    def get_action(self, state, context, deterministic=False):
        mean, log_std = self.forward(state, context)
        if deterministic:
            return torch.tanh(mean)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class PreferenceMixtureExpertPolicyNet(nn.Module):
    """Mixture-of-experts actor with exact tanh-Gaussian mixture log-prob."""

    def __init__(self, state_dim, action_dim, context_dim, reward_dim,
                 hidden=256, gate_temperature=0.08):
        super().__init__()
        self.action_dim = action_dim
        self.reward_dim = int(reward_dim)
        self.gate_temperature = float(gate_temperature)
        templates = preference_expert_templates(self.reward_dim)
        self.register_buffer("templates", torch.from_numpy(templates))
        self.experts = nn.ModuleList([
            PolicyNet(state_dim, action_dim, context_dim, hidden)
            for _ in range(len(templates))
        ])

    def gate_weights(self, context):
        pref = context[:, :self.reward_dim].clamp_min(0.0)
        pref = pref / pref.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        d2 = (pref.unsqueeze(1) - self.templates.unsqueeze(0)).pow(2).sum(dim=-1)
        return torch.softmax(-d2 / max(self.gate_temperature, 1e-6), dim=-1)

    def expert_outputs(self, state, context):
        means = []
        log_stds = []
        for expert in self.experts:
            mean_i, log_std_i = expert(state, context)
            means.append(mean_i)
            log_stds.append(log_std_i)
        return torch.stack(means, dim=1), torch.stack(log_stds, dim=1)

    def _mixture_log_prob(self, means, log_stds, gates, pre_tanh, action):
        stds = log_stds.exp()
        dist = Normal(means, stds)
        x = pre_tanh.unsqueeze(1)
        log_det = torch.log(1 - action.pow(2) + 1e-6).unsqueeze(1)
        expert_log_probs = (dist.log_prob(x) - log_det).sum(dim=-1)
        log_gate = torch.log(gates.clamp_min(1e-8))
        return torch.logsumexp(log_gate + expert_log_probs, dim=-1, keepdim=True)

    def get_action(self, state, context, deterministic=False):
        means, log_stds = self.expert_outputs(state, context)
        gates = self.gate_weights(context)
        batch_idx = torch.arange(state.shape[0], device=state.device)

        if deterministic:
            expert_idx = gates.argmax(dim=-1)
            mean = means[batch_idx, expert_idx]
            return torch.tanh(mean)

        expert_idx = torch.multinomial(gates.detach(), num_samples=1).squeeze(-1)
        mean = means[batch_idx, expert_idx]
        log_std = log_stds[batch_idx, expert_idx]
        dist = Normal(mean, log_std.exp())
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = self._mixture_log_prob(means, log_stds, gates, pre_tanh, action)
        return action, log_prob


class PreferenceHeadMixturePolicyNet(nn.Module):
    """MoE actor with shared state/context trunk and expert output heads."""

    def __init__(self, state_dim, action_dim, context_dim, reward_dim,
                 hidden=256, gate_temperature=0.08):
        super().__init__()
        self.action_dim = action_dim
        self.reward_dim = int(reward_dim)
        self.gate_temperature = float(gate_temperature)
        templates = preference_expert_templates(self.reward_dim)
        self.register_buffer("templates", torch.from_numpy(templates))
        in_dim = state_dim + context_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_heads = nn.ModuleList([
            nn.Linear(hidden, action_dim) for _ in range(len(templates))
        ])
        self.log_std_heads = nn.ModuleList([
            nn.Linear(hidden, action_dim) for _ in range(len(templates))
        ])

    def gate_weights(self, context):
        pref = context[:, :self.reward_dim].clamp_min(0.0)
        pref = pref / pref.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        d2 = (pref.unsqueeze(1) - self.templates.unsqueeze(0)).pow(2).sum(dim=-1)
        return torch.softmax(-d2 / max(self.gate_temperature, 1e-6), dim=-1)

    def expert_outputs(self, state, context):
        h = self.trunk(torch.cat([state, context], dim=-1))
        means = torch.stack([head(h) for head in self.mean_heads], dim=1)
        log_stds = torch.stack([
            head(h).clamp(PolicyNet.LOG_STD_MIN, PolicyNet.LOG_STD_MAX)
            for head in self.log_std_heads
        ], dim=1)
        return means, log_stds

    def _mixture_log_prob(self, means, log_stds, gates, pre_tanh, action):
        stds = log_stds.exp()
        dist = Normal(means, stds)
        x = pre_tanh.unsqueeze(1)
        log_det = torch.log(1 - action.pow(2) + 1e-6).unsqueeze(1)
        expert_log_probs = (dist.log_prob(x) - log_det).sum(dim=-1)
        log_gate = torch.log(gates.clamp_min(1e-8))
        return torch.logsumexp(log_gate + expert_log_probs, dim=-1, keepdim=True)

    def get_action(self, state, context, deterministic=False):
        means, log_stds = self.expert_outputs(state, context)
        gates = self.gate_weights(context)
        batch_idx = torch.arange(state.shape[0], device=state.device)

        if deterministic:
            expert_idx = gates.argmax(dim=-1)
            mean = means[batch_idx, expert_idx]
            return torch.tanh(mean)

        expert_idx = torch.multinomial(gates.detach(), num_samples=1).squeeze(-1)
        mean = means[batch_idx, expert_idx]
        log_std = log_stds[batch_idx, expert_idx]
        dist = Normal(mean, log_std.exp())
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = self._mixture_log_prob(means, log_stds, gates, pre_tanh, action)
        return action, log_prob


class PreferenceDualBranchPolicyNet(nn.Module):
    """Fixed-mixture dual actor: independent experts + shared-head experts.

    The stochastic policy trains both branches from the same replay stream.
    Deterministic branch selection is handled by the trainer using the critic,
    which keeps this policy independent of Q-network ownership.
    """

    def __init__(self, state_dim, action_dim, context_dim, reward_dim,
                 hidden=256, gate_temperature=0.08, independent_prior=0.5):
        super().__init__()
        self.action_dim = action_dim
        self.reward_dim = int(reward_dim)
        self.independent = PreferenceMixtureExpertPolicyNet(
            state_dim, action_dim, context_dim, reward_dim,
            hidden=hidden, gate_temperature=gate_temperature)
        self.shared_head = PreferenceHeadMixturePolicyNet(
            state_dim, action_dim, context_dim, reward_dim,
            hidden=hidden, gate_temperature=gate_temperature)
        p = float(np.clip(independent_prior, 1e-3, 1.0 - 1e-3))
        self.register_buffer(
            "branch_prior", torch.tensor([p, 1.0 - p], dtype=torch.float32))

    @staticmethod
    def _mixture_log_prob_from_outputs(means, log_stds, gates, pre_tanh, action):
        stds = log_stds.exp()
        dist = Normal(means, stds)
        x = pre_tanh.unsqueeze(1)
        log_det = torch.log(1 - action.pow(2) + 1e-6).unsqueeze(1)
        expert_log_probs = (dist.log_prob(x) - log_det).sum(dim=-1)
        log_gate = torch.log(gates.clamp_min(1e-8))
        return torch.logsumexp(log_gate + expert_log_probs, dim=-1, keepdim=True)

    def get_action(self, state, context, deterministic=False):
        ind_means, ind_log_stds = self.independent.expert_outputs(state, context)
        ind_gates = self.independent.gate_weights(context)
        head_means, head_log_stds = self.shared_head.expert_outputs(state, context)
        head_gates = self.shared_head.gate_weights(context)
        batch_idx = torch.arange(state.shape[0], device=state.device)

        if deterministic:
            ind_idx = ind_gates.argmax(dim=-1)
            return torch.tanh(ind_means[batch_idx, ind_idx])

        branch_prob = self.branch_prior.to(state.device).unsqueeze(0).expand(
            state.shape[0], -1)
        branch_idx = torch.multinomial(branch_prob.detach(), num_samples=1).squeeze(-1)

        ind_idx = torch.multinomial(ind_gates.detach(), num_samples=1).squeeze(-1)
        head_idx = torch.multinomial(head_gates.detach(), num_samples=1).squeeze(-1)
        ind_mean = ind_means[batch_idx, ind_idx]
        ind_log_std = ind_log_stds[batch_idx, ind_idx]
        head_mean = head_means[batch_idx, head_idx]
        head_log_std = head_log_stds[batch_idx, head_idx]

        use_ind = (branch_idx == 0).unsqueeze(-1)
        mean = torch.where(use_ind, ind_mean, head_mean)
        log_std = torch.where(use_ind, ind_log_std, head_log_std)
        dist = Normal(mean, log_std.exp())
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)

        ind_log_prob = self._mixture_log_prob_from_outputs(
            ind_means, ind_log_stds, ind_gates, pre_tanh, action)
        head_log_prob = self._mixture_log_prob_from_outputs(
            head_means, head_log_stds, head_gates, pre_tanh, action)
        log_prior = torch.log(self.branch_prior.to(state.device).clamp_min(1e-8))
        branch_log_probs = torch.cat(
            [log_prior[0] + ind_log_prob, log_prior[1] + head_log_prob], dim=-1)
        log_prob = torch.logsumexp(branch_log_probs, dim=-1, keepdim=True)
        return action, log_prob

    def deterministic_branch_actions(self, state, context):
        ind_action = self.independent.get_action(state, context, deterministic=True)
        head_action = self.shared_head.get_action(state, context, deterministic=True)
        return torch.stack([ind_action, head_action], dim=1)

    def training_branches(self):
        return (self.independent, self.shared_head)


# ================================================================
# Replay Buffer
# ================================================================

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, weight_dim,
                 context_dim, reward_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.weights = np.zeros((capacity, weight_dim), dtype=np.float32)
        self.contexts = np.zeros((capacity, context_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, reward_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_contexts = np.zeros((capacity, context_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.timesteps = np.zeros((capacity, 1), dtype=np.int64)
        self.regimes = np.full((capacity, 1), -1, dtype=np.int64)

    def push(self, s, a, w, c, r, s_next, c_next, done, t=0, regime=-1):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.weights[self.ptr] = w
        self.contexts[self.ptr] = c
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s_next
        self.next_contexts[self.ptr] = c_next
        self.dones[self.ptr] = float(done)
        self.timesteps[self.ptr] = int(t)
        self.regimes[self.ptr] = int(regime)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idx]).to(self.device),
            torch.FloatTensor(self.actions[idx]).to(self.device),
            torch.FloatTensor(self.weights[idx]).to(self.device),
            torch.FloatTensor(self.contexts[idx]).to(self.device),
            torch.FloatTensor(self.rewards[idx]).to(self.device),
            torch.FloatTensor(self.next_states[idx]).to(self.device),
            torch.FloatTensor(self.next_contexts[idx]).to(self.device),
            torch.FloatTensor(self.dones[idx]).to(self.device),
        )

    def sample_encoder(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idx]).to(self.device),
            torch.FloatTensor(self.actions[idx]).to(self.device),
            torch.FloatTensor(self.rewards[idx]).to(self.device),
            torch.FloatTensor(self.next_states[idx]).to(self.device),
            torch.LongTensor(self.timesteps[idx].reshape(-1)).to(self.device),
            torch.LongTensor(self.regimes[idx].reshape(-1)).to(self.device),
        )

    def sample_encoder_sequence(self, batch_size, seq_len):
        if self.size < seq_len:
            return None

        starts = []
        max_start = self.size - seq_len
        attempts = 0
        while len(starts) < batch_size and attempts < batch_size * 20:
            start = np.random.randint(0, max_start + 1)
            # Avoid sequences that cross episode boundaries.  The final
            # transition may be terminal; earlier terminal transitions would
            # splice two unrelated episodes into one recurrent context.
            if self.dones[start:start + seq_len - 1].any():
                attempts += 1
                continue
            starts.append(start)
            attempts += 1
        if not starts:
            return None
        if len(starts) < batch_size:
            starts.extend(np.random.choice(starts, size=batch_size - len(starts)))

        idx = np.stack([np.arange(s, s + seq_len) for s in starts])
        end = idx[:, -1]
        return (
            torch.FloatTensor(self.states[idx]).to(self.device),
            torch.FloatTensor(self.actions[idx]).to(self.device),
            torch.FloatTensor(self.rewards[idx]).to(self.device),
            torch.FloatTensor(self.next_states[idx]).to(self.device),
            torch.LongTensor(self.timesteps[end].reshape(-1)).to(self.device),
            torch.LongTensor(self.regimes[end].reshape(-1)).to(self.device),
        )


# ================================================================
# Trainer
# ================================================================

class MOSACTrainer:
    """Weight-conditioned multi-objective SAC trainer."""

    critic_target_rule = "legacy_scalar_soft_vector_reward"
    bootstrap_terminal_rule = "terminated_or_truncated"
    critic_candidate_context_rule = "candidate_preference"

    @staticmethod
    def replay_terminal(terminated, truncated):
        return terminated or truncated

    def critic_targets(self, reward, weights, done, next_scalar, next_vector):
        target_scalar = (weights * reward).sum(dim=-1, keepdim=True) + \
                        self.gamma * (1 - done) * next_scalar
        target_vector = reward + self.gamma * (1 - done) * next_vector
        return target_scalar, target_vector

    def __init__(self, state_dim, action_dim, reward_dim, hidden=256,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 buffer_capacity=500_000, batch_size=256, device="auto",
                 context_dim=None, reward_normalization=False,
                 expert_policy=False, expert_threshold=0.2,
                 soft_expert_policy=False, soft_expert_temperature=0.04,
                 moe_expert_policy=False, moe_expert_temperature=0.08,
                 moe_head_policy=False, pairwise_residual_policy=False,
                 dual_branch_policy=False,
                 dual_branch_update=False, critic_select_policy=False,
                 preference_relabeling=False, preference_relabeling_rate=1.0,
                 preference_relabeling_strategy="mixed",
                 critic_relabel_copies=1, envelope_target_candidates=1,
                 vector_critic_coef=0.0, envelope_policy_candidates=1,
                 behavior_clone_coef=0.0):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.reward_dim = reward_dim
        self.context_dim = context_dim or reward_dim
        self.reward_normalization = reward_normalization
        self.expert_policy = expert_policy
        self.soft_expert_policy = soft_expert_policy
        self.moe_expert_policy = moe_expert_policy
        self.moe_head_policy = moe_head_policy
        self.pairwise_residual_policy = pairwise_residual_policy
        self.dual_branch_policy = dual_branch_policy
        self.dual_branch_update = dual_branch_update
        self.critic_select_policy = critic_select_policy
        self.preference_relabeling = preference_relabeling
        self.preference_relabeling_rate = float(preference_relabeling_rate)
        self.preference_relabeling_strategy = str(preference_relabeling_strategy)
        self.critic_relabel_copies = int(critic_relabel_copies)
        self.envelope_target_candidates = int(envelope_target_candidates)
        self.vector_critic_coef = float(vector_critic_coef)
        self.envelope_policy_candidates = int(envelope_policy_candidates)
        self.behavior_clone_coef = float(behavior_clone_coef)
        self.reward_norm_initialized = False
        self.reward_norm_beta = 0.01
        self.reward_norm_eps = 1e-4
        self.reward_mean = torch.zeros(reward_dim, device=self.device)
        self.reward_var = torch.ones(reward_dim, device=self.device)

        # Networks
        self.q1 = SoftQNet(state_dim, action_dim, self.context_dim, reward_dim, hidden).to(self.device)
        self.q2 = SoftQNet(state_dim, action_dim, self.context_dim, reward_dim, hidden).to(self.device)
        self.q1_target = SoftQNet(state_dim, action_dim, self.context_dim, reward_dim, hidden).to(self.device)
        self.q2_target = SoftQNet(state_dim, action_dim, self.context_dim, reward_dim, hidden).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        if dual_branch_policy:
            self.policy = PreferenceDualBranchPolicyNet(
                state_dim, action_dim, self.context_dim, reward_dim,
                hidden=hidden, gate_temperature=moe_expert_temperature).to(self.device)
        elif moe_head_policy:
            self.policy = PreferenceHeadMixturePolicyNet(
                state_dim, action_dim, self.context_dim, reward_dim,
                hidden=hidden, gate_temperature=moe_expert_temperature).to(self.device)
        elif moe_expert_policy:
            self.policy = PreferenceMixtureExpertPolicyNet(
                state_dim, action_dim, self.context_dim, reward_dim,
                hidden=hidden, gate_temperature=moe_expert_temperature).to(self.device)
        elif soft_expert_policy:
            self.policy = PreferenceSoftExpertPolicyNet(
                state_dim, action_dim, self.context_dim, reward_dim,
                hidden=hidden, gate_temperature=soft_expert_temperature).to(self.device)
        elif expert_policy:
            self.policy = PreferenceExpertPolicyNet(
                state_dim, action_dim, self.context_dim, reward_dim,
                hidden=hidden, expert_threshold=expert_threshold).to(self.device)
        elif pairwise_residual_policy:
            self.policy = PairwiseResidualPolicyNet(
                state_dim, action_dim, self.context_dim, reward_dim,
                hidden=hidden).to(self.device)
        else:
            self.policy = PolicyNet(
                state_dim, action_dim, self.context_dim, hidden).to(self.device)

        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.update_work = dict(critic_optimizer_steps=0, actor_optimizer_steps=0,
                                critic_rows=0, target_candidate_rows=0, actor_candidate_rows=0)

        # Buffer
        self.buffer = ReplayBuffer(
            buffer_capacity, state_dim, action_dim, reward_dim,
            self.context_dim, reward_dim, self.device
        )

    def sample_preference_batch(self, n: int, strategy: str = "mixed"):
        """Sample training preferences on the simplex directly on the trainer device."""
        n = int(n)
        if n <= 0:
            return torch.empty((0, self.reward_dim), device=self.device)

        strategy = str(strategy or "mixed")
        alpha = torch.ones(self.reward_dim, device=self.device)
        if strategy == "uniform":
            return torch.distributions.Dirichlet(alpha).sample((n,))

        if strategy == "stratified":
            if self.reward_dim == 2:
                idx = torch.randint(0, 11, (n,), device=self.device)
                w0 = idx.float() / 10.0
                return torch.stack([w0, 1.0 - w0], dim=-1)
            idx = torch.randint(0, self.reward_dim + 1, (n,), device=self.device)
            prefs = torch.zeros(n, self.reward_dim, device=self.device)
            corner = idx < self.reward_dim
            if corner.any():
                prefs[corner, idx[corner]] = 1.0
            if (~corner).any():
                prefs[~corner] = 1.0 / float(self.reward_dim)
            return prefs

        if strategy == "edge":
            idx = torch.randint(0, self.reward_dim + 1, (n,), device=self.device)
            prefs = torch.zeros(n, self.reward_dim, device=self.device)
            corner = idx < self.reward_dim
            if corner.any():
                prefs[corner, idx[corner]] = 1.0
            if (~corner).any():
                prefs[~corner] = 1.0 / float(self.reward_dim)
            return prefs

        sampled = torch.distributions.Dirichlet(alpha).sample((n,))
        mask = torch.rand(n, device=self.device) < 0.5
        n_mask = int(mask.sum().item())
        if n_mask:
            sampled[mask] = self.sample_preference_batch(n_mask, "stratified")
        return sampled

    def replace_context_preferences(self, context, prefs):
        out = context.clone()
        out[:, :self.reward_dim] = prefs
        return out

    def relabel_preferences(self, w, c, c_next):
        """Reuse vector-reward transitions under fresh scalarisation weights."""
        if not self.preference_relabeling:
            return w, c, c_next
        rate = float(np.clip(self.preference_relabeling_rate, 0.0, 1.0))
        if rate <= 0.0:
            return w, c, c_next

        batch = w.shape[0]
        relabel_mask = torch.rand(batch, device=self.device) < rate
        n_relabel = int(relabel_mask.sum().item())
        if n_relabel == 0:
            return w, c, c_next

        sampled = self.sample_preference_batch(
            n_relabel, self.preference_relabeling_strategy)

        new_w = w.clone()
        new_w[relabel_mask] = sampled

        c = self.replace_context_preferences(c, new_w)
        c_next = self.replace_context_preferences(c_next, new_w)
        return new_w, c, c_next

    def select_action(self, state, context, deterministic=False):
        if (deterministic and self.critic_select_policy
                and hasattr(self.policy, "deterministic_branch_actions")):
            branch_actions = self.policy.deterministic_branch_actions(state, context)
            batch_size, num_branches, action_dim = branch_actions.shape
            s_rep = state.unsqueeze(1).expand(-1, num_branches, -1).reshape(
                batch_size * num_branches, -1)
            c_rep = context.unsqueeze(1).expand(-1, num_branches, -1).reshape(
                batch_size * num_branches, -1)
            a_rep = branch_actions.reshape(batch_size * num_branches, action_dim)
            q1 = self.q1(s_rep, a_rep, c_rep).view(
                batch_size, num_branches, self.reward_dim)
            q2 = self.q2(s_rep, a_rep, c_rep).view(
                batch_size, num_branches, self.reward_dim)
            q = torch.min(q1, q2)
            pref = context[:, :self.reward_dim].clamp_min(0.0)
            pref = pref / pref.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            q_scalar = (q * pref.unsqueeze(1)).sum(dim=-1)
            branch_idx = q_scalar.argmax(dim=-1)
            batch_idx = torch.arange(batch_size, device=state.device)
            return branch_actions[batch_idx, branch_idx]
        return self.policy.get_action(state, context, deterministic=deterministic)

    def normalize_reward_batch(self, reward):
        """Normalize objectives for critic learning while keeping eval raw."""
        if not self.reward_normalization:
            return reward
        with torch.no_grad():
            batch_mean = reward.mean(dim=0)
            batch_var = reward.var(dim=0, unbiased=False).clamp_min(self.reward_norm_eps)
            if not self.reward_norm_initialized:
                self.reward_mean.copy_(batch_mean)
                self.reward_var.copy_(batch_var)
                self.reward_norm_initialized = True
            else:
                beta = self.reward_norm_beta
                self.reward_mean.lerp_(batch_mean, beta)
                self.reward_var.lerp_(batch_var, beta)
        return (reward - self.reward_mean) / torch.sqrt(self.reward_var + self.reward_norm_eps)

    def actor_objective(self, s, w, c, q_scalar, log_prob):
        return (self.alpha * log_prob - q_scalar).mean()

    def critic_objective(self, q1_vec, q2_vec, w, target_scalar, target_vec):
        q1_val = (w * q1_vec).sum(dim=-1, keepdim=True)
        q2_val = (w * q2_vec).sum(dim=-1, keepdim=True)
        loss = F.mse_loss(q1_val, target_scalar) + F.mse_loss(q2_val, target_scalar)
        vector_coef = max(0.0, float(self.vector_critic_coef))
        if vector_coef > 0.0:
            loss = loss + vector_coef * (
                F.mse_loss(q1_vec, target_vec) + F.mse_loss(q2_vec, target_vec))
        return loss

    def behavior_clone_loss(self, action, replay_action):
        return F.mse_loss(action, replay_action.detach())

    def envelope_actor_loss(self, s, w, c):
        """Legacy candidate-policy gradient, retained for the existing methods."""
        candidate_contexts = [c]
        for _ in range(max(1, int(self.envelope_policy_candidates)) - 1):
            cand_w = self.sample_preference_batch(w.shape[0], self.preference_relabeling_strategy)
            candidate_contexts.append(self.replace_context_preferences(c, cand_w))

        action_candidates = []
        log_prob_candidates = []
        scalar_candidates = []
        for c_i in candidate_contexts:
            a_i, log_prob_i = self.policy.get_action(s, c_i)
            # Ranking uses w; the selected action keeps its generating context's gradient.
            q_i = torch.min(self.q1(s, a_i, c), self.q2(s, a_i, c))
            scalar_i = (w * q_i).sum(dim=-1, keepdim=True)
            action_candidates.append(a_i)
            log_prob_candidates.append(log_prob_i)
            scalar_candidates.append(scalar_i)

        action_stack = torch.stack(action_candidates, dim=0)
        log_prob_stack = torch.stack(log_prob_candidates, dim=0)
        scalar_stack = torch.stack(scalar_candidates, dim=0)
        best_idx = scalar_stack.detach().squeeze(-1).argmax(dim=0)
        batch_idx = torch.arange(w.shape[0], device=self.device)
        new_a = action_stack[best_idx, batch_idx]
        log_prob = log_prob_stack[best_idx, batch_idx]
        q_scalar = scalar_stack[best_idx, batch_idx]
        return new_a, q_scalar, self.actor_objective(s, w, c, q_scalar, log_prob)

    def target_critic_context(self, update_context, candidate_context):
        return candidate_context

    @torch.no_grad()
    def envelope_next_values(self, s_next, w, c_next):
        candidate_contexts = [c_next]
        for _ in range(max(1, int(self.envelope_target_candidates)) - 1):
            cand_w = self.sample_preference_batch(
                w.shape[0], self.preference_relabeling_strategy)
            candidate_contexts.append(
                self.replace_context_preferences(c_next, cand_w))

        q_next_candidates = []
        scalar_candidates = []
        for c_next_i in candidate_contexts:
            next_a_i, next_log_prob_i = self.policy.get_action(s_next, c_next_i)
            q_context = self.target_critic_context(c_next, c_next_i)
            q1_next_i = self.q1_target(s_next, next_a_i, q_context)
            q2_next_i = self.q2_target(s_next, next_a_i, q_context)
            q_next_i = torch.min(q1_next_i, q2_next_i)
            q_next_candidates.append(q_next_i)
            q_next_scalar_i = (w * q_next_i).sum(dim=-1, keepdim=True)
            scalar_candidates.append(q_next_scalar_i - self.alpha * next_log_prob_i)

        q_next_stack = torch.stack(q_next_candidates, dim=0)
        scalar_stack = torch.stack(scalar_candidates, dim=0)
        best_idx = scalar_stack.squeeze(-1).argmax(dim=0)
        batch_idx = torch.arange(w.shape[0], device=self.device)
        return scalar_stack[best_idx, batch_idx], q_next_stack[best_idx, batch_idx]

    def update(self, w_anchor=None, anchor_eta=0.01):
        """One gradient step. Returns (mean_q, target_q_vec) for diagnostics."""
        if self.buffer.size < self.batch_size:
            return None, None

        s, a, w, c, r, s_next, c_next, done = self.buffer.sample(self.batch_size)
        w, c, c_next = self.relabel_preferences(w, c, c_next)
        copies = max(1, int(self.critic_relabel_copies))
        if copies > 1:
            batch = w.shape[0]
            pref_blocks = [w]
            for _ in range(copies - 1):
                pref_blocks.append(self.sample_preference_batch(
                    batch, self.preference_relabeling_strategy))
            w = torch.cat(pref_blocks, dim=0)
            s = s.repeat(copies, 1)
            a = a.repeat(copies, 1)
            r = r.repeat(copies, 1)
            s_next = s_next.repeat(copies, 1)
            done = done.repeat(copies, 1)
            c = self.replace_context_preferences(c.repeat(copies, 1), w)
            c_next = self.replace_context_preferences(c_next.repeat(copies, 1), w)
        r_train = self.normalize_reward_batch(r)

        # --- Critic update ---
        with torch.no_grad():
            next_scalar, q_next = self.envelope_next_values(s_next, w, c_next)
            target_scalar, target_vec = self.critic_targets(
                r_train, w, done, next_scalar, q_next)

        q1_vec = self.q1(s, a, c)
        q2_vec = self.q2(s, a, c)
        q_loss = self.critic_objective(q1_vec, q2_vec, w, target_scalar, target_vec)

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()
        self.update_work["critic_optimizer_steps"] += 1
        self.update_work["critic_rows"] += len(s)
        self.update_work["target_candidate_rows"] += len(s) * max(1, self.envelope_target_candidates)

        # --- Policy update ---
        if self.dual_branch_update and hasattr(self.policy, "training_branches"):
            branch_losses = []
            branch_q_scalars = []
            for branch in self.policy.training_branches():
                new_a, log_prob = branch.get_action(s, c)
                q1_new = self.q1(s, new_a, c)
                q2_new = self.q2(s, new_a, c)
                q_new = torch.min(q1_new, q2_new)
                q_scalar_i = (w * q_new).sum(dim=-1, keepdim=True)
                branch_losses.append((self.alpha * log_prob - q_scalar_i).mean())
                branch_q_scalars.append(q_scalar_i)
            actor_rows = len(s) * len(branch_losses)
            policy_loss = torch.stack(branch_losses).mean()
            q_scalar = torch.stack(branch_q_scalars, dim=0).mean(dim=0)
        else:
            policy_candidates = max(1, int(self.envelope_policy_candidates))
            actor_rows = len(s) * policy_candidates
            if policy_candidates > 1:
                new_a, q_scalar, policy_loss = self.envelope_actor_loss(s, w, c)
            else:
                new_a, log_prob = self.policy.get_action(s, c)
                q1_new = self.q1(s, new_a, c)
                q2_new = self.q2(s, new_a, c)
                q_new = torch.min(q1_new, q2_new)
                q_scalar = (w * q_new).sum(dim=-1, keepdim=True)
                policy_loss = self.actor_objective(s, w, c, q_scalar, log_prob)

        bc_coef = max(0.0, float(self.behavior_clone_coef))
        if bc_coef > 0.0 and "new_a" in locals():
            policy_loss = policy_loss + bc_coef * self.behavior_clone_loss(new_a, a)

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
        self.update_work["actor_optimizer_steps"] += 1
        self.update_work["actor_candidate_rows"] += actor_rows

        # --- Target update ---
        for p, p_tgt in zip(self.q1.parameters(), self.q1_target.parameters()):
            p_tgt.data.lerp_(p.data, self.tau)
        for p, p_tgt in zip(self.q2.parameters(), self.q2_target.parameters()):
            p_tgt.data.lerp_(p.data, self.tau)

        mean_q = q_scalar.mean().item()
        target_q_mean = target_vec.mean(dim=0).detach().cpu().numpy()
        return mean_q, target_q_mean


class ConsistentTargetMOSACTrainer(MOSACTrainer):
    """V24 repair: continuing-task masks and one common soft critic target."""

    critic_target_rule = "scalar_projection_of_soft_vector"
    bootstrap_terminal_rule = "terminated_only"

    @staticmethod
    def replay_terminal(terminated, truncated):
        return terminated

    def critic_targets(self, reward, weights, done, next_scalar, next_vector):
        # The selected candidate's entropy bonus is common to all objectives.
        # Simplex weights then project the vector onto the same soft target.
        entropy_bonus = next_scalar - (weights * next_vector).sum(dim=-1, keepdim=True)
        target_vector = reward + self.gamma * (1 - done) * (next_vector + entropy_bonus)
        target_scalar = (weights * target_vector).sum(dim=-1, keepdim=True)
        return target_scalar, target_vector


class CommonTargetMOSACTrainer(ConsistentTargetMOSACTrainer):
    """One own-preference soft target and full-vector loss for every sampler."""

    critic_candidate_context_rule = "own_preference"
    critic_loss_rule = "mean_twin_full_vector_mse"
    actor_update_rule = "own_preference_soft_sac"

    def __init__(self, *args, **kwargs):
        kwargs.update(preference_relabeling=True, preference_relabeling_rate=1.0,
                      preference_relabeling_strategy="uniform", critic_relabel_copies=1,
                      envelope_target_candidates=1, envelope_policy_candidates=1,
                      vector_critic_coef=0.0, behavior_clone_coef=0.0)
        super().__init__(*args, **kwargs)

    def critic_objective(self, q1_vec, q2_vec, w, target_scalar, target_vec):
        return 0.5 * (F.mse_loss(q1_vec, target_vec) + F.mse_loss(q2_vec, target_vec))


class ProjectedCommonTargetMOSACTrainer(CommonTargetMOSACTrainer):
    """Emphasize the actor's value direction without losing vector supervision."""

    critic_loss_rule = "trace_normalized_vector_plus_projection_mse"

    def critic_objective(self, q1_vec, q2_vec, w, target_scalar, target_vec):
        errors = torch.stack((q1_vec - target_vec, q2_vec - target_vec))
        vector_error = errors.square().mean(dim=-1)
        projected_error = (errors * w).sum(dim=-1).square()
        # I/d + ww^T has trace 1 + ||w||^2; preserve the baseline's unit trace.
        return ((vector_error + projected_error) / (1 + w.square().sum(dim=-1))).mean()


def optimizer_step_count(optimizer):
    if optimizer is None:
        return 0
    return max((int(state["step"]) for state in optimizer.state.values() if "step" in state), default=0)


def network_parameter_counts(policy, critics, encoder=None):
    return dict(actor=sum(p.numel() for p in policy.parameters()),
                critic=sum(p.numel() for q in critics for p in q.parameters()),
                encoder=sum(p.numel() for p in encoder.parameters()) if encoder is not None else 0)


class PreferenceAlignedTargetMOSACTrainer(ConsistentTargetMOSACTrainer):
    """Evaluate proposal Q at the update preference; keep proposal log-probs."""

    critic_candidate_context_rule = "update_preference"

    def target_critic_context(self, update_context, candidate_context):
        return update_context


class MeanValueMOSACTrainer(PreferenceAlignedTargetMOSACTrainer):
    """Blend selected-sample and own-preference mean-action values at fixed scale."""

    actor_update_rule = "half_selected_half_own_preference_mean_value"
    mean_value_fraction = 0.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.single_candidate_updates = 0
        self.envelope_updates = 0

    def actor_objective(self, s, w, c, q_scalar, log_prob):
        mean_action = self.policy.get_action(s, c, deterministic=True)
        mean_q = torch.min(self.q1(s, mean_action, c), self.q2(s, mean_action, c))
        mean_scalar = (w * mean_q).sum(dim=-1, keepdim=True)
        if self.envelope_policy_candidates > 1:
            self.envelope_updates += 1
        else:
            self.single_candidate_updates += 1
        mixed_value = (1 - self.mean_value_fraction) * q_scalar + self.mean_value_fraction * mean_scalar
        return (self.alpha * log_prob - mixed_value).mean()

    def actor_update_diagnostics(self):
        return {"rule": self.actor_update_rule, "mean_value_fraction": self.mean_value_fraction,
                "single_candidate_updates": self.single_candidate_updates,
                "envelope_updates": self.envelope_updates,
                "updates": self.single_candidate_updates + self.envelope_updates}


class AlignedEnvelopeMOSACTrainer(ConsistentTargetMOSACTrainer):
    """Transfer detached proposals to the current preference, not their source."""

    actor_update_rule = "own_preference_sac_plus_detached_proposal_projection"
    projection_coef = 0.1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.projection_updates = 0
        self.proposal_samples = 0
        self.proposal_wins = 0
        self.projection_mse_sum = 0.0

    def envelope_actor_loss(self, s, w, c):
        candidate_contexts = [c]
        for _ in range(max(1, int(self.envelope_policy_candidates)) - 1):
            cand_w = self.sample_preference_batch(w.shape[0], self.preference_relabeling_strategy)
            candidate_contexts.append(self.replace_context_preferences(c, cand_w))

        own_action, own_log_prob = self.policy.get_action(s, c)
        own_q = torch.min(self.q1(s, own_action, c), self.q2(s, own_action, c))
        own_scalar = (w * own_q).sum(dim=-1, keepdim=True)
        with torch.no_grad():
            actions, values = [own_action.detach()], [own_scalar.detach()]
            for candidate_context in candidate_contexts[1:]:
                action, _ = self.policy.get_action(s, candidate_context)
                q = torch.min(self.q1(s, action, c), self.q2(s, action, c))
                actions.append(action)
                values.append((w * q).sum(dim=-1, keepdim=True))
            action_stack, value_stack = torch.stack(actions), torch.stack(values)
            best = value_stack.squeeze(-1).argmax(dim=0)
            batch = torch.arange(w.shape[0], device=self.device)
            proposal = action_stack[best, batch]
            improves = (value_stack[best, batch] > own_scalar.detach()).float()

        mean_action = self.policy.get_action(s, c, deterministic=True)
        projection_mse = (improves * (mean_action - proposal).square().mean(-1, keepdim=True)).mean()
        loss = (self.alpha * own_log_prob - own_scalar).mean() + self.projection_coef * projection_mse
        self.projection_updates += 1
        self.proposal_samples += w.shape[0]
        self.proposal_wins += int(improves.sum().item())
        self.projection_mse_sum += float(projection_mse.detach().item())
        return own_action, own_scalar, loss

    def actor_update_diagnostics(self):
        return {
            "rule": self.actor_update_rule,
            "projection_coef": self.projection_coef,
            "projection_updates": self.projection_updates,
            "proposal_samples": self.proposal_samples,
            "proposal_wins": self.proposal_wins,
            "proposal_win_fraction": self.proposal_wins / max(1, self.proposal_samples),
            "mean_projection_mse": self.projection_mse_sum / max(1, self.projection_updates),
        }


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


@lru_cache(maxsize=None)
def simplex_lattice_points(reward_dim: int, grid_size: int):
    """Deterministic simplex lattice used by stratified preference sampling."""
    total = max(int(grid_size) - 1, 1)
    points = []

    def rec(prefix, remaining, dims_left):
        if dims_left == 1:
            points.append(tuple(prefix + [remaining]))
            return
        for value in range(remaining + 1):
            rec(prefix + [value], remaining - value, dims_left - 1)

    rec([], total, int(reward_dim))
    arr = np.asarray(points, dtype=np.float32) / float(total)
    return tuple(tuple(float(x) for x in row) for row in arr)


def sample_stratified_weight(reward_dim, step: int, grid_size: int = 11):
    """Deterministic simplex coverage for seed-robust preference training."""
    if reward_dim == 2:
        idx = (int(step) - 1) % grid_size
        w0 = idx / max(grid_size - 1, 1)
        w = np.array([w0, 1.0 - w0], dtype=np.float32)
        return w, None

    # Higher-dimensional fallback: cycle one-hot corners plus the center.
    idx = (int(step) - 1) % (reward_dim + 1)
    if idx < reward_dim:
        w = np.zeros(reward_dim, dtype=np.float32)
        w[idx] = 1.0
    else:
        w = np.ones(reward_dim, dtype=np.float32) / reward_dim
    return w, None


def sample_lattice_weight(reward_dim, step: int, grid_size: int = 11):
    """Higher-dimensional simplex lattice sampler for explicit ablations."""
    if reward_dim == 2:
        return sample_stratified_weight(reward_dim, step, grid_size)

    lattice = simplex_lattice_points(int(reward_dim), int(grid_size))
    stride = max(1, len(lattice) // 2)
    while np.gcd(stride, len(lattice)) != 1:
        stride += 1
    idx = ((int(step) - 1) * stride) % len(lattice)
    w = np.asarray(lattice[idx], dtype=np.float32)
    return w, None


def sample_expert_template_weight(reward_dim, step: int):
    """Cycle all preference-expert templates for balanced expert coverage."""
    templates = preference_expert_templates(int(reward_dim))
    idx = (int(step) - 1) % len(templates)
    return templates[idx].astype(np.float32), None


def sample_edge_weight(reward_dim, step: int):
    """Cycle preference corners plus center to prevent objective-region collapse."""
    reward_dim = int(reward_dim)
    idx = (int(step) - 1) % (reward_dim + 1)
    if idx < reward_dim:
        w = np.zeros(reward_dim, dtype=np.float32)
        w[idx] = 1.0
    else:
        w = np.ones(reward_dim, dtype=np.float32) / float(reward_dim)
    return w, None


class AdaptiveOperatorSelector:
    """Online selector over BAMOR training operators.

    The selector adapts lightweight training operators rather than replacing
    the actor-critic architecture.  Each operator defines a preference sampler
    and a replay preference-relabeling rate.
    """

    def __init__(self, reward_dim: int, epsilon: float = 0.15,
                 temperature: float = 0.35, lr: float = 0.04,
                 baseline_beta: float = 0.01, scale_beta: float = 0.01,
                 operators: list[str] | tuple[str, ...] | None = None):
        self.reward_dim = int(reward_dim)
        self.epsilon = float(epsilon)
        self.temperature = float(temperature)
        self.lr = float(lr)
        self.baseline_beta = float(baseline_beta)
        self.scale_beta = float(scale_beta)
        self.operators = list(operators) if operators is not None else [
            "uniform",
            "stratified",
            "mixed",
            "uniform_relabel",
            "stratified_relabel",
            "mixed_relabel",
        ]
        self.scores = {name: 0.0 for name in self.operators}
        self.counts = {name: 0 for name in self.operators}
        self.reward_baseline = 0.0
        self.reward_scale = 1.0
        self.initialized = False
        self.last_probs = {name: 1.0 / len(self.operators) for name in self.operators}

    def probabilities(self) -> dict[str, float]:
        scores = np.asarray([self.scores[name] for name in self.operators], dtype=np.float64)
        scores = scores - scores.max()
        temp = max(self.temperature, 1e-6)
        probs = np.exp(scores / temp)
        probs = probs / probs.sum()
        probs = (1.0 - self.epsilon) * probs + self.epsilon / len(self.operators)
        self.last_probs = {name: float(prob) for name, prob in zip(self.operators, probs)}
        return self.last_probs

    def select(self) -> str:
        probs = self.probabilities()
        return str(np.random.choice(self.operators, p=[probs[name] for name in self.operators]))

    def relabel_rate(self, operator: str) -> float:
        return 1.0 if "relabel" in str(operator).split("_") else 0.0

    def sample_weight(self, operator: str, reward_dim: int, step: int):
        base = str(operator).split("_")[0]
        if base == "uniform":
            return sample_uniform_weight(reward_dim)
        if base == "stratified":
            return sample_stratified_weight(reward_dim, step)
        if base == "mixed":
            if step % 2 == 0:
                return sample_stratified_weight(reward_dim, step // 2)
            return sample_uniform_weight(reward_dim)
        return sample_uniform_weight(reward_dim)

    def update(self, operator: str, scalar_reward: float) -> None:
        r = float(scalar_reward)
        if not np.isfinite(r):
            return
        if not self.initialized:
            self.reward_baseline = r
            self.reward_scale = 1.0
            self.initialized = True
        delta = r - self.reward_baseline
        beta = self.baseline_beta
        self.reward_baseline = (1.0 - beta) * self.reward_baseline + beta * r
        self.reward_scale = (1.0 - self.scale_beta) * self.reward_scale + self.scale_beta * abs(delta)
        normalized = float(np.tanh(delta / max(self.reward_scale, 1e-6)))
        self.scores[operator] = (1.0 - self.lr) * self.scores[operator] + self.lr * normalized
        self.counts[operator] += 1

    def state_dict(self) -> dict:
        return {
            "operators": list(self.operators),
            "scores": {k: float(v) for k, v in self.scores.items()},
            "counts": {k: int(v) for k, v in self.counts.items()},
            "last_probs": {k: float(v) for k, v in self.last_probs.items()},
            "reward_baseline": float(self.reward_baseline),
            "reward_scale": float(self.reward_scale),
            "epsilon": self.epsilon,
            "temperature": self.temperature,
        }


class BlockAdaptiveOperatorSelector(AdaptiveOperatorSelector):
    """Block-level operator selector with delayed credit assignment.

    V12 updated the operator score from one-step scalarized reward, which is
    too noisy in HalfCheetah.  V13 holds an operator fixed for a short training
    block and updates from the block's average scalarized reward.
    """

    def __init__(self, reward_dim: int, block_steps: int = 1000,
                 operators: list[str] | tuple[str, ...] | None = None,
                 epsilon: float = 0.20, temperature: float = 0.30,
                 lr: float = 0.18, baseline_beta: float = 0.15,
                 scale_beta: float = 0.10):
        super().__init__(
            reward_dim,
            epsilon=epsilon,
            temperature=temperature,
            lr=lr,
            baseline_beta=baseline_beta,
            scale_beta=scale_beta,
            operators=operators,
        )
        self.block_steps = int(block_steps)
        self.current_operator = None
        self.block_start_step = 1
        self.block_scalar_sum = 0.0
        self.block_count = 0
        self.step_counts = {name: 0 for name in self.operators}
        self.block_history = []

    def select_for_step(self, step: int) -> str:
        if self.current_operator is None:
            self.current_operator = self.select()
            self.block_start_step = int(step)
            self.block_scalar_sum = 0.0
            self.block_count = 0
        return self.current_operator

    def observe_step(self, operator: str, scalar_reward: float) -> None:
        if operator not in self.step_counts:
            return
        r = float(scalar_reward)
        if not np.isfinite(r):
            return
        self.block_scalar_sum += r
        self.block_count += 1
        self.step_counts[operator] += 1

    def maybe_finish_block(self, step: int, force: bool = False) -> None:
        if self.current_operator is None:
            return
        elapsed = int(step) - int(self.block_start_step) + 1
        if not force and elapsed < self.block_steps:
            return
        if self.block_count > 0:
            block_mean = self.block_scalar_sum / float(self.block_count)
            operator = self.current_operator
            super().update(operator, block_mean)
            self.block_history.append({
                "end_step": int(step),
                "operator": operator,
                "steps": int(self.block_count),
                "mean_scalar_reward": float(block_mean),
                "score_after": float(self.scores[operator]),
            })
            if len(self.block_history) > 200:
                self.block_history = self.block_history[-200:]
        self.current_operator = None
        self.block_scalar_sum = 0.0
        self.block_count = 0

    def state_dict(self) -> dict:
        out = super().state_dict()
        out.update({
            "block_steps": int(self.block_steps),
            "current_operator": self.current_operator,
            "block_start_step": int(self.block_start_step),
            "block_count": int(self.block_count),
            "step_counts": {k: int(v) for k, v in self.step_counts.items()},
            "block_history": list(self.block_history),
        })
        return out


class EnvelopeBlockAdaptiveOperatorSelector(BlockAdaptiveOperatorSelector):
    """Block selector whose operators change both sampling and critic updates."""

    def __init__(self, reward_dim: int, block_steps: int = 2000):
        operators = [
            "uniform",
            "stratified_relabel",
            "mixed_relabel",
            "uniform_relabel_envelope",
            "stratified_relabel_envelope",
            "mixed_relabel_envelope",
        ]
        super().__init__(
            reward_dim,
            block_steps=block_steps,
            operators=operators,
            epsilon=0.25,
            temperature=0.35,
            lr=0.16,
            baseline_beta=0.12,
            scale_beta=0.10,
        )
        self.round_robin_index = 0

    def select(self) -> str:
        untried = [name for name in self.operators if self.counts[name] == 0]
        if untried:
            choice = untried[0]
            probs = {name: 0.0 for name in self.operators}
            probs[choice] = 1.0
            self.last_probs = probs
            return choice
        return super().select()

    def trainer_config(self, operator: str) -> dict:
        tokens = set(str(operator).split("_"))
        base = str(operator).split("_")[0]
        strategy = base if base in {"uniform", "stratified"} else "mixed"
        use_envelope = "envelope" in tokens
        use_relabel = "relabel" in tokens
        return {
            "preference_relabeling_strategy": strategy,
            "critic_relabel_copies": 2 if use_relabel else 1,
            "envelope_target_candidates": 4 if use_envelope else 1,
            "vector_critic_coef": 0.03 if (use_envelope or use_relabel) else 0.0,
        }

    def state_dict(self) -> dict:
        out = super().state_dict()
        out["operator_trainer_configs"] = {
            name: self.trainer_config(name) for name in self.operators
        }
        return out


class PolicyEnvelopeBlockAdaptiveOperatorSelector(EnvelopeBlockAdaptiveOperatorSelector):
    """More aggressive selector that also adapts envelope policy improvement."""

    def __init__(self, reward_dim: int, block_steps: int = 2000):
        super().__init__(reward_dim, block_steps=block_steps)
        self.operators = [
            "uniform",
            "mixed_relabel",
            "stratified_relabel_envelope",
            "mixed_relabel_envelope",
            "mixed_relabel_policy",
            "uniform_relabel_envelope_policy",
            "stratified_relabel_envelope_policy",
            "mixed_relabel_envelope_policy_bc",
        ]
        self.scores = {name: 0.0 for name in self.operators}
        self.counts = {name: 0 for name in self.operators}
        self.last_probs = {name: 1.0 / len(self.operators) for name in self.operators}
        self.step_counts = {name: 0 for name in self.operators}

    def trainer_config(self, operator: str) -> dict:
        tokens = set(str(operator).split("_"))
        base = str(operator).split("_")[0]
        strategy = base if base in {"uniform", "stratified"} else "mixed"
        use_relabel = "relabel" in tokens
        use_envelope = "envelope" in tokens
        use_policy = "policy" in tokens
        use_bc = "bc" in tokens
        return {
            "preference_relabeling_strategy": strategy,
            "critic_relabel_copies": 2 if use_relabel else 1,
            "envelope_target_candidates": 4 if use_envelope else 1,
            "vector_critic_coef": 0.035 if (use_envelope or use_relabel) else 0.0,
            "envelope_policy_candidates": 4 if use_policy else 1,
            "behavior_clone_coef": 0.015 if use_bc else 0.0,
        }


class RiskAwarePolicyEnvelopeBlockAdaptiveOperatorSelector(PolicyEnvelopeBlockAdaptiveOperatorSelector):
    """Risk-sensitive operator portfolio for difficult MuJoCo transfer.

    V15 showed that actor-side envelope improvement can rescue some seeds, but
    it is slow and sometimes amplifies optimistic critic errors.  V16 keeps a
    smaller policy-envelope option in the portfolio and scores each block by a
    lower-confidence objective, so noisy/high-variance operators need a better
    mean return before the selector commits to them.
    """

    def __init__(self, reward_dim: int, block_steps: int = 2500,
                 risk_coef: float = 0.10, tail_coef: float = 0.05):
        super().__init__(reward_dim, block_steps=block_steps)
        self.operators = [
            "uniform",
            "mixed_relabel",
            "stratified_relabel_envelope",
            "mixed_relabel_envelope",
            "uniform_relabel_envelope_policy",
            "mixed_relabel_envelope_policy_bc",
        ]
        self.epsilon = 0.18
        self.temperature = 0.28
        self.lr = 0.14
        self.baseline_beta = 0.10
        self.scale_beta = 0.10
        self.risk_coef = float(risk_coef)
        self.tail_coef = float(tail_coef)
        self.scores = {name: 0.0 for name in self.operators}
        self.counts = {name: 0 for name in self.operators}
        self.last_probs = {name: 1.0 / len(self.operators) for name in self.operators}
        self.step_counts = {name: 0 for name in self.operators}
        self.block_scalar_sumsq = 0.0
        self.block_scalar_min = float("inf")

    def observe_step(self, operator: str, scalar_reward: float) -> None:
        if operator not in self.step_counts:
            return
        r = float(scalar_reward)
        if not np.isfinite(r):
            return
        self.block_scalar_sum += r
        self.block_scalar_sumsq += r * r
        self.block_scalar_min = min(self.block_scalar_min, r)
        self.block_count += 1
        self.step_counts[operator] += 1

    def maybe_finish_block(self, step: int, force: bool = False) -> None:
        if self.current_operator is None:
            return
        elapsed = int(step) - int(self.block_start_step) + 1
        if not force and elapsed < self.block_steps:
            return
        if self.block_count > 0:
            block_mean = self.block_scalar_sum / float(self.block_count)
            block_var = max(
                0.0,
                self.block_scalar_sumsq / float(self.block_count) - block_mean * block_mean,
            )
            block_std = float(np.sqrt(block_var))
            block_min = float(self.block_scalar_min)
            downside = max(0.0, block_mean - block_min)
            robust_score = (
                block_mean
                - self.risk_coef * block_std
                - self.tail_coef * downside
            )
            operator = self.current_operator
            AdaptiveOperatorSelector.update(self, operator, robust_score)
            self.block_history.append({
                "end_step": int(step),
                "operator": operator,
                "steps": int(self.block_count),
                "mean_scalar_reward": float(block_mean),
                "std_scalar_reward": float(block_std),
                "min_scalar_reward": float(block_min),
                "robust_scalar_reward": float(robust_score),
                "score_after": float(self.scores[operator]),
            })
            if len(self.block_history) > 200:
                self.block_history = self.block_history[-200:]
        self.current_operator = None
        self.block_scalar_sum = 0.0
        self.block_scalar_sumsq = 0.0
        self.block_scalar_min = float("inf")
        self.block_count = 0

    def trainer_config(self, operator: str) -> dict:
        tokens = set(str(operator).split("_"))
        base = str(operator).split("_")[0]
        strategy = base if base in {"uniform", "stratified"} else "mixed"
        use_relabel = "relabel" in tokens
        use_envelope = "envelope" in tokens
        use_policy = "policy" in tokens
        use_bc = "bc" in tokens
        return {
            "preference_relabeling_strategy": strategy,
            "critic_relabel_copies": 2 if use_relabel else 1,
            "envelope_target_candidates": 3 if use_envelope else 1,
            "vector_critic_coef": 0.04 if (use_envelope or use_relabel) else 0.0,
            "envelope_policy_candidates": 2 if use_policy else 1,
            "behavior_clone_coef": 0.008 if use_bc else 0.0,
        }

    def state_dict(self) -> dict:
        out = super().state_dict()
        out.update({
            "risk_coef": float(self.risk_coef),
            "tail_coef": float(self.tail_coef),
        })
        return out


class CoverageArchiveRiskAdaptiveOperatorSelector(RiskAwarePolicyEnvelopeBlockAdaptiveOperatorSelector):
    """Risk selector with a coverage floor for difficult preference regions.

    V17's archive preserves useful policies once they have been learned, but it
    cannot recover preference regions that a seed never trains well.  V18 keeps
    the risk-aware BAMOR operator adaptation while forcing a small minimum
    amount of edge/stratified relabeling and envelope policy work.
    """

    def __init__(self, reward_dim: int, block_steps: int = 2000,
                 min_step_fraction: float = 0.055):
        super().__init__(reward_dim, block_steps=block_steps,
                         risk_coef=0.08, tail_coef=0.04)
        self.operators = [
            "uniform",
            "edge_relabel",
            "stratified_relabel",
            "mixed_relabel",
            "edge_relabel_envelope",
            "stratified_relabel_envelope",
            "mixed_relabel_envelope",
            "edge_relabel_envelope_policy_bc",
        ]
        self.epsilon = 0.22
        self.temperature = 0.32
        self.lr = 0.12
        self.baseline_beta = 0.10
        self.scale_beta = 0.10
        self.min_step_fraction = float(min_step_fraction)
        self.scores = {name: 0.0 for name in self.operators}
        self.counts = {name: 0 for name in self.operators}
        self.last_probs = {name: 1.0 / len(self.operators) for name in self.operators}
        self.step_counts = {name: 0 for name in self.operators}
        self.block_history = []

    def _force_choice(self, choice: str) -> str:
        self.last_probs = {name: 0.0 for name in self.operators}
        self.last_probs[choice] = 1.0
        return choice

    def select(self) -> str:
        untried = [name for name in self.operators if self.counts[name] == 0]
        if untried:
            return self._force_choice(untried[0])

        total_steps = sum(self.step_counts.values())
        if total_steps > 0:
            target = self.min_step_fraction * float(total_steps)
            deficits = {
                name: target - float(self.step_counts.get(name, 0))
                for name in self.operators
            }
            choice, deficit = max(deficits.items(), key=lambda kv: kv[1])
            if deficit > 0.0:
                return self._force_choice(choice)

        return AdaptiveOperatorSelector.select(self)

    def sample_weight(self, operator: str, reward_dim: int, step: int):
        base = str(operator).split("_")[0]
        if base == "edge":
            return sample_edge_weight(reward_dim, step)
        if base == "stratified":
            return sample_stratified_weight(reward_dim, step)
        if base == "mixed":
            if step % 3 == 0:
                return sample_edge_weight(reward_dim, step // 3)
            if step % 2 == 0:
                return sample_stratified_weight(reward_dim, step // 2)
            return sample_uniform_weight(reward_dim)
        return sample_uniform_weight(reward_dim)

    def trainer_config(self, operator: str) -> dict:
        tokens = set(str(operator).split("_"))
        base = str(operator).split("_")[0]
        strategy = base if base in {"uniform", "stratified", "edge"} else "mixed"
        use_relabel = "relabel" in tokens
        use_envelope = "envelope" in tokens
        use_policy = "policy" in tokens
        use_bc = "bc" in tokens
        edge_like = base == "edge"
        return {
            "preference_relabeling_strategy": strategy,
            "critic_relabel_copies": 3 if (use_relabel and (use_envelope or edge_like)) else (2 if use_relabel else 1),
            "envelope_target_candidates": 4 if use_envelope else 1,
            "vector_critic_coef": 0.05 if (use_envelope or use_relabel) else 0.0,
            "envelope_policy_candidates": 3 if use_policy else 1,
            "behavior_clone_coef": 0.006 if use_bc else 0.0,
        }

    def state_dict(self) -> dict:
        out = super().state_dict()
        out.update({
            "min_step_fraction": float(self.min_step_fraction),
            "operator_trainer_configs": {
                name: self.trainer_config(name) for name in self.operators
            },
        })
        return out


class HybridArchiveRiskAdaptiveOperatorSelector(CoverageArchiveRiskAdaptiveOperatorSelector):
    """Hybrid v17/v18 operator pool for single-run policy-set robustness."""

    def __init__(self, reward_dim: int, block_steps: int = 2000,
                 min_step_fraction: float = 0.045):
        super().__init__(
            reward_dim,
            block_steps=block_steps,
            min_step_fraction=min_step_fraction,
        )
        self.operators = [
            "uniform",
            "edge_relabel",
            "stratified_relabel",
            "mixed_relabel",
            "edge_relabel_envelope",
            "stratified_relabel_envelope",
            "mixed_relabel_envelope",
            "uniform_relabel_envelope_policy",
            "mixed_relabel_envelope_policy_bc",
            "edge_relabel_envelope_policy_bc",
        ]
        self.epsilon = 0.20
        self.temperature = 0.30
        self.lr = 0.12
        self.scores = {name: 0.0 for name in self.operators}
        self.counts = {name: 0 for name in self.operators}
        self.last_probs = {name: 1.0 / len(self.operators) for name in self.operators}
        self.step_counts = {name: 0 for name in self.operators}
        self.block_history = []


class HierarchicalArchiveRiskAdaptiveOperatorSelector(RiskAwarePolicyEnvelopeBlockAdaptiveOperatorSelector):
    """Two-level selector over coverage and recovery operator families.

    V18 is strongest when edge/stratified coverage is already learning useful
    policies; V19 improves tail seeds by adding policy-envelope recovery, but a
    single large pool can spend too much budget on recovery in seeds where V18
    was already good.  V20 first chooses a family, then an operator inside that
    family, while keeping a small family-level coverage floor.
    """

    COVERAGE_OPS = (
        "edge_relabel",
        "stratified_relabel",
        "mixed_relabel",
        "edge_relabel_envelope",
        "stratified_relabel_envelope",
        "mixed_relabel_envelope",
        "edge_relabel_envelope_policy_bc",
    )
    RECOVERY_OPS = (
        "mixed_relabel",
        "stratified_relabel_envelope",
        "mixed_relabel_envelope",
        "mixed_relabel_policy",
        "uniform_relabel_envelope_policy",
        "stratified_relabel_envelope_policy",
        "mixed_relabel_envelope_policy_bc",
    )

    def __init__(self, reward_dim: int, block_steps: int = 1500):
        super().__init__(
            reward_dim,
            block_steps=block_steps,
            risk_coef=0.08,
            tail_coef=0.04,
        )
        self.families = {
            "coverage": [f"coverage:{name}" for name in self.COVERAGE_OPS],
            "recovery": [f"recovery:{name}" for name in self.RECOVERY_OPS],
        }
        self.operators = self.families["coverage"] + self.families["recovery"]
        self.warmup_sequence = [
            "coverage:mixed_relabel_envelope",
            "coverage:edge_relabel",
            "coverage:stratified_relabel",
            "coverage:edge_relabel_envelope_policy_bc",
            "recovery:mixed_relabel",
            "recovery:stratified_relabel_envelope",
            "recovery:uniform_relabel_envelope_policy",
            "recovery:mixed_relabel_envelope_policy_bc",
        ]
        self.epsilon = 0.17
        self.family_epsilon = 0.14
        self.temperature = 0.30
        self.family_temperature = 0.22
        self.lr = 0.13
        self.baseline_beta = 0.10
        self.scale_beta = 0.10
        self.min_family_fraction = {"coverage": 0.45, "recovery": 0.25}
        self.scores = {name: 0.0 for name in self.operators}
        self.counts = {name: 0 for name in self.operators}
        self.last_probs = {name: 1.0 / len(self.operators) for name in self.operators}
        self.step_counts = {name: 0 for name in self.operators}
        self.block_history = []

    def _split_operator(self, operator: str) -> tuple[str, str]:
        text = str(operator)
        if ":" in text:
            family, inner = text.split(":", 1)
            return family, inner
        return "coverage", text

    def _force_choice(self, choice: str) -> str:
        self.last_probs = {name: 0.0 for name in self.operators}
        self.last_probs[choice] = 1.0
        return choice

    def _family_step_counts(self) -> dict[str, int]:
        return {
            family: int(sum(self.step_counts.get(name, 0) for name in names))
            for family, names in self.families.items()
        }

    def _family_scores(self) -> dict[str, float]:
        out = {}
        for family, names in self.families.items():
            values = sorted((float(self.scores.get(name, 0.0)) for name in names), reverse=True)
            top = values[:2] if len(values) >= 2 else values
            out[family] = float(np.mean(top)) if top else 0.0
        return out

    def _sample_from_names(self, names: list[str]) -> str:
        scores = np.asarray([self.scores[name] for name in names], dtype=np.float64)
        scores = scores - scores.max()
        probs = np.exp(scores / max(self.temperature, 1e-6))
        probs = probs / probs.sum()
        probs = (1.0 - self.epsilon) * probs + self.epsilon / len(names)
        return str(np.random.choice(names, p=probs))

    def select(self) -> str:
        completed_blocks = sum(self.counts.values())
        if completed_blocks < len(self.warmup_sequence):
            return self._force_choice(self.warmup_sequence[completed_blocks])

        family_steps = self._family_step_counts()
        total_steps = sum(family_steps.values())
        if total_steps > 0:
            deficits = {
                family: self.min_family_fraction[family] * float(total_steps) - float(steps)
                for family, steps in family_steps.items()
            }
            family, deficit = max(deficits.items(), key=lambda kv: kv[1])
            if deficit > 0.0:
                return self._force_choice(self._sample_from_names(self.families[family]))

        family_scores = self._family_scores()
        names = list(self.families.keys())
        scores = np.asarray([family_scores[name] for name in names], dtype=np.float64)
        scores = scores - scores.max()
        probs = np.exp(scores / max(self.family_temperature, 1e-6))
        probs = probs / probs.sum()
        probs = (1.0 - self.family_epsilon) * probs + self.family_epsilon / len(names)
        family = str(np.random.choice(names, p=probs))
        choice = self._sample_from_names(self.families[family])

        self.last_probs = {name: 0.0 for name in self.operators}
        for family_name, family_prob in zip(names, probs):
            family_names = self.families[family_name]
            inner_scores = np.asarray([self.scores[name] for name in family_names], dtype=np.float64)
            inner_scores = inner_scores - inner_scores.max()
            inner_probs = np.exp(inner_scores / max(self.temperature, 1e-6))
            inner_probs = inner_probs / inner_probs.sum()
            inner_probs = (1.0 - self.epsilon) * inner_probs + self.epsilon / len(inner_probs)
            for name, prob in zip(family_names, inner_probs):
                self.last_probs[name] = float(family_prob) * float(prob)
        return choice

    def relabel_rate(self, operator: str) -> float:
        _, inner = self._split_operator(operator)
        return super().relabel_rate(inner)

    def sample_weight(self, operator: str, reward_dim: int, step: int):
        _, inner = self._split_operator(operator)
        base = inner.split("_")[0]
        if base == "edge":
            return sample_edge_weight(reward_dim, step)
        if base == "stratified":
            return sample_stratified_weight(reward_dim, step)
        if base == "mixed":
            if step % 3 == 0:
                return sample_edge_weight(reward_dim, step // 3)
            if step % 2 == 0:
                return sample_stratified_weight(reward_dim, step // 2)
            return sample_uniform_weight(reward_dim)
        return sample_uniform_weight(reward_dim)

    def trainer_config(self, operator: str) -> dict:
        family, inner = self._split_operator(operator)
        tokens = set(inner.split("_"))
        base = inner.split("_")[0]
        strategy = base if base in {"uniform", "stratified", "edge"} else "mixed"
        use_relabel = "relabel" in tokens
        use_envelope = "envelope" in tokens
        use_policy = "policy" in tokens
        use_bc = "bc" in tokens
        edge_like = base == "edge"
        recovery = family == "recovery"
        return {
            "preference_relabeling_strategy": strategy,
            "critic_relabel_copies": (
                3 if (use_relabel and (use_envelope or edge_like)) else
                (2 if use_relabel else 1)
            ),
            "envelope_target_candidates": 4 if use_envelope else 1,
            "vector_critic_coef": 0.05 if (use_envelope or use_relabel) else 0.0,
            "envelope_policy_candidates": (
                4 if (use_policy and recovery) else
                (3 if use_policy else 1)
            ),
            "behavior_clone_coef": 0.010 if (use_bc and recovery) else (0.006 if use_bc else 0.0),
        }

    def state_dict(self) -> dict:
        out = super().state_dict()
        out.update({
            "families": {k: list(v) for k, v in self.families.items()},
            "warmup_sequence": list(self.warmup_sequence),
            "family_scores": self._family_scores(),
            "family_step_counts": self._family_step_counts(),
            "family_epsilon": float(self.family_epsilon),
            "family_temperature": float(self.family_temperature),
            "min_family_fraction": {
                k: float(v) for k, v in self.min_family_fraction.items()
            },
            "operator_trainer_configs": {
                name: self.trainer_config(name) for name in self.operators
            },
        })
        return out


class CoverageBiasedHierarchicalArchiveRiskAdaptiveOperatorSelector(
    HierarchicalArchiveRiskAdaptiveOperatorSelector
):
    """V21: keep V20's recovery path, but require stronger evidence to leave coverage."""

    def __init__(self, reward_dim: int, block_steps: int = 1500):
        super().__init__(reward_dim, block_steps=block_steps)
        self.coverage_prior = 0.04
        self.family_epsilon = 0.10
        self.family_temperature = 0.20
        self.min_family_fraction = {"coverage": 0.55, "recovery": 0.18}

    def _raw_family_scores(self) -> dict[str, float]:
        out = {}
        for family, names in self.families.items():
            values = sorted((float(self.scores.get(name, 0.0)) for name in names), reverse=True)
            top = values[:2] if len(values) >= 2 else values
            out[family] = float(np.mean(top)) if top else 0.0
        return out

    def _family_scores(self) -> dict[str, float]:
        out = self._raw_family_scores()
        out["coverage"] = float(out.get("coverage", 0.0) + self.coverage_prior)
        return out

    def state_dict(self) -> dict:
        out = super().state_dict()
        out.update({
            "raw_family_scores": self._raw_family_scores(),
            "coverage_prior": float(self.coverage_prior),
        })
        return out


class ExtremeHierarchicalArchiveRiskAdaptiveOperatorSelector(
    HierarchicalArchiveRiskAdaptiveOperatorSelector
):
    """V23: add a frontier family for HalfCheetah endpoint policies.

    V22 improves robust policy-set output with a dense archive, but its weak
    HalfCheetah seeds still miss the high-velocity frontier that CAPQL often
    finds.  This selector keeps V20's coverage/recovery hierarchy and adds a
    small protected extreme family that repeatedly trains edge-conditioned
    policy-envelope operators.  The family remains scored by online rewards,
    so it is an adaptive BAMOR operator rather than a fixed baseline wrapper.
    """

    EXTREME_OPS = (
        "edge_relabel_envelope_policy",
        "edge_envelope_policy",
        "edge_relabel_envelope_policy_bc",
        "edge_relabel",
    )

    def __init__(self, reward_dim: int, block_steps: int = 1500):
        super().__init__(reward_dim, block_steps=block_steps)
        self.families["extreme"] = [
            f"extreme:{name}" for name in self.EXTREME_OPS
        ]
        self.operators = (
            self.families["coverage"]
            + self.families["recovery"]
            + self.families["extreme"]
        )
        self.warmup_sequence = [
            "coverage:mixed_relabel_envelope",
            "coverage:edge_relabel",
            "coverage:stratified_relabel",
            "recovery:mixed_relabel",
            "recovery:uniform_relabel_envelope_policy",
            "extreme:edge_relabel_envelope_policy",
            "extreme:edge_envelope_policy",
            "extreme:edge_relabel_envelope_policy_bc",
            "coverage:edge_relabel_envelope_policy_bc",
            "recovery:mixed_relabel_envelope_policy_bc",
        ]
        self.epsilon = 0.16
        self.family_epsilon = 0.16
        self.temperature = 0.28
        self.family_temperature = 0.22
        self.min_family_fraction = {
            "coverage": 0.36,
            "recovery": 0.22,
            "extreme": 0.18,
        }
        self.scores = {name: 0.0 for name in self.operators}
        self.counts = {name: 0 for name in self.operators}
        self.last_probs = {name: 1.0 / len(self.operators) for name in self.operators}
        self.step_counts = {name: 0 for name in self.operators}
        self.block_history = []

    def trainer_config(self, operator: str) -> dict:
        family, inner = self._split_operator(operator)
        config = super().trainer_config(operator)
        if family == "extreme":
            tokens = set(inner.split("_"))
            use_relabel = "relabel" in tokens
            use_bc = "bc" in tokens
            config.update({
                "preference_relabeling_strategy": "edge",
                "critic_relabel_copies": 3 if use_relabel else 1,
                "envelope_target_candidates": 5,
                "vector_critic_coef": 0.06,
                "envelope_policy_candidates": 5,
                "behavior_clone_coef": 0.004 if use_bc else 0.0,
            })
        return config


class FixedExtremeHierarchicalOperatorSelector(
    ExtremeHierarchicalArchiveRiskAdaptiveOperatorSelector
):
    """Keep v23 warmup and family quotas, with fixed zero selection scores."""

    def select(self) -> str:
        observed_scores = self.scores
        self.scores = dict.fromkeys(self.operators, 0.0)
        try:
            return super().select()
        finally:
            self.scores = observed_scores


class OwnPreferenceFixedExtremeOperatorSelector(FixedExtremeHierarchicalOperatorSelector):
    """Ablate actor-envelope search, retaining critic operators and existing BC."""

    def trainer_config(self, operator: str) -> dict:
        config = super().trainer_config(operator)
        config["envelope_policy_candidates"] = 1
        return config

    def state_dict(self) -> dict:
        out = super().state_dict()
        out["actor_update_rule"] = "own_preference_with_existing_bc"
        return out


class OwnPreferenceCriticFixedExtremeOperatorSelector(FixedExtremeHierarchicalOperatorSelector):
    """Ablate critic candidate search without changing relabeling or losses."""

    def trainer_config(self, operator: str) -> dict:
        config = super().trainer_config(operator)
        config["envelope_target_candidates"] = 1
        return config

    def state_dict(self) -> dict:
        out = super().state_dict()
        out["critic_candidate_rule"] = "own_preference_only"
        return out


class OwnPreferenceBothFixedExtremeOperatorSelector(OwnPreferenceCriticFixedExtremeOperatorSelector):
    """Disable both candidate searches, retaining all other fixed-portfolio work."""

    def trainer_config(self, operator: str) -> dict:
        config = super().trainer_config(operator)
        config["envelope_policy_candidates"] = 1
        return config

    def state_dict(self) -> dict:
        out = super().state_dict()
        out["actor_update_rule"] = "own_preference_with_existing_bc"
        return out


class ProgressExtremeHierarchicalOperatorSelector(
    ExtremeHierarchicalArchiveRiskAdaptiveOperatorSelector
):
    """Score within-block vector progress without the sampled preference weight."""

    def __init__(self, reward_dim: int, block_steps: int = 1500,
                 learning_starts: int = 5000):
        super().__init__(reward_dim, block_steps=block_steps)
        self.learning_starts = int(learning_starts)
        self.vector_mean = np.zeros(reward_dim, dtype=np.float64)
        self.vector_var = np.zeros(reward_dim, dtype=np.float64)
        self.vector_initialized = False
        self.block_scale = np.ones(reward_dim, dtype=np.float64)
        self.half_sums = np.zeros((2, reward_dim), dtype=np.float64)
        self.half_counts = np.zeros(2, dtype=np.int64)

    def select_for_step(self, step: int) -> str:
        # The action at learning_starts precedes the first gradient update.
        if step <= self.learning_starts:
            return self.warmup_sequence[0]
        if self.current_operator is None:
            self.block_scale = np.sqrt(self.vector_var).clip(min=1e-6)
            self.half_sums.fill(0)
            self.half_counts.fill(0)
        return super().select_for_step(step)

    def observe_transition(self, operator: str, reward, step: int) -> None:
        reward = np.asarray(reward, dtype=np.float64)
        if not np.all(np.isfinite(reward)):
            return
        if not self.vector_initialized:
            self.vector_mean[:] = reward
            self.vector_initialized = True
        else:
            beta = 1.0 / self.block_steps
            delta = reward - self.vector_mean
            self.vector_mean += beta * delta
            self.vector_var = (1.0 - beta) * (self.vector_var + beta * delta ** 2)
        if step <= self.learning_starts:
            return
        half = min(1, self.block_count // (self.block_steps // 2))
        self.half_sums[half] += reward
        self.half_counts[half] += 1
        self.block_count += 1
        self.step_counts[operator] += 1

    def maybe_finish_block(self, step: int, force: bool = False) -> None:
        if self.current_operator is None:
            return
        elapsed = int(step) - self.block_start_step + 1
        if not force and elapsed < self.block_steps:
            return
        operator = self.current_operator
        complete = self.block_count == self.block_steps
        record = {"end_step": int(step), "operator": operator,
                  "steps": int(self.block_count), "complete": complete,
                  "half_counts": self.half_counts.tolist(),
                  "scale_before_block": self.block_scale.tolist()}
        if complete:
            means = self.half_sums / self.half_counts[:, None]
            gain = (means[1] - means[0]) / self.block_scale
            score = float(np.tanh(gain).mean())
            self.scores[operator] = (1 - self.lr) * self.scores[operator] + self.lr * score
            self.counts[operator] += 1
            record.update({"half_mean_rewards": means.tolist(),
                           "normalized_vector_gain": gain.tolist(),
                           "progress_score": score})
        record["score_after"] = float(self.scores[operator])
        self.block_history.append(record)
        self.block_history = self.block_history[-200:]
        self.current_operator = None
        self.block_count = 0

    def state_dict(self) -> dict:
        out = super().state_dict()
        for key in ["risk_coef", "tail_coef", "reward_baseline", "reward_scale"]:
            out.pop(key, None)
        out.update({"score_signal": "within_block_normalized_vector_progress",
                    "learning_starts": self.learning_starts,
                    "vector_reward_mean": self.vector_mean.tolist(),
                    "vector_reward_variance": self.vector_var.tolist()})
        return out


class CommonTargetSamplingSelector(ProgressExtremeHierarchicalOperatorSelector):
    """Allocate collection preferences while leaving the learning objective fixed."""

    selection_uses_scores = True

    def __init__(self, reward_dim, block_steps=1500, learning_starts=5000):
        super().__init__(reward_dim, block_steps, learning_starts)
        self.operators = ["uniform", "stratified", "mixed"]
        self.warmup_sequence = list(self.operators)
        self.scores = dict.fromkeys(self.operators, 0.0)
        self.counts = dict.fromkeys(self.operators, 0)
        self.step_counts = dict.fromkeys(self.operators, 0)
        self.last_probs = dict.fromkeys(self.operators, 1.0 / len(self.operators))
        self.epsilon, self.temperature = 0.15, 0.35

    def select(self):
        completed = sum(self.counts.values())
        if completed < len(self.operators):
            return self._force_choice(self.warmup_sequence[completed])
        probs = self.probabilities()
        return str(np.random.choice(self.operators, p=[probs[name] for name in self.operators]))

    def probabilities(self):
        return AdaptiveOperatorSelector.probabilities(self)

    def sample_weight(self, operator, reward_dim, step):
        return AdaptiveOperatorSelector.sample_weight(self, operator, reward_dim, step)

    def relabel_rate(self, operator):
        return 1.0

    def trainer_config(self, operator):
        return dict(preference_relabeling_strategy="uniform", critic_relabel_copies=1,
                    envelope_target_candidates=1, envelope_policy_candidates=1,
                    vector_critic_coef=0.0, behavior_clone_coef=0.0)

    def state_dict(self):
        out = BlockAdaptiveOperatorSelector.state_dict(self)
        out.update(score_signal="within_block_normalized_vector_progress",
                   selection_uses_scores=self.selection_uses_scores,
                   learning_starts=self.learning_starts,
                   vector_reward_mean=self.vector_mean.tolist(),
                   vector_reward_variance=self.vector_var.tolist(),
                   learning_preference_distribution="uniform_simplex",
                   operator_trainer_configs={op: self.trainer_config(op) for op in self.operators})
        return out


class FixedCommonTargetSamplingSelector(CommonTargetSamplingSelector):
    selection_uses_scores = False

    def probabilities(self):
        self.last_probs = dict.fromkeys(self.operators, 1.0 / len(self.operators))
        return self.last_probs


class EpisodeReturnSamplingSelector(AdaptiveOperatorSelector):
    """Score full single-sampler episodes against a causal return baseline."""

    selection_uses_scores = True

    def __init__(self, reward_dim, learning_starts=5000):
        super().__init__(reward_dim, epsilon=0.15, temperature=0.35, lr=0.13,
                         operators=["uniform", "stratified", "mixed"])
        self.learning_starts = int(learning_starts)
        self.detector_config = dict(ema_alpha=1 / 1500, warmup=2000,
            surprise_threshold=18.0, switch_cooldown=2000,
            cusum_required_consecutive=5, cusum_decay=0.5)
        self.surprise = SurpriseComputer(reward_dim, ema_alpha=self.detector_config["ema_alpha"],
                                        warmup=self.detector_config["warmup"])
        self.belief = BeliefTracker(1, **{key: value for key, value in self.detector_config.items()
                                       if key not in ("ema_alpha", "warmup")})
        self.epoch = 0
        self.segment_counts = dict.fromkeys(self.operators, 0)
        self.step_counts = dict.fromkeys(self.operators, 0)
        self.baseline_ready = False
        self.return_mean = self.return_var = 0.0
        self.current_operator = None
        self.episode_return = np.zeros(reward_dim, dtype=np.float64)
        self.episode_length = 0
        self.episode_mixed = False
        self.episode_history, self.reset_history = [], []

    def select_for_step(self, step):
        if self.current_operator is None:
            completed = sum(self.segment_counts.values())
            self.current_operator = (self.operators[0] if step <= self.learning_starts else
                self.operators[completed] if completed < len(self.operators) else self.select())
            self.episode_start = int(step)
            self.episode_epoch = self.epoch
            self._freeze_baseline(step)
        return self.current_operator

    def _freeze_baseline(self, step):
        self.baseline_before = self.return_mean if self.baseline_ready else None
        self.scale_before = max(np.sqrt(self.return_var), 1e-6) if self.baseline_ready else None

    def _reset_baseline(self):
        self.baseline_ready = False
        self.return_mean = self.return_var = 0.0

    def _observe_completed_return(self, utility):
        if not self.baseline_ready:
            self.return_mean, self.return_var, self.baseline_ready = utility, 0.0, True
        else:
            delta = utility - self.return_mean
            self.return_mean += self.lr * delta
            self.return_var = (1 - self.lr) * (self.return_var + self.lr * delta ** 2)

    def observe_transition(self, operator, reward, step, done):
        reward = np.asarray(reward, dtype=np.float64)
        changed = self.belief.update(self.surprise.compute(reward))["mode_switched"]
        if changed:
            self.epoch += 1
            self.scores = dict.fromkeys(self.operators, 0.0)
            self.segment_counts = dict.fromkeys(self.operators, 0)
            self._reset_baseline()
            self.episode_mixed = True
            self.reset_history.append(dict(step=int(step), epoch=self.epoch))
            self.surprise.reset()
        self.episode_return += reward
        self.episode_length += 1
        self.step_counts[operator] += 1
        if not done:
            return
        utility = float(self.episode_return.mean())
        record = dict(start_step=self.episode_start, end_step=int(step), operator=operator,
            length=self.episode_length, **{"return": self.episode_return.tolist()},
            utility=utility, epoch_start=self.episode_epoch, epoch_end=self.epoch,
            mixed_epoch=self.episode_mixed, baseline_before=self.baseline_before,
            scale_before=self.scale_before, credited=False, score=None)
        if not self.episode_mixed:
            if self.episode_start > self.learning_starts and self.baseline_before is not None:
                score = float(np.tanh((utility - self.baseline_before) / self.scale_before))
                self.scores[operator] = (1 - self.lr) * self.scores[operator] + self.lr * score
                self.counts[operator] += 1
                self.segment_counts[operator] += 1
                record.update(credited=True, score=score)
            self._observe_completed_return(utility)
        record["score_after"] = float(self.scores[operator])
        self.episode_history.append(record)
        self.current_operator = None
        self.episode_return.fill(0)
        self.episode_length = 0
        self.episode_mixed = False

    def relabel_rate(self, operator):
        return 1.0

    def trainer_config(self, operator):
        return CommonTargetSamplingSelector.trainer_config(self, operator)

    def state_dict(self):
        return dict(operators=list(self.operators), scores=dict(self.scores), counts=dict(self.counts),
            step_counts=dict(self.step_counts), last_probs=dict(self.last_probs),
            epsilon=self.epsilon, temperature=self.temperature, score_rate=self.lr,
            learning_starts=self.learning_starts, selection_uses_scores=self.selection_uses_scores,
            score_signal="causal_episode_return_advantage", credit_unit="complete_single_sampler_episode",
            utility_rule="equal_weight_undiscounted_raw_episode_return", baseline_rate=self.lr,
            detector_config=dict(self.detector_config), epoch=self.epoch,
            segment_counts=dict(self.segment_counts), return_baseline=self.return_mean,
            return_variance=self.return_var, episode_history=list(self.episode_history),
            reset_history=list(self.reset_history), partial_episode_length=self.episode_length,
            operator_trainer_configs={op: self.trainer_config(op) for op in self.operators})


class FixedEpisodeReturnSamplingSelector(EpisodeReturnSamplingSelector):
    selection_uses_scores = False

    def probabilities(self):
        self.last_probs = dict.fromkeys(self.operators, 1 / len(self.operators))
        return self.last_probs


class ForecastEpisodeReturnSamplingSelector(EpisodeReturnSamplingSelector):
    """Isolated collection-credit ablation with unchanged learning updates."""

    def __init__(self, reward_dim, learning_starts=5000, credit_rule="blend", adaptive=True):
        super().__init__(reward_dim, learning_starts)
        self.credit_rule = credit_rule
        self.selection_uses_scores = adaptive
        self.forecaster = EpisodeCreditBlend(self.lr)
        self.forecast_fits = 0
        self.first_free_start_step = None

    def _freeze_baseline(self, step):
        self.forecast_before = self.forecaster.predict(step)
        before = self.forecast_before
        self.forecast_fits += before["trend_prediction"] is not None
        self.credit_trend_weight = (before["trend_weight"] if self.credit_rule == "blend" else
                                   .5 if self.credit_rule == "equal_mixture" else 0.)
        self.scale_before = before["scale"]
        self.baseline_before = None
        if before["trend_prediction"] is not None and self.scale_before is not None:
            self.baseline_before = ((1 - self.credit_trend_weight) * before["ema_prediction"]
                                    + self.credit_trend_weight * before["trend_prediction"])
        completed = sum(self.segment_counts.values())
        self.selection_phase = ("warmup" if step <= self.learning_starts else
                                "forced" if completed < len(self.operators) else "free")
        if self.selection_phase == "free" and self.first_free_start_step is None:
            self.first_free_start_step = int(step)

    def _reset_baseline(self):
        super()._reset_baseline()
        self.forecaster.reset()

    def _observe_completed_return(self, utility):
        self.forecaster.observe(self.episode_start, utility, self.forecast_before, False)
        scale = self.forecaster.baseline.scale
        self.return_mean = scale.mean
        self.return_var = scale.variance if scale.variance is not None else 0.
        self.baseline_ready = scale.variance is not None

    def observe_transition(self, operator, reward, step, done):
        super().observe_transition(operator, reward, step, done)
        if done:
            self.episode_history[-1].update(forecast_before=dict(self.forecast_before),
                credit_trend_weight=self.credit_trend_weight, credit_rule=self.credit_rule,
                selection_phase=self.selection_phase)

    def probabilities(self):
        if self.selection_uses_scores:
            return super().probabilities()
        self.last_probs = dict.fromkeys(self.operators, 1 / len(self.operators))
        return self.last_probs

    def state_dict(self):
        out = super().state_dict()
        out.update(score_signal="causal_episode_forecast_advantage", credit_rule=self.credit_rule,
            baseline_initial_episodes=INITIAL_EPISODES, trend_window=TREND_WINDOW,
            blend_rule="clipped_past_pair_squared_error", statistical_baseline_fits=self.forecast_fits,
            first_free_start_step=self.first_free_start_step,
            forecast_evidence=dict(cross_product=self.forecaster.cross_product,
                disagreement_energy=self.forecaster.disagreement_energy,
                comparisons=self.forecaster.comparisons,
                scale_observations=self.forecaster.baseline.scale.count))
        return out


class FixedProgressExtremeHierarchicalOperatorSelector(
    ProgressExtremeHierarchicalOperatorSelector
):
    """Same progress statistics and learning-only warmup, but ignore the scores."""

    def select(self) -> str:
        observed_scores = self.scores
        self.scores = dict.fromkeys(self.operators, 0.0)
        try:
            return super().select()
        finally:
            self.scores = observed_scores


class OwnPreferenceProgressExtremeOperatorSelector(
    ProgressExtremeHierarchicalOperatorSelector
):
    """Keep progress-driven routing while training the actor at its own preference."""

    def trainer_config(self, operator: str) -> dict:
        config = super().trainer_config(operator)
        config["envelope_policy_candidates"] = 1
        return config

    def state_dict(self) -> dict:
        out = super().state_dict()
        out["actor_update_rule"] = "own_preference_with_existing_bc"
        return out


class OwnPreferenceFixedProgressExtremeOperatorSelector(
    FixedProgressExtremeHierarchicalOperatorSelector
):
    """Keep fixed v25 routing while removing cross-preference actor search."""

    def trainer_config(self, operator: str) -> dict:
        config = super().trainer_config(operator)
        config["envelope_policy_candidates"] = 1
        return config

    def state_dict(self) -> dict:
        out = super().state_dict()
        out["actor_update_rule"] = "own_preference_with_existing_bc"
        return out


class OwnPreferenceBothFixedProgressExtremeOperatorSelector(
    OwnPreferenceFixedProgressExtremeOperatorSelector
):
    """Ablate critic-envelope search after the own-preference actor repair."""

    def trainer_config(self, operator: str) -> dict:
        config = super().trainer_config(operator)
        config["envelope_target_candidates"] = 1
        return config

    def state_dict(self) -> dict:
        out = super().state_dict()
        out["critic_candidate_rule"] = "own_preference_only"
        return out


class OwnPreferenceBothProgressExtremeOperatorSelector(
    OwnPreferenceProgressExtremeOperatorSelector
):
    """Retain feedback-driven routing with own-preference actor and critic updates."""

    def trainer_config(self, operator: str) -> dict:
        config = super().trainer_config(operator)
        config["envelope_target_candidates"] = 1
        return config

    def state_dict(self) -> dict:
        out = super().state_dict()
        out["critic_candidate_rule"] = "own_preference_only"
        return out


def uses_transition_context(method: str) -> bool:
    return method in COMMON_TARGET_METHODS | EXTREME_PORTFOLIO_METHODS or method in {
        "bamor_v2_contrast_hybrid",
        "bamor_v2_proto_hybrid",
        "bamor_v4_contrast_expert_mix",
        "bamor_v4_contrast_expert_norm_mix",
        "bamor_v4_contrast_soft_expert_mix",
        "bamor_v5_contrast_expert_template_mix",
        "bamor_v5_contrast_moe_sharp_template_mix",
        "bamor_v5_contrast_moe_template_mix",
        "bamor_v6_contrast_moe_head_template_mix",
        "bamor_v7_contrast_critic_dual_template_mix",
        "bamor_v8_contrast_dual_update_template_mix",
        "bamor_v11_contrast_relabel_mix",
        "bamor_v12_adaptive_operator_mix",
        "bamor_v13_block_operator_mix",
        "bamor_v14_adaptive_envelope_mix",
        "bamor_v15_adaptive_policy_envelope_mix",
        "bamor_v16_risk_adaptive_mix",
        "bamor_v17_archive_risk_mix",
        "bamor_v18_coverage_archive_mix",
        "bamor_v19_hybrid_archive_mix",
        "bamor_v20_hier_archive_mix",
        "bamor_v21_covbias_hier_archive_mix",
        "bamor_v22_dense_hier_archive_mix",
        "bamor_v23_extreme_hier_archive_mix",
        "bamor_v3_contrast_norm_mix",
        "bamor_v3_contrast_lattice_mix",
        "bamor_v3_contrast_mix",
        "bamor_v3_contrast_stratified",
    }


def uses_generic_context(method: str) -> bool:
    return method in {"bamor_v2_proto_generic", "bamor_v3_seq_belief"}


def uses_learned_context(method: str) -> bool:
    return method in SINGLE_LATENT_METHODS


def uses_sequence_context(method: str) -> bool:
    return method in SEQUENCE_CONTEXT_METHODS


def uses_proto_context(method: str) -> bool:
    return method in {
        "bamor_v2_proto_hybrid",
        "bamor_v2_proto_generic",
        "bamor_v3_seq_belief",
    }


def uses_stratified_weights(method: str) -> bool:
    return method in {"bamor_v3_contrast_stratified", "bamor_v3_seq_belief"}


def uses_mixed_weights(method: str) -> bool:
    return method in {
        "bamor_v3_contrast_mix",
        "bamor_v3_contrast_norm_mix",
        "bamor_v4_contrast_expert_mix",
        "bamor_v4_contrast_expert_norm_mix",
        "bamor_v4_contrast_soft_expert_mix",
        "bamor_v11_contrast_relabel_mix",
    }


def uses_template_mixed_weights(method: str) -> bool:
    return method in {
        "bamor_v5_contrast_expert_template_mix",
        "bamor_v5_contrast_moe_sharp_template_mix",
        "bamor_v5_contrast_moe_template_mix",
        "bamor_v6_contrast_moe_head_template_mix",
        "bamor_v7_contrast_critic_dual_template_mix",
        "bamor_v8_contrast_dual_update_template_mix",
    }


def uses_lattice_mixed_weights(method: str) -> bool:
    return method in {"bamor_v3_contrast_lattice_mix"}


def uses_reward_normalization(method: str) -> bool:
    return method in {
        "uniform_norm",
        "bamor_v3_contrast_norm_mix",
        "bamor_v4_contrast_expert_norm_mix",
    }


def uses_expert_policy(method: str) -> bool:
    return method in {
        "bamor_v4_contrast_expert_mix",
        "bamor_v4_contrast_expert_norm_mix",
        "bamor_v5_contrast_expert_template_mix",
    }


def uses_soft_expert_policy(method: str) -> bool:
    return method in {"bamor_v4_contrast_soft_expert_mix"}


def uses_moe_expert_policy(method: str) -> bool:
    return method in {
        "bamor_v5_contrast_moe_sharp_template_mix",
        "bamor_v5_contrast_moe_template_mix",
    }


def uses_moe_head_policy(method: str) -> bool:
    return method in {"bamor_v6_contrast_moe_head_template_mix"}


def uses_dual_branch_policy(method: str) -> bool:
    return method in {
        "bamor_v7_contrast_critic_dual_template_mix",
        "bamor_v8_contrast_dual_update_template_mix",
    }


def uses_dual_branch_update(method: str) -> bool:
    return method in {"bamor_v8_contrast_dual_update_template_mix"}


def uses_critic_select_policy(method: str) -> bool:
    return method in {
        "bamor_v7_contrast_critic_dual_template_mix",
        "bamor_v8_contrast_dual_update_template_mix",
    }


def uses_preference_relabeling(method: str) -> bool:
    return method in COMMON_TARGET_METHODS | EXTREME_PORTFOLIO_METHODS or method in {
        "bamor_v11_contrast_relabel_mix",
        "bamor_v12_adaptive_operator_mix",
        "bamor_v13_block_operator_mix",
        "bamor_v14_adaptive_envelope_mix",
        "bamor_v15_adaptive_policy_envelope_mix",
        "bamor_v16_risk_adaptive_mix",
        "bamor_v17_archive_risk_mix",
        "bamor_v18_coverage_archive_mix",
        "bamor_v19_hybrid_archive_mix",
        "bamor_v20_hier_archive_mix",
        "bamor_v21_covbias_hier_archive_mix",
        "bamor_v22_dense_hier_archive_mix",
        "bamor_v23_extreme_hier_archive_mix",
    }


def uses_adaptive_operator_selection(method: str) -> bool:
    return method in COMMON_TARGET_METHODS | EXTREME_PORTFOLIO_METHODS or method in {
        "bamor_v12_adaptive_operator_mix",
        "bamor_v13_block_operator_mix",
        "bamor_v14_adaptive_envelope_mix",
        "bamor_v15_adaptive_policy_envelope_mix",
        "bamor_v16_risk_adaptive_mix",
        "bamor_v17_archive_risk_mix",
        "bamor_v18_coverage_archive_mix",
        "bamor_v19_hybrid_archive_mix",
        "bamor_v20_hier_archive_mix",
        "bamor_v21_covbias_hier_archive_mix",
        "bamor_v22_dense_hier_archive_mix",
        "bamor_v23_extreme_hier_archive_mix",
    }


def uses_block_adaptive_operator_selection(method: str) -> bool:
    return method in (COMMON_TARGET_METHODS - EPISODIC_ALLOCATION_METHODS) | EXTREME_PORTFOLIO_METHODS or method in {
        "bamor_v13_block_operator_mix",
        "bamor_v14_adaptive_envelope_mix",
        "bamor_v15_adaptive_policy_envelope_mix",
        "bamor_v16_risk_adaptive_mix",
        "bamor_v17_archive_risk_mix",
        "bamor_v18_coverage_archive_mix",
        "bamor_v19_hybrid_archive_mix",
        "bamor_v20_hier_archive_mix",
        "bamor_v21_covbias_hier_archive_mix",
        "bamor_v22_dense_hier_archive_mix",
        "bamor_v23_extreme_hier_archive_mix",
    }


def uses_envelope_adaptive_operator_selection(method: str) -> bool:
    return method in EXTREME_PORTFOLIO_METHODS or method in {
        "bamor_v14_adaptive_envelope_mix",
        "bamor_v15_adaptive_policy_envelope_mix",
        "bamor_v16_risk_adaptive_mix",
        "bamor_v17_archive_risk_mix",
        "bamor_v18_coverage_archive_mix",
        "bamor_v19_hybrid_archive_mix",
        "bamor_v20_hier_archive_mix",
        "bamor_v21_covbias_hier_archive_mix",
        "bamor_v22_dense_hier_archive_mix",
        "bamor_v23_extreme_hier_archive_mix",
    }


def uses_policy_envelope_adaptive_operator_selection(method: str) -> bool:
    return method in EXTREME_PORTFOLIO_METHODS or method in {
        "bamor_v15_adaptive_policy_envelope_mix",
        "bamor_v16_risk_adaptive_mix",
        "bamor_v17_archive_risk_mix",
        "bamor_v18_coverage_archive_mix",
        "bamor_v19_hybrid_archive_mix",
        "bamor_v20_hier_archive_mix",
        "bamor_v21_covbias_hier_archive_mix",
        "bamor_v22_dense_hier_archive_mix",
        "bamor_v23_extreme_hier_archive_mix",
    }


def uses_risk_adaptive_operator_selection(method: str) -> bool:
    return method in EXTREME_PORTFOLIO_METHODS or method in {
        "bamor_v16_risk_adaptive_mix",
        "bamor_v17_archive_risk_mix",
        "bamor_v18_coverage_archive_mix",
        "bamor_v19_hybrid_archive_mix",
        "bamor_v20_hier_archive_mix",
        "bamor_v21_covbias_hier_archive_mix",
        "bamor_v22_dense_hier_archive_mix",
        "bamor_v23_extreme_hier_archive_mix",
    }


def uses_policy_archive(method: str) -> bool:
    return method in COMMON_TARGET_METHODS | EXTREME_PORTFOLIO_METHODS or method in {
        "bamor_v17_archive_risk_mix",
        "bamor_v18_coverage_archive_mix",
        "bamor_v19_hybrid_archive_mix",
        "bamor_v20_hier_archive_mix",
        "bamor_v21_covbias_hier_archive_mix",
        "bamor_v22_dense_hier_archive_mix",
        "bamor_v23_extreme_hier_archive_mix",
    }


def policy_archive_interval_for_method(method: str) -> int:
    if method in EXTREME_PORTFOLIO_METHODS or method in {
        "bamor_v22_dense_hier_archive_mix",
    }:
        return 2_500
    return 5_000 if method in {
        "bamor_v18_coverage_archive_mix",
        "bamor_v19_hybrid_archive_mix",
        "bamor_v20_hier_archive_mix",
        "bamor_v21_covbias_hier_archive_mix",
        "bamor_v22_dense_hier_archive_mix",
        "bamor_v23_extreme_hier_archive_mix",
    } else 10_000


def policy_archive_limit_for_method(method: str) -> int:
    if method in EXTREME_PORTFOLIO_METHODS or method in {
        "bamor_v22_dense_hier_archive_mix",
    }:
        return 20
    return 10 if method in {
        "bamor_v18_coverage_archive_mix",
        "bamor_v19_hybrid_archive_mix",
        "bamor_v20_hier_archive_mix",
        "bamor_v21_covbias_hier_archive_mix",
        "bamor_v22_dense_hier_archive_mix",
        "bamor_v23_extreme_hier_archive_mix",
    } else 6


def moe_expert_temperature_for_method(method: str) -> float:
    if method == "bamor_v5_contrast_moe_sharp_template_mix":
        return 0.04
    return 0.08


def context_dim_for_method(method: str, obs_dim: int, act_dim: int,
                           reward_dim: int, num_modes: int) -> int:
    dim = reward_dim
    if uses_transition_context(method):
        dim += 1 + reward_dim
    if uses_generic_context(method):
        dim += 2 * obs_dim + act_dim + reward_dim
    if uses_learned_context(method):
        dim += LATENT_CONTEXT_DIM
    if uses_sequence_context(method):
        dim += LATENT_CONTEXT_DIM
    if uses_proto_context(method):
        dim += num_modes + 2
    return dim


def proto_feature_dim_for_method(method: str, obs_dim: int, act_dim: int,
                                 reward_dim: int) -> int:
    dim = 0
    if uses_transition_context(method):
        dim += 1 + reward_dim
    if uses_generic_context(method):
        dim += 2 * obs_dim + act_dim + reward_dim
    if uses_learned_context(method):
        dim += LATENT_CONTEXT_DIM
    if uses_sequence_context(method):
        dim += LATENT_CONTEXT_DIM
    return dim


def build_mujoco_context(method: str, weight, trans_tracker=None,
                         generic_tracker=None, learned_tracker=None,
                         seq_tracker=None, proto_tracker=None):
    parts = [np.asarray(weight, dtype=np.float32)]
    if uses_transition_context(method):
        parts.append(trans_tracker.features())
    if uses_generic_context(method):
        parts.append(generic_tracker.features())
    if uses_learned_context(method):
        parts.append(learned_tracker.features())
    if uses_sequence_context(method):
        parts.append(seq_tracker.features())
    if uses_proto_context(method):
        parts.append(proto_tracker.features())
    return np.concatenate(parts).astype(np.float32)


def mujoco_proto_feature(method: str, trans_tracker=None, generic_tracker=None,
                         learned_tracker=None, seq_tracker=None):
    parts = []
    if uses_transition_context(method):
        parts.append(trans_tracker.features())
    if uses_generic_context(method):
        parts.append(generic_tracker.features())
    if uses_learned_context(method):
        parts.append(learned_tracker.features())
    if uses_sequence_context(method):
        parts.append(seq_tracker.features())
    return np.concatenate(parts).astype(np.float32)


def make_detector_bamor(num_modes: int, reward_dim: int, gamma: float):
    return BAMORPipeline(
        num_modes=num_modes,
        reward_dim=reward_dim,
        gamma=gamma,
        surprise_threshold=18.0,
        concentration=15.0,
        switch_cooldown=2000,
        warmup=2000,
        disable_cone=True,
    )


def reference_point_for_env(env_name: str, reward_dim: int):
    if "mountaincarcontinuous" in env_name:
        return np.array([-1100.0, -100.0], dtype=np.float64)
    if "halfcheetah" in env_name:
        return np.array([-50.0, -3000.0], dtype=np.float64)
    if reward_dim == 3 and any(k in env_name for k in ("hopper", "walker2d", "ant")):
        # Ant uses x/y/control components, each with survival/contact bonuses.
        # This fixed floor keeps HV comparable across methods within an env.
        return np.array([-50.0, -100.0, -3000.0], dtype=np.float64)
    if reward_dim == 2:
        return np.array([-1e3, -1e3], dtype=np.float64)
    if reward_dim == 3:
        return np.array([-1e3, -1e3, -1e4], dtype=np.float64)
    return None


def compute_hv_nd(returns, ref_point):
    """Dominated hypervolume for the 2-D/3-D MuJoCo fronts we evaluate."""
    pts = np.asarray(returns, dtype=np.float64)
    ref = np.asarray(ref_point, dtype=np.float64)
    if pts.size == 0 or ref_point is None:
        return None
    if pts.ndim != 2 or pts.shape[1] != ref.shape[0]:
        raise ValueError(f"HV shape mismatch: points={pts.shape}, ref={ref.shape}")
    if pts.shape[1] == 2:
        return compute_hv_2d(pts, ref)
    if pts.shape[1] != 3:
        return None

    pts = pts[np.all(pts > ref, axis=1)]
    if len(pts) == 0:
        return 0.0

    z_levels = np.unique(np.concatenate(([ref[2]], pts[:, 2])))
    z_levels.sort()
    hv = 0.0
    for z_lo, z_hi in zip(z_levels[:-1], z_levels[1:]):
        if z_hi <= z_lo:
            continue
        active = pts[pts[:, 2] >= z_hi]
        if len(active) == 0:
            continue
        hv += compute_hv_2d(active[:, :2], ref[:2]) * (z_hi - z_lo)
    return float(hv)


# ================================================================
# Evaluation
# ================================================================

def full_simplex_grid(reward_dim):
    if reward_dim == 2:
        w = np.linspace(0, 1, 21, dtype=np.float32)
        return np.stack([w, 1 - w], axis=-1)
    if reward_dim == 3:
        return np.asarray([(i / 5, j / 5, (5 - i - j) / 5)
                           for i in range(6) for j in range(6 - i)], dtype=np.float32)
    raise ValueError("The matched protocol supports the studied 2-D and 3-D tasks")


def evaluate_locked_modes(env, evaluate_front, seed, ref_point):
    """Compare policy sets within a mode, never union returns across modes."""
    start = time.time()
    weights = full_simplex_grid(env.reward_dim)
    result = {"evaluation_protocol": "locked_full_simplex_v1",
              "evaluation_weights": weights.tolist(), "episode_seeds_per_mode": [],
              "ref_point": ref_point.tolist(), "pareto_per_mode": [],
              "hv_per_mode": [], "final_hv_per_mode": [], "eum_per_mode": []}
    previous_mode, previous_lock = env.current_mode_idx, env._lock_mode
    try:
        env._lock_mode = True
        for mode in range(env.num_modes):
            env._apply_mode(mode)
            seeds = [100000 + int(seed) * 1000 + mode * 10 + i for i in range(3)]
            front = evaluate_front(weights, seeds)
            returns = np.asarray([point["return"] for point in front])
            final_step = max(point.get("archive_step", 0) for point in front)
            final_returns = np.asarray([point["return"] for point in front
                                        if point.get("archive_step", 0) == final_step])
            result["episode_seeds_per_mode"].append(seeds)
            result["pareto_per_mode"].append(front)
            result["hv_per_mode"].append(compute_hv_nd(returns, ref_point))
            result["final_hv_per_mode"].append(compute_hv_nd(final_returns, ref_point))
            result["eum_per_mode"].append(float(np.max(returns @ weights.T, axis=0).mean()))
            print(f"Matched mode {mode}: archive HV={result['hv_per_mode'][-1]:.1f}, "
                  f"final HV={result['final_hv_per_mode'][-1]:.1f}", flush=True)
    finally:
        env._apply_mode(previous_mode)
        env._lock_mode = previous_lock
    result["hv_mean_mode"] = float(np.mean(result["hv_per_mode"]))
    result["final_hv_mean_mode"] = float(np.mean(result["final_hv_per_mode"]))
    result["evaluation_seconds"] = time.time() - start
    return result


def evaluate_pareto_with_weights(env, trainer, weights, num_episodes=3, device="cpu",
                                 episode_seeds=None):
    """Evaluate policy on the *given* weight grid → approximate Pareto front."""
    weights = np.asarray(weights, dtype=np.float32)
    results = []
    for w in weights:
        ep_returns = []
        for episode in range(num_episodes):
            reset_seed = None if episode_seeds is None else int(episode_seeds[episode])
            obs, _ = env.reset(seed=reset_seed)
            total_r = np.zeros(trainer.reward_dim)
            done = False
            steps = 0
            while not done and steps < 1000:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    w_t = torch.FloatTensor(w).unsqueeze(0).to(device)
                    action = trainer.select_action(obs_t, w_t, deterministic=True)
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


def evaluate_morl_agent_with_weights(env, agent, reward_dim, weights, num_episodes=3,
                                     episode_seeds=None):
    """Evaluate a morl-baselines-style agent exposing .eval(obs, w)."""
    weights = np.asarray(weights, dtype=np.float32)
    results = []
    for w in weights:
        ep_returns = []
        for episode in range(num_episodes):
            reset_seed = None if episode_seeds is None else int(episode_seeds[episode])
            obs, _ = env.reset(seed=reset_seed)
            total_r = np.zeros(reward_dim)
            done = False
            steps = 0
            while not done and steps < 1000:
                action = agent.eval(obs, w)
                if isinstance(action, tuple):
                    action = action[0]
                obs, rew, term, trunc, _ = env.step(np.asarray(action).flatten())
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


def evaluate_pareto_v2(env, trainer, method, encoder, weights,
                       num_episodes=3, device="cpu", gamma=0.99, episode_seeds=None):
    """Evaluate context-conditioned v2 methods with online trackers."""
    weights = np.asarray(weights, dtype=np.float32)
    reward_dim = trainer.reward_dim
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    results = []

    for w in weights:
        ep_returns = []
        for episode in range(num_episodes):
            reset_seed = None if episode_seeds is None else int(episode_seeds[episode])
            obs, _ = env.reset(seed=reset_seed)
            trans_tracker = (TransitionContextTracker(reward_dim)
                             if uses_transition_context(method) else None)
            generic_tracker = (GenericTransitionContextTracker(
                obs_dim, act_dim, reward_dim)
                if uses_generic_context(method) else None)
            learned_tracker = (LearnedTransitionContextTracker(
                encoder, latent_dim=LATENT_CONTEXT_DIM)
                if uses_learned_context(method) else None)
            seq_tracker = (SequenceTransitionContextTracker(
                encoder, V3_SEQ_LEN, obs_dim, act_dim, reward_dim,
                latent_dim=LATENT_CONTEXT_DIM)
                if uses_sequence_context(method) else None)
            proto_tracker = None
            detector = None
            if uses_proto_context(method):
                detector = make_detector_bamor(env.num_modes, reward_dim, gamma)
                proto_tracker = RegimePrototypeBeliefTracker(
                    env.num_modes,
                    proto_feature_dim_for_method(
                        method, obs_dim, act_dim, reward_dim))

            total_r = np.zeros(reward_dim)
            done = False
            steps = 0
            while not done and steps < 1000:
                c = build_mujoco_context(
                    method, w, trans_tracker=trans_tracker,
                    generic_tracker=generic_tracker,
                    learned_tracker=learned_tracker,
                    seq_tracker=seq_tracker,
                    proto_tracker=proto_tracker)
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    c_t = torch.FloatTensor(c).unsqueeze(0).to(device)
                    action = trainer.select_action(
                        obs_t, c_t, deterministic=True)
                    if isinstance(action, tuple):
                        action = action[0]
                action_np = action.cpu().numpy().flatten()
                next_obs, reward, term, trunc, _ = env.step(action_np)
                total_r += reward

                detector_info = None
                if detector is not None:
                    detector_info = detector.observe_reward(reward)
                if trans_tracker is not None:
                    trans_tracker.update(obs, action_np, reward, next_obs)
                if generic_tracker is not None:
                    generic_tracker.update(obs, action_np, reward, next_obs)
                if learned_tracker is not None:
                    learned_tracker.update(obs, action_np, reward, next_obs, device)
                if seq_tracker is not None:
                    seq_tracker.update(obs, action_np, reward, next_obs, device)
                if proto_tracker is not None:
                    proto_tracker.update(
                        mujoco_proto_feature(
                            method, trans_tracker=trans_tracker,
                            generic_tracker=generic_tracker,
                            learned_tracker=learned_tracker,
                            seq_tracker=seq_tracker),
                        detector_label=int(detector.belief_tracker.current_mode),
                        mode_switched=bool(
                            detector_info and detector_info.get("mode_switched")),
                    )
                obs = next_obs
                done = term or trunc
                steps += 1
            ep_returns.append(total_r)
        results.append({"weight": w.tolist(),
                        "return": np.mean(ep_returns, axis=0).tolist()})
    return results


def ensure_wandb_stub():
    """Provide a no-op wandb module for morl-baselines when logging is disabled."""
    try:
        __import__("wandb")
        return
    except ImportError:
        pass

    stub = types.ModuleType("wandb")
    stub.config = {}
    stub.log = lambda *args, **kwargs: None
    stub.init = lambda *args, **kwargs: None
    stub.finish = lambda *args, **kwargs: None
    stub.define_metric = lambda *args, **kwargs: None
    stub.watch = lambda *args, **kwargs: None
    stub.Table = lambda *args, **kwargs: None
    sys.modules["wandb"] = stub


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
    training_probe=None,
    update_router=None,
    archive_steps=None,
    resume_state=None,
    state_probe=None,
    trainer_factory=None,
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
    num_modes = int(env.num_modes)
    print(f"[{method}] env={env_name}, obs={obs_dim}, act={act_dim}, rew={reward_dim}, modes={num_modes}")

    context_dim = context_dim_for_method(
        method, obs_dim, act_dim, reward_dim, num_modes)

    # --- Trainer ---
    if method in {"bamor_common_target_projected_fixed", "bamor_common_target_projected_adaptive"} | EPISODIC_ALLOCATION_METHODS:
        trainer_class = ProjectedCommonTargetMOSACTrainer
    elif method in COMMON_TARGET_METHODS:
        trainer_class = CommonTargetMOSACTrainer
    elif method in V28_MUJOCO_METHODS:
        trainer_class = MeanValueMOSACTrainer
    elif method in V27_MUJOCO_METHODS:
        trainer_class = PreferenceAlignedTargetMOSACTrainer
    elif method in V26_MUJOCO_METHODS:
        trainer_class = AlignedEnvelopeMOSACTrainer
    else:
        trainer_class = (ConsistentTargetMOSACTrainer
                         if method in V24_MUJOCO_METHODS | V25_MUJOCO_METHODS else MOSACTrainer)
    trainer = (trainer_factory or trainer_class)(
        obs_dim, act_dim, reward_dim, hidden=hidden,
        lr=lr, gamma=gamma, batch_size=batch_size, device=device,
        context_dim=context_dim,
        reward_normalization=uses_reward_normalization(method),
        expert_policy=uses_expert_policy(method),
        soft_expert_policy=uses_soft_expert_policy(method),
        moe_expert_policy=uses_moe_expert_policy(method),
        moe_expert_temperature=moe_expert_temperature_for_method(method),
        moe_head_policy=uses_moe_head_policy(method),
        pairwise_residual_policy=method in {
            "bamor_v25_pairwise_progress_archive_mix",
            "bamor_v25_pairwise_progress_fixed_archive_mix",
        },
        dual_branch_policy=uses_dual_branch_policy(method),
        dual_branch_update=uses_dual_branch_update(method),
        critic_select_policy=uses_critic_select_policy(method),
        preference_relabeling=uses_preference_relabeling(method),
        preference_relabeling_rate=0.0 if uses_adaptive_operator_selection(method) else 1.0,
    )
    dev = trainer.device

    encoder = None
    encoder_opt = None
    if uses_learned_context(method):
        encoder = LatentTransitionEncoder(
            obs_dim, act_dim, reward_dim, latent_dim=LATENT_CONTEXT_DIM,
            hidden=hidden,
        ).to(dev)
        encoder_opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    elif uses_sequence_context(method):
        encoder = SequenceTransitionEncoder(
            obs_dim, act_dim, reward_dim, latent_dim=LATENT_CONTEXT_DIM,
            hidden=hidden,
        ).to(dev)
        encoder_opt = torch.optim.Adam(encoder.parameters(), lr=lr)

    # --- BAMOR Pipeline ---
    # gpt diagnosis #1+#2: in MuJoCo, threshold=6 caused 134 spurious switches
    # per 20K steps. We now use threshold=18 + CUSUM-style required_consecutive=5
    # (defaults inside BeliefTracker), which empirically gave 0 spurious /
    # correct detection at the real switch point.  switch_cooldown is also
    # bumped from default 80 → 2000 because mujoco episodes are ~1000 steps
    # and the policy reward drift is far slower than 1D-synthetic dynamics.
    bamor = None
    if method in OLD_BAMOR_METHODS:
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
    elif method in DETECTOR_CONTEXT_METHODS:
        bamor = make_detector_bamor(num_modes, reward_dim, gamma)

    trans_tracker = (TransitionContextTracker(reward_dim)
                     if uses_transition_context(method) else None)
    generic_tracker = (GenericTransitionContextTracker(obs_dim, act_dim, reward_dim)
                       if uses_generic_context(method) else None)
    learned_tracker = (LearnedTransitionContextTracker(
        encoder, latent_dim=LATENT_CONTEXT_DIM)
        if uses_learned_context(method) else None)
    seq_tracker = (SequenceTransitionContextTracker(
        encoder, V3_SEQ_LEN, obs_dim, act_dim, reward_dim,
        latent_dim=LATENT_CONTEXT_DIM)
        if uses_sequence_context(method) else None)
    proto_tracker = (RegimePrototypeBeliefTracker(
        num_modes,
        proto_feature_dim_for_method(method, obs_dim, act_dim, reward_dim))
        if uses_proto_context(method) else None)
    if method in FORECAST_EPISODIC_METHODS:
        credit_rule, adaptive = FORECAST_EPISODIC_METHODS[method]
        adaptive_selector = ForecastEpisodeReturnSamplingSelector(reward_dim,
            learning_starts=learning_starts, credit_rule=credit_rule, adaptive=adaptive)
    elif method in EPISODIC_ALLOCATION_METHODS:
        selector_class = (FixedEpisodeReturnSamplingSelector if method.endswith("_fixed")
                          else EpisodeReturnSamplingSelector)
        adaptive_selector = selector_class(reward_dim, learning_starts=learning_starts)
    elif method in COMMON_TARGET_METHODS:
        selector_class = (FixedCommonTargetSamplingSelector if method.endswith("_fixed")
                          else CommonTargetSamplingSelector)
        adaptive_selector = selector_class(reward_dim, learning_starts=learning_starts)
    elif method in {"bamor_v25_ownpref_both_progress_fixed_archive_mix",
                  "bamor_v25_pairwise_progress_fixed_archive_mix"}:
        adaptive_selector = OwnPreferenceBothFixedProgressExtremeOperatorSelector(
            reward_dim, learning_starts=learning_starts)
    elif method in {"bamor_v25_ownpref_both_progress_archive_mix",
                    "bamor_v25_pairwise_progress_archive_mix"}:
        adaptive_selector = OwnPreferenceBothProgressExtremeOperatorSelector(
            reward_dim, learning_starts=learning_starts)
    elif method == "bamor_v25_ownpref_actor_progress_archive_mix":
        adaptive_selector = OwnPreferenceProgressExtremeOperatorSelector(
            reward_dim, learning_starts=learning_starts)
    elif method == "bamor_v25_ownpref_actor_progress_fixed_archive_mix":
        adaptive_selector = OwnPreferenceFixedProgressExtremeOperatorSelector(
            reward_dim, learning_starts=learning_starts)
    elif method in {"bamor_v25_progress_archive_mix",
                  "bamor_v27_aligned_target_progress_archive_mix"}:
        adaptive_selector = ProgressExtremeHierarchicalOperatorSelector(
            reward_dim, learning_starts=learning_starts)
    elif method in {"bamor_v25_progress_fixed_archive_mix",
                    "bamor_v27_aligned_target_progress_fixed_archive_mix"}:
        adaptive_selector = FixedProgressExtremeHierarchicalOperatorSelector(
            reward_dim, learning_starts=learning_starts)
    elif method in {"bamor_v24_ownpref_actor_fixed_archive_mix",
                    "bamor_v27_aligned_target_ownpref_actor_fixed_archive_mix"}:
        adaptive_selector = OwnPreferenceFixedExtremeOperatorSelector(reward_dim)
    elif method == "bamor_v24_ownpref_critic_fixed_archive_mix":
        adaptive_selector = OwnPreferenceCriticFixedExtremeOperatorSelector(reward_dim)
    elif method == "bamor_v24_ownpref_both_fixed_archive_mix":
        adaptive_selector = OwnPreferenceBothFixedExtremeOperatorSelector(reward_dim)
    elif method in {"bamor_v23_fixed_selector_archive_mix",
                   "bamor_v24_target_consistent_fixed_archive_mix",
                   "bamor_v26_aligned_actor_fixed_archive_mix",
                   "bamor_v27_aligned_target_fixed_archive_mix",
                   "bamor_v28_mean_value_fixed_archive_mix"}:
        adaptive_selector = FixedExtremeHierarchicalOperatorSelector(reward_dim)
    elif method in {"bamor_v23_extreme_hier_archive_mix",
                    "bamor_v24_target_consistent_archive_mix"}:
        adaptive_selector = ExtremeHierarchicalArchiveRiskAdaptiveOperatorSelector(reward_dim)
    elif method == "bamor_v21_covbias_hier_archive_mix":
        adaptive_selector = CoverageBiasedHierarchicalArchiveRiskAdaptiveOperatorSelector(reward_dim)
    elif method in {
        "bamor_v20_hier_archive_mix",
        "bamor_v22_dense_hier_archive_mix",
    }:
        adaptive_selector = HierarchicalArchiveRiskAdaptiveOperatorSelector(reward_dim)
    elif method == "bamor_v19_hybrid_archive_mix":
        adaptive_selector = HybridArchiveRiskAdaptiveOperatorSelector(reward_dim)
    elif method == "bamor_v18_coverage_archive_mix":
        adaptive_selector = CoverageArchiveRiskAdaptiveOperatorSelector(reward_dim)
    elif uses_risk_adaptive_operator_selection(method):
        adaptive_selector = RiskAwarePolicyEnvelopeBlockAdaptiveOperatorSelector(reward_dim)
    elif uses_policy_envelope_adaptive_operator_selection(method):
        adaptive_selector = PolicyEnvelopeBlockAdaptiveOperatorSelector(reward_dim)
    elif uses_envelope_adaptive_operator_selection(method):
        adaptive_selector = EnvelopeBlockAdaptiveOperatorSelector(reward_dim)
    elif uses_block_adaptive_operator_selection(method):
        adaptive_selector = BlockAdaptiveOperatorSelector(reward_dim)
    elif uses_adaptive_operator_selection(method):
        adaptive_selector = AdaptiveOperatorSelector(reward_dim)
    else:
        adaptive_selector = None

    # --- Logging ---
    log_data = {
        "method": method, "env": env_name, "seed": seed,
        "total_steps": total_steps, "switch_interval": switch_interval,
        "steps": [], "episode_rewards": [], "mode_switches": [],
        "pareto_evals": [], "bamor_metrics": [],
        "ground_truth": env.get_ground_truth(),
        "context_dim": context_dim,
        "reward_normalization": uses_reward_normalization(method),
        "expert_policy": uses_expert_policy(method),
        "soft_expert_policy": uses_soft_expert_policy(method),
        "moe_expert_policy": uses_moe_expert_policy(method),
        "moe_head_policy": uses_moe_head_policy(method),
        "dual_branch_policy": uses_dual_branch_policy(method),
        "dual_branch_update": uses_dual_branch_update(method),
        "critic_select_policy": uses_critic_select_policy(method),
        "preference_relabeling": uses_preference_relabeling(method),
        "adaptive_operator_selection": uses_adaptive_operator_selection(method),
        "block_adaptive_operator_selection": uses_block_adaptive_operator_selection(method),
        "envelope_adaptive_operator_selection": uses_envelope_adaptive_operator_selection(method),
        "policy_envelope_adaptive_operator_selection": uses_policy_envelope_adaptive_operator_selection(method),
        "risk_adaptive_operator_selection": uses_risk_adaptive_operator_selection(method),
        "policy_archive": uses_policy_archive(method),
        "critic_target_rule": trainer.critic_target_rule,
        "bootstrap_terminal_rule": trainer.bootstrap_terminal_rule,
        "critic_candidate_context_rule": trainer.critic_candidate_context_rule,
    }
    if method in CONTEXT_MUJOCO_METHODS:
        log_data["v2_context"] = {
            "transition_context": uses_transition_context(method),
            "generic_context": uses_generic_context(method),
            "learned_latent_dim": LATENT_CONTEXT_DIM if uses_learned_context(method) else 0,
            "sequence_latent_dim": LATENT_CONTEXT_DIM if uses_sequence_context(method) else 0,
            "sequence_len": V3_SEQ_LEN if uses_sequence_context(method) else 0,
            "prototype_belief": uses_proto_context(method),
            "stratified_weights": uses_stratified_weights(method),
            "mixed_weights": uses_mixed_weights(method),
            "template_mixed_weights": uses_template_mixed_weights(method),
            "lattice_mixed_weights": uses_lattice_mixed_weights(method),
            "expert_policy": uses_expert_policy(method),
            "soft_expert_policy": uses_soft_expert_policy(method),
            "moe_expert_policy": uses_moe_expert_policy(method),
            "moe_head_policy": uses_moe_head_policy(method),
            "dual_branch_policy": uses_dual_branch_policy(method),
            "dual_branch_update": uses_dual_branch_update(method),
            "critic_select_policy": uses_critic_select_policy(method),
            "preference_relabeling": uses_preference_relabeling(method),
            "adaptive_operator_selection": uses_adaptive_operator_selection(method),
            "block_adaptive_operator_selection": uses_block_adaptive_operator_selection(method),
            "envelope_adaptive_operator_selection": uses_envelope_adaptive_operator_selection(method),
            "policy_envelope_adaptive_operator_selection": uses_policy_envelope_adaptive_operator_selection(method),
            "risk_adaptive_operator_selection": uses_risk_adaptive_operator_selection(method),
            "policy_archive": uses_policy_archive(method),
        }

    policy_archive = []
    policy_archive_interval = policy_archive_interval_for_method(method)
    policy_archive_limit = policy_archive_limit_for_method(method)
    if archive_steps is not None:
        archive_steps = set(archive_steps)
        assert min(archive_steps) >= learning_starts and max(archive_steps) == total_steps
        assert len(archive_steps) <= policy_archive_limit

    def clone_policy_state_cpu():
        return {
            k: v.detach().cpu().clone()
            for k, v in trainer.policy.state_dict().items()
        }

    def maybe_store_policy_archive(step_value: int, reason: str) -> None:
        if not uses_policy_archive(method):
            return
        if step_value < learning_starts:
            return
        if policy_archive and policy_archive[-1]["step"] == int(step_value):
            return
        snapshot = {
            "step": int(step_value),
            "reason": str(reason),
            "policy": clone_policy_state_cpu(),
        }
        if archive_steps is not None or update_router is not None:
            snapshot["interaction_step"] = (update_router.interaction_step(step_value)
                                            if update_router is not None else int(step_value))
        policy_archive.append(snapshot)
        while len(policy_archive) > policy_archive_limit:
            policy_archive.pop(0)

    # --- Training ---
    env.action_space.seed(seed)
    obs, info = env.reset(seed=seed)
    episode_reward = np.zeros(reward_dim)
    episode_steps = 0
    episode_count = 0
    mode_switch_count = 0
    last_mode = info["mode_idx"]

    start_time = time.time()
    run_config = dict(env_name=env_name, switch_interval=switch_interval, seed=seed,
                      hidden=hidden, lr=lr, gamma=gamma, batch_size=batch_size,
                      learning_starts=learning_starts, context_dim=context_dim)
    first_step = 1
    if resume_state is not None:
        assert str(dev) == "cpu" and resume_state["run_config"] == run_config
        env.close()
        eval_env.close()
        (trainer, encoder, encoder_opt, bamor, trans_tracker, generic_tracker,
         learned_tracker, seq_tracker, proto_tracker, adaptive_selector, env,
         eval_env, obs, episode_reward, episode_steps, episode_count,
         mode_switch_count, last_mode, log_data, policy_archive) = [resume_state[k] for k in (
             "trainer", "encoder", "encoder_opt", "bamor", "trans_tracker", "generic_tracker",
             "learned_tracker", "seq_tracker", "proto_tracker", "adaptive_selector", "env",
             "eval_env", "obs", "episode_reward", "episode_steps", "episode_count",
             "mode_switch_count", "last_mode", "log_data", "policy_archive")]
        log_data["method"] = method
        first_step = resume_state["step"] + 1
        start_time -= resume_state["training_seconds"]
        np.random.set_state(resume_state["numpy_rng"])
        torch.set_rng_state(resume_state["torch_rng"])

    for step in range(first_step, total_steps + 1):
        # Sample weight
        operator_name = None
        if adaptive_selector is not None:
            if uses_block_adaptive_operator_selection(method) or method in EPISODIC_ALLOCATION_METHODS:
                operator_name = adaptive_selector.select_for_step(step)
            else:
                operator_name = adaptive_selector.select()
            w_np, w_anchor_np = adaptive_selector.sample_weight(
                operator_name, reward_dim, step)
            trainer.preference_relabeling_rate = adaptive_selector.relabel_rate(
                operator_name)
            if hasattr(adaptive_selector, "trainer_config"):
                for key, value in adaptive_selector.trainer_config(operator_name).items():
                    setattr(trainer, key, value)
        elif method in OLD_BAMOR_METHODS:
            w_np, w_anchor_np = bamor.sample_weight()
        elif method in {"uniform", "uniform_norm"}:
            w_np, w_anchor_np = sample_uniform_weight(reward_dim)
        elif method == "fixed":
            w_np, w_anchor_np = sample_fixed_weight(reward_dim)
        elif uses_lattice_mixed_weights(method):
            if step % 2 == 0:
                w_np, w_anchor_np = sample_lattice_weight(reward_dim, step // 2)
            else:
                w_np, w_anchor_np = sample_uniform_weight(reward_dim)
        elif uses_template_mixed_weights(method):
            if step % 2 == 0:
                w_np, w_anchor_np = sample_expert_template_weight(reward_dim, step // 2)
            else:
                w_np, w_anchor_np = sample_uniform_weight(reward_dim)
        elif uses_stratified_weights(method):
            w_np, w_anchor_np = sample_stratified_weight(reward_dim, step)
        elif uses_mixed_weights(method):
            if step % 2 == 0:
                w_np, w_anchor_np = sample_stratified_weight(reward_dim, step // 2)
            else:
                w_np, w_anchor_np = sample_uniform_weight(reward_dim)
        else:
            w_np, w_anchor_np = sample_uniform_weight(reward_dim)
        if update_router is not None:
            update_router.configure(step, trainer)
        c_np = (build_mujoco_context(
            method, w_np, trans_tracker=trans_tracker,
            generic_tracker=generic_tracker,
            learned_tracker=learned_tracker,
            seq_tracker=seq_tracker,
            proto_tracker=proto_tracker)
            if method in CONTEXT_MUJOCO_METHODS else w_np)

        # Act
        if step < learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(dev)
                c_t = torch.FloatTensor(c_np).unsqueeze(0).to(dev)
                action, _ = trainer.select_action(obs_t, c_t)
                action = action.cpu().numpy().flatten()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if adaptive_selector is not None and operator_name is not None:
            scalar_reward = float(np.dot(w_np, reward))
            if isinstance(adaptive_selector, EpisodeReturnSamplingSelector):
                adaptive_selector.observe_transition(operator_name, reward, step, done)
            elif isinstance(adaptive_selector, ProgressExtremeHierarchicalOperatorSelector):
                adaptive_selector.observe_transition(operator_name, reward, step)
                adaptive_selector.maybe_finish_block(step)
            elif uses_block_adaptive_operator_selection(method):
                adaptive_selector.observe_step(operator_name, scalar_reward)
                adaptive_selector.maybe_finish_block(step)
            else:
                adaptive_selector.update(operator_name, scalar_reward)

        # BAMOR observe — capture belief BEFORE / AFTER to detect its own switch
        prev_bamor_mode = (bamor.belief_tracker.current_mode
                           if bamor is not None else None)
        bamor_info = None
        if bamor is not None:
            bamor_info = bamor.observe_reward(reward)
        cur_bamor_mode = (bamor.belief_tracker.current_mode
                          if bamor is not None else None)

        if method in DETECTOR_CONTEXT_METHODS and bamor_info and bamor_info.get("mode_switched"):
            if trans_tracker is not None:
                trans_tracker.reset()
            if generic_tracker is not None:
                generic_tracker.reset()
            if learned_tracker is not None:
                learned_tracker.reset()
            if seq_tracker is not None:
                seq_tracker.history.clear()
                seq_tracker.z[:] = 0.0
        if trans_tracker is not None:
            trans_tracker.update(obs, action, reward, next_obs)
        if generic_tracker is not None:
            generic_tracker.update(obs, action, reward, next_obs)
        if learned_tracker is not None:
            learned_tracker.update(obs, action, reward, next_obs, dev)
        if seq_tracker is not None:
            seq_tracker.update(obs, action, reward, next_obs, dev)
        if proto_tracker is not None:
            proto_tracker.update(
                mujoco_proto_feature(
                    method, trans_tracker=trans_tracker,
                    generic_tracker=generic_tracker,
                    learned_tracker=learned_tracker,
                    seq_tracker=seq_tracker),
                detector_label=int(bamor.belief_tracker.current_mode),
                mode_switched=bool(bamor_info and bamor_info.get("mode_switched")),
            )
        c_next_np = (build_mujoco_context(
            method, w_np, trans_tracker=trans_tracker,
            generic_tracker=generic_tracker,
            learned_tracker=learned_tracker,
            seq_tracker=seq_tracker,
            proto_tracker=proto_tracker)
            if method in CONTEXT_MUJOCO_METHODS else w_np)

        # Store after online context update so c_next matches next state.
        regime_label = (
            int(bamor.belief_tracker.current_mode)
            if method in DETECTOR_CONTEXT_METHODS and bamor is not None else -1
        )
        trainer.buffer.push(
            obs, action, w_np, c_np, reward, next_obs, c_next_np,
            trainer.replay_terminal(terminated, truncated),
            t=step, regime=regime_label)

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
            if uses_learned_context(method) and trainer.buffer.size >= batch_size:
                es, ea, er, esn, et, ey = trainer.buffer.sample_encoder(batch_size)
                er_s = torch.tanh(er / 3.0)
                ez, pred_r, pred_delta = encoder(es, ea, er_s, esn)
                target_delta = torch.tanh((esn - es) / 3.0)
                enc_loss = (
                    F.mse_loss(pred_r, er_s)
                    + F.mse_loss(pred_delta, target_delta)
                    + 1e-4 * ez.pow(2).mean()
                )
                ezp = encoder.project(ez)
                enc_loss = enc_loss + 0.2 * temporal_contrastive_loss(
                    ezp, et, pos_window=256, temperature=0.2)
                if method in V2_DETECTOR_METHODS:
                    enc_loss = enc_loss + 0.25 * pseudo_regime_contrastive_loss(
                        ezp, ey, temperature=0.2)
                encoder_opt.zero_grad()
                enc_loss.backward()
                encoder_opt.step()
            elif uses_sequence_context(method) and trainer.buffer.size >= max(batch_size, V3_SEQ_LEN):
                seq_batch = trainer.buffer.sample_encoder_sequence(batch_size, V3_SEQ_LEN)
                if seq_batch is not None:
                    es, ea, er, esn, et, ey = seq_batch
                    ez, pred_r, pred_delta = encoder(es, ea, er, esn)
                    target_r = torch.tanh(er[:, -1] / 3.0)
                    target_delta = torch.tanh((esn[:, -1] - es[:, -1]) / 3.0)
                    enc_loss = (
                        F.mse_loss(pred_r, target_r)
                        + F.mse_loss(pred_delta, target_delta)
                        + 1e-4 * ez.pow(2).mean()
                    )
                    ezp = encoder.project(ez)
                    enc_loss = enc_loss + 0.2 * temporal_contrastive_loss(
                        ezp, et, pos_window=512, temperature=0.2)
                    enc_loss = enc_loss + 0.25 * pseudo_regime_contrastive_loss(
                        ezp, ey, temperature=0.2)
                    encoder_opt.zero_grad()
                    enc_loss.backward()
                    encoder_opt.step()

        if update_router is not None:
            update_router.after_step(step, trainer, encoder, env, operator_name)

        if (uses_policy_archive(method)
                and step >= learning_starts
                and (step in archive_steps if archive_steps is not None
                     else step % policy_archive_interval == 0)):
            maybe_store_policy_archive(step, "periodic")

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

        # Capture after episode bookkeeping, before any evaluation consumes RNG.
        if state_probe is not None and step == state_probe.step:
            assert str(dev) == "cpu"
            stop_after_capture = state_probe(dict(
                step=step, run_config=run_config, training_seconds=time.time() - start_time,
                numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                trainer=trainer, encoder=encoder, encoder_opt=encoder_opt, bamor=bamor,
                trans_tracker=trans_tracker, generic_tracker=generic_tracker,
                learned_tracker=learned_tracker, seq_tracker=seq_tracker,
                proto_tracker=proto_tracker, adaptive_selector=adaptive_selector,
                env=env, eval_env=eval_env, obs=obs, episode_reward=episode_reward,
                episode_steps=episode_steps, episode_count=episode_count,
                mode_switch_count=mode_switch_count, last_mode=last_mode,
                log_data=log_data, policy_archive=policy_archive))
            # A non-stopping capture leaves the factual continuation uninterrupted.
            if stop_after_capture is not False:
                log_data["ground_truth"] = env.get_ground_truth()
                env.close()
                eval_env.close()
                return log_data

        # A diagnostic can inspect live optimizer/replay state without serializing it.
        if training_probe is not None and training_probe(
                step, trainer, encoder, env, operator_name):
            log_data["ground_truth"] = env.get_ground_truth()
            env.close()
            eval_env.close()
            return log_data

        # Evaluation
        if step % eval_freq == 0:
            if method in CONTEXT_MUJOCO_METHODS:
                if reward_dim == 2:
                    weight_grid = np.linspace(0, 1, 11)
                    weights = np.stack([weight_grid, 1 - weight_grid], axis=-1).astype(np.float32)
                else:
                    weights = np.random.dirichlet(np.ones(reward_dim), size=11).astype(np.float32)
                pareto = evaluate_pareto_v2(
                    eval_env, trainer, method, encoder, weights,
                    num_episodes=3, device=dev, gamma=gamma)
            else:
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
    training_seconds = time.time() - start_time
    GLOBAL_REF = reference_point_for_env(env_name, reward_dim)
    if uses_policy_archive(method):
        maybe_store_policy_archive(total_steps, "final")
        log_data["policy_archive_snapshots"] = [
            {"index": int(i), "step": int(snap["step"]), "reason": snap["reason"],
             **({"interaction_step": snap["interaction_step"]} if "interaction_step" in snap else {})}
            for i, snap in enumerate(policy_archive)
        ]

    def evaluate_current_policy(eval_env_obj, weights, num_episodes, episode_seeds=None):
        if method in CONTEXT_MUJOCO_METHODS:
            return evaluate_pareto_v2(
                eval_env_obj, trainer, method, encoder, weights,
                num_episodes=num_episodes, device=dev, gamma=gamma,
                episode_seeds=episode_seeds)
        return evaluate_pareto_with_weights(
            eval_env_obj, trainer, weights, num_episodes=num_episodes, device=dev,
            episode_seeds=episode_seeds)

    def evaluate_policy_archive(eval_env_obj, weights, num_episodes, episode_seeds=None):
        if not uses_policy_archive(method):
            return evaluate_current_policy(eval_env_obj, weights, num_episodes, episode_seeds)

        final_state = clone_policy_state_cpu()
        fronts = []
        eval_candidates = list(policy_archive)
        if not eval_candidates:
            eval_candidates = [{
                "step": int(total_steps),
                "reason": "final_fallback",
                "policy": final_state,
            }]
        for idx, snap in enumerate(eval_candidates):
            trainer.policy.load_state_dict(snap["policy"])
            front_i = evaluate_current_policy(eval_env_obj, weights, num_episodes, episode_seeds)
            for item in front_i:
                row = dict(item)
                row["archive_index"] = int(idx)
                row["archive_step"] = int(snap["step"])
                row["archive_reason"] = str(snap["reason"])
                if "interaction_step" in snap:
                    row["archive_interaction_step"] = snap["interaction_step"]
                fronts.append(row)
        trainer.policy.load_state_dict(final_state)
        return fronts

    log_data.update(evaluate_locked_modes(
        eval_env,
        lambda weights, seeds: evaluate_policy_archive(
            eval_env, weights, len(seeds), seeds),
        seed, GLOBAL_REF))
    log_data["archive_context"] = "final_encoder"
    log_data["training_seconds"] = training_seconds
    log_data["training_interactions"] = int(env._total_steps)
    log_data["evaluation_interactions"] = int(eval_env._total_steps)
    log_data["environment_initialization_probe_steps"] = 2
    log_data["learning_work"] = {
        **trainer.update_work, "encoder_optimizer_steps": optimizer_step_count(encoder_opt),
        "network_parameters": network_parameter_counts(trainer.policy, (trainer.q1, trainer.q2), encoder)}
    if method in COMMON_TARGET_METHODS:
        log_data["common_target_learning"] = dict(
            critic_loss_rule=trainer.critic_loss_rule, actor_update_rule=trainer.actor_update_rule,
            learning_preference_distribution="uniform_simplex", candidate_count=1,
            behavior_clone_coef=0.0, critic_relabel_copies=1)
    if resume_state is not None:
        log_data["continuation_from_step"] = first_step - 1
        log_data["training_interactions_this_call"] = total_steps - first_step + 1
    log_data["ground_truth"] = env.get_ground_truth()
    if update_router is not None:
        log_data["update_routing"] = update_router.state_dict()
    if method in V26_MUJOCO_METHODS | V28_MUJOCO_METHODS:
        log_data["actor_update"] = trainer.actor_update_diagnostics()
    if adaptive_selector is not None:
        if uses_block_adaptive_operator_selection(method):
            adaptive_selector.maybe_finish_block(total_steps, force=True)
        log_data["adaptive_operator_final"] = adaptive_selector.state_dict()

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
        "encoder": encoder.state_dict() if encoder is not None else None,
        "policy_archive": [
            {
                "step": int(snap["step"]),
                "reason": snap["reason"],
                "policy": snap["policy"],
            }
            for snap in policy_archive
        ] if uses_policy_archive(method) else None,
    }, model_path)

    env.close()
    eval_env.close()
    return log_data


# ================================================================
# CAPQL baseline (uses morl-baselines directly)
# ================================================================

class UniformSimplexWeightSampler:
    """Full-simplex control for CAPQL's hard-coded angle-sampler interface."""

    def __init__(self, rwd_dim, angle):
        self.distribution = torch.distributions.Dirichlet(torch.ones(rwd_dim))

    def sample(self, n_sample):
        return self.distribution.sample((n_sample,))


def capql_replica_index(method: str) -> int | None:
    if method in {"capql", "capql_archive", "capql_full_simplex"}:
        return 0
    for prefix in ("capql_rep", "capql_r"):
        if method.startswith(prefix):
            suffix = method[len(prefix):]
            if suffix.isdigit():
                return int(suffix)
    return None


def is_capql_method(method: str) -> bool:
    return capql_replica_index(method) is not None


def train_capql(env_name, total_steps, switch_interval, seed, save_dir,
                eval_freq=20_000, device="auto", method="capql",
                train_seed=None, eval_seed=None):
    """Train CAPQL from morl-baselines on piecewise-stationary env."""
    replica_idx = capql_replica_index(method)
    if replica_idx is None:
        raise ValueError(f"Not a CAPQL method: {method}")
    base_seed = int(seed)
    train_seed = int(base_seed + 10000 * replica_idx if train_seed is None else train_seed)
    eval_seed = int(base_seed + 1000 if eval_seed is None else eval_seed)

    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    random.seed(train_seed)
    ensure_wandb_stub()
    try:
        from morl_baselines.multi_policy.capql import capql as capql_module
        CAPQL = capql_module.CAPQL
    except ImportError:
        print("CAPQL not available in morl-baselines. Skipping.")
        return None

    env = make_piecewise_mo_env(env_name, switch_interval=switch_interval, seed=train_seed)
    eval_env = make_piecewise_mo_env(env_name, switch_interval=switch_interval, seed=eval_seed)
    # CAPQL samples random actions before learning_starts; Gym's action-space RNG is independent.
    env.action_space.seed(train_seed)

    ref_point = reference_point_for_env(env_name, env.reward_dim)

    archive_enabled = method == "capql_archive"
    archive_interval = 2_500
    archive_limit = 20

    class ArchivingCAPQL(CAPQL):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.policy_archive = []

        def store_policy(self):
            if self.policy_archive and self.policy_archive[-1]["step"] == self.global_step:
                return
            self.policy_archive.append({
                "step": int(self.global_step),
                "policy": {k: v.detach().cpu().clone()
                           for k, v in self.policy.state_dict().items()},
            })
            if len(self.policy_archive) > archive_limit:
                self.policy_archive.pop(0)

        def update(self):
            result = super().update()
            if self.global_step % archive_interval == 0:
                self.store_policy()
            return result

    agent = (ArchivingCAPQL if archive_enabled else CAPQL)(
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_size=500_000,
        net_arch=[256, 256],
        batch_size=256,
        alpha=0.2,
        learning_starts=5000,
        seed=train_seed,
        device=device,
        log=False,  # disable wandb
    )

    print(
        f"[{method}] Training on {env_name} for {total_steps} steps "
        f"(base_seed={base_seed}, train_seed={train_seed}, eval_seed={eval_seed})..."
    )
    training_start = time.time()
    # CAPQL constructs its sampler inside train(); isolate this experimental substitution.
    native_sampler = capql_module.WeightSamplerAngle
    if method == "capql_full_simplex":
        capql_module.WeightSamplerAngle = UniformSimplexWeightSampler
    try:
        agent.train(
            total_timesteps=total_steps,
            eval_env=eval_env,
            ref_point=ref_point,
            eval_freq=eval_freq,
            num_eval_weights_for_front=50,
            num_eval_episodes_for_front=3,
        )
    finally:
        capql_module.WeightSamplerAngle = native_sampler

    training_seconds = time.time() - training_start
    if archive_enabled:
        agent.store_policy()

    def evaluate_archive(weights, episode_seeds):
        if not archive_enabled:
            return evaluate_morl_agent_with_weights(
                eval_env, agent, env.reward_dim, weights, num_episodes=len(episode_seeds),
                episode_seeds=episode_seeds)
        final_state = {k: v.detach().cpu().clone()
                       for k, v in agent.policy.state_dict().items()}
        fronts = []
        for idx, snapshot in enumerate(agent.policy_archive):
            agent.policy.load_state_dict(snapshot["policy"])
            front = evaluate_morl_agent_with_weights(
                eval_env, agent, env.reward_dim, weights, num_episodes=len(episode_seeds),
                episode_seeds=episode_seeds)
            for point in front:
                fronts.append({**point, "archive_index": idx,
                               "archive_step": snapshot["step"]})
        agent.policy.load_state_dict(final_state)
        return fronts

    evaluation = evaluate_locked_modes(eval_env, evaluate_archive, base_seed, ref_point)

    # Save CAPQL results in the same schema used by BAMOR MuJoCo runs.
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, f"{method}_seed{base_seed}.json")
    with open(result_path, "w") as f:
        json.dump({
            "method": method, "env": env_name, "seed": base_seed,
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "capql_replica_index": replica_idx,
            "learning_work": {
                "critic_optimizer_steps": optimizer_step_count(agent.q_optim),
                "actor_optimizer_steps": optimizer_step_count(agent.policy_optim),
                "encoder_optimizer_steps": 0,
                "critic_rows": optimizer_step_count(agent.q_optim) * agent.batch_size,
                "target_candidate_rows": optimizer_step_count(agent.q_optim) * agent.batch_size,
                "actor_candidate_rows": optimizer_step_count(agent.policy_optim) * agent.batch_size,
                "network_parameters": network_parameter_counts(agent.policy, agent.q_nets)},
            "training_preference_sampler": (
                {"distribution": "Dirichlet", "concentration": [1.0] * env.reward_dim,
                 "support": "full_simplex"} if method == "capql_full_simplex" else
                {"distribution": "WeightSamplerAngle", "center": "equal",
                 "angle_degrees": 22.5}),
            "total_steps": total_steps,
            "ref_point": ref_point.tolist() if ref_point is not None else None,
            "policy_archive_snapshots": ([
                {"index": i, "step": snapshot["step"]}
                for i, snapshot in enumerate(agent.policy_archive)
            ] if archive_enabled else None),
            **evaluation,
            "training_seconds": training_seconds,
            "training_interactions": int(env._total_steps),
            "evaluation_interactions": int(eval_env._total_steps),
            "environment_initialization_probe_steps": 2,
            "switch_interval": switch_interval,
            "ground_truth": env.get_ground_truth(),
        }, f, indent=2, default=str)

    torch.save({
        "policy": agent.policy.state_dict(),
        "policy_archive": agent.policy_archive if archive_enabled else None,
    }, os.path.join(save_dir, f"{method}_seed{base_seed}_model.pt"))
    env.close()
    eval_env.close()
    print(f"[{method}] Done. Saved to {result_path}")


# ================================================================
# Entry point
# ================================================================

METHODS = [
    "bamor", "bamor_nc", "bamor_ns", "uniform", "uniform_norm", "fixed", "capql", "capql_archive",
    "capql_rep1", "capql_rep2", "capql_full_simplex",
    "bamor_common_target_adaptive", "bamor_common_target_fixed",
    "bamor_common_target_projected_adaptive", "bamor_common_target_projected_fixed",
    "bamor_common_target_episode_adaptive", "bamor_common_target_episode_fixed",
    *FORECAST_EPISODIC_METHODS,
    "bamor_v2_contrast_hybrid",
    "bamor_v2_proto_hybrid",
    "bamor_v2_proto_generic",
    "bamor_v4_contrast_expert_mix",
    "bamor_v4_contrast_expert_norm_mix",
    "bamor_v4_contrast_soft_expert_mix",
    "bamor_v5_contrast_expert_template_mix",
    "bamor_v5_contrast_moe_sharp_template_mix",
    "bamor_v5_contrast_moe_template_mix",
    "bamor_v6_contrast_moe_head_template_mix",
    "bamor_v7_contrast_critic_dual_template_mix",
    "bamor_v8_contrast_dual_update_template_mix",
    "bamor_v11_contrast_relabel_mix",
    "bamor_v12_adaptive_operator_mix",
    "bamor_v13_block_operator_mix",
    "bamor_v14_adaptive_envelope_mix",
    "bamor_v15_adaptive_policy_envelope_mix",
    "bamor_v16_risk_adaptive_mix",
    "bamor_v17_archive_risk_mix",
    "bamor_v18_coverage_archive_mix",
    "bamor_v19_hybrid_archive_mix",
    "bamor_v20_hier_archive_mix",
    "bamor_v21_covbias_hier_archive_mix",
    "bamor_v22_dense_hier_archive_mix",
    "bamor_v23_extreme_hier_archive_mix",
    "bamor_v23_fixed_selector_archive_mix",
    "bamor_v24_target_consistent_archive_mix",
    "bamor_v24_target_consistent_fixed_archive_mix",
    "bamor_v24_ownpref_actor_fixed_archive_mix",
    "bamor_v24_ownpref_critic_fixed_archive_mix",
    "bamor_v24_ownpref_both_fixed_archive_mix",
    "bamor_v25_progress_archive_mix",
    "bamor_v25_progress_fixed_archive_mix",
    "bamor_v25_ownpref_actor_progress_fixed_archive_mix",
    "bamor_v25_ownpref_actor_progress_archive_mix",
    "bamor_v25_ownpref_both_progress_fixed_archive_mix",
    "bamor_v25_ownpref_both_progress_archive_mix",
    "bamor_v25_pairwise_progress_archive_mix",
    "bamor_v25_pairwise_progress_fixed_archive_mix",
    "bamor_v26_aligned_actor_fixed_archive_mix",
    "bamor_v27_aligned_target_fixed_archive_mix",
    "bamor_v27_aligned_target_ownpref_actor_fixed_archive_mix",
    "bamor_v27_aligned_target_progress_archive_mix",
    "bamor_v27_aligned_target_progress_fixed_archive_mix",
    "bamor_v28_mean_value_fixed_archive_mix",
    "bamor_v3_contrast_norm_mix",
    "bamor_v3_contrast_lattice_mix",
    "bamor_v3_contrast_mix",
    "bamor_v3_contrast_stratified",
    "bamor_v3_seq_belief",
]

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

            if is_capql_method(method):
                train_capql(
                    args.env, args.total_steps, args.switch_interval,
                    seed, save_dir, args.eval_freq, args.device, method=method,
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
