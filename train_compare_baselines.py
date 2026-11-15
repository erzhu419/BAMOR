#!/usr/bin/env python3
"""
train_compare_baselines.py — Unified BAMOR vs. Baselines Comparison

Methods:
  Single-policy weight-conditioned (fair comparison):
    bamor       : BAMOR (Target Cone + belief update)
    bamor_nc    : BAMOR ablation — no Target Cone (uniform sampling)
    bamor_ns    : frozen-global Target Cone (no surprise detection)
    bamor_oracle: oracle mode + learned cone
    bamor_oracle_cone: oracle mode + oracle cone
    bamor_fixed_cone: learned mode + oracle cone
    bamor_ns_eps: frozen-global Target Cone + residual full-simplex sampling
    capql       : CAPQL (ICLR 2023), uniform weight sampling
    uniform     : uniform Dirichlet weight sampling
    fixed       : fixed weight [0.5, 0.5]

  Multi-policy weight-conditioned (extended comparison):
    gpi_pd      : GPI-PD continuous (NeurIPS 2023), maintains weight support set

All methods share:
  - Same SyntheticMOEnv with identical mode-switch schedule
  - Same total_timesteps budget
  - Same eval protocol: HV over 21 weight vectors, repeated 3 episodes
  - Same ref_point for HV (estimated from first run, shared across methods)

Usage:
  python train_compare_baselines.py --total_steps 50000 --seed 0
  python train_compare_baselines.py --total_steps 50000 --seeds 5 --method all
"""

import os, sys, json, argparse, math, copy
from collections import deque
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Auto-add morl-baselines to path — search up to 5 parent levels
def _find_morl_baselines():
    p = Path(__file__).resolve().parent
    for _ in range(5):
        candidate = p / 'morl-baselines'
        if candidate.exists():
            return candidate
        p = p.parent
    return None
_morl_path = _find_morl_baselines()
if _morl_path and str(_morl_path) not in sys.path:
    sys.path.insert(0, str(_morl_path))

from synthetic_mo_env import SyntheticMOEnv, ModeConfig
from bamor_core import BAMORPipeline
from train_synthetic_bamor import (
    SmallQNet, SmallPolicy, Buffer,
    compute_hv_2d, eval_pareto,
    create_env, get_canonical_modes,
)

# ── morl-baselines import guard ──────────────────────────────────────────────
MORL_BASELINES_PATH = _morl_path  # already resolved above


# ════════════════════════════════════════════════════════════════════════════
# mo-gymnasium wrapper  (required by CAPQL / GPI-PD)
# ════════════════════════════════════════════════════════════════════════════

import gymnasium as gym
from gymnasium import spaces

# Fixed conservative reference point for cross-method overall HV comparison.
# Synthetic env rewards are non-negative (Gaussian fns, ≥0 per step).
# GLOBAL_REF is set well below any policy's per-episode return so that
# even a random policy gets nonzero HV and comparisons are on a common scale.
GLOBAL_REF = (-5.0, -5.0)


class ContextBuffer:
    """Replay buffer storing both scalarization weights and policy contexts."""

    def __init__(self, cap, s_dim, a_dim, w_dim, c_dim, r_dim):
        self.cap, self.ptr, self.size = cap, 0, 0
        self.s = np.zeros((cap, s_dim), np.float32)
        self.a = np.zeros((cap, a_dim), np.float32)
        self.w = np.zeros((cap, w_dim), np.float32)
        self.c = np.zeros((cap, c_dim), np.float32)
        self.r = np.zeros((cap, r_dim), np.float32)
        self.sn = np.zeros((cap, s_dim), np.float32)
        self.cn = np.zeros((cap, c_dim), np.float32)
        self.d = np.zeros((cap, 1), np.float32)
        self.t = np.zeros((cap, 1), np.int64)
        self.y = np.full((cap, 1), -1, np.int64)

    def push(self, s, a, w, c, r, sn, cn, d, t=0, regime=-1):
        i = self.ptr
        self.s[i], self.a[i], self.w[i], self.c[i] = s, a, w, c
        self.r[i], self.sn[i], self.cn[i], self.d[i] = r, sn, cn, float(d)
        self.t[i] = int(t)
        self.y[i] = int(regime)
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n, device):
        idx = np.random.randint(0, self.size, n)
        to = lambda x: torch.FloatTensor(x[idx]).to(device)
        return (
            to(self.s), to(self.a), to(self.w), to(self.c),
            to(self.r), to(self.sn), to(self.cn), to(self.d)
        )

    def sample_encoder(self, n, device):
        idx = np.random.randint(0, self.size, n)
        to = lambda x: torch.FloatTensor(x[idx]).to(device)
        tt = torch.LongTensor(self.t[idx].reshape(-1)).to(device)
        yy = torch.LongTensor(self.y[idx].reshape(-1)).to(device)
        return to(self.s), to(self.a), to(self.r), to(self.sn), tt, yy

    def sample_sequences(self, n, seq_len, device):
        if self.size < seq_len:
            raise ValueError(f"buffer size {self.size} < seq_len {seq_len}")

        starts = []
        max_start = self.size - seq_len
        attempts = 0
        while len(starts) < n and attempts < n * 20:
            start = np.random.randint(0, max_start + 1)
            ts = self.t[start:start + seq_len].reshape(-1)
            if len(ts) == seq_len and int(ts[-1] - ts[0]) == seq_len - 1:
                starts.append(start)
            attempts += 1
        if len(starts) < n:
            starts.extend(np.random.randint(0, max_start + 1, size=n - len(starts)).tolist())

        idx = np.asarray([
            np.arange(start, start + seq_len) for start in starts[:n]
        ])
        s = torch.FloatTensor(self.s[idx]).to(device)
        a = torch.FloatTensor(self.a[idx]).to(device)
        r = torch.FloatTensor(self.r[idx]).to(device)
        sn = torch.FloatTensor(self.sn[idx]).to(device)
        tt = torch.LongTensor(self.t[idx[:, -1]].reshape(-1)).to(device)
        return s, a, r, sn, tt


class TransitionContextTracker:
    """Online context from observable transition and reward signatures.

    This is intentionally non-oracle: it uses the previous action, next
    observed velocity, and reward vector, but never reads env mode labels.
    In SyntheticMOEnv the velocity/action response is a direct signature of
    the active substrate dynamics; in a larger system this slot should be
    replaced by a learned latent dynamics/reward encoder.
    """

    def __init__(self, reward_dim: int, ema_alpha: float = 0.2):
        self.reward_dim = reward_dim
        self.ema_alpha = ema_alpha
        self.reward_ema = np.zeros(reward_dim, dtype=np.float32)
        self.speed_response = 0.0
        self.has_speed_response = False

    @property
    def feature_dim(self) -> int:
        return 1 + self.reward_dim

    def features(self) -> np.ndarray:
        if self.has_speed_response:
            speed = np.clip((self.speed_response - 1.0) / 0.5, -3.0, 3.0)
        else:
            speed = 0.0
        reward = np.tanh(self.reward_ema / 3.0)
        return np.concatenate([
            np.array([speed], dtype=np.float32),
            reward.astype(np.float32),
        ]).astype(np.float32)

    def update(self, obs, action, reward, next_obs):
        del obs  # current implementation only needs action and next velocity
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        reward = np.asarray(reward, dtype=np.float32)

        if action.size and next_obs.size >= 2 and abs(float(action[0])) > 0.05:
            response = float(next_obs[1]) / float(action[0])
            if np.isfinite(response) and abs(response) < 10.0:
                if self.has_speed_response:
                    self.speed_response = (
                        (1.0 - self.ema_alpha) * self.speed_response
                        + self.ema_alpha * response
                    )
                else:
                    self.speed_response = response
                    self.has_speed_response = True

        self.reward_ema = (
            (1.0 - self.ema_alpha) * self.reward_ema
            + self.ema_alpha * reward
        ).astype(np.float32)

    def reset(self):
        self.reward_ema[:] = 0.0
        self.speed_response = 0.0
        self.has_speed_response = False


class GenericTransitionContextTracker:
    """Environment-agnostic online transition statistic.

    Unlike `TransitionContextTracker`, this does not assume a particular
    observation coordinate is velocity.  It tracks generic state delta,
    action-normalized state response, action, and reward EMAs.
    """

    def __init__(self, state_dim: int, action_dim: int, reward_dim: int,
                 ema_alpha: float = 0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.ema_alpha = ema_alpha
        self.delta_ema = np.zeros(state_dim, dtype=np.float32)
        self.response_ema = np.zeros(state_dim, dtype=np.float32)
        self.action_ema = np.zeros(action_dim, dtype=np.float32)
        self.reward_ema = np.zeros(reward_dim, dtype=np.float32)

    @property
    def feature_dim(self) -> int:
        return 2 * self.state_dim + self.action_dim + self.reward_dim

    def features(self) -> np.ndarray:
        return np.concatenate([
            self.delta_ema,
            self.response_ema,
            self.action_ema,
            self.reward_ema,
        ]).astype(np.float32)

    def update(self, obs, action, reward, next_obs):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        reward = np.asarray(reward, dtype=np.float32).reshape(-1)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)

        delta_raw = next_obs - obs
        delta = np.tanh(delta_raw / 3.0)
        act_norm = float(np.linalg.norm(action))
        response = np.tanh(delta_raw / max(act_norm, 0.1) / 3.0)
        action_feat = np.tanh(action)
        reward_feat = np.tanh(reward / 3.0)

        self.delta_ema = (
            (1.0 - self.ema_alpha) * self.delta_ema
            + self.ema_alpha * delta
        ).astype(np.float32)
        self.response_ema = (
            (1.0 - self.ema_alpha) * self.response_ema
            + self.ema_alpha * response
        ).astype(np.float32)
        self.action_ema = (
            (1.0 - self.ema_alpha) * self.action_ema
            + self.ema_alpha * action_feat
        ).astype(np.float32)
        self.reward_ema = (
            (1.0 - self.ema_alpha) * self.reward_ema
            + self.ema_alpha * reward_feat
        ).astype(np.float32)

    def reset(self):
        self.delta_ema[:] = 0.0
        self.response_ema[:] = 0.0
        self.action_ema[:] = 0.0
        self.reward_ema[:] = 0.0


class RegimePrototypeBeliefTracker:
    """Soft online belief from detector-labeled transition prototypes."""

    def __init__(self, num_modes: int, feature_dim: int,
                 ema_alpha: float = 0.08, belief_alpha: float = 0.35,
                 temperature: float = 0.35, detector_prior: float = 0.20,
                 max_age: int = 300):
        self.num_modes = num_modes
        self.feature_dim = feature_dim
        self.ema_alpha = ema_alpha
        self.belief_alpha = belief_alpha
        self.temperature = temperature
        self.detector_prior = detector_prior
        self.max_age = max_age
        self.prototypes = np.zeros((num_modes, feature_dim), dtype=np.float32)
        self.counts = np.zeros(num_modes, dtype=np.int64)
        self.belief = np.ones(num_modes, dtype=np.float32) / num_modes
        self.confidence = 1.0 / num_modes
        self.age = 0

    @property
    def context_dim(self) -> int:
        return self.num_modes + 2

    def features(self) -> np.ndarray:
        return np.concatenate([
            self.belief.astype(np.float32),
            np.array([
                self.confidence,
                np.tanh(float(self.age) / max(float(self.max_age), 1.0)),
            ], dtype=np.float32),
        ]).astype(np.float32)

    def _normalize(self, feature) -> np.ndarray:
        feature = np.asarray(feature, dtype=np.float32).reshape(-1)
        if feature.size != self.feature_dim:
            raise ValueError(
                f"prototype feature dim {feature.size} != {self.feature_dim}")
        feature = np.nan_to_num(feature, nan=0.0, posinf=3.0, neginf=-3.0)
        norm = float(np.linalg.norm(feature))
        if norm > 1e-6:
            feature = feature / norm
        return feature.astype(np.float32)

    def update(self, feature, detector_label: int = -1,
               mode_switched: bool = False):
        feature = self._normalize(feature)
        label = int(detector_label)
        if 0 <= label < self.num_modes:
            if self.counts[label] == 0:
                self.prototypes[label] = feature
            else:
                self.prototypes[label] = (
                    (1.0 - self.ema_alpha) * self.prototypes[label]
                    + self.ema_alpha * feature
                ).astype(np.float32)
                pn = float(np.linalg.norm(self.prototypes[label]))
                if pn > 1e-6:
                    self.prototypes[label] /= pn
            self.counts[label] += 1

        known = self.counts > 0
        if known.any():
            logits = np.full(self.num_modes, -8.0, dtype=np.float32)
            diffs = self.prototypes[known] - feature[None, :]
            logits[known] = -np.sum(diffs * diffs, axis=1) / max(self.temperature, 1e-6)
            logits -= logits.max()
            belief_new = np.exp(logits)
            belief_new /= np.maximum(belief_new.sum(), 1e-8)
        else:
            belief_new = np.ones(self.num_modes, dtype=np.float32) / self.num_modes

        if 0 <= label < self.num_modes:
            prior = np.zeros(self.num_modes, dtype=np.float32)
            prior[label] = 1.0
            belief_new = (
                (1.0 - self.detector_prior) * belief_new
                + self.detector_prior * prior
            ).astype(np.float32)
            belief_new /= np.maximum(belief_new.sum(), 1e-8)

        if mode_switched and 0 <= label < self.num_modes:
            self.age = 0
            self.belief = (0.65 * np.eye(self.num_modes, dtype=np.float32)[label]
                           + 0.35 * belief_new).astype(np.float32)
        else:
            self.age += 1
            self.belief = (
                (1.0 - self.belief_alpha) * self.belief
                + self.belief_alpha * belief_new
            ).astype(np.float32)
        self.belief /= np.maximum(self.belief.sum(), 1e-8)
        self.confidence = float(np.max(self.belief))


class LatentTransitionEncoder(nn.Module):
    """Self-supervised encoder for online latent regime context.

    The encoder maps one observed transition `(s, a, r, s')` into a compact
    latent `z`.  Auxiliary heads reconstruct reward and state delta so the
    latent is forced to retain substrate-relevant dynamics/reward structure.
    """

    def __init__(self, state_dim: int, action_dim: int, reward_dim: int,
                 latent_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.latent_dim = latent_dim
        in_dim = state_dim + action_dim + reward_dim + state_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, reward_dim),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def _features(self, s, a, r, sn):
        delta = sn - s
        return torch.cat([s, a, r, delta], dim=-1)

    def encode(self, s, a, r, sn):
        z = self.encoder(self._features(s, a, r, sn))
        return torch.tanh(z)

    def forward(self, s, a, r, sn):
        z = self.encode(s, a, r, sn)
        return z, self.reward_head(z), self.delta_head(z)

    def project(self, z):
        return F.normalize(self.projector(z), dim=-1)


def temporal_contrastive_loss(z, timesteps, pos_window: int = 64,
                              temperature: float = 0.2):
    """Supervised-contrastive-style loss using only temporal proximity.

    Positive pairs are transitions sampled within `pos_window` training steps.
    All other non-self samples are negatives.  This is deliberately weaker
    than oracle mode labels but stronger than reconstruction-only training.
    """
    if z.shape[0] < 2:
        return z.new_tensor(0.0)

    proj = F.normalize(z, dim=-1)
    sim = proj @ proj.T / temperature
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    dt = torch.abs(timesteps.view(-1, 1) - timesteps.view(1, -1))
    positive = (dt <= pos_window) & ~eye
    valid = positive.any(dim=1)
    if not bool(valid.any()):
        return z.new_tensor(0.0)

    logits = sim.masked_fill(eye, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = positive.sum(dim=1).clamp_min(1)
    per_anchor = -(log_prob * positive.float()).sum(dim=1) / pos_count
    return per_anchor[valid].mean()


def pseudo_regime_contrastive_loss(z, labels, temperature: float = 0.2):
    """Supervised contrastive loss from non-oracle detector regime labels."""
    if z.shape[0] < 2:
        return z.new_tensor(0.0)

    valid_label = labels >= 0
    if int(valid_label.sum().item()) < 2:
        return z.new_tensor(0.0)

    proj = F.normalize(z, dim=-1)
    sim = proj @ proj.T / temperature
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    same_label = labels.view(-1, 1) == labels.view(1, -1)
    valid_pair = valid_label.view(-1, 1) & valid_label.view(1, -1)
    positive = same_label & valid_pair & ~eye
    valid = positive.any(dim=1)
    if not bool(valid.any()):
        return z.new_tensor(0.0)

    logits = sim.masked_fill(eye | ~valid_pair, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = positive.sum(dim=1).clamp_min(1)
    per_anchor = -(log_prob * positive.float()).sum(dim=1) / pos_count
    return per_anchor[valid].mean()


class SequenceTransitionEncoder(nn.Module):
    """GRU encoder over recent raw transition features."""

    def __init__(self, state_dim: int, action_dim: int, reward_dim: int,
                 latent_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.latent_dim = latent_dim
        self.feature_dim = state_dim + action_dim + reward_dim + state_dim
        self.gru = nn.GRU(self.feature_dim, hidden, batch_first=True)
        self.latent_head = nn.Linear(hidden, latent_dim)
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, reward_dim),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    @staticmethod
    def make_features(s, a, r, sn):
        delta = torch.tanh((sn - s) / 3.0)
        rew = torch.tanh(r / 3.0)
        return torch.cat([s, a, rew, delta], dim=-1)

    def encode_features(self, seq_features):
        out, _ = self.gru(seq_features)
        return torch.tanh(self.latent_head(out[:, -1]))

    def encode_sequence(self, s, a, r, sn):
        return self.encode_features(self.make_features(s, a, r, sn))

    def forward(self, s, a, r, sn):
        z = self.encode_sequence(s, a, r, sn)
        return z, self.reward_head(z), self.delta_head(z)

    def project(self, z):
        return F.normalize(self.projector(z), dim=-1)


class LearnedTransitionContextTracker:
    """Online EMA of learned transition latents."""

    def __init__(self, encoder: LatentTransitionEncoder,
                 latent_dim: int = 8, ema_alpha: float = 0.35,
                 reward_scale: float = 3.0, delta_scale: float = 3.0):
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.ema_alpha = ema_alpha
        self.reward_scale = reward_scale
        self.delta_scale = delta_scale
        self.z = np.zeros(latent_dim, dtype=np.float32)
        self.count = 0

    def features(self) -> np.ndarray:
        return self.z.astype(np.float32)

    def update(self, obs, action, reward, next_obs, device):
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        reward = np.asarray(reward, dtype=np.float32).reshape(1, -1)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(1, -1)
        s = torch.FloatTensor(obs).to(device)
        a = torch.FloatTensor(action).to(device)
        r = torch.FloatTensor(np.tanh(reward / self.reward_scale)).to(device)
        sn = torch.FloatTensor(next_obs).to(device)
        with torch.no_grad():
            was_training = self.encoder.training
            self.encoder.eval()
            z = self.encoder.encode(s, a, r, sn).cpu().numpy().reshape(-1)
            if was_training:
                self.encoder.train()
        if self.count == 0:
            self.z = z.astype(np.float32)
        else:
            self.z = (
                (1.0 - self.ema_alpha) * self.z
                + self.ema_alpha * z
            ).astype(np.float32)
        self.count += 1

    def reset(self):
        self.z[:] = 0.0
        self.count = 0


class SequenceTransitionContextTracker:
    """Online fixed-window tracker for sequence transition latents."""

    def __init__(self, encoder: SequenceTransitionEncoder, seq_len: int,
                 state_dim: int, action_dim: int, reward_dim: int,
                 latent_dim: int = 8):
        self.encoder = encoder
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.feature_dim = state_dim + action_dim + reward_dim + state_dim
        self.history = deque(maxlen=seq_len)
        self.z = np.zeros(latent_dim, dtype=np.float32)

    def features(self) -> np.ndarray:
        return self.z.astype(np.float32)

    def update(self, obs, action, reward, next_obs, device):
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        reward = np.asarray(reward, dtype=np.float32).reshape(-1)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        delta = np.tanh((next_obs - obs) / 3.0)
        rew = np.tanh(reward / 3.0)
        feat = np.concatenate([obs, action, rew, delta]).astype(np.float32)
        self.history.append(feat)

        seq = np.zeros((self.seq_len, self.feature_dim), dtype=np.float32)
        hist = list(self.history)
        if hist:
            seq[-len(hist):] = np.stack(hist)
        st = torch.FloatTensor(seq).unsqueeze(0).to(device)
        with torch.no_grad():
            was_training = self.encoder.training
            self.encoder.eval()
            z = self.encoder.encode_features(st).cpu().numpy().reshape(-1)
            if was_training:
                self.encoder.train()
        self.z = z.astype(np.float32)


def _one_hot(size, idx):
    v = np.zeros(size, dtype=np.float32)
    v[int(idx)] = 1.0
    return v


def _policy_context(weight, mode_idx, num_modes):
    return np.concatenate([
        np.asarray(weight, dtype=np.float32),
        _one_hot(num_modes, mode_idx),
    ]).astype(np.float32)


def _uses_oracle_context(method):
    return method in CONTEXT_ORACLE_METHODS


def _uses_transition_context(method):
    return method in TRANSITION_CONTEXT_METHODS


def _uses_generic_context(method):
    return method in GENERIC_CONTEXT_METHODS


def _uses_learned_context(method):
    return method in LEARNED_CONTEXT_METHODS


def _uses_contrastive_context(method):
    return method in CONTRASTIVE_CONTEXT_METHODS


def _uses_boundary_context(method):
    return method in BOUNDARY_CONTEXT_METHODS


def _uses_proto_context(method):
    return method in PROTO_CONTEXT_METHODS


def _uses_sequence_context(method):
    return method in SEQUENCE_CONTEXT_METHODS


def _uses_belief_context(method):
    return (
        method in CONTEXT_BELIEF_METHODS
        or method in TRANSITION_BAMOR_CONTEXT_METHODS
        or method in LEARNED_BAMOR_CONTEXT_METHODS
    )


def _context_dim_for_method(method, env, r_dim):
    dim = r_dim
    if _uses_oracle_context(method) or _uses_belief_context(method):
        dim += env.num_modes
    if _uses_transition_context(method):
        dim += 1 + r_dim  # speed-response signature + reward EMA
    if _uses_generic_context(method):
        s_dim = int(np.prod(env.observation_space.shape))
        a_dim = int(np.prod(env.action_space.shape))
        dim += 2 * s_dim + a_dim + r_dim
    if _uses_learned_context(method):
        dim += LATENT_CONTEXT_DIM
    if _uses_proto_context(method):
        dim += env.num_modes + 2
    if _uses_sequence_context(method):
        dim += LATENT_CONTEXT_DIM
    return dim


def _proto_feature_dim_for_method(method, env, r_dim):
    dim = 0
    if _uses_transition_context(method):
        dim += 1 + r_dim
    if _uses_generic_context(method):
        s_dim = int(np.prod(env.observation_space.shape))
        a_dim = int(np.prod(env.action_space.shape))
        dim += 2 * s_dim + a_dim + r_dim
    if _uses_learned_context(method):
        dim += LATENT_CONTEXT_DIM
    return dim


def _prototype_feature(method, trans_tracker=None, generic_tracker=None,
                       learned_tracker=None):
    parts = []
    if _uses_transition_context(method):
        if trans_tracker is None:
            raise ValueError(f"{method} requires a transition context tracker")
        parts.append(trans_tracker.features())
    if _uses_generic_context(method):
        if generic_tracker is None:
            raise ValueError(f"{method} requires a generic transition context tracker")
        parts.append(generic_tracker.features())
    if _uses_learned_context(method):
        if learned_tracker is None:
            raise ValueError(f"{method} requires a learned transition context tracker")
        parts.append(learned_tracker.features())
    if not parts:
        raise ValueError(f"{method} has no prototype feature source")
    return np.concatenate(parts).astype(np.float32)


def _build_context(method, weight, active_mode, env, bamor=None,
                   trans_tracker=None, generic_tracker=None,
                   learned_tracker=None, proto_tracker=None,
                   seq_tracker=None):
    parts = [np.asarray(weight, dtype=np.float32)]
    if _uses_oracle_context(method):
        parts.append(_one_hot(env.num_modes, active_mode))
    elif _uses_belief_context(method):
        if bamor is None:
            raise ValueError(f"{method} requires a BAMOR belief context")
        parts.append(bamor.belief_tracker.belief.astype(np.float32))
    if _uses_transition_context(method):
        if trans_tracker is None:
            raise ValueError(f"{method} requires a transition context tracker")
        parts.append(trans_tracker.features())
    if _uses_generic_context(method):
        if generic_tracker is None:
            raise ValueError(f"{method} requires a generic transition context tracker")
        parts.append(generic_tracker.features())
    if _uses_learned_context(method):
        if learned_tracker is None:
            raise ValueError(f"{method} requires a learned transition context tracker")
        parts.append(learned_tracker.features())
    if _uses_proto_context(method):
        if proto_tracker is None:
            raise ValueError(f"{method} requires a prototype belief tracker")
        parts.append(proto_tracker.features())
    if _uses_sequence_context(method):
        if seq_tracker is None:
            raise ValueError(f"{method} requires a sequence transition context tracker")
        parts.append(seq_tracker.features())
    return np.concatenate(parts).astype(np.float32)


def eval_pareto_context_oracle(env, pi, r_dim, dev, num_modes,
                               num_w=21, num_ep=3, ref_point=None):
    """Evaluate a mode-conditioned policy using true mode one-hot context."""
    wg = np.linspace(0.01, 0.99, num_w)
    weights = np.stack([1 - wg, wg], axis=-1).astype(np.float32)
    results = []
    for w in weights:
        ep_rets = []
        for _ in range(num_ep):
            obs, info = env.reset()
            tr = np.zeros(r_dim)
            done = False
            while not done:
                mode_idx = int(getattr(env, 'current_mode_idx', info.get('mode_idx', 0)))
                c = _policy_context(w, mode_idx, num_modes)
                with torch.no_grad():
                    st = torch.FloatTensor(obs).unsqueeze(0).to(dev)
                    ct = torch.FloatTensor(c).unsqueeze(0).to(dev)
                    action = pi.get_action(st, ct, deterministic=True)
                    if isinstance(action, tuple):
                        action = action[0]
                obs, rew, term, trunc, info = env.step(action.cpu().numpy().flatten())
                tr += np.asarray(rew)
                done = term or trunc
            ep_rets.append(tr)
        results.append({'weight': w.tolist(),
                        'return': np.mean(ep_rets, axis=0).tolist()})

    all_returns = np.array([p['return'] for p in results])
    if ref_point is None:
        ref_point = (all_returns[:, 0].min() - abs(all_returns[:, 0].min()) * 0.1 - 1.0,
                     all_returns[:, 1].min() - abs(all_returns[:, 1].min()) * 0.1 - 1.0)
    hv = compute_hv_2d(all_returns, ref_point)
    return results, hv, ref_point


def eval_pareto_context_belief(env, pi, bamor, r_dim, dev,
                               num_w=21, num_ep=3, ref_point=None):
    """Evaluate a belief-conditioned policy with online BAMOR belief updates."""
    eval_bamor = copy.deepcopy(bamor)
    wg = np.linspace(0.01, 0.99, num_w)
    weights = np.stack([1 - wg, wg], axis=-1).astype(np.float32)
    results = []
    for w in weights:
        ep_rets = []
        for _ in range(num_ep):
            obs, info = env.reset()
            tr = np.zeros(r_dim)
            done = False
            while not done:
                c = np.concatenate([
                    w,
                    eval_bamor.belief_tracker.belief.astype(np.float32),
                ]).astype(np.float32)
                active_mode = int(getattr(env, 'current_mode_idx', info.get('mode_idx', 0)))
                with torch.no_grad():
                    st = torch.FloatTensor(obs).unsqueeze(0).to(dev)
                    ct = torch.FloatTensor(c).unsqueeze(0).to(dev)
                    action = pi.get_action(st, ct, deterministic=True)
                    if isinstance(action, tuple):
                        action = action[0]
                obs, rew, term, trunc, info = env.step(action.cpu().numpy().flatten())
                tr += np.asarray(rew)
                if eval_bamor.mode_source == 'oracle':
                    eval_bamor.observe_reward(rew, true_mode=active_mode)
                else:
                    eval_bamor.observe_reward(rew)
                done = term or trunc
            ep_rets.append(tr)
        results.append({'weight': w.tolist(),
                        'return': np.mean(ep_rets, axis=0).tolist()})

    all_returns = np.array([p['return'] for p in results])
    if ref_point is None:
        ref_point = (all_returns[:, 0].min() - abs(all_returns[:, 0].min()) * 0.1 - 1.0,
                     all_returns[:, 1].min() - abs(all_returns[:, 1].min()) * 0.1 - 1.0)
    hv = compute_hv_2d(all_returns, ref_point)
    return results, hv, ref_point


def eval_pareto_context_transition(env, pi, method, bamor, r_dim, dev,
                                   num_w=21, num_ep=3, ref_point=None):
    """Evaluate a transition-context policy with online non-oracle signatures."""
    eval_bamor = copy.deepcopy(bamor) if bamor is not None else None
    trans_tracker = TransitionContextTracker(r_dim)
    wg = np.linspace(0.01, 0.99, num_w)
    weights = np.stack([1 - wg, wg], axis=-1).astype(np.float32)
    results = []
    for w in weights:
        ep_rets = []
        for _ in range(num_ep):
            obs, info = env.reset()
            tr = np.zeros(r_dim)
            done = False
            while not done:
                active_mode = int(getattr(env, 'current_mode_idx', info.get('mode_idx', 0)))
                c = _build_context(
                    method, w, active_mode, env,
                    bamor=eval_bamor, trans_tracker=trans_tracker)
                with torch.no_grad():
                    st = torch.FloatTensor(obs).unsqueeze(0).to(dev)
                    ct = torch.FloatTensor(c).unsqueeze(0).to(dev)
                    action = pi.get_action(st, ct, deterministic=True)
                    if isinstance(action, tuple):
                        action = action[0]
                action_np = action.cpu().numpy().flatten()
                nobs, rew, term, trunc, info = env.step(action_np)
                tr += np.asarray(rew)
                if eval_bamor is not None:
                    if eval_bamor.mode_source == 'oracle':
                        eval_bamor.observe_reward(rew, true_mode=active_mode)
                    else:
                        eval_bamor.observe_reward(rew)
                trans_tracker.update(obs, action_np, rew, nobs)
                obs = nobs
                done = term or trunc
            ep_rets.append(tr)
        results.append({'weight': w.tolist(),
                        'return': np.mean(ep_rets, axis=0).tolist()})

    all_returns = np.array([p['return'] for p in results])
    if ref_point is None:
        ref_point = (all_returns[:, 0].min() - abs(all_returns[:, 0].min()) * 0.1 - 1.0,
                     all_returns[:, 1].min() - abs(all_returns[:, 1].min()) * 0.1 - 1.0)
    hv = compute_hv_2d(all_returns, ref_point)
    return results, hv, ref_point


def eval_pareto_context_generic(env, pi, method, bamor, r_dim, dev,
                                num_w=21, num_ep=3, ref_point=None):
    """Evaluate generic transition-statistic context online."""
    eval_bamor = copy.deepcopy(bamor) if bamor is not None else None
    s_dim = int(np.prod(env.observation_space.shape))
    a_dim = int(np.prod(env.action_space.shape))
    generic_tracker = GenericTransitionContextTracker(s_dim, a_dim, r_dim)
    wg = np.linspace(0.01, 0.99, num_w)
    weights = np.stack([1 - wg, wg], axis=-1).astype(np.float32)
    results = []
    for w in weights:
        ep_rets = []
        for _ in range(num_ep):
            obs, info = env.reset()
            tr = np.zeros(r_dim)
            done = False
            while not done:
                active_mode = int(getattr(env, 'current_mode_idx', info.get('mode_idx', 0)))
                c = _build_context(
                    method, w, active_mode, env,
                    bamor=eval_bamor, generic_tracker=generic_tracker)
                with torch.no_grad():
                    st = torch.FloatTensor(obs).unsqueeze(0).to(dev)
                    ct = torch.FloatTensor(c).unsqueeze(0).to(dev)
                    action = pi.get_action(st, ct, deterministic=True)
                    if isinstance(action, tuple):
                        action = action[0]
                action_np = action.cpu().numpy().flatten()
                nobs, rew, term, trunc, info = env.step(action_np)
                tr += np.asarray(rew)
                if eval_bamor is not None:
                    if eval_bamor.mode_source == 'oracle':
                        eval_bamor.observe_reward(rew, true_mode=active_mode)
                    else:
                        eval_bamor.observe_reward(rew)
                generic_tracker.update(obs, action_np, rew, nobs)
                obs = nobs
                done = term or trunc
            ep_rets.append(tr)
        results.append({'weight': w.tolist(),
                        'return': np.mean(ep_rets, axis=0).tolist()})

    all_returns = np.array([p['return'] for p in results])
    if ref_point is None:
        ref_point = (all_returns[:, 0].min() - abs(all_returns[:, 0].min()) * 0.1 - 1.0,
                     all_returns[:, 1].min() - abs(all_returns[:, 1].min()) * 0.1 - 1.0)
    hv = compute_hv_2d(all_returns, ref_point)
    return results, hv, ref_point


def eval_pareto_context_learned(env, pi, method, bamor, encoder, r_dim, dev,
                                num_w=21, num_ep=3, ref_point=None):
    """Evaluate BAMOR-v2 learned latent context online."""
    eval_bamor = copy.deepcopy(bamor) if bamor is not None else None
    trans_tracker = (TransitionContextTracker(r_dim)
                     if _uses_transition_context(method) else None)
    s_dim = int(np.prod(env.observation_space.shape))
    a_dim = int(np.prod(env.action_space.shape))
    generic_tracker = (GenericTransitionContextTracker(s_dim, a_dim, r_dim)
                       if _uses_generic_context(method) else None)
    learned_tracker = LearnedTransitionContextTracker(
        encoder, latent_dim=LATENT_CONTEXT_DIM)
    proto_tracker = (RegimePrototypeBeliefTracker(
        env.num_modes, _proto_feature_dim_for_method(method, env, r_dim))
        if _uses_proto_context(method) else None)
    wg = np.linspace(0.01, 0.99, num_w)
    weights = np.stack([1 - wg, wg], axis=-1).astype(np.float32)
    results = []
    for w in weights:
        ep_rets = []
        for _ in range(num_ep):
            obs, info = env.reset()
            tr = np.zeros(r_dim)
            done = False
            while not done:
                active_mode = int(getattr(env, 'current_mode_idx', info.get('mode_idx', 0)))
                c = _build_context(
                    method, w, active_mode, env,
                    bamor=eval_bamor, trans_tracker=trans_tracker,
                    generic_tracker=generic_tracker,
                    learned_tracker=learned_tracker,
                    proto_tracker=proto_tracker)
                with torch.no_grad():
                    st = torch.FloatTensor(obs).unsqueeze(0).to(dev)
                    ct = torch.FloatTensor(c).unsqueeze(0).to(dev)
                    action = pi.get_action(st, ct, deterministic=True)
                    if isinstance(action, tuple):
                        action = action[0]
                action_np = action.cpu().numpy().flatten()
                nobs, rew, term, trunc, info = env.step(action_np)
                tr += np.asarray(rew)
                bamor_info = None
                if eval_bamor is not None:
                    if eval_bamor.mode_source == 'oracle':
                        bamor_info = eval_bamor.observe_reward(rew, true_mode=active_mode)
                    else:
                        bamor_info = eval_bamor.observe_reward(rew)
                if _uses_boundary_context(method) and bamor_info and bamor_info.get('mode_switched'):
                    if trans_tracker is not None:
                        trans_tracker.reset()
                    if generic_tracker is not None:
                        generic_tracker.reset()
                    learned_tracker.reset()
                if trans_tracker is not None:
                    trans_tracker.update(obs, action_np, rew, nobs)
                if generic_tracker is not None:
                    generic_tracker.update(obs, action_np, rew, nobs)
                learned_tracker.update(obs, action_np, rew, nobs, dev)
                if proto_tracker is not None:
                    label = (int(eval_bamor.belief_tracker.current_mode)
                             if eval_bamor is not None else -1)
                    proto_tracker.update(
                        _prototype_feature(
                            method, trans_tracker=trans_tracker,
                            generic_tracker=generic_tracker,
                            learned_tracker=learned_tracker),
                        detector_label=label,
                        mode_switched=bool(
                            bamor_info and bamor_info.get('mode_switched')))
                obs = nobs
                done = term or trunc
            ep_rets.append(tr)
        results.append({'weight': w.tolist(),
                        'return': np.mean(ep_rets, axis=0).tolist()})

    all_returns = np.array([p['return'] for p in results])
    if ref_point is None:
        ref_point = (all_returns[:, 0].min() - abs(all_returns[:, 0].min()) * 0.1 - 1.0,
                     all_returns[:, 1].min() - abs(all_returns[:, 1].min()) * 0.1 - 1.0)
    hv = compute_hv_2d(all_returns, ref_point)
    return results, hv, ref_point


def eval_pareto_context_sequence(env, pi, method, bamor, seq_encoder,
                                 r_dim, dev, num_w=21, num_ep=3, ref_point=None):
    """Evaluate sequence-regime context online with no oracle mode labels."""
    eval_bamor = copy.deepcopy(bamor) if bamor is not None else None
    s_dim = int(np.prod(env.observation_space.shape))
    a_dim = int(np.prod(env.action_space.shape))
    seq_tracker = SequenceTransitionContextTracker(
        seq_encoder, SEQUENCE_CONTEXT_LEN, s_dim, a_dim, r_dim,
        latent_dim=LATENT_CONTEXT_DIM)
    wg = np.linspace(0.01, 0.99, num_w)
    weights = np.stack([1 - wg, wg], axis=-1).astype(np.float32)
    results = []
    for w in weights:
        ep_rets = []
        for _ in range(num_ep):
            obs, info = env.reset()
            tr = np.zeros(r_dim)
            done = False
            while not done:
                active_mode = int(getattr(env, 'current_mode_idx', info.get('mode_idx', 0)))
                c = _build_context(
                    method, w, active_mode, env,
                    bamor=eval_bamor, seq_tracker=seq_tracker)
                with torch.no_grad():
                    st = torch.FloatTensor(obs).unsqueeze(0).to(dev)
                    ct = torch.FloatTensor(c).unsqueeze(0).to(dev)
                    action = pi.get_action(st, ct, deterministic=True)
                    if isinstance(action, tuple):
                        action = action[0]
                action_np = action.cpu().numpy().flatten()
                nobs, rew, term, trunc, info = env.step(action_np)
                tr += np.asarray(rew)
                if eval_bamor is not None:
                    if eval_bamor.mode_source == 'oracle':
                        eval_bamor.observe_reward(rew, true_mode=active_mode)
                    else:
                        eval_bamor.observe_reward(rew)
                seq_tracker.update(obs, action_np, rew, nobs, dev)
                obs = nobs
                done = term or trunc
            ep_rets.append(tr)
        results.append({'weight': w.tolist(),
                        'return': np.mean(ep_rets, axis=0).tolist()})

    all_returns = np.array([p['return'] for p in results])
    if ref_point is None:
        ref_point = (all_returns[:, 0].min() - abs(all_returns[:, 0].min()) * 0.1 - 1.0,
                     all_returns[:, 1].min() - abs(all_returns[:, 1].min()) * 0.1 - 1.0)
    hv = compute_hv_2d(all_returns, ref_point)
    return results, hv, ref_point


class MOGymWrapper(gym.Env):
    """Adapts SyntheticMOEnv to the mo-gymnasium API (reward_space attribute).

    CAPQL and GPI-PD require:
        env.reward_space : Box(reward_dim,)
        env.reward_dim   : int
        step()           → 5-tuple with np.ndarray reward
    """
    metadata = {"render_modes": []}

    def __init__(self, switch_interval=300, max_steps=200, seed=None, modes=None):
        super().__init__()
        kwargs = dict(switch_interval=switch_interval, max_steps=max_steps, seed=seed)
        if modes is not None:
            kwargs['modes'] = modes
        self._env = SyntheticMOEnv(**kwargs)
        self.observation_space = self._env.observation_space
        self.action_space      = self._env.action_space
        self.reward_space      = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self._env.reward_dim,), dtype=np.float64)
        self.reward_dim        = self._env.reward_dim
        self.num_modes         = self._env.num_modes

    # Pass-through mode locking for per-mode eval
    @property
    def _lock_mode(self):
        return getattr(self._env, '_lock_mode', False)
    @_lock_mode.setter
    def _lock_mode(self, v):
        self._env._lock_mode = v

    @property
    def current_mode_idx(self):
        return self._env.current_mode_idx
    @current_mode_idx.setter
    def current_mode_idx(self, v):
        self._env.current_mode_idx = v

    def reset(self, *, seed=None, options=None):
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        obs, rew, term, trunc, info = self._env.step(action)
        return obs, np.asarray(rew, dtype=np.float64), term, trunc, info

    def close(self):
        self._env.close()


# ════════════════════════════════════════════════════════════════════════════
# Unified eval: works for BAMOR (policy net) and morl-baselines agents
# ════════════════════════════════════════════════════════════════════════════

def eval_front_bamor(env, pi, r_dim, dev, num_w=21, num_ep=3, ref_point=None):
    """Evaluate BAMOR policy (SmallPolicy) → returns (results, hv, ref_point)."""
    return eval_pareto(env, pi, r_dim, dev, num_w=num_w, num_ep=num_ep,
                       ref_point=ref_point)


def eval_front_agent(env, agent, r_dim, num_w=21, num_ep=3, ref_point=None):
    """Evaluate a morl-baselines agent (has .eval(obs, w) → action)."""
    wg = np.linspace(0.01, 0.99, num_w)
    weights = np.stack([1 - wg, wg], axis=-1).astype(np.float32)
    results = []
    for w in weights:
        ep_rets = []
        for _ in range(num_ep):
            obs, _ = env.reset()
            tr = np.zeros(r_dim)
            done = False
            while not done:
                action = agent.eval(obs, w)
                obs, rew, term, trunc, _ = env.step(action)
                tr += np.asarray(rew)
                done = term or trunc
            ep_rets.append(tr)
        results.append({'weight': w.tolist(),
                        'return': np.mean(ep_rets, axis=0).tolist()})

    all_returns = np.array([p['return'] for p in results])
    if ref_point is None:
        ref_point = (all_returns[:, 0].min() - abs(all_returns[:, 0].min()) * 0.1 - 1.0,
                     all_returns[:, 1].min() - abs(all_returns[:, 1].min()) * 0.1 - 1.0)
    hv = compute_hv_2d(all_returns, ref_point)
    return results, hv, ref_point


def eval_per_mode(env_or_wrapper, eval_fn, num_modes, r_dim, ref_point=None):
    """Run eval locked per mode, return list of (results, hv) per mode."""
    per_mode = []
    for m in range(num_modes):
        env_or_wrapper._lock_mode = True
        env_or_wrapper.current_mode_idx = m
        out = eval_fn(ref_point=ref_point)
        env_or_wrapper._lock_mode = False
        per_mode.append(out)
        if ref_point is None:
            ref_point = out[2]   # reuse ref from first mode
    return per_mode


# ════════════════════════════════════════════════════════════════════════════
# BAMOR / Uniform / Fixed training  (reuses train_synthetic_bamor logic)
# ════════════════════════════════════════════════════════════════════════════

BAMOR_FAMILY = {
    'bamor',
    'bamor_nc',
    'bamor_ns',
    'bamor_oracle',
    'bamor_oracle_cone',
    'bamor_fixed_cone',
    'bamor_ns_eps',
    'bamor_ctx',
    'bamor_ctx_nc',
    'bamor_ctx_fixed_cone',
    'bamor_ctx_oracle',
    'bamor_ctx_oracle_cone',
    'bamor_fastdet',
    'bamor_fastdet_ctx',
    'bamor_fastdet_ctx_nc',
    'bamor_fastdet_trans_ctx',
    'bamor_fastdet_trans_ctx_nc',
    'bamor_v2_belief',
    'bamor_v2_cone',
    'bamor_v2_hybrid_belief',
    'bamor_v2_hybrid_cone',
    'bamor_v2_contrast_hybrid_cone',
    'bamor_v2_seq_contrast_cone',
    'bamor_v2_contrast_generic_cone',
    'bamor_v2_boundary_hybrid',
    'bamor_v2_boundary_hybrid_cone',
    'bamor_v2_boundary_generic',
    'bamor_v2_proto_hybrid',
    'bamor_v2_proto_hybrid_cone',
    'bamor_v2_proto_generic',
}

CONTEXT_ORACLE_METHODS = {
    'uniform_ctx_oracle',
    'bamor_ctx_oracle',
    'bamor_ctx_oracle_cone',
}

CONTEXT_BELIEF_METHODS = {
    'bamor_ctx',
    'bamor_ctx_nc',
    'bamor_ctx_fixed_cone',
    'bamor_fastdet_ctx',
    'bamor_fastdet_ctx_nc',
}

TRANSITION_CONTEXT_METHODS = {
    'uniform_trans_ctx',
    'bamor_fastdet_trans_ctx',
    'bamor_fastdet_trans_ctx_nc',
    'bamor_v2_hybrid',
    'bamor_v2_hybrid_belief',
    'bamor_v2_hybrid_cone',
    'bamor_v2_contrast_hybrid',
    'bamor_v2_contrast_hybrid_cone',
    'bamor_v2_boundary_hybrid',
    'bamor_v2_boundary_hybrid_cone',
    'bamor_v2_proto_hybrid',
    'bamor_v2_proto_hybrid_cone',
}

TRANSITION_BAMOR_CONTEXT_METHODS = {
    'bamor_fastdet_trans_ctx',
    'bamor_fastdet_trans_ctx_nc',
}

GENERIC_CONTEXT_METHODS = {
    'bamor_v2_generic',
    'bamor_v2_contrast_generic',
    'bamor_v2_contrast_generic_cone',
    'bamor_v2_boundary_generic',
    'bamor_v2_proto_generic',
}

LATENT_CONTEXT_DIM = 8

LEARNED_CONTEXT_METHODS = {
    'bamor_v2',
    'bamor_v2_belief',
    'bamor_v2_cone',
    'bamor_v2_hybrid',
    'bamor_v2_hybrid_belief',
    'bamor_v2_hybrid_cone',
    'bamor_v2_contrast',
    'bamor_v2_contrast_hybrid',
    'bamor_v2_contrast_hybrid_cone',
    'bamor_v2_contrast_generic',
    'bamor_v2_contrast_generic_cone',
    'bamor_v2_boundary_hybrid',
    'bamor_v2_boundary_hybrid_cone',
    'bamor_v2_boundary_generic',
    'bamor_v2_proto_hybrid',
    'bamor_v2_proto_hybrid_cone',
    'bamor_v2_proto_generic',
}

LEARNED_BAMOR_CONTEXT_METHODS = {
    'bamor_v2_belief',
    'bamor_v2_cone',
    'bamor_v2_hybrid_belief',
    'bamor_v2_hybrid_cone',
    'bamor_v2_contrast_hybrid_cone',
    'bamor_v2_seq_contrast_cone',
    'bamor_v2_contrast_generic_cone',
    'bamor_v2_boundary_hybrid_cone',
    'bamor_v2_proto_hybrid_cone',
}

CONTRASTIVE_CONTEXT_METHODS = {
    'bamor_v2_contrast',
    'bamor_v2_contrast_hybrid',
    'bamor_v2_contrast_hybrid_cone',
    'bamor_v2_contrast_generic',
    'bamor_v2_contrast_generic_cone',
    'bamor_v2_seq_contrast',
    'bamor_v2_seq_contrast_cone',
    'bamor_v2_boundary_hybrid',
    'bamor_v2_boundary_hybrid_cone',
    'bamor_v2_boundary_generic',
    'bamor_v2_proto_hybrid',
    'bamor_v2_proto_hybrid_cone',
    'bamor_v2_proto_generic',
}

BOUNDARY_CONTEXT_METHODS = {
    'bamor_v2_boundary_hybrid',
    'bamor_v2_boundary_hybrid_cone',
    'bamor_v2_boundary_generic',
    'bamor_v2_proto_hybrid',
    'bamor_v2_proto_hybrid_cone',
    'bamor_v2_proto_generic',
}

PROTO_CONTEXT_METHODS = {
    'bamor_v2_proto_hybrid',
    'bamor_v2_proto_hybrid_cone',
    'bamor_v2_proto_generic',
}

SEQUENCE_CONTEXT_LEN = 16

SEQUENCE_CONTEXT_METHODS = {
    'bamor_v2_seq',
    'bamor_v2_seq_contrast',
    'bamor_v2_seq_contrast_cone',
}

UNIFORM_WEIGHT_METHODS = {
    'uniform',
    'uniform_ctx_oracle',
    'uniform_trans_ctx',
    'bamor_v2',
    'bamor_v2_hybrid',
    'bamor_v2_contrast',
    'bamor_v2_contrast_hybrid',
    'bamor_v2_generic',
    'bamor_v2_contrast_generic',
    'bamor_v2_proto_hybrid',
    'bamor_v2_proto_generic',
    'bamor_v2_seq',
    'bamor_v2_seq_contrast',
}


def _base_bamor_method(method):
    return {
        'bamor_ctx': 'bamor',
        'bamor_ctx_nc': 'bamor_nc',
        'bamor_ctx_fixed_cone': 'bamor_fixed_cone',
        'bamor_ctx_oracle': 'bamor_oracle',
        'bamor_ctx_oracle_cone': 'bamor_oracle_cone',
        'bamor_fastdet': 'bamor',
        'bamor_fastdet_ctx': 'bamor',
        'bamor_fastdet_ctx_nc': 'bamor_nc',
        'bamor_fastdet_trans_ctx': 'bamor',
        'bamor_fastdet_trans_ctx_nc': 'bamor_nc',
        'bamor_v2_belief': 'bamor_nc',
        'bamor_v2_cone': 'bamor',
        'bamor_v2_hybrid_belief': 'bamor_nc',
        'bamor_v2_hybrid_cone': 'bamor',
        'bamor_v2_contrast_hybrid_cone': 'bamor',
        'bamor_v2_seq_contrast_cone': 'bamor',
        'bamor_v2_contrast_generic_cone': 'bamor',
        'bamor_v2_boundary_hybrid': 'bamor_nc',
        'bamor_v2_boundary_hybrid_cone': 'bamor',
        'bamor_v2_boundary_generic': 'bamor_nc',
        'bamor_v2_proto_hybrid': 'bamor_nc',
        'bamor_v2_proto_hybrid_cone': 'bamor',
        'bamor_v2_proto_generic': 'bamor_nc',
    }.get(method, method)


def _oracle_cones_from_env(env):
    anchors = np.stack([m.true_anchor for m in env.modes]).astype(np.float64)
    # Moderate radius: tight enough to test the cone mechanism, not a point mass.
    radii = np.full(env.num_modes, 0.25, dtype=np.float64)
    return anchors, radii


def _make_bamor_pipeline(method, env, reward_dim, gamma):
    original_method = method
    method = _base_bamor_method(method)
    fast_detector = original_method in {
        'bamor_fastdet',
        'bamor_fastdet_ctx',
        'bamor_fastdet_ctx_nc',
        'bamor_fastdet_trans_ctx',
        'bamor_fastdet_trans_ctx_nc',
        'bamor_v2_belief',
        'bamor_v2_cone',
        'bamor_v2_hybrid_belief',
        'bamor_v2_hybrid_cone',
        'bamor_v2_contrast_hybrid_cone',
        'bamor_v2_seq_contrast_cone',
        'bamor_v2_contrast_generic_cone',
        'bamor_v2_boundary_hybrid',
        'bamor_v2_boundary_hybrid_cone',
        'bamor_v2_boundary_generic',
        'bamor_v2_proto_hybrid',
        'bamor_v2_proto_hybrid_cone',
        'bamor_v2_proto_generic',
    }
    fixed_anchors = fixed_radii = None
    if method in ('bamor_oracle_cone', 'bamor_fixed_cone'):
        fixed_anchors, fixed_radii = _oracle_cones_from_env(env)

    is_oracle_mode = method in ('bamor_oracle', 'bamor_oracle_cone')
    is_frozen_global = method in ('bamor_ns', 'bamor_ns_eps')

    return BAMORPipeline(
        num_modes=env.num_modes,
        reward_dim=reward_dim,
        gamma=gamma,
        surprise_threshold=3.0 if fast_detector else 7.0,
        concentration=20.0,
        switch_cooldown=250 if fast_detector else 290,
        warmup=20 if fast_detector else 150,
        cusum_required_consecutive=2 if fast_detector else 5,
        cusum_decay=0.25 if fast_detector else 0.5,
        disable_surprise=is_frozen_global or is_oracle_mode,
        disable_cone=(method == 'bamor_nc'),
        freeze_belief=is_frozen_global or is_oracle_mode,
        mode_source='oracle' if is_oracle_mode else ('global' if is_frozen_global else 'belief'),
        fixed_anchors=fixed_anchors,
        fixed_radii=fixed_radii,
        residual_uniform_prob=0.05 if method == 'bamor_ns_eps' else 0.0,
    )


def train_bamor_family(method, total_steps, switch_interval, seed, save_dir,
                       device='cpu', ref_point=None, eval_freq=None):
    """Train one of: bamor, bamor_nc, bamor_ns, uniform, fixed."""
    np.random.seed(seed); torch.manual_seed(seed)
    dev = torch.device(device)

    env      = create_env(switch_interval=switch_interval, seed=seed)
    eval_env = create_env(switch_interval=switch_interval, seed=seed + 1000)

    s_dim, a_dim, r_dim = 2, 1, 2
    gamma, tau, alpha, lr, bs = 0.99, 0.005, 0.2, 3e-4, 128
    use_oracle_context = _uses_oracle_context(method)
    use_belief_context = _uses_belief_context(method)
    use_transition_context = _uses_transition_context(method)
    use_generic_context = _uses_generic_context(method)
    use_learned_context = _uses_learned_context(method)
    use_contrastive_context = _uses_contrastive_context(method)
    use_boundary_context = _uses_boundary_context(method)
    use_proto_context = _uses_proto_context(method)
    use_sequence_context = _uses_sequence_context(method)
    use_context = (
        use_oracle_context or use_belief_context
        or use_transition_context or use_generic_context
        or use_learned_context
        or use_proto_context
        or use_sequence_context
    )
    context_dim = _context_dim_for_method(method, env, r_dim)
    q1  = SmallQNet(s_dim, a_dim, context_dim, r_dim).to(dev)
    q2  = SmallQNet(s_dim, a_dim, context_dim, r_dim).to(dev)
    q1t = SmallQNet(s_dim, a_dim, context_dim, r_dim).to(dev); q1t.load_state_dict(q1.state_dict())
    q2t = SmallQNet(s_dim, a_dim, context_dim, r_dim).to(dev); q2t.load_state_dict(q2.state_dict())
    pi  = SmallPolicy(s_dim, a_dim, context_dim).to(dev)
    q_opt  = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    pi_opt = torch.optim.Adam(pi.parameters(), lr=lr)
    encoder = None
    enc_opt = None
    seq_encoder = None
    seq_opt = None
    if use_learned_context:
        encoder = LatentTransitionEncoder(
            s_dim, a_dim, r_dim, latent_dim=LATENT_CONTEXT_DIM).to(dev)
        enc_opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    if use_sequence_context:
        seq_encoder = SequenceTransitionEncoder(
            s_dim, a_dim, r_dim, latent_dim=LATENT_CONTEXT_DIM).to(dev)
        seq_opt = torch.optim.Adam(seq_encoder.parameters(), lr=lr)
    if use_context:
        buf = ContextBuffer(100_000, s_dim, a_dim, r_dim, context_dim, r_dim)
    else:
        buf = Buffer(100_000, s_dim, a_dim, r_dim, r_dim)

    bamor = None
    if method in BAMOR_FAMILY:
        bamor = _make_bamor_pipeline(method, env, r_dim, gamma)
    trans_tracker = TransitionContextTracker(r_dim) if use_transition_context else None
    generic_tracker = (GenericTransitionContextTracker(s_dim, a_dim, r_dim)
                       if use_generic_context else None)
    learned_tracker = (LearnedTransitionContextTracker(
        encoder, latent_dim=LATENT_CONTEXT_DIM) if use_learned_context else None)
    proto_tracker = (RegimePrototypeBeliefTracker(
        env.num_modes, _proto_feature_dim_for_method(method, env, r_dim))
        if use_proto_context else None)
    seq_tracker = (SequenceTransitionContextTracker(
        seq_encoder, SEQUENCE_CONTEXT_LEN, s_dim, a_dim, r_dim,
        latent_dim=LATENT_CONTEXT_DIM) if use_sequence_context else None)

    log = {'method': method, 'seed': seed, 'hv_curve': [], 'mode_switches': []}
    if use_oracle_context:
        log['policy_context'] = 'oracle_mode_one_hot'
    elif use_transition_context and use_learned_context:
        log['policy_context'] = 'transition_signature+learned_transition_latent'
    elif use_generic_context and use_learned_context:
        log['policy_context'] = 'generic_transition_stat+learned_transition_latent'
    elif use_generic_context:
        log['policy_context'] = 'generic_transition_stat'
    elif use_proto_context:
        log['policy_context'] = 'prototype_regime_belief'
    elif use_transition_context:
        log['policy_context'] = 'transition_signature'
    elif use_learned_context:
        log['policy_context'] = 'learned_transition_latent'
    elif use_sequence_context:
        log['policy_context'] = 'sequence_transition_latent'
    elif use_belief_context:
        log['policy_context'] = 'online_bamor_belief'
    else:
        log['policy_context'] = 'none'
    if bamor is not None:
        log['bamor_config'] = {
            'mode_source': bamor.mode_source,
            'freeze_belief': bamor.freeze_belief,
            'disable_surprise': bamor.disable_surprise,
            'disable_cone': bamor.disable_cone,
            'freeze_cone': bamor.freeze_cone,
            'residual_uniform_prob': bamor.weight_sampler.residual_uniform_prob,
        }
    if use_learned_context:
        log['latent_context'] = {
            'latent_dim': LATENT_CONTEXT_DIM,
            'encoder': (
                'boundary_temporal_contrastive_transition_encoder'
                if use_boundary_context else (
                    'temporal_contrastive_transition_encoder'
                    if use_contrastive_context else 'transition_autoencoder'
                )
            ),
            'input': 'state,action,tanh(reward/3),next_state-state',
            'online_filter': 'ema_alpha=0.35',
        }
        if use_contrastive_context:
            log['latent_context'].update({
                'contrastive_pos_window': 64,
                'contrastive_temperature': 0.2,
                'contrastive_weight': 0.2,
            })
        if use_boundary_context:
            log['latent_context'].update({
                'pseudo_regime_labels': 'bamor_detector_current_mode',
                'pseudo_regime_contrastive_weight': 0.25,
                'reset_context_on_detected_switch': True,
            })
    if use_proto_context:
        log['prototype_belief_context'] = {
            'source': 'non_oracle_bamor_detector_labels',
            'feature_dim': _proto_feature_dim_for_method(method, env, r_dim),
            'context': 'soft_belief,max_probability,tanh(age/max_age)',
            'prototype_ema_alpha': 0.08,
            'belief_alpha': 0.35,
            'detector_prior': 0.20,
        }
    if use_sequence_context:
        log['sequence_context'] = {
            'latent_dim': LATENT_CONTEXT_DIM,
            'seq_len': SEQUENCE_CONTEXT_LEN,
            'encoder': 'gru_temporal_contrastive_transition_encoder'
                       if use_contrastive_context else 'gru_transition_autoencoder',
            'input': 'state,action,tanh(reward/3),tanh((next_state-state)/3)',
        }
    obs, info = env.reset()
    ep_r = np.zeros(r_dim)
    last_mode = 0
    _ref = ref_point

    for step in range(total_steps):
        active_mode = int(getattr(env, 'current_mode_idx', info.get('mode_idx', 0)))
        if bamor and bamor.mode_source == 'oracle':
            bamor.set_mode(active_mode, one_hot=True)

        # Weight sampling
        if bamor:
            w_np, w_anchor_np = bamor.sample_weight()
        elif method in UNIFORM_WEIGHT_METHODS:
            w_np = np.random.dirichlet(np.ones(r_dim)).astype(np.float32)
            w_anchor_np = None
        elif method == 'fixed':
            w_np = np.array([0.5, 0.5], np.float32)
            w_anchor_np = None
        else:
            w_np = np.random.dirichlet(np.ones(r_dim)).astype(np.float32)
            w_anchor_np = None
        c_np = _build_context(
            method, w_np, active_mode, env,
            bamor=bamor, trans_tracker=trans_tracker,
            generic_tracker=generic_tracker,
            learned_tracker=learned_tracker,
            proto_tracker=proto_tracker,
            seq_tracker=seq_tracker) if use_context else w_np

        # Action
        if step < 200:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                st = torch.FloatTensor(obs).unsqueeze(0).to(dev)
                ct = torch.FloatTensor(c_np).unsqueeze(0).to(dev)
                action, _ = pi.get_action(st, ct)
                action = action.cpu().numpy().flatten()

        nobs, rew, term, trunc, info = env.step(action)
        done = term or trunc
        bamor_info = None
        if bamor:
            if bamor.mode_source == 'oracle':
                bamor_info = bamor.observe_reward(rew, true_mode=active_mode)
            else:
                bamor_info = bamor.observe_reward(rew)
        if use_boundary_context and bamor_info and bamor_info.get('mode_switched'):
            if trans_tracker is not None:
                trans_tracker.reset()
            if generic_tracker is not None:
                generic_tracker.reset()
            if learned_tracker is not None:
                learned_tracker.reset()
        if use_transition_context:
            trans_tracker.update(obs, action, rew, nobs)
        if use_generic_context:
            generic_tracker.update(obs, action, rew, nobs)
        if use_learned_context:
            learned_tracker.update(obs, action, rew, nobs, dev)
        if use_proto_context:
            proto_label = (
                int(bamor.belief_tracker.current_mode)
                if bamor is not None else -1
            )
            proto_tracker.update(
                _prototype_feature(
                    method, trans_tracker=trans_tracker,
                    generic_tracker=generic_tracker,
                    learned_tracker=learned_tracker),
                detector_label=proto_label,
                mode_switched=bool(
                    bamor_info and bamor_info.get('mode_switched')))
        if use_sequence_context:
            seq_tracker.update(obs, action, rew, nobs, dev)
        next_mode = int(info.get('mode_idx', getattr(env, 'current_mode_idx', active_mode)))
        cn_np = _build_context(
            method, w_np, next_mode, env,
            bamor=bamor, trans_tracker=trans_tracker,
            generic_tracker=generic_tracker,
            learned_tracker=learned_tracker,
            proto_tracker=proto_tracker,
            seq_tracker=seq_tracker) if use_context else w_np
        if use_context:
            regime_label = (
                int(bamor.belief_tracker.current_mode)
                if use_boundary_context and bamor is not None else -1
            )
            buf.push(
                obs, action, w_np, c_np, rew, nobs, cn_np, done,
                t=step, regime=regime_label)
        else:
            buf.push(obs, action, w_np, rew, nobs, done)

        cur_mode = info.get('mode_idx', 0)
        if cur_mode != last_mode:
            log['mode_switches'].append(step)
            last_mode = cur_mode

        ep_r += rew
        if done:
            obs, info = env.reset()
            ep_r[:] = 0
        else:
            obs = nobs

        # Train
        if buf.size >= bs and step >= 200 and step % 4 == 0:
            if use_context:
                s, a, w, c, r, sn, cn, d = buf.sample(bs, dev)
            else:
                s, a, w, r, sn, d = buf.sample(bs, dev)
                c = cn = w
            with torch.no_grad():
                na, nlp = pi.get_action(sn, cn)
                q1n = q1t(sn, na, cn); q2n = q2t(sn, na, cn)
                qn  = torch.min(q1n, q2n)
                if bamor:
                    bamor.record_target_q(qn.mean(0).cpu().numpy())
                qn_s  = (w * qn).sum(-1, keepdim=True)
                tgt_s = (w * r).sum(-1, keepdim=True) + gamma * (1-d) * (qn_s - alpha * nlp)
            q1v = (w * q1(s, a, c)).sum(-1, keepdim=True)
            q2v = (w * q2(s, a, c)).sum(-1, keepdim=True)
            qloss = F.mse_loss(q1v, tgt_s) + F.mse_loss(q2v, tgt_s)
            q_opt.zero_grad(); qloss.backward(); q_opt.step()

            if use_learned_context:
                enc_r = torch.tanh(r / 3.0)
                z, pred_r, pred_delta = encoder(s, a, enc_r, sn)
                target_delta = torch.tanh((sn - s) / 3.0)
                enc_loss = (
                    F.mse_loss(pred_r, enc_r)
                    + F.mse_loss(pred_delta, target_delta)
                    + 1e-4 * z.pow(2).mean()
                )
                if use_contrastive_context:
                    es, ea, er, esn, et, ey = buf.sample_encoder(bs, dev)
                    er = torch.tanh(er / 3.0)
                    ez = encoder.encode(es, ea, er, esn)
                    ezp = encoder.project(ez)
                    enc_loss = enc_loss + 0.2 * temporal_contrastive_loss(
                        ezp, et, pos_window=64, temperature=0.2)
                    if use_boundary_context:
                        enc_loss = enc_loss + 0.25 * pseudo_regime_contrastive_loss(
                            ezp, ey, temperature=0.2)
                enc_opt.zero_grad(); enc_loss.backward(); enc_opt.step()

            if use_sequence_context and buf.size >= SEQUENCE_CONTEXT_LEN:
                qs, qa, qr, qsn, qt = buf.sample_sequences(
                    bs, SEQUENCE_CONTEXT_LEN, dev)
                qz, qpred_r, qpred_delta = seq_encoder(qs, qa, qr, qsn)
                qtarget_r = torch.tanh(qr[:, -1] / 3.0)
                qtarget_delta = torch.tanh((qsn[:, -1] - qs[:, -1]) / 3.0)
                seq_loss = (
                    F.mse_loss(qpred_r, qtarget_r)
                    + F.mse_loss(qpred_delta, qtarget_delta)
                    + 1e-4 * qz.pow(2).mean()
                )
                if use_contrastive_context:
                    seq_loss = seq_loss + 0.2 * temporal_contrastive_loss(
                        seq_encoder.project(qz), qt, pos_window=64, temperature=0.2)
                seq_opt.zero_grad(); seq_loss.backward(); seq_opt.step()

            na2, lp2 = pi.get_action(s, c)
            q_s   = (w * torch.min(q1(s, na2, c), q2(s, na2, c))).sum(-1, keepdim=True)
            ploss = (alpha * lp2 - q_s).mean()
            # Anchor-shifted regularization changes the scalarization geometry
            # over weights; as a policy-loss add-on it is constant w.r.t. the
            # actor for replayed weights, so the diagnostic implementation keeps
            # the actor gradient focused on the sampled scalarized Q objective.
            pi_opt.zero_grad(); ploss.backward(); pi_opt.step()

            for p, pt in zip(q1.parameters(), q1t.parameters()): pt.data.lerp_(p.data, tau)
            for p, pt in zip(q2.parameters(), q2t.parameters()): pt.data.lerp_(p.data, tau)

        # Periodic eval
        if eval_freq is None:
            eval_freq = max(total_steps // 10, 1)
        if eval_freq > 0 and (step + 1) % eval_freq == 0:
            if use_oracle_context:
                pareto, hv, _ref = eval_pareto_context_oracle(
                    eval_env, pi, r_dim, dev, env.num_modes, ref_point=_ref)
            elif use_sequence_context:
                pareto, hv, _ref = eval_pareto_context_sequence(
                    eval_env, pi, method, bamor, seq_encoder, r_dim, dev, ref_point=_ref)
            elif use_learned_context:
                pareto, hv, _ref = eval_pareto_context_learned(
                    eval_env, pi, method, bamor, encoder, r_dim, dev, ref_point=_ref)
            elif use_transition_context:
                pareto, hv, _ref = eval_pareto_context_transition(
                    eval_env, pi, method, bamor, r_dim, dev, ref_point=_ref)
            elif use_generic_context:
                pareto, hv, _ref = eval_pareto_context_generic(
                    eval_env, pi, method, bamor, r_dim, dev, ref_point=_ref)
            elif use_belief_context:
                pareto, hv, _ref = eval_pareto_context_belief(
                    eval_env, pi, bamor, r_dim, dev, ref_point=_ref)
            else:
                pareto, hv, _ref = eval_pareto(eval_env, pi, r_dim, dev, ref_point=_ref)
            log['hv_curve'].append({'step': step + 1, 'hv': hv})
            if bamor:
                m = bamor.get_metrics(w_np)
                dl, dg = float(m.get('delta_local', 0)), float(m.get('delta_global', 0))
                ratio  = dl / (dg + 1e-8)
            else:
                dl, dg, ratio = 0., 0., 0.
            print(f'  [{method}|s={seed}] step={step+1:>6d} HV={hv:.1f} '
                  f'Δl/Δg={dl:.3f}/{dg:.3f}({ratio:.2f})')

    # Final eval (overall + per-mode)
    if use_oracle_context:
        overall, hv_overall, _ref = eval_pareto_context_oracle(
            eval_env, pi, r_dim, dev, env.num_modes, num_w=21, ref_point=_ref)
    elif use_sequence_context:
        overall, hv_overall, _ref = eval_pareto_context_sequence(
            eval_env, pi, method, bamor, seq_encoder, r_dim, dev, num_w=21, ref_point=_ref)
    elif use_learned_context:
        overall, hv_overall, _ref = eval_pareto_context_learned(
            eval_env, pi, method, bamor, encoder, r_dim, dev, num_w=21, ref_point=_ref)
    elif use_transition_context:
        overall, hv_overall, _ref = eval_pareto_context_transition(
            eval_env, pi, method, bamor, r_dim, dev, num_w=21, ref_point=_ref)
    elif use_generic_context:
        overall, hv_overall, _ref = eval_pareto_context_generic(
            eval_env, pi, method, bamor, r_dim, dev, num_w=21, ref_point=_ref)
    elif use_belief_context:
        overall, hv_overall, _ref = eval_pareto_context_belief(
            eval_env, pi, bamor, r_dim, dev, num_w=21, ref_point=_ref)
    else:
        overall, hv_overall, _ref = eval_pareto(eval_env, pi, r_dim, dev,
                                                num_w=21, ref_point=_ref)
    log['hv_overall'] = hv_overall
    log['pareto_overall'] = overall
    log['pareto_per_mode'] = []
    log['hv_per_mode'] = []

    for m in range(env.num_modes):
        eval_env._lock_mode = True
        eval_env.current_mode_idx = m
        # Per-mode: use GLOBAL_REF so HV values are comparable across methods
        if use_oracle_context:
            pm, hv_m, _ = eval_pareto_context_oracle(
                eval_env, pi, r_dim, dev, env.num_modes, num_w=11, ref_point=GLOBAL_REF)
        elif use_sequence_context:
            pm, hv_m, _ = eval_pareto_context_sequence(
                eval_env, pi, method, bamor, seq_encoder, r_dim, dev, num_w=11, ref_point=GLOBAL_REF)
        elif use_learned_context:
            pm, hv_m, _ = eval_pareto_context_learned(
                eval_env, pi, method, bamor, encoder, r_dim, dev, num_w=11, ref_point=GLOBAL_REF)
        elif use_transition_context:
            pm, hv_m, _ = eval_pareto_context_transition(
                eval_env, pi, method, bamor, r_dim, dev, num_w=11, ref_point=GLOBAL_REF)
        elif use_generic_context:
            pm, hv_m, _ = eval_pareto_context_generic(
                eval_env, pi, method, bamor, r_dim, dev, num_w=11, ref_point=GLOBAL_REF)
        elif use_belief_context:
            pm, hv_m, _ = eval_pareto_context_belief(
                eval_env, pi, bamor, r_dim, dev, num_w=11, ref_point=GLOBAL_REF)
        else:
            pm, hv_m, _ = eval_pareto(eval_env, pi, r_dim, dev, num_w=11, ref_point=GLOBAL_REF)
        eval_env._lock_mode = False
        returns = np.array([p['return'] for p in pm])
        corr = np.corrcoef(returns[:, 0], returns[:, 1])[0, 1] if len(returns) > 2 else 0
        log['pareto_per_mode'].append(pm)
        log['hv_per_mode'].append(hv_m)
        print(f'    Mode {m}: HV={hv_m:.2f}  corr={corr:+.2f}')

    # Diagnostic: BAMOR's own detected-switch count (for comparison with env's true switches)
    if bamor is not None:
        log['bamor_switches_detected'] = int(getattr(bamor.belief_tracker, 'mode_switch_count', -1))
        log['env_switches_true'] = len(log['mode_switches'])
        print(f'    BAMOR detected {log["bamor_switches_detected"]} switches (env truth: {log["env_switches_true"]})')

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{method}_seed{seed}.json')
    with open(path, 'w') as f:
        json.dump(log, f, indent=2, default=str)
    print(f'  Saved → {path}')
    env.close(); eval_env.close()
    return log, _ref


# ════════════════════════════════════════════════════════════════════════════
# CAPQL wrapper
# ════════════════════════════════════════════════════════════════════════════

def train_capql(total_steps, switch_interval, seed, save_dir,
                ref_point=None, device='cpu'):
    try:
        from morl_baselines.multi_policy.capql.capql import CAPQL
    except ImportError as _e:
        print(f'  [capql] import failed ({_e}) — skipping')
        return None, ref_point

    _modes = get_canonical_modes()
    env      = MOGymWrapper(switch_interval=switch_interval, seed=seed,      modes=_modes)
    eval_env = MOGymWrapper(switch_interval=switch_interval, seed=seed+1000, modes=_modes)
    r_dim    = env.reward_dim

    _ref_np = np.array(ref_point) if ref_point is not None else np.array(GLOBAL_REF)

    agent = CAPQL(env, learning_rate=3e-4, gamma=0.99, batch_size=128,
                  net_arch=[64, 64], log=False, seed=seed)

    print(f'  [capql|s={seed}] training {total_steps} steps...')
    agent.train(total_timesteps=total_steps,
                eval_env=eval_env,
                ref_point=_ref_np,
                num_eval_weights_for_front=21,
                num_eval_episodes_for_front=3,
                eval_freq=max(total_steps // 10, 1000),
                checkpoints=False)

    # Eval
    overall, hv_overall, _ref = eval_front_agent(eval_env, agent, r_dim,
                                                  ref_point=ref_point)
    log = {'method': 'capql', 'seed': seed,
           'hv_overall': hv_overall, 'pareto_overall': overall,
           'pareto_per_mode': [], 'hv_per_mode': []}

    print(f'  [capql|s={seed}] HV={hv_overall:.1f}')
    for m in range(eval_env.num_modes):
        eval_env._lock_mode = True
        eval_env.current_mode_idx = m
        # Per-mode: use GLOBAL_REF so HV is comparable across methods
        pm, hv_m, _ = eval_front_agent(eval_env, agent, r_dim, ref_point=GLOBAL_REF)
        eval_env._lock_mode = False
        log['pareto_per_mode'].append(pm)
        log['hv_per_mode'].append(hv_m)
        print(f'    Mode {m}: HV={hv_m:.2f}')

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'capql_seed{seed}.json')
    with open(path, 'w') as f:
        json.dump(log, f, indent=2, default=str)
    print(f'  Saved → {path}')
    env.close(); eval_env.close()
    return log, _ref


# ════════════════════════════════════════════════════════════════════════════
# GPI-PD wrapper
# ════════════════════════════════════════════════════════════════════════════

def train_gpi_pd(total_steps, switch_interval, seed, save_dir,
                 ref_point=None, device='cpu'):
    try:
        from morl_baselines.multi_policy.gpi_pd.gpi_pd_continuous_action import GPIPDContinuousAction
    except ImportError as _e:
        print(f'  [gpi_pd] import failed ({_e}) — skipping')
        print('  [gpi_pd] hint: install pycddlib==2.1.6 (requires libcdd-dev system library)')
        return None, ref_point

    _modes = get_canonical_modes()
    env      = MOGymWrapper(switch_interval=switch_interval, seed=seed,      modes=_modes)
    eval_env = MOGymWrapper(switch_interval=switch_interval, seed=seed+1000, modes=_modes)
    r_dim    = env.reward_dim

    _ref_np = np.array(ref_point) if ref_point is not None else np.array(GLOBAL_REF)

    agent = GPIPDContinuousAction(
        env,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=128,
        net_arch=[64, 64],
        buffer_size=100_000,
        learning_starts=200,
        gradient_updates=1,
        use_gpi=True,         # enable GPI weight selection
        log=False,
        seed=seed,
        device=device,
    )

    print(f'  [gpi_pd|s={seed}] training {total_steps} steps...')
    agent.train(total_timesteps=total_steps,
                eval_env=eval_env,
                ref_point=_ref_np,
                num_eval_weights_for_front=21,
                num_eval_episodes_for_front=3,
                timesteps_per_iter=max(total_steps // 10, 1000),
                eval_freq=max(total_steps // 10, 1000),
                checkpoints=False)

    # Eval
    overall, hv_overall, _ref = eval_front_agent(eval_env, agent, r_dim,
                                                  ref_point=ref_point)
    log = {'method': 'gpi_pd', 'seed': seed,
           'hv_overall': hv_overall, 'pareto_overall': overall,
           'pareto_per_mode': [], 'hv_per_mode': []}

    print(f'  [gpi_pd|s={seed}] HV={hv_overall:.1f}')
    for m in range(eval_env.num_modes):
        eval_env._lock_mode = True
        eval_env.current_mode_idx = m
        pm, hv_m, _ = eval_front_agent(eval_env, agent, r_dim, ref_point=_ref)
        eval_env._lock_mode = False
        log['pareto_per_mode'].append(pm)
        log['hv_per_mode'].append(hv_m)
        print(f'    Mode {m}: HV={hv_m:.2f}')

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'gpi_pd_seed{seed}.json')
    with open(path, 'w') as f:
        json.dump(log, f, indent=2, default=str)
    print(f'  Saved → {path}')
    env.close(); eval_env.close()
    return log, _ref


# ════════════════════════════════════════════════════════════════════════════
# Summary table
# ════════════════════════════════════════════════════════════════════════════

def print_summary(all_logs, num_modes):
    methods = list(all_logs.keys())
    print(f'\n{"="*65}')
    print(f'  Results summary  (mean ± std across seeds)')
    print(f'{"="*65}')
    print(f'  {"Method":<12} {"HV overall":>12}  ' +
          '  '.join(f'HV Mode{m}' for m in range(num_modes)))
    print(f'  {"-"*63}')
    for method, logs in all_logs.items():
        if not logs:
            continue
        hvs_overall = [l['hv_overall'] for l in logs if l and 'hv_overall' in l]
        hvs_mode    = [[l['hv_per_mode'][m] for l in logs if l and 'hv_per_mode' in l]
                       for m in range(num_modes)]
        if not hvs_overall:
            continue
        overall_str = f'{np.mean(hvs_overall):>8.1f}±{np.std(hvs_overall):<5.1f}'
        mode_strs   = [f'{np.mean(hm):>6.1f}±{np.std(hm):<4.1f}'
                       for hm in hvs_mode]
        print(f'  {method:<12} {overall_str}  ' + '  '.join(mode_strs))
    print(f'{"="*65}')


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

SINGLE_POLICY = [
    'uniform_ctx_oracle',
    'bamor',
    'bamor_nc',
    'bamor_ns',
    'bamor_oracle',
    'bamor_oracle_cone',
    'bamor_fixed_cone',
    'bamor_ns_eps',
    'bamor_ctx',
    'bamor_ctx_nc',
    'bamor_ctx_fixed_cone',
    'bamor_ctx_oracle',
    'bamor_ctx_oracle_cone',
    'bamor_fastdet',
    'bamor_fastdet_ctx',
    'bamor_fastdet_ctx_nc',
    'uniform_trans_ctx',
    'bamor_fastdet_trans_ctx',
    'bamor_fastdet_trans_ctx_nc',
    'bamor_v2',
    'bamor_v2_belief',
    'bamor_v2_cone',
    'bamor_v2_hybrid',
    'bamor_v2_hybrid_belief',
    'bamor_v2_hybrid_cone',
    'bamor_v2_contrast',
    'bamor_v2_contrast_hybrid',
    'bamor_v2_contrast_hybrid_cone',
    'bamor_v2_generic',
    'bamor_v2_contrast_generic',
    'bamor_v2_contrast_generic_cone',
    'bamor_v2_boundary_hybrid',
    'bamor_v2_boundary_hybrid_cone',
    'bamor_v2_boundary_generic',
    'bamor_v2_proto_hybrid',
    'bamor_v2_proto_hybrid_cone',
    'bamor_v2_proto_generic',
    'bamor_v2_seq',
    'bamor_v2_seq_contrast',
    'bamor_v2_seq_contrast_cone',
    'capql',
    'uniform',
    'fixed',
]
MULTI_POLICY  = ['gpi_pd']
ALL_METHODS   = SINGLE_POLICY + MULTI_POLICY


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', default='all',
                   choices=ALL_METHODS + ['all', 'single_policy', 'multi_policy'])
    p.add_argument('--total_steps',    type=int, default=50_000)
    p.add_argument('--switch_interval',type=int, default=300)
    p.add_argument('--seed',           type=int, default=0)
    p.add_argument('--seeds',          type=int, default=3,
                   help='Number of seeds starting from --seed')
    p.add_argument('--save_dir',       default='results/compare')
    p.add_argument('--device',         default='cpu')
    p.add_argument('--eval_freq',      type=int, default=None,
                   help='Periodic eval frequency in env steps; 0 disables periodic eval')
    args = p.parse_args()

    if args.method == 'all':
        methods = ALL_METHODS
    elif args.method == 'single_policy':
        methods = SINGLE_POLICY
    elif args.method == 'multi_policy':
        methods = MULTI_POLICY
    else:
        methods = [args.method]

    seeds = list(range(args.seed, args.seed + args.seeds))

    print(f'Methods : {methods}')
    print(f'Seeds   : {seeds}')
    print(f'Steps   : {args.total_steps}')
    print(f'Switch  : every {args.switch_interval} steps')
    print(f'Save    : {args.save_dir}')

    all_logs = {m: [] for m in methods}
    shared_ref = GLOBAL_REF   # fixed conservative ref for fair cross-method HV

    for seed in seeds:
        for method in methods:
            print(f'\n{"="*55}\n  {method}  seed={seed}\n{"="*55}')

            if method in BAMOR_FAMILY or method in (
                'uniform', 'uniform_ctx_oracle', 'uniform_trans_ctx',
                'bamor_v2', 'bamor_v2_hybrid',
                'bamor_v2_contrast', 'bamor_v2_contrast_hybrid',
                'bamor_v2_generic', 'bamor_v2_contrast_generic',
                'bamor_v2_proto_hybrid', 'bamor_v2_proto_generic',
                'bamor_v2_seq', 'bamor_v2_seq_contrast',
                'fixed'):
                log, shared_ref = train_bamor_family(
                    method, args.total_steps, args.switch_interval,
                    seed, args.save_dir, args.device, ref_point=shared_ref,
                    eval_freq=args.eval_freq)

            elif method == 'capql':
                log, shared_ref = train_capql(
                    args.total_steps, args.switch_interval,
                    seed, args.save_dir, ref_point=shared_ref, device=args.device)

            elif method == 'gpi_pd':
                log, shared_ref = train_gpi_pd(
                    args.total_steps, args.switch_interval,
                    seed, args.save_dir, ref_point=shared_ref, device=args.device)

            if log is not None:
                all_logs[method].append(log)

    # Save combined
    combined_path = os.path.join(args.save_dir, 'all_results.json')
    os.makedirs(args.save_dir, exist_ok=True)
    with open(combined_path, 'w') as f:
        json.dump(all_logs, f, indent=2, default=str)
    print(f'\nAll results → {combined_path}')

    # Print summary table
    num_modes = 3   # SyntheticMOEnv default
    print_summary(all_logs, num_modes)


if __name__ == '__main__':
    main()
