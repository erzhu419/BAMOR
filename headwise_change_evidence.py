"""Frozen headwise evidence with separately learned normal residual scales."""

import numpy as np
import torch
from torch import nn

from conditional_change_evidence import ConditionalChangeEvidence, CONFIG as BASE_CONFIG


CONFIG = {**BASE_CONFIG, "updates": 2000, "scale_updates": 2000,
          "residual_floor": 1e-6, "action_validation_mse": 1e-4}
ARMS = ("global_scale", "conditional_scale")


def action_basis(actions):
    actions = np.asarray(actions, np.float64)
    return np.column_stack([np.ones(len(actions)), actions, actions ** 2])


class HeadwiseChangeEvidence:
    predictor_type = ConditionalChangeEvidence

    def __init__(self, input_dim, output_dim, reward_dim, **kwargs):
        self.input_dim, self.output_dim, self.reward_dim = input_dim, output_dim, reward_dim
        self.obs_dim = output_dim - reward_dim
        self.act_dim = input_dim - self.obs_dim
        assert self.obs_dim > 0 and self.act_dim > 0
        self.config = {**CONFIG, **kwargs}
        self.head_names = [f"reward_{index}" for index in range(reward_dim)] + ["dynamics"]

    def fit(self, inputs, targets, split_at, seed):
        inputs, targets = np.asarray(inputs, np.float32), np.asarray(targets, np.float32)
        assert inputs.shape == (len(targets), self.input_dim)
        assert targets.shape[1] == self.output_dim and 0 < split_at < len(inputs)
        predictor_config = {key: self.config[key] for key in BASE_CONFIG}
        self.reward_predictor = self.predictor_type(
            self.input_dim, self.reward_dim, self.reward_dim, **predictor_config)
        self.dynamics_predictor = self.predictor_type(
            self.input_dim, self.obs_dim, self.obs_dim, **predictor_config)
        reward_work = self.reward_predictor.fit(inputs, targets[:, :self.reward_dim], seed)
        dynamics_work = self.dynamics_predictor.fit(inputs, targets[:, self.reward_dim:], seed + 1)
        self.x_mean, self.x_scale = self.reward_predictor.x_mean, self.reward_predictor.x_scale
        self.y_scale = np.concatenate([self.reward_predictor.y_scale, self.dynamics_predictor.y_scale])
        self.reference = self.reward_predictor.reference
        basis = action_basis(inputs[:, self.obs_dim:])
        reward = targets[:, :self.reward_dim]
        errors = []
        # A reduced footprint must predict the other whole fit source in both directions.
        for train, validation in ((slice(None, split_at), slice(split_at, None)),
                                  (slice(split_at, None), slice(None, split_at))):
            coefficients = np.linalg.lstsq(basis[train], reward[train], rcond=None)[0]
            residual = (basis[validation] @ coefficients - reward[validation]) / self.y_scale[:self.reward_dim]
            errors.append(np.mean(residual ** 2, axis=0))
        self.action_heads = np.max(errors, axis=0) <= self.config["action_validation_mse"]
        self.action_coefficients = np.linalg.lstsq(basis, reward, rcond=None)[0]
        self.domains = ["action" if accepted else "state_action" for accepted in self.action_heads] + ["state_action"]
        self.predictor_optimizer_steps = reward_work["optimizer_steps"] + dynamics_work["optimizer_steps"]
        return dict(optimizer_steps=self.predictor_optimizer_steps, reward=reward_work,
                    dynamics=dynamics_work, action_validation_mse=np.asarray(errors).tolist(),
                    action_heads=self.action_heads.tolist(), support_domains=self.domains)

    def _features(self, inputs, head):
        normalized = (np.asarray(inputs, np.float32) - self.x_mean) / self.x_scale
        return normalized[:, self.obs_dim:] if self.domains[head] == "action" else normalized

    def raw_losses(self, inputs, targets):
        prediction = np.concatenate([self.reward_predictor.predict(inputs),
                                     self.dynamics_predictor.predict(inputs)], axis=1)
        action_prediction = action_basis(np.asarray(inputs)[:, self.obs_dim:]) @ self.action_coefficients
        for index in np.flatnonzero(self.action_heads):
            prediction[:, index] = action_prediction[:, index]
        squared = ((prediction - np.asarray(targets)) / self.y_scale) ** 2
        return np.column_stack([squared[:, :self.reward_dim], squared[:, self.reward_dim:].mean(1)])

    def fit_scales(self, inputs, targets, seed):
        return self.fit_scale_losses(inputs, self.raw_losses(inputs, targets), seed)

    def fit_scale_losses(self, inputs, losses, seed):
        inputs, losses = np.asarray(inputs, np.float32), np.asarray(losses, np.float32)
        assert inputs.shape == (len(losses), self.input_dim)
        assert losses.shape[1] == len(self.head_names)
        self.global_scales = np.maximum(losses.mean(0), self.config["residual_floor"])
        self.scale_models, work = [], []
        for head in range(len(self.head_names)):
            features = torch.as_tensor(self._features(inputs, head))
            normalized_loss = torch.as_tensor(losses[:, head] / self.global_scales[head], dtype=torch.float32)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + head)
                hidden = self.config["hidden"]
                model = nn.Sequential(nn.Linear(features.shape[1], hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Softplus())
                optimizer = torch.optim.Adam(model.parameters(), lr=self.config["lr"],
                                             weight_decay=self.config["weight_decay"])
                for _ in range(self.config["scale_updates"]):
                    indices = torch.randint(len(features), (self.config["batch_size"],))
                    scale = model(features[indices]).squeeze(1) + 1e-3
                    loss = (normalized_loss[indices] / scale + scale.log()).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            model.eval().requires_grad_(False)
            self.scale_models.append(model)
            work.append(dict(head=self.head_names[head], optimizer_steps=self.config["scale_updates"],
                             final_minibatch_loss=float(loss.detach())))
        return dict(optimizer_steps=sum(row["optimizer_steps"] for row in work), heads=work,
                    global_scales=self.global_scales.tolist())

    @torch.no_grad()
    def scales(self, inputs, arm):
        assert arm in ARMS
        if arm == "global_scale":
            return np.broadcast_to(self.global_scales, (len(inputs), len(self.head_names)))
        multipliers = np.column_stack([
            model(torch.as_tensor(self._features(inputs, head))).squeeze(1).numpy() + 1e-3
            for head, model in enumerate(self.scale_models)])
        return np.maximum(multipliers * self.global_scales, self.config["residual_floor"])

    @torch.no_grad()
    def distances(self, inputs):
        full = torch.as_tensor(self._features(inputs, len(self.head_names) - 1))
        full_distance = torch.cat([torch.cdist(chunk, self.reference).min(1).values
                                   for chunk in full.split(128)]).numpy()
        if any(self.action_heads):
            actions = full[:, self.obs_dim:]
            reference = self.reference[:, self.obs_dim:]
            action_distance = torch.cat([torch.cdist(chunk, reference).min(1).values
                                         for chunk in actions.split(128)]).numpy()
        return np.column_stack([action_distance if domain == "action" else full_distance
                                for domain in self.domains])

    def _windows(self, losses, supported):
        width = self.config["window"]
        assert len(losses) % width == 0
        windows = []
        for start in range(0, len(losses), width):
            mask = supported[start:start + width]
            heads = {}
            for index, name in enumerate(self.head_names):
                eligible = bool(mask[:, index].mean() >= self.config["minimum_supported_fraction"])
                statistic = float(losses[start:start + width, index][mask[:, index]].mean()) if eligible else None
                heads[name] = dict(eligible=eligible, statistic=statistic,
                                  supported_steps=int(mask[:, index].sum()), domain=self.domains[index])
            values = [row["statistic"] for row in heads.values() if row["eligible"]]
            windows.append(dict(start_step=start + 1, end_step=start + width,
                supported_steps=int(mask.any(1).sum()), eligible=bool(values),
                statistic=max(values) if values else None, heads=heads))
        return windows

    def calibrate(self, inputs, targets):
        distances = self.distances(inputs)
        self.radii = np.quantile(distances, self.config["quantile"], axis=0)
        supported = distances <= self.radii
        raw = self.raw_losses(inputs, targets)
        self.thresholds, report = {}, {}
        for arm in ARMS:
            windows = self._windows(raw / self.scales(inputs, arm), supported)
            eligible = [window["statistic"] for window in windows if window["eligible"]]
            assert len(eligible) >= 2, "Calibration must contain multiple supported windows"
            self.thresholds[arm] = float(np.quantile(eligible, self.config["quantile"]))
            report[arm] = dict(error_threshold=self.thresholds[arm],
                support_radii=dict(zip(self.head_names, self.radii.tolist())),
                eligible_windows=len(eligible), total_windows=len(windows), windows=windows)
        return report

    def evaluate(self, inputs, targets, arm):
        distances = self.distances(inputs)
        head_supported = distances <= self.radii
        errors = self.raw_losses(inputs, targets) / self.scales(inputs, arm)
        windows = self._windows(errors, head_supported)
        for window in windows:
            window["alarm"] = bool(window["eligible"] and window["statistic"] > self.thresholds[arm])
            for row in window["heads"].values():
                row["alarm"] = bool(row["eligible"] and row["statistic"] > self.thresholds[arm])
        return dict(windows=windows, supported=head_supported.any(1), head_supported=head_supported,
                    errors=errors, distances=distances)
