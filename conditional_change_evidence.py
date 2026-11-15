"""Frozen state/action-only prediction and held-out change evidence."""

import numpy as np
import torch
from torch import nn


CONFIG = dict(hidden=64, updates=500, batch_size=128, lr=1e-3,
              weight_decay=1e-4, scale_floor=1e-3, quantile=.99,
              window=100, minimum_supported_fraction=.8)


class ConditionalChangeEvidence:
    def __init__(self, input_dim, output_dim, reward_dim, **kwargs):
        self.input_dim, self.output_dim, self.reward_dim = input_dim, output_dim, reward_dim
        self.config = {**CONFIG, **kwargs}

    def fit(self, inputs, targets, seed):
        inputs, targets = np.asarray(inputs, np.float32), np.asarray(targets, np.float32)
        assert inputs.shape == (len(targets), self.input_dim)
        assert targets.shape[1] == self.output_dim
        floor = self.config["scale_floor"]
        self.x_mean, self.y_mean = inputs.mean(0), targets.mean(0)
        self.x_scale = np.maximum(inputs.std(0), floor)
        self.y_scale = np.maximum(targets.std(0), floor)
        self.reference = torch.as_tensor((inputs - self.x_mean) / self.x_scale)
        normalized_targets = torch.as_tensor((targets - self.y_mean) / self.y_scale)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            hidden = self.config["hidden"]
            self.model = nn.Sequential(nn.Linear(self.input_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, self.output_dim))
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"],
                                         weight_decay=self.config["weight_decay"])
            for _ in range(self.config["updates"]):
                indices = torch.randint(len(inputs), (self.config["batch_size"],))
                loss = (self.model(self.reference[indices]) - normalized_targets[indices]).square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.model.eval()
        self.model.requires_grad_(False)
        self.optimizer_steps = self.config["updates"]
        return dict(optimizer_steps=self.optimizer_steps,
                    parameters=sum(p.numel() for p in self.model.parameters()),
                    final_minibatch_normalized_mse=float(loss.detach()))

    @torch.no_grad()
    def predict(self, inputs):
        normalized = torch.as_tensor((np.asarray(inputs, np.float32) - self.x_mean) / self.x_scale)
        return self.model(normalized).numpy() * self.y_scale + self.y_mean

    @torch.no_grad()
    def distances(self, inputs):
        normalized = torch.as_tensor((np.asarray(inputs, np.float32) - self.x_mean) / self.x_scale)
        return torch.cat([torch.cdist(chunk, self.reference).min(1).values
                          for chunk in normalized.split(128)]).numpy()

    def errors(self, inputs, targets):
        residual = (self.predict(inputs) - np.asarray(targets)) / self.y_scale
        reward = np.mean(residual[:, :self.reward_dim] ** 2, axis=1)
        dynamics = np.mean(residual[:, self.reward_dim:] ** 2, axis=1)
        return np.maximum(reward, dynamics)

    def _windows(self, errors, supported):
        width = self.config["window"]
        assert len(errors) % width == 0
        windows = []
        for start in range(0, len(errors), width):
            mask = supported[start:start + width]
            eligible = bool(mask.mean() >= self.config["minimum_supported_fraction"])
            statistic = float(errors[start:start + width][mask].mean()) if eligible else None
            windows.append(dict(start_step=start + 1, end_step=start + width,
                supported_steps=int(mask.sum()), eligible=eligible, statistic=statistic))
        return windows

    def calibrate(self, inputs, targets):
        distances = self.distances(inputs)
        self.support_radius = float(np.quantile(distances, self.config["quantile"]))
        windows = self._windows(self.errors(inputs, targets), distances <= self.support_radius)
        eligible = [window["statistic"] for window in windows if window["eligible"]]
        assert len(eligible) >= 2, "Calibration must contain multiple supported windows"
        self.threshold = float(np.quantile(eligible, self.config["quantile"]))
        return dict(support_radius=self.support_radius, error_threshold=self.threshold,
                    eligible_windows=len(eligible), total_windows=len(windows), windows=windows)

    def evaluate(self, inputs, targets):
        distances = self.distances(inputs)
        supported = distances <= self.support_radius
        errors = self.errors(inputs, targets)
        windows = self._windows(errors, supported)
        for window in windows:
            window["alarm"] = bool(window["eligible"] and window["statistic"] > self.threshold)
        return dict(windows=windows, supported=supported, errors=errors, distances=distances)
