"""Normal-fit affine prediction with a frozen nonlinear residual correction."""

import numpy as np
import torch
from torch import nn

from conditional_change_evidence import ConditionalChangeEvidence
from headwise_change_evidence import HeadwiseChangeEvidence, action_basis


class AffineResidualPredictor(ConditionalChangeEvidence):
    def fit(self, inputs, targets, seed):
        inputs, targets = np.asarray(inputs, np.float32), np.asarray(targets, np.float32)
        assert inputs.shape == (len(targets), self.input_dim)
        assert targets.shape[1] == self.output_dim
        floor = self.config["scale_floor"]
        self.x_mean, self.y_mean = inputs.mean(0), targets.mean(0)
        self.x_scale = np.maximum(inputs.std(0), floor)
        self.y_scale = np.maximum(targets.std(0), floor)
        normalized_x = (inputs - self.x_mean) / self.x_scale
        normalized_y = (targets - self.y_mean) / self.y_scale
        self.reference = torch.as_tensor(normalized_x)
        basis = np.column_stack([np.ones(len(inputs)), normalized_x]).astype(np.float64)
        coefficients, _, rank, singular = np.linalg.lstsq(basis, normalized_y, rcond=None)
        self.affine_coefficients = coefficients.astype(np.float32)
        residual_targets = torch.as_tensor(
            normalized_y - self._affine(normalized_x), dtype=torch.float32)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            hidden = self.config["hidden"]
            self.model = nn.Sequential(nn.Linear(self.input_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, self.output_dim))
            # Start from the fitted affine predictor, not an arbitrary correction.
            nn.init.zeros_(self.model[-1].weight)
            nn.init.zeros_(self.model[-1].bias)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"],
                                         weight_decay=self.config["weight_decay"])
            for _ in range(self.config["updates"]):
                indices = torch.randint(len(inputs), (self.config["batch_size"],))
                loss = (self.model(self.reference[indices]) - residual_targets[indices]).square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.model.eval().requires_grad_(False)
        self.optimizer_steps = self.config["updates"]
        return dict(optimizer_steps=self.optimizer_steps,
            parameters=sum(p.numel() for p in self.model.parameters()),
            affine_parameters=int(coefficients.size), affine_fit_rows=len(inputs),
            affine_rank=int(rank), affine_smallest_singular_value=float(singular[-1]),
            final_minibatch_normalized_mse=float(loss.detach()))

    def _affine(self, normalized):
        return normalized @ self.affine_coefficients[1:] + self.affine_coefficients[0]

    def affine_predict(self, inputs):
        normalized = (np.asarray(inputs, np.float32) - self.x_mean) / self.x_scale
        return self._affine(normalized) * self.y_scale + self.y_mean

    @torch.no_grad()
    def predict(self, inputs):
        normalized = (np.asarray(inputs, np.float32) - self.x_mean) / self.x_scale
        return (self._affine(normalized) + self.model(torch.as_tensor(normalized)).numpy()) * self.y_scale + self.y_mean


class AffineResidualHeadwiseEvidence(HeadwiseChangeEvidence):
    predictor_type = AffineResidualPredictor

    def affine_losses(self, inputs, targets):
        prediction = np.concatenate([self.reward_predictor.affine_predict(inputs),
                                     self.dynamics_predictor.affine_predict(inputs)], axis=1)
        actions = action_basis(np.asarray(inputs)[:, self.obs_dim:]) @ self.action_coefficients
        for index in np.flatnonzero(self.action_heads):
            prediction[:, index] = actions[:, index]
        squared = ((prediction - np.asarray(targets)) / self.y_scale) ** 2
        return np.column_stack([squared[:, :self.reward_dim], squared[:, self.reward_dim:].mean(1)])
