"""Normal-startup mean correction without changing frozen predictors or scales."""

import numpy as np

from headwise_change_evidence import action_basis
from run_sensitivity_guarded_selection import GuardedEvidence


class MeanOffsetEvidence(GuardedEvidence):
    def normalized_residuals(self, inputs, targets):
        prediction = np.concatenate([self.reward_predictor.predict(inputs),
            self.dynamics_predictor.predict(inputs)], axis=1)
        actions = action_basis(np.asarray(inputs)[:, self.obs_dim:]) @ self.action_coefficients
        for head in np.flatnonzero(self.action_heads):
            prediction[:, head] = actions[:, head]
        return (prediction - np.asarray(targets)) / self.y_scale

    def fit_offset(self, inputs, targets):
        residuals = self.normalized_residuals(inputs, targets)
        supported = self.distances(inputs) <= self.radii
        self.normalized_offset = np.zeros(self.output_dim, np.float32)
        heads = {}
        for head, name in enumerate(self.head_names):
            mask = supported[:, head]
            outputs = slice(head, head + 1) if head < self.reward_dim else slice(self.reward_dim, None)
            accepted = self.domains[head] == "state_action" and bool(mask.any())
            if accepted:
                self.normalized_offset[outputs] = residuals[mask, outputs].mean(0, dtype=np.float64)
            heads[name] = dict(domain=self.domains[head], supported_steps=int(mask.sum()),
                corrected=accepted, normalized_offset=self.normalized_offset[outputs].tolist())
        return dict(fit_steps=len(inputs), fit_calls=1, optimizer_steps=0, heads=heads,
            normalized_offset=self.normalized_offset.tolist(),
            physical_prediction_shift=(-self.normalized_offset * self.y_scale).tolist())

    def corrected_losses(self, inputs, targets):
        squared = (self.normalized_residuals(inputs, targets) - self.normalized_offset) ** 2
        return np.column_stack([squared[:, :self.reward_dim], squared[:, self.reward_dim:].mean(1)])
