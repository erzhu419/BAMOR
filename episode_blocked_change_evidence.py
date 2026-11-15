"""Frozen ensemble with input-dependent disagreement scaling."""

import numpy as np

from conditional_change_evidence import ConditionalChangeEvidence, CONFIG


ENSEMBLE_SIZE = 3
ENSEMBLE_CONFIG = {**CONFIG, "updates": 2000}


class EpisodeBlockedChangeEvidence(ConditionalChangeEvidence):
    def __init__(self, input_dim, output_dim, reward_dim, **kwargs):
        super().__init__(input_dim, output_dim, reward_dim, **{**ENSEMBLE_CONFIG, **kwargs})

    def fit(self, inputs, targets, seed):
        self.members, work = [], []
        for index in range(ENSEMBLE_SIZE):
            member = ConditionalChangeEvidence(self.input_dim, self.output_dim,
                                               self.reward_dim, **self.config)
            work.append(member.fit(inputs, targets, seed + index))
            self.members.append(member)
        first = self.members[0]
        self.x_mean, self.x_scale = first.x_mean, first.x_scale
        self.y_mean, self.y_scale = first.y_mean, first.y_scale
        self.reference = first.reference
        self.optimizer_steps = sum(row["optimizer_steps"] for row in work)
        return dict(optimizer_steps=self.optimizer_steps, members=work)

    def predict(self, inputs):
        return np.mean([member.predict(inputs) for member in self.members], axis=0)

    def components(self, inputs, targets):
        predictions = np.stack([member.predict(inputs) for member in self.members])
        residual = (predictions.mean(0) - targets) / self.y_scale
        variance = predictions.var(0) / self.y_scale ** 2
        squared = residual ** 2

        def combine(values):
            return np.maximum(values[:, :self.reward_dim].mean(1),
                              values[:, self.reward_dim:].mean(1))

        return dict(raw_errors=combine(squared), errors=combine(squared / (1. + variance)),
                    disagreement=combine(variance))

    def errors(self, inputs, targets):
        return self.components(inputs, targets)["errors"]
