"""Leave-whole-source-out normal residuals for conditional scale fitting."""

import numpy as np

from affine_residual_change_evidence import AffineResidualPredictor
from conditional_change_evidence import CONFIG as PREDICTOR_CONFIG
from headwise_change_evidence import action_basis


def cross_source_residuals(evidence, sources, seed):
    assert len(sources) == 2 and sources[0][2]["seed"] != sources[1][2]["seed"]
    inputs, losses, work = [], [], []
    config = {key: evidence.config[key] for key in PREDICTOR_CONFIG}
    for direction in range(2):
        train_x, train_y, train_meta = sources[1 - direction]
        held_x, held_y, held_meta = sources[direction]
        reward = AffineResidualPredictor(evidence.input_dim, evidence.reward_dim,
                                         evidence.reward_dim, **config)
        dynamics = AffineResidualPredictor(evidence.input_dim, evidence.obs_dim,
                                           evidence.obs_dim, **config)
        reward_work = reward.fit(train_x, train_y[:, :evidence.reward_dim], seed + 2 * direction)
        dynamics_work = dynamics.fit(train_x, train_y[:, evidence.reward_dim:], seed + 2 * direction + 1)
        prediction = np.concatenate([reward.predict(held_x), dynamics.predict(held_x)], axis=1)
        # Reuse validated footprints, but exclude held-source targets from coefficients.
        coefficients = np.linalg.lstsq(action_basis(train_x[:, evidence.obs_dim:]),
                                      train_y[:, :evidence.reward_dim], rcond=None)[0]
        action_prediction = action_basis(held_x[:, evidence.obs_dim:]) @ coefficients
        for head in np.flatnonzero(evidence.action_heads):
            prediction[:, head] = action_prediction[:, head]
        squared = ((prediction - held_y) / evidence.y_scale) ** 2
        head_losses = np.column_stack([squared[:, :evidence.reward_dim],
                                      squared[:, evidence.reward_dim:].mean(1)])
        inputs.append(held_x)
        losses.append(head_losses)
        work.append(dict(train_seed=train_meta["seed"], residual_seed=held_meta["seed"],
            train_steps=len(train_x), residual_steps=len(held_x), reward=reward_work, dynamics=dynamics_work,
            optimizer_steps=reward_work["optimizer_steps"] + dynamics_work["optimizer_steps"],
            normalized_head_mse=dict(zip(evidence.head_names, head_losses.mean(0).tolist()))))
    return np.concatenate(inputs), np.concatenate(losses), dict(
        optimizer_steps=sum(row["optimizer_steps"] for row in work), sources=work)
