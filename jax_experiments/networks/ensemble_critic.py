"""Vectorized ensemble Q-network — Multi-Objective (BAMOR).

Output: [ensemble_size, batch, num_objectives] Q-values.
Corresponds to BAMOR.lean: Q : S × A → Fin m → ℝ
"""
import jax
import jax.numpy as jnp
from flax import nnx


class SingleCritic(nnx.Module):
    """Single Q(s,a) → [batch, m] multi-objective network."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 num_objectives: int = 1, n_layers: int = 3, *, rngs: nnx.Rngs):
        layers = []
        in_d = obs_dim + act_dim
        for _ in range(n_layers):
            layers.append(nnx.Linear(in_d, hidden_dim, rngs=rngs))
            in_d = hidden_dim
        layers.append(nnx.Linear(in_d, num_objectives, rngs=rngs))
        self.layers = nnx.List(layers)
        self.n_hidden = n_layers
        self.num_objectives = num_objectives

    def __call__(self, obs, act):
        x = jnp.concatenate([obs, act], axis=-1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.n_hidden:
                x = nnx.relu(x)
        # x: [batch, m]
        if self.num_objectives == 1:
            return x.squeeze(-1)  # backward compat: [batch]
        return x  # [batch, m]


class EnsembleCritic(nnx.Module):
    """Ensemble of N independent multi-objective Q-networks.

    Forward pass returns:
      - num_objectives == 1: [N, batch] (backward compatible with BAPR)
      - num_objectives > 1:  [N, batch, m]
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 ensemble_size: int = 10, num_objectives: int = 1,
                 n_layers: int = 3, *, rngs: nnx.Rngs):
        critics = [
            SingleCritic(obs_dim, act_dim, hidden_dim,
                         num_objectives=num_objectives, n_layers=n_layers,
                         rngs=nnx.Rngs(params=rngs.params()))
            for _ in range(ensemble_size)
        ]
        self.critics = nnx.List(critics)
        self.ensemble_size = ensemble_size
        self.num_objectives = num_objectives

    def __call__(self, obs, act):
        """Returns [ensemble_size, batch, m] or [ensemble_size, batch]."""
        qs = jnp.stack([c(obs, act) for c in self.critics], axis=0)
        return qs

    def scalarize(self, obs, act, w):
        """Scalarize Q with preference weight w.

        Args:
            obs: [batch, obs_dim]
            act: [batch, act_dim]
            w: [m] preference weights
        Returns:
            [ensemble_size, batch] scalarized Q-values
        """
        qs = self(obs, act)  # [ensemble, batch, m]
        return (qs * w[None, None, :]).sum(axis=-1)  # [ensemble, batch]

    def compute_reg_norm(self):
        """L1 regularization norm per critic member. Returns [ensemble_size]."""
        norms = []
        for critic in self.critics:
            total = 0.0
            for layer in critic.layers:
                total = total + jnp.sum(jnp.abs(layer.kernel))
                if layer.bias is not None:
                    total = total + jnp.sum(jnp.abs(layer.bias))
            norms.append(total)
        return jnp.array(norms)

    def compute_reg_norm_per_objective(self):
        """Per-objective L1 norm. Returns [ensemble_size, m].

        Only the last layer is split per objective; hidden layers broadcast.
        Corresponds to BAMOR.lean: κ_ale(i)
        """
        m = self.num_objectives
        norms = []
        for critic in self.critics:
            hidden_total = 0.0
            output_total = jnp.zeros(m)
            for i, layer in enumerate(critic.layers):
                if i < len(critic.layers) - 1:
                    # Hidden layer: aggregate
                    hidden_total = hidden_total + jnp.sum(jnp.abs(layer.kernel))
                    if layer.bias is not None:
                        hidden_total = hidden_total + jnp.sum(jnp.abs(layer.bias))
                else:
                    # Output layer [hidden, m]: per-objective
                    output_total = output_total + jnp.sum(jnp.abs(layer.kernel), axis=0)
                    if layer.bias is not None:
                        output_total = output_total + jnp.abs(layer.bias).squeeze(0)
            # Broadcast hidden to all objectives
            norms.append(hidden_total + output_total)  # [m]
        return jnp.stack(norms, axis=0)  # [ensemble, m]
