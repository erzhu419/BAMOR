"""BAMOR: Multi-Objective BAPR + Target Cone + Anchor-Shifted Penalty — scan-fused.

Based on BAPR (ESCP + BOCD), extended with:
  - Multi-objective Q: [ensemble, batch, m]
  - w-conditioned policy (PSL paradigm)
  - Scalarized policy loss with anchor-shifted regularization (Part XI)
  - Mahalanobis surprise (Part IV) → belief update
  - Target Cone sampling (Part XII) for Focused Search

Corresponds to BAMOR.lean Parts I, IV, VII, IX, X, XI, XII.
"""
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax import nnx
from copy import deepcopy

from jax_experiments.networks.policy import GaussianPolicy
from jax_experiments.networks.ensemble_critic import EnsembleCritic
from jax_experiments.networks.context_net import ContextNetwork, compute_rmdm_loss
from jax_experiments.common.belief_tracker import BeliefTracker
from jax_experiments.common.mahalanobis import (
    MahalanobisSurpriseComputer, ContextAwareWeightSampler)


class BAMOR:
    """BAMOR = BAPR + Multi-Objective Q + Target Cone + Anchor Penalty — scan-fused."""

    def __init__(self, obs_dim: int, act_dim: int, config, seed: int = 0):
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_objectives = config.num_objectives
        self.rngs = nnx.Rngs(seed)

        # Networks
        self.context_net = ContextNetwork(
            obs_dim, config.ep_dim, hidden_dim=128, rngs=self.rngs)
        self.policy = GaussianPolicy(
            obs_dim, act_dim, config.hidden_dim,
            ep_dim=config.ep_dim, w_dim=config.num_objectives,
            n_layers=2, rngs=self.rngs)
        # Critic input = obs + ep; output = [ensemble, batch, m]
        self.critic = EnsembleCritic(
            obs_dim + config.ep_dim, act_dim, config.hidden_dim,
            ensemble_size=config.ensemble_size,
            num_objectives=config.num_objectives,
            n_layers=3, rngs=self.rngs)
        self.target_critic = deepcopy(self.critic)

        # Auto-alpha
        self.log_alpha = jnp.array(jnp.log(config.alpha))
        self.target_entropy = -float(act_dim)

        # Optimizers
        self.policy_opt = optax.adam(config.lr)
        self.critic_opt = optax.adam(config.lr)
        self.context_opt = optax.adam(config.lr)
        self.alpha_opt = optax.adam(config.lr)

        self.policy_opt_state = self.policy_opt.init(nnx.state(self.policy, nnx.Param))
        self.critic_opt_state = self.critic_opt.init(nnx.state(self.critic, nnx.Param))
        self.context_opt_state = self.context_opt.init(nnx.state(self.context_net, nnx.Param))
        self.alpha_opt_state = self.alpha_opt.init(self.log_alpha)

        # BOCD Belief + Mahalanobis Surprise + Target Cone
        self.belief_tracker = BeliefTracker(
            max_run_length=config.max_run_length,
            hazard_rate=config.hazard_rate,
            base_variance=config.base_variance,
            variance_growth=config.variance_growth)
        self.mahal_computer = MahalanobisSurpriseComputer(
            num_objectives=config.num_objectives,
            ema_alpha=config.surprise_ema_alpha)
        self.cone_sampler = ContextAwareWeightSampler()

        # Current preference weights (updated each multi_update)
        w_init = np.array(config.preference_weights, dtype=np.float32)
        self._current_w = w_init
        self._current_w_anchor = w_init.copy()

        self.update_count = 0
        self._current_weighted_lambda = 0.0
        self._build_scan_fn()

    def _build_scan_fn(self):
        gamma = self.config.gamma
        tau = self.config.tau
        beta_base = self.config.beta
        penalty_scale = self.config.penalty_scale
        eta_anchor = self.config.eta_anchor
        auto_alpha = self.config.auto_alpha
        target_entropy = jnp.array(self.target_entropy)
        rbf_r = self.config.rbf_radius
        cons_w = self.config.consistency_loss_weight
        div_w = self.config.diversity_loss_weight
        m = self.config.num_objectives

        gd_policy = nnx.graphdef(self.policy)
        gd_critic = nnx.graphdef(self.critic)
        gd_target = nnx.graphdef(self.target_critic)
        gd_ctx = nnx.graphdef(self.context_net)

        p_opt = self.policy_opt
        c_opt = self.critic_opt
        x_opt = self.context_opt
        a_opt = self.alpha_opt

        @jax.jit
        def _scan_update(critic_params, target_params, policy_params,
                         ctx_params, log_alpha,
                         c_opt_state, p_opt_state, x_opt_state, a_opt_state,
                         all_obs, all_act, all_rew, all_next_obs, all_done,
                         all_task_ids, rng_key, warmup, weighted_lambda,
                         w_jax, w_anchor_jax):
            """BAMOR scan: multi-objective Q + scalarized policy + anchor penalty."""
            effective_beta = beta_base - weighted_lambda * penalty_scale

            def body_fn(carry, batch_data):
                (c_p, t_p, p_p, x_p, la, c_os, p_os, x_os, a_os, key) = carry
                (obs, act, rew, next_obs, done, tids) = batch_data
                key, k1, k2 = jax.random.split(key, 3)
                alpha = jnp.exp(la)

                # === Context RMDM ===
                def ctx_loss_fn(xp):
                    xm = nnx.merge(gd_ctx, xp)
                    return compute_rmdm_loss(xm(obs), tids, rbf_r, cons_w, div_w)

                x_loss, x_grads = jax.value_and_grad(ctx_loss_fn)(x_p)
                x_upd, new_x_os = x_opt.update(x_grads, x_os, x_p)
                new_x_p = optax.apply_updates(x_p, x_upd)

                ctx_m = nnx.merge(gd_ctx, new_x_p)
                ep = ctx_m(obs)
                ep = jnp.where(warmup, jnp.zeros_like(ep), ep)
                next_ep = ctx_m(next_obs)
                next_ep = jnp.where(warmup, jnp.zeros_like(next_ep), next_ep)

                obs_aug = jnp.concatenate([obs, ep], axis=-1)
                next_aug = jnp.concatenate([next_obs, next_ep], axis=-1)

                # === Critic loss: per-component MSE on [ensemble, batch, m] ===
                def critic_loss_fn(cp):
                    tm = nnx.merge(gd_target, t_p)
                    pm = nnx.merge(gd_policy, p_p)
                    na, nlp = pm.sample(next_obs, k1, next_ep, w_jax)

                    # Target Q: [ensemble, batch, m] — independent per Q_i
                    tq_all = tm(next_aug, na)  # [E, B, m]
                    # Scalarize per ensemble member: [E, B]
                    tq_scalar = (tq_all * w_jax[None, None, :]).sum(axis=-1)
                    tq_scalar = tq_scalar - alpha * nlp  # [E, B]
                    # Independent target per Q_i per objective
                    tv_all = rew[None] + gamma * (1 - done[None]) * tq_scalar[:, :, None]  # [E, B, m]

                    cm = nnx.merge(gd_critic, cp)
                    pq = cm(obs_aug, act)  # [E, B, m]
                    # MSE per component, mean over all
                    return jnp.mean((pq - tv_all) ** 2), pq

                (c_loss, pq), c_grads = jax.value_and_grad(
                    critic_loss_fn, has_aux=True)(c_p)
                c_upd, new_c_os = c_opt.update(c_grads, c_os, c_p)
                new_c_p = optax.apply_updates(c_p, c_upd)

                # === Policy loss: scalarized LCB + anchor penalty (Part XI) ===
                def policy_loss_fn(pp):
                    pm = nnx.merge(gd_policy, pp)
                    cm = nnx.merge(gd_critic, new_c_p)
                    na, lp = pm.sample(obs, k2, ep, w_jax)
                    oa = jnp.concatenate([obs, ep], axis=-1)
                    qv = cm(oa, na)  # [E, B, m]
                    # Scalarize: [E, B]
                    qv_scalar = (qv * w_jax[None, None, :]).sum(axis=-1)
                    lcb = qv_scalar.mean(axis=0) + effective_beta * qv_scalar.std(axis=0)
                    base_loss = (jnp.exp(la) * lp - lcb).mean()

                    # Anchor-shifted regularization (Part XI)
                    anchor_penalty = eta_anchor * jnp.sum((w_jax - w_anchor_jax) ** 2)
                    return base_loss + anchor_penalty, lp

                (p_loss, lp), p_grads = jax.value_and_grad(
                    policy_loss_fn, has_aux=True)(p_p)
                p_upd, new_p_os = p_opt.update(p_grads, p_os, p_p)
                new_p_p = optax.apply_updates(p_p, p_upd)

                # === Alpha ===
                a_grad = -(lp.mean() + target_entropy)
                a_upd, new_a_os = a_opt.update(a_grad, a_os, la)
                new_la = jnp.where(auto_alpha, la + a_upd, la)
                new_a_os = jax.tree.map(
                    lambda n, o: jnp.where(auto_alpha, n, o), new_a_os, a_os)

                # === Target soft update ===
                new_t_p = jax.tree.map(
                    lambda tp, cp: tp * (1 - tau) + cp * tau, t_p, new_c_p)

                new_carry = (new_c_p, new_t_p, new_p_p, new_x_p, new_la,
                             new_c_os, new_p_os, new_x_os, new_a_os, key)

                # Metrics: per-objective Q mean/std for Mahalanobis
                pq_scalar = (pq * w_jax[None, None, :]).sum(axis=-1)  # [E, B]
                # Per-objective ensemble std: [m]
                pq_per_obj_std = pq.std(axis=0).mean(axis=0)  # [m]

                metrics = (c_loss, p_loss, x_loss, jnp.exp(new_la),
                           pq_scalar.mean(), pq_scalar.std(axis=0).mean(),
                           lp.mean(), pq_per_obj_std)
                return new_carry, metrics

            init = (critic_params, target_params, policy_params,
                    ctx_params, log_alpha,
                    c_opt_state, p_opt_state, x_opt_state, a_opt_state, rng_key)
            batches = (all_obs, all_act, all_rew, all_next_obs, all_done,
                       all_task_ids)
            return jax.lax.scan(body_fn, init, batches)

        self._scan_update = _scan_update

    @property
    def alpha(self):
        return jnp.exp(self.log_alpha)

    def select_action(self, obs, deterministic=False):
        obs_jax = jnp.array(obs)[None] if np.asarray(obs).ndim == 1 else jnp.array(obs)
        ep = self.context_net(obs_jax)
        w_jax = jnp.array(self._current_w)
        if deterministic:
            return np.array(self.policy.deterministic(obs_jax, ep, w_jax)[0])
        key = self.rngs.params()
        action, _ = self.policy.sample(obs_jax, key, ep, w_jax)
        return np.array(action[0])

    def reset_episode(self):
        self.belief_tracker.reset()
        self.mahal_computer.reset()

    def _compute_weighted_lambda(self):
        ew = self.belief_tracker.effective_window
        max_h = self.belief_tracker.max_H - 1
        return float(np.clip(ew / max_h, 0.0, 1.0))

    def multi_update(self, stacked_batch, current_iter=0, recent_rewards=None, **kwargs):
        rng_key = self.rngs.params()
        warmup = jnp.array(current_iter < self.config.context_warmup_iters)

        # --- Mahalanobis surprise from per-objective Q-std ---
        last_obs = jnp.array(stacked_batch["obs"][-1])
        last_act = jnp.array(stacked_batch["act"][-1])
        c_p = nnx.state(self.critic, nnx.Param)
        x_p = nnx.state(self.context_net, nnx.Param)

        # Compute per-objective Q-std on last batch
        ctx_m = self.context_net
        ep = ctx_m(last_obs)
        ep = jnp.where(warmup, jnp.zeros_like(ep), ep)
        obs_aug = jnp.concatenate([last_obs, ep], axis=-1)
        pq = self.critic(obs_aug, last_act)  # [E, B, m]
        per_obj_q_std = np.array(pq.std(axis=0).mean(axis=0))  # [m]

        # Mahalanobis surprise → belief update
        mahal_surprise, per_obj_surp = self.mahal_computer.compute(per_obj_q_std)
        self.belief_tracker.update(mahal_surprise)
        weighted_lambda = self._compute_weighted_lambda()
        self._current_weighted_lambda = weighted_lambda

        # --- Target Cone sampling (Part XI/XII) ---
        if self.config.enable_target_cone:
            w_restricted, w_anchor = self.cone_sampler.sample_and_get_anchor(
                self.belief_tracker.belief)
        else:
            w_restricted = np.array(self.config.preference_weights, dtype=np.float32)
            w_anchor = w_restricted.copy()
        self._current_w = w_restricted
        self._current_w_anchor = w_anchor

        # --- Run scan ---
        t_p = nnx.state(self.target_critic, nnx.Param)
        p_p = nnx.state(self.policy, nnx.Param)

        obs = jnp.array(stacked_batch["obs"])
        act = jnp.array(stacked_batch["act"])
        rew = jnp.array(stacked_batch["rew"])
        nobs = jnp.array(stacked_batch["next_obs"])
        done = jnp.array(stacked_batch["done"])
        tids = jnp.array(stacked_batch["task_id"])

        w_jax = jnp.array(w_restricted)
        w_anchor_jax = jnp.array(w_anchor)

        final, metrics = self._scan_update(
            c_p, t_p, p_p, x_p, self.log_alpha,
            self.critic_opt_state, self.policy_opt_state,
            self.context_opt_state, self.alpha_opt_state,
            obs, act, rew, nobs, done, tids, rng_key, warmup,
            jnp.array(weighted_lambda), w_jax, w_anchor_jax)

        (new_c, new_t, new_p, new_x, new_la,
         self.critic_opt_state, self.policy_opt_state,
         self.context_opt_state, self.alpha_opt_state, _) = final
        nnx.update(self.critic, new_c)
        nnx.update(self.target_critic, new_t)
        nnx.update(self.policy, new_p)
        nnx.update(self.context_net, new_x)
        self.log_alpha = new_la

        n = obs.shape[0]
        self.update_count += n

        c_loss, p_loss, x_loss, alpha, qm, qs, lp, pq_obj_std = metrics
        return {
            "critic_loss": float(c_loss.mean()),
            "policy_loss": float(p_loss.mean()),
            "rmdm_loss": float(x_loss.mean()),
            "alpha": float(alpha[-1]),
            "q_mean": float(qm.mean()),
            "q_std_mean": float(qs.mean()),
            "log_prob": float(lp.mean()),
            "weighted_lambda": weighted_lambda,
            "belief_entropy": float(self.belief_tracker.entropy),
            "effective_window": float(self.belief_tracker.effective_window),
            "mahalanobis_surprise": float(mahal_surprise),
            "target_cone_w0": float(w_restricted[0]),
            "w_anchor_0": float(w_anchor[0]),
            "warmup": bool(warmup),
        }
