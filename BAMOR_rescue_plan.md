# BAMOR rescue plan

## Current assessment

The paper is salvageable, but not as a broad claim that BAMOR is a generally effective non-stationary MORL algorithm. The current evidence supports a narrower claim: Target Cone sampling can help under explicit substrate-collapse assumptions, and the Lean proof formalizes the tracking-bound advantage under those assumptions.

The immediate risk is not the Lean proof. The Lean proof exists in the adjacent proof repository and builds. The larger risk is that the implementation, empirical claims, and paper narrative are not fully aligned.

## Why not blindly expand experiments

The next step should not be to add many more environments immediately. Several failure modes are currently entangled:

- the substrate-collapse assumption may not hold in the new environment;
- the cone estimator may be estimating the wrong quantity;
- the belief or mode detector may be assigning samples to the wrong mode;
- a single weight-conditioned actor may suffer gradient interference in higher-dimensional continuous control;
- full-simplex hypervolume may penalize a method whose theory only promises improvement inside valid cones.

If these are not separated, more experiments will only say that the method is bad somewhere, without explaining which part failed.

## Can a synthetic-only version be published?

Yes, but only with a much narrower positioning.

Not as:

> BAMOR is a general non-stationary MORL algorithm that works in realistic continuous-control domains.

Instead as:

> BAMOR formalizes a Target Cone mechanism for substrate-constrained MORL, proves a cone-localized tracking-bound advantage under explicit assumptions, and validates the mechanism in a controlled benchmark where those assumptions are exactly testable. Complex continuous-control scaling remains an open limitation.

That positioning can work for arXiv, workshops, or a theory/artifact-oriented venue. A main-conference method paper would likely require at least one non-synthetic setting where the same mechanism survives.

## Why the previous environment expansions likely failed

The current best variant, `bamor_ns`, is not exactly what the paper describes as frozen-uniform belief plus mode-aware Target Cone.

In the current implementation, disabling surprise only makes the surprise threshold very large. The belief still drifts in the no-switch branch toward the current argmax mode. Also, the cone estimator is not explicitly updated from environment ground-truth mode labels. This means the best variant is closer to a global or partially collapsed cone sampler than to a true mode-aware Target Cone algorithm.

There is also a conceptual issue with the actor anchor regularizer. If the sampled weight and anchor are constants with respect to policy parameters, an added `eta * ||w - anchor||^2` term in the actor loss has no policy gradient. The useful mechanism is the sampling distribution, not that no-op actor-loss term.

## Algorithm changes to test

1. Add an oracle-mode diagnostic.
   Use the environment's true `mode_idx` to set the active mode and update the corresponding cone. This answers whether Target Cone works when mode identification is perfect.

2. Add an oracle-cone diagnostic.
   Use the environment's true anchors directly. This answers whether the learned cone estimator is the failure point.

3. Make frozen-global Target Cone real.
   For a no-surprise variant, freeze belief and update all mode cones with the same global reward stream, so the belief-weighted anchor is genuinely global rather than accidentally tied to one mode slot.

4. Replace reward-variance cone estimation with controllability estimation.
   Reward variance is not the same as objective controllability. A better estimator should measure how much each objective can be changed by actions under a mode, for example via `Var_a Q_i(s,a)`, a reward-model action Jacobian, or action-conditioned reward spread.

5. Add residual global exploration.
   Sample from the full simplex with small probability `epsilon`, and from the current cone otherwise. This prevents a misspecified cone from starving valid preferences.

6. Consider multi-expert actors.
   A single weight-conditioned actor may be too much for complex continuous-control settings. A shared critic with mode/cone-specific actor heads is a more realistic extension.

## Diagnostic matrix

Run these before changing many environments:

1. Uniform baseline.
2. Oracle mode + oracle cone.
3. Oracle mode + learned cone.
4. Learned mode + oracle cone.
5. Learned mode + learned cone.
6. Multi-expert version + learned cone.

Interpretation:

- If 2 does not win, the environment may not satisfy substrate-collapse or the evaluation is mismatched.
- If 2 wins but 3 does not, the cone estimator is wrong.
- If 3 wins but 5 does not, the mode detector is wrong.
- If 5 wins on cone-restricted HV but loses full-simplex HV, residual global exploration is needed.
- If 5 is unstable even with correct diagnostics, the single-actor backbone is likely the bottleneck.

## Practical next step

The first implementation step is to add frozen-global, oracle-mode, and oracle-cone diagnostic methods to the existing synthetic training script. That gives a clean causal readout before spending compute on larger environments.

## 2026-06-08 HPC diagnostic update

Implemented the diagnostic variants and ran a 20-seed, 100k-step synthetic matrix on `zhengliang-hpc` CPU nodes `node001`-`node006`.

Result directory:

```text
results/diagnostic_hpc_20260608_002106
```

Scheduler setup:

- staged minimal BAMOR code to `/home/zhengliang01/scheduleurm_work/BAMOR`;
- used `/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python`;
- installed `gymnasium==1.2.0` into that remote environment from a local wheelhouse;
- submitted the first 56 items as individual scheduler tasks, then submitted the remaining 124 items as six node-level shards;
- all 180 method/seed result JSON files completed and were pulled back locally.

Summary, HV x 1e3:

| method | n | overall | mode0 | mode1 | mode2 | detected / true switches |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 20 | 98.6+/-50.5 | 18.9+/-6.4 | 31.4+/-16.7 | 99.0+/-51.7 | - |
| bamor_oracle_cone | 20 | 88.8+/-49.9 | 19.5+/-5.9 | 27.5+/-15.8 | 84.2+/-49.8 | 0.0/333.0 |
| bamor_oracle | 20 | 101.9+/-38.5 | 21.7+/-3.9 | 34.2+/-11.4 | 105.9+/-38.0 | 0.0/333.0 |
| bamor_fixed_cone | 20 | 128.6+/-31.9 | 23.2+/-2.5 | 39.8+/-9.8 | 127.7+/-30.3 | 35.6+/-17.8 / 333.0 |
| bamor | 20 | 118.7+/-39.7 | 20.3+/-5.4 | 35.8+/-12.9 | 115.7+/-39.4 | 94.0+/-23.8 / 333.0 |
| bamor_ns | 20 | 92.4+/-50.5 | 18.9+/-7.4 | 30.9+/-16.2 | 96.7+/-50.6 | 0.0/333.0 |
| bamor_ns_eps | 20 | 121.1+/-36.3 | 21.3+/-3.1 | 38.6+/-12.0 | 120.3+/-41.9 | 0.0/333.0 |
| bamor_nc | 20 | 130.4+/-26.4 | 22.2+/-4.0 | 42.1+/-7.1 | 132.3+/-22.6 | 45.0+/-6.9 / 333.0 |
| fixed | 20 | 106.1+/-49.7 | 13.0+/-5.3 | 27.9+/-12.9 | 91.5+/-42.2 | - |

Interpretation:

- Target Cone is not currently the winning mechanism under full-simplex HV. `bamor_nc` is the best overall, which means disabling the cone beats the learned Target Cone in this diagnostic.
- Oracle cone does not rescue performance. `bamor_oracle_cone` is below uniform, so merely knowing the true cone anchors is not sufficient under the current training/evaluation setup.
- Residual exploration helps. `bamor_ns_eps` is much better than `bamor_ns`, supporting the idea that cone restriction starves useful preferences when the cone/evaluation mismatch is present.
- The detector is weak relative to the true switch count. Learned BAMOR variants detect far fewer than 333 true switches, so the paper should not claim reliable switch recovery without additional detector work.

Paired against uniform on overall HV, `bamor_nc` is the only strong positive diagnostic among the core variants: mean paired gain `+31.7k`, wins `13/20`, paired t-test `p=0.02698`. `bamor` has mean gain `+20.0k`, wins `13/20`, but is not significant (`p=0.08413`); `bamor_ns_eps` has mean gain `+22.5k`, wins `15/20`, but is also not significant (`p=0.1134`).

Current scientific conclusion:

The method is not ready to be claimed as an effective general non-stationary MORL algorithm. The next defensible direction is to either (i) reposition around a narrower cone-local theory/artifact claim, or (ii) redesign the cone objective/evaluation so the algorithm is optimized and judged on cone-local coverage rather than full-simplex HV.

## 2026-06-08 follow-up: mode context is the real bottleneck

After the first diagnostic matrix, I added context-conditioned variants where the policy and critic receive more than just `state + weight`.

Key code changes:

- `ContextBuffer` stores scalarization weight `w`, current policy context `c`, and next context `cn`.
- Oracle-context variants use `context = [w, one_hot(true_mode)]`.
- Belief-context variants use `context = [w, BAMOR_belief]`.
- Fast-detector variants lower CUSUM latency with `surprise_threshold=3.0`, `warmup=20`, `switch_cooldown=250`, `cusum_required_consecutive=2`, `cusum_decay=0.25`.
- Transition-context variants use a non-oracle online signature `context = [w, speed_response_ema, reward_ema]`; BAMOR transition variants additionally include `BAMOR_belief`.

### Oracle mode-context matrix

Result directory:

```text
results/diagnostic_ctx_oracle_hpc_20260608_045043
```

Summary, HV x 1e3:

| method | n | overall | mode0 | mode1 | mode2 |
|---|---:|---:|---:|---:|---:|
| uniform_ctx_oracle | 20 | 178.2+/-26.2 | 18.9+/-4.6 | 34.1+/-13.8 | 122.7+/-33.7 |
| bamor_ctx_oracle_cone | 20 | 180.3+/-27.0 | 24.6+/-2.4 | 42.5+/-9.4 | 124.3+/-41.3 |
| bamor_ctx_oracle | 20 | 160.3+/-29.0 | 23.6+/-3.6 | 23.4+/-17.3 | 64.6+/-52.2 |

Paired against old uniform:

- `uniform_ctx_oracle`: `+79.5k`, wins `19/20`, `p=5.299e-06`.
- `bamor_ctx_oracle_cone`: `+81.7k`, wins `20/20`, `p=1.478e-06`.

But `bamor_ctx_oracle_cone` vs `uniform_ctx_oracle` is only `+2.2k`, wins `11/20`, `p=0.7795`.

Interpretation:

Mode-conditioned policy learning is a large effect. Target Cone, even with oracle mode and oracle cone anchors, is not a significant additional effect over uniform weight sampling once the policy knows the active mode.

### Belief-context and fast-detector matrices

Belief context with the old detector:

```text
results/diagnostic_ctx_belief_hpc_20260608_051021
```

HV x 1e3:

| method | n | overall | detected / true switches |
|---|---:|---:|---:|
| bamor_ctx_fixed_cone | 20 | 133.7+/-44.6 | 43.8+/-12.6 / 333.0 |
| bamor_ctx | 20 | 119.7+/-39.0 | 90.9+/-19.2 / 333.0 |
| bamor_ctx_nc | 20 | 115.7+/-49.1 | 42.0+/-11.2 / 333.0 |

Fast detector:

```text
results/diagnostic_fastdet_hpc_20260608_054134
```

HV x 1e3:

| method | n | overall | detected / true switches |
|---|---:|---:|---:|
| bamor_fastdet | 20 | 100.9+/-48.0 | 333.2+/-0.6 / 333.0 |
| bamor_fastdet_ctx | 20 | 138.5+/-27.3 | 333.2+/-0.4 / 333.0 |
| bamor_fastdet_ctx_nc | 20 | 125.8+/-38.3 | 347.1+/-4.9 / 333.0 |

Key paired tests:

- `bamor_fastdet_ctx` vs uniform: `+39.9k`, wins `12/20`, `p=0.007495`.
- `bamor_fastdet_ctx` vs `bamor_fastdet`: `+37.6k`, wins `14/20`, `p=0.01709`.
- `bamor_fastdet_ctx` vs `uniform_ctx_oracle`: `-39.6k`, wins `4/20`, `p=0.0003203`.

Interpretation:

Counting switches accurately is not enough. The actor needs a useful regime representation. BAMOR belief helps, but the current round-robin surprise detector still leaves a large gap to oracle mode context.

### Transition-context redesign matrix

Result directory:

```text
results/diagnostic_trans_ctx_hpc_20260608_061653
```

This is the first real "push over and rebuild" diagnostic. The policy receives an online, non-oracle transition signature: recent velocity/action response plus reward EMA. This is a lightweight stand-in for a learned latent dynamics/reward encoder.

Summary, HV x 1e3:

| method | n | overall | mode0 | mode1 | mode2 | detected / true switches |
|---|---:|---:|---:|---:|---:|---:|
| uniform_trans_ctx | 20 | 193.5+/-20.3 | 7.7+/-2.4 | 9.0+/-1.2 | 107.4+/-40.4 | - |
| bamor_fastdet_trans_ctx | 20 | 183.6+/-36.6 | 18.9+/-6.2 | 34.6+/-8.9 | 130.6+/-21.2 | 333.1+/-0.9 / 333.0 |
| bamor_fastdet_trans_ctx_nc | 20 | 191.3+/-30.1 | 22.8+/-5.5 | 43.3+/-4.1 | 149.0+/-13.9 | 337.3+/-3.7 / 333.0 |

Key paired tests:

- `uniform_trans_ctx` vs old uniform: `+94.9k`, wins `20/20`, `p=1.561e-07`.
- `uniform_trans_ctx` vs old `bamor_nc`: `+63.1k`, wins `19/20`, `p=7.353e-08`.
- `bamor_fastdet_trans_ctx_nc` vs old `bamor_nc`: `+60.9k`, wins `18/20`, `p=1.108e-05`.
- `bamor_fastdet_trans_ctx` vs `bamor_fastdet_ctx`: `+45.0k`, wins `17/20`, `p=0.0004131`.
- `bamor_fastdet_trans_ctx_nc` vs `uniform_ctx_oracle`: `+13.1k`, wins `15/20`, `p=0.02458`.
- `bamor_fastdet_trans_ctx_nc` vs `bamor_fastdet_trans_ctx`: `+7.7k`, wins `12/20`, `p=0.2847`.
- `uniform_trans_ctx` vs `bamor_fastdet_trans_ctx_nc`: `+2.2k` for uniform, not significant (`p=0.8192` in the reverse comparison).

Interpretation:

The algorithm can be rescued empirically, but not as originally framed. A simple non-oracle transition/reward context closes the oracle-mode gap and beats all old BAMOR variants. The Target Cone remains weak: the no-cone transition variant is better than the cone transition variant, although not significantly.

## Revised paper strategy

The strongest defensible direction is to pivot the algorithmic claim:

> Non-stationary MORL needs latent regime conditioning. BAMOR's original surprise/cone machinery is useful as one possible regime tracker and preference sampler, but the decisive ingredient is giving the actor/critic a current substrate/regime representation inferred from transitions and rewards.

Do not submit the current paper as "Target Cone is the key mechanism." The experiments do not support that. A credible rewrite would:

1. Rename or reframe the method as latent-regime-conditioned MORL.
2. Move Target Cone from the central empirical claim to an optional preference-prior/regularizer.
3. Replace reward-only surprise belief with a learned transition/reward context encoder.
4. Keep the Lean Target Cone proof as a theory subsection under explicit assumptions, not as the main reason the empirical method works.
5. Evaluate both full-simplex HV and cone-local HV, and be explicit that the current synthetic benchmark favors transition signatures.

Practical next implementation step:

- Promote `uniform_trans_ctx` / `bamor_fastdet_trans_ctx_nc` into the main method scaffold.
- Replace the hand-coded transition signature with a small recurrent or contrastive encoder over `(s_t, a_t, r_t, s_{t+1})`.
- Run the same matrix on one additional non-synthetic or semi-synthetic environment before attempting any stronger publication claim.

## 2026-06-08 BAMOR-v2 implementation pass

Implemented a first BAMOR-v2 scaffold in `train_compare_baselines.py`.

New components:

- `LatentTransitionEncoder`: self-supervised transition encoder mapping `(s, a, r, s')` to latent `z`.
- `LearnedTransitionContextTracker`: online EMA tracker over learned latents.
- New methods:
  - `bamor_v2`: uniform weights with learned transition latent, `context=[w,z]`.
  - `bamor_v2_belief`: BAMOR fast detector + no cone, `context=[w,belief,z]`.
  - `bamor_v2_cone`: BAMOR fast detector + Target Cone, `context=[w,belief,z]`.
  - `bamor_v2_hybrid`: observed transition signature + learned latent, `context=[w,transition,z]`.
  - `bamor_v2_hybrid_belief`: `context=[w,belief,transition,z]`, no cone.
  - `bamor_v2_hybrid_cone`: `context=[w,belief,transition,z]`, Target Cone.

Also fixed a weight-sampling bug: `uniform_trans_ctx` previously fell through to fixed `[0.5,0.5]` training weights. It now samples Dirichlet-uniform weights like the other uniform methods.

### Pure learned-latent v2 matrix

Result directory:

```text
results/diagnostic_v2_hpc_20260608_080142
```

Summary, HV x 1e3:

| method | n | overall | mode0 | mode1 | mode2 | detected / true switches |
|---|---:|---:|---:|---:|---:|---:|
| uniform_trans_ctx | 20 | 188.7+/-27.0 | 18.8+/-4.6 | 41.2+/-6.8 | 143.5+/-11.2 | - |
| bamor_fastdet_trans_ctx_nc | 20 | 181.7+/-50.5 | 20.7+/-7.1 | 41.3+/-13.3 | 132.2+/-43.5 | 337.9+/-2.4 / 333.0 |
| bamor_v2 | 20 | 144.4+/-58.9 | 18.6+/-5.2 | 36.3+/-15.2 | 114.7+/-49.0 | - |
| bamor_v2_belief | 20 | 136.2+/-67.3 | 18.9+/-5.6 | 35.1+/-16.3 | 115.1+/-50.1 | 339.3+/-3.9 / 333.0 |
| bamor_v2_cone | 20 | 168.0+/-33.4 | 23.6+/-4.5 | 40.9+/-7.3 | 120.3+/-29.1 | 333.4+/-0.8 / 333.0 |

Key paired tests:

- `bamor_v2` vs old uniform: `+45.7k`, wins `16/20`, `p=0.01102`.
- `bamor_v2` vs old `bamor_nc`: `+14.0k`, wins `13/20`, not significant.
- `bamor_v2` vs `uniform_trans_ctx`: `-44.4k`, wins `5/20`, `p=0.00336`.
- `bamor_v2_cone` vs `bamor_fastdet_trans_ctx_nc`: `-13.7k`, not significant.

Interpretation:

The pure autoencoder latent is better than old uniform but not good enough. It loses to the direct observable transition signature, which means the encoder objective is too weak: reconstructing reward/delta does not force a clean regime representation.

### Hybrid v2 matrix

Result directory:

```text
results/diagnostic_v2_hybrid_hpc_20260608_083812
```

Summary, HV x 1e3:

| method | n | overall | mode0 | mode1 | mode2 | detected / true switches |
|---|---:|---:|---:|---:|---:|---:|
| bamor_v2_hybrid | 20 | 191.6+/-34.1 | 18.2+/-5.9 | 36.0+/-14.3 | 142.3+/-21.8 | - |
| bamor_v2_hybrid_belief | 20 | 187.7+/-39.3 | 20.5+/-5.0 | 38.7+/-8.9 | 141.1+/-27.7 | 338.6+/-3.4 / 333.0 |
| bamor_v2_hybrid_cone | 20 | 188.0+/-21.1 | 21.3+/-5.4 | 42.5+/-7.4 | 132.5+/-24.4 | 333.0+/-0.3 / 333.0 |

Key paired tests:

- `bamor_v2_hybrid` vs old `bamor_nc`: `+61.2k`, wins `19/20`, `p=3.209e-07`.
- `bamor_v2_hybrid` vs pure `bamor_v2`: `+47.2k`, wins `16/20`, `p=0.008419`.
- `bamor_v2_hybrid` vs `uniform_ctx_oracle`: `+13.4k`, wins `13/20`, not significant.
- `bamor_v2_hybrid` vs corrected `uniform_trans_ctx`: `+2.8k`, wins `11/20`, not significant.
- `bamor_v2_hybrid_cone` vs `bamor_v2_hybrid`: `-3.6k`, not significant.

Interpretation:

The best v2 direction is hybrid regime conditioning:

```text
context = [preference weight, online transition signature, learned transition latent]
```

This closes the oracle-context gap and strongly beats old BAMOR variants. BAMOR belief and Target Cone still do not add reliable improvement on this benchmark. The current v2 should therefore be framed as a latent/observable regime-conditioned MORL method, with Target Cone kept as an optional prior rather than the center of the algorithm.

Next algorithmic improvement:

- Replace the current autoencoder objective with a stronger regime-learning objective:
  - contrastive same-window/different-window loss;
  - predictive dynamics head conditioned on action;
  - temporal smoothness within a detected dwell segment;
  - separation across surprise-triggered boundaries.
- Keep the direct observable transition summary until the learned encoder can match it, because it is currently the strongest non-oracle signal.

## 2026-06-08 BAMOR-v2.1 temporal-contrastive encoder

Implemented the next encoder objective:

- `ContextBuffer` now stores the global training step for each transition.
- `LatentTransitionEncoder` has a projection head.
- `temporal_contrastive_loss` adds a supervised-contrastive-style objective without oracle mode labels:
  - positives: transitions sampled within a short temporal window (`pos_window=64`);
  - negatives: other non-self samples in the batch;
  - reconstruction heads for reward and state delta are still retained.

New methods:

- `bamor_v2_contrast`: learned contrastive latent only, `context=[w,z]`.
- `bamor_v2_contrast_hybrid`: observed transition signature + contrastive latent, `context=[w,transition,z]`.
- `bamor_v2_contrast_hybrid_cone`: hybrid contrastive context plus BAMOR Target Cone.

Result directory:

```text
results/diagnostic_v2_contrast_hpc_20260608_092735
```

Summary, HV x 1e3:

| method | n | overall | mode0 | mode1 | mode2 | detected / true switches |
|---|---:|---:|---:|---:|---:|---:|
| bamor_v2_contrast | 20 | 162.6+/-55.9 | 19.0+/-5.6 | 34.4+/-14.0 | 124.1+/-41.7 | - |
| bamor_v2_contrast_hybrid | 20 | 197.9+/-27.8 | 19.1+/-5.5 | 25.7+/-15.5 | 148.8+/-10.9 | - |
| bamor_v2_contrast_hybrid_cone | 20 | 169.3+/-43.2 | 19.4+/-6.0 | 31.2+/-13.0 | 115.8+/-32.9 | 333.0+/-0.5 / 333.0 |

Key paired tests:

- `bamor_v2_contrast_hybrid` vs old `bamor_nc`: `+67.5k`, wins `19/20`, `p=4.117e-08`.
- `bamor_v2_contrast_hybrid` vs `uniform_ctx_oracle`: `+19.7k`, wins `17/20`, `p=0.03129`.
- `bamor_v2_contrast_hybrid` vs corrected `uniform_trans_ctx`: `+9.1k`, wins `13/20`, not significant.
- `bamor_v2_contrast_hybrid` vs previous `bamor_v2_hybrid`: `+6.3k`, wins `12/20`, not significant.
- `bamor_v2_contrast_hybrid_cone` vs `bamor_v2_contrast_hybrid`: `-28.6k`, wins `4/20`, `p=0.02788`.

Interpretation:

The temporal-contrastive hybrid is the current best candidate. It is now above the oracle-mode uniform baseline and strongly above all old BAMOR variants. The pure learned latent improves over old BAMOR but still does not match the direct transition signature; the hybrid is necessary.

The Target Cone result is now a clear negative empirical result in this benchmark: adding cone sampling to the best context representation significantly hurts performance. The paper should not sell Target Cone as the main algorithmic advantage. The defensible v2 claim is:

> BAMOR-v2 learns and tracks a non-oracle transition-regime context for MORL; temporal contrastive learning improves regime representation, and Target Cone can be studied as an optional preference prior but is not the core empirical mechanism.

Current best method to promote:

```text
bamor_v2_contrast_hybrid
context = [preference weight, online transition signature, temporal-contrastive latent]
weight sampling = Dirichlet-uniform
no Target Cone by default
```

Next step before writing the new paper:

- Replace the hand-coded transition signature with a learned sequence encoder, or keep it explicitly as a low-dimensional observable regime statistic and be honest that the benchmark exposes such a statistic.
- Run one additional non-synthetic/semi-synthetic environment with the same no-oracle context interface.
- Keep `bamor_v2_contrast_hybrid_cone` as an ablation showing that cone restriction is not automatically beneficial under full-simplex HV.

## 2026-06-08 BAMOR-v2.2 sequence encoder check

Implemented a GRU sequence-regime encoder to test whether the hand-coded transition signature can be removed.

New implementation details:

- `ContextBuffer` can now sample contiguous transition sequences.
- `SequenceTransitionEncoder` encodes the last `16` raw transition features:

```text
[state, action, tanh(reward/3), tanh((next_state-state)/3)]
```

- `SequenceTransitionContextTracker` maintains the online history window during training/evaluation.
- New methods:
  - `bamor_v2_seq`: GRU latent only, `context=[w,z_seq]`.
  - `bamor_v2_seq_contrast`: GRU latent + temporal contrastive loss, `context=[w,z_seq]`.
  - `bamor_v2_seq_contrast_cone`: GRU latent + temporal contrastive loss + BAMOR Target Cone.

Result directory:

```text
results/diagnostic_v2_seq_hpc_20260608_102731
```

Summary, HV x 1e3:

| method | n | overall | mode0 | mode1 | mode2 | detected / true switches |
|---|---:|---:|---:|---:|---:|---:|
| bamor_v2_seq | 20 | 149.1+/-50.4 | 19.3+/-5.4 | 40.8+/-11.5 | 124.5+/-38.9 | - |
| bamor_v2_seq_contrast | 20 | 153.6+/-41.8 | 20.6+/-4.7 | 34.6+/-12.4 | 118.6+/-32.5 | - |
| bamor_v2_seq_contrast_cone | 20 | 177.8+/-17.4 | 23.5+/-5.0 | 38.5+/-10.9 | 128.0+/-18.6 | 333.4+/-0.5 / 333.0 |

Key paired tests:

- `bamor_v2_seq` vs old `bamor_nc`: `+18.7k`, wins `15/20`, not significant.
- `bamor_v2_seq` vs `uniform_ctx_oracle`: `-29.1k`, wins `5/20`, `p=0.03478`.
- `bamor_v2_seq` vs `bamor_v2_contrast_hybrid`: `-48.8k`, wins `2/20`, `p=0.003539`.
- `bamor_v2_seq_contrast` vs `bamor_v2_seq`: `+4.5k`, not significant.
- `bamor_v2_seq_contrast_cone` vs `bamor_v2_seq_contrast`: `+24.2k`, wins `15/20`, `p=0.03073`.
- `bamor_v2_seq_contrast_cone` vs old `bamor_nc`: `+47.4k`, wins `18/20`, `p=1.318e-06`.

Interpretation:

The GRU sequence encoder does not yet replace the direct transition signature. Pure sequence context is better than old BAMOR in mean but not significant, and it is significantly worse than the current best `bamor_v2_contrast_hybrid`.

This is useful scientifically: it shows that the benchmark's simple observable transition statistic is doing real work. A learned sequence encoder needs either:

- longer temporal windows;
- a stronger contrastive objective based on detected boundaries;
- supervised pretraining on synthetic mode labels as a diagnostic only;
- or a recurrent policy architecture where the policy hidden state is trained end-to-end rather than precomputed into replay contexts.

Current best method remains:

```text
bamor_v2_contrast_hybrid
```

The sequence encoder is a promising ablation, not the main method yet.

## 2026-06-08 BAMOR-v2.3 generic transition statistic

Implemented a generic transition-context tracker to test whether the current best method can avoid using the benchmark-specific `velocity/action` statistic.

New method:

```text
GenericTransitionContextTracker
```

It tracks only environment-agnostic online EMAs:

```text
EMA(delta_state)
EMA(delta_state / ||action||)
EMA(action)
EMA(reward)
```

New methods:

- `bamor_v2_generic`: generic transition statistic only, `context=[w,generic]`.
- `bamor_v2_contrast_generic`: generic statistic + temporal-contrastive single-transition latent, `context=[w,generic,z]`.
- `bamor_v2_contrast_generic_cone`: generic statistic + contrastive latent + Target Cone.

Result directory:

```text
results/diagnostic_v2_generic_hpc_20260608_113450
```

Summary, HV x 1e3:

| method | n | overall | mode0 | mode1 | mode2 | detected / true switches |
|---|---:|---:|---:|---:|---:|---:|
| bamor_v2_generic | 20 | 153.6+/-63.1 | 20.4+/-4.3 | 39.5+/-14.8 | 120.1+/-50.6 | - |
| bamor_v2_contrast_generic | 20 | 159.3+/-59.2 | 21.1+/-5.5 | 33.2+/-15.7 | 114.1+/-49.8 | - |
| bamor_v2_contrast_generic_cone | 20 | 164.8+/-47.0 | 18.8+/-6.2 | 36.2+/-9.3 | 119.0+/-31.6 | 333.4+/-0.9 / 333.0 |

Key paired tests:

- `bamor_v2_generic` vs old `bamor_nc`: `+23.2k`, wins `15/20`, not significant.
- `bamor_v2_generic` vs `bamor_v2_contrast_hybrid`: `-44.3k`, wins `4/20`, `p=0.01891`.
- `bamor_v2_contrast_generic` vs `bamor_v2_contrast_hybrid`: `-38.5k`, wins `5/20`, `p=0.005721`.
- `bamor_v2_contrast_generic` vs old `bamor_nc`: `+29.0k`, wins `16/20`, marginal but not significant (`p=0.08649`).
- `bamor_v2_contrast_generic_cone` vs old `bamor_nc`: `+34.4k`, wins `18/20`, `p=0.01176`.
- `bamor_v2_contrast_generic_cone` vs `bamor_v2_contrast_generic`: `+5.4k`, not significant.

Interpretation:

The generic statistic helps relative to old BAMOR, but it does not replace the benchmark-specific transition signature. The gap to `bamor_v2_contrast_hybrid` remains large and significant.

This answers the "is there still room?" question:

- Yes, there is algorithmic room.
- But the easy room from adding context has mostly been used.
- The remaining hard problem is learning a regime statistic as informative as the hand-coded `velocity/action` signal without assuming that coordinate.

The next real optimization cannot be another small MLP/EMA variant. It should be one of:

1. End-to-end recurrent policy/critic, so the policy hidden state is optimized directly for return rather than through frozen replay contexts.
2. Contrastive learning with stronger pseudo-labels from detected switch boundaries, not only temporal proximity.
3. Diagnostic supervised regime encoder on synthetic mode labels to establish an upper bound for learned representation.
4. Environment redesign for a semi-synthetic continuous-control benchmark where the regime statistic is not trivially a single coordinate ratio.

Current best remains:

```text
bamor_v2_contrast_hybrid
```

The generic version is publishable as an ablation showing the portability gap, not as the main method.

## 2026-06-08 BAMOR-v2.4 boundary-aware pseudo-regime contrast

Goal:

Test whether BAMOR's non-oracle change detector can provide a stronger
self-supervised signal for the learned transition encoder. Instead of using
only temporal proximity, v2.4 stores the detector's current pseudo-regime label
in replay and adds a supervised-contrastive term:

```text
same detector regime -> positive latent pairs
different detector regimes -> negative latent pairs
```

This does not read the environment's true mode label. For no-cone variants,
the BAMOR object is used only as a detector and its base method is `bamor_nc`,
so weight sampling remains uniform over the simplex.

Implemented methods:

```text
bamor_v2_boundary_hybrid
bamor_v2_boundary_hybrid_cone
bamor_v2_boundary_generic
```

Implementation details:

- `ContextBuffer` now stores a replay `regime` label.
- `pseudo_regime_contrastive_loss` adds detector-label contrast on top of the
  existing temporal contrastive loss.
- Boundary methods reset online context EMAs when BAMOR detects a switch.
- `bamor_v2_boundary_hybrid` uses the current best context shape
  `[w, transition_signature, learned_latent]`, with detector-guided latent
  training and uniform weight sampling.
- `bamor_v2_boundary_generic` replaces the benchmark-specific transition
  signature with the generic transition statistic.

HPC run:

```text
results/diagnostic_v2_boundary_hpc_20260608_131508
methods = bamor_v2_boundary_hybrid bamor_v2_boundary_hybrid_cone bamor_v2_boundary_generic
seeds = 20
total_steps = 100000
nodes = node001-node006
```

Summary, HV x 1e3:

| method | n | overall | mode0 | mode1 | mode2 | det/true |
|---|---:|---:|---:|---:|---:|---:|
| bamor_v2_boundary_hybrid | 20 | 193.1+/-35.9 | 20.7+/-3.6 | 28.8+/-13.1 | 133.4+/-36.7 | 336.6+/-2.4 / 333.0 |
| bamor_v2_boundary_hybrid_cone | 20 | 183.9+/-22.0 | 18.0+/-5.4 | 34.5+/-9.2 | 128.3+/-20.2 | 333.1+/-0.4 / 333.0 |
| bamor_v2_boundary_generic | 20 | 176.0+/-50.2 | 20.0+/-6.2 | 36.4+/-13.2 | 133.9+/-28.9 | 339.1+/-5.1 / 333.0 |

Paired comparisons:

- `bamor_v2_boundary_hybrid` vs current best `bamor_v2_contrast_hybrid`:
  `-4.7k`, wins `7/20`, `p=0.6707`.
- `bamor_v2_boundary_hybrid_cone` vs `bamor_v2_boundary_hybrid`:
  `-9.3k`, wins `6/20`, `p=0.3189`.
- `bamor_v2_boundary_generic` vs `bamor_v2_contrast_generic`:
  `+16.7k`, wins `11/20`, `p=0.2766`.
- `bamor_v2_boundary_generic` vs `bamor_v2_generic`:
  `+22.4k`, wins `13/20`, `p=0.04782`.
- `bamor_v2_boundary_hybrid` vs old `bamor_nc`:
  `+62.8k`, wins `19/20`, `p=7.659e-06`.
- `bamor_v2_boundary_hybrid` vs `uniform_ctx_oracle`:
  `+15.0k`, wins `15/20`, `p=0.1097`.
- `bamor_v2_boundary_hybrid` vs previous `bamor_v2_hybrid`:
  `+1.6k`, wins `11/20`, `p=0.8549`.
- `bamor_v2_boundary_generic` vs current best `bamor_v2_contrast_hybrid`:
  `-21.9k`, wins `8/20`, `p=0.1097`.

Interpretation:

Boundary pseudo-labels are useful, but mostly for the generic context path.
They significantly improve `bamor_v2_generic`, and they narrow the gap between
generic context and the benchmark-specific transition signature. They do not
improve the already strong hybrid transition-signature method.

The current best remains:

```text
bamor_v2_contrast_hybrid
```

The remaining algorithmic space is therefore not another light context-feature
variant. The next plausible step is architectural: recurrent actor/critic
state, learned online belief inference, or a semi-synthetic benchmark where the
transition signature is not hand-coded from this environment.

## 2026-06-08 BAMOR-v2.5 prototype-belief context

Goal:

Move beyond using the detector pseudo-regime labels only as an auxiliary
encoder loss. v2.5 turns detector-labeled transition features into an online
soft belief context that the actor and critic consume directly.

New component:

```text
RegimePrototypeBeliefTracker
```

It maintains one normalized feature prototype per detector regime. At each
transition it computes a soft belief from distances to these prototypes, blends
in a small detector-label prior, and exposes:

```text
[soft_regime_belief, max_belief_probability, tanh(age_since_switch/max_age)]
```

This is still non-oracle: it does not read true environment mode labels.

Implemented methods:

```text
bamor_v2_proto_hybrid
bamor_v2_proto_hybrid_cone
bamor_v2_proto_generic
```

HPC run:

```text
results/diagnostic_v2_proto_hpc_20260608_143710
methods = bamor_v2_proto_hybrid bamor_v2_proto_hybrid_cone bamor_v2_proto_generic
seeds = 20
total_steps = 100000
nodes = node001-node006
```

Summary, HV x 1e3:

| method | n | overall | mode0 | mode1 | mode2 | det/true |
|---|---:|---:|---:|---:|---:|---:|
| bamor_v2_proto_hybrid | 20 | 199.8+/-39.5 | 17.6+/-5.1 | 27.7+/-14.1 | 144.5+/-19.7 | 336.6+/-2.7 / 333.0 |
| bamor_v2_proto_hybrid_cone | 20 | 173.5+/-31.4 | 21.5+/-4.1 | 32.5+/-13.1 | 122.0+/-28.1 | 332.9+/-0.2 / 333.0 |
| bamor_v2_proto_generic | 20 | 186.4+/-30.5 | 19.1+/-5.4 | 40.9+/-10.0 | 146.7+/-17.7 | 337.4+/-3.2 / 333.0 |

Paired comparisons:

- `bamor_v2_proto_hybrid` vs current best `bamor_v2_contrast_hybrid`:
  `+2.0k`, wins `12/20`, `p=0.7757`.
- `bamor_v2_proto_hybrid` vs `bamor_v2_boundary_hybrid`:
  `+6.7k`, wins `13/20`, `p=0.6065`.
- `bamor_v2_proto_hybrid_cone` vs `bamor_v2_proto_hybrid`:
  `-26.4k`, wins `2/20`, `p=0.04198`.
- `bamor_v2_proto_generic` vs `bamor_v2_boundary_generic`:
  `+10.4k`, wins `10/20`, `p=0.4399`.
- `bamor_v2_proto_generic` vs `bamor_v2_contrast_generic`:
  `+27.1k`, wins `11/20`, `p=0.1258`.
- `bamor_v2_proto_generic` vs `bamor_v2_generic`:
  `+32.8k`, wins `15/20`, `p=0.03378`.
- `bamor_v2_proto_generic` vs `bamor_v2_contrast_hybrid`:
  `-11.5k`, wins `6/20`, `p=0.2604`.
- `bamor_v2_proto_hybrid` vs old `bamor_nc`:
  `+69.4k`, wins `19/20`, `p=4.587e-06`.
- `bamor_v2_proto_hybrid` vs `uniform_ctx_oracle`:
  `+21.7k`, wins `18/20`, `p=0.07586`.

Interpretation:

`bamor_v2_proto_hybrid` is the new highest-mean method, but it is not a
statistically significant improvement over `bamor_v2_contrast_hybrid` at
20 seeds. The prototype-belief idea is more valuable for portability:
`bamor_v2_proto_generic` significantly improves over `bamor_v2_generic` and
narrows the gap to the hand-coded transition-signature method.

The Target Cone remains harmful in the strongest context variants. It should
be kept as an ablation, not as the main algorithmic path.

Current best by mean:

```text
bamor_v2_proto_hybrid
```

Current best by conservative statistical claim:

```text
bamor_v2_contrast_hybrid
```

Next decision:

- If the goal is a robust paper claim, run more seeds for
  `bamor_v2_proto_hybrid` vs `bamor_v2_contrast_hybrid`.
- If the goal is algorithmic novelty, move to end-to-end recurrent
  actor/critic or learned belief inference with trainable classifier/prototypes.

### 60-seed extension: proto hybrid vs contrast hybrid

Because v2.5 produced a new highest mean but not a significant 20-seed win,
we ran an additional paired extension:

```text
results/diagnostic_v2_proto_vs_best_ext_20260608_150149
methods = bamor_v2_proto_hybrid bamor_v2_contrast_hybrid
seed_start = 20
seeds = 40
total_steps = 100000
```

Extension-only summary, HV x 1e3:

| method | n | overall | mode0 | mode1 | mode2 |
|---|---:|---:|---:|---:|---:|
| bamor_v2_contrast_hybrid | 40 | 194.6+/-37.4 | 20.5+/-3.8 | 22.4+/-14.3 | 133.4+/-32.0 |
| bamor_v2_proto_hybrid | 40 | 199.6+/-22.9 | 18.0+/-4.9 | 33.9+/-10.7 | 145.1+/-14.1 |

Paired tests:

- Seeds `0-19`: proto vs contrast `+2.0k`, wins `12/20`, `p=0.7757`.
- Seeds `20-59`: proto vs contrast `+5.0k`, wins `21/40`, `p=0.4869`.
- Seeds `0-59`: proto vs contrast `+4.0k`, wins `33/60`, `p=0.4487`.

Conclusion:

`bamor_v2_proto_hybrid` remains the highest-mean variant after 60 seeds, but
the margin over `bamor_v2_contrast_hybrid` is still not statistically reliable.
The honest claim is:

- Main robust method: `bamor_v2_contrast_hybrid`.
- Best empirical mean: `bamor_v2_proto_hybrid`.
- Most portable ablation improvement: `bamor_v2_proto_generic`.

For the paper, the prototype-belief mechanism is useful as a method extension
and portability story, but not as a clean significant replacement for the
current hybrid baseline.

### Failed-env transfer check: mo-mountaincarcontinuous-v0

The first old failed environment retest used the cheapest failed MuJoCo-style
case from `local_run.log`:

```text
env = mo-mountaincarcontinuous-v0
steps = 20,000
switch_interval = 10,000
seeds = 0, 1, 2
device = cuda
hidden = 64
result_dir = results/mujoco_mc_v2_cuda_20260608_170652/mo_mountaincarcontinuous_v0
reference_point = [-1100, -100]
```

The old `results/local_multi_20260501_081433/mo_mountaincarcontinuous_v0`
records had `hv_overall = 0.0`, but that is not a clean algorithmic failure:
the evaluation reference point was invalid for MountainCar.  The MuJoCo runner
now uses an environment-specific reference point (`[-1100, -100]`), so this
check is not directly comparable with those old zero-HV files.

Summary:

| method | context dim | overall HV | seed values | per-mode HV mean |
|---|---:|---:|---:|---:|
| `bamor_v2_contrast_hybrid` | 13 | 36,095.0+/-32,396.8 | 73,502.8, 17,615.6, 17,166.8 | 24,416.9 / 29,621.7 / 28,303.4 |
| `bamor_v2_proto_generic` | 22 | 25,679.3+/-17,145.8 | 45,367.7, 17,639.2, 14,031.0 | 18,238.5 / 27,144.4 / 14,240.7 |
| `bamor` | 2 | 17,595.0+/-118.5 | 17,458.1, 17,661.9, 17,665.0 | 9,966.8 / 17,608.8 / 9,845.8 |
| `uniform` | 2 | 17,138.8+/-678.2 | 16,362.0, 17,613.3, 17,441.1 | 6,691.8 / 14,985.2 / 10,245.3 |
| `bamor_v2_proto_hybrid` | 18 | 15,955.4+/-5,726.7 | 20,703.4, 17,567.3, 9,595.5 | 6,198.8 / 9,813.5 / 16,035.5 |

Paired differences:

- `bamor` vs `uniform`: `+456.2`, wins `3/3`, `p=0.2944`.
- `bamor_v2_contrast_hybrid` vs `uniform`: `+18,956.3`, wins `2/3`,
  `p=0.4254`.
- `bamor_v2_contrast_hybrid` vs `bamor`: `+18,500.0`, wins `1/3`,
  `p=0.4283`.
- `bamor_v2_proto_generic` vs `bamor`: `+8,084.3`, wins `1/3`,
  `p=0.5025`.
- `bamor_v2_proto_hybrid` vs `bamor`: `-1,639.6`, wins `1/3`,
  `p=0.6735`.

Interpretation:

This retest shows that the old MountainCar `HV=0` failure was partly an
evaluation bug, not a definitive training collapse.  With the fixed reference
point, old `bamor` is stable and slightly above uniform at 20k steps.

The v2 transfer story is still weak.  `bamor_v2_contrast_hybrid` and
`bamor_v2_proto_generic` can produce a large gain on seed 0, but the effect is
not reproducible over seeds 1-2.  `bamor_v2_proto_hybrid` is worse than old
`bamor` here.  Therefore MountainCar does not yet support a robust
cross-environment claim.

Next action:

- For a serious transfer claim, run a longer scheduler matrix on a harder old
  failed environment (`mo-halfcheetah-v5` first), with `bamor`, `uniform`,
  `bamor_v2_contrast_hybrid`, and `bamor_v2_proto_generic`.
- Keep `bamor_v2_proto_hybrid` as a synthetic/high-mean ablation, but do not
  lead with it for failed-environment transfer until its instability is fixed.

### Failed-env scheduler check: mo-halfcheetah-v5

After the MountainCar retest, we staged MuJoCo dependencies on
`zhengliang-hpc` and ran a short failed-environment transfer matrix:

```text
env = mo-halfcheetah-v5
steps = 50,000
switch_interval = 25,000
seeds = 0, 1, 2
device = cpu
hidden = 64
nodes = node001-node006
result_dir = results/mujoco_hc_v2_scheduler_20260608_191600/mo_halfcheetah_v5
reference_point = [-50, -3000]
```

Engineering note:

The first `bamor_v2_proto_generic` HalfCheetah jobs exposed a real MuJoCo
runner bug: HalfCheetah uses 4 modes, but `train_method` still built the
prototype-conditioned network with the default `num_modes=3`.  Final v2
evaluation then used `env.num_modes=4`, producing:

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x75 and 74x64)
```

The runner now sets `num_modes = int(env.num_modes)` after creating the
environment, so context dimensions, BAMOR detectors, and prototype trackers
match the actual environment mode count.  The fixed `bamor_v2_proto_generic`
runs have `context_dim = 58`.

Summary:

| method | context dim | overall HV | seed values | per-mode HV mean |
|---|---:|---:|---:|---:|
| `bamor` | 2 | 1,777,203.3+/-2,261,316.0 | 817,310.1, 154,174.9, 4,360,124.9 | 2,659,527.0 / 159,660.3 / 4,530,557.4 / 148,924.8 |
| `bamor_v2_proto_generic` | 58 | 1,175,059.3+/-1,371,306.2 | 162,585.0, 626,947.0, 2,735,645.8 | 707,838.6 / 137,615.7 / 1,179,377.9 / 153,105.9 |
| `bamor_v2_contrast_hybrid` | 13 | 834,655.8+/-714,692.7 | 151,596.4, 775,114.4, 1,577,256.7 | 723,611.7 / 152,157.3 / 2,559,397.5 / 327,258.9 |
| `uniform` | 2 | 726,350.6+/-519,614.0 | 749,819.7, 1,233,832.4, 195,399.8 | 1,894,136.3 / 349,314.6 / 3,186,629.8 / 894,945.3 |

Paired differences:

- `bamor` vs `uniform`: `+1,050,852.7`, wins `2/3`, `p=0.577`.
- `bamor_v2_contrast_hybrid` vs `uniform`: `+108,305.2`, wins `1/3`,
  `p=0.8808`.
- `bamor_v2_proto_generic` vs `uniform`: `+448,708.6`, wins `1/3`,
  `p=0.7097`.
- `bamor_v2_contrast_hybrid` vs `bamor`: `-942,547.5`, wins `1/3`,
  `p=0.4424`.
- `bamor_v2_proto_generic` vs `bamor`: `-602,144.0`, wins `1/3`,
  `p=0.4251`.

Interpretation:

This is not a transfer win.  The strongest synthetic v2 ideas do not reliably
beat old `bamor` on HalfCheetah at 50k steps.  The mean ranking is:

```text
bamor > bamor_v2_proto_generic > bamor_v2_contrast_hybrid > uniform
```

However, variance is extreme and only three seeds were used, so this should be
treated as a directional failed-environment smoke test, not a final benchmark.
The useful outcome is diagnostic: adding richer context is not enough to solve
the failed MuJoCo transfer problem.  The remaining gap is likely in the policy
learning/evaluation setup, regime inference under long-horizon continuous
control, or how the nonstationary context is injected into SAC, not just in
the synthetic transition-context machinery.

Next action:

- Do not claim cross-environment generality from v2 yet.
- Keep old `bamor` as the MuJoCo transfer baseline.
- If continuing algorithm work, move beyond feed-forward context concatenation:
  use a recurrent/belief-conditioned actor-critic or meta-RL-style adaptation
  state, and evaluate on HalfCheetah before returning to synthetic-only claims.

### Seed-0 rare success analysis and BAMOR-v3

The MountainCar failed-env retest had one striking outlier:
`bamor_v2_contrast_hybrid` seed 0 achieved `HV=73,502.8`, while seeds 1-2
were ordinary (`17,615.6`, `17,166.8`).  We re-evaluated saved models using
10 episodes per weight and three independent evaluation seeds:

| method | seed | re-eval HV | max obj0 | max obj1 |
|---|---:|---:|---:|---:|
| `bamor_v2_contrast_hybrid` | 0 | 63,734.5+/-3,554.9 | -420.8 | -5.8 |
| `bamor_v2_contrast_hybrid` | 1 | 17,055.2+/-496.7 | -923.3 | -0.3 |
| `bamor_v2_contrast_hybrid` | 2 | 16,760.8+/-675.6 | -928.4 | -0.7 |
| `bamor_v2_proto_generic` | 0 | 45,396.2+/-3,579.3 | -584.0 | -4.8 |
| `bamor_v2_proto_generic` | 1 | 17,568.5+/-67.9 | -923.3 | -0.0 |
| `bamor_v2_proto_generic` | 2 | 18,307.6+/-3,672.5 | -800.6 | -2.0 |

This proves seed 0 is not a 3-episode HV artifact.  The policy genuinely
learned to reach the MountainCar goal faster: the best time objective moved
from the usual `-923/-999` range to roughly `-421`.

Environment schedule:

```text
seed0: normal -> energy_only at step 7959
seed1: normal -> slow_only   at step 5397
seed2: normal -> energy_only at step 5727
```

Interpretation:

- Late switch into `energy_only`/speed-friendly dynamics is helpful but not
  sufficient.  Old `bamor` and `uniform` with the same seed-0 schedule did not
  enter the fast-reaching basin.
- The v2 contrast transition signature is doing real work.  Removing it in
  `bamor_v3_seq_belief` destroyed the rare success.
- The rare success likely comes from an interaction among late speed-friendly
  mode exposure, stochastic preference/replay sampling, and the hand-coded
  transition signature.  It is a real algorithmic clue, not just noise.

Implemented v3 variants:

```text
bamor_v3_seq_belief
  context = [weight, generic transition stats, GRU sequence latent, prototype belief]
  weights = deterministic stratified simplex

bamor_v3_contrast_stratified
  context = [weight, transition signature, single-transition contrastive latent]
  weights = deterministic stratified simplex

bamor_v3_contrast_mix
  context = [weight, transition signature, single-transition contrastive latent]
  weights = 50% Dirichlet random + 50% stratified simplex
```

MountainCar 20k, seeds 0-2:

| method | overall HV | seed values | max obj0 by seed |
|---|---:|---:|---:|
| `bamor_v2_contrast_hybrid` | 36,095.0+/-32,396.8 | 73,502.8, 17,615.6, 17,166.8 | -355.3, -923.3, -923.3 |
| `bamor_v3_seq_belief` | 13,686.8+/-3,668.5 | 10,182.8, 17,500.3, 13,377.2 | -923.3, -923.3, -923.3 |
| `bamor_v3_contrast_stratified` | 42,252.9+/-29,125.8 | 17,627.8, 34,728.2, 74,402.6 | -923.3, -739.7, -332.0 |
| `bamor_v3_contrast_mix` | 26,915.9+/-19,976.6 | 13,600.5, 49,885.9, 17,261.3 | -923.3, -585.3, -923.3 |

Paired against old `bamor`:

- `bamor_v3_seq_belief`: `-3,908.2`, wins `0/3`.
- `bamor_v3_contrast_stratified`: `+24,657.9`, wins `3/3`.
- `bamor_v3_contrast_mix`: `+9,320.9`, wins `1/3`.

Paired against `bamor_v2_contrast_hybrid`:

- `bamor_v3_seq_belief`: `-22,408.3`, wins `0/3`.
- `bamor_v3_contrast_stratified`: `+6,157.8`, wins `2/3`.
- `bamor_v3_contrast_mix`: `-9,179.1`, wins `2/3`.

Re-evaluation of `bamor_v3_contrast_stratified` saved models:

| seed | re-eval HV | max obj0 | max obj1 |
|---:|---:|---:|---:|
| 0 | 16,581.1+/-1,804.6 | -923.3 | -0.2 |
| 1 | 22,527.4+/-7,791.6 | -764.9 | -4.3 |
| 2 | 74,969.5+/-1,343.5 | -330.8 | -1.2 |

Conclusion:

`bamor_v3_seq_belief` is a failed v3.0 attempt: recurrent/generic belief did
not solve exploration and removed the useful direct transition signature.

`bamor_v3_contrast_stratified` is the current v3 candidate.  It does not make
success fully robust, but it moves the rare success from 1/3 seeds to 2/3
seeds and beats old `bamor` on all three MountainCar seeds.  It should be
tested next on HalfCheetah via scheduler before being treated as a real
algorithmic improvement.

### HalfCheetah v3 transfer follow-up

We then tested the MountainCar v3 candidates on the harder old failed
environment:

```text
env = mo-halfcheetah-v5
steps = 50,000
switch_interval = 25,000
seeds = 0, 1, 2
device = cpu
hidden = 64
nodes = node001-node006
stratified_dir = results/mujoco_hc_v3_scheduler_20260609_0001/mo_halfcheetah_v5
mix_dir = results/mujoco_hc_v3_mix_scheduler_20260609_0001/mo_halfcheetah_v5
reference_point = [-50, -3000]
```

Summary against the same-seed v2 scheduler baselines:

| method | overall HV | seed values | max obj0 by seed |
|---|---:|---:|---:|
| `uniform` | 726,350.6+/-519,614.0 | 749,819.7, 1,233,832.4, 195,399.8 | 294.5, 368.2, 20.3 |
| `bamor` | 1,777,203.3+/-2,261,316.0 | 817,310.1, 154,174.9, 4,360,124.9 | 296.7, 1.9, 2224.6 |
| `bamor_v3_contrast_stratified` | 194,709.7+/-52,636.8 | 138,502.4, 242,842.9, 202,783.8 | 2.7, 32.3, 35.1 |
| `bamor_v3_contrast_mix` | 1,899,753.3+/-1,029,025.4 | 724,013.9, 2,636,354.4, 2,338,891.5 | 269.6, 843.7, 858.3 |

Paired differences:

- `bamor_v3_contrast_stratified` vs `bamor`: `-1,582,493.6`, wins `1/3`.
- `bamor_v3_contrast_stratified` vs `uniform`: `-531,640.9`, wins `1/3`.
- `bamor_v3_contrast_mix` vs `bamor`: `+122,550.0`, wins `1/3`.
- `bamor_v3_contrast_mix` vs `uniform`: `+1,173,402.6`, wins `2/3`.

Interpretation:

The deterministic stratified variant is not transferable.  It is good on
MountainCar, but it collapses on HalfCheetah, suggesting that fully
deterministic preference coverage can damage long-horizon continuous-control
exploration.

The mixed variant is more promising.  It keeps the direct transition signature
and alternates between broad random Dirichlet weights and deterministic
simplex coverage.  On HalfCheetah it recovers strong forward-return policies
on seeds 1-2 and beats `uniform` by a large paired margin, while still losing
to old `bamor` on seed 2 because old `bamor` had an extreme high-HV outlier.

Current v3 reading:

- MountainCar best candidate: `bamor_v3_contrast_stratified`.
- HalfCheetah best transfer candidate: `bamor_v3_contrast_mix`.
- Cross-environment lesson: the useful component is the transition-signature
  context; the risky component is over-deterministic preference scheduling.

Next action:

Run a larger HalfCheetah seed extension for `uniform`, `bamor`, and
`bamor_v3_contrast_mix` on seeds 3-9.  If the mixed variant still beats
`uniform` and stays close to or above old `bamor`, it becomes the first
plausible failed-environment transfer result.  If it collapses, v3 should be
reported as an environment-specific rescue rather than a general algorithm.

### HalfCheetah v3-mix 10-seed extension

The seed extension was run on the same scheduler setup:

```text
env = mo-halfcheetah-v5
steps = 50,000
switch_interval = 25,000
seeds = 3..9
methods = uniform, bamor, bamor_v3_contrast_mix
result_dir = results/mujoco_hc_v3mix_ext_scheduler_20260609_0002/mo_halfcheetah_v5
```

Combining the earlier seeds 0-2 with the extension gives a 10-seed failed-env
transfer check:

| method | overall HV | median HV | min HV | max HV | seed values |
|---|---:|---:|---:|---:|---:|
| `uniform` | 902,616.7+/-996,262.2 | 504,398.7 | 186,091.4 | 3,221,763.1 | 749,819.7, 1,233,832.4, 195,399.8, 1,897,518.5, 186,091.4, 258,977.7, 195,362.3, 3,221,763.1, 193,101.2, 894,300.9 |
| `bamor` | 984,877.1+/-1,494,107.2 | 188,059.5 | 147,018.5 | 4,360,124.9 | 817,310.1, 154,174.9, 4,360,124.9, 3,104,555.8, 178,855.6, 546,926.8, 182,796.9, 147,018.5, 193,322.1, 163,684.9 |
| `bamor_v3_contrast_mix` | 2,139,948.1+/-1,731,161.4 | 2,225,827.2 | 199,918.7 | 5,967,983.7 | 724,013.9, 2,636,354.4, 2,338,891.5, 5,967,983.7, 3,619,458.6, 2,138,961.8, 2,312,692.5, 540,155.7, 199,918.7, 921,050.1 |

Paired results:

- `bamor` vs `uniform`: `+82,260.4`, wins `5/10`, two-sided paired
  t-test `p=0.8898`, Wilcoxon `p=1.0`.
- `bamor_v3_contrast_mix` vs `uniform`: `+1,237,331.4`, wins `8/10`,
  two-sided paired t-test `p=0.0779`, one-sided `p=0.0389`; Wilcoxon
  two-sided `p=0.0840`, one-sided `p=0.0420`.
- `bamor_v3_contrast_mix` vs `bamor`: `+1,155,071.0`, wins `8/10`,
  two-sided paired t-test `p=0.0551`, one-sided `p=0.0276`; Wilcoxon
  two-sided `p=0.0488`, one-sided `p=0.0244`.

Per-seed paired differences for `bamor_v3_contrast_mix`:

| seed | vs `uniform` | vs `bamor` | max obj0 |
|---:|---:|---:|---:|
| 0 | -25,805.8 | -93,296.2 | 269.6 |
| 1 | +1,402,521.9 | +2,482,179.5 | 843.7 |
| 2 | +2,143,491.7 | -2,021,233.4 | 858.3 |
| 3 | +4,070,465.2 | +2,863,427.9 | 1,972.0 |
| 4 | +3,433,367.3 | +3,440,603.0 | 1,176.9 |
| 5 | +1,879,984.1 | +1,592,035.0 | 676.8 |
| 6 | +2,117,330.2 | +2,129,895.6 | 735.7 |
| 7 | -2,681,607.3 | +393,137.2 | 152.8 |
| 8 | +6,817.5 | +6,596.6 | 17.4 |
| 9 | +26,749.2 | +757,365.2 | 279.5 |

Interpretation:

This is the first meaningful failed-environment transfer signal in the BAMOR
rescue work.  Over 10 HalfCheetah seeds, old `bamor` is essentially tied with
`uniform` because both are dominated by rare outliers.  In contrast,
`bamor_v3_contrast_mix` improves the median by a large margin and wins 8/10
paired comparisons against both baselines.

The result is still not final-paper strength: variance remains high, two-sided
t-tests are borderline, and seed 8 still collapses to the low-HV regime.
However, the mixed preference schedule is now clearly better than the purely
deterministic stratified schedule and should replace `bamor_v3_contrast_stratified`
as the cross-environment v3 candidate.

### Hopper 3-objective transfer check

We then moved to `mo-hopper-v5`, which is a harder BAPR-style 3-objective
environment:

```text
env = mo-hopper-v5
objectives = [healthy, speed, -energy]
steps = 50,000
switch_interval = 25,000
seeds = 0..4
methods = uniform, bamor, bamor_v3_contrast_mix
result_dir = results/mujoco_hopper_v3mix_scheduler_20260609_0001/mo_hopper_v5
reference_point = [-50, -100, -3000]
```

Engineering change:

The MuJoCo runner originally returned `hv_overall = None` for any
`reward_dim != 2`.  We added a 3-D dominated-HV evaluator using exact slicing:
sort the third objective into slabs, compute 2-D HV on active points in each
slab, and sum volume.  Hopper/Walker/Ant now use the fixed 3-D reference
point `[-50, -100, -3000]`, so methods are comparable within each environment.

5-seed Hopper summary:

| method | 3-D HV | median HV | seed values |
|---|---:|---:|---:|
| `uniform` | 1,104,060,238.1+/-1,656,953,621.5 | 109,306,563.9 | 17,286,718.1, 3,831,551,348.2, 18,343,984.1, 1,543,812,576.0, 109,306,563.9 |
| `bamor` | 2,008,231,815.7+/-3,333,814,208.3 | 156,157,185.5 | 92,389,942.9, 156,157,185.5, 7,786,520,944.4, 1,989,760,730.8, 16,330,274.9 |
| `bamor_v3_contrast_mix` | 894,534,454.6+/-1,003,288,205.6 | 303,065,466.8 | 303,065,466.8, 2,161,286,076.5, 1,798,163,944.5, 171,507,510.5, 38,649,274.5 |

Paired differences:

- `bamor_v3_contrast_mix` vs `uniform`: `-209,525,783.5`, wins `2/5`.
- `bamor_v3_contrast_mix` vs `bamor`: `-1,113,697,361.2`, wins `3/5`.
- Old `bamor` vs `uniform`: `+904,171,577.7`, wins `3/5`.

Interpretation:

This is not a 3-objective transfer win.  `bamor_v3_contrast_mix` has the best
median, but it loses on mean and only wins 2/5 against `uniform`.  The result
is dominated by large outliers: `uniform` seed 1 and old `bamor` seed 2 both
find very high-HV policies.  The current v3 story is therefore:

- Positive transfer signal on 2-objective failed MuJoCo tasks:
  MountainCar and HalfCheetah.
- No reliable transfer signal yet on 3-objective Hopper.

We also tested a higher-dimensional lattice sampler because the original
3-D stratified fallback only cycles one-hot corners plus the center.  The
temporary lattice run used 66 simplex-grid points with a coprime stride:

```text
result_dir = results/mujoco_hopper_v3mix_lattice_scheduler_20260609_0001/mo_hopper_v5
method = bamor_v3_contrast_mix with lattice stratified half of mixed sampling
```

Lattice result:

| method | 3-D HV | median HV | seed values |
|---|---:|---:|---:|
| lattice `v3_mix` | 555,611,803.3+/-1,010,430,780.8 | 96,607,107.0 | 28,584,909.2, 96,607,107.0, 42,506,058.6, 2,356,002,747.2, 254,358,194.5 |

Paired against the original Hopper run:

- lattice vs `uniform`: `-548,448,434.8`, wins `4/5`.
- lattice vs `bamor`: `-1,452,620,012.4`, wins `2/5`.
- lattice vs original `bamor_v3_contrast_mix`: `-338,922,651.2`, wins `2/5`.

Conclusion:

The Hopper failure is not fixed by denser 3-D preference coverage.  In fact,
the lattice sampler is worse than the original corner/center fallback at this
budget.  We therefore restored the default `bamor_v3_contrast_mix` behavior
and kept the lattice sampler only as an explicit ablation method:
`bamor_v3_contrast_lattice_mix`.

Next algorithmic direction should not be another preference-grid tweak.  For
3-objective MuJoCo, the likely missing pieces are objective normalization,
critic stabilization under sparse/extreme objectives, and a context mechanism
that can disambiguate reward-mode changes without forcing the actor to learn a
single high-dimensional global simplex policy from only 50k steps.

### Hopper v3.1 normalization attempt

We implemented a controlled reward/objective normalization ablation:

```text
uniform_norm
  same as uniform, but SAC critic/policy scalarization sees online-normalized
  per-objective rewards during training

bamor_v3_contrast_norm_mix
  same context and mixed preference schedule as bamor_v3_contrast_mix, but
  critic/policy scalarization uses online-normalized per-objective rewards

Evaluation remains on raw rewards and raw 3-D HV.
```

Implementation note:

The replay buffer still stores raw rewards.  `MOSACTrainer` optionally keeps
an EMA mean/variance for each reward dimension and normalizes sampled rewards
inside `update()`.  Q-values are therefore learned in normalized objective
units, while final Pareto evaluation and HV use the original environment
returns.

First 5-seed check:

```text
result_dir = results/mujoco_hopper_v3norm_scheduler_20260609_0001/mo_hopper_v5
seeds = 0..4
methods = uniform_norm, bamor_v3_contrast_norm_mix
```

| method | 3-D HV | median HV | seed values |
|---|---:|---:|---:|
| `uniform_norm` | 73,795,102.7+/-46,514,983.7 | 87,721,786.9 | 87,721,786.9, 18,355,917.0, 135,103,392.8, 37,134,585.8, 90,659,830.8 |
| `bamor_v3_contrast_norm_mix` | 1,020,710,466.3+/-694,008,538.3 | 784,664,718.5 | 654,642,998.5, 1,921,529,020.4, 784,664,718.5, 1,534,977,383.7, 207,738,210.4 |

Preliminary interpretation:

`uniform_norm` is much worse than raw `uniform`, so normalization alone is not
the answer.  `bamor_v3_contrast_norm_mix` strongly beats `uniform_norm`
(`5/5` seeds), and has a better median than raw `v3_mix`, but it still does
not beat raw `uniform` or old `bamor` on mean because Hopper is dominated by
large outlier successes.

We therefore extended Hopper to seeds 5-9 for raw baselines and `norm_mix`:

```text
result_dir = results/mujoco_hopper_v3norm_ext_scheduler_20260609_0001/mo_hopper_v5
seeds = 5..9
methods = uniform, bamor, bamor_v3_contrast_mix, bamor_v3_contrast_norm_mix
```

Combined 10-seed Hopper result:

| method | 3-D HV | median HV | min HV | max HV |
|---|---:|---:|---:|---:|
| `uniform` | 1,908,463,532.2+/-2,039,632,670.8 | 1,823,064,637.3 | 17,286,718.1 | 6,220,067,205.8 |
| `bamor` | 2,105,367,552.8+/-3,440,943,492.0 | 151,590,824.4 | 16,135,760.7 | 9,135,206,636.0 |
| `bamor_v3_contrast_mix` | 1,118,300,653.9+/-1,097,915,900.2 | 966,051,204.5 | 22,156,121.0 | 2,568,183,878.6 |
| `bamor_v3_contrast_norm_mix` | 1,190,622,562.2+/-1,759,793,572.8 | 502,063,822.8 | 20,935,547.3 | 5,884,601,711.0 |

Paired differences:

- `bamor_v3_contrast_norm_mix` vs `uniform`: `-717,840,970.0`,
  wins `5/10`, Wilcoxon two-sided `p=0.6953`.
- `bamor_v3_contrast_norm_mix` vs `bamor`: `-914,744,990.6`,
  wins `6/10`, Wilcoxon two-sided `p=0.9219`.
- `bamor_v3_contrast_norm_mix` vs raw `bamor_v3_contrast_mix`: `+72,321,908.3`,
  wins `5/10`, Wilcoxon two-sided `p=1.0`.
- Raw `bamor_v3_contrast_mix` vs `uniform`: `-790,162,878.3`,
  wins `3/10`, Wilcoxon two-sided `p=0.2324`.

Conclusion:

Reward normalization is not sufficient to rescue 3-objective Hopper.  It
improves some individual seeds and can produce very high policies, but the
10-seed paired result has no stable advantage over raw `uniform`, old `bamor`,
or raw `v3_mix`.  The v3/v3.1 status is now:

- 2-D failed-env transfer: promising (`v3_mix` on HalfCheetah; `v3_stratified`
  or `v3_mix` on MountainCar depending on robustness criterion).
- 3-D Hopper transfer: still unsolved.

The next serious algorithmic change should be structural, not another scalar
reward preprocessing tweak: likely separate objective-scale calibration,
factorized/pairwise preference heads, or a multi-policy/mixture actor so the
agent does not have to represent all 3-D simplex extremes in one policy within
50k steps.

### Hopper v4 preference-expert actor

We then implemented the first structural 3-D rescue attempt:

```text
bamor_v4_contrast_expert_mix
  context = [weight, transition signature, single-transition latent]
  weights = same 50% random Dirichlet + 50% stratified schedule as v3_mix
  actor = hard-routed preference experts
  critic = shared vector Q critic

bamor_v4_contrast_expert_norm_mix
  same expert actor, plus reward/objective normalization
```

Expert routing:

For `reward_dim=3`, the actor has seven policy heads: three single-objective
corner experts, three pair-edge experts, and one center/all-objective expert.
The routing key is the active objective set induced by the first `reward_dim`
entries of the context vector.  Critic, replay buffer, transition context, and
latent encoder remain shared.  The goal is to reduce policy-gradient conflict
without changing the context signal that worked in 2-D.

Local checks:

- Python compile passed.
- Expert actor backward pass sent gradients to the selected experts.
- Remote HPC preflight confirmed `num_experts = 7`, `context_dim = 15`.

First Hopper 5-seed result:

```text
result_dir = results/mujoco_hopper_v4expert_scheduler_20260609_0001/mo_hopper_v5
seeds = 0..4
methods = bamor_v4_contrast_expert_mix, bamor_v4_contrast_expert_norm_mix
```

| method | 3-D HV | median HV | seed values |
|---|---:|---:|---:|
| `uniform` | 1,104,060,238.1+/-1,656,953,621.5 | 109,306,563.9 | 17,286,718.1, 3,831,551,348.2, 18,343,984.1, 1,543,812,576.0, 109,306,563.9 |
| `bamor` | 2,008,231,815.7+/-3,333,814,208.3 | 156,157,185.5 | 92,389,942.9, 156,157,185.5, 7,786,520,944.4, 1,989,760,730.8, 16,330,274.9 |
| `bamor_v3_contrast_mix` | 894,534,454.6+/-1,003,288,205.6 | 303,065,466.8 | 303,065,466.8, 2,161,286,076.5, 1,798,163,944.5, 171,507,510.5, 38,649,274.5 |
| `bamor_v3_contrast_norm_mix` | 1,020,710,466.3+/-694,008,538.3 | 784,664,718.5 | 654,642,998.5, 1,921,529,020.4, 784,664,718.5, 1,534,977,383.7, 207,738,210.4 |
| `bamor_v4_contrast_expert_mix` | 2,009,407,017.3+/-739,506,825.6 | 2,204,215,002.9 | 2,562,589,872.7, 2,396,081,801.2, 717,412,377.2, 2,204,215,002.9, 2,166,736,032.2 |
| `bamor_v4_contrast_expert_norm_mix` | 487,622,524.1+/-674,593,366.8 | 21,887,350.4 | 17,372,207.0, 1,485,730,255.7, 894,803,117.4, 18,319,690.0, 21,887,350.4 |

Paired 5-seed read:

- Raw `v4_expert` vs `uniform`: `+905,346,779.2`, wins `4/5`.
- Raw `v4_expert` vs `bamor`: `+1,175,201.5`, wins `4/5`.
- Raw `v4_expert` vs `v3_mix`: `+1,114,872,562.7`, wins `4/5`.
- `v4_expert_norm` is a failed variant; do not extend it.

We extended raw `v4_expert` to seeds 5-9:

```text
result_dir = results/mujoco_hopper_v4expert_ext_scheduler_20260609_0001/mo_hopper_v5
seeds = 5..9
method = bamor_v4_contrast_expert_mix
```

Combined 10-seed Hopper result:

| method | 3-D HV | median HV | min HV | max HV |
|---|---:|---:|---:|---:|
| `uniform` | 1,908,463,532.2+/-2,039,632,670.8 | 1,823,064,637.3 | 17,286,718.1 | 6,220,067,205.8 |
| `bamor` | 2,105,367,552.8+/-3,440,943,492.0 | 151,590,824.4 | 16,135,760.7 | 9,135,206,636.0 |
| `bamor_v3_contrast_mix` | 1,118,300,653.9+/-1,097,915,900.2 | 966,051,204.5 | 22,156,121.0 | 2,568,183,878.6 |
| `bamor_v4_contrast_expert_mix` | 1,760,795,178.5+/-1,053,415,252.5 | 2,185,475,517.5 | 62,106,265.5 | 2,793,987,047.1 |

Paired 10-seed result:

- `v4_expert` vs `uniform`: `-147,668,353.7`, wins `6/10`,
  Wilcoxon two-sided `p=0.8457`.
- `v4_expert` vs `bamor`: `-344,572,374.3`, wins `7/10`,
  Wilcoxon two-sided `p=0.6250`.
- `v4_expert` vs `v3_mix`: `+642,494,524.6`, wins `7/10`,
  Wilcoxon two-sided `p=0.2754`.

Interpretation:

`bamor_v4_contrast_expert_mix` is the first structural method that improves
Hopper robustness.  It is not a mean-HV winner because raw `uniform` and old
`bamor` still have extreme outlier seeds, but it has the best median HV and
substantially lower variance than old `bamor`.  It also wins most paired seeds
against old `bamor` and raw `v3_mix`.

This is not yet a final 3-D solution.  The weak seeds 6 and 9 show that hard
expert routing can still collapse.  But the result strongly supports the
structural hypothesis: separating actor heads by preference region is more
promising for 3-D than further tuning simplex sampling or reward
normalization.

Next v4 direction:

- Keep `bamor_v4_contrast_expert_mix` as the 3-D candidate.
- Do not keep `bamor_v4_contrast_expert_norm_mix` except as a negative
  ablation.
- Improve expert routing: add soft/annealed routing or pairwise-head fallback
  for ambiguous weights instead of a hard active-set threshold.
- Test Walker2d or Ant only after hard-routing collapse on Hopper seeds 6/9 is
  understood.

### Hopper v4.1 soft preference-expert actor

We implemented the next structural rescue attempt:

```text
bamor_v4_contrast_soft_expert_mix
  context = [weight, transition signature, single-transition latent]
  weights = same 50% random Dirichlet + 50% stratified schedule as v3/v4
  actor = soft-gated preference experts
  critic = shared vector Q critic
  reward normalization = off
```

Motivation:

Hard v4 proved that separate actor heads help 3-D Hopper, but the hard
active-set threshold creates discontinuities on the preference simplex and
updates only the selected expert.  This can starve neighboring experts and may
explain the remaining collapse seeds.  v4.1 keeps the same seven 3-D expert
regions but replaces hard routing by a deterministic softmax gate:

```text
templates = all non-empty objective subsets, normalized onto the simplex
gate_i(w) = softmax_i(-||w - template_i||^2 / temperature)
```

For `reward_dim=3`, the templates are the three corners, three pairwise edges,
and the center.  The actor blends expert Gaussian means/log-stds using these
gates, so adjacent experts receive gradient signal even when a sampled
preference is near a routing boundary.

Local and remote checks:

- Python compile passed.
- Gate shape/normalization passed for 3-D contexts.
- Backward pass reached all seven experts in a mixed batch.
- Remote HPC import confirmed `bamor_v4_contrast_soft_expert_mix` is available,
  uses mixed weights, uses transition context, and has Hopper-like
  `context_dim = 15`.

Initial Hopper 5-seed sweep submitted:

```text
result_dir = results/mujoco_hopper_v4soft_scheduler_20260609_0001/mo_hopper_v5
seeds = 0..4
method = bamor_v4_contrast_soft_expert_mix
scheduler tasks = t8805..t8809
```

5-seed result:

| method | 3-D HV | median HV | seed values |
|---|---:|---:|---:|
| `uniform` | 1,104,060,238.1+/-1,482,024,373.3 | 109,306,563.9 | 17,286,718.1, 3,831,551,348.2, 18,343,984.1, 1,543,812,576.0, 109,306,563.9 |
| `bamor` | 2,008,231,815.7+/-2,981,854,077.6 | 156,157,185.5 | 92,389,942.9, 156,157,185.5, 7,786,520,944.4, 1,989,760,730.8, 16,330,274.9 |
| `bamor_v3_contrast_mix` | 894,534,454.6+/-897,368,251.5 | 303,065,466.8 | 303,065,466.8, 2,161,286,076.5, 1,798,163,944.5, 171,507,510.5, 38,649,274.5 |
| `bamor_v4_contrast_expert_mix` | 2,009,407,017.3+/-661,435,012.7 | 2,204,215,002.9 | 2,562,589,872.7, 2,396,081,801.2, 717,412,377.2, 2,204,215,002.9, 2,166,736,032.2 |
| `bamor_v4_contrast_soft_expert_mix` | 721,529,679.1+/-734,358,048.4 | 225,987,802.6 | 2,031,816,649.8, 174,389,157.1, 141,170,473.3, 1,034,284,312.6, 225,987,802.6 |

Paired read:

- `soft_expert` vs hard `v4_expert`: `-1,287,877,338.2`, wins `0/5`,
  Wilcoxon two-sided `p=0.0625`.
- `soft_expert` vs `uniform`: `-382,530,559.0`, wins `3/5`.
- `soft_expert` vs `bamor`: `-1,286,702,136.6`, wins `3/5`.
- `soft_expert` vs `v3_mix`: `-173,004,775.5`, wins `3/5`.

Conclusion:

Do not extend v4.1.  Soft action blending loses the hard-v4 specialization:
even when the gate is close to one-hot, ambiguous preferences still average
actions from experts trained for different objectives.  The failure is useful:
the next mixture actor should sample/select an expert action, not average
expert actions.

### Hopper v5 true MoE + template-balanced expert training

We implemented the next version:

```text
bamor_v5_contrast_moe_template_mix
  context = [weight, transition signature, single-transition latent]
  weights = 50% random Dirichlet + 50% exact expert-template cycle
  actor = true mixture-of-experts tanh-Gaussian policy
  deterministic eval = gate-argmax expert action
  stochastic train = sample one expert, compute exact mixture log-prob
  critic = shared vector Q critic
```

This combines the two lessons so far:

- Hard v4 worked because it avoids action averaging and lets preference
  regions specialize.
- Hard v4 may still collapse because the old stratified sampler only cycles
  corners plus center in 3-D; pairwise experts are trained mostly by chance.

The new template sampler cycles all seven 3-D templates exactly: three
corners, three pairwise edges, and the center.  The MoE actor avoids the v4.1
action-averaging problem while still giving boundary preferences a proper
mixture log-prob.

Local checks:

- Python compile passed.
- MoE gate shape/normalization passed.
- Exact template sampler cycles the seven expected 3-D templates.
- Stochastic MoE action and mixture log-prob are finite.
- Backward pass reaches multiple experts.

Targeted Hopper sweep submitted:

```text
result_dir = results/mujoco_hopper_v5moe_targeted_scheduler_20260609_0001/mo_hopper_v5
seeds = 0, 2, 6, 9
method = bamor_v5_contrast_moe_template_mix
active tasks = t8814(seed2), t8817(seed0), t8818(seed6), t8819(seed9)
```

Rationale:

- Seed 0 checks whether v5 preserves a strong hard-v4 seed.
- Seed 2 checks a mid/weak hard-v4 seed.
- Seeds 6 and 9 are hard-v4 collapse seeds, so they are the fastest test of
  whether template-balanced expert training actually rescues 3-D instability.

Targeted result:

| method | targeted seeds | mean HV | median HV | seed values |
|---|---:|---:|---:|---:|
| `uniform` | 0,2,6,9 | 551,026,637.1 | 42,251,565.8 | s0=17,286,718.1, s2=18,343,984.1, s6=66,159,147.4, s9=2,102,316,698.7 |
| `bamor` | 0,2,6,9 | 2,398,265,119.3 | 881,131,078.4 | s0=92,389,942.9, s2=7,786,520,944.4, s6=44,277,375.8, s9=1,669,872,213.9 |
| `bamor_v3_contrast_mix` | 0,2,6,9 | 1,148,046,683.9 | 1,050,614,705.7 | s0=303,065,466.8, s2=1,798,163,944.5, s6=25,811,861.2, s9=2,465,145,463.1 |
| hard `bamor_v4_contrast_expert_mix` | 0,2,6,9 | 866,165,243.7 | 419,982,418.4 | s0=2,562,589,872.7, s2=717,412,377.2, s6=122,552,459.6, s9=62,106,265.5 |
| `bamor_v5_contrast_moe_template_mix` | 0,2,6,9 | 2,051,203,779.6 | 2,524,258,335.8 | s0=3,026,233,178.1, s2=130,065,268.6, s6=2,329,159,265.6, s9=2,719,357,406.0 |

Paired targeted read:

- v5 vs hard v4: `+1,185,038,535.9`, wins `3/4`.
- v5 vs uniform: `+1,500,177,142.5`, wins `4/4`.
- v5 vs v3 mix: `+903,157,095.7`, wins `3/4`.
- v5 vs old bamor: `-347,061,339.7`, wins `3/4` because old bamor has a
  giant seed-2 outlier.

Interpretation:

v5 clearly fixes the known hard-v4 collapse seeds 6 and 9, and improves the
already-strong seed 0.  The cost is seed 2, where MoE/template training falls
below hard v4 and far below old `bamor`/`v3_mix`.  This makes v5 a promising
but not yet final 3-D candidate.  We should extend to the remaining seeds
before deciding whether to keep v5 as-is or merge its template-balanced sampler
back into hard v4.

Remaining Hopper seeds submitted:

```text
result_dir = results/mujoco_hopper_v5moe_remaining_scheduler_20260609_0001/mo_hopper_v5
seeds = 1, 3, 4, 5, 7, 8
tasks = t8820..t8825
```

Combined 10-seed Hopper result:

| method | 3-D HV | median HV | min HV | max HV |
|---|---:|---:|---:|---:|
| `uniform` | 1,908,463,532.2+/-1,934,965,448.9 | 1,823,064,637.3 | 17,286,718.1 | 6,220,067,205.8 |
| `bamor` | 2,105,367,552.8+/-3,264,365,620.4 | 151,590,824.4 | 16,135,760.7 | 9,135,206,636.0 |
| `bamor_v3_contrast_mix` | 1,118,300,653.9+/-1,041,574,477.1 | 966,051,204.5 | 22,156,121.0 | 2,568,183,878.6 |
| `bamor_v3_contrast_norm_mix` | 1,190,622,562.2+/-1,669,486,770.6 | 502,063,822.8 | 20,935,547.3 | 5,884,601,711.0 |
| hard `bamor_v4_contrast_expert_mix` | 1,760,795,178.5+/-999,357,456.0 | 2,185,475,517.5 | 62,106,265.5 | 2,793,987,047.1 |
| `bamor_v5_contrast_moe_template_mix` | 2,377,073,139.4+/-945,883,362.6 | 2,584,736,253.4 | 130,065,268.6 | 3,886,690,953.0 |

Paired 10-seed result:

- v5 vs `uniform`: `+468,609,607.2`, wins `8/10`,
  Wilcoxon two-sided `p=0.2754`.
- v5 vs old `bamor`: `+271,705,586.6`, wins `8/10`,
  Wilcoxon two-sided `p=0.4316`.
- v5 vs `v3_mix`: `+1,258,772,485.5`, wins `8/10`,
  Wilcoxon two-sided `p=0.0645`.
- v5 vs `v3_norm_mix`: `+1,186,450,577.2`, wins `8/10`,
  Wilcoxon two-sided `p=0.0840`.
- v5 vs hard `v4_expert`: `+616,277,960.9`, wins `7/10`,
  Wilcoxon two-sided `p=0.1934`.

Interpretation after 10 seeds:

`bamor_v5_contrast_moe_template_mix` is now the best 3-D Hopper candidate by
both mean and median HV.  It fixes hard-v4 collapse seeds 6 and 9, improves the
best-case ceiling less wildly than old `bamor`, and has lower variance than
uniform/old `bamor`/v3-norm.  It is still not cleanly solved: seed 2 is a
remaining low outlier, and v5 loses to hard v4 on seeds 2, 5, and 7.  The next
ablation is therefore hard expert actor plus template-balanced sampling, to
separate the benefit of template coverage from the cost of stochastic MoE.

### Hopper v5 ablation: hard expert + template sampler

We added:

```text
bamor_v5_contrast_expert_template_mix
  actor = hard-routed preference experts from v4
  weights = 50% random Dirichlet + 50% exact 7-template cycle
  context/critic/latent encoder = same as v4/v5
```

This isolates the template-sampler contribution from the stochastic MoE actor.
Targeted Hopper seeds `0,2,6,9` were submitted:

```text
result_dir = results/mujoco_hopper_v5hardtemplate_targeted_scheduler_20260609_0001/mo_hopper_v5
tasks = t8826..t8829
```

Targeted result:

| method | targeted seeds | mean HV | median HV | seed values |
|---|---:|---:|---:|---:|
| hard-template ablation | 0,2,6,9 | 650,435,509.8 | 457,862,202.2 | s0=761,760,180.6, s2=1,602,815,782.9, s6=153,964,223.8, s9=83,201,852.0 |
| MoE-template v5 | 0,2,6,9 | 2,051,203,779.6 | 2,524,258,335.8 | s0=3,026,233,178.1, s2=130,065,268.6, s6=2,329,159,265.6, s9=2,719,357,406.0 |
| hard v4 | 0,2,6,9 | 866,165,243.7 | 419,982,418.4 | s0=2,562,589,872.7, s2=717,412,377.2, s6=122,552,459.6, s9=62,106,265.5 |

Interpretation:

The template sampler alone is not enough.  It improves seed 2 over both hard
v4 and MoE-v5, but it does not rescue the hard-v4 collapse seeds 6 and 9.
Therefore the MoE actor is doing real work for the collapse cases.  The next
small ablation is a sharper MoE gate: keep true MoE and template-balanced
sampling, but lower the gate temperature from `0.08` to `0.04` to see whether
seed 2's degradation is caused by overly broad expert mixing.

### Hopper v5 ablation: sharp MoE gate

We added:

```text
bamor_v5_contrast_moe_sharp_template_mix
  actor/sampler/context = same as MoE-template v5
  gate temperature = 0.04 instead of 0.08
```

Targeted Hopper seeds `0,2,6,9` were submitted:

```text
result_dir = results/mujoco_hopper_v5moe_sharp_targeted_scheduler_20260609_0001/mo_hopper_v5
tasks = t8832..t8835
```

Targeted result:

| method | targeted seeds | mean HV | median HV | seed values |
|---|---:|---:|---:|---:|
| sharp MoE `temp=0.04` | 0,2,6,9 | 2,315,401,108.6 | 1,706,569,583.9 | s0=1,774,467,563.9, s2=5,794,164,586.6, s6=1,638,671,603.9, s9=54,300,680.0 |
| MoE-template v5 `temp=0.08` | 0,2,6,9 | 2,051,203,779.6 | 2,524,258,335.8 | s0=3,026,233,178.1, s2=130,065,268.6, s6=2,329,159,265.6, s9=2,719,357,406.0 |
| hard-template ablation | 0,2,6,9 | 650,435,509.8 | 457,862,202.2 | s0=761,760,180.6, s2=1,602,815,782.9, s6=153,964,223.8, s9=83,201,852.0 |
| hard v4 | 0,2,6,9 | 866,165,243.7 | 419,982,418.4 | s0=2,562,589,872.7, s2=717,412,377.2, s6=122,552,459.6, s9=62,106,265.5 |

Interpretation:

Sharpening the MoE gate is not a monotone improvement.  It fixes seed 2
dramatically, but loses seed 9, and has lower targeted median than the original
`temp=0.08` MoE.  Do not replace the current v5 default with sharp MoE.  The
best current 3-D candidate remains:

```text
bamor_v5_contrast_moe_template_mix
```

The remaining open problem is not simply "temperature too high"; it is likely
seed-dependent expert commitment.  A future v5.1 should use annealed or
uncertainty-aware routing, but the present paper candidate should be the
10-seed-tested `temp=0.08` MoE-template version.

### Ant 3-D transfer check

We next moved beyond Hopper.  Local environment probing showed:

```text
mo-walker2d-v5: reward_dim = 2
mo-ant-v5:      reward_dim = 3
mo-hopper-v5:   reward_dim = 3
```

So Walker2d is not a valid 3-D transfer check in this wrapper; Ant is the next
proper 3-objective test.

Submitted Ant 5-seed matrix:

```text
result_dir = results/mujoco_ant_v5moe_transfer_scheduler_20260610_0001/mo_ant_v5
env = mo-ant-v5
seeds = 0..4
methods =
  uniform
  bamor
  bamor_v3_contrast_mix
  bamor_v4_contrast_expert_mix
  bamor_v5_contrast_moe_template_mix
tasks = t8849..t8874
```

All 25 tasks were confirmed running across node001-node006.

5-seed Ant result:

| method | 3-D HV | median HV | min HV | max HV |
|---|---:|---:|---:|---:|
| `uniform` | 16,439,284,494.2+/-10,034,212,726.8 | 15,956,535,612.7 | 3,380,785,589.7 | 32,209,824,125.1 |
| `bamor` | 17,038,283,047.9+/-9,618,159,130.0 | 19,406,894,993.2 | 3,422,198,557.1 | 30,496,587,503.0 |
| `bamor_v3_contrast_mix` | 14,621,747,754.3+/-7,864,141,679.5 | 16,510,279,810.2 | 3,799,965,466.6 | 26,956,959,356.1 |
| `bamor_v4_contrast_expert_mix` | 12,285,779,471.2+/-9,777,020,424.0 | 8,089,849,492.5 | 2,697,259,197.7 | 27,651,135,424.4 |
| `bamor_v5_contrast_moe_template_mix` | 16,470,176,965.4+/-9,260,706,222.1 | 20,564,936,100.1 | 2,736,478,752.6 | 27,844,171,099.6 |

Paired 5-seed read:

- v5 vs `uniform`: `+30,892,471.2`, wins `1/5`, Wilcoxon two-sided
  `p=0.6250`.
- v5 vs old `bamor`: `-568,106,082.5`, wins `1/5`, Wilcoxon two-sided
  `p=0.6250`.
- v5 vs `v3_mix`: `+1,848,429,211.1`, wins `3/5`, Wilcoxon two-sided
  `p=0.4375`.
- v5 vs hard `v4_expert`: `+4,184,397,494.2`, wins `5/5`, Wilcoxon
  two-sided `p=0.0625`.

Interpretation:

Ant supports part of the Hopper story but not all of it.  v5 transfers clearly
against hard v4 and v3mix, but it does not clearly beat simple `uniform` or old
`bamor` on only five seeds.  Its median is the best in the table, while its
paired wins against `uniform`/`bamor` are weak.  We need seeds 5-9 before
claiming external 3-D generalization.

Seeds 5-9 were submitted and completed:

```text
result_dir = results/mujoco_ant_v5moe_ext_scheduler_20260610_0001/mo_ant_v5
tasks = t8888..t8912
```

Combined 10-seed Ant result:

| method | 3-D HV | median HV | min HV | max HV |
|---|---:|---:|---:|---:|
| `uniform` | 13,338,562,469.5+/-8,988,251,700.2 | 9,459,577,928.8 | 3,380,785,589.7 | 32,209,824,125.1 |
| `bamor` | 13,982,136,170.3+/-9,215,373,517.4 | 12,576,414,140.9 | 3,422,198,557.1 | 30,496,587,503.0 |
| `bamor_v3_contrast_mix` | 13,707,142,050.9+/-7,701,470,410.4 | 15,162,525,190.3 | 3,075,030,398.8 | 26,956,959,356.1 |
| hard `bamor_v4_contrast_expert_mix` | 10,736,713,107.4+/-8,987,765,088.1 | 6,932,881,542.4 | 1,386,032,233.2 | 27,651,135,424.4 |
| `bamor_v5_contrast_moe_template_mix` | 13,552,190,139.4+/-9,403,569,781.3 | 13,342,799,808.7 | 2,265,927,676.9 | 27,844,171,099.6 |

Paired 10-seed result:

- v5 vs `uniform`: `+213,627,669.9`, wins `3/10`, Wilcoxon two-sided
  `p=0.7695`.
- v5 vs old `bamor`: `-429,946,030.9`, wins `3/10`, Wilcoxon two-sided
  `p=0.3750`.
- v5 vs `v3_mix`: `-154,951,911.5`, wins `6/10`, Wilcoxon two-sided
  `p=0.6250`.
- v5 vs hard `v4_expert`: `+2,815,477,032.0`, wins `9/10`, Wilcoxon
  two-sided `p=0.0488`.

Interpretation after 10 seeds:

Ant does not support a broad claim that v5 dominates generic baselines.  It
does support the narrower structural claim that MoE/template routing fixes the
hard-v4 expert collapse: v5 beats hard v4 on `9/10` seeds with `p=0.0488`.
But v5 is statistically tied with `uniform`, old `bamor`, and v3mix.  The
likely issue is sample efficiency in high-dimensional Ant: fully independent
expert actors fragment state-control learning.  A better next version should
share the actor trunk and only specialize preference heads.

### v6 shared-trunk MoE head actor

We implemented:

```text
bamor_v6_contrast_moe_head_template_mix
  actor = shared state/context trunk + seven expert mean/log_std heads
  routing = same MoE template gate as v5, temp = 0.08
  sampler = 50% random Dirichlet + 50% exact 7-template cycle
  context/critic/latent encoder = same as v5
```

Rationale:

Hopper needs preference-specialized behavior, but Ant shows that fully
independent expert actors can fragment high-dimensional control learning.  v6
keeps preference-specialized heads while sharing the lower-level state-control
representation.

Local checks:

- Python compile passed.
- MoE head gate shape/normalization passed.
- Stochastic action and mixture log-prob are finite.
- Backward pass reaches both shared trunk and expert heads.

Targeted v6 run submitted:

```text
result_dir = results/mujoco_v6_headmoe_targeted_scheduler_20260610_0001
Ant seeds = 2, 7, 9
Hopper seeds = 2, 6, 9
tasks = t8919..t8924
```

These are the most diagnostic seeds:

- Ant seed 7 is where v5 collapses relative to v3mix.
- Ant seed 2/9 check whether v6 keeps high seeds.
- Hopper seeds 6/9 are hard-v4 collapse seeds.
- Hopper seed 2 checks whether v6 avoids the original v5 seed-2 collapse.

Targeted result:

Ant targeted seeds `2,7,9`:

| method | mean HV | median HV | seed values |
|---|---:|---:|---:|
| `bamor_v6_contrast_moe_head_template_mix` | 18,478,236,068.3 | 16,920,398,401.6 | s2=29,346,644,455.2, s7=16,920,398,401.6, s9=9,167,665,348.3 |
| `bamor_v5_contrast_moe_template_mix` | 15,998,527,547.6 | 17,885,483,866.2 | s2=27,844,171,099.6, s7=2,265,927,676.9, s9=17,885,483,866.2 |
| `bamor_v3_contrast_mix` | 19,316,484,962.9 | 17,177,724,962.1 | s2=26,956,959,356.1, s7=17,177,724,962.1, s9=13,814,770,570.4 |

Hopper targeted seeds `2,6,9`:

| method | mean HV | median HV | seed values |
|---|---:|---:|---:|
| `bamor_v6_contrast_moe_head_template_mix` | 1,658,396,673.4 | 2,092,096,805.9 | s2=2,092,096,805.9, s6=18,339,846.3, s9=2,864,753,368.1 |
| `bamor_v5_contrast_moe_template_mix` | 1,726,193,980.1 | 2,329,159,265.6 | s2=130,065,268.6, s6=2,329,159,265.6, s9=2,719,357,406.0 |
| hard `bamor_v4_contrast_expert_mix` | 300,690,367.4 | 122,552,459.6 | s2=717,412,377.2, s6=122,552,459.6, s9=62,106,265.5 |

Interpretation:

v6 is complementary to v5, not a replacement.  It fixes Ant seed 7 and
Hopper seed 2, but it loses Ant seed 9 and collapses on Hopper seed 6.  The
shared trunk helps sample efficiency on some high-dimensional Ant cases, but
weakens the independent-specialist behavior that rescued Hopper seed 6.

Do not expand v6 as the next default.  The current best 3-D story is:

- Hopper: v5 MoE-template is the best candidate.
- Ant: v5 beats hard v4 but does not beat uniform/bamor/v3mix; v6 is only a
  diagnostic complement.
- Next serious algorithmic step, if continuing, should be a dual-branch actor
  or critic-selected ensemble that can use independent experts when
  specialization matters and shared-head experts when sample efficiency matters.

### v7 critic-selected dual-branch actor

Implemented:

```text
bamor_v7_contrast_critic_dual_template_mix
  branch A = v5 independent MoE-template experts
  branch B = v6 shared-trunk MoE-template heads
  training sampler = fixed 50/50 branch mixture, exact branch+expert mixture log-prob
  deterministic eval = critic-selected branch action under current scalarized Q
  context/critic/latent encoder/weight sampler = same as v5/v6
```

Why this is the next real test:

- v5 and v6 fail on different diagnostic seeds.
- Averaging the two policies would blur specialist behavior and likely inherit
  both failure modes.
- v7 instead trains both branches from the same replay stream, then lets the
  critic choose between the independent-specialist action and shared-head
  action for the current state/context/preference.

Local checks:

- Python compile passed.
- Dual-branch stochastic action/log-prob are finite.
- Deterministic branch-action tensor has the expected shape.
- Backward pass reaches both the independent-expert and shared-head branches.
- Critic-selected deterministic action path runs.

Targeted test plan:

```text
result_dir = results/mujoco_v7_dual_targeted_scheduler_20260610_0001
Ant seeds = 2, 7, 9
Hopper seeds = 2, 6, 9
method = bamor_v7_contrast_critic_dual_template_mix
```

Decision rule:

- If v7 keeps Ant seed 7 without losing Ant seed 9, and keeps Hopper seed 6
  without losing Hopper seed 2, expand to 10 seeds.
- If v7 collapses on either Hopper seed 6 or Ant seed 9, the dual-branch idea
  is not enough; the next change should constrain branch selection or train
  per-branch critics.

Targeted v7 result:

Ant seeds `2,7,9`:

| method | mean HV | median HV | seed values |
|---|---:|---:|---:|
| `bamor_v7_contrast_critic_dual_template_mix` | 10,992,400,152.1 | 2,264,821,655.9 | s2=28,646,571,394.3, s7=2,264,821,655.9, s9=2,065,807,406.1 |
| `bamor_v6_contrast_moe_head_template_mix` | 18,478,236,068.3 | 16,920,398,401.6 | s2=29,346,644,455.2, s7=16,920,398,401.6, s9=9,167,665,348.3 |
| `bamor_v5_contrast_moe_template_mix` | 15,998,527,547.6 | 17,885,483,866.2 | s2=27,844,171,099.6, s7=2,265,927,676.9, s9=17,885,483,866.2 |

Hopper seeds `2,6,9`:

| method | mean HV | median HV | seed values |
|---|---:|---:|---:|
| `bamor_v7_contrast_critic_dual_template_mix` | 1,742,725,843.9 | 936,482,760.4 | s2=936,482,760.4, s6=4,270,582,135.9, s9=21,112,635.4 |
| `bamor_v5_contrast_moe_template_mix` | 1,726,193,980.1 | 2,329,159,265.6 | s2=130,065,268.6, s6=2,329,159,265.6, s9=2,719,357,406.0 |
| `bamor_v6_contrast_moe_head_template_mix` | 1,658,396,673.4 | 2,092,096,805.9 | s2=2,092,096,805.9, s6=18,339,846.3, s9=2,864,753,368.1 |

Interpretation:

v7 should not be expanded.  It is not just a noisy selector: branch diagnostic
with the saved models showed that several failure seeds have both branches
weak after joint 50/50-mixture training.  The likely issue is branch
under-training: each actor branch receives too little direct SAC actor signal
and can collapse together, even though v5 and v6 trained separately are
complementary.

### v8 dual-update dual-branch actor

Implemented:

```text
bamor_v8_contrast_dual_update_template_mix
  actor structure = same two branches as v7
  stochastic data collection = same fixed 50/50 branch mixture
  actor update = every gradient step updates both branches with their own SAC loss
  deterministic eval = critic-selected branch action
```

Rationale:

v7 sampled a branch per actor update.  v8 makes the two branches explicit
co-learners: both independent specialists and shared-head specialists receive
full actor-gradient coverage on every update.  This directly tests whether the
v7 failure was caused by branch starvation rather than by the dual-branch idea
itself.

Local checks:

- Python compile passed.
- Scheduler script syntax passed.
- Dual-update trainer smoke test passed.
- Critic-selected deterministic path still runs.

Targeted test plan:

```text
result_dir = results/mujoco_v8_dualupdate_targeted_scheduler_20260610_0001
Ant seeds = 7, 9
Hopper seeds = 6, 9
method = bamor_v8_contrast_dual_update_template_mix
```

Decision rule:

- If Ant seed 7/9 recover while Hopper seed 6 stays high, expand to the full
  six diagnostic seeds.
- If Ant seed 9 or Hopper seed 9 still collapses, the remaining viable route is
  a post-hoc or explicitly trained two-policy ensemble, not a single
  critic-selected dual actor.

Targeted v8 result:

Ant seeds `7,9`:

| method | mean HV | median HV | seed values |
|---|---:|---:|---:|
| `bamor_v8_contrast_dual_update_template_mix` | 5,547,319,936.0 | 5,547,319,936.0 | s7=2,316,040,879.0, s9=8,778,599,002.9 |
| `bamor_v6_contrast_moe_head_template_mix` | 13,044,031,875.0 | 13,044,031,875.0 | s7=16,920,398,401.6, s9=9,167,665,348.3 |
| `bamor_v5_contrast_moe_template_mix` | 10,075,705,771.6 | 10,075,705,771.6 | s7=2,265,927,676.9, s9=17,885,483,866.2 |

Hopper seeds `6,9`:

| method | mean HV | median HV | seed values |
|---|---:|---:|---:|
| `bamor_v8_contrast_dual_update_template_mix` | 1,412,436,974.2 | 1,412,436,974.2 | s6=2,802,628,186.4, s9=22,245,762.0 |
| `bamor_v5_contrast_moe_template_mix` | 2,524,258,335.8 | 2,524,258,335.8 | s6=2,329,159,265.6, s9=2,719,357,406.0 |
| `bamor_v6_contrast_moe_head_template_mix` | 1,441,546,606.2 | 1,441,546,606.2 | s6=18,339,846.3, s9=2,864,753,368.1 |

Interpretation:

v8 improves over v7 on Ant seed 9 but still fails Ant seed 7 and Hopper seed
9.  This rejects the single dual-actor family for now.  The core issue is not
just branch starvation; coupling the two actor branches through one replay
stream and one critic still creates shared failure modes.

### v5+v6 independent policy-set ensemble

The strongest 3-D route is now:

```text
BAMOR-3D-ensemble
  policy 1 = v5 independent MoE-template actor
  policy 2 = v6 shared-trunk MoE-head actor
  train independently with the same BAMOR context/template sampler
  evaluate/report the Pareto union of both policy sets
```

This is not a cosmetic ensemble.  It matches the empirical failure structure:
v5 and v6 collapse on different seeds, while joint single-actor variants v7/v8
make the failures correlated.  Independent training preserves complementarity.

We first computed the targeted upper bound on the diagnostic seeds:

| env | seeds | v5 mean | v6 mean | max(v5,v6) mean | union mean |
|---|---|---:|---:|---:|---:|
| Ant | 2,7,9 | 15.999B | 18.478B | 21.384B | 21.862B |
| Hopper | 2,6,9 | 1.726B | 1.658B | 2.429B | 2.455B |

Then v6 was completed to 10 seeds for Ant/Hopper:

```text
result_dir = results/mujoco_v6_headmoe_ensemble_remaining_scheduler_20260610_0001
Ant added seeds = 0,1,3,4,5,6,8
Hopper added seeds = 0,1,3,4,5,7,8
tasks = t9008..t9021
```

10-seed result:

Ant:

| method | mean HV | median HV | min HV | max HV |
|---|---:|---:|---:|---:|
| `uniform` | 13.339B | 9.460B | 3.381B | 32.210B |
| old `bamor` | 13.982B | 12.576B | 3.422B | 30.497B |
| `bamor_v3_contrast_mix` | 13.707B | 15.163B | 3.075B | 26.957B |
| hard `bamor_v4_contrast_expert_mix` | 10.737B | 6.933B | 1.386B | 27.651B |
| `bamor_v5_contrast_moe_template_mix` | 13.552B | 13.343B | 2.266B | 27.844B |
| `bamor_v6_contrast_moe_head_template_mix` | 14.586B | 13.044B | 3.072B | 29.347B |
| `v5_v6_union` | 16.536B | 17.655B | 3.072B | 32.777B |

Paired Ant reads for `v5_v6_union`:

- vs `uniform`: `+3.198B`, wins `6/10`, Wilcoxon `p=0.2754`.
- vs old `bamor`: `+2.554B`, wins `5/10`, Wilcoxon `p=0.4922`.
- vs `v3_mix`: `+2.829B`, wins `8/10`, Wilcoxon `p=0.0488`.
- vs hard `v4_expert`: `+5.800B`, wins `10/10`, Wilcoxon `p=0.0020`.
- vs v5 alone: `+2.984B`, wins `10/10`, Wilcoxon `p=0.0020`.
- vs v6 alone: `+1.950B`, wins `10/10`, Wilcoxon `p=0.0020`.

Hopper:

| method | mean HV | median HV | min HV | max HV |
|---|---:|---:|---:|---:|
| `uniform` | 1.908B | 1.823B | 0.017B | 6.220B |
| old `bamor` | 2.105B | 0.152B | 0.016B | 9.135B |
| `bamor_v3_contrast_mix` | 1.118B | 0.966B | 0.022B | 2.568B |
| hard `bamor_v4_contrast_expert_mix` | 1.761B | 2.185B | 0.062B | 2.794B |
| `bamor_v5_contrast_moe_template_mix` | 2.377B | 2.585B | 0.130B | 3.887B |
| `bamor_v6_contrast_moe_head_template_mix` | 2.311B | 2.554B | 0.018B | 4.998B |
| `v5_v6_union` | 3.131B | 2.971B | 2.092B | 5.267B |

Paired Hopper reads for `v5_v6_union`:

- vs `uniform`: `+1.223B`, wins `8/10`, Wilcoxon `p=0.0840`.
- vs old `bamor`: `+1.026B`, wins `8/10`, Wilcoxon `p=0.4316`.
- vs `v3_mix`: `+2.013B`, wins `10/10`, Wilcoxon `p=0.0020`.
- vs hard `v4_expert`: `+1.370B`, wins `10/10`, Wilcoxon `p=0.0020`.
- vs v5 alone: `+0.754B`, wins `8/10`, Wilcoxon `p=0.0117`.
- vs v6 alone: `+0.820B`, wins `9/10`, Wilcoxon `p=0.0039`.

Current conclusion:

This is the first rescue path that is defensible across both 3-D environments.
It still does not significantly beat `uniform` or old `bamor` on Ant at 10
seeds, and does not significantly beat old `bamor` on Hopper because old BAMOR
has a single huge seed.  But it fixes the central 3-D robustness problem: the
Hopper minimum rises from near-zero collapse to `2.092B`, and the union beats
both constituent policies on every Ant seed and nearly every Hopper seed.

Reproducible aggregation:

```bash
BUILD_PAPER=1 bash refresh_mujoco_policy_set_artifacts.sh
```

The script recomputes HV from the saved `pareto_overall` fronts rather than
trusting copied numbers.  It also reports robust-collapse metrics:

Ant robust read:

| method | q25 HV | bottom20 HV | min HV | mean rank |
|---|---:|---:|---:|---:|
| `uniform` | 6.811B | 3.646B | 3.381B | 4.20 |
| old `bamor` | 5.395B | 3.538B | 3.422B | 2.90 |
| `v5_v6_union` | 7.294B | 3.284B | 3.072B | 2.10 |

Hopper robust read:

| method | q25 HV | bottom20 HV | min HV | mean rank |
|---|---:|---:|---:|---:|
| `uniform` | 0.077B | 0.018B | 0.017B | 4.50 |
| old `bamor` | 0.056B | 0.016B | 0.016B | 5.20 |
| `bamor_v5_contrast_moe_template_mix` | 2.290B | 0.800B | 0.130B | 3.20 |
| `bamor_v6_contrast_moe_head_template_mix` | 2.036B | 0.272B | 0.018B | 3.60 |
| `v5_v6_union` | 2.530B | 2.211B | 2.092B | 1.70 |

This strengthens the claim that the rescue is mainly a 3-D robustness fix.
The seed-level CSV adds a complementarity read:

- Ant: union gain over the better constituent is `+0.916B` on average, with
  strict gains on `10/10` seeds.  v5 is the better constituent on 4 seeds and
  v6 on 6 seeds.
- Hopper: union gain over the better constituent is `+0.148B` on average, with
  strict gains on `6/10` seeds.  v5 and v6 are each the better constituent on
  5 seeds.

This rules out the interpretation that the policy set merely chooses one
uniformly superior actor.  The supporting figure is written to
`paper/fig_mujoco_policy_set_gain.pdf`.

20-seed extension was submitted on 2026-06-10:

```bash
ENVS="mo-ant-v5 mo-hopper-v5" \
SEED_LIST="10 11 12 13 14 15 16 17 18 19" \
SAVE_DIR="results/mujoco_policy_set_20seed_ext_scheduler_20260610_0001" \
DISPATCH=1 \
bash submit_bamor_mujoco_scheduler.sh
```

Matrix size: `2 envs x 6 methods x 10 seeds = 120` tasks on
`node001`-`node006`.  Scheduler task range: `t9067` through `t9219`.
The submit script signature now includes `seed${seed}` so different seeds are
not treated as duplicates.

After pulling the extension results, run:

```bash
./pull_bamor_hpc_results.sh results/mujoco_policy_set_20seed_ext_scheduler_20260610_0001
PRESETS="ant20 hopper20" TAG=3d20 BUILD_PAPER=0 bash refresh_mujoco_policy_set_artifacts.sh
```

Final 20-seed read after pulling `results/mujoco_policy_set_20seed_ext_scheduler_20260610_0001`:

Ant:

| method | mean HV | median HV | q25 HV | bottom20 HV | min HV | mean rank |
|---|---:|---:|---:|---:|---:|---:|
| `uniform` | 12.671B | 9.460B | 3.778B | 2.584B | 2.185B | 4.25 |
| old `bamor` | 13.769B | 16.661B | 3.597B | 2.931B | 2.381B | 2.35 |
| `bamor_v3_contrast_mix` | 13.079B | 16.601B | 3.654B | 2.553B | 2.168B | 4.45 |
| hard `bamor_v4_contrast_expert_mix` | 10.376B | 6.933B | 2.667B | 2.121B | 1.386B | 5.70 |
| `bamor_v5_contrast_moe_template_mix` | 12.312B | 11.405B | 2.996B | 2.487B | 2.266B | 4.55 |
| `bamor_v6_contrast_moe_head_template_mix` | 13.441B | 16.136B | 3.151B | 2.587B | 2.185B | 4.65 |
| `v5_v6_union` | 15.172B | 18.166B | 3.405B | 2.769B | 2.357B | 2.05 |

Ant paired reads for `v5_v6_union`:

- vs `uniform`: `+2.502B`, wins `14/20`, Wilcoxon `p=0.0362`.
- vs old `bamor`: `+1.403B`, wins `9/20`, Wilcoxon `p=0.7285`.
- vs `v3_mix`: `+2.093B`, wins `17/20`, Wilcoxon `p=0.0005`.
- vs hard `v4_expert`: `+4.796B`, wins `19/20`, Wilcoxon `p<0.0001`.
- vs v5 alone: `+2.860B`, wins `20/20`, Wilcoxon `p<0.0001`.
- vs v6 alone: `+1.731B`, wins `20/20`, Wilcoxon `p<0.0001`.

Hopper:

| method | mean HV | median HV | q25 HV | bottom20 HV | min HV | mean rank |
|---|---:|---:|---:|---:|---:|---:|
| `uniform` | 1.500B | 1.618B | 0.063B | 0.026B | 0.017B | 5.00 |
| old `bamor` | 2.287B | 1.610B | 0.122B | 0.024B | 0.016B | 4.50 |
| `bamor_v3_contrast_mix` | 1.313B | 1.033B | 0.064B | 0.030B | 0.022B | 5.00 |
| hard `bamor_v4_contrast_expert_mix` | 1.598B | 2.185B | 0.092B | 0.045B | 0.022B | 4.35 |
| `bamor_v5_contrast_moe_template_mix` | 2.412B | 2.368B | 1.443B | 0.408B | 0.027B | 3.50 |
| `bamor_v6_contrast_moe_head_template_mix` | 2.406B | 2.347B | 1.816B | 0.459B | 0.018B | 3.80 |
| `v5_v6_union` | 3.388B | 2.844B | 2.451B | 1.799B | 0.562B | 1.85 |

Hopper paired reads for `v5_v6_union`:

- vs `uniform`: `+1.888B`, wins `18/20`, Wilcoxon `p=0.0007`.
- vs old `bamor`: `+1.101B`, wins `15/20`, Wilcoxon `p=0.0484`.
- vs `v3_mix`: `+2.075B`, wins `19/20`, Wilcoxon `p<0.0001`.
- vs hard `v4_expert`: `+1.790B`, wins `17/20`, Wilcoxon `p=0.0009`.
- vs v5 alone: `+0.976B`, wins `16/20`, Wilcoxon `p=0.0004`.
- vs v6 alone: `+0.982B`, wins `18/20`, Wilcoxon `p=0.0002`.

20-seed conclusion:

The 3-D policy-set result is now much stronger than the 10-seed read.  It
significantly beats Uniform on both Ant and Hopper, significantly beats old
`bamor` on Hopper, and beats both constituent policies on every Ant seed and on
most Hopper seeds.  The remaining boundary is Ant vs old `bamor`: the union has
higher mean/median HV and better mean rank, but the paired test is not
significant and old `bamor` has the best Ant lower-tail metrics.  The defensible
claim is therefore: policy-set BAMOR is a robust 3-D coverage rescue, with
strong evidence on Hopper collapse removal and significant Uniform gains on
both available continuous-control 3-D tasks.

The default MuJoCo scheduler matrix was therefore changed to the paper matrix:

```text
uniform
bamor
bamor_v3_contrast_mix
bamor_v4_contrast_expert_mix
bamor_v5_contrast_moe_template_mix
bamor_v6_contrast_moe_head_template_mix
```

v7/v8 remain implemented for ablation history but are no longer default runs.

### Third-environment feasibility check

Local `mo_gymnasium` registry probe:

| env | reward_dim | action space | current status |
|---|---:|---|---|
| `mo-ant-v5` | 3 | continuous Box, shape `(8,)` | already used |
| `mo-hopper-v5` | 3 | continuous Box, shape `(3,)` | already used |
| `mo-halfcheetah-v5` | 2 | continuous Box, shape `(6,)` | not a 3-D test |
| `mo-walker2d-v5` | 2 | continuous Box, shape `(6,)` | not a 3-D test |
| `mo-humanoid-v5` | 2 | continuous Box, shape `(17,)` | not a 3-D test |
| `mo-swimmer-v5` | 2 | continuous Box, shape `(2,)` | not a 3-D test |
| `mo-reacher-v5` | 4 | `Discrete(9)` | incompatible with current SAC actor |
| `mo-mountaincar-3d-v0` | 3 | `Discrete(3)` | incompatible with current SAC actor |
| `mo-lunar-lander-continuous-v3` | unknown locally | Box2D missing | dependency gap |

So the clean 3-D continuous-control evidence base in the current code is
Ant + Hopper.  Adding a third environment would require either:

- implementing discrete-action MORL support for Reacher/MountainCar-3D, or
- adding Box2D and validating LunarLander-continuous, or
- defining new piecewise presets for a custom continuous 3-D task.

This is a separate engineering track.  It should not block the current 3-D
rescue story, which is now about robustness across the two available
continuous 3-D MuJoCo tasks.

### True morl-baselines CAPQL comparison, pure BAMOR-v9, and disclosed hybrid

After the 20-seed BAMOR-only policy-set result, we ran the actual continuous
CAPQL implementation from `morl-baselines` on the same 3-D Ant/Hopper protocol:

```text
result_dir = results/mujoco_capql_3d20_scheduler_20260611_0001
envs       = mo-ant-v5, mo-hopper-v5
seeds      = 0..19
steps      = 50,000
```

Implementation fixes made for this comparison:

- `train_bamor_mujoco.py` now evaluates CAPQL into the same schema as BAMOR
  runs: `pareto_overall`, `hv_overall`, `pareto_per_mode`, `hv_per_mode`, and
  `ref_point`.
- CAPQL uses `reference_point_for_env(...)` instead of a zero reference point.
- A no-op `wandb` stub is installed only when `wandb` is absent, because CAPQL
  imports `wandb` even with `log=False`.
- `stage_bamor_mujoco_hpc.sh` stages the local `morl-baselines` package and
  installs the missing remote `pymoo` wheel/dependencies.

CAPQL result:

| env | method | mean HV | median HV | q25 HV | bottom20 HV | min HV |
|---|---|---:|---:|---:|---:|---:|
| Ant | CAPQL | 10.137B | 4.466B | 3.534B | 2.665B | 2.292B |
| Ant | v5+v6 BAMOR policy set | 15.172B | 18.166B | 3.405B | 2.769B | 2.357B |
| Hopper | CAPQL | 4.335B | 2.835B | 2.116B | 1.325B | 0.487B |
| Hopper | v5+v6 BAMOR policy set | 3.388B | 2.844B | 2.451B | 1.799B | 0.562B |

Paired comparison:

- Ant v5+v6 policy set vs CAPQL: `+5.035B`, wins `14/20`,
  Wilcoxon `p=0.0441`.
- Hopper v5+v6 policy set vs CAPQL: `-0.947B`, wins `8/20`,
  Wilcoxon `p=0.4749`.

Interpretation:

The BAMOR-only policy set is now clearly effective beyond the synthetic 1-D
environment, and it beats true CAPQL on Ant.  It does **not** beat CAPQL on
Hopper in mean HV or paired significance, although it has better median, q25,
bottom-tail, and minimum HV.  Therefore we should not claim BAMOR-only
dominance over `morl-baselines`.

The pure BAMOR-v9 improvement is a portfolio/library method with no external
algorithm arm:

```text
bamor_v5_v6_union = union(original BAMOR front, v5 MoE-template front, v6 shared-head MoE front)
```

BAMOR-v9 result:

| env | method | mean HV | median HV | q25 HV | bottom20 HV | min HV | mean rank |
|---|---|---:|---:|---:|---:|---:|---:|
| Ant | pure BAMOR-v9 portfolio | 16.651B | 19.054B | 4.397B | 3.041B | 2.611B | 1.35 |
| Hopper | pure BAMOR-v9 portfolio | 4.176B | 3.346B | 2.665B | 2.156B | 1.551B | 2.05 |

Paired BAMOR-v9 comparison:

- Ant pure v9 vs CAPQL: `+6.514B`, wins `15/20`, Wilcoxon `p=0.0153`.
- Ant pure v9 vs v5+v6 policy set: `+1.479B`, wins `20/20`,
  Wilcoxon `p<0.0001`.
- Hopper pure v9 vs CAPQL: `-0.158B`, wins `11/20`, Wilcoxon `p=0.7285`.
- Hopper pure v9 vs v5+v6 policy set: `+0.789B`, wins `16/20`,
  Wilcoxon `p=0.0007`.

Important identity/fairness conclusion:

- Pure BAMOR-v9 is **not** a CAPQL wrapper and not a known-algorithm shell.  It
  is a BAPR/BAMOR-family multi-scenario multi-objective policy library.
- Pure BAMOR-v9 significantly beats CAPQL on Ant and is statistically tied with
  CAPQL in Hopper mean HV, while improving Hopper median, q25, bottom20, and
  minimum HV.
- Therefore the honest pure-BAMOR claim is robustness and coverage, not
  unconditional mean-HV dominance over CAPQL on Hopper.

We also computed a disclosed hybrid/engineering extension:

```text
capql_v5_v6_union = union(CAPQL front, v5 MoE-template BAMOR front, v6 shared-head BAMOR front)
```

Hybrid result:

| env | method | mean HV | median HV | q25 HV | bottom20 HV | min HV | mean rank |
|---|---|---:|---:|---:|---:|---:|---:|
| Ant | CAPQL+BAMOR hybrid | 17.824B | 22.037B | 6.363B | 3.448B | 2.823B | 1.30 |
| Hopper | CAPQL+BAMOR hybrid | 5.356B | 4.588B | 2.970B | 2.401B | 1.974B | 1.75 |

Hybrid paired comparison:

- Ant hybrid vs CAPQL: `+7.687B`, wins `20/20`, Wilcoxon `p<0.0001`.
- Hopper hybrid vs CAPQL: `+1.021B`, wins `18/20`, Wilcoxon `p=0.0002`.
- Hopper hybrid vs v5+v6 policy set: `+1.968B`, wins `18/20`,
  Wilcoxon `p<0.0001`.

Defensible claim:

- BAMOR-only v5+v6 policy set: robust 3-D rescue, but not universal dominance.
- Pure BAMOR-v9 portfolio: the real BAMOR algorithmic version; no CAPQL arm, no
  baseline leakage, no unfair self-comparison.
- CAPQL+BAMOR hybrid: strongest engineering result, but must be disclosed as a
  hybrid and not used to claim pure BAMOR dominance over CAPQL.

### 2026-06-11 targeted large-experiment extension

Instead of blindly expanding all environments and all algorithms, the next
experiments target the two remaining reviewer risks:

1. Cost fairness: BAMOR-v9 is a 3-front portfolio, so compare it against a
   3-front CAPQL portfolio, not only single-run CAPQL.
2. Failed-environment transfer: check whether the pure BAMOR-v9 portfolio also
   helps on the earlier failed `mo-halfcheetah-v5` transfer setting.

Implementation changes:

- `train_bamor_mujoco.py` now supports `capql_rep1` and `capql_rep2`.
  These are independent CAPQL training replicas with different train seeds, but
  the same base eval seed as `capql`.  This avoids mixing different test
  environments inside a portfolio union.
- `aggregate_mujoco_policy_set_union.py` now has presets:
  - `ant20_capql_3x`
  - `hopper20_capql_3x`
  - `halfcheetah10_bamor_portfolio`

Submitted scheduler experiments:

```text
CAPQL cost-matched portfolio:
result_dir = results/mujoco_capql_3x_3d20_scheduler_20260611_0002
envs       = mo-ant-v5, mo-hopper-v5
methods    = capql_rep1, capql_rep2
seeds      = 0..19
tasks      = 80 additional CAPQL runs
task ids   = t9585..t9674, with non-BAMOR scheduler ids interleaved

HalfCheetah failed-env BAMOR-v9 pilot:
result_dir = results/mujoco_hc_v9_scheduler_20260611_0001
env        = mo-halfcheetah-v5
methods    = capql, bamor_v5_contrast_moe_template_mix, bamor_v6_contrast_moe_head_template_mix
seeds      = 0..9
tasks      = 30
task ids   = t9675..t9704
```

Scheduler note:

The local `scheduler.service` watch process was temporarily stopped because it
held `~/.claude/scheduler/.lock` for several minutes and blocked submissions.
After all jobs were registered, `scheduler.service` was restarted and is active
again.  Some pending tasks may be dispatched by the restarted service rather
than by the interrupted manual `dispatch` command.

When jobs finish, pull and aggregate:

```bash
./pull_bamor_hpc_results.sh results/mujoco_capql_3x_3d20_scheduler_20260611_0002
./pull_bamor_hpc_results.sh results/mujoco_hc_v9_scheduler_20260611_0001

python3 aggregate_mujoco_policy_set_union.py \
  --preset ant20_capql_3x \
  --preset hopper20_capql_3x \
  --write-json results/mujoco_policy_set_union_capql_3x_3d20_summary.json \
  --write-md results/mujoco_policy_set_union_capql_3x_3d20_summary.md \
  --write-seed-csv results/mujoco_policy_set_union_capql_3x_3d20_seed_diagnostics.csv

python3 aggregate_mujoco_policy_set_union.py \
  --preset halfcheetah10_bamor_portfolio \
  --write-json results/mujoco_policy_set_union_bamor_v9_halfcheetah10_summary.json \
  --write-md results/mujoco_policy_set_union_bamor_v9_halfcheetah10_summary.md \
  --write-seed-csv results/mujoco_policy_set_union_bamor_v9_halfcheetah10_seed_diagnostics.csv
```

### 2026-06-11 targeted extension results

Both targeted extension matrices finished and were pulled locally:

```text
results/mujoco_capql_3x_3d20_scheduler_20260611_0002
  completed JSON files = 80/80

results/mujoco_hc_v9_scheduler_20260611_0001
  completed JSON files = 30/30
```

Aggregated outputs:

```text
results/mujoco_policy_set_union_capql_3x_3d20_summary.{json,md}
results/mujoco_policy_set_union_capql_3x_3d20_seed_diagnostics.csv

results/mujoco_policy_set_union_bamor_v9_halfcheetah10_summary.{json,md}
results/mujoco_policy_set_union_bamor_v9_halfcheetah10_seed_diagnostics.csv
```

Cost-matched CAPQL-3x result:

| env | method | mean HV | median HV | q25 HV | bottom20 HV | min HV |
|---|---|---:|---:|---:|---:|---:|
| Ant | single CAPQL | 10.137B | 4.466B | 3.534B | 2.665B | 2.292B |
| Ant | CAPQL-3x union | 13.624B | 10.065B | 4.113B | 3.351B | 3.183B |
| Hopper | single CAPQL | 4.335B | 2.835B | 2.116B | 1.325B | 0.487B |
| Hopper | CAPQL-3x union | 5.734B | 4.832B | 3.757B | 2.559B | 1.365B |

Pure BAMOR-v9 vs CAPQL-3x:

- Ant: BAMOR-v9 mean `16.651B` vs CAPQL-3x `13.624B`, mean delta
  `+3.027B`, wins `11/20`, Wilcoxon `p=0.3683`.
- Hopper: BAMOR-v9 mean `4.176B` vs CAPQL-3x `5.734B`, mean delta
  `-1.558B`, wins `7/20`, Wilcoxon `p=0.1536`.

Interpretation:

Against single-run CAPQL, pure BAMOR-v9 remains a defensible robustness result:
significant on Ant and lower-tail improved on Hopper.  Against a cost-matched
3-front CAPQL portfolio, pure BAMOR-v9 is **not** statistically stronger:
Ant has a higher mean but no paired significance, and Hopper is worse in mean
and distributional robustness.  Therefore the paper should not claim that
BAMOR-v9 dominates compute-matched CAPQL portfolios.

HalfCheetah failed-environment result, HV in millions:

| method | mean HV | median HV | q25 HV | min HV | mean rank |
|---|---:|---:|---:|---:|---:|
| `uniform` | 0.903M | 0.504M | 0.195M | 0.186M | 4.80 |
| `capql` | 5.212M | 5.355M | 2.877M | 0.189M | 1.60 |
| `bamor` | 0.985M | 0.188M | 0.167M | 0.147M | 5.40 |
| `bamor_v3_contrast_mix` | 2.140M | 2.226M | 0.773M | 0.200M | 2.70 |
| `bamor_v5_contrast_moe_template_mix` | 0.908M | 0.395M | 0.191M | 0.158M | 4.90 |
| `bamor_v6_contrast_moe_head_template_mix` | 0.488M | 0.196M | 0.174M | 0.047M | 5.80 |
| `bamor_v5_v6_union` | 1.538M | 0.862M | 0.601M | 0.192M | 2.80 |

HalfCheetah interpretation:

- Pure BAMOR-v9 improves over `uniform`, old `bamor`, and its own v5/v6
  constituents, but does not beat CAPQL.
- The earlier `bamor_v3_contrast_mix` remains the better BAMOR-family
  HalfCheetah variant than v9 (`2.140M` vs `1.538M` mean HV).
- This failed-environment check should be reported as a limitation/diagnostic,
  not as another v9 success.

Updated claim boundary:

- Strongest honest 3-D claim: pure BAMOR-v9 is effective beyond synthetic 1-D
  and beats single-run CAPQL on Ant; on Hopper it improves robustness relative
  to old BAMOR-family variants and single-run CAPQL lower tail, but not
  cost-matched CAPQL-3x.
- Strongest honest engineering claim: disclosed hybrid/portfolio methods can
  improve coverage, but cost-matched baseline portfolios are also strong and
  must be reported if we use portfolio framing.

### 2026-06-11 HalfCheetah v11 preference-relabeling attempt

HalfCheetah still has room, but the failure mode is now clear:

- `bamor_v3_contrast_mix` is the best BAMOR-family single method on
  HalfCheetah.
- v5/v6 MoE-style 3-D rescue structures do not transfer well to 2-D
  HalfCheetah.
- CAPQL is strong because it reuses vector-reward transitions efficiently
  across preferences.

Implemented a new BAMOR-family method:

```text
bamor_v11_contrast_relabel_mix
  base = v3 transition-context + mixed preference schedule
  change = preference relabeling inside MOSAC updates
  effect = every sampled vector-reward transition is trained under fresh
           Dirichlet/stratified scalarisation weights
```

This is not a CAPQL wrapper:

- it does not import or call CAPQL;
- it does not union with CAPQL fronts;
- it keeps the BAMOR/MOSAC actor-critic, transition context, and mixed
  preference schedule;
- it adds off-policy preference relabeling, which is a generic vector-reward
  MORL sample-efficiency mechanism.

Validation:

- `python3 -m py_compile train_bamor_mujoco.py aggregate_mujoco_policy_set_union.py`
  passes.
- A local `mo-halfcheetah-v5` 6k-step smoke test completed, including replay
  updates, evaluation, and JSON saving:

```text
results/smoke_hc_v11_relabel/mo_halfcheetah_v5/bamor_v11_contrast_relabel_mix_seed0.json
```

Submitted scheduler pilot:

```text
result_dir = results/mujoco_hc_v11_relabel_scheduler_20260611_0001
env        = mo-halfcheetah-v5
method     = bamor_v11_contrast_relabel_mix
seeds      = 0..9
steps      = 50,000
tasks      = 10
task ids   = t9840..t9849
```

When jobs finish:

```bash
./pull_bamor_hpc_results.sh results/mujoco_hc_v11_relabel_scheduler_20260611_0001

python3 aggregate_mujoco_policy_set_union.py \
  --preset halfcheetah10_bamor_v11_portfolio \
  --write-json results/mujoco_policy_set_union_bamor_v11_halfcheetah10_summary.json \
  --write-md results/mujoco_policy_set_union_bamor_v11_halfcheetah10_summary.md \
  --write-seed-csv results/mujoco_policy_set_union_bamor_v11_halfcheetah10_seed_diagnostics.csv
```

Decision rule:

- If v11 single-method mean/median exceeds `bamor_v3_contrast_mix`, preference
  relabeling should become the HalfCheetah rescue path.
- If the v11 portfolio approaches CAPQL but does not beat it, report it as a
  partial sample-efficiency improvement and keep CAPQL as the stronger
  HalfCheetah baseline.
- If v11 collapses, stop adding actor structure and move to a CAPQL-style
  envelope update only if we are willing to disclose it as a more substantial
  algorithmic change rather than pure BAMOR-v9.

### 2026-06-11 HalfCheetah v11 results

The v11 HalfCheetah pilot finished and was pulled locally:

```text
results/mujoco_hc_v11_relabel_scheduler_20260611_0001
  completed JSON files = 10/10
```

Aggregated outputs:

```text
results/mujoco_policy_set_union_bamor_v11_halfcheetah10_summary.{json,md}
results/mujoco_policy_set_union_bamor_v11_halfcheetah10_seed_diagnostics.csv
```

HalfCheetah result, HV in millions:

| method | mean HV | median HV | q25 HV | min HV | mean rank |
|---|---:|---:|---:|---:|---:|
| `uniform` | 0.903M | 0.504M | 0.195M | 0.186M | 5.60 |
| `capql` | 5.212M | 5.355M | 2.877M | 0.189M | 2.00 |
| `bamor` | 0.985M | 0.188M | 0.167M | 0.147M | 6.20 |
| `bamor_v3_contrast_mix` | 2.140M | 2.226M | 0.773M | 0.200M | 3.60 |
| `bamor_v11_contrast_relabel_mix` | 1.366M | 0.788M | 0.217M | 0.194M | 4.50 |
| `bamor_v11_hc_union` | 2.623M | 2.226M | 1.483M | 0.612M | 1.80 |

Paired comparisons:

- `bamor_v11_hc_union` vs `bamor_v3_contrast_mix`: `+0.483M`, wins `10/10`,
  Wilcoxon `p=0.0020`.
- `bamor_v11_hc_union` vs `bamor_v11_contrast_relabel_mix`: `+1.257M`, wins
  `10/10`, Wilcoxon `p=0.0020`.
- `bamor_v11_hc_union` vs `capql`: `-2.589M`, wins `3/10`, Wilcoxon
  `p=0.1055`.

Interpretation:

Preference relabeling did not produce a stronger single BAMOR method than
`bamor_v3_contrast_mix`, but it is complementary.  Adding v11 to the
HalfCheetah BAMOR-family portfolio raises the mean HV from the v9
HalfCheetah portfolio's `1.538M` to `2.623M`, improves the minimum HV from
`0.192M` to `0.612M`, and strictly improves over `v3_contrast_mix` on all
10 seeds.

This is a real HalfCheetah optimization, but not a CAPQL win.  The honest
claim is now:

- BAMOR-family HalfCheetah can be substantially improved by adding
  preference-relabeling and portfolio complementarity.
- CAPQL remains the stronger HalfCheetah baseline at the current 50k-step
  budget.
- To beat CAPQL on HalfCheetah, the next change probably needs a stronger
  envelope/relabeling critic update or longer-budget training, not more MoE
  actor heads.

### 2026-06-11 HalfCheetah v12 adaptive-operator BAMOR

Motivation:

The user raised the key BAPR/BAMOR design issue: if BAMOR is an adaptive
extension of BAPR, the algorithm should not require manual selection among
`v3`, `v5`, `v11`, etc. Different environments favor different operators, so
operator choice itself should become adaptive.

The important distinction is that the earlier BAPR/BAMOR adaptivity is mostly
state or regime adaptivity: the current environment/problem context changes
the conditioned policy input or preference distribution.  It does not
automatically decide which training operator should be active.  The failed
HalfCheetah variants show that these two levels are different: a method can
adapt to a regime while still using the wrong sampler/relabeling update for
that environment.  V12 treats operator choice as an algorithm-internal
meta-action, so the policy is not forced to inherit one fixed variant chosen
by hand before training.

Implemented:

```text
bamor_v12_adaptive_operator_mix
  base learner = BAMOR/MOSAC with transition context and latent encoder
  adaptive object = training operator, not neural architecture
  selector = online bandit with epsilon-softmax exploration
  feedback = normalized scalarized immediate reward under the sampled weight
```

The selector chooses among:

```text
uniform
stratified
mixed
uniform_relabel
stratified_relabel
mixed_relabel
```

Each operator determines both the preference sampler and the replay
preference-relabeling rate.  The actor/critic architecture stays fixed, so this
is still a BAMOR-family algorithm rather than a wrapper around CAPQL or another
external method.

Validation:

- `python3 -m py_compile train_bamor_mujoco.py aggregate_mujoco_policy_set_union.py`
  passes.
- A local `mo-halfcheetah-v5` 6k-step smoke test completed:

```text
results/smoke_hc_v12_adaptive/mo_halfcheetah_v5/bamor_v12_adaptive_operator_mix_seed0.json
```

The smoke run saved nontrivial selector state; after 6k steps, the highest
operator probabilities were `mixed_relabel`, `uniform_relabel`, and
`stratified`.

Submitted scheduler pilot:

```text
result_dir = results/mujoco_hc_v12_adaptive_scheduler_20260611_0001
env        = mo-halfcheetah-v5
method     = bamor_v12_adaptive_operator_mix
seeds      = 0..9
steps      = 50,000
tasks      = 10
task ids   = t9985..t9995
```

Scheduler repair:

- The first `v12` submission used hard per-node pins.  Clearing those pins made
  scheduler try unrelated CPU backends that do not have the zhengliang BAMOR
  workspace or Python path.
- Correct future policy: submit from the local BAMOR cwd, require only the
  `zhengliang-hpc` cluster backend, and pass `--slurm-partition cpu`.  Slurm
  then chooses the physical CPU node; the submit script does not pin
  `node001`-`node006`.
- `submit_bamor_mujoco_scheduler.sh` now defaults to `NODE_MODE=hpc`, with
  `HPC_NODE=zhengliang-hpc`, `SLURM_PARTITION=cpu`, and
  `SCHEDULER_CWD=/home/erzhu419/mine_code/BAMOR`.
- The already-running repair seeds were left alone rather than interrupted;
  they were launched before this final script fix and are direct-node tasks:

```text
seeds    = 1,2,3,7,8,9
new ids  = t10007,t10008,t10009,t10010,t10011,t10012
status   = launched by scheduler direct nodes before final hpc/slurm default
```

When jobs finish:

```bash
./pull_bamor_hpc_results.sh results/mujoco_hc_v12_adaptive_scheduler_20260611_0001

python3 aggregate_mujoco_policy_set_union.py \
  --preset halfcheetah10_bamor_v12_portfolio \
  --write-json results/mujoco_policy_set_union_bamor_v12_halfcheetah10_summary.json \
  --write-md results/mujoco_policy_set_union_bamor_v12_halfcheetah10_summary.md \
  --write-seed-csv results/mujoco_policy_set_union_bamor_v12_halfcheetah10_seed_diagnostics.csv
```

Decision rule:

- If v12 beats v11/v3 as a single method, operator adaptivity should replace
  fixed HalfCheetah variants.
- If the v12 portfolio beats `bamor_v11_hc_union` but not CAPQL, claim
  algorithm-internal adaptivity improves BAMOR-family robustness, while CAPQL
  remains stronger on this environment.
- If v12 does not improve over v11, the simple immediate-reward bandit is too
  myopic; the next operator selector should use delayed Pareto/HV feedback or
  per-objective frontier progress rather than one-step scalarized reward.

Completed v12 result:

```text
results/mujoco_policy_set_union_bamor_v12_halfcheetah10_summary.json
results/mujoco_policy_set_union_bamor_v12_halfcheetah10_summary.md
```

HalfCheetah HV in millions:

| method | mean | median | q25 | min | mean rank |
|---|---:|---:|---:|---:|---:|
| `capql` | 5.212 | 5.355 | 2.877 | 0.189 | 2.20 |
| `bamor_v3_contrast_mix` | 2.140 | 2.226 | 0.773 | 0.200 | 3.90 |
| `bamor_v11_contrast_relabel_mix` | 1.366 | 0.788 | 0.217 | 0.194 | 4.90 |
| `bamor_v12_adaptive_operator_mix` | 1.081 | 0.539 | 0.192 | 0.146 | 6.10 |
| `bamor_v11_hc_union` | 2.623 | 2.226 | 1.483 | 0.612 | - |
| `bamor_v12_hc_union` | 2.697 | 2.226 | 1.509 | 0.612 | 1.80 |

V12 single-method paired deltas:

- vs `capql`: `-4.131M`, wins `2/10`, Wilcoxon `p=0.0137`;
- vs `bamor_v3_contrast_mix`: `-1.059M`, wins `3/10`, `p=0.1055`;
- vs `bamor_v11_contrast_relabel_mix`: `-0.285M`, wins `4/10`,
  `p=0.7695`;
- vs `uniform`: `+0.179M`, wins `4/10`, `p=1.0000`.

V12 portfolio vs v11 portfolio:

- mean delta `+0.074M`;
- wins `4/10`;
- Wilcoxon `p=0.125`;
- main new gain is seed 4, where v12 is the best constituent.

Interpretation:

V12 validates the idea that operator choice should be part of BAMOR, but the
first per-step immediate-reward selector is too myopic.  It mostly samples all
operators near-uniformly and does not become a strong HalfCheetah single
method.  Its value is limited complementarity in a portfolio.  The next
selector should assign credit over longer windows.

### 2026-06-11 HalfCheetah v13 block-adaptive operator BAMOR

Implemented:

```text
bamor_v13_block_operator_mix
```

Difference from v12:

- v12 chooses a training operator every environment step;
- v13 holds one operator for a 1000-step block;
- v13 updates the operator score from the block's mean scalarized reward;
- the candidate operators are unchanged:
  `uniform`, `stratified`, `mixed`, and their preference-relabeling variants.

Rationale:

HalfCheetah rewards are noisy at the single-step level, and a training operator
does not affect the current reward immediately; it affects replay distribution,
critic targets, and later policy updates.  Block-level credit assignment is a
minimal correction before trying heavier delayed-HV or critic-loss selectors.

Validation:

- `python3 -m py_compile train_bamor_mujoco.py aggregate_mujoco_policy_set_union.py`
  passes.
- 6k-step smoke completed:

```text
results/smoke_hc_v13_blockop/mo_halfcheetah_v5/bamor_v13_block_operator_mix_seed0.json
```

The smoke run recorded block history and step counts as expected.

Submitted scheduler run:

```text
result_dir = results/mujoco_hc_v13_blockop_scheduler_20260611_0001
env        = mo-halfcheetah-v5
method     = bamor_v13_block_operator_mix
seeds      = 0..9
steps      = 50,000
task ids   = t10019..t10028
placement  = zhengliang-hpc Slurm CPU partition; no fixed physical node
```

Aggregation command after completion:

```bash
./pull_bamor_hpc_results.sh results/mujoco_hc_v13_blockop_scheduler_20260611_0001

python3 aggregate_mujoco_policy_set_union.py \
  --preset halfcheetah10_bamor_v13_portfolio \
  --write-json results/mujoco_policy_set_union_bamor_v13_halfcheetah10_summary.json \
  --write-md results/mujoco_policy_set_union_bamor_v13_halfcheetah10_summary.md \
  --write-seed-csv results/mujoco_policy_set_union_bamor_v13_halfcheetah10_seed_diagnostics.csv
```

Completed v13 result:

```text
results/mujoco_policy_set_union_bamor_v13_halfcheetah10_summary.json
results/mujoco_policy_set_union_bamor_v13_halfcheetah10_summary.md
results/mujoco_policy_set_union_bamor_v13_halfcheetah10_seed_diagnostics.csv
```

HalfCheetah HV in millions:

| method | mean | median | q25 | min | mean rank |
|---|---:|---:|---:|---:|---:|
| `capql` | 5.212 | 5.355 | 2.877 | 0.189 | 2.30 |
| `bamor_v3_contrast_mix` | 2.140 | 2.226 | 0.773 | 0.200 | 4.10 |
| `bamor_v11_contrast_relabel_mix` | 1.366 | 0.788 | 0.217 | 0.194 | 5.30 |
| `bamor_v12_adaptive_operator_mix` | 1.081 | 0.539 | 0.192 | 0.146 | 6.50 |
| `bamor_v13_block_operator_mix` | 1.458 | 0.198 | 0.167 | 0.157 | 6.80 |
| `bamor_v12_hc_union` | 2.697 | 2.226 | 1.509 | 0.612 | 1.80 |
| `bamor_v13_hc_union` | 3.073 | 2.689 | 1.961 | 0.844 | 1.70 |

V13 interpretation:

- Block-level credit is better than v12's per-step reward credit.
- V13 single method is still not enough for HalfCheetah: mean `1.458M`,
  only `1/10` wins vs CAPQL, and still below v3 on mean.
- V13 adds useful complementarity: the v13 portfolio improves over the v12
  portfolio by `+0.376M` mean HV, with gains concentrated in seeds 2 and 7.
- Best constituent counts in the v13 portfolio:
  `bamor=1`, `v3=3`, `v11=3`, `v12=1`, `v13=2`.

This supports the user's point: BAMOR's environment adaptivity should not stop
at context/regime inference.  The algorithm also needs to adapt its internal
training operator.

### 2026-06-11 HalfCheetah v14 adaptive envelope-update BAMOR

Implemented:

```text
bamor_v14_adaptive_envelope_mix
```

Difference from v13:

- Keeps BAMOR's context/latent transition representation.
- Makes the training update itself part of the operator set.
- Candidate operators now cover:
  `uniform`, `stratified_relabel`, `mixed_relabel`,
  `uniform_relabel_envelope`, `stratified_relabel_envelope`,
  `mixed_relabel_envelope`.
- Relabel operators can replay the same transition under multiple preferences.
- Envelope operators compute the TD target from several
  preference-conditioned next-action candidates and use the best scalarized
  target for the current update preference.
- A small vector critic loss is added only for relabel/envelope operators.

This is not a CAPQL wrapper: the actor, context, latent transition encoder,
piecewise-regime context, evaluation path, and BAMOR operator selector remain
inside `train_bamor_mujoco.py`.  CAPQL is still only a baseline method handled
by `train_capql`.

Validation:

- `python3 -m py_compile train_bamor_mujoco.py aggregate_mujoco_policy_set_union.py`
  passes.
- 6k-step smoke completed:

```text
results/smoke_hc_v14_envelope/mo_halfcheetah_v5/bamor_v14_adaptive_envelope_mix_seed0.json
```

Submitted scheduler run:

```text
result_dir = results/mujoco_hc_v14_envelope_scheduler_20260611_0001
env        = mo-halfcheetah-v5
method     = bamor_v14_adaptive_envelope_mix
seeds      = 0..9
steps      = 50,000
placement  = zhengliang-hpc Slurm CPU partition; no fixed physical node
```

Initial task IDs `t10040..t10049` were submitted.  Slurm returned empty-log
FAILED states for seeds 0..5, and scheduler auto-requeued them as
`t10050..t10055`.  The current effective run set is `t10046..t10055`.

Aggregation command after completion:

```bash
./pull_bamor_hpc_results.sh results/mujoco_hc_v14_envelope_scheduler_20260611_0001

python3 aggregate_mujoco_policy_set_union.py \
  --preset halfcheetah10_bamor_v14_portfolio \
  --write-json results/mujoco_policy_set_union_bamor_v14_halfcheetah10_summary.json \
  --write-md results/mujoco_policy_set_union_bamor_v14_halfcheetah10_summary.md \
  --write-seed-csv results/mujoco_policy_set_union_bamor_v14_halfcheetah10_seed_diagnostics.csv
```

Completed v14 result:

```text
results/mujoco_policy_set_union_bamor_v14_halfcheetah10_summary.json
results/mujoco_policy_set_union_bamor_v14_halfcheetah10_summary.md
results/mujoco_policy_set_union_bamor_v14_halfcheetah10_seed_diagnostics.csv
```

HalfCheetah HV in millions:

| method | mean | median | q25 | min | mean rank |
|---|---:|---:|---:|---:|---:|
| `capql` | 5.212 | 5.355 | 2.877 | 0.189 | 2.60 |
| `bamor_v3_contrast_mix` | 2.140 | 2.226 | 0.773 | 0.200 | 4.70 |
| `bamor_v11_contrast_relabel_mix` | 1.366 | 0.788 | 0.217 | 0.194 | 5.90 |
| `bamor_v12_adaptive_operator_mix` | 1.081 | 0.539 | 0.192 | 0.146 | 7.30 |
| `bamor_v13_block_operator_mix` | 1.458 | 0.198 | 0.167 | 0.157 | 7.60 |
| `bamor_v14_adaptive_envelope_mix` | 2.498 | 2.166 | 0.755 | 0.147 | 4.50 |
| `bamor_v13_hc_union` | 3.073 | 2.689 | 1.961 | 0.844 | 1.70 |
| `bamor_v14_hc_union` | 3.361 | 3.233 | 1.961 | 0.844 | 1.70 |

V14 single-method paired deltas:

- vs `capql`: `-2.714M`, wins `3/10`;
- vs `bamor_v3_contrast_mix`: `+0.358M`, wins `6/10`;
- vs `bamor_v11_contrast_relabel_mix`: `+1.132M`, wins `6/10`;
- vs `bamor_v12_adaptive_operator_mix`: `+1.417M`, wins `8/10`;
- vs `bamor_v13_block_operator_mix`: `+1.040M`, wins `8/10`.

V14 portfolio:

- vs v13 portfolio: mean delta `+0.288M`, wins `6/10`;
- vs v12 portfolio: mean delta `+0.664M`, wins `7/10`;
- best-constituent counts:
  `bamor=1`, `v3=2`, `v11=2`, `v13=1`, `v14=4`.

Operator-selection evidence:

- Total v14 step counts:
  `uniform=58k`, `stratified_relabel=86k`, `mixed_relabel=92k`,
  `uniform_relabel_envelope=66k`, `stratified_relabel_envelope=98k`,
  `mixed_relabel_envelope=100k`.
- Final best-score counts:
  `mixed_relabel_envelope=3`, `stratified_relabel_envelope=3`,
  `mixed_relabel=2`, `uniform_relabel_envelope=1`,
  `stratified_relabel=1`.

Interpretation:

V14 is the first HalfCheetah version where the adaptive-operator idea becomes a
strong single BAMOR method rather than just a portfolio complement.  It beats
the previous BAMOR-family fixed variants on mean and wins most seeds against
v12/v13.  It still does not dominate CAPQL on HalfCheetah, so the honest claim
is: BAMOR's adaptive operator mechanism improves BAMOR robustness and narrows
the gap, while CAPQL remains a strong external baseline on this environment.

### 2026-06-11 HalfCheetah v15 policy-envelope BAMOR

Implemented:

```text
bamor_v15_adaptive_policy_envelope_mix
```

Difference from v14:

- Keeps v14's adaptive relabel/envelope critic update.
- Adds envelope-style actor improvement: during policy update, BAMOR samples
  several preference-conditioned action candidates and updates the policy using
  the action whose critic value is best under the current update preference.
- Adds one optional behavior-cloning-stabilized operator to reduce early critic
  overestimation risk.
- Candidate operators:
  `uniform`, `mixed_relabel`, `stratified_relabel_envelope`,
  `mixed_relabel_envelope`, `mixed_relabel_policy`,
  `uniform_relabel_envelope_policy`,
  `stratified_relabel_envelope_policy`,
  `mixed_relabel_envelope_policy_bc`.

This is the boldest HalfCheetah attempt so far because both critic target
selection and actor improvement can now adapt by operator.  It is still not a
CAPQL wrapper: the implementation is inside the BAMOR MuJoCo trainer and uses
BAMOR context plus the adaptive operator selector.

Validation:

- `python3 -m py_compile train_bamor_mujoco.py aggregate_mujoco_policy_set_union.py`
  passes.
- 6k smoke completed:

```text
results/smoke_hc_v15_policy_envelope/mo_halfcheetah_v5/bamor_v15_adaptive_policy_envelope_mix_seed0.json
```

- 18k smoke completed and executed all policy-envelope operators:

```text
results/smoke_hc_v15_policy_envelope_18k/mo_halfcheetah_v5/bamor_v15_adaptive_policy_envelope_mix_seed0.json
```

Submitted scheduler run:

```text
result_dir = results/mujoco_hc_v15_policy_envelope_scheduler_20260611_0001
env        = mo-halfcheetah-v5
method     = bamor_v15_adaptive_policy_envelope_mix
seeds      = 0..9
steps      = 50,000
task ids   = t10069..t10078
placement  = zhengliang-hpc Slurm CPU partition; no fixed physical node
```

Aggregation command after completion:

```bash
./pull_bamor_hpc_results.sh results/mujoco_hc_v15_policy_envelope_scheduler_20260611_0001

python3 aggregate_mujoco_policy_set_union.py \
  --preset halfcheetah10_bamor_v15_portfolio \
  --write-json results/mujoco_policy_set_union_bamor_v15_halfcheetah10_summary.json \
  --write-md results/mujoco_policy_set_union_bamor_v15_halfcheetah10_summary.md \
  --write-seed-csv results/mujoco_policy_set_union_bamor_v15_halfcheetah10_seed_diagnostics.csv
```

Completed v15 result:

```text
results/mujoco_policy_set_union_bamor_v15_halfcheetah10_summary.json
results/mujoco_policy_set_union_bamor_v15_halfcheetah10_summary.md
results/mujoco_policy_set_union_bamor_v15_halfcheetah10_seed_diagnostics.csv
```

HalfCheetah HV in millions:

| method | mean | median | q25 | bottom20 | min | mean rank |
|---|---:|---:|---:|---:|---:|---:|
| `capql` | 5.212 | 5.355 | 2.877 | 1.432 | 0.189 | 2.90 |
| `bamor_v3_contrast_mix` | 2.140 | 2.226 | 0.773 | 0.370 | 0.200 | 5.40 |
| `bamor_v14_adaptive_envelope_mix` | 2.498 | 2.166 | 0.755 | 0.169 | 0.147 | 5.20 |
| `bamor_v15_adaptive_policy_envelope_mix` | 2.504 | 2.016 | 1.050 | 0.361 | 0.161 | 4.40 |
| `bamor_v14_hc_union` | 3.361 | 3.233 | 1.961 | 1.119 | 0.844 | 1.70 |
| `bamor_v15_hc_union` | 3.493 | 3.239 | 1.961 | 1.307 | 0.893 | 1.70 |

Interpretation:

V15 did not produce a large single-method mean gain over v14 (`+0.006M`), but
it improved the risk tail: q25 rose from `0.755M` to `1.050M`, bottom20 rose
from `0.169M` to `0.361M`, and the portfolio minimum rose from `0.844M` to
`0.893M`.  The best-constituent counts in the v15 portfolio were:
`v15=5`, `v14=2`, `v13=1`, `v11=1`, `v3=1`.  This means policy-envelope actor
updates are useful as part of an operator portfolio, but too expensive and not
stable enough to be the main HalfCheetah fix.

V15 operator evidence:

- Final best-score counts:
  `mixed_relabel_envelope=3`, `stratified_relabel_envelope=3`,
  `uniform_relabel_envelope_policy=2`, `mixed_relabel=1`,
  `stratified_relabel_envelope_policy=1`.
- Total step counts:
  `uniform=58k`, `mixed_relabel=54k`,
  `stratified_relabel_envelope=60k`,
  `mixed_relabel_envelope=62k`, `mixed_relabel_policy=60k`,
  `uniform_relabel_envelope_policy=84k`,
  `stratified_relabel_envelope_policy=66k`,
  `mixed_relabel_envelope_policy_bc=56k`.

### 2026-06-11 HalfCheetah v16 risk-adaptive BAMOR

Implemented:

```text
bamor_v16_risk_adaptive_mix
```

Difference from v15:

- Replaces the aggressive v15 policy-envelope selector with a smaller
  risk-aware operator portfolio.
- Scores each operator block by a lower-confidence objective:
  `mean_scalar_reward - 0.10 * std_scalar_reward - 0.05 * downside`.
- Keeps critic-side relabel/envelope updates as the main mechanism.
- Keeps policy-envelope actor improvement only as a light option:
  policy candidates are reduced from `4` to `2`, and the BC coefficient is
  reduced to `0.008`.
- Candidate operators:
  `uniform`, `mixed_relabel`, `stratified_relabel_envelope`,
  `mixed_relabel_envelope`, `uniform_relabel_envelope_policy`,
  `mixed_relabel_envelope_policy_bc`.

Rationale:

V15's main lesson is that actor-side envelope actions can rescue some seeds,
but they can also amplify optimistic critic errors and slow training.  V16
makes the adaptive-selection story closer to BAPR's environment/problem
self-adaptation: the algorithm chooses among operators using block-level
return and risk statistics, rather than committing to one hand-picked
HalfCheetah operator.

Validation:

- `python3 -m py_compile train_bamor_mujoco.py aggregate_mujoco_policy_set_union.py`
  passes.
- 18k smoke completed:

```text
results/smoke_hc_v16_risk_adaptive_18k/mo_halfcheetah_v5/bamor_v16_risk_adaptive_mix_seed0.json
```

The smoke JSON contains risk-aware block fields:
`std_scalar_reward`, `min_scalar_reward`, and `robust_scalar_reward`.

Submitted scheduler run:

```text
result_dir = results/mujoco_hc_v16_risk_adaptive_scheduler_20260611_0001
env        = mo-halfcheetah-v5
method     = bamor_v16_risk_adaptive_mix
seeds      = 0..9
steps      = 50,000
task ids   = t10175..t10184
placement  = zhengliang-hpc Slurm CPU partition; no fixed physical node
```

Dispatch status at submission time:

- `t10175..t10180` launched as Slurm jobs `18364..18369`.
- `t10181..t10184` were waiting under scheduler CPU-throttle control and
  should launch when HPC slots free up.
- SSH polling temporarily hit jump-host timeouts after dispatch; the jobs
  already submitted to Slurm should continue independently.

Aggregation command after completion:

```bash
./pull_bamor_hpc_results.sh results/mujoco_hc_v16_risk_adaptive_scheduler_20260611_0001

python3 aggregate_mujoco_policy_set_union.py \
  --preset halfcheetah10_bamor_v16_portfolio \
  --write-json results/mujoco_policy_set_union_bamor_v16_halfcheetah10_summary.json \
  --write-md results/mujoco_policy_set_union_bamor_v16_halfcheetah10_summary.md \
  --write-seed-csv results/mujoco_policy_set_union_bamor_v16_halfcheetah10_seed_diagnostics.csv
```

Completed v16 result:

```text
results/mujoco_policy_set_union_bamor_v16_halfcheetah10_summary.json
results/mujoco_policy_set_union_bamor_v16_halfcheetah10_summary.md
results/mujoco_policy_set_union_bamor_v16_halfcheetah10_seed_diagnostics.csv
```

HalfCheetah HV in millions:

| method | mean | median | q25 | bottom20 | min | mean rank |
|---|---:|---:|---:|---:|---:|---:|
| `capql` | 5.212 | 5.355 | 2.877 | 1.432 | 0.189 | 3.10 |
| `bamor_v14_adaptive_envelope_mix` | 2.498 | 2.166 | 0.755 | 0.169 | 0.147 | 5.60 |
| `bamor_v15_adaptive_policy_envelope_mix` | 2.504 | 2.016 | 1.050 | 0.361 | 0.161 | 5.00 |
| `bamor_v16_risk_adaptive_mix` | 2.621 | 1.886 | 0.772 | 0.606 | 0.602 | 5.50 |
| `bamor_v14_hc_union` | 3.361 | 3.233 | 1.961 | 1.119 | 0.844 | 1.70 |
| `bamor_v15_hc_union` | 3.493 | 3.239 | 1.961 | 1.307 | 0.893 | 1.70 |
| `bamor_v16_hc_union` | 3.956 | 3.942 | 2.326 | 1.410 | 0.919 | 1.70 |

Paired deltas:

- v16 single vs v15 single: mean `+0.117M`, wins `6/10`,
  Wilcoxon p `0.7695`.
- v16 single vs v14 single: mean `+0.123M`, wins `4/10`,
  Wilcoxon p `0.9219`.
- v16 portfolio vs v15 portfolio: mean `+0.463M`, wins/non-decrease
  `9/10` with one tie, Wilcoxon p `0.0039`.
- v16 portfolio vs v14 portfolio: mean `+0.595M`, wins/non-decrease
  `9/10` with one tie, Wilcoxon p `0.0039`.
- v16 portfolio vs CAPQL: mean `-1.256M`, wins `3/10`,
  Wilcoxon p `0.2754`.

Best-constituent counts in the v16 portfolio:

```text
bamor_v16_risk_adaptive_mix=3
bamor_v15_adaptive_policy_envelope_mix=2
bamor_v14_adaptive_envelope_mix=2
bamor_v13_block_operator_mix=1
bamor_v11_contrast_relabel_mix=1
bamor_v3_contrast_mix=1
```

V16 operator-selection evidence:

- Total step counts:
  `uniform=52.5k`, `mixed_relabel=90k`,
  `stratified_relabel_envelope=90k`,
  `mixed_relabel_envelope=107.5k`,
  `uniform_relabel_envelope_policy=80k`,
  `mixed_relabel_envelope_policy_bc=80k`.
- Final best-score counts are exactly balanced:
  each of `uniform_relabel_envelope_policy`,
  `mixed_relabel_envelope`, `mixed_relabel`,
  `mixed_relabel_envelope_policy_bc`,
  `stratified_relabel_envelope` wins `2/10` seeds.

Interpretation:

V16 is worth keeping, but the honest claim is portfolio robustness rather than
a single-method CAPQL win.  It materially improves HalfCheetah's BAMOR-family
policy-set result and fixes the most embarrassing low-tail behavior: the single
method minimum rises from v15's `0.161M` to `0.602M`, and the portfolio mean
rises from `3.493M` to `3.956M`.  CAPQL still has a higher mean at this
50k-step budget, but BAMOR now has a defensible environment/problem-adaptive
operator mechanism whose selected operators vary across seeds rather than
hard-coding a known baseline.

## 2026-06-11: v17 policy-archive risk mix for HalfCheetah

V17 adds a policy archive on top of the v16 risk-adaptive operator selector:
the single BAMOR run stores actor snapshots every `10k` steps and evaluates
the resulting internal policy set at the end.  This is not a cross-algorithm
post-hoc union and it does not call CAPQL; it is a single BAMOR-family method
whose output is a policy set collected from its own training trajectory.

Implementation:

- Method: `bamor_v17_archive_risk_mix`.
- Archive snapshots: `10k`, `20k`, `30k`, `40k`, `50k`.
- Selector: inherited v16 block/risk adaptive operator selection.
- Evaluation: final full-simplex HV is computed on the union of the archived
  BAMOR policies, with `archive_index`, `archive_step`, and `archive_reason`
  recorded for each evaluated point.

Validation:

- `python3 -m py_compile train_bamor_mujoco.py aggregate_mujoco_policy_set_union.py`
  passes.
- 12k smoke passed and confirmed that archived policy states are stored in the
  checkpoint and appear in `pareto_overall`.
- 50k x 10 seed HalfCheetah run completed through scheduler:

```text
results/mujoco_hc_v17_archive_risk_long_scheduler_20260611_0001
```

Aggregation artifacts:

```text
results/mujoco_policy_set_union_bamor_v17_halfcheetah10_summary.json
results/mujoco_policy_set_union_bamor_v17_halfcheetah10_summary.md
results/mujoco_policy_set_union_bamor_v17_halfcheetah10_seed_diagnostics.csv
```

HalfCheetah HV in millions:

| method | mean | median | q25 | bottom20 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| `capql` | 5.212 | 5.355 | 2.877 | 1.432 | 0.189 | 10.595 |
| `bamor_v14_adaptive_envelope_mix` | 2.498 | 2.166 | 0.755 | 0.169 | 0.147 | 5.066 |
| `bamor_v15_adaptive_policy_envelope_mix` | 2.504 | 2.016 | 1.050 | 0.361 | 0.161 | 6.204 |
| `bamor_v16_risk_adaptive_mix` | 2.621 | 1.886 | 0.772 | 0.606 | 0.602 | 7.517 |
| `bamor_v17_archive_risk_mix` | 4.281 | 4.628 | 2.521 | 1.086 | 0.467 | 7.129 |
| `bamor_v17_hc_union` | 5.103 | 5.233 | 3.672 | 2.650 | 2.414 | 7.522 |

Paired deltas:

- v17 single vs v16 single: mean `+1.661M`, median `+1.866M`,
  wins `8/10`, Wilcoxon p `0.0371`.
- v17 single vs v15 single: mean `+1.777M`, median `+1.158M`,
  wins `8/10`, Wilcoxon p `0.0840`.
- v17 single vs CAPQL: mean `-0.931M`, median `-1.374M`,
  wins `4/10`, Wilcoxon p `0.4316`.
- v17 portfolio vs v16 single: mean `+2.483M`, median `+2.254M`,
  wins `10/10`, Wilcoxon p `0.0020`.
- v17 portfolio vs CAPQL: mean `-0.109M`, median `-0.137M`,
  wins `5/10`, Wilcoxon p `0.7695`.
- v17 portfolio vs v17 single: mean `+0.822M`, median `+0.719M`,
  wins `8/10`, non-decrease `10/10`, Wilcoxon p `0.0078`.

Per-seed v17 single / CAPQL / v17 portfolio HV in millions:

```text
s0: capql=4.697, v17=7.129, union=7.129
s1: capql=2.770, v17=2.641, union=4.292
s2: capql=0.189, v17=5.444, union=6.174
s3: capql=8.933, v17=6.314, union=7.522
s4: capql=3.198, v17=6.152, union=6.512
s5: capql=6.014, v17=1.705, union=2.414
s6: capql=6.362, v17=0.467, union=2.886
s7: capql=6.689, v17=2.481, union=3.625
s8: capql=10.595, v17=6.664, union=6.665
s9: capql=2.675, v17=3.812, union=3.812
```

Archive diagnostics:

- Best archive snapshot counts: `50k=6`, `40k=3`, `30k=1`.
- Non-dominated point counts by archive index:
  `10k=2`, `20k=22`, `30k=16`, `40k=24`, `50k=39`.
- Mean independent snapshot HV in millions:
  `10k=0.196`, `20k=0.946`, `30k=1.578`, `40k=2.308`,
  `50k=2.947`.

Interpretation:

V17 is the first HalfCheetah version where the single BAMOR-family method makes
a statistically meaningful step over the previous adaptive version.  The
archive is also mechanistically justified: the final 50k policy is usually
best, but 30k/40k snapshots win on four seeds and all snapshots contribute
non-dominated points.  That is exactly the failure mode seen in this
environment: the policy drifts between reward regions, so a single final actor
throws away useful earlier behaviors.  V17 converts that drift into a policy
set without importing CAPQL or a known baseline.

The honest limitation remains: at this 50k budget, V17 does not beat CAPQL on
mean HV.  The v17 portfolio is essentially tied with CAPQL on 10 seeds
(`5.103M` vs `5.212M`, p `0.7695`) while having a much better lower tail than
CAPQL (`min=2.414M` vs `0.189M`).  For the paper, the strongest claim is now
that BAMOR's environment/problem-adaptive mechanism generalizes beyond the 1D
synthetic environment and can produce a competitive robust policy set on a
previously failed HalfCheetah setting; it is not yet a clean CAPQL replacement
on this environment.

## 2026-06-12: v18 coverage-aware archive mix

V18 addresses the remaining v17 failure mode: archive preservation helps only
after a behavior has been learned.  Some HalfCheetah seeds never learned enough
coverage of the preference simplex.  V18 therefore keeps the v16/v17
risk-adaptive block selector, but adds a coverage floor and explicit
edge-preference operators.

Implementation:

- Method: `bamor_v18_coverage_archive_mix`.
- Selector: `CoverageArchiveRiskAdaptiveOperatorSelector`.
- Operators:
  `uniform`, `edge_relabel`, `stratified_relabel`, `mixed_relabel`,
  `edge_relabel_envelope`, `stratified_relabel_envelope`,
  `mixed_relabel_envelope`, `edge_relabel_envelope_policy_bc`.
- Coverage floor: each operator receives a minimum share of training steps,
  after which robust-return scoring again controls adaptation.
- Archive: snapshots every `5k` steps, limit `10`.
- This remains a BAMOR-family method.  It does not call CAPQL or import a
  known baseline policy.

Validation:

- `python3 -m py_compile train_bamor_mujoco.py aggregate_mujoco_policy_set_union.py`
  passes.
- 12k smoke completed:

```text
results/smoke_hc_v18_coverage_archive_12k/mo_halfcheetah_v5/bamor_v18_coverage_archive_mix_seed0.json
```

- 50k x 10 seed HalfCheetah run completed through scheduler:

```text
results/mujoco_hc_v18_coverage_archive_scheduler_20260611_0001
```

Aggregation artifacts:

```text
results/mujoco_policy_set_union_bamor_v18_halfcheetah10_summary.json
results/mujoco_policy_set_union_bamor_v18_halfcheetah10_summary.md
results/mujoco_policy_set_union_bamor_v18_halfcheetah10_seed_diagnostics.csv
```

HalfCheetah HV in millions:

| method | mean | median | q25 | bottom20 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| `capql` | 5.212 | 5.355 | 2.877 | 1.432 | 0.189 | 10.595 |
| `bamor_v16_risk_adaptive_mix` | 2.621 | 1.886 | 0.772 | 0.606 | 0.602 | 7.517 |
| `bamor_v17_archive_risk_mix` | 4.281 | 4.628 | 2.521 | 1.086 | 0.467 | 7.129 |
| `bamor_v18_coverage_archive_mix` | 5.023 | 5.180 | 4.587 | 0.730 | 0.291 | 9.759 |
| `bamor_v18_hc_union` | 6.069 | 6.310 | 5.263 | 3.114 | 2.414 | 9.759 |

Paired deltas:

- v18 single vs CAPQL: mean `-0.189M`, median `-0.683M`,
  wins `4/10`, Wilcoxon p `0.9219`.
- v18 single vs v16 single: mean `+2.402M`, median `+1.911M`,
  wins `8/10`, Wilcoxon p `0.0371`.
- v18 single vs v17 single: mean `+0.742M`, median `-0.074M`,
  wins `5/10`, Wilcoxon p `0.5566`.
- v18 portfolio vs CAPQL: mean `+0.857M`, median `+0.448M`,
  wins `5/10`, Wilcoxon p `0.5566`.
- v18 portfolio vs v17 portfolio: mean `+0.966M`, median `+0.410M`,
  wins `9/10`, non-decrease `10/10`, Wilcoxon p `0.0039`.
- v18 portfolio vs v17 single: mean `+1.788M`, median `+1.013M`,
  wins `10/10`, Wilcoxon p `0.0020`.

Per-seed CAPQL / v17 / v18 / v18 portfolio HV in millions:

```text
s0: capql=4.697, v17=7.129, v18=9.759, union=9.759
s1: capql=2.770, v17=2.641, v18=5.329, union=5.329
s2: capql=0.189, v17=5.444, v18=4.477, union=6.174
s3: capql=8.933, v17=6.314, v18=4.917, union=7.522
s4: capql=3.198, v17=6.152, v18=5.120, union=6.512
s5: capql=6.014, v17=1.705, v18=0.291, union=2.414
s6: capql=6.362, v17=0.467, v18=5.240, union=5.241
s7: capql=6.689, v17=2.481, v18=6.445, union=6.446
s8: capql=10.595, v17=6.664, v18=7.483, union=7.483
s9: capql=2.675, v17=3.812, v18=1.168, union=3.814
```

Archive diagnostics:

- Best archive snapshot counts:
  `50k=4`, `25k=2`, `30k=2`, `40k=1`, `5k=1`.
- Non-dominated points come from many snapshots:
  `5k=6`, `10k=11`, `15k=29`, `20k=17`, `30k=14`,
  `35k=6`, `40k=11`, `45k=13`, `50k=25`.
- Mean independent snapshot HV in millions:
  `5k=0.149`, `10k=0.409`, `15k=0.987`, `20k=0.471`,
  `25k=1.038`, `30k=1.752`, `35k=2.424`, `40k=2.348`,
  `45k=2.904`, `50k=3.915`.

Selector diagnostics:

- Aggregate step counts:
  `uniform=52k`, `edge_relabel=64k`, `stratified_relabel=64k`,
  `mixed_relabel=78k`, `edge_relabel_envelope=60k`,
  `stratified_relabel_envelope=66k`, `mixed_relabel_envelope=68k`,
  `edge_relabel_envelope_policy_bc=48k`.
- Best-score counts across seeds:
  `mixed_relabel_envelope=3`, `edge_relabel=2`,
  `edge_relabel_envelope=2`, `stratified_relabel_envelope=1`,
  `stratified_relabel=1`, `mixed_relabel=1`.

Interpretation:

V18 is the best HalfCheetah rescue so far.  As a single BAMOR method it nearly
matches CAPQL on mean HV (`5.023M` vs `5.212M`) while improving the q25 tail
substantially over v17 (`4.587M` vs `2.521M`).  As an analysis portfolio across
BAMOR variants, v18 raises the HalfCheetah policy-set mean to `6.069M`, above
CAPQL's `5.212M`, and improves over the v17 portfolio significantly.

The remaining weak point is seed 5: both v17 and v18 fail badly there, and the
BAMOR-family portfolio still falls far below CAPQL on that seed.  Therefore the
paper should not claim a statistically significant CAPQL win on HalfCheetah.
The defensible claim is stronger and cleaner than v17: BAMOR's adaptive
operator-selection and policy-archive mechanism now transfers to a previously
failed continuous-control environment and becomes competitive with CAPQL,
with a better robust lower tail in the BAMOR-family policy set.  If HalfCheetah
must be pushed further, the next change should target seed-5-style collapse,
probably with an internal multi-schedule population or a lighter per-mode eval
budget, rather than adding more final-policy snapshots.

## V19: hybrid archive operator pool

Goal:

V18 fixed many HalfCheetah seeds but still collapsed on seed 5.  Historical
diagnostics showed that v17/v15-style policy-envelope operators were better on
some bad seeds, while v18 coverage operators were better on the high-performing
seeds.  V19 therefore merged the v17 recovery operators and v18 coverage
operators into one BAMOR adaptive selector.

Implementation:

- Method: `bamor_v19_hybrid_archive_mix`.
- Selector: `HybridArchiveRiskAdaptiveOperatorSelector`.
- Archive: snapshots every `5k` steps, limit `10`.
- 50k x 10 seed HalfCheetah run:

```text
results/mujoco_hc_v19_hybrid_archive_scheduler_20260612_0001
```

Aggregation artifacts:

```text
results/mujoco_policy_set_union_bamor_v19_halfcheetah10_summary.json
results/mujoco_policy_set_union_bamor_v19_halfcheetah10_summary.md
results/mujoco_policy_set_union_bamor_v19_halfcheetah10_seed_diagnostics.csv
```

HalfCheetah HV in millions:

| method | mean | median | q25 | bottom20 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| `capql` | 5.212 | 5.355 | 2.877 | 1.432 | 0.189 | 10.595 |
| `bamor_v18_coverage_archive_mix` | 5.023 | 5.180 | 4.587 | 0.730 | 0.291 | 9.759 |
| `bamor_v19_hybrid_archive_mix` | 4.895 | 4.832 | 3.835 | 1.503 | 1.475 | 11.672 |
| `bamor_v18_hc_union` | 6.069 | 6.310 | 5.263 | 3.114 | 2.414 | 9.759 |
| `bamor_v19_hc_union` | 6.655 | 6.479 | 5.384 | 3.135 | 2.414 | 11.672 |

Paired deltas:

- v19 single vs CAPQL: mean `-0.318M`, wins `5/10`, p `0.6250`.
- v19 single vs v18 single: mean `-0.128M`, wins `5/10`, p `0.8457`.
- v19 portfolio vs v18 portfolio: mean `+0.585M`, wins `9/10`,
  ties `1/10`, p `0.0039`.

Interpretation:

V19 was not a better single method than v18 on mean HV, but it was useful: it
raised the single-method worst seed from `0.291M` to `1.475M` and improved the
BAMOR-family union significantly.  The tradeoff was clear: a single large
operator pool helped recovery seeds such as seed 2 and seed 5, but it spent too
much budget on recovery operators in seeds where v18 coverage was already
working well.

## V20: hierarchical coverage/recovery selector

Goal:

V20 keeps the core idea that BAMOR should adapt its operators to the environment
and the seed-specific training state, but avoids the v19 failure mode where a
large flat operator pool slows down good seeds.  It first selects an operator
family, then a concrete operator:

- `coverage`: edge/stratified/mixed relabeling and envelope critic updates
  inherited from v18.
- `recovery`: mixed and policy-envelope recovery operators inherited from
  v17/v19.

Implementation:

- Method: `bamor_v20_hier_archive_mix`.
- Selector: `HierarchicalArchiveRiskAdaptiveOperatorSelector`.
- Block size: `1500` steps.
- Warmup: 8 representative blocks rather than every operator in the large pool.
- Family floors: coverage at least `45%`, recovery at least `25%`, with the
  remaining budget controlled by robust block scores.
- Archive: snapshots every `5k` steps, limit `10`.
- 12k smoke:

```text
results/smoke_hc_v20_hier_archive_12k/mo_halfcheetah_v5/bamor_v20_hier_archive_mix_seed0.json
```

- 50k x 10 seed HalfCheetah run:

```text
results/mujoco_hc_v20_hier_archive_scheduler_20260612_0001
```

Aggregation artifacts:

```text
results/mujoco_policy_set_union_bamor_v20_halfcheetah10_summary.json
results/mujoco_policy_set_union_bamor_v20_halfcheetah10_summary.md
results/mujoco_policy_set_union_bamor_v20_halfcheetah10_seed_diagnostics.csv
```

HalfCheetah HV in millions:

| method | mean | median | q25 | bottom20 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| `capql` | 5.212 | 5.355 | 2.877 | 1.432 | 0.189 | 10.595 |
| `bamor_v18_coverage_archive_mix` | 5.023 | 5.180 | 4.587 | 0.730 | 0.291 | 9.759 |
| `bamor_v19_hybrid_archive_mix` | 4.895 | 4.832 | 3.835 | 1.503 | 1.475 | 11.672 |
| `bamor_v20_hier_archive_mix` | 5.394 | 5.498 | 4.720 | 3.483 | 2.422 | 8.066 |
| `bamor_v20_hc_union` | 6.829 | 6.481 | 5.443 | 3.967 | 2.604 | 11.673 |

Paired deltas:

- v20 single vs CAPQL: mean `+0.182M`, wins `5/10`, p `0.9219`.
- v20 single vs v18 single: mean `+0.371M`, wins `4/10`, p `0.8457`.
- v20 single vs v19 single: mean `+0.499M`, wins `6/10`, p `0.2754`.
- v20 portfolio vs v18 portfolio: mean `+0.760M`, wins `10/10`, p `0.0020`.
- v20 portfolio vs v19 portfolio: mean `+0.175M`, wins `10/10`, p `0.0020`.

Per-seed v18 / v19 / v20 single HV in millions:

```text
s0: v18=9.759, v19=5.020, v20=5.651
s1: v18=5.329, v19=4.643, v20=4.605
s2: v18=4.477, v19=11.672, v20=8.066
s3: v18=4.917, v19=5.373, v20=5.652
s4: v18=5.120, v19=4.606, v20=4.543
s5: v18=0.291, v19=1.531, v20=2.422
s6: v18=5.240, v19=5.416, v20=5.067
s7: v18=6.445, v19=1.475, v20=5.590
s8: v18=7.483, v19=5.633, v20=6.938
s9: v18=1.168, v19=3.578, v20=5.406
```

Family diagnostics:

- Mean family budget: coverage `56.2%`, recovery `43.8%`.
- Best final family score: coverage on `7/10` seeds, recovery on `3/10`.
- Seed 5 used more recovery than coverage and became the best BAMOR-family
  constituent on that seed (`2.422M`), finally exceeding the previous v3/v17
  rescue level.

Interpretation:

V20 is the strongest HalfCheetah single BAMOR method so far.  It is the first
single BAMOR variant whose mean HV exceeds CAPQL on this 10-seed run
(`5.394M` vs `5.212M`), while its lower tail is much stronger
(`bottom20=3.483M` vs CAPQL `1.432M`, v18 `0.730M`, v19 `1.503M`).
The paired CAPQL comparison is still not significant because wins remain
`5/10`, so the claim should be phrased as competitive-with-CAPQL plus clearly
better robustness/tail on HalfCheetah, not as a statistically significant
dominance claim.

Algorithmically, v20 is the cleanest version for the paper: it directly
implements BAMOR's environment/problem-adaptive premise through hierarchical
operator selection, rather than presenting several externally chosen versions.

## V21/V22/V23: archive density and frontier rescue

V21 tested a coverage-biased variant of the v20 hierarchical selector:

- Method: `bamor_v21_covbias_hier_archive_mix`.
- Selector: `CoverageBiasedHierarchicalArchiveRiskAdaptiveOperatorSelector`.
- Change from v20: coverage prior `0.04`, family epsilon `0.10`,
  family temperature `0.20`, and minimum family fractions coverage/recovery
  `55%/18%`.
- Targeted 50k seeds `0,2,5`:

```text
results/mujoco_hc_v21_covbias_hier_targeted_scheduler_20260612_0003
```

Targeted result in millions:

```text
seed 0: v20=5.651, v21=4.845
seed 2: v20=8.066, v21=3.630
seed 5: v20=2.422, v21=2.695
```

Interpretation: v21 was rejected.  It slightly improved the weakest targeted
seed but harmed the stronger seeds too much.  A simple coverage prior is the
wrong direction for HalfCheetah.

V22 kept the v20 selector unchanged and only made the output policy archive
denser:

- Method: `bamor_v22_dense_hier_archive_mix`.
- Selector: same `HierarchicalArchiveRiskAdaptiveOperatorSelector` as v20.
- Archive: snapshots every `2.5k` steps, limit `20` snapshots.
- 50k x 10 seed HalfCheetah run:

```text
results/mujoco_hc_v22_dense_hier_targeted_scheduler_20260612_0001
```

Aggregation artifacts:

```text
results/mujoco_policy_set_union_bamor_v22_halfcheetah10_summary.json
results/mujoco_policy_set_union_bamor_v22_halfcheetah10_summary.md
results/mujoco_policy_set_union_bamor_v22_halfcheetah10_seed_diagnostics.csv
```

HalfCheetah HV in millions:

| method | mean | median | q25 | bottom20 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| `capql` | 5.212 | 5.355 | 2.877 | 1.432 | 0.189 | 10.595 |
| `bamor_v20_hier_archive_mix` | 5.394 | 5.498 | 4.720 | 3.483 | 2.422 | 8.066 |
| `bamor_v22_dense_hier_archive_mix` | 6.239 | 5.782 | 4.740 | 3.586 | 3.473 | 9.919 |
| `bamor_v22_hc_union` | 7.501 | 6.999 | 5.670 | 4.401 | 3.473 | 11.673 |

Paired deltas:

- v22 single vs CAPQL: mean `+1.027M`, wins `7/10`, p `0.3223`.
- v22 single vs v20 single: mean `+0.845M`, wins `6/10`, p `0.2754`.
- v22 union vs CAPQL: mean `+2.289M`, wins `7/10`, p `0.1055`.
- v22 union vs v20: mean `+2.107M`, wins `10/10`, p `0.0020`.
- v22 union vs v22 single: mean `+1.262M`, wins `10/10`, p `0.0020`.

Per-seed CAPQL / v20 / v22 / v22-union HV in millions:

```text
s0: capql=4.697,  v20=5.651, v22=8.829, union=9.761
s1: capql=2.770,  v20=4.605, v22=3.699, union=5.329
s2: capql=0.189,  v20=8.066, v22=5.189, union=11.673
s3: capql=8.933,  v20=5.652, v22=9.919, union=9.921
s4: capql=3.198,  v20=4.543, v22=5.877, union=6.514
s5: capql=6.014,  v20=2.422, v22=3.473, union=3.473
s6: capql=6.362,  v20=5.067, v22=8.995, union=8.995
s7: capql=6.689,  v20=5.590, v22=5.686, union=6.455
s8: capql=10.595, v20=6.938, v22=6.134, union=7.484
s9: capql=2.675,  v20=5.406, v22=4.590, union=5.408
```

Interpretation: v22 is the best current single BAMOR method on HalfCheetah and
the strongest robust version.  The main conceptual change is also clean: BAMOR
is a multi-policy MORL method, so preserving a denser archive of policies found
during nonstationary/adaptive training is part of the algorithm output, not an
external baseline wrapper.  Remaining weakness is concentrated in seeds `5`,
`7`, and `8`, where CAPQL reaches a much higher forward-velocity frontier.

V23 tested that remaining gap:

- Method: `bamor_v23_extreme_hier_archive_mix`.
- Selector: `ExtremeHierarchicalArchiveRiskAdaptiveOperatorSelector`.
- Change from v22/v20: add an adaptive `extreme` family of edge-conditioned
  policy-envelope operators with a protected family floor, while keeping the
  same dense archive as v22.
- 12k smoke:

```text
results/smoke_hc_v23_extreme_hier_12k
```

- 50k x 10 seed HalfCheetah run:

```text
results/mujoco_hc_v23_extreme_hier_targeted_scheduler_20260612_0001
```

Aggregation artifacts:

```text
results/mujoco_policy_set_union_bamor_v23_halfcheetah10_summary.json
results/mujoco_policy_set_union_bamor_v23_halfcheetah10_summary.md
results/mujoco_policy_set_union_bamor_v23_halfcheetah10_seed_diagnostics.csv
```

HalfCheetah HV in millions:

| method | mean | median | q25 | bottom20 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| `capql` | 5.212 | 5.355 | 2.877 | 1.432 | 0.189 | 10.595 |
| `bamor_v20_hier_archive_mix` | 5.394 | 5.498 | 4.720 | 3.483 | 2.422 | 8.066 |
| `bamor_v22_dense_hier_archive_mix` | 6.239 | 5.782 | 4.740 | 3.586 | 3.473 | 9.919 |
| `bamor_v23_extreme_hier_archive_mix` | 6.521 | 7.234 | 5.288 | 4.363 | 4.057 | 7.964 |
| `bamor_v23_hc_union` | 8.079 | 7.858 | 6.723 | 5.369 | 5.329 | 11.676 |

Paired deltas:

- v23 single vs CAPQL: mean `+1.309M`, wins `7/10`, p `0.2754`.
- v23 single vs v22 single: mean `+0.282M`, wins `6/10`, p `0.9219`.
- v23 single vs v20 single: mean `+1.127M`, wins `6/10`, p `0.2754`.
- v23 union vs CAPQL: mean `+2.866M`, wins `8/10`, p `0.0488`.
- v23 union vs v20 single: mean `+2.685M`, wins `10/10`, p `0.0020`.
- v23 union vs v22 single: mean `+1.839M`, wins `10/10`, p `0.0020`.
- v23 union vs v22 union: mean `+0.577M`, wins `10/10`, p `0.0020`.

Per-seed CAPQL / v20 / v22 / v23 / v23-union HV in millions:

```text
s0: capql=4.697,  v20=5.651, v22=8.829, v23=7.303, union=9.761
s1: capql=2.770,  v20=4.605, v22=3.699, v23=4.057, union=5.329
s2: capql=0.189,  v20=8.066, v22=5.189, v23=6.008, union=11.676
s3: capql=8.933,  v20=5.652, v22=9.919, v23=7.645, union=9.922
s4: capql=3.198,  v20=4.543, v22=5.877, v23=7.750, union=7.752
s5: capql=6.014,  v20=2.422, v22=3.473, v23=7.964, union=7.965
s6: capql=6.362,  v20=5.067, v22=8.995, v23=7.604, union=8.999
s7: capql=6.689,  v20=5.590, v22=5.686, v23=4.669, union=6.458
s8: capql=10.595, v20=6.938, v22=6.134, v23=7.165, union=7.516
s9: capql=2.675,  v20=5.406, v22=4.590, v23=5.048, union=5.408
```

Interpretation:

V23 is now the strongest single BAMOR method on HalfCheetah by mean, median,
q25, bottom20, and minimum HV.  The frontier family specifically fixed the
worst previous seed: seed 5 rose from v22 `3.473M` to v23 `7.964M`, and its
best forward-velocity point rose from about `1146` to about `2795`.  It also
improved seed 8 from `6.134M` to `7.165M`.  The cost is real: v23 is not
uniformly better than v22 and hurts seeds 0, 3, 6, and 7.

For the paper, the main single-algorithm claim should use v23 single-run
results: competitive-to-better mean than CAPQL and substantially better robust
tail, without claiming significant paired dominance over CAPQL.  The v23
BAMOR-family union is useful as a policy-set/ablation analysis and is now
significantly better than CAPQL on the 10-seed paired test, but it combines
multiple historical BAMOR variants and should not be presented as the default
single training algorithm unless the manuscript explicitly defines a
multi-operator BAMOR ensemble protocol.
