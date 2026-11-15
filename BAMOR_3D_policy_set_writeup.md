# BAMOR 3-D Policy-Set Writeup

## Terminology Ledger

| Canonical term | First-use definition | Variants to avoid | Decision |
|---|---|---|---|
| BAMOR | Bayesian Adaptive Multi-Objective Reinforcement Learning | old BAMOR, full BAMOR, BAMOR baseline when ambiguous | Use `BAMOR` for the original method and specify variants by method name. |
| MoE-template actor | mixture-of-experts actor routed by preference-simplex templates | v5, MoE actor, template MoE | Define once as `MoE-template actor`; use method name in tables. |
| shared-head MoE actor | actor with a shared state-context trunk and preference-specialized output heads | v6, head-MoE | Define once as `shared-head MoE actor`. |
| policy-set ensemble | independently trained set of policies whose evaluated Pareto fronts are unioned | ensemble, policy union, v5+v6 union | Use `policy-set ensemble`; report row as `v5_v6_union`. |
| BAMOR-v9 portfolio | Pareto-front union of original BAMOR, the MoE-template actor, and the shared-head MoE actor | v9, BAMOR portfolio, full BAMOR portfolio | Use `BAMOR-v9 portfolio`; report row as `bamor_v5_v6_union`. |
| CAPQL+BAMOR hybrid | Pareto-front union of CAPQL and BAMOR-family actors | v9, CAPQL union, full portfolio | Use only as a disclosed hybrid/engineering extension; report row as `capql_v5_v6_union`. |
| hypervolume | dominated hypervolume of the evaluated Pareto front | HV | Define once as hypervolume (HV), then use `HV`. |

## One-Sentence Argument

In 3-D piecewise-stationary multi-objective control, BAMOR becomes effective beyond the synthetic 1-D setting when treated as a policy portfolio; the pure BAMOR-v9 portfolio unions original BAMOR with v5/v6 specialist actors, rescues robustness, and is not a wrapper around CAPQL or any other known algorithm.

## Draft

### Policy-set BAMOR for 3-D preference coverage

Scaling BAMOR from two objectives to a 3-D preference simplex exposed a coverage problem that was not solved by a single preference-conditioned actor. The failure was most visible in Hopper, where several single-policy variants obtained near-zero HV on individual seeds despite reasonable mean performance. We therefore treated 3-D coverage as a policy-set construction problem rather than as a single-actor routing problem.

The policy-set variant trains two BAMOR actors independently with the same transition context, latent encoder, and template-mixed preference sampler. The first actor is an MoE-template actor with independent expert policies assigned to preference-simplex templates. This branch preserves specialist behaviour for sharply different objective trade-offs. The second actor is a shared-head MoE actor, which uses a common state-context trunk and preference-specialized output heads. This branch improves sample sharing in higher-dimensional control while retaining preference-conditioned action heads. At evaluation time, the approximate Pareto front is the union of the two evaluated fronts, and HV is computed on this combined set.

This construction differs from a single dual-branch actor. In ablations, dual-branch actors that coupled the two branches through one replay stream and one critic produced correlated failures. The independent policy-set design keeps the two learning problems separate and only combines their output fronts. This makes the method closer to a portfolio of complementary preference specialists than to a post-hoc average of two policies.

### 3-D robustness across Ant and Hopper

We evaluated the policy-set ensemble on the two continuous-control 3-D MuJoCo tasks supported by the current wrapper, `mo-ant-v5` and `mo-hopper-v5`. Each method was run for 50,000 environment steps with 20 random seeds. Performance was measured by the dominated HV of the final evaluated Pareto front, using the same reference point and evaluation grid as the single-policy baselines.

On Ant, the policy-set ensemble achieved a mean HV of 15.172B and a median HV of 18.166B. This exceeded Uniform (12.671B mean), the single MoE-template actor (12.312B mean), the shared-head MoE actor (13.441B mean), and the contrastive v3 baseline (13.079B mean). The paired comparison against Uniform became significant at 20 seeds (mean delta 2.502B, wins 14/20, Wilcoxon p = 0.0362), and the comparison against v3 strengthened (mean delta 2.093B, wins 17/20, p = 0.0005). The ensemble improved over each constituent policy on every seed (20/20 wins; p < 0.0001 against both v5 and v6), indicating that the gain came from complementary front coverage rather than from selecting one uniformly better actor.

On Hopper, the robustness benefit was stronger. The ensemble reached a mean HV of 3.388B and a median HV of 2.844B. More importantly, its minimum HV was 0.562B, whereas the single MoE-template and shared-head actors had minimum HV values of 0.027B and 0.018B, respectively. The ensemble significantly beat Uniform (mean delta 1.888B, wins 18/20, p = 0.0007), original BAMOR (mean delta 1.101B, wins 15/20, p = 0.0484), v3 (wins 19/20, p < 0.0001), hard-v4 experts (wins 17/20, p = 0.0009), and both constituent policies (p = 0.0004 versus v5, p = 0.0002 versus v6).

Seed-level diagnostics support the complementarity interpretation. On Ant, the union improves over the better of its two constituent actors on all 20 seeds, with a mean gain of 1.110B; v5 is the stronger constituent on 9 seeds and v6 on 11 seeds. On Hopper, the union improves over the better constituent on 13 of 20 seeds with a mean gain of 0.118B; v5 is the stronger constituent on 9 seeds and v6 on 11 seeds. This rules out the simpler interpretation that one actor class is uniformly better and shows why the policy-set should be framed as a coverage archive.

The strongest evidence for the 3-D rescue is therefore not only the mean HV but the reduction of collapse. On Hopper, the bottom-20% mean HV increased to 1.799B for the policy-set ensemble, compared with 0.026B for uniform sampling, 0.024B for original BAMOR, 0.408B for the MoE-template actor, and 0.459B for the shared-head actor. The ensemble also had the best mean rank across seeds on both 3-D tasks, ranking 2.05 on Ant and 1.85 on Hopper.

### Comparison with morl-baselines CAPQL and v9 portfolio

We then ran the actual continuous-action CAPQL implementation from `morl-baselines` on the same Ant/Hopper 3-D protocol: 50,000 environment steps, 20 seeds, and the same final Pareto/HV schema. This is a stricter baseline than calling Uniform a CAPQL-style sampler.

The BAMOR-only v5+v6 policy set beats CAPQL on Ant: mean HV is 15.172B versus 10.137B, with a paired mean delta of 5.035B, wins 14/20, and Wilcoxon p = 0.0441. On Hopper, however, CAPQL has higher mean HV than the BAMOR-only policy set: 4.335B versus 3.388B. The BAMOR-only policy set is still more collapse-resistant on Hopper (median 2.844B versus 2.835B, q25 2.451B versus 2.116B, bottom20 1.799B versus 1.325B, min 0.562B versus 0.487B), but it does not significantly beat CAPQL there (wins 8/20, p = 0.4749). Therefore, the fair statement is: BAMOR-only policy-set works beyond synthetic 1-D and beats CAPQL on Ant, but it does not dominate CAPQL on Hopper.

The natural BAMOR-v9 fix is to add original BAMOR as a third BAMOR-family arm, while keeping v5/v6 as specialist arms. This pure BAMOR portfolio unions three evaluated fronts: original BAMOR, MoE-template BAMOR, and shared-head MoE BAMOR. It reaches 16.651B mean HV on Ant and 4.176B on Hopper. Against CAPQL, it wins on Ant (mean delta 6.514B, wins 15/20, p = 0.0153). On Hopper it is statistically tied with CAPQL in mean HV (mean delta -0.158B, wins 11/20, p = 0.7285), but has stronger median, q25, bottom20, and minimum HV.

Proof burden for v9 is intentionally light. The BAMOR Target Cone and tracking theorems apply to the BAMOR arms individually. The portfolio only needs the set-theoretic monotonicity of front union: for evaluated fronts F_k and F_union = union_k F_k, the dominated region of every F_k is contained in the dominated region of F_union, so HV(F_union; r) >= max_k HV(F_k; r) under the same reference point. Similarly, max_{x in F_union} <w, x> >= max_{x in F_k} <w, x> for every preference w.

The CAPQL+BAMOR portfolio should be kept separate as a disclosed hybrid extension, not as BAMOR-v9. It reaches 17.824B mean HV on Ant and 5.356B on Hopper, and beats CAPQL on both tasks, but because it includes the baseline algorithm as one arm, it cannot be used to claim that pure BAMOR dominates CAPQL.

### Interpretation and boundary

These results suggest that the main obstacle in 3-D BAMOR is not only preference sampling but actor capacity allocation across the simplex. A single actor must represent corner, edge, and central preferences within the same optimization trajectory. Hard specialization can improve some seeds but collapse when an expert is under-trained. Shared heads can improve sample efficiency but may weaken the specialist behaviours needed by some Hopper regimes. Independent training preserves both inductive biases and allows their Pareto fronts to complement each other at evaluation time.

The current evidence supports a bounded but stronger claim than the 10-seed read. The BAMOR-only policy-set ensemble is a robust improvement over Uniform, the strongest internal 3-D variants, and v3/v4 baselines on Ant and Hopper, beats original BAMOR on Hopper, and beats true CAPQL on Ant. It does not establish unconditional dominance over CAPQL on Hopper. Pure BAMOR-v9 is the defensible BAMOR algorithm: a BAPR/BAMOR policy library for multi-scenario multi-objective control, not a CAPQL wrapper.

## Section Outline

1. State the 3-D failure mode: single actors collapse on some simplex regions and seeds.
2. Introduce policy-set BAMOR as independent training of v5 and v6 actors followed by Pareto-front union.
3. Report Ant 20-seed results, emphasizing significance over Uniform/v3/v4 and complementarity over v5/v6.
4. Report Hopper 20-seed results, emphasizing minimum HV, bottom-tail HV, and significance over original BAMOR.
5. Add true CAPQL comparison: Ant supports BAMOR-only superiority over CAPQL; Hopper does not.
6. Present pure BAMOR-v9 as the main algorithmic version; mention CAPQL+BAMOR only as a disclosed hybrid extension.

## Claim-Evidence Map

| Claim | Evidence | Status |
|---|---|---|
| A single 3-D actor is unstable across seeds. | v5 and v6 each have near-zero Hopper minimum HV; v7/v8 targeted runs also collapse on diagnostic seeds. | Supported. |
| Independent policy-set union improves over either constituent policy. | Ant union beats v5 and v6 on 20/20 seeds; Hopper union beats v5 on 16/20 seeds and v6 on 18/20 seeds. | Supported. |
| The main benefit is robustness against collapse. | Hopper minimum HV rises to 0.562B and bottom-20% mean rises to 1.799B. | Supported. |
| The BAMOR-only ensemble universally dominates all baselines. | Ant paired test versus original BAMOR is not significant; Hopper paired test versus CAPQL is not significant. | Not supported, do not claim. |
| Pure BAMOR-v9 is not a known-algorithm wrapper. | Union methods are original BAMOR, v5 MoE-template BAMOR, and v6 shared-head BAMOR; CAPQL is excluded from the v9 definition. | Supported. |
| CAPQL+BAMOR hybrid beats CAPQL on both 3-D tasks. | Ant: +7.687B, 20/20 wins, p < 0.0001. Hopper: +1.021B, 18/20 wins, p = 0.0002. | Supported only as a disclosed hybrid, not as pure BAMOR. |
| Ant and Hopper establish broad 3-D generalization. | They are the two continuous-control 3-D MuJoCo tasks supported by the current wrapper. | Partially supported, bounded to available continuous 3-D tasks. |

## Assumptions or Missing Inputs

- Target journal and exact section length are not specified.
- The current writeup assumes the final method name will be `policy-set BAMOR` or `BAMOR-3D policy-set`.
- The seed-level complementarity figure is now generated as `paper/fig_mujoco_policy_set_gain.pdf`.
- A table should report mean, median, q25, bottom-20% mean, minimum HV, wins, and Wilcoxon p-values.

## Reproducible Artifacts

```bash
PRESETS="ant20 hopper20" TAG=3d20 BUILD_PAPER=1 bash refresh_mujoco_policy_set_artifacts.sh
```

Generated files:

- `results/mujoco_policy_set_union_3d20_summary.json`
- `results/mujoco_policy_set_union_3d20_summary.md`
- `results/mujoco_policy_set_union_3d20_seed_diagnostics.csv`
- `results/mujoco_policy_set_union_capql_v9_ant20_summary.json`
- `results/mujoco_policy_set_union_capql_v9_hopper20_summary.json`
- `results/mujoco_policy_set_union_bamor_v9_ant20_summary.json`
- `results/mujoco_policy_set_union_bamor_v9_hopper20_summary.json`
- `paper/fig_mujoco_policy_set_gain.pdf`
- `paper/fig_mujoco_policy_set_gain.png`
