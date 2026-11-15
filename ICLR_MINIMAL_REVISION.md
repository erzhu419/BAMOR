# BAMOR: minimal conference revision (2026-09-13)

The research question is whether feedback-driven allocation among fixed update
operators improves mode-wise policy-set coverage beyond retaining past actors.
Keep v23; do not add a new algorithm version before this comparison.

## Evidence and correction

The historical 20-seed continuous-control table compares 19 v23 actor snapshots
against a final CAPQL actor. Its overall evaluation also advances a persistent
mode schedule between weights and snapshots. That metric does not compare
policies on one common task distribution and cannot establish adaptive-selection
benefit. Historical files are retained, not merged into the corrected study.

Existing locked-mode, cone-grid data still show archive gains. Mean across modes:
HalfCheetah 3.032M vs CAPQL 2.221M; Ant 2.096B vs 1.838B; Hopper 3.357B vs
1.914B. These remain archive-vs-final comparisons with unmatched episode seeds
and restricted grids. HalfCheetah mode 3 favors CAPQL (0.827M vs 1.653M).

## Frozen pilot protocol

- Three environments: the existing piecewise HalfCheetah, Ant and Hopper presets.
- Three methods: v23 adaptive, v23 score-blind selector, CAPQL with an archive.
- New seeds 20, 21, 22; 50k training steps; exponential dwell mean 25k.
- Archive actor parameters at 5k, 7.5k, ..., 50k: 19 snapshots in every method.
  v23 uses the final transition encoder with every actor, matching its historical
  composed-policy definition. These are not historical encoder checkpoints.
- The score-blind selector retains all 18 arms, family quotas, warmup, network,
  sampling and update configurations. It uses zero scores during selection.
- Lock each evaluation mode; use 21 full-simplex weights (2-D uniform grid or
  3-D degree-5 simplex lattice including vertices) and three episodes per weight.
  Reset seeds are 100000 + 1000 * training_seed + 10 * mode + episode_index.
  Use the same reset seeds for every weight and archived policy.
- Report per-mode archive HV, final-actor HV, mean-mode HV and grid EUM; never
  union return vectors across modes. Retain existing environment reference points.
  Archive and final metrics use the same rollouts. Evaluation does not feed the
  selector; its score uses online training rewards only.
- Seed both the training environment reset and warmup action space in the new
  runs. Keep these reproducible reruns separate from historical outputs.
- Disable intermediate evaluation in the pilot (eval_freq=100000) and record
  training/evaluation time separately. Equal interactions do not imply equal
  compute: CAPQL retains its existing 256-wide network, v23 its 64-wide network.
- Route via the current scheduler to the HPC compute pool, with no single-node pin.
  Transfer only source and small logs/JSON summaries; checkpoints stay remote.

## Decision and remaining content work

Pilot launched: scheduler tasks t93490-t93516, automatically placed across
node001-node006. Remote outputs: `results/iclr_matched_v23_pilot_20260913`.
Five focused tests and short training/archival smoke runs passed for all three
methods before dispatch. These checks establish execution, not algorithm quality.

The pilot diagnoses protocol execution and effect direction, not significance.
If adaptive selection has no useful benefit over the score-blind archive,
do not describe archive gains as adaptive-selection gains or tune on these seeds.
If the comparison is informative, freeze the code and use seeds 23-42 for an
independent 20-seed confirmation with paired differences and confidence intervals.
Cross-environment claims need consistent evidence; online recovery additionally
needs switch-aligned measurements, not just locked-mode end-of-training fronts.

For the manuscript, distinguish BAMOR-TC, BAMOR-v23 and the abstract componentwise
operator. Its independently maximizing coordinates need not be jointly achievable.
Remove the actor-constant anchor penalty, the claimed preference/Pareto bijection,
and the 1%-of-error arithmetic. Keep valid conditional contraction and archive
set-inclusion results as supporting theory, not proofs of neural training or
empirical superiority. The synthetic TC-vs-uniform paired test is not significant.

ICLR main-track readiness remains open. A clean, bounded MORL/continual-learning
workshop contribution is a possible fallback, but changing venue does not remove
the need for matched evaluation. No positive result is assumed in this plan.

## Completed pilot and target repair (2026-09-13)

All 27 pilot outputs passed the matched-protocol checks. Mean-mode archive HV:

| Environment | v23 adaptive | v23 score-blind | CAPQL archive |
| --- | ---: | ---: | ---: |
| HalfCheetah | 2.469M | 2.704M | 2.744M |
| Ant | 2.187B | 2.177B | 2.151B |
| Hopper | 4.817B | 4.800B | 3.546B |

Adaptive beats score-blind in only 1/3, 2/3 and 1/3 seeds, respectively.
This does not justify selector superiority or a 20-seed confirmation yet.
Small result summary: `results/iclr_matched_v23_pilot_20260913_summary.json`.

Next: HalfCheetah, seeds 23-25, five methods (v23 adaptive/score-blind,
v24 target-consistent adaptive/score-blind, CAPQL archive), 15 tasks.
Keep the pilot's 50k budget, 19 snapshots, weights, reset formula and mode schedule.
V24 changes only the replay terminal mask to `terminated` and adds the selected
candidate's entropy bonus to every vector-target coordinate, so its scalar
projection equals the soft scalar target. Keep v23, networks and selectors intact.
This tests the two-fix bundle, not separate attribution to either fix.
Output root: `results/iclr_v24_target_repair_20260913`.
Submitted tasks `t93550-t93564`; scheduler selects nodes from the HPC compute pool.
All 12 focused tests passed in the remote runtime before submission, including
short real MuJoCo training for both v24 variants and updates with every operator.

Compare repaired vs original within both selector conditions, repaired adaptive
vs repaired score-blind, and repaired adaptive vs CAPQL; inspect final actors and
archive coverage separately. Do not promote based only on a better archive mean.
Warmup credit before learning and preference-dependent reward scoring remain
selector issues for the following step. Manuscript work stays paused.
Seeds 23-25 are now development data; any later 20-seed confirmation starts at 26.

## Target-repair result and weak-seed regression (2026-09-14)

All 15 outputs completed with matched evaluation and training-mode schedules.
HalfCheetah archive/final mean-mode HV (millions): v23 adaptive 2.923/2.540,
v23 score-blind 3.239/3.138, v24 adaptive 4.554/4.242,
v24 score-blind 4.923/4.592, CAPQL archive 4.718/3.887.
Both selector conditions improve over their respective v23 controls in all three
seeds. V24 adaptive loses to v24 score-blind in all three seeds on both metrics.
Seeds 24/25 never leave training mode 0; these results do not establish recovery
from the earlier weak nonstationary runs or an adaptive-selector benefit.

Next run only the two frozen v24 variants on HalfCheetah seeds 20-22 (six tasks).
Keep 50k steps, mean dwell 25k and the original matched evaluation unchanged.
Write new method files alongside existing controls in
`results/iclr_matched_v23_pilot_20260913`; do not rerun or overwrite the controls.
This is development regression on known weak seeds, not independent confirmation.
No training-code changes or selector tuning in this step. Manuscript stays paused.
Submitted regression tasks: `t93585-t93590`, with scheduler-selected HPC placement.

Regression completed: v24 adaptive archive/final HV is 2.222M/1.542M,
v24 score-blind 2.307M/1.447M, v23 adaptive 2.469M/0.929M, and CAPQL
2.744M/1.708M. Final HV improves for seeds 21/22 but declines for seed 20;
archive coverage improves over v23 adaptive in only one of the three seeds.
Across development seeds 20-25, v24 adaptive beats its score-blind archive in
only 1/6 seeds. V24 is not a confirmed cross-schedule improvement.

## V25 feedback diagnostic

Identical vector rewards produce different v24 risk scores under different
preference samplers. Its first three scored blocks also precede gradient learning.
V25 retains v24 targets, networks, 18 arms, family quotas and evaluation. After
the first gradient update, score each complete 1500-step block by
`mean(tanh((second_half_mean_reward - first_half_mean_reward) / prior_std))`.
Estimate per-objective variance causally with EMA rate 1/1500; freeze its square
root at block start with numerical floor 1e-6. The first 5000 transitions estimate
scales only, with the first warmup arm used throughout. Ignore incomplete blocks
for score updates. Preserve the existing score EMA rate and ten-arm warmup order.
The matched v25 score-blind variant has identical statistics and learning timing.
This removes direct sampled-weight scoring, not state-distribution or mode-change
confounding; balanced normalized reward progress is not a Pareto-coverage metric.

Next: HalfCheetah seeds 26-28, v25 adaptive/score-blind, v24 adaptive/score-blind
and CAPQL archive (15 tasks). Same 50k steps, mean dwell 25k and locked-full-simplex
evaluation. Freeze before launch; compare final actors and archives separately.
Output root: `results/iclr_v25_progress_pilot_20260914`. These are diagnostic
development seeds, not the 20-seed confirmation; reserve seeds 29 onward for that.
If v25 does not improve on its own score-blind control, do not expand it.
All 18 tests passed locally and in the server runtime before submission.
Submitted tasks: `t93762-t93776`, with scheduler-selected HPC placement.

## V25 result and update-stability boundary

All 15 tasks completed with matched schedules and evaluation. Archive/final HV
(millions): v25 adaptive 4.288/3.508, v25 score-blind 3.999/3.267,
v24 adaptive 4.409/4.195, v24 score-blind 4.048/3.583, CAPQL 3.939/3.421.
V25 wins archive HV against its own control in 3/3 seeds (+7.23% mean), but v24
also wins 3/3 (+8.93%). V25 is lower than v24 adaptive by 2.76% archive and
16.39% final HV. The difference between their within-version selector effects
is -0.072M archive / -0.371M final. These three seeds do not establish that the
new feedback rule improves on the old rule; v25 remains an unadopted candidate.

Seed 27 snapshot diagnosis uses cached return fronts, with the final encoder for
every actor, not contemporaneous online evaluations. V25's 45k snapshot scores
3.112M, versus 1.400M final. Mode 1 (the mode trained since step 13866) falls
only from 0.926M to 0.899M; modes 0/2/3 fall from 4.123/5.299/2.101M to
2.370/2.041/0.292M. V25 score-blind also falls from its best single snapshot
3.156M to 1.456M. This is not evidence that feedback alone causes degradation.
Diagnostic: `results/iclr_v25_seed27_snapshot_diagnostic.json`.

Do not launch a larger confirmation or alter the manuscript on these results.
Next isolate actor-envelope stability on the repaired v24 backbone: compare the
current actor update with the own-preference SAC actor update, holding critic
envelope, relabeling, network, score-blind selector and archive protocol fixed.
This is a proposed ablation, not an implemented or executed improvement. Only
after that comparison should selector changes and independent confirmation resume.

## Own-preference actor ablation (2026-09-14)

Implemented `bamor_v24_ownpref_actor_fixed_archive_mix`: set actor candidates to
one for all 18 arms, retaining existing BC, critic envelope, relabeling, soft
targets, networks and score-blind quotas/warmup. Keep arm labels even where this
makes their update configurations identical. No other training change.

Run only HalfCheetah seeds 26-28 at 50k steps / mean dwell 25k / hidden 64,
with the same locked-full-simplex evaluation and 19 actor snapshots. Reuse v24
score-blind and CAPQL controls in `results/iclr_v25_progress_pilot_20260914`;
write the ablation under its distinct method name. This is known-seed mechanism
regression, not independent confirmation. Seeds 29 onward remain unused.

Compare paired archive and final HV, per-mode coverage and seed-27 snapshot
retention. An archive gain alone does not establish a more stable final actor;
a final-actor gain with lost archive coverage is a tradeoff, not an overall win.
Only a joint improvement warrants testing the adaptive selector on this update.
Equal interaction budgets do not imply equal compute: actor sampling is reduced,
and changed Torch RNG consumption can change subsequent training trajectories.
Local tests: 21 passed, including all-arm update isolation and real-env routing.
The same 21 tests passed in the server runtime. Submitted only `t93891-t93893`
(seeds 26/27/28); scheduler chooses HPC placement without a required node.

Completed all three. Archive/final mean HV (millions): own-preference actor
3.989/3.444, original v24 score-blind 4.048/3.583, CAPQL 3.939/3.421.
Relative to v24 score-blind, means fall 1.47%/3.88% despite winning 2/3 seeds;
seed 28 falls 14.14%/21.59%. The actual operator sequences match in all three
pairs, and recorded trainer configurations differ only in actor candidates.

Seed 27 final HV improves 9.24%, but the trained terminal mode 1 falls 38.57%;
the gain comes from unvisited modes 2/3. Final/best-snapshot retention is 80.77%
versus 92.45% in the original. Seed 28's best single snapshot is 3.931M versus
4.661M, so its deficit is not just the final checkpoint. Snapshot comparisons
use the final encoder, not online learning curves. The three-seed minimum rises,
but this does not establish a stability or general robustness improvement.

Do not adopt the own-preference ablation or expand it to adaptive selection.
Small summaries: `results/iclr_v24_ownpref_actor_ablation_20260914_summary.json`
and `results/iclr_v24_ownpref_actor_snapshot_diagnostic.json`.
Next proposed diagnostic: complete the actor-envelope x critic-envelope 2x2
ablation on the same fixed-selector seeds, reusing the two finished cells.
Disable only critic candidates in each missing cell (six new runs), retaining
relabeling, vector-loss coefficients, BC, quotas and evaluation. Compare critic
effects within each actor setting and their interaction before choosing a new
training rule. This follow-up is not implemented or submitted yet; no new seeds
or manuscript changes in this step.

## Envelope factorial follow-up

Implemented the two missing cells: `bamor_v24_ownpref_critic_fixed_archive_mix`
(actor on / critic off) and `bamor_v24_ownpref_both_fixed_archive_mix`
(both off). Each overrides only the indicated candidate count to one; all
relabeling, losses, BC, networks, fixed-selector quotas and arm labels remain.
Reuse the original v24 score-blind (both on) and own-preference actor (actor off /
critic on). Run the two new methods on seeds 26-28 only, 50k steps, mean dwell
25k, hidden 64, eval frequency 100k, under the existing result root above.

For each seed and each archive/final metric, let Y_ac denote actor/critic search
enabled by a/c. Report critic effects Y_11-Y_10 and Y_01-Y_00, and interaction
(Y_11-Y_10)-(Y_01-Y_00), including per-mode values and seen/unseen exposure.
Candidate counts when enabled retain their original per-arm settings. Budget and
evaluation remain frozen; this is a development diagnostic, not confirmation.
No adaptive-selector change or manuscript revision. All 24 tests passed locally
and in the server runtime before launch; no duplicate tasks or result files found.
Submitted `t93924-t93929`: critic-off / both-off for seed 26, then 27, then 28.
Scheduler selects placement from the HPC pool; no required node is set.

### Completed factorial (2026-09-19)

All six tasks are done; all 15 results including reused controls are complete.
Evaluation grids, reset seeds, mode histories and 19 snapshots match. Actual
operator sequences, quotas and all non-ablated trainer settings also match.
Mean archive/final HV (millions): both on 4.048/3.583, actor off 3.989/3.444,
critic off 4.068/2.744, both off 4.098/3.407, CAPQL 3.939/3.421.
Critic-off is +0.50% archive but -23.41% final against both-on. Both-off is
+1.23%/-4.90%; its three-seed minima rise from 2.890/2.125M to 3.316/2.774M,
but seed 26 loses coverage and seed 28 loses final HV. Retain it as a lower-tail
tradeoff, not an adopted replacement or demonstrated general robustness gain.

The mean critic effects with actor on/off are -0.020/-0.109M archive and
+0.839/+0.036M final; interaction is +0.089M archive / +0.802M final. Final
interaction varies across seeds (-0.098, +2.127, +0.378M), so the large mean
is driven by seed 27 rather than evidence of a uniform coupling effect.
For critic-off seed 27, best/final snapshot HV is 2.921/0.450M (15.4% retained);
the final trained mode 1 remains 0.900M, while modes 0/2/3 are 0.260/0.448/0.193M.
Both-off restores seed-27 final HV to 2.774M but seed-28 best/final is
5.099/4.176M (81.9% retained). All snapshots use the final encoder.

Saved `results/iclr_v24_envelope_factorial_20260919_summary.json` (paired effects,
including per-mode and seen/unseen metrics) and the companion `_diagnostic.json`.
No new training rule adopted, no new tasks, and no manuscript changes.
Next proposed mechanism check: actor-envelope currently ranks with preference w
but backpropagates through the selected candidate's generating preference w_i.
Test preference-aligned improvement using detached candidate proposals while
retaining critic search, relabeling and BC, rather than deleting all search.
This is a code-grounded hypothesis, not an established cause or implemented fix.

## BAPR transfer and aligned actor pilot (2026-09-19)

Read BAPR v28/v32 code and the completed local v32 analysis: single-reference
polarity compensation has 98.55% causal mode accuracy and +44.2% mean return
versus equal-budget SAC, but fails the registered consistency gate (7/10 seeds,
23/30 events). Its remaining bottleneck is reference-controller headroom.
The sign-inverse transform does not apply to BAMOR's mass/gravity/damping and
nonlinear reward modes. Reuse the separation of identification, controller
capacity and retention diagnostics, not its estimator weights or compensation
formula. Source: `../BAPR/reports/regime_polarity_action_compensation_power_v32_2026-09-13.md`.

Implemented `bamor_v26_aligned_actor_fixed_archive_mix`. Keep the original fixed
selector and all critic, relabeling, BC and candidate-count settings. For
multi-candidate actor arms, retain own-preference SAC and add 0.1 times the
action-dimension-averaged MSE from its deterministic action to the detached
highest-Q proposal, masked to proposals strictly better than the own sampled
action under the same update preference. Average over the entire batch; no
projection when the own candidate wins. Coefficient fixed before results.
Single-candidate updates remain unchanged; BC acts on the own sampled action.
Record proposal wins/samples, projection updates and mean projection MSE.
These are critic-estimated gains, not guaranteed return improvements; shared
parameters can still interfere. This tests a new actor objective, not only an
otherwise identical gradient-path toggle, and adds no frozen policy bank.

Run only seeds 26-28 at 50k steps / mean dwell 25k / hidden 64 / eval frequency
100k, same locked evaluation and existing result root. Reuse both-on and
actor-off controls plus CAPQL. Compare archive, final, per-mode and minimum
HV; preserve negative results and keep seeds 29 onward reserved. Local tests:
29 passed, including exact legacy actor value/gradient/RNG parity. No paper edits.
The same 29 tests passed in the server runtime. Submitted `t94041-t94043`
(seeds 26/27/28), with scheduler-selected HPC placement and no required node.

### V26 result

All three tasks completed; the 12 matched results including reused controls
are complete. Actual operator sequences/configs agree with the original fixed
selector. Projection update/sample counts exactly match the active arm schedule:
33,500/27,000/24,000 updates, with proposal win fractions 64.79%/62.78%/61.22%.
Mean masked projection MSE is 0.1354/0.0932/0.0835. The mechanism was active;
candidate wins are critic estimates, not verified return gains.

Archive/final mean HV (millions): v26 3.681/2.789, original fixed v24
4.048/3.583, actor-off v24 3.989/3.444, CAPQL 3.939/3.421.
V26 is -9.06% archive / -22.15% final versus original v24, and -7.70%/-19.00%
versus actor-off. Its final actor loses all three seeds to each comparator.
Seed 26's best snapshot is only 2.936M; mode-3 archive HV is 0.297M versus
2.278M originally. Seeds 27/28 have best/final HV 2.903/1.820M and 4.647/3.613M
(62.7%/77.7% retention). Snapshots are evaluated with the final encoder.
Do not adopt v26 or tune its coefficient on these results.

Small artifacts: `results/iclr_v26_aligned_actor_20260919_summary.json` and
`results/iclr_v26_aligned_actor_20260919_diagnostic.json`. No paper changes.
Next proposed diagnostic, not yet submitted: use existing server-side models
and frozen diagnostic states/reset streams to compare critic-ranked proposals
with paired finite-horizon rollout returns, holding the continuation policy and
preference fixed. Separate candidate-value calibration from transfer/retention;
short-horizon agreement is not a full-horizon guarantee. Do not use diagnostic
mode labels or holdout returns for online selection. No new training in this step.

### Frozen candidate diagnostic

Compare only existing final v24 fixed and v26 models, seeds 26-28. Use common
states generated by the v24 fixed deterministic actor: 4 locked modes x weights
(0,1), (.5,.5), (1,0) x 2 resets, after 128 transitions. Reset seeds are
800000 + 1000*seed + 10*mode + stream, separate from locked-front evaluation.
Reconstruct each model's causal tracker from the same observed prefix.
Five mixed-preference sampled candidates are ranked under the current preference.
Compare own action, highest-Q candidate and highest-Q of five own-preference
samples, with the same Gaussian draws for the latter control.

For each first action, four paired stochastic continuations share RNG seeds.
Record discounted raw and bonus-adjusted differences at 1/32/128/512 steps,
without critic bootstrapping. Continuations are own-preference sampling or a
frozen five-candidate envelope recipe. The latter scores Q at generating
preferences, as in critic targets, but uses final online critics because target
critics were not saved; its component log-probability bonus is not the entropy
of the selected-action policy. These are fixed-policy ranking probes, not an
exact evaluation of the changing training Bellman operator. Prefix states are
off-policy for v26 and do not reconstruct training replay. No confidence claims
from treating states or rollout replicates as independent training seeds.

If Q-positive actions lose under both continuations, investigate candidate-value
bias before further actor distillation. If they help only under envelope
continuation, investigate continuation mismatch. If gains survive both, inspect
training transfer/retention next. Also compare the own-resampling control before
attributing any benefit to cross-preference search. No coefficient sweep,
training, reserved seeds, online ground-truth inputs or manuscript changes.

Four focused tests passed locally and on the server; both real model types
passed short runtime preflights. Submitted six evaluation-only tasks:
v24/v26 seed 26 `t94279/t94280`, seed 27 `t94301/t94302`, seed 28
`t94303/t94304`. All launched with scheduler-selected HPC placement, no required
node. Checkpoints remain on the server; results go to
`results/iclr_candidate_calibration_20260919`. Only summaries/logs will be pulled.

### Candidate diagnostic result

All six tasks completed: 24 states each, four paired repeats, all four horizons.
Protocol fields, paired seeds/preferences, candidate rankings and derived means
agree; unchanged first actions have zero paired effects. At horizon 512, mean
cross-candidate raw gains under own/envelope continuation are +0.664/+0.290 for
v24 and +0.615/+0.213 for v26. Among Q-positive proposals, positive observed
fractions are 23/42 and 18/42 for v24, 22/45 and 24/45 for v26. Search is not
uniformly harmful, but a positive Q difference is not a reliable acceptance test
on these probes. Versus own-preference resampling, cross-search gains are
+0.621/-0.104 for v24 and +0.910/-0.280 for v26, depending on continuation.

The balanced preference (.5,.5) exposes a specific failure: v26 predicts mean
gain +0.674 but realizes -0.908/-0.213; bonus-adjusted gains also remain negative
(-0.631/-0.255). Only 8/22 Q-positive cases improve under each continuation;
10/22 lose under both. V24 balanced gains are +0.735/+0.698, with 13/20 and
12/20 positive cases. V26's pooled positive mean is driven by the speed-only
preference (+2.730/+0.854), not uniform multi-objective improvement. Seed and
repeat variation is substantial; this is not a significance or causal-training
failure claim. States come from v24, horizons are finite, and the saved final
critics do not reconstruct the training operator or its historical targets.

Keep v26 non-adopted. Next proposed step: on the existing probe states, test
whether separate-critic advantage agreement, or the stricter rule min(Q1,Q2)
for a candidate exceeding max(Q1,Q2) for the own action, actually filters losses
while retaining useful candidates. Use scalarized critic values and fixed rules,
not a tuned margin or diagnostic mode labels. These are disagreement heuristics,
not statistical confidence bounds. Check acceptance rate and own-resampling
controls before implementing another training objective; do not fit the gate to
these rollout returns. V24 fixed remains the comparison base, not a newly proven
winner. This follow-up is not implemented or submitted yet.

Saved only `results/iclr_candidate_calibration_20260919_summary.json` (71,460
bytes) locally. No raw rollout files/checkpoints pulled, no new tasks or training
changes, and no manuscript edits.

### Fixed candidate-gate audit

Freeze two filters on the already selected candidates: both scalarized critics
must predict a strict improvement, or the candidate's smaller scalarized Q must
strictly exceed the own action's larger Q. Both also require the original
componentwise-minimum ranking gain to be positive. Rejection returns the own
action; no reranking, margin sweep or rollout-dependent gate input. Apply the
same filters to own-preference resampling. Reconstruct 128-step prefixes and
candidate draws for the existing six models, verify recorded Q values, and
reuse branch returns. Report acceptance, zero-filled overall gains, useful
gain discarded and loss removed, especially at (.5,.5). No new training.

Eight tests passed locally and remotely. Tasks `t94552-t94557` (v24/v26 for
seeds 26, 27, 28) all completed with scheduler-selected placement. Each took
3.5-3.8 seconds; all 144 reconstructed candidate Q vectors exactly match the
original records. Only 63,062 bytes of derived summary were pulled:
`results/iclr_candidate_gates_20260919_compact.json`.

Neither gate is adopted. Original/agreement/separated accept 42/32/18 cross
candidates for v24, 45/30/12 for v26. Overall 512-step raw gains (own/envelope)
are v24 +0.664/+0.290, +0.311/+0.093, +0.246/-0.158, and v26 +0.615/+0.213,
+0.451/+0.122, +0.026/-0.160. At balanced preference, v26 agreement remains
negative (-0.779/-0.161); separation changes -0.908/-0.213 to +0.098/-0.379.
Of nine balanced proposals retained by separation, eight lose under envelope
continuation. V24 separation also discards useful balanced proposals and yields
+0.492/-0.125 versus +0.735/+0.698 originally. Own-resampling controls likewise
lose pooled envelope gains under separation. Stricter disagreement filtering is
not established as a reliable acceptance rule; these remain development probes.

Next proposed algorithm ablation: use v24 fixed as the base, keep candidate
actions generated at their sampled preferences, but evaluate target critics at
the original update preference. Current code evaluates target Q at the candidate
preference before scalarizing by the update preference. Change only that Q
conditioning, retaining candidate log-probability bonuses, actor updates,
relabeling and scheduling. This tests a backup-design hypothesis, not a proven
bug fix or fully policy-consistent entropy target. Reuse seeds 26-28 and existing
v24 controls; no new gate or v26 projection. Not implemented/submitted yet.
No new training or manuscript edits in this gate-audit step.

### V27 aligned target pilot

Implemented `bamor_v27_aligned_target_fixed_archive_mix`: candidate policies
still use sampled preferences, while both target critics use the current update
preference. Preserve proposal log-probability bonuses, v24 soft vector targets,
terminated-only masks, legacy actor loss, BC, relabeling and the fixed selector.
No v26 projection or candidate gate. Log `critic_candidate_context_rule` as
`update_preference`; older methods remain `candidate_preference`.

Run HalfCheetah seeds 26-28 only, 50k interactions, mean dwell 25k, hidden 64,
eval frequency 100k, with the same 19-actor locked-full-simplex evaluation.
Use `results/iclr_v25_progress_pilot_20260914`; reuse original v24 fixed as the
primary control, with critic-off and CAPQL archive as existing secondary
controls. Compare archive/final and per-mode gains, paired seeds, the observed
minimum and balanced-preference returns. Seeds 29 onward remain reserved.
Local 42-test suite passed, including exact legacy target/RNG parity, unchanged
single-candidate updates, known multi-candidate target differences and real
training routing. This is a target-conditioning ablation, not a guarantee of
policy-consistent entropy or improved performance. No manuscript changes.

The same 42 tests passed in the server runtime. Submitted `t94563-t94565`
(seeds 26/27/28), all running with scheduler-selected HPC placement and no
required node. Models and raw results stay on the server; only logs and derived
summaries will be pulled.

### V27 result

All three runs and nine reused controls completed. Evaluation grids, reset
seeds, training mode histories and 19 snapshots match. V27 records the aligned
target rule; actual fixed-selector configs, arm sequences and family quotas
match v24. Recomputed final HV agrees with the saved results.
Mean archive/final HV (millions): v27 4.436/3.354, v24 fixed 4.048/3.583,
critic-off 4.068/2.744, CAPQL archive 3.939/3.421. V27 archive gains are
+9.57% versus v24 and +12.60% versus CAPQL, winning all three seeds against
every comparator; all 12 mode/seed archive values improve versus v24.
Final means remain -6.40%/-1.98% versus v24/CAPQL. Against v24, final changes
are +2.30%, +48.67%, -38.90% for seeds 26/27/28. Observed final minimum rises
from 2.125M to 2.848M, but three development seeds do not establish robustness.

Seed 28 best/final snapshot HV is 5.103M at 45k / 2.848M at 50k (55.8%
retained). Mode-2 HV falls from 10.468M to 1.756M; its training exposure ended
at step 41,135. Maximum evaluated speed return falls from 3,497.7 at 45k to
542.8 at 50k across the full preference grid, so this is not only a shifted
best preference. All snapshots use the final encoder, not historical encoders.
Direct balanced-preference final scalar-return differences versus v24 are
-14.49/-11.21/+95.20 across seeds, improving only 4/12 mode/seed pairs.
Do not claim a general balanced-preference repair or adopt v27 as the final actor.

Next proposed experiment: complete the target-context x actor-envelope 2x2
comparison. Reuse v24 original, v24 actor-off and v27; add only v27 with actor
candidates set to one on seeds 26-28. Keep critic search, aligned targets, BC,
relabeling, fixed schedules and evaluation unchanged. This tests whether
cross-preference actor updates interact with aligned targets in retention,
without a projection coefficient, gate or new regularizer. Existing actor-off
results are complete and matched. Not implemented/submitted yet; no new tasks.
Pulled only the 19,957-byte summary and 82,726-byte compact snapshot diagnostic:
`results/iclr_v27_aligned_target_20260919_summary.json` and `_compact.json`.
No checkpoints/raw fronts pulled and no manuscript edits.

### V27 target-context x actor-envelope completion

Implemented `bamor_v27_aligned_target_ownpref_actor_fixed_archive_mix` by reusing
the v27 trainer and existing actor-off fixed selector. Only actor candidates
become one; keep target alignment, critic search, relabeling, BC and scheduling.
Run seeds 26-28 at the same 50k/25k/hidden64/eval100k settings and result root.
Let Y_TA index aligned targets T and actor search A: reuse v24 original Y_01,
v24 actor-off Y_00 and v27 Y_11; add Y_10. Compare target effects Y_11-Y_01
and Y_10-Y_00, their interaction, archive/final/per-mode values and balanced
returns. Focus on seed-28 mode-2 retention without selecting an oracle snapshot.
All nine reused outputs are complete and matched. Local 42-test suite passed
with expanded aligned-target actor-isolation and real-route coverage.
No new loss, gate, reserved seed or manuscript changes.

The expanded 42-test suite also passed on the server. Submitted `t95245-t95247`
(seeds 26/27/28), launched with scheduler-selected HPC placement and no required
node. Raw outputs and checkpoints remain remote; only logs/summaries are pulled.

### Target-context x actor-envelope result (2026-09-19)

All three new runs completed. All 15 outputs (four factorial cells plus CAPQL)
match the frozen evaluation and training schedules. Recorded arm sequences,
family counts and configs match; only the intended actor-candidate setting
differs within each target condition. New runs record aligned Q conditioning,
consistent soft targets, terminated-only masks and own-preference actor updates.
Archive and final HV recomputed from the saved fronts agree with logged values.

Mean archive/final HV in millions, seeds 26-28:

| Target context | Actor search | Archive | Final |
| --- | --- | ---: | ---: |
| Candidate (v24) | On | 4.048 | 3.583 |
| Candidate (v24) | Off | 3.989 | 3.444 |
| Update (v27) | On | 4.436 | 3.354 |
| Update (v27, new) | Off | 3.801 | 3.341 |
| CAPQL archive | Standard | 3.939 | 3.421 |

Do not adopt the new combination: archive/final means are -14.32%/-0.37%
versus v27 actor-on and -3.52%/-2.34% versus CAPQL. New final HV by seed is
2.642/1.703/5.679M, versus v27 actor-on 4.054/3.159/2.848M. Thus actor-off
changes final HV by -34.84%/-46.08%/+99.41%; the observed minimum falls
from 2.848M to 1.703M. Seed 28 mode-2 final HV rises from 1.756M to 12.267M,
with maximum speed return 4,101.4 and the best mean-HV snapshot at 50k.
Seed 26 also peaks at 50k, but at only 2.642M; its deficit is not just a late
collapse. Seed 27 peaks at 32.5k (2.554M), retaining 66.7% at 50k.

The final-HV interaction (Y11-Y01)-(Y10-Y00) is +1.805/+1.652/-3.837M
by seed. Target alignment is beneficial with actor search for seeds 26/27,
but harmful without it; seed 28 reverses that pattern. New balanced-preference
final scalar return improves over v27 actor-on in only 2/12 mode/seed pairs;
the positive pooled difference is driven by seed 28. These are development
observations, not a validated routing rule. Snapshots still use the final encoder.

Next: stop global on/off variants. Before designing a feedback gate, run short
paired training forks at predeclared steps on development seeds, holding the
starting model, replay batches and encoder fixed across the two update choices.
Measure preference-gradient interference and independent training-rollout return
changes to test whether training-time signals distinguish helpful from harmful
actor search. Do not route by seed, true mode or locked evaluation returns.
The new result motivates this diagnostic, not an adaptive-selection claim.
No new training submitted, no reserved seeds consumed and no manuscript edits.
Only the small derived `results/iclr_v27_target_actor_factorial_20260919_summary.json`
and `_compact.json` are retained locally; raw fronts and checkpoints stay remote.

### Live paired-update diagnostic (2026-09-19)

Implemented `diagnose_update_forks.py` and a live-training callback. Rerun v27
actor-on, seeds 26-28, unchanged 50k/25k/hidden64 settings; probe at 10k, 15k,
..., 50k. Clone models, target critics and optimizer moments; exclude 1024 audit
transitions from shared read-only replay. Freeze encoder and active arm config;
only set actor candidates to one in the off branch. Perform 128 updates per
branch, recoupling RNG each update. Single-candidate arms are identity controls.

At weights (0,1), (.5,.5), (1,0), measure pre-update gradient conflict/alignment
(four draws) and post-update held-out soft-loss changes with the pre-fork critic
and existing BC. Compare base/on/off discounted raw returns at 32/128/512 steps,
three fresh paired resets each. Environment copies retain the current training
schedule, not locked modes; no true-mode inputs or diagnostic feedback to training.
Extra cost: 6,912 updates and 373,248 simulator steps. Main RNG/state are preserved;
completion also requires reproducing original episode steps, mode history and
returns (absolute tolerance 1e-6). Local focused tests and all 49 remote tests pass.

Next: inspect surrogate sign agreement and gradient relationships per preference
and seed, without fitting thresholds. These correlated probes test local effects,
not old-mode retention or final archive gains. Output:
`results/iclr_update_forks_20260919`; pull only derived summaries. No reserved
seeds or manuscript changes.

Submitted `t95592-t95594` (seeds 26/27/28), 2 CPUs and 4 GiB per task,
HPC pool placement without a required node. Only the changed trainer and the
new diagnostic/test files were uploaded; no checkpoints were transferred.
Scheduler placed all three on node007. Training logs and remote JSONs confirm
the first 10k probe completed for every seed with two Torch threads; runs continue.

### Live-fork result

All three tasks completed with zero episode-return reproduction error. Recomputed
effects match saved returns; all nine identity controls have exactly zero on/off
difference. The remaining 18 positions give 54 preference comparisons. At 512
steps, own-soft-loss feedback selects off in all 54, agreeing with actual return
direction in only 19/54. Search wins 35/54 comparisons, yet mean on-minus-off is
-4.578 (seed-macro -3.348), illustrating that win count and mean gain differ.
Per-seed mean differences are -14.273/+0.195/+4.034 for seeds 26/27/28. Seeds
26/28 have opposite directions to the previous full-training actor-off ablation.
Gradient-alignment advantage has speed-only Spearman 0.728 pooled and
0.821/0.357/0.800 within seeds (7/7/4 positions), but is positive in only 1/54
comparisons overall. Neither proxy is an accepted adaptive gate.

Next: decompose held-out loss into Q, entropy and BC, and compare deterministic
action Q with deterministic rollout labels, reusing the frozen fork outcomes.
Do not tune a threshold on these observations. These are correlated, short,
fixed-replay probes on the current training distribution, not old-mode retention
or long-run selection evidence. No new training submitted or manuscript changes.
Four summary tests pass locally/remotely; `summarize_update_forks.py` produces
`results/iclr_update_forks_20260919_summary.json` (116,794 bytes, the only pull).

### Fork-loss decomposition (2026-09-20)

Implemented `diagnose_fork_loss_components.py`. Replay the same seeds 26-28,
50k/25k/hidden64 training and nine fork positions; reproduce the same 128-update
branches. Split held-out own-policy loss into negative scalar Q, alpha-log-prob
and BC. All three branches must exactly reproduce their original total losses
before reusing cached rollout labels. Also query deterministic-action Q on both
held-out replay and the original rollout reset states, using one common pre-fork
critic. Main episode returns/schedules must reproduce again. No training changes.

Next analysis: compare component contributions, stochastic Q-only and the two
deterministic Q signals per seed/preference, without fitting thresholds. These
are still soft critics versus finite-horizon raw returns, not new independent
confirmation. Reruns cost 150k main steps plus 6,912 fork updates and 81 diagnostic
resets, but zero new rollout steps (reuse all previous outcomes). Output root:
`results/iclr_fork_loss_components_20260920`. Only source is uploaded; checkpoints
stay remote, seeds 29 onward remain reserved, and manuscript work stays paused.

All 17 focused tests pass locally and remotely. Submitted `t96111-t96113`
(seeds 26/27/28), 2 CPUs and 4 GiB each. Scheduler selected node001 for all
three; no required node or new algorithm variant.

### Fork-loss result

All three tasks completed. All 27 reconstructed cases match original losses
exactly; main episode-return errors are zero and all nine identity controls
remain zero. For the 54 active preference comparisons, mean on advantages are
Q -0.371673, entropy +0.237421, BC -0.000109, total -0.134361. Q favors off
in 54/54, whereas entropy favors on in 54/54; removing entropy or BC from the
audit loss changes zero decisions. This does not test their removal from training.

At 512 steps, sign agreement is 19/54 for total loss or stochastic Q-only,
20/54 for deterministic replay Q, and 26/54 for deterministic reset-state Q.
The last signal has agreement 15/21, 9/21, 2/12 by seed 26/27/28 and 6/18,
10/18, 10/18 by energy/balanced/speed preference. Its selected-minus-always-off
raw discounted gain is +7.836/+0.086/-0.559 by seed; pooled +2.956 hides the
seed-28 reversal. It also loses to always-on for seeds 27/28. Do not adopt it.

Next: separate operator evaluation from the training soft critic. Test an
independent raw-return evaluator with the deployed deterministic continuation,
using training-only data and counting any extra interactions. Require useful
paired return decisions across development seeds before using it for routing;
keep the existing on/off controls and reserved seeds 29 onward. This is a
proposed experiment, not an implemented v28 or a newly submitted job.

Limitations: these reuse correlated finite-horizon fork labels, not independent
confirmation; changing action/state queries does not change the soft critic's
continuation target. Loss decomposition does not establish the cause of final
policy degradation. No manuscript edits. Added a server-side summary and three
passing local summary tests; only the 120,095-byte derived
`results/iclr_fork_loss_components_20260920_summary.json` was pulled.

### Independent raw-return scoring (2026-09-20)

Implemented `diagnose_raw_return_forks.py`: reconstruct the same v27 on/off
128-update forks at 10k, 15k, ..., 50k on seeds 26-28. Score each branch with
direct raw Monte Carlo returns, deterministic continuation and gamma 0.99,
without Q queries, entropy or bootstrap. For each of three preferences, use
three new paired resets: fork seed + 20000 + repeat, disjoint from the cached
outcome resets (+10000). Old losses only verify exact branch reconstruction.
The original training schedule remains active in simulator copies. Main
training receives no diagnostic data or feedback and must reproduce exactly.

Primary comparison: positive mean paired 512-step score chooses on, otherwise
off; test against the cached independent-reset 512-step return difference.
Also report 32/128-step score prefixes against that same long outcome, without
selecting a window from results. Keep nine identity controls separate from
18 active positions and report gains versus both constant choices per seed
and preference, not only pooled sign accuracy. No threshold fitting.

Budget: 150,000 main steps + 248,832 new diagnostic steps = 398,832 interactions,
plus 6,912 fork updates and 486 diagnostic resets across three tasks. Raw
rollouts/models stay remote; pull only the derived summary from
`summarize_raw_return_forks.py`. Output: `results/iclr_raw_return_forks_20260920`.
Seeds 29 onward remain reserved; no manuscript or production algorithm edits.

Limitations: this is a direct rollout evaluator, not a learned value network.
It needs resettable simulator copies and tests decisions on the current training
distribution, not deployment under an equal interaction budget or old-mode
retention. Cached development outcomes are not fresh independent confirmation.
Only a consistent result warrants designing a budgeted online selector.

All 16 focused tests passed locally and in the remote runtime, including real
MuJoCo branch reconstruction, critic-free rollout scoring, heldout-label
independence, main model/optimizer/RNG preservation and summary direction.
Submitted `t97431-t97433` (seeds 26/27/28), requesting 2 CPUs and 4 GiB each, HPC pool
placement without a required node. Only three new scripts (21,263 bytes) were
uploaded; no automatic result-directory sync is configured.
Scheduler selected node004/node006/node005 respectively; logs confirm training
has reached approximately 6k/7k/7k steps. Results are pending.

### Raw-return scoring result

All three tasks completed with exit code zero. All 27 forks exactly reproduce
the cached losses, main episode-return error is zero, and nine identity controls
remain zero. The primary 512-step score agrees with independent-reset outcome
direction in 48/54 comparisons (88.9%), versus 19/54 for the old soft-loss score.
By seed 26/27/28, agreement is 19/21, 18/21, 11/12. Mean selected-minus-off
raw discounted returns are +10.335/+1.364/+4.226; selected-minus-on returns
are +24.608/+1.168/+0.191. These are local return differences, not final HV.

Energy/balanced/speed agreement is 17/18, 18/18, 13/18. Five of six errors
are speed comparisons; seed-28 speed selection loses 1.102 to always-on.
Its aggregate advantage over always-on is small and becomes negative if its
largest positive contribution is removed (descriptive sensitivity only).
The 128-step prefix makes exactly the same 54 choices as 512; 32 steps change
two choices. Keep the predeclared 512-step primary comparison unchanged.

Next: design a budgeted, executable routing pilot on development seeds. Specify
how preference-level scores produce one committed update branch; charge all trial
interactions to the same total budget as fixed-on/off controls. Evaluate final
and archive HV separately, including old-mode retention. This diagnostic supports
separate return-based scoring, not robust online routing or superiority to CAPQL.
Simulator-copy access and correlated development outcomes remain limitations.
No new training submitted, no production algorithm or manuscript edits. Pulled
only `results/iclr_raw_return_forks_20260920_summary.json` (87,046 bytes).

### Budgeted routing pilot

Implemented `run_budgeted_return_router.py` with two optional training hooks:
apply the persistent actor-search gate after normal arm configuration; trial
and commit a branch after learning, before archiving. At main steps 10k/20k/
30k/40k, clone full trainers and optimizer moments, hold encoder/replay fixed,
and run 128 paired updates with actor search on/off. Score deterministic raw
512-step returns at three preferences and three fresh paired resets. Use the
mean of the three scalar-return differences, not a vote; positive chooses on,
otherwise off. Commit networks, target critics and optimizers together, retain
live replay, and hold the gate until the next decision. Identity arms preserve
the prior gate. Reset seeds are 30M + 100k*seed + 100*step + 20000 + repeat;
no cached diagnostic labels or locked evaluation results enter decisions.

Five conditions, seeds 26-28: route; forced_on/forced_off with identical trials
and committed update counts but ignored scores; train_on/train_off spending
the entire budget on normal training. Trial groups use 50,000 main + 36,864
trial steps; train-only groups use 86,864 main steps. All 15 runs therefore
cost 1,302,960 training interactions in total, excluding common final evaluation.
Keep hidden64, mean dwell25k and locked-full-simplex evaluation. Align 19 archive
points by cumulative interactions, including trial cost, and archive after commit.
Each trial run additionally computes 1,024 fork updates, of which 512 are kept.

Compare routing with forced controls for feedback value and with train-only
controls for interaction efficiency. Inspect final/archived HV, per-mode and
old-mode retention separately. Trial groups share the main mode schedule;
train-only groups may see additional modes during their longer main trajectories.
Limitations: resettable simulator access, three-preference average utility,
different compute costs, development seeds, and final-encoder snapshot evaluation.
No CAPQL superiority claim or manuscript edits. Output root:
`results/iclr_budgeted_router_20260920`; models/raw fronts remain remote.
All 72 tests pass locally and remotely, including all five real short-run
conditions and the existing exact-reproduction diagnostics.

Submitted 15 tasks, seeds 26/27/28 in each group: route `t99131-t99133`,
forced_on `t99134-t99136`, forced_off `t99137-t99139`, train_on `t99140-t99142`,
train_off `t99143-t99145`. Requested 2 CPUs/4 GiB per task, no required or
preferred node, and no automatic directory downloads. Only this matrix was
passed to dispatch; placement is selected by the scheduler.
The manual dispatch client remained blocked on an existing global dispatcher
and was stopped before placement. All 15 experiments remain queued for the
active background watcher; no experiment was cancelled. Startup is not yet
confirmed. No scheduler code/state or other dispatch process was modified.

### Budgeted routing result (2026-09-20)

All 15 tasks `t99131-t99145` completed with exit code zero. Server-side
recomputation confirms per-mode HV, complete evaluation grids, routing choices,
19 matched archive budgets and the common main-schedule prefix. Each run costs
86,864 training interactions; total 1,302,960, excluding terminal evaluation.
HV below is in millions, across development seeds 26/27/28:

| Condition | Archive mean | Final mean | Minimum final |
| --- | ---: | ---: | ---: |
| route | 4.403305 | 3.516479 | 2.855555 |
| forced_on | 4.357651 | 3.630168 | 3.353349 |
| forced_off | 3.681771 | 2.704440 | 0.541777 |
| train_on | 5.122109 | 3.970266 | 2.008537 |
| train_off | 4.178420 | 2.404638 | 1.533061 |

Decision: do not adopt persistent routing. Against the same-trial forced_on
control, archive mean rises 1.05% but final mean falls 3.13%, with only 1/3
paired final wins. Against train_on, archive/final means fall 14.03%/11.43%.
Route final HV by seed is 4.550149/3.143732/2.855555M. Seed 28's old mode 2
falls from 9.842033M at the 45k snapshot to 1.111046M final (11.29% retained);
forced_on retains 40.96%, forced_off 99.94% over its observed snapshot maximum.

Four of 12 route decisions are identity comparisons, including all three at
40k; their interactions remain charged. Seed 28 chooses on at 30k and receives
no distinguishing evidence at 40k, leaving the gate on across switches at
32,642 and 41,135. The score measures a 128-update fork, but its gate controls
up to 20k subsequent training steps: this time-scale mismatch is a hypothesis
to isolate, not a proven cause of forgetting.

Next: compare impulse-only route/on/off branches, committing only the scored
128 updates and restoring the same default portfolio on the next normal step.
All three start with that same default; do not reuse this pilot's persistent
forced_off trajectory as the impulse-off control. Keep seeds 26-28, scoring,
budget and evaluation fixed; no coefficient search or seed expansion yet.
Limitations: three development seeds, resettable simulators, unequal compute
and longer exposure for train-only controls; snapshot retention uses the final
encoder. This pilot does not establish CAPQL superiority.
Added a small summary script and five focused tests, passing locally/remotely.
Pulled only the derived summary JSON (312,903 bytes), no raw fronts or models.
No new training submitted and no algorithm or manuscript edits in this step.

### Impulse-only routing pilot (2026-09-20)

Implemented `run_impulse_return_router.py`: normal steps always use the default
portfolio, including before the first trial in the forced_off group. A selected
fork commits its learned networks and optimizer moments; its actor-search gate
expires before the next normal step. Reuse the existing paired-update/scoring
implementation unchanged. Identity trials remain charged and do not carry an
earlier off decision. Protocol: `impulse_raw_return_router_v1`.

Freeze three conditions (route/forced_on/forced_off), seeds 26-28, hidden64,
50k main steps, decisions at 10k/20k/30k/40k, 128 updates per branch, 512-step
raw scoring at three preferences and three paired resets. Same RNG seeds,
36,864 trial steps, 86,864 total interactions, 1,024 computed/512 committed
fork updates per run. Nine new runs cost 781,776 training interactions, excluding
terminal evaluation. Keep the same 19 budget-aligned archives and locked grid.

Primary comparison: route versus both impulse controls on final HV, paired
seeds and the minimum; report archive HV and old-mode retention separately.
Compare against the previous persistent route to test the gate-duration change.
Fresh forced_on is a reproduction bridge to the old forced_on implementation;
investigate any mismatch before interpreting cross-pilot differences. Reuse
existing train-only controls for secondary interaction-efficiency comparisons.
Do not replace the new impulse-off control with the old persistent forced_off.
No seed expansion or manuscript changes. Results/models stay on the server at
`results/iclr_impulse_router_20260920`; retrieve only a derived summary.

The 21 focused tests pass locally and remotely, including identical pre-trial
policies across conditions, gate expiry, optimizer/RNG preservation, accounting,
post-commit archives and exact short-run forced_on reproduction. Uploaded only
the two new scripts (11,891 bytes); the shared training implementation is unchanged.
Submitted route `t99576-t99578`, forced_on `t99579-t99581`, forced_off
`t99582-t99584`, seeds 26/27/28 respectively. Each requests 2 CPUs/4 GiB, uses
the HPC pool without required/preferred placement, and disables automatic result
directory download. The background scheduler placed all nine on node001/004/005/006;
all are running, with individual logs confirming approximately 5k-7k main steps.
No further variants submitted. Await complete final/archive and retention results.

### Impulse-only result

All nine tasks completed with exit code zero. Budgets, decisions and per-mode
HV recompute correctly. Fresh forced_on exactly reproduces the previous three
seeds' episode logs, routing cases and evaluated fronts. HV in millions:

| Condition | Archive mean | Final mean | Minimum final |
| --- | ---: | ---: | ---: |
| impulse route | 4.292176 | 3.372941 | 2.819388 |
| impulse forced_on | 4.357651 | 3.630168 | 3.353349 |
| impulse forced_off | 4.439108 | 3.470554 | 2.819388 |

Route final mean is 7.09% below impulse-on, 2.81% below impulse-off, 4.08%
below persistent routing and 15.04% below the existing train_on control.
Final HV for seeds 26/27/28 is 3.901706/3.397730/2.819388M; only seed 27
beats both impulse controls. Seed 28 exactly matches impulse-off's logged
trajectory and evaluated fronts. Its old mode 2 retains 21.00% of its best
snapshot (9.920662M -> 2.083797M), versus 40.96% for forced_on.

Decision: do not adopt impulse routing. Gate expiry alone does not repair the
long-run objective/retention problem. Stop tuning this short-return binary gate.
Next investigate representation consistency: the encoder updates every training
step, while replay samples stored collection-time contexts without refreshing
their latent component. Measure stored versus current-encoder causal-history
contexts and resulting actor/critic changes on fixed historical batches before
choosing a single-factor repair. Preserve the tracker's EMA/history semantics;
do not re-encode a single transition as though it were the full context, and
do not feed true mode labels into training. This is a hypothesis, not an
established explanation of forgetting.
Nine summary tests pass locally/remotely. Pulled only the 204,350-byte derived
summary; no models/raw fronts. No new jobs or manuscript changes. These remain
three-seed development results, with final-encoder snapshot retention.

### Replay-context drift diagnostic (2026-09-20)

Implemented `diagnose_context_drift.py`, an observation-only probe on the original
v27 fixed-portfolio runs, seeds 26-28. Reproduce 50k main steps per seed against
the existing v27 episode logs; stop before terminal evaluation and model saving.
Budget: 150,000 main interactions, zero extra environment trials or learner updates.

Observe at 10k/20k/30k/35k/40k/42.5k/45k/47.5k/50k. At each point, re-encode the
available raw transition prefix with the current encoder and reconstruct the
actual continuous EMA, including across episode resets. Pre-transition latent
excludes that transition; next latent includes it. Replace only the learned
latent suffix, retaining stored nonlearned transition features; both paths use
the same audit preference.
Use 256 fixed, seeded samples from each completed 10k time bin. Ground-truth
mode counts annotate these bins only after training; they never select samples.

Report latent drift and deterministic action/Q sensitivity at energy/balanced/
speed preferences. Cross stored/refreshed actor contexts with stored/refreshed
critic contexts to separate input effects. Each bin also keeps its first-probe
context, actor actions and Q outputs as fixed anchors: later measurements at
those same inputs expose network-weight drift separately from encoding drift.
Anchors/samples stay in memory; only aggregate measurements are written.

Interpretation: this measures input sensitivity, not policy improvement or a
calibrated value advantage. Re-encoding is retrospective at the current training
prefix, not the encoder historically available at each old transition. Preserve
the negative routing results; choose a single-factor repair only after examining
all three seeds and the late-training old cohorts. No manuscript edits.
Output: `results/iclr_context_drift_20260920/seed{26,27,28}.json` on the server.

All 12 focused tests pass locally and remotely, covering frozen-encoder EMA
reconstruction, prefix invariance, latent-only replacement, input/weight-effect
separation and exact preservation of short-run parameters, optimizer state,
replay and RNG. Uploaded only two new files (19,922 bytes). Submitted
`t99598-t99600` for seeds 26/27/28, requesting 2 CPUs/4 GiB each with no node
pin/preference and no automatic result download. All three are running on
scheduler-selected node004/node006/node005, with logs confirming about 6k/6k/7k
steps. Diagnostic results remain pending; completion requires the full trajectory
reproduction check, not only the final probe having run.

### Replay-context drift results and next intervention (2026-09-20)

`t99598-t99600` completed with exit 0. All three 50k trajectories match the
reference schedule and episode returns exactly; all 90 cohort/checkpoint panels
are present. Total: 150,000 main interactions, no extra trials or updates.

At 50k, on the fixed 30k-40k time bin (256 samples, network anchor at 40k):

| Seed | Latent RMS: refreshed - stored | Actor context RMS | Actor weight RMS at fixed anchor context |
|---|---:|---:|---:|
| 26 | 0.11796 | 0.03191 | 0.24518 |
| 27 | 0.09355 | 0.01709 | 0.11396 |
| 28 | 0.11577 | 0.05371 | 0.31009 |

Action RMS pools the same samples/action dimensions across three preferences
via squared RMS, not an average of RMS. Seed 28's latent drift is similar to
seed 26's, while its policy is more context-sensitive and continues changing
at fixed inputs. Its speed-preference actor weight RMS grows from 0.24971 at
45k to 0.34694 at 50k. Existing final-encoder snapshot evaluation shows mode-2
HV falling from 10.46825M to 1.75625M over those actor snapshots. Speed-Q weight
drift at 50k is not unique to failure: 38.28074 for seed 26 versus 35.04767 for
seed 28. Neither larger Q nor a Q delta establishes better returns.

Limitations: the seed-28 bin mixes 72 mode-0 and 184 mode-2 samples; these are
not mode-2-specific action measurements. Input sensitivity and anchor-relative
weight drift are different contrasts, not additive causal shares. Small direct
input effects do not exclude accumulated training effects from stale replay
contexts. This diagnostic establishes no improvement in policy performance.

Next: one replay-cache intervention, not another binary routing variant. Keep
v27's encoder learning, actor/critic updates, portfolio and interaction budget;
refresh only replay's current/next latent suffixes every fixed 1,000 main steps,
using the current encoder and the full observed-prefix continuous EMA. Preserve
nonlearned features/preferences and the online tracker; do not reset on episode
boundaries or use mode labels. Use seeds 26-28 at 50k, the existing matched v27
control and locked-mode evaluation. Compare paired final HV, its minimum and
old-mode retention; keep archive HV separate and record added compute time.
This tests stale replay as a training cause, not as an inference-time fix.

Three new summary tests pass locally/remotely. Only the 254,433-byte derived
`results/iclr_context_drift_20260920_summary.json` was pulled; no models/raw
fronts. No new jobs submitted and no manuscript changes in this result review.

### Periodic replay-context refresh pilot (2026-09-20)

Implemented `run_replay_context_refresh.py` using the existing end-of-step hook;
the shared trainer is unchanged. Frozen protocol:
`periodic_causal_replay_context_refresh_v1`, v27 fixed portfolio, HalfCheetah
seeds 26-28, 50k steps each, dwell 25k, hidden 64, CPU. Refresh after the normal
learner/encoder updates at 5k, 6k, ..., 49k: 45 refreshes and 1,215,000 existing
transitions re-encoded per seed. The first affected learner update is step 5001.
Skip pre-learning and terminal refreshes. Preserve the online tracker; only
replay current/next latent suffixes change, reconstructing the continuous EMA
from the available raw prefix. Newly collected entries retain online contexts
until the next refresh. This is a periodic cache intervention, not per-update
consistency. Log re-encoding time separately within total training time.

Reuse original v27 controls in `iclr_v25_progress_pilot_20260914`; preserve
19 actor snapshots, final-encoder evaluation, four locked modes, 21 preferences
and three paired resets. Check the pre-intervention episode returns, mode
schedule and evaluation settings against each control. New training budget:
150,000 interactions and zero extra trial interactions/learner updates; terminal
evaluation is unchanged and never feeds training. Output stays separate under
`results/iclr_replay_context_refresh_20260920` on the server.

Decision: advance only if final mean and minimum HV both improve and seed 28's
old-mode retention recovers; inspect paired results for all seeds and report
compute overhead. Archive-only or Q-only gains do not qualify. These are
development seeds, not confirmation evidence. Do not tune the interval against
these results or change the manuscript before assessing the completed pilot.

Eight focused tests pass locally and remotely, including exact latent-only
mutation, continuous-history reconstruction, RNG/network/optimizer preservation,
unchanged online tracker, matched short-run prefix, next-update effect, archive
timing and paired evaluation. Uploaded only the runner and tests (13,358 bytes).
Submitted `t99603-t99605` for seeds 26/27/28 with 2 CPUs/4 GiB each, no node
pin/preference and no automatic result download. All are running on
scheduler-selected node004/node006/node005; logs confirm refreshes at 5k and 6k.
No performance result yet; no manuscript edits.

### Replay refresh outcome (2026-09-20)

All three training/evaluation runs finished, but `t99603-t99605` exited 1 in
postprocessing: `paired_protocol_check` compared live tuple mode-history entries
with JSON lists. Fixed with the existing `np.array_equal` convention and added
live-versus-JSON coverage plus rejection of a genuinely different schedule.
Saved histories match exactly. The server-side summary independently validates
all 45 refreshes, pre-intervention returns and four 19x21 grids, then recomputes
HV. Original result/trace files and failed task states remain unchanged; no
training or evaluation rerun was needed.

| Final HV (M) | Stored context | Refreshed context | Change |
|---|---:|---:|---:|
| Seed 26 | 4.054126 | 5.073773 | +25.15% |
| Seed 27 | 3.158804 | 2.932878 | -7.15% |
| Seed 28 | 2.847842 | 4.993631 | +75.35% |
| Mean | 3.353591 | 4.333427 | +29.22% |
| Minimum | 2.847842 | 2.932878 | +2.99% |

Archive mean improves 5.64%. Seed 28 mode-2 final HV rises 1.756247M ->
9.436752M; final/best snapshot retention rises 16.78% -> 91.97%. All three
frozen development gates pass. Refresh compute is 4.94-5.07 seconds per run,
0.42-0.53% of measured training time, with no additional training interactions.

Limitations: seed 27 loses mostly on unseen mode 3 (2.284520M -> 1.233487M).
Seed 28 mode-2 snapshot HV still drops to 2.699730M at 47.5k before recovering
at 50k. A posthoc 45k/47.5k/50k average across modes/seeds improves only
3.571168M -> 3.718178M; its worst-seed average worsens 2.537375M -> 1.924636M.
These final-encoder snapshot diagnostics show residual instability, not online
performance or extra independent samples. Historical wall times are not a
compute-matched benchmark. No robustness or baseline-superiority claim yet.

Next: freeze this variant and compare stored/refreshed contexts on five fresh
paired seeds 29-33, 50k steps each (10 runs, 500k training interactions). Keep
final HV primary and prerecord the same late-snapshot average/minimum as
secondary stability measures. Do not retune the interval, reconnect adaptive
routing or expand to 20 seeds before that check. Six focused tests pass locally
and remotely. This turn only fixes postprocessing and records the result; no
new jobs or manuscript edits. Derived artifact:
`results/iclr_replay_context_refresh_20260920_summary.json`.

### Fresh-seed replay-context confirmation (2026-09-20)

Frozen `replay_context_confirmation_v1`: seeds 29-33, stored/refreshed contexts,
50k steps per arm, dwell 25k, hidden 64, fixed v27 portfolio. Ten independent
jobs, 500,000 planned training interactions; no extra trials or updates. Reuse
the unchanged 5k-49k/1k refresh and final-encoder 19-snapshot locked evaluation.
Both arms run without reading the other's results; the summary checks all five
paired mode schedules, warmup episode returns and evaluation grids afterwards.
New output: `results/iclr_replay_context_confirmation_20260920/{stored,refresh}`.

Primary: final HV averaged across four separately evaluated modes; report all
paired differences, wins, mean and minimum across seeds. Secondary: per-seed
mean and minimum over the fixed 45k/47.5k/50k snapshots, then mean and minimum
across seeds for both quantities. Proceed to larger confirmation only if final
mean/minimum improve and none of those four secondary summaries regress.
Also retain per-mode exposure/old-mode retention, archive HV and refresh time.
No interval tuning, adaptive routing, mixing with seeds 26-28 or manuscript
changes. Five pairs are a replication check, not a significance or broad
robustness claim.

Accounting correction discovered during scheduler preflight: the prior three
postprocessing failures each triggered three automatic retries, all reaching
50k. The pilot therefore consumed 600,000 training interactions across 12
attempts, not 150,000 in actual execution. Its statistical sample remains
three seeds. The latest saved archive/final HV matches the reviewed summary.
Failed attempt IDs and logs are preserved; resolve the old application-error
escalations without retrying those completed experiments.

Eight focused tests pass locally/remotely. Uploaded only three new files
(11,874 bytes); shared training and refresh code are unchanged. Resolved
`t99628-t99630` escalations as validated results requiring no retry, preserving
failed task histories. Submitted `t99651-t99660`, ordered stored/refresh for
each seed 29-33, requesting 2 CPUs/4 GiB per task with no node pin/preference
and no result auto-download. All ten are running on scheduler-selected
node003-node006; log checks show 6k-7k steps and active refreshes in all five
treatment runs. Confirmation results remain pending.

### Fresh-seed confirmation outcome (2026-09-20)

`t99651-t99660` all completed with exit 0, with no retry descendants: 500,000
training interactions. All five schedule/warmup/evaluation pairs pass; full
four-mode 19x21 grids and all refresh traces validate, and HV is recomputed.
Pulled only `results/iclr_replay_context_confirmation_20260920_summary.json`
(156,573 bytes); no models or raw fronts.

| Final HV (M) | Stored | Refresh | Change |
|---|---:|---:|---:|
| Seed 29 | 5.339719 | 5.510677 | +3.20% |
| Seed 30 | 5.115327 | 3.864873 | -24.45% |
| Seed 31 | 2.056414 | 2.583208 | +25.62% |
| Seed 32 | 5.227232 | 5.049799 | -3.39% |
| Seed 33 | 2.660999 | 3.335050 | +25.33% |
| Mean | 4.079938 | 4.068721 | -0.27% |
| Minimum | 2.056414 | 2.583208 | +25.62% |

Three of five final-HV pairs improve, but the mean does not. Archive mean falls
2.14%. All four prespecified late-snapshot summaries regress: mean of per-seed
late means -6.25%, minimum of those means -30.29%, mean of per-seed late minima
-9.03%, and minimum of those minima -84.21%. The final-minimum gain therefore
does not establish stable improvement. Do not adopt unconditional refresh or
expand it to 20 seeds; preserve it as an ablation, without pooling development
and confirmation seeds or retuning the interval on this result.

Seed 30 loses 1.404310M/3.283198M on seen modes 0/2. Seed 31's refresh
snapshots have mean HV 0.154687M/0.153991M/2.583208M at 45k/47.5k/50k;
its final gain includes +1.584986M on unseen mode 2, while old mode 0 loses
0.422299M. This is not uniform old-scenario retention.

Limitations: the snapshot metrics use the final encoder, so they cannot locate
an online collapse or identify its cause. Next diagnostic: reproduce seed 31's
two arms (100k training interactions), retain contemporaneous encoders at the
three late snapshots, and compare the same actor with contemporaneous versus
final encoder on identical locked-mode resets. Record extra evaluation
interactions separately and feed no evaluation back into training. This can
separate snapshot representation mismatch from actor degradation; it cannot
erase seed 30's final-actor loss. Seeds 29-33 are now observed; reserve new seeds
for any subsequently modified method. No new jobs, training-code changes or
manuscript changes in this result review.

### Frozen snapshot/encoder diagnostic (2026-09-20)

Reproduce seed 31 independently for stored and refreshed replay. Capture actor
and encoder after the normal updates at 45k/47.5k/50k; stop before the usual
archive evaluation. Require the full episode-return sequence and mode schedule
to reproduce each corresponding confirmation run before interpreting results.
For each actor, compare its contemporaneous encoder with the final encoder on
the existing 4 locked modes, 21 preferences and 3 matched resets. Reproduce the
saved final-encoder fronts within 1e-6 absolute return error. Evaluate the shared
50k/50k cell once. No training/evaluation feedback or algorithm change.

Budget: 2 x 50k training interactions, plus 5 x 4 x 21 x 3 x 1000 = 1.26M
diagnostic evaluation steps per arm (2.52M total maximum); report actual steps.
The three one-step environment initialization probes per arm are counted
separately. Only small actor/encoder snapshots and results are saved remotely;
no replay/optimizer dumps or local checkpoint pulls. Scheduler placement is
automatic across the available HPC pool.

Interpretation: contemporaneous-encoder recovery would implicate representation
mismatch in these archived troughs. Continued low returns would leave policy
degradation/update dynamics as the next target, not prove them causal. Compare
all modes and both arms, without choosing a favorable reset or time. This is a
single-seed mechanism diagnostic, not a successful refresh confirmation; seed
30's final loss and the failed five-seed stability gate remain unchanged.

Implementation: `diagnose_snapshot_encoders.py`; two tests in
`test_snapshot_encoders.py` pass locally and in the remote CPU environment.
The short MuJoCo test reproduces both training arms and their final-encoder
fronts exactly, checks actor capture, shared-cell reuse and rollout counts.
Submitted `t99661` (stored) and `t99662` (refresh), each CPU=2/RAM=4096MB,
with no required/preferred node or automatic result download. Scheduler placed
them on node006/node005; startup logs confirm training. Full 50k reproduction
and paired diagnostic results remain pending.

### Snapshot/encoder outcome (2026-09-20)

`t99661`/`t99662` completed with exit 0 and no retry descendants. Both complete
training trajectories and all six final-encoder fronts reproduce exactly
(maximum return error 0). All ten 4x21 paired cells validate and HV recomputes.
Actual budget: 100,000 training steps, 2,520,000 diagnostic evaluation steps,
plus six environment-initialization probe steps. Pulled only the 36,585-byte
`results/iclr_snapshot_encoders_20260920_summary.json`; no models/raw fronts.

| Mean mode HV (M) | Final encoder | Contemporaneous encoder | Change |
|---|---:|---:|---:|
| Stored, 45k | 1.116801 | 1.155283 | +3.45% |
| Stored, 47.5k | 0.975007 | 1.006589 | +3.24% |
| Refresh, 45k | 0.154687 | 0.153951 | -0.48% |
| Refresh, 47.5k | 0.153991 | 0.154276 | +0.18% |

The refresh trough is not rescued by the contemporaneous encoder. Its maximum
speed-objective return across all four modes and 21 preferences remains only
5.2323/4.7500 at 45k/47.5k. The shared 50k cell is unchanged: stored 2.056414M,
refresh 2.583208M. Historical/final encoder mismatch is not the main explanation
for this seed's late deterministic-policy trough; it does not repair the failed
five-seed confirmation or justify adopting unconditional replay refresh.

Next: use the already saved paired networks to compare stochastic versus
deterministic execution at these three times, recording action means/standard
deviations under fixed preferences, locked-mode resets and action-noise seeds.
This needs no new training. Limitations: the existing stochastic training returns
use different preferences, contexts and mode trajectories, so their positive
speed returns do not yet establish an execution-policy mismatch. Keep the
original deterministic evaluation as the primary result; a stochastic diagnostic
is not a replacement score. No new jobs or manuscript changes in this review.

### Frozen execution-policy diagnostic (2026-09-20)

`diagnose_execution_policy.py` loads seed 31's stored/refreshed actor and
contemporaneous encoder at 45k/47.5k/50k. Six independent jobs compare
`tanh(mean)` with `tanh(Normal(mean, std).sample())`, using unchanged online
episode trackers, all four locked modes, 21 preferences and three matched
resets. Action-noise seeds are `31000000 + 1000*mode + 10*weight_index + episode`,
shared across conditions/times and isolated from environment-reset RNG.
Deterministic replay must reproduce its saved contemporaneous-encoder front
within 1e-6 before interpreting the stochastic comparison.

Record per-episode raw returns/lengths and per-preference action statistics:
pre-tanh Gaussian mean/spread, deterministic/executed action RMS, noise
displacement, saturation, and empirical per-dimension action mean/std.
Empirical action std mixes state visitation and noise; pre-tanh std is not
post-tanh spread. Compare both objectives and same-preference scalarized return,
not just the best speed preference or archive union. No critic or updates.

Budget: zero training; 6 x 2 x 4 x 21 x 3 x 1000 = 3.024M maximum evaluation
steps, plus six initialization probes. Deterministic reruns collect the action
statistics absent from prior fronts. Log actual counts and retain individual
episode returns. Output: `results/iclr_execution_policy_20260920`, server only.

Interpretation: recovery under sampled actions would implicate execution-mode
sensitivity in these snapshots, not establish an algorithmic fix or reverse
the failed refresh confirmation. No recovery redirects attention to training
updates. Three reset/noise pairs on one seed are a diagnostic, not independent
training replications. Keep primary deterministic scores and the manuscript
unchanged; scheduler chooses among available HPC nodes without a pin/preference.

Three tests in `test_execution_policy.py` pass locally and remotely: native
action equivalence, statistics, deterministic-front reproduction, fixed-noise
repeatability, unchanged networks, per-episode returns and interaction counts.
All six remote network pairs load successfully. Submitted `t99663-t99668`,
ordered stored 45k/47.5k/50k then refresh 45k/47.5k/50k, CPU=2/RAM=2048MB each.
All are running on scheduler-selected node003-node006 with no automatic result
download; logs show the first locked-mode evaluation. Results remain pending.

### Execution-policy outcome (2026-09-20)

`t99663-t99668` all exited 0, without retries. All six deterministic fronts
reproduce exactly. Validated all 12 grids, 3,024 individual episodes, action
noise seeds, pooled-step counts and recomputed HV. Actual budget: zero training,
3,024,000 evaluation steps plus six initialization probes. Only the 162,519-byte
`results/iclr_execution_policy_20260920_summary.json` was pulled; no checkpoints.

| Mean mode HV (M) | Deterministic | Stochastic | Change |
|---|---:|---:|---:|
| Stored, 45k | 1.155283 | 0.628629 | -45.59% |
| Stored, 47.5k | 1.006589 | 0.531953 | -47.15% |
| Stored, 50k | 2.056414 | 2.010280 | -2.24% |
| Refresh, 45k | 0.153951 | 0.658614 | +327.81% |
| Refresh, 47.5k | 0.154276 | 0.495460 | +221.15% |
| Refresh, 50k | 2.583208 | 1.662189 | -35.65% |

Refresh's two troughs recover some movement, but same-preference utility improves
in only 28/84 and 26/84 mode/preference cells. Mean `w dot R(w)` falls from
-8.821/-14.946 to -245.544/-264.886. On the sampled trajectories, mean-action RMS
is 0.10670/0.08797 while noise-displacement RMS is 0.52061/0.53094, pooled via
squared RMS and step weights. Do not adopt stochastic execution as a repair.

Next single-factor training pilot: keep stored-context v27's critic, encoder,
exploration, entropy/BC terms and fixed portfolio; replace its actor value term
`Q_selected` with `0.5*Q_selected + 0.5*Q_mean`, where `Q_mean` evaluates the
same-preference deterministic action. The total value coefficient stays one.
Freeze controls before running and judge deterministic raw-return performance,
including late stability, not the training Q score. This is a proposed training
intervention, not a new evaluation score or a demonstrated improvement.

Limitations: one training seed and three reset/noise pairs; stochastic and
deterministic trajectories visit different states. This does not isolate the
training cause or reverse the failed five-seed refresh confirmation. No new jobs,
algorithm changes or manuscript edits in this result review.

### Frozen mean-value actor pilot (2026-09-20)

`bamor_v28_mean_value_fixed_archive_mix` changes the actor value contribution
to `0.5*Q_selected + 0.5*Q_mean` in both single-candidate and envelope updates.
`Q_mean` uses `tanh(mu(s,c_w))` at the current update preference, with gradients
through that action. Preserve candidate selection, entropy coefficient, BC,
critic targets, encoder, replay storage, exploration and the fixed v27 portfolio.
No replay refresh, variance clamp, adaptive routing or coefficient sweep.

Run HalfCheetah seeds 29-33 at 50k/25k dwell/hidden64/CPU, reusing the five
completed stored-context v27 controls in `iclr_replay_context_confirmation_20260920`.
These seeds are observed development cases, including the problematic 30/31,
not independent confirmation. Check warmup returns, mode/operator schedules and
evaluation pairing; require 45,001 mean-value updates covering both paths.
New training budget: 250k steps. Unchanged deterministic 19-snapshot evaluation
costs up to 23.94M additional steps in total, counted separately, plus ten
initialization probes. Results/models stay remote; no automatic downloads.

Primary: paired final mean-mode HV, with mean and minimum over all five seeds.
Secondary: each seed's mean/minimum over 45k/47.5k/50k, then mean/minimum over
seeds. Advance only if final mean and minimum improve and none of the four
secondary aggregates regress. Report every pair, per-mode returns/retention,
archive HV and compute separately. Keep the final encoder convention; these
snapshot curves are not online performance. An increase in training Q is not
evidence of success. A positive pilot would still require fresh confirmation.

Implementation: `run_mean_value_pilot.py`; output
`results/iclr_mean_value_pilot_20260920`. The shared trainer's legacy objective
retains its original formula; only v28 overrides it. Shared terminal logs now
also record actual main/evaluation environment-step counters. No paper changes.

Local and remote: 28 tests passed, including value/entropy gradients, all 18
operators in 2D/3D, unchanged first critic updates/RNG, legacy actor equivalence,
and a real paired HalfCheetah run exercising both actor paths and step accounting.
Submitted on 2026-09-21: `t99669-t99673`, respectively seeds 29-33,
CPU=2/RAM=4096MB each, scheduler-selected node001-node007 with no node preference
or automatic result downloads. Results pending.

### Mean-value actor outcome (2026-09-21)

`t99669-t99673` all exited 0 on their first launch. All five warmup/schedule/
evaluation pairs pass. Recomputed HV from all 10 four-mode 19x21 grids;
each new run has 45,001 mixed-value updates covering both actor paths.
Actual new budget: 250,000 training, 23,940,000 evaluation steps and ten
initialization probes. Only the 179,973-byte derived
`results/iclr_mean_value_pilot_20260920_summary.json` was pulled.

| Final mean-mode HV (M) | V27 stored | V28 mean-value | Change |
|---|---:|---:|---:|
| Seed 29 | 5.339719 | 4.438357 | -16.88% |
| Seed 30 | 5.115327 | 3.760511 | -26.49% |
| Seed 31 | 2.056414 | 3.557694 | +73.00% |
| Seed 32 | 5.227232 | 5.092257 | -2.58% |
| Seed 33 | 2.660999 | 1.545467 | -41.92% |
| Mean | 4.079938 | 3.678857 | -9.83% |
| Minimum | 2.056414 | 1.545467 | -24.85% |

All six frozen gates fail. Late mean/minimum aggregates change by -10.68%,
-20.95%, -9.07%, -36.33% in the registered order; archive mean falls 14.66%.
Seed 31 improves in every mode, but seed 33 becomes the worst final run.
For seed 33, mode-3 final HV falls 1.715210M -> 0.149295M; maximum speed
return over the preference grid falls 551.964 -> 0.440. Its archive mean-mode
HV is only 1.879362M versus v27's 3.943023M, so choosing an earlier saved actor
cannot close that archive gap. This is not evidence for robust improvement.

Keep v27 as this comparison's control; retain v28 as a failed development
ablation, with no 20-seed expansion or coefficient retuning. Next proposed
diagnostic: on fixed seed-31/33 models, test whether the mean-action Q-gradient
direction agrees with independently measured return changes under deterministic
continuation. Freeze the perturbation and charge rollout steps before running;
do not reuse the training Q score as the outcome. No new jobs or paper edits.
Eight summary/regression tests pass locally and remotely. These observed seeds
and final-encoder snapshot curves do not identify the causal training failure.

### Frozen mean-action gradient diagnostic (2026-09-21)

`diagnose_mean_action_gradient.py` uses final v27/v28 models for observed seeds
31/33, all four locked modes, weights (0,1), (0.5,0.5), (1,0), three resets.
Reset rule: `41000000 + seed*1000 + mode*10 + repeat`, shared across methods
and preferences, disjoint from prior evaluation resets. Build the usual online
contexts over 128 deterministic steps, then branch at the same observed state.

Compute the gradient of `w dot min(Q1,Q2)(s,tanh(z),c_w)` with respect to the
pre-tanh mean output z. Normalize its L2 norm and perturb z by +/-0.05 once;
zero gradients produce zero perturbations. Continue the unchanged deterministic
policy with separately updated contexts. Four branches: base, plus, minus and
an uncopied identity continuation. Restore fork physics and MuJoCo integration
state explicitly: Gym deepcopy alone resets the simulator. Require exact base
versus uncopied returns and final observations in every case.

Primary: discounted raw 512-step plus-minus and plus-base return contrasts,
gamma 0.99, without entropy or bootstrap. Record 32/128-step prefixes as
secondary diagnostics, not selectable horizons. Average three resets within
each mode/preference cell; report all 12 cells separately for each model/seed,
including finite Q contrasts, actual action displacements and zero/non-ascent
Q cases. Positive Q but negative return contrasts challenge this local proxy;
positive agreement does not authorize routing or establish a training cause.

Budget: four jobs, each 36 cases * (128 + 4*512) = 78,336 diagnostic steps;
313,344 in total plus four environment initialization probes. Zero training or
optimizer updates. No coefficients/horizons fitted to these results, no new
algorithm version, no manuscript edits. Outputs stay remote under
`results/iclr_mean_action_gradient_20260921/{v27,v28}/seed{31,33}.json`.
Limitations: two observed seeds, finite-horizon local action interventions, not
parameter gradients; visited states are paired within, not across, models.

Seven focused tests passed locally and remotely: analytic gradient/chain rule,
zero direction, real simulator identity/native-evaluator equivalence, no Q
queries during continuation, unchanged networks, complete step accounting and
outcome-based summaries. All four source models load and match their 50k archive
actors. Submitted `t99762-t99765`, ordered v27 seed31/33 then v28 seed31/33,
CPU=2/RAM=2048MB each, scheduler placement across node001-node007 without node
pin/preference or result auto-download. Results pending.

### Mean-action gradient outcome (2026-09-21)

All four tasks `t99762-t99765` exited 0 on their first launch. Validated every
case/reset/weight/horizon and recomputed summaries; all 144 copied-base versus
uncopied continuations match exactly. Actual cost: 313,344 diagnostic steps,
four initialization probes, zero training. Pulled only the 103,040-byte derived
`results/iclr_mean_action_gradient_20260921_summary.json`.

Every model/seed has positive mean finite Q plus-minus contrasts in all 12
mode/preference cells. Raw discounted 512-step results are:

| Model/seed | Positive return cells / 12 | Mean plus-minus | Mean plus-base |
|---|---:|---:|---:|
| V27 / 31 | 6 | 1.390404 | 1.344799 |
| V27 / 33 | 7 | 0.618602 | 0.089987 |
| V28 / 31 | 7 (one zero) | 0.028583 | -0.187826 |
| V28 / 33 | 7 | 0.000497 | 0.201444 |

In total, 27/48 cells agree in sign, 20 reverse and one is zero. Energy-only
preferences have 5 positive, 10 negative and one zero cell; balanced and speed
each have 11 positive and five negative. The 32/128-step counts are 26/27
positive, respectively; the primary horizon remains 512. V28 seed31's mode-2
balanced cell predicts Q contrast +0.047139 but returns -1.262865 versus the
opposite direction and -1.693945 versus base. V28 seed33's mode-2 speed cell
is better than the negative perturbation (+3.654812) but worse than base
(-4.785170): directional ranking is not itself an improvement guarantee.

Do not adopt the current soft-Q mean-action direction as a generally reliable
repair. Next proposed experiment: learn an independent raw-return critic with
the deterministic continuation and no entropy, initially shadow-only, and test
its local directions against held-out rollout outcomes before coupling it to
the actor or routing. Charge all new interactions; keep original-Q controls.
Limitations: some effects are nearly zero, so sign counts are descriptive, not
significance tests. These local visited-state interventions do not identify
the cause of the failed training pilot. No new algorithm or paper changes.

### Frozen shadow return-critic pilot (2026-09-21)

`diagnose_return_critic.py` fits independent twin raw-return critics for each
frozen v27/v28 seed31/33 policy. This is fixed-policy TD evaluation, not a new
BAMOR actor or routing variant. Collect 5 episodes * 1000 steps at each of the
four modes and three existing diagnostic preferences: 60,000 transitions/model.
Behavior adds fixed N(0,0.1^2) pre-tanh action noise; stored target actions are
the frozen policy's deterministic means at the observed next state/context.

Use fresh hidden64 twin Q networks, Adam 3e-4, batch256, 20,000 updates,
target tau=0.005 and gamma=0.99. Target is raw vector reward plus discounted
minimum target-Q under that deterministic action, without entropy, envelope
search or preference relabeling. Bootstrap through time limits, not true
termination. Input statistics and reward scales use training data only;
output Q remains in raw reward units. No early stopping or held-out tuning.

Train reset seeds: `42000000 + seed*1000 + mode*10 + repeat`; held-out resets
use base 43000000 and three repeats. Action-noise seeds use
`44000000 + seed*10000 + mode*1000 + weight_index*100 + repeat`.
Finish fitting and save the critic before creating the held-out environment.
No original policy/encoder/soft-Q parameter updates or mode-label inputs.

Held-out protocol retains warmup128, pre-tanh perturbation L2=0.05, deterministic
continuation, horizons32/128/512 and primary512. At each common state, evaluate
base, old-Q plus/minus, raw-Q plus/minus and an uncopied identity continuation.
Require exact copied-base/native agreement. Compare both critics' predictions
on both action pairs; report contrast calibration, not just absolute Q offsets.
Primary evidence is actual plus-base return and plus-minus direction per
model/seed/mode/preference. A new shadow replication is warranted only if all
four model/seed mean plus-base gains are positive and no worse than old-Q, and
the total positive plus-minus cell count exceeds old-Q. This does not authorize
actor/routing deployment; lower TD loss alone never passes the check.

Budget across four jobs: 240,000 critic-data steps + 460,800 held-out steps =
700,800 interactions, plus eight initialization probes and 80,000 critic-only
optimizer steps. Zero actor updates. Results and critics stay remote under
`results/iclr_return_critic_20260921`; no model/dataset downloads.
Limitations: observed seeds, fresh balanced-mode data and new normalization
confound attribution to entropy removal alone; not an equal-resource algorithm
comparison. Continuing TD values are compared with finite512 rollout contrasts.

Eight tests pass locally and remotely: raw-unit gradients, terminal/time-limit
targets, deterministic train-only fitting, no old-Q queries during collection,
actor-mean bootstrap actions, fit-before-test ordering, unchanged original
networks, copied/native continuations and interaction accounting. All four
source checkpoints load. Submitted `t99772-t99775`, ordered v27 seed31/33 then
v28 seed31/33, CPU=2/RAM=2048MB each, no required/preferred node or automatic
result downloads. Results pending; no actor or manuscript changes.

### Shadow return-critic outcome (2026-09-21)

`t99772-t99775` all exited 0 on their first launch. Recomputed all four
summaries after validating the complete collection/test grids, fixed fit
settings, reset separation and accounting. All 144 copied/native continuations
match. Actual budget: 240,000 data + 460,800 evaluation steps, eight probes and
80,000 critic-only updates; zero actor updates. Pulled only the 77,866-byte
`results/iclr_return_critic_20260921_summary.json`; models stay remote.

| Model/seed | Soft-Q mean plus-base | Raw-Q mean plus-base | Soft/raw positive plus-minus cells |
|---|---:|---:|---:|
| V27 / 31 | 0.000809 | 0.315467 | 3 / 10 |
| V27 / 33 | -0.423506 | -0.626560 | 7 / 3 |
| V28 / 31 | -0.141295 | 0.211221 | 6 / 8 |
| V28 / 33 | -1.722627 | -2.007414 | 7 / 11 |

The positive-direction count improves 23/48 -> 32/48, but both all-group
plus-base gates fail on seed33. Do not expand this critic or connect it to the
actor/router. V28/33's mode2 speed cell loses 15.929064 versus base in the raw
positive direction and 25.311102 in the negative direction; all three positive
perturbations lose. Its mode0 speed cell loses 8.252171 on average, dominated
by one reset (-24.766849). Thus 11/12 positive plus-minus cells do not establish
improvement over the original action. Both seed33 groups also have negative
raw plus-base means at the secondary 32/128 horizons.

Next proposed diagnostic: hold these critics/policies fixed and compare a
preregistered smaller-radius ladder around the same states, retaining base and
both directions, to distinguish a wrong local direction from finite-step
curvature. Report every radius, without selecting a winning deployment step
size or revising this failed gate. Freeze its budget before launch. No new jobs
or manuscript changes. Ten tests pass locally and remotely, including missing
artifacts, stale summaries and a single failed group that pooling would hide.

Limitations: observed development models and three resets per cell; sign counts
are descriptive. Added balanced data/normalization confound entropy attribution,
and continuing TD values differ from finite-horizon returns. All four normalized
TD-loss traces rise over fitting, but moving-target training loss alone does not
identify the cause of the return-direction failures.

### Frozen action-radius diagnostic (2026-09-21)

`diagnose_action_radius.py` keeps the four v27/v28 seed31/33 actors, encoders,
soft critics and saved independent raw critics fixed. Retain all four modes,
three preferences and three original reset states per cell. Reuse the recorded
0.05 radius; evaluate 0.01 and 0.002 along each critic's unchanged normalized
pre-tanh mean-output gradient, perturbing only the first action. At each new
radius rerun warmup128, base, soft/raw plus/minus and uncopied identity; require
exact historical-base return replay and current copied/native agreement.
Also check anchor mean outputs, gradients and base Q against the source record.

Primary remains discounted raw horizon512, gamma0.99; retain 32/128 prefixes.
Report every radius and cell: plus-base, minus-base, plus-minus, centered slope
`(R_plus-R_minus)/(2*epsilon)` and curvature
`(R_plus+R_minus-2*R_base)/epsilon^2`. A cell is consistent with an oversized
step only when its 0.05 plus-base mean is negative but both smaller radii have
positive plus-base means and positive centered slopes. A persistent direction
mismatch requires negative return slopes and positive Q contrasts at both
smaller radii. Apply both descriptive rules to raw and soft critics, averaging
three resets within each cell; keep mixed/zero cases unclassified.

Budget: four jobs * 36 cases * 2 new radii * (128 + 6*512) = 921,600 new
evaluation steps, plus four initialization probes. No collection, fitting,
actor/critic/encoder updates or new policy version. Existing 0.05 interactions
are reused, not charged again. Results stay remote under
`results/iclr_action_radius_20260921`; no checkpoint downloads.

Limitations: these are observed development states, not fresh confirmation.
Finite-radius slopes need not equal infinitesimal derivatives in contact
dynamics; descriptive sign counts are not significance tests. No winning-radius
selection, deployment approval or revision of the failed shadow-critic gate.

Thirteen tests pass locally and remotely, covering checkpoint normalization and
action gradients, analytic curvature versus reversed direction, exact historical
replay, unchanged networks and new-only step accounting. All four production
sources load. Submitted `t99776-t99779`, ordered v27 seed31/33 then v28 seed31/33,
CPU=2/RAM=2048MB each, scheduler placement across node001-node007 with no
required/preferred node or result auto-download. Results pending.

### Action-radius outcome (2026-09-21)

`t99776-t99779` all exited 0 on their first launch. Recomputed every radius/cell
summary after checking the complete grid, actual perturbation radii, reused
source records and step accounting. All 288 new-radius cases match their
historical bases and copied/native continuations. Actual cost: 921,600 new
evaluation steps and four probes, zero training/updates. Pulled only the
314,734-byte `results/iclr_action_radius_20260921_summary.json`.

| Raw-critic mean plus-base return | Radius 0.05 (reused) | 0.01 | 0.002 |
|---|---:|---:|---:|
| V27 / 31 | 0.315467 | -0.363233 | -1.439616 |
| V27 / 33 | -0.626560 | 0.160680 | -0.198072 |
| V28 / 31 | 0.211221 | -0.038721 | 0.071349 |
| V28 / 33 | -2.007414 | -1.294293 | -1.738924 |

Under the frozen descriptive rules, raw critics have 3/48 cells consistent
with an oversized step and 9/48 with a direction mismatch at both small radii;
soft critics have 1/48 and 20/48, respectively. The remaining cells are
unclassified. V28/33 has 12/12 positive plus-minus cells at radius0.01 but
negative mean plus-base return. Its mode2 speed cell loses 15.929064, 15.220371
and 19.429972 versus base at the three radii, with both signs below base in
every radius. Shrinking the perturbation does not provide a consistent repair.

Next proposed experiment: test a learned paired-advantage evaluator using
finite-horizon Monte Carlo candidate-minus-base targets, without bootstrapping,
while keeping the actor fixed and retaining no-change as the reference. Use
separate training/validation resets and test prediction of unseen return gains;
freeze the data/fit/evaluation budget before launch. This differs from the
earlier direct-rollout fork router, whose equal-budget training pilot failed.
No new jobs, actor/routing integration, radius selection or manuscript edits.
Five focused tests pass locally and remotely.

Limitations: observed models/states, three resets per cell and finite radii,
not independent confirmation or infinitesimal-gradient identification. Counts
are descriptive; one raw mismatch cell has effects around 1e-8 at radius0.002.

### Frozen paired-advantage shadow pilot (2026-09-21)

`diagnose_paired_advantage.py` retains the four v27/v28 seed31/33 policies,
encoders and both frozen critic types. Candidate set: unchanged mean action,
soft-Q plus/minus and raw-Q plus/minus, all at pre-tanh radius0.05. No radius
selection or learned candidate generation. After deterministic warmup128, every
candidate receives the same-state 512-step deterministic continuation, gamma0.99;
keep 32/128 prefixes and copied/native identity controls.

Per model collect 16 training resets per mode/preference (192 anchors, 768
nonbase labels), then eight validation resets per cell (96 anchors). Train
resets: `47000000 + seed*10000 + mode*100 + repeat`; validation uses base48000000.
The same reset schedule is shared across preferences/methods, not mixed across
splits. No earlier diagnostic outcomes become labels in this pilot.

Learn scalar paired advantage `w dot (R_candidate - R_base)` directly, without
entropy or bootstrap. Inputs: observed state, online context, mean action and
candidate action delta; no mode/reset/candidate-name inputs. Use a hidden64x64
ReLU MLP, train-only input and target scaling, Adam3e-4, batch128, exactly 2000
updates. Subtract its output at zero action delta to anchor no-change at zero;
opposite actions may both have negative advantage. Initialization/sampling seeds
are 49000000+seed and 50000000+seed. No validation stopping or threshold fitting.
Save/freeze the evaluator before creating the validation environment. Predictions
are computed before the candidate rollouts, using only the anchor and actions.

Select the highest predicted advantage among five actions, with base first for
ties. Controls use the identical candidate set: centered soft-Q, centered raw-Q,
no-change, and a state-independent per-preference candidate-mean rule fitted
only on training labels (including base=0). Primary: realized selected-minus-base
512-step gain, averaging eight resets within cells and all 12 cells per model.
Report per-reset gains, rejection counts, calibration and negative-part means.
Only another shadow replication is eligible if EACH of all four model groups
has positive mean gain, no lower mean than soft/raw/fixed-preference controls,
and no more negative mean negative-part gain than those three controls. No-change
is tested through strictly positive mean gain, not a zero-downside requirement.
Lower prediction RMSE alone cannot pass; this is not deployment approval.

Budget across four jobs: 2,457,600 label-collection + 1,228,800 validation =
3,686,400 interactions, plus eight initialization probes and 8,000 evaluator
updates. Zero policy/encoder/original-critic updates. Results, labels and fitted
models stay remote under `results/iclr_paired_advantage_20260921`.
Limitations: observed policies, balanced-mode sampling and simulator-copy access;
finite-horizon one-action selection is not training-operator routing, cross-seed
generalization or an equal-resource algorithm comparison. No manuscript edits.

Fifteen tests pass locally and remotely: exact zero reference with non-antisymmetric
advantages, train-only deterministic fitting, disjoint reset sets, fit/save before
validation, predictions before branch outcomes, unchanged source networks,
outcome-independent choices, downside gating and interaction accounting. All
four production sources load frozen. Submitted `t99780-t99783`, ordered v27
seed31/33 then v28 seed31/33, CPU=2/RAM=2048MB each, scheduler placement across
node001-node007 without a required/preferred node or automatic result download.
Results pending; no policy, routing or paper changes.

### Paired-advantage outcome (2026-09-21)

`t99780-t99783` all exited 0 on their first launch. Server-side recomputation
confirmed the complete train/validation grids, disjoint resets, train-only fixed
controls, stored choices/gates, 1,152 copied/native identities and frozen budget:
2,457,600 collection + 1,228,800 validation steps, eight probes and 8,000 evaluator
updates; zero source-network updates. Only the 68,946-byte
`results/iclr_paired_advantage_20260921_summary.json` was pulled.

| Group | Learned mean gain | Fixed-preference mean gain | Train / validation RMSE | Frozen gate |
|---|---:|---:|---:|---|
| V27 / 31 | -0.686934 | -0.608538 | 1.106569 / 7.272985 | fail |
| V27 / 33 | 0.289630 | 0.208430 | 0.257155 / 2.615432 | pass |
| V28 / 31 | -1.986704 | -0.864410 | 1.253408 / 20.882117 | fail |
| V28 / 33 | -0.287350 | -0.189146 | 0.732446 / 7.503150 | fail |

The all-four-group gate fails. V28/33 also worsens the negative-part mean
relative to soft-Q (-0.619592 versus -0.586832). Learned validation RMSE is
worse than every control in all four groups. Final saved-model training error
is much smaller than validation error, showing a substantial generalization
gap. The learned selector keeps base in 19/384 cases and loses in 159; 120 of
those losses occur despite a positive-return alternative. V28/31's worst
selected action loses 110.983674, with all four changes below base at that state.

Next proposed step: use existing training labels only for reset-grouped
cross-validation, keeping each reset's preferences and candidates together.
Compare a low-capacity, regularized evaluator with the current MLP, training-only
fixed-preference rule and no-change; judge out-of-fold selected gains and downside,
not training fit. This requires no new simulator interactions. Do not tune on
the inspected validation set; any resulting model requires fresh validation.
No new jobs, seed expansion, actor/routing integration or manuscript edits.
Six focused tests pass locally and remotely.

Limitations: the gap does not identify model capacity, data coverage or state
representation as its unique cause. Positive-alternative counts use exact signs
and may include tiny gains. These are extra-budget one-action shadow decisions
on observed policies, not an equal-budget BAMOR improvement.

### Frozen training-only grouped CV (2026-09-21)

Reuse only the 192 training anchors per v27/v28 seed31/33 model from the paired
pilot; the inspected validation cases and saved full-data evaluators are not
inputs. Four folds use `repeat % 4`, identically across modes/preferences/models.
Keep all preferences and candidate labels for a reset together. Each fold fits
144 anchors (576 nonbase rows), predicts 48 anchors, and holds out 16 reset IDs.
Every training anchor receives exactly one out-of-fold prediction.

Compare the original anchored hidden64x64 MLP with one fixed quadratic ridge
evaluator. Both receive the same 42 causal features. For ridge, normalize the
36 state/context/base-action coordinates with fold-training mean/std and the
six action deltas with fold-training RMS, floor1e-3. Features are six deltas,
216 state-by-delta products and 21 upper-triangular delta products: 243
coefficients, no intercept, exactly zero at no-change. Scale each feature by
its training RMS and targets by training std (floors1e-3). Solve normalized
mean squared error plus `1.0 * ||beta||^2`; no penalty search. Quadratic terms
allow opposite actions to both lose.

Refit the original MLP for exactly 2,000 updates/fold with its original optimizer
and scaling; pass seed `policy_seed + 1000*(fold+1)` to the existing fitter.
Recompute the per-preference fixed-candidate rule using fitting-fold labels only.
Retain soft-Q, raw-Q and no-change controls on the same five actions. Selection
is argmax predicted gain, base wins zero ties. No clipping or rejection tuning.

Primary: OOF selected-minus-base return at horizon512, gamma0.99, averaging 16
resets within each of 12 cells, then equal cells per model. Ridge is eligible
for a proposed fresh-validation pilot only if ALL FOUR models have positive
mean gain, no lower mean than MLP/fixed-preference/soft/raw, and no more negative
mean negative-part gain than those four controls. Report all models and controls,
fit/OOF RMSE and action counts; no pooled override or winning-fold selection.

Cost: zero new simulator interactions or policy/original-critic updates; reuse
768 anchors/3,072 nonbase labels across models. Sixteen MLP fits total 32,000
optimizer updates, plus 16 ridge solves. Four CPU tasks via scheduler with no
required/preferred node; results stay remote. This is development diagnosis,
not independent confirmation, routing integration or a revised prior result.
No manuscript edits. Implementation and tests precede submission.

### Grouped-CV outcome (2026-09-21)

`t99784-t99787` completed on first launch, exit0, with scheduler-selected
placement and no required/preferred node. Five focused tests pass locally and
remotely. Recomputed all OOF decisions, fold membership and gates; cost is zero
new interactions, 32,000 MLP updates and 16 ridge solves. Pulled only the
62,714-byte `results/iclr_advantage_cv_20260921_summary.json`.

| Group | Ridge OOF gain | MLP OOF gain | Ridge gate |
|---|---:|---:|---|
| V27 / 31 | 0.303198 | 0.012660 | pass |
| V27 / 33 | 0.517191 | 0.431694 | pass |
| V28 / 31 | 0.275253 | 0.226984 | fail |
| V28 / 33 | 0.678099 | 0.304046 | fail |

Ridge improves mean gain, negative-part mean and OOF RMSE over MLP in all four
groups, but the all-group gate still fails. V28/31 trails soft-Q gain1.154253
and downside-0.155874 (ridge-0.342265); V28/33 trails raw-Q gain0.818445 and
downside-0.732996 (ridge-0.817115). No fresh-validation expansion is approved.

Post-hoc paired-gap inspection, without changing metrics or excluding states:
V28/31 mode2/speed/repeat8 loses36.814556 under ridge versus a soft-Q gain of
171.246077; repeat4 favors ridge by54.619546. V28/33 trails raw-Q on143/192
anchors and exceeds it on13 (36 ties); sign counts can include tiny differences.
The latter failure is not exclusively one bad outlier.

Next proposal: keep frozen soft/raw candidate contrasts as causal prediction
inputs and test a low-capacity residual correction on training-only grouped
folds, including the uncorrected critics and current ridge/MLP as controls.
No per-environment/seed version switching, validation-set tuning, or new rollout
budget yet. This comparison supports a smaller regularized model relative to
this MLP; it does not isolate capacity from regularization, prove reliable tail
prediction, or establish independent policy improvement. No main-algorithm or
manuscript changes.

### Frozen critic-residual grouped CV (2026-09-21)

Reuse the paired pilot's training cases and the same `repeat % 4` folds for
v27/v28 seed31/33. All preferences/candidates for a reset remain together.
Reuse the prior CV's recorded OOF ridge/MLP/fixed-preference/soft/raw/no-change
controls, paired by case identity and unchanged gains. No old validation labels,
full-data evaluators, policy checkpoints or simulator calls enter this run.

For each fitting fold, divide soft/raw candidate-minus-base Q contrasts by
their training RMS and targets by training std (floor1e-3). Fit a nonnegative
two-coefficient ridge mixture by SciPy NNLS with an augmented identity penalty.
This calibrated prior is also a separate control. Fit its normalized return
residual using 77 features: two Q contrasts, 72 products with standardized
state/context/base-action coordinates, and three quadratic Q products. Scale
features by fitting-fold RMS without centering. Both objectives use mean squared
error plus `1.0 * squared coefficient norm`; no search, intercept or clipping.
The residual has unrestricted coefficients; all terms vanish at base and the
quadratic terms allow both signs to lose. Action dependence is through critic
contrasts, not action-delta features or candidate/mode/seed identifiers.

Freeze predictions before scoring held-out labels. Primary remains equal-cell
OOF selected-minus-base horizon512 return, gamma0.99, argmax with base winning
zero ties. Eligibility for a proposed fresh-validation pilot requires ALL FOUR
groups to have positive mean gain and no worse mean or negative-part mean than
the calibrated prior, previous ridge/MLP, fixed-preference, soft and raw controls.
No pooled override. Cost: 16 two-variable NNLS fits and 16 77-variable ridge
solves, zero new interactions, MLP updates or source-network updates. Four
scheduler tasks without a required/preferred node; compact results only.
This is continued development on inspected training folds, not confirmation.

### Critic-residual CV outcome (2026-09-21)

`t99788-t99791` all exited0 on first launch, with scheduler-selected placement.
Recomputed the folds, cached-control equality, decisions and gates. Cost:
16 NNLS fits + 16 residual solves, zero new interactions or network updates.
Ten tests pass locally and remotely; only the 83,675-byte
`results/iclr_residual_advantage_20260921_summary.json` was pulled.

| Group | Calibrated prior gain | Previous ridge gain | Residual gain |
|---|---:|---:|---:|
| V27 / 31 | 0.026981 | 0.303198 | -0.228325 |
| V27 / 33 | 0.178501 | 0.517191 | 0.346539 |
| V28 / 31 | 0.378089 | 0.275253 | 0.241478 |
| V28 / 33 | 0.818445 | 0.678099 | -0.006440 |

Zero groups pass. Residual correction worsens the negative-part mean versus
the calibrated prior in every group and improves its mean only in V27/33.
Residual training RMSE improves in all16 folds, but OOF RMSE worsens versus
the calibrated prior in all4 groups. Relative to the previous ridge model,
OOF RMSE improves in all4 while selected gains and downside worsen in all4.
Thus better aggregate squared-error fit does not establish a better selector.

Stop this residual variant; no fresh-validation expansion or main-algorithm
integration. Next proposal: one fixed, training-only cost-sensitive selection
pilot using candidate-minus-base returns directly, retaining no-change and all
current controls. Judge realized OOF selection gains and downside; do not select
by RMSE or tune on the old validation split. The current result concerns this
77-feature residual model, not impossibility of residual learning. Continued
reuse of these development folds will still require fresh confirmation before
any policy-improvement claim. No manuscript edits.

### Frozen cost-sensitive grouped CV (2026-09-21)

Retain training-only v27/v28 seed31/33 cases, `repeat % 4` folds, five actions
and all eight cached OOF selectors from the residual pilot. Use exactly the
previous ridge model's 243 quadratic action features with fitting-fold-only
scales, now shared as a feature transform. No critic forecasts, candidate IDs,
mode IDs or reset IDs are additional model inputs. Base features remain zero.

For realized candidate-minus-base gains g, set utility `u(g)=g+min(g,0)`:
positive gains unchanged, negative gains doubled. Divide utilities by the
fitting-fold std of nonbase gains, floor1e-3. Fit 243 zero-initialized linear
logit coefficients with the full-batch objective
`-mean(sum(softmax(features @ beta) * u / scale)) + ||beta||^2`.
Use float64, Adam0.01, exactly2,000 updates/fold, temperature1, no early stopping,
threshold search, entropy bonus or clipping. Execution is deterministic argmax,
with base winning zero ties. Logits are not return predictions; omit their RMSE.

Predictions precede held-label scoring. Keep the existing primary: equal-cell
OOF selected gain and negative-part mean at discounted horizon512. ALL FOUR
groups must have positive mean and be no worse on either metric than residual,
calibrated prior, ridge, MLP, fixed-preference, soft and raw controls. No-change
is tested through positive mean. Reuse old OOF predictions without refitting.
Budget:16 selector fits/32,000 full-batch updates, zero new environment steps,
policy/critic updates, old MLP refits or checkpoint reads. Scheduler placement
is not pinned; only compact summaries are pulled. This is another development
comparison, not fresh confirmation; no main-algorithm or manuscript changes.

### Cost-sensitive CV outcome (2026-09-21)

`t99792-t99795` all exited0 on first launch with scheduler-selected placement.
Recomputed decisions, cached-control equality and all frozen gates. Sixteen
tests pass locally and remotely. Cost:32,000 selector updates, zero new
interactions or old-model refits. Pulled only the 98,573-byte
`results/iclr_cost_sensitive_20260921_summary.json`.

| Group | Previous ridge gain | Cost-sensitive gain | Negative-part mean |
|---|---:|---:|---:|
| V27 / 31 | 0.303198 | -0.213714 | -1.107180 |
| V27 / 33 | 0.517191 | 0.461248 | -0.271589 |
| V28 / 31 | 0.275253 | 0.255079 | -0.382856 |
| V28 / 33 | 0.678099 | 0.428916 | -1.101064 |

Zero groups pass; ridge retains better mean and downside in every group.
Average fitting-fold gains are0.884028/0.652216/1.246733/2.116400 in table
order, above their OOF gains. Base is selected only1/768 times. Recorded
objectives change little between500 and2,000 updates; extra fitting alone
has no clear support from these logs.

Post-hoc execution diagnostic uses the saved logits and all five observed
returns, without fitting or new rollouts. Median maximum softmax masses are
0.262/0.336/0.255/0.257. Softmax-weighted expected gains are
-0.028250/0.266413/0.405287/0.403946; negative-part means are
-0.729735/-0.262530/-0.258971/-0.869367, less negative than hard argmax in
every group. This does not pass the frozen gate or justify switching to a
stochastic selector; these masses are not calibrated success probabilities.

Stop this variant. Next proposal: a fixed cost-sensitive margin objective
that directly trains deterministic candidate ordering, including base, on the
same training-only folds. Retain current controls and gates, with no temperature,
penalty or rejection-threshold sweep. The observed train/OOF gap and soft/hard
execution mismatch warrant this test but do not identify a unique failure cause.
No fresh-validation expansion, policy integration or manuscript edits.

### Frozen deterministic margin CV (2026-09-21)

Replace only the softmax objective with a cost-sensitive structured hinge.
For each training anchor, let `u=(g+min(g,0))/scale`, `b=argmax(u)` and
`s=features @ beta`. Minimize
`mean(max_a(s_a + u_b - u_a) - s_b) + ||beta||^2`.
The hinge upper-bounds normalized utility regret of deterministic argmax on
the labeled training anchors; this is not a held-out performance guarantee.
Base wins ties. Retain 243 features, fitting-fold-only normalization, zero
initialization, float64 Adam0.01 and exactly2,000 full-batch updates/fold.
No loss, penalty, threshold or step-count sweep; no extra model inputs.

Use the same192 training anchors/group, four reset-grouped folds and horizon512.
Retain every cached OOF control, including the failed softmax selector. All four
groups must show positive mean gain and no worse mean gain or negative-part
mean than every nonbase control. Logits have no return RMSE. This pilot uses
32,000 selector updates and zero new interactions or policy/critic updates.
Do not read checkpoints or use old validation labels. Schedule without pinned
nodes; pull only a compact summary. This remains development CV, not fresh
confirmation. Stop if the gate fails; no algorithm or paper edits in this pilot.

### Deterministic margin CV outcome (2026-09-21)

`t99797-t99800` completed with exit0 on first launch, with scheduler-selected
placement and no pinned/preferred nodes. All21 tests pass locally and remotely.
Recomputed OOF decisions, cached-control equality and frozen gates on the server;
pulled only `results/iclr_margin_selection_20260921_summary.json` (108,566 bytes).
Cost:32,000 selector updates, zero new interactions or policy/critic updates.

| Group | Ridge gain | Margin gain | Ridge negative-part | Margin negative-part |
|---|---:|---:|---:|---:|
| V27 / 31 | 0.303198 | 0.081318 | -0.635539 | -0.720409 |
| V27 / 33 | 0.517191 | 0.429165 | -0.203875 | -0.256442 |
| V28 / 31 | 0.275253 | 0.255970 | -0.342265 | -0.403806 |
| V28 / 33 | 0.678099 | 0.276915 | -0.817115 | -1.093860 |

Zero groups pass. All means are positive, but ridge is better on both metrics
in every group. Base choices rise from1/768 to10/768 without resolving downside.
Fitting-fold gains exceed OOF gains in every group; matching the loss to hard
selection has not repaired generalization in this fixed pilot. Stop this variant.

Next: diagnose existing32/128/512-step paired gains for sign/ranking stability
and tail concentration across reset groups, without fitting or new interactions.
Keep512 as the primary outcome. This can distinguish early errors from later
trajectory divergence before proposing a multi-horizon evaluator; it cannot
establish label noise from deterministic returns alone. No fresh-validation
expansion, policy integration or manuscript edits yet.

### Frozen horizon attribution diagnostic (2026-09-21)

Use only the same192 training anchors in each v27/v28 seed31/33 group and
their cached margin-pilot OOF decisions. No fitting, checkpoints, old validation
inputs or new interactions. Keep all existing512-step results and decisions.
Compute scalar candidate-minus-base gains from stored discounted vector returns
at32/128/512; segment increments retain the original discount origin.

Report sign reversals and pairwise ranking reversals for32->128,128->512 and
32->512, with denominators excluding differences within1e-6 at either endpoint.
Report hindsight best-action agreement and the512-step return of a prefix-best
action as descriptive label statistics, not executable selectors. For all old
selectors, evaluate their unchanged OOF choices at each horizon.

For margin-minus-ridge, ridge-minus-soft and ridge-minus-raw, decompose the
mean gain gap by0:32,32:128,128:512. Report top1/top4 reset-group shares of
absolute label mass and adverse gap mass, retaining all three preferences per
reset. Also report equal-cell leave-one-reset-out gap ranges without refitting;
this is influence analysis, not additional CV or grounds for dropping samples.
These are descriptive development diagnostics, not new performance gates.

### Horizon attribution outcome (2026-09-21)

All four groups analyzed;10 tests pass locally and remotely. No experiment
model updates or new interactions. Only the110,758-byte
`results/iclr_horizon_attribution_20260921_summary.json` was pulled.

In v27/31,v27/33,v28/31,v28/33 order, nonbase sign reversal rates32->512 are
16.94/27.47/21.47/7.81%, versus5.83/9.64/9.75/1.30% for128->512 (decisive
comparisons only). The128-step hindsight best action retains98.69/94.64/98.77/
95.73% of the512-step oracle mean gain. This is label evidence, not a learned
selector result or proof that128-step targets are easier to predict.

For v28/31 ridge-minus-soft, the full gap is-0.879000: -0.123449 in0:32,
-0.684097 in32:128 and-0.071453 in128:512. Reset47310208 accounts for83.45%
of adverse gap mass; removing its three preferences changes the equal-cell gap
to+0.221708. Keep that reset. The problem is not exclusively post128 divergence;
the same group's largest-reset share of absolute label mass is already40.58%
at128. Margin-minus-ridge remains negative after any single-reset removal in
three groups; only v28/31 is sign-sensitive. Prior failure decisions stand.

Next proposal: one fixed128-step-target quadratic ridge pilot, keeping the
same features, penalty and training-only reset folds, with16 closed-form fits.
Compare its unchanged argmax choices on the original512-step mean/downside
gates against every cached control. No horizon sweep or sample removal. Test
predictability first, before adding multi-horizon heads or collecting fresh data.

### Frozen128-step ridge pilot (2026-09-21)

Fit the existing243-coefficient quadratic ridge to128-step candidate-minus-base
discounted returns. Keep the original float32 input/target construction,
float64 solve, fitting-fold feature/target scales and penalty1. Only labels
change; no critic forecasts or additional inputs. Use the same training-only
v27/v28 seed31/33 grid and `repeat % 4` reset-grouped folds. Predict held features
before seeing either horizon's held labels; execute argmax with base-first ties.

Retain every cached margin-pilot control. ALL FOUR groups must have positive
512-step mean gain and be no worse on mean gain AND negative-part mean than
every nonbase control. Report128-step calibration/choices separately; never
score128-step predictions as512-step value estimates. No horizon/penalty sweep,
sample exclusion, checkpoint reads, policy/critic changes or old validation
inputs. Budget:16 closed-form solves, zero optimizer updates/new interactions.
Scheduler chooses placement; pull only a compact summary. This is development
CV, not fresh confirmation; failure stops this variant without paper edits.

### Prefix128 ridge outcome (2026-09-21)

`t99807-t99810` completed on first launch with exit0, scheduler-selected placement
and no pinned/preferred node. Fourteen tests pass locally and remotely. Server
aggregation checked both horizons against source labels and retained all cached
controls. Pulled only the115,914-byte `results/iclr_prefix_ridge_20260921_summary.json`.
Cost:16 closed-form solves, zero new interactions or optimizer updates.

| Group | Original ridge512 gain | Prefix128 gain at512 | Prefix128 negative-part at512 |
|---|---:|---:|---:|
| V27 / 31 | 0.303198 | 0.302130 | -0.631427 |
| V27 / 33 | 0.517191 | 0.455450 | -0.247762 |
| V28 / 31 | 0.275253 | 0.254922 | -0.343741 |
| V28 / 33 | 0.678099 | 0.510499 | -0.952553 |

Zero groups pass. Every512-step mean trails the original ridge; downside improves
only for v27/31. V28/31 already has negative mean gain at its own128-step training
horizon (-0.077615 versus soft+0.742235). Prefix-oracle quality did not translate
into a better learned selector. Stop this variant; no policy or paper change.

Next: retain512-step targets and diagnose the existing ridge's fitting-fold
normalized error as orthogonal feature-projection residual plus ridge shrinkage
error, using the same frozen features/folds. This separates representational
limits from regularization bias on training labels before choosing an algorithm
change. It is not a new selector or held-out improvement claim; no penalty sweep,
sample removal, new labels or deployment integration.

### Frozen ridge spectral diagnosis (2026-09-21)

Reconstruct the original512-step ridge on the same16 fitting folds, using its
float32 labels,243 normalized features, fitting-fold scales and penalty1.
Let `A=Phi/sqrt(n)`, `b=(y/scale)/sqrt(n)` and take the float64 thin SVD.
Use the fixed numerical rank rule `s > eps64 * max(A.shape) * s_max`.
Decompose normalized training MSE into squared residual orthogonal to the
feature span plus squared error from ridge shrinkage inside that span. Check
the cross term and reconstruction identity numerically, and match original
training RMSE and cached held scores before interpreting the decomposition.

Report each fold's rank, effective degrees of freedom, singular-value range,
zero-prediction MSE and the two error components. Group summaries average four
overlapping fitting folds; component shares use summed normalized MSE, not
independent-sample inference. Budget:16 SVD/ridge reconstructions, no new selector
variants, optimizer updates, environment interactions, checkpoints or old
validation inputs. No penalty sweep or unregularized held-out selector. Large
shrinkage identifies training bias, not evidence that weaker regularization
improves generalization. Use scheduler placement without pinning and small pulls.

### Ridge spectral outcome (2026-09-21)

`t99811-t99814` completed with exit0 on first launch and unpinned scheduler
placement. Nine tests pass locally and remotely. All16 reconstructions match
original training RMSE and cached held predictions (maximum score difference
3.575e-14). Pulled only the21,390-byte spectral summary. No new selector,
optimizer updates, environment interactions or manuscript changes.

| Group | Normalized MSE | Feature-span residual | Shrinkage error | Shrinkage share |
|---|---:|---:|---:|---:|
| V27 / 31 | 0.853866 | 0.343947 | 0.509918 | 59.72% |
| V27 / 33 | 0.893256 | 0.752583 | 0.140673 | 15.75% |
| V28 / 31 | 0.901467 | 0.348878 | 0.552589 | 61.30% |
| V28 / 33 | 0.823524 | 6.69e-9 | 0.823524 | ~100% |

All folds have numerical rank237 of243; group-mean effective degrees of freedom
are50.97/53.85/52.67/40.58. V27/33 is mainly feature-span training residual.
V28/33 admits near interpolation, but its retained-subspace condition number
is3.74e8-5.58e8: the interpolation floor is not evidence of stable estimation.
The bottleneck is not uniformly excessive regularization, and MSE components
alone do not determine action-selection quality.

Next proposal: one fixed penalty0.1 ridge control, common to all four groups,
retaining512-step labels, the same features/folds and every cached control.
Keep the original512-step mean/downside gates. No zero-penalty interpolation,
per-seed settings or grid search. This tests whether reduced shrinkage changes
actual OOF decisions favorably before considering a larger evaluator.

### Frozen weak-ridge control (2026-09-21)

Change only ridge penalty1 to0.1, identically for v27/v28 seed31/33. Keep512-step
paired gain labels,243 quadratic features, float32 data/float64 solve and all
fitting-fold normalization. Use the same192 training anchors and reset-grouped
`repeat % 4` folds; predict held features before accessing held labels. Base
wins argmax ties. Retain all cached prefix-pilot controls without refitting.

ALL FOUR groups must have positive512-step mean gain and no worse mean gain
AND negative-part mean than every nonbase control. Training RMSE and OOF RMSE
are diagnostics, not substitute gates. Exclude128-step predictions and decision
logits from512-step value calibration. No additional penalties, seed-specific
settings, horizon changes, sample removal or old validation inputs. Budget:
16 closed-form solves; zero optimizer, policy or critic updates, new interactions
or checkpoint reads. Scheduler chooses placement. Only compact summaries return
locally. This is development CV, not confirmation or a main-algorithm change.

### Weak-ridge outcome (2026-09-21)

`t99815-t99818` all exited0 on first launch with unpinned scheduler placement.
Thirty-three tests pass locally and remotely, including unchanged default
penalty1 behavior. Recomputed decisions and all gates; pulled only the118,817-byte
weak-ridge summary. Cost:16 solves, zero new interactions or optimizer updates.

| Group | Original gain | Weak-ridge gain | Original negative-part | Weak negative-part |
|---|---:|---:|---:|---:|
| V27 / 31 | 0.303198 | 0.047608 | -0.635539 | -0.830478 |
| V27 / 33 | 0.517191 | 0.463098 | -0.203875 | -0.203670 |
| V28 / 31 | 0.275253 | 0.301736 | -0.342265 | -0.349651 |
| V28 / 33 | 0.678099 | 0.348112 | -0.817115 | -1.088298 |

Zero groups pass. Training normalized MSE falls in all four groups, but OOF
RMSE rises in all four (5.067->5.979,2.200->2.288,8.950->9.338,8.131->9.365).
Only v28/31 improves mean gain, with worse downside. Stop this variant; the
training shrinkage diagnosis did not justify weaker regularization in OOF use.

Next proposal: retain penalty1 and512-step targets, but add one structured
state-dependent even-response term. Current features have `d`, `z*d` and
`vech(d*d')`; the quadratic coefficients themselves do not depend on state.
Append `z*||d||^2` using the same fitting-fold-normalized36 state/base-action
features, adding36 coefficients (243->279), with no mode IDs or extra data.
Keep base exactly zero, all cached controls and the original gates. This is
a test of state-dependent curvature, not evidence it will improve generalization;
no further penalty sweep, actor changes or manuscript edits.

### Frozen state-curvature ridge (2026-09-21)

Use the original penalty1 ridge and512-step labels, appending only
`z*sum(d**2)` to the existing243 features. Here z is the36 fitting-fold-standardized
observation/context/base-action inputs, and d is the6 fitting-fold-RMS-scaled
action increments. The36 new columns use the same fitting-fold RMS normalization
and penalty1 as the old columns. Keep base exactly zero and the old feature block
unchanged. Total279 coefficients; no extra mode/reset IDs, critic forecasts,
candidate IDs, horizon/penalty choices or additional data.

Keep the same192 training anchors/group, v27/v28 seed31/33 and `repeat % 4`
reset-grouped folds. Predict held features before scoring their labels. Reuse
every cached weak-ridge-pilot control. ALL FOUR groups must show positive512-step
mean gain and no worse mean gain AND negative-part mean than every nonbase
control. Calibration uses only512-step value predictors, excluding prefix values
and logits. Budget:16 closed-form solves, no optimizer/policy/critic updates,
new interactions, checkpoint reads or old validation inputs. Unpinned scheduler
placement and compact pulls only. This is a structural development comparison,
not fresh confirmation; no paper or main-algorithm changes on partial success.

### State-curvature outcome (2026-09-21)

`t99819-t99822` all exited0 on first launch, with unpinned scheduler placement.
Eighteen tests pass locally and remotely. All cached controls and source gains
match; only the128,810-byte summary was pulled. Cost:16 solves, zero new
interactions or optimizer updates.

| Group | Original gain | Curvature gain | Original negative-part | Curvature negative-part |
|---|---:|---:|---:|---:|
| V27 / 31 | 0.303198 | 0.261636 | -0.635539 | -0.631112 |
| V27 / 33 | 0.517191 | 0.536242 | -0.203875 | -0.167694 |
| V28 / 31 | 0.275253 | 0.230837 | -0.342265 | -0.334087 |
| V28 / 33 | 0.678099 | 0.706927 | -0.817115 | -0.730556 |

Zero groups pass. Negative-part means improve against original ridge in all
four, but mean gains improve in only two. V27/33 passes the mean gate but not
downside; v28/33 passes downside but not mean. Training normalized MSE falls
in all four, while OOF RMSE rises in all four (5.067->5.149,2.200->2.239,
8.950->9.144,8.131->8.388). Stop this variant; no main-algorithm or paper change.

Next proposal: a fixed-model learning-curve diagnostic using only existing
training labels. Keep original243-feature penalty1 ridge and held reset groups
unchanged; compare nested16/32/48 fitting reset groups (all three preferences
per reset), selected without labels and balanced across the four modes. This
tests whether more fitting groups improve held decisions and value error;
it is not another selector search or permission to collect new interactions.
Freeze the subset rule and analysis before running. No further feature expansion
or penalty sweep based on the present four-group results.

### Frozen ridge learning curve (2026-09-21)

Keep original243-feature penalty1 ridge,512-step targets and the existing four
reset-grouped folds for v27/v28 seed31/33. Compare16/32/48 fitting reset groups,
each with all three preferences; held groups are identical at every size.
Within each repeat block b=0..3, sort the three non-held repeat indices. Chain
c=0..2 cyclically orders these from index `(b+c)%3`; use the first1/2/3 for
sizes16/32/48. Apply the same rule in all four modes. Thus small subsets are
nested, balanced and label-independent; the three16-group subsets partition
the full fit set. Normalize separately on each fitting subset. Never ensemble
chain predictions or select a chain/size using outcomes.

Fit the shared48-group endpoint once per fold and require its scores/choices
to reproduce cached original ridge (score tolerance rtol1e-10,atol1e-10).
Retain cached controls without refitting. Report each chain and size's OOF mean
gain, negative-part mean and512-step RMSE; equal-average chain gains/downside
and pool squared errors before taking RMSE. Show both16->32 and32->48 changes
and chain sign counts. Broad support for collecting more labels requires ALL
FOUR groups to improve mean gain and RMSE with non-worse negative-part mean
on32->48. This is a diagnostic decision rule, not a selector acceptance gate;
mixed results do not authorize a collection run. Overlapping folds/chains are
not independent replications, and this remains reused development data.

Budget:112 closed-form solves (four groups x four folds x seven fits), zero
optimizer/policy/critic updates, new interactions, checkpoint reads or old
validation inputs. Scheduler chooses nodes; only a compact summary is pulled.

### Ridge learning-curve outcome (2026-09-21)

`t99823-t99826` all exited0 on first launch, without node pinning. Fourteen tests
pass locally and remotely. All48-group endpoint scores exactly reproduce the
cached ridge (max error0); controls are unchanged. Cost:112 solves, zero new
interactions. Only the29,612-byte summary was pulled.

| Group | Gain16 | Gain32 | Gain48 | Negative-part32->48 | RMSE32->48 |
|---|---:|---:|---:|---:|---:|
| V27 / 31 | -0.104036 | -0.072480 | 0.303198 | -0.849415 -> -0.635539 | 5.166569 -> 5.066707 |
| V27 / 33 | 0.278712 | 0.423728 | 0.517191 | -0.237965 -> -0.203875 | 2.219774 -> 2.200027 |
| V28 / 31 | 0.646693 | 0.261748 | 0.275253 | -0.376635 -> -0.342265 | 9.005937 -> 8.950314 |
| V28 / 33 | 0.470204 | 0.416988 | 0.678099 | -1.017203 -> -0.817115 | 8.315230 -> 8.130724 |

All four satisfy the prespecified32->48 data-scaling diagnostic. Gain/downside
improve for all three chains in every group; RMSE improves for three chains
except v28/31 (two). Limitations:16->48 is not uniformly monotone, notably for
v28/31; reused folds/chains are not independent confirmations. The unchanged
48-group ridge still passes zero of four all-control selector gates.

Next proposal: collect64 additional disjoint training resets per group, retaining
all three preferences and the frozen policy/candidates. Assign new resets to
the same four-fold rule; add48 new fitting resets to each old48-reset fit while
keeping the old held set unchanged. Compare48 vs96 at penalty1 with all cached
controls, without architecture tuning. Planned collection is768 new anchors
across four groups, about2.46 million environment steps including warmups and
six branches per anchor, plus initialization probes. Count every interaction;
no collection started here. Fresh confirmation remains a separate later stage.

### Frozen ridge label expansion (2026-09-21)

For v27/v28 seed31/33, collect training repeats16..31 using the existing
`paired_advantage.reset_seed("train", seed, mode, repeat)` rule. These64 reset
groups/group are disjoint from old training repeats0..15 and old validation.
Keep all four modes, three preferences, frozen policy/encoder/critics, epsilon0.05,
128-step warmup, six512-step branches and gamma0.99. Reuse the existing fork
collector and checkpoint paths; no policy, critic or MLP training.

Each fold retains its original48 held anchors and fits on144 old plus144 new
anchors with `repeat % 4 != fold`:96 reset groups,1152 nonbase targets. New
same-fold resets are excluded from fitting and are not an evaluation split.
Keep243 features, penalty1,512-step targets, fitting-only normalization and
base-first ties. Reuse cached48-group ridge and all13 state-curvature-pilot
selectors; learning-curve subset models are diagnostics, not candidate controls.

Primary gates remain positive mean gain and no worse mean AND negative-part
mean than EVERY nonbase control in ALL FOUR groups. Also report48->96 gain,
downside and RMSE changes without substituting these for the primary gates.
No seed-specific decisions, label filtering, new feature/penalty choice or
old validation outcomes. Even a pass only warrants a fresh-validation proposal.

New budget:768 anchors x (128 + 6x512) =2,457,600 collection steps, plus four
initialization probes =2,457,604 total steps;16 closed-form solves and zero
optimizer updates. Complete label files can be reused; incomplete collections
must not be silently overwritten/recollected. Scheduler chooses placement;
labels/checkpoints remain on the server and only compact summaries return.

Launched as `t99827-t99830` after14 local and14 remote tests passed. All four
started on scheduler-selected nodes, without preferred/required node settings.

### Ridge label-expansion outcome (2026-09-21)

All four jobs exited0 on first launch. Actual new cost:2,457,604 environment
steps,16 ridge solves and zero optimizer updates. Pulled only the137,614-byte
summary; labels and checkpoints remain remote. Cached controls are unchanged.

| Group | Gain48 | Gain96 | Negative-part48->96 | RMSE48->96 |
|---|---:|---:|---:|---:|
| V27 / 31 | 0.303198 | -0.000932 | -0.635539 -> -0.593842 | 5.066707 -> 5.091531 |
| V27 / 33 | 0.517191 | 0.319229 | -0.203875 -> -0.306089 | 2.200027 -> 2.206225 |
| V28 / 31 | 0.275253 | 0.823905 | -0.342265 -> -0.195045 | 8.950314 -> 8.904471 |
| V28 / 33 | 0.678099 | 0.414122 | -0.817115 -> -0.876056 | 8.130724 -> 8.102526 |

Zero groups pass the primary gates; only v28/31 improves all three scaling
metrics. The32->48 within-batch trend did not extend uniformly to new labels.
Largest cell changes are all at preference(1,0): mode2 for v27/31,v27/33,v28/31
(-3.746,-2.239,+7.315), and mode0 for v28/33 (-3.766). These are descriptive
cell changes, not grounds to drop cases or choose a seed-specific method.

Next proposal: stop new collection. Use existing labels for bidirectional
batch cross-evaluation: old-only48, new-only48 and combined96 fitting resets,
each scored on both old and new same-fold held resets, with unchanged model,
fold rule and labels. Report endpoint cells alongside aggregate gain/downside
and RMSE. This diagnoses training-batch sensitivity without assuming physical
distribution shift or treating the reused batches as fresh confirmation.

### Frozen ridge batch cross-evaluation (2026-09-21)

Reuse the two192-anchor training batches for v27/v28 seed31/33. For each
`repeat % 4` fold, fit old-only48, new-only48 and combined96 reset groups;
predict both old and new same-fold held anchors before reading their returns.
All preferences for a reset stay together. Keep243 features, penalty1,512-step
targets, fitting-only normalization and base-first ties. Batch/mode/reset IDs
are metadata, not model inputs. Reproduce cached old48 and combined96 scores
and choices on old holdouts (score rtol1e-10,atol1e-10).

Report the full three-training-set by two-evaluation-batch matrix, with mean
gain, negative-part mean, nonbase RMSE and choice counts. Retain all12 mode/
preference cells with those metrics, including both endpoints. Report paired
old48->new48, old48->combined96 and new48->combined96 differences separately
on each batch; same-batch old48/new48 gain advantages are descriptive only.
Soft/raw/no-change provide same-action references, not a replacement acceptance
gate. No batch-specific winner, ensemble, label filtering or pooled override.
The previous all-control0/4 result remains; this is not fresh confirmation or
evidence of physical distribution shift.

Budget:48 closed-form solves; zero new environment interactions, optimizer
updates, checkpoint reads or old validation inputs. Scheduler chooses nodes;
raw cases remain remote and only the compact cross-evaluation summary returns.

### Ridge batch cross-evaluation outcome (2026-09-21)

`t99831-t99834` exited0 on first launch with unpinned placement. Thirteen tests
pass locally and remotely; cached old48/combined96 scores reproduce exactly.
Cost:48 solves, zero new interactions. Pulled only the311,844-byte summary.

| Group | Combined96 gain on old holdouts | Combined96 gain on new holdouts |
|---|---:|---:|
| V27 / 31 | -0.000932 | -0.099193 |
| V27 / 33 | 0.319229 | -0.071559 |
| V28 / 31 | 0.823905 | -1.401861 |
| V28 / 33 | 0.414122 | -0.303704 |

Combined96 has negative mean gain on all four new-batch holdouts and is worse
than both separate fits on five of eight group/batch holdouts. All24 ridge
RMSEs exceed the corresponding zero-gain predictor. The largest combined-minus-
old gain changes remain at preference(1,0); v28/31 mode2 flips from+7.315 on old
holdouts to-13.865 on new holdouts. Same-batch training is not uniformly better:
both same-batch gain advantages are positive only for v27. This is batch/decision
sensitivity, not evidence of physical distribution shift. Original gates remain0/4.

Next proposal: keep combined96 fitting labels fixed, but separately use the
old48 and new48 fitting-feature transforms instead of the combined transform.
Evaluate both on the same old/new holdouts, without batch-specific selection.
This isolates input centering/scaling and its implicit regularization from the
added labels. Keep243 features, penalty1 and512-step targets;32 solves, zero
new interactions. No further collection or capacity/penalty tuning at this stage.

### Frozen ridge input-transform control (2026-09-21)

Keep each fold's combined96 training resets,1152 nonbase512-step labels,
target scale,243 coefficients and penalty1 unchanged. Fit two controls using
the old48 or new48 fitting-input transform: state mean/std, delta RMS and
feature RMS, with the existing floors and basis. Donor transforms receive
only576 nonbase input rows, never held inputs or return labels. Target scaling
continues to use the combined training labels. Score both old/new same-fold
holdouts with base-first ties; no batch/seed-dependent transform choice.

Reuse all six batch-cross predictors without refitting. Report each transform's
mean gain, negative-part mean, nonbase RMSE and aggregate choice counts on both
batches, retaining all12 cells. Compare against cached combined96 and the zero-
gain predictor; neither RMSE alone nor a partial improvement is acceptance.
The previous all-control0/4 result remains. This isolates an input-transform
effect under the fixed ridge penalty, not new confirmation or a main-algorithm
change. Budget:32 solves, zero new interactions, optimizer updates, checkpoint
reads or old validation inputs. Unpinned scheduler placement; compact pulls only.

### Ridge input-transform outcome (2026-09-21)

`t99835-t99838` exited0 on first launch with scheduler-selected placement.
Thirteen tests pass locally and remotely; all six cached controls are unchanged.
Cost:32 solves, zero new interactions. Pulled only the224,455-byte summary.

| Group | New holdout gain: combined transform | Old-only transform | New-only transform |
|---|---:|---:|---:|
| V27 / 31 | -0.099193 | -0.220494 | -0.076230 |
| V27 / 33 | -0.071559 | -0.064777 | -0.055328 |
| V28 / 31 | -1.401861 | -1.397190 | -1.403119 |
| V28 / 33 | -0.303704 | -0.297743 | -0.153239 |

Both replacement transforms retain negative new-batch mean gain in all four
groups. All16 replacement-transform RMSEs exceed the zero-gain predictor.
Each transform improves gain, negative-part mean and RMSE together on only
one of eight group/batch evaluations. Old-only/new-only transforms change
5-16/0-10 of192 choices, versus128-147 when changing old/new fitting batches.
Input normalization affects decisions but neither tested replacement repairs
the generalization failure. Stop this repair; the prior gate remains0/4.

Next proposal: in a fixed combined-input coordinate system, decompose the
old-to-combined coefficient and held-score changes into fitting-reset
contributions, including the ridge term. Summarize influence by reset and
preference to distinguish concentrated large-return influence from broad
label conflict. Retain every case; no sample removal, new selector, loss tuning
or collection. This should guide whether a robust-loss intervention is justified.

### Frozen fixed-coordinate label influence (2026-09-21)

Use v27/v28 seeds31/33 and the existing four reset-grouped folds on both batches.
Both fits use combined96 fitting-input normalization, combined target scale,
243 coefficients, penalty1 and512-step labels. The old-only fit is a coordinate-
aligned diagnostic, not the cached old48 selector. Reproduce combined96 scores
and choices; preserve all six cached selectors. No held labels enter fitting
or attribution, and no old validation inputs are read.

Let H = Xc'Xc/nc + I, alpha = nn/nc = 1/2, and beta_o be the old-label fit.
The exact change is H^-1 [Xn'(yn-Xn beta_o)/nc - alpha beta_o]. Separate label,
old-prediction feedback and ridge terms. Aggregate new-anchor contributions by
reset and preference; verify their sum reconstructs coefficients, held scores
and the score-margin change between old and combined choices. The margin is
averaged over every held anchor, with unchanged choices contributing zero.
Report raw-label/residual mass and top1/top5 shares of summed reset score-RMS
magnitudes on each held batch, keeping signed cancellation and the ridge term.
Keep aggregate/cell return metrics for context, not as a new acceptance gate.

Budget:32 model solves and16 multiple-right-hand-side attribution solves;
zero new interactions, optimizer updates or checkpoint reads. Automatic scheduler
placement; raw rows stay remote. Limitations: fixed-coordinate attribution is
not deletion/retraining influence, overlapping folds are not independent, and
RMS-mass shares are not additive fractions of total score change. No sample
removal, loss/penalty search, new selector selection or manuscript change.

### Fixed-coordinate influence outcome (2026-09-21)

`t99839-t99842` exited0 on first launch without pinned/preferred nodes. Twelve
tests pass locally and remotely. Combined96 cached scores reproduce exactly;
maximum coefficient/held-score reconstruction errors are1.78e-15/1.78e-14.
Cost:32 model plus16 attribution solves, zero new interactions. Pulled only
the396,248-byte summary; all raw cases remain remote.

| Group | Top5/48 reset RMS-mass share on new holdouts, fold range | New gain: old labels -> combined labels, fixed coordinates |
|---|---:|---:|
| V27 / 31 | 44.26%-56.72% | -0.031759 -> -0.099193 |
| V27 / 33 | 40.00%-42.13% | -0.217669 -> -0.071559 |
| V28 / 31 | 43.88%-81.57% | -0.171480 -> -1.401861 |
| V28 / 33 | 37.15%-50.14% | -0.035648 -> -0.303704 |

For v28/31, the most influential reset in each fold accounts for66.57%-96.71%
of new-fitting residual squared mass and24.34%-46.39% of new-held reset RMS
mass: reset47310226 in folds0/1/3, reset47310229 when26 is held out. Other
groups are more diffuse. The(1,0) preference component has the largest mean
score RMS in every group on both held batches. Positive and negative reset
margin contributions coexist, and the ridge term is nonzero. These results
support concentrated large-residual influence, especially for v28/31, rather
than a universal single-bad-reset explanation. They do not identify corrupt
labels or establish that downweighting improves expected return.

Next proposal: one common Huber-loss control with normalized threshold1,
combined fitting transform/target scale,243 coefficients and penalty1.
Use mean huber_1(y/scale - Phi beta) + 0.5*||beta||^2, preserving the ridge
quadratic-region penalty convention. Retain every label and unchanged argmax
execution; assess the original512-step mean/downside on both batches against
cached controls. No per-group settings, threshold sweep, return clipping,
new collection or main-algorithm integration. A failed control stops this
robust-loss variant; the present diagnostic is not a new acceptance result.

### Frozen combined-label Huber control (2026-09-21)

Change only the fitting loss to mean huber_1(y/scale-Phi beta)+0.5*||beta||^2.
Keep v27/v28 seeds31/33, four reset-grouped folds, combined96 fitting transforms
and target scale,243 coefficients,512-step labels, all1152 fitting rows and
base-first argmax. Use float64 IRLS from zero, original sample-count divisor,
at most100 solves/fold and gradient infinity-norm <=1e-8; reject nonconvergence.
No held labels enter fitting; no residual weights are applied to evaluation.

Preserve every expansion-pilot old-batch control/gate. Additionally require
positive mean and no worse mean AND negative-part mean than old48/new48/
combined96/soft/raw on EACH held batch, in ALL FOUR groups. No-change stays
a zero-gain/error reference, not a negative-part comparator. Retain all cells,
RMSE, choices and training downweights by batch/preference. RMSE is diagnostic,
not a substitute gate. Budget:16 Huber fits, at most1600 small linear solves;
zero new interactions, policy/critic/MLP updates or checkpoint reads. No threshold
search, sample removal or fresh-confirmation claim. Scheduler chooses nodes;
only a compact summary returns. Failure stops this fixed Huber variant.

### Huber control outcome (2026-09-21)

`t99843-t99846` exited0 on first launch with unpinned placement. Thirteen tests
pass locally/remotely; all16 fits converged in8-33 solves,291 solves total,
maximum gradient8.62e-9. Zero new interactions; pulled only233,277 summary bytes.

| Group | New-batch gain: ridge96 -> Huber | Old-batch gain: ridge96 -> Huber |
|---|---:|---:|
| V27 / 31 | -0.099193 -> -0.060950 | -0.000932 -> -0.032291 |
| V27 / 33 | -0.071559 -> -0.009639 | 0.319229 -> 0.237540 |
| V28 / 31 | -1.401861 -> -0.557107 | 0.823905 -> 0.264670 |
| V28 / 33 | -0.303704 -> 0.233816 | 0.414122 -> 0.627678 |

All eight RMSEs improve but remain worse than zero-gain prediction. Three old-
batch groups lose both mean gain and negative-part mean. Zero groups pass even
the original old-batch gates; none of eight batch evaluations passes all common
controls. Downweighting is active and concentrated at(1,0), but no uniform
decision improvement follows. Stop this fixed Huber variant without deployment.

Next proposal: fit two objective-wise512-step paired-return heads, retaining
preference in the context, then scalarize with the current preference. Use the
same combined fitting data/basis and ridge penalty1: two243-coefficient heads,
486 total, no Huber threshold search or new labels. Preserve both-batch and old
legacy gates. This tests a structured vector predictor, not proof that scalar
coupling caused failure; parameter count also changes. No main-algorithm or
manuscript edits until there is decision-quality evidence.

### Frozen objective-wise paired-return heads (2026-09-21)

Use the same combined96 folds for v27/v28 seeds31/33. Keep preference in the
unchanged inputs and243-feature basis. Fit both512-step candidate-minus-base
return components, including the component with zero current preference.
Use float32 paired vector targets, float64 solves, fitting-only per-component
std scales (floor1e-3), penalty1 for each head and the original sample divisor.
Two243-coefficient heads share one Gram matrix; scalarize physical-unit vector
predictions with the current preference, then use base-first argmax.

Reuse all seven Huber-pilot predictors. Preserve the old legacy gates and both-
batch common-control gates, also retaining Huber as a comparator. ALL FOUR
groups must pass. Report all cells, scalar/vector RMSE, zero predictors and
choice counts; no clipping, label removal or objective/preference selection.
Budget:16 two-right-hand-side solves,32 fitted heads, zero new interactions,
policy/critic/MLP updates or checkpoint reads. Scheduler placement is unpinned;
only a compact summary returns. Limitations:486 coefficients versus243 and
reused development folds; success is not fresh confirmation or causal proof
of harmful scalar coupling. No main-algorithm or paper change in this pilot.

### Objective-wise head outcome (2026-09-21)

`t99853-t99856` exited0 on first launch with scheduler-selected placement.
Fourteen tests pass locally/remotely; cached Huber/common controls and summaries
are unchanged. Cost:16 two-RHS solves, zero new interactions; pulled263,168 bytes.

| Group | New-batch gain: ridge96 -> vector heads | Old-batch gain: ridge96 -> vector heads |
|---|---:|---:|
| V27 / 31 | -0.099193 -> -0.020314 | -0.000932 -> -0.031656 |
| V27 / 33 | -0.071559 -> -0.050527 | 0.319229 -> 0.314842 |
| V28 / 31 | -1.401861 -> -1.394725 | 0.823905 -> 0.855259 |
| V28 / 33 | -0.303704 -> -0.307276 | 0.414122 -> 0.410550 |

Zero groups pass, including the original old-batch gates. Scalar RMSE improves
in all eight group/batch evaluations, but seven remain worse than zero-gain
prediction. All four new-batch(1,0) endpoints have negative selected gain and
MSE above zero prediction, so this is not just error on zero-weight objectives.
Stop this objective-wise variant; doubled output capacity did not repair it.

Next proposal: hold the486-coefficient structure and input basis fixed, but fit
the scalar paired return using preference-projected features w tensor Phi.
Keep penalty1, current-preference argmax and all controls, with no new labels.
This capacity-matched loss comparison tests decision-target alignment. Its
latent heads are not separately return-calibrated. If it also fails, stop this
quadratic-evaluator branch rather than extend its loss/capacity variants.

### Frozen preference-projected loss (2026-09-21)

Use the same combined96 grouped folds for v27/v28 seeds31/33. Keep the existing
243-feature Phi, including preference context, and its fitting-only transforms.
Fit the scalar512-step paired return on head-major w tensor Phi, without any
second feature normalization:486 coefficients, penalty1, sample divisor n,
float32 scalar targets and float64 solve. Use the fitting-only scalar target
std (floor1e-3) and restore physical units before base-first argmax. Latent heads
are not calibrated objective returns. At constant w this is scalar ridge with
effective penalty1/||w||^2, not penalty1 except at the simplex endpoints.

Reuse all eight cached predictors, retaining vector heads and Huber among the
common controls. Keep original old-batch gates and positive mean/no worse mean
AND negative-part mean on EACH batch against ALL common controls; ALL FOUR
groups must pass. Retain all12 cells, RMSE versus zero and choices. Budget:
16 small486-dimensional solves, zero new interactions or policy/critic/MLP
updates. Unpinned scheduler placement; pull only a compact summary. Limitations:
reused development folds, not fresh confirmation. If this fails, close the
quadratic-evaluator branch without deployment or further loss/capacity tuning.

### Projected-loss outcome and branch closure (2026-09-21)

`t99858-t99861` exited0 on first launch with scheduler-selected nodes and no
pin/preference. Sixteen tests pass locally/remotely. All cached controls and
vector summaries are unchanged;16 solves, maximum residual8.33e-17, zero new
interactions. Pulled only258,831 summary bytes.

| Group | New-batch gain: vector -> projected | Old-batch gain: vector -> projected |
|---|---:|---:|
| V27 / 31 | -0.020314 -> -0.205476 | -0.031656 -> -0.001364 |
| V27 / 33 | -0.050527 -> -0.016062 | 0.314842 -> 0.276509 |
| V28 / 31 | -1.394725 -> -1.399108 | 0.855259 -> 0.886317 |
| V28 / 33 | -0.307276 -> -0.283761 | 0.410550 -> 0.412423 |

Zero of four groups pass, including the original old-batch gates; zero of eight
batch evaluations passes. New-batch gains remain negative in all four groups.
Projected RMSE beats vector heads in only two of eight evaluations and beats
zero prediction only for V28/31's old batch. Close this quadratic-evaluator
branch as prespecified: no deployment, threshold search, extra labels or further
loss/capacity variants. The independent raw-return evaluation idea remains
unproven, not disproved in general. No main-algorithm or manuscript change.
Next work must return to the end-to-end adaptive-versus-blind operator question,
not promote another local action-perturbation predictor from these reused folds.

### Frozen end-to-end aligned-target selector pilot (2026-09-21)

The existing v27 experiments used a score-blind portfolio. Test the existing
v25 progress selector on v27's aligned-target backbone, paired with its own
progress-statistics/learning-only-warmup blind control. Add only the two method
routes; retain all18 operators, quotas, ten warmup blocks,1500-step blocks,
feedback formula and optimizer settings. No new feedback model or tuning.

HalfCheetah seeds26-28,50k steps per arm, mean dwell25k, hidden64, batch256,
lr3e-4/gamma.99, learning starts5k, no intermediate evaluation. Six new runs:
300,000 planned training interactions, plus28,728,000 locked-mode evaluation
steps and12 initialization probes. Reuse the six v25 progress adaptive/blind
results to report the target-context x selector interaction. Report actual
attempt cost separately if any run fails/retries. Do not reuse legacy v27 fixed
as the paired control: its pre-learning selection/scoring schedule differs.

Primary: paired final-policy four-mode mean HV. Advance only if its seed mean
improves, its observed worst-seed value does not decrease, and the seed mean
of45k/47.5k/50k snapshot HV does not decrease. Report every seed, mode, late
minimum and archive separately; archive-only gains cannot pass this decision.
Keep the19x21x3 locked full-simplex grid/reset protocol and final-encoder archive.
Validate equal warmup trajectories and all ten warmup blocks within each pair;
report post-warmup arm differences without requiring identical later schedules.

Limitations: observed development seeds, only three pairs, final-encoder late
snapshots are not online recovery curves. Passing motivates a fresh-seed test,
not an adaptive-superiority claim. No new seeds, online probe interactions,
replay refresh, quadratic evaluator or paper edits. Scheduler chooses the HPC
node; only compact summaries return locally.

Twenty-six focused tests pass locally/remotely, including real MuJoCo routing,
paired warmup, unchanged target/arm behavior, summary gates and result-only
postprocessing recovery. All six historical v25 controls validate remotely;
their first20k trajectories match within pairs. The old files predate the
candidate-context log field; the known v25 route supplies that interpretation,
while new v27 outputs must explicitly log update-preference conditioning.
Submitted `t99863-t99868`, ordered adaptive/blind for seeds26/27/28. All six
launched once on scheduler-selected node003-node006 with no pin/preference;
logs show6k-7k learning steps. Results remain pending. Existing terminal results
are reused on runner re-entry; interrupted training still lacks full-state
resume, so count any restart attempts separately rather than hiding their cost.

### Aligned-target selector outcome (2026-09-21)

`t99863-t99868` all exited0 on their first launch, with no retry descendants.
All12 new/reused results validate: matched schedules,20k warmup trajectories,
score reconstruction and full evaluation grids; HV is recomputed from fronts.
Actual new cost:300,000 training steps,28,728,000 evaluation steps,12 initialization
probes. Pulled only246,754 bytes in `iclr_aligned_progress_20260921_summary.json`.

| Metric (HV millions) | Aligned blind | Aligned adaptive | Change |
|---|---:|---:|---:|
| Final seed mean | 3.103402 | 3.350822 | +7.97% |
| Final seed minimum | 0.155758 | 0.393128 | +152.40% |
| Late-three seed mean | 3.449974 | 3.707065 | +7.45% |
| Archive seed mean | 4.492315 | 4.359646 | -2.95% |

All three frozen advancement checks pass. Final paired changes for seeds26/27/28
are -13.87%/+152.40%/+26.13% (2/3 wins); post-warmup arm differences are3/12/10.
The final selector effect is+0.247421M, versus+0.240819M on v25: the mean
target-context x selector interaction is only+0.006601M, with mixed seed signs.

Limitations: seed27's adaptive HV falls from2.570739M at47.5k to0.393128M at50k
(-84.71%, under the same final encoder). Against old v25 adaptive, aligned
adaptive loses4.47% in final mean and71.93% in final minimum. The larger minimum
versus aligned blind is not an absolute robustness repair. These three observed
development seeds warrant replication, not adoption or manuscript claims.

Next proposal: freeze both aligned methods and test five unused seeds, retaining
old v25 adaptive as an absolute-performance reference alongside aligned blind.
Keep final-policy and fixed late-snapshot metrics, report archive separately,
and count all interactions. No parameter tuning, fresh jobs or paper changes
in this outcome review; the quadratic-evaluator branch remains closed.

### Frozen three-arm fresh-seed replication (2026-09-21)

Use seeds1001-1005, chosen before inspecting any new outcomes or mode schedules.
Seeds34-38 were already used by early BAMOR v2 diagnostics. No1001-1005 matches
were found in local/remote result names or the current BAMOR scheduler records.
Run aligned-target progress adaptive, its progress blind control, and old v25
progress adaptive. All three use the existing methods unchanged; no new loss,
feedback, operator configuration, refresh or action evaluator.

Fifteen independent50k HalfCheetah runs, dwell25k, hidden64, batch256, lr3e-4,
gamma.99, learning starts5k, no intermediate evaluation. Keep30 learning-only
1500-step blocks, the existing19 snapshots and locked21-weight/3-episode grids
in each of four modes. Planned cost:750,000 training steps,71,820,000 evaluation
steps and30 initialization probes. Model files stay remote. Reuse terminal
results after postprocessing failures; count any restarted training separately.

Primary: final-policy four-mode mean HV. Replicate the selector signal only if
aligned adaptive improves the five-seed final mean over aligned blind, without
lower final seed minimum or lower seed mean of45k/47.5k/50k snapshot HV. Separately
require those same three summaries to be no worse than v25 adaptive before
advancing to larger confirmation. Report every paired difference and win count,
all modes, late minima and archives. An archive gain cannot replace either
decision, and failure against v25 does not erase a positive within-backbone
selector result. Require all15 results; no partial-seed decision or pooling with
26-28/29-33. Align schedules/evaluation across the triplet; match the full20k
warmup only within the aligned pair and the prelearning5k prefix against v25.

Limitations: five seeds and observed minima are not significance or population
robustness guarantees. Late snapshots use the final encoder, not an online
curve. Passing supports further confirmation, not adoption or paper claims.
Freeze all parameters through completion; unpinned resource-aware scheduling.

Thirteen focused tests pass locally/remotely, covering all three method routes,
real MuJoCo smoke runs, the different warmup boundaries, separate selector and
legacy decisions, all15-result aggregation, budget accounting and saved-result
recovery. Submitted `t99882-t99896`, ordered adaptive/blind/legacy for each seed
1001 through1005. All15 launched once on scheduler-selected node003-node006,
with no node pin/preference and no automatic result download. Individual logs
show7k-12k learning steps. Results are pending; no algorithm or paper edits.

### Fresh-seed replication outcome (2026-09-21)

All15 tasks `t99882-t99896` finished with exit0, one dispatch each, no retries.
Server-side recomputation validates all15 runs, paired schedules/evaluation,
20 warmup episodes within each aligned pair and5 prelearning episodes against
v25. Actual cost:750,000 training interactions,71,820,000 evaluation interactions
and30 initialization probes. Only the276,375-byte summary was downloaded:
`results/iclr_aligned_progress_confirmation_20260921_summary.json`.

| Metric (HV millions) | Aligned adaptive | Aligned blind | v25 adaptive |
|---|---:|---:|---:|
| Final seed mean | 3.916043 | 3.768957 | 4.152535 |
| Final seed minimum | 3.229238 | 1.706898 | 3.317151 |
| Late-three seed mean | 3.923809 | 3.905047 | 3.949030 |
| Minimum over seeds and late snapshots | 2.610097 | 1.706898 | 3.158811 |
| Archive seed mean | 4.696545 | 4.719336 | 4.640734 |

| Seed | Adaptive final | Blind final | v25 final | Adaptive - blind | Adaptive - v25 |
|---|---:|---:|---:|---:|---:|
| 1001 | 4.265907 | 4.661133 | 4.848988 | -0.395226 | -0.583081 |
| 1002 | 3.690333 | 3.690333 | 3.812076 | 0.000000 | -0.121742 |
| 1003 | 4.235783 | 4.316016 | 4.358253 | -0.080233 | -0.122470 |
| 1004 | 3.229238 | 1.706898 | 4.426210 | +1.522340 | -1.196971 |
| 1005 | 4.158951 | 4.470407 | 3.317151 | -0.311456 | +0.841800 |

The selector passes all frozen replication checks: final mean+3.90%, observed
minimum+89.19%, late-three mean+0.48% versus aligned blind. Final outcomes are
1 win,1 tie,3 losses; the positive mean is driven by seed1004. Post-warmup arm
differences are8/0/1/10/3, explaining the identical seed1002 aligned outcomes.
Against v25, all three regression checks fail: final mean-5.69%, minimum-2.65%,
late-three mean-0.64%, with1 win and4 losses. Archive mean+1.20% does not replace
the final-policy decision. Per-mode values and all paired metrics are in the
summary. The frozen larger-confirmation gate is false; do not promote v27.

Limitations: passing the selector thresholds is not broad seed-wise improvement
or statistical significance. These five seeds remain separate from the pilot;
late snapshots use the final encoder. No fresh v25 blind arm exists here, so
this batch cannot establish the target-context x selector interaction.

Next proposal: retain v25 as the reference and add only its blind control on the
same five seeds, reusing these15 runs to complete the2x2 attribution comparison.
Freeze its protocol before launch; no parameter tuning or20-seed expansion.
No new jobs, algorithm changes or paper edits in this readout.

### Frozen missing-arm completion (2026-09-21)

Add only `bamor_v25_progress_fixed_archive_mix` on seeds1001-1005, unchanged,
under `progress_factorial_completion_v1`. Reuse the15 validated three-arm runs
and their saved summary; compare the new v25 pair's full20k warmup, schedules
and evaluation grids. All training, selector blocks, snapshots and evaluation
settings remain those of `aligned_progress_confirmation_v1`. New budget:
250,000 training interactions,23,940,000 evaluation interactions,10 probes.
The full20-run terminal budget is1,000,000/95,760,000/40, with any failed attempts
counted separately. Recover saved terminal results without retraining.

Report paired adaptive-minus-blind effects on each backbone, aligned-minus-v25
effects with each selector, and interaction `(aligned adaptive - aligned blind)
- (v25 adaptive - v25 blind)`. Final four-mode HV is primary; retain every seed,
four-mode values, observed minima, fixed late-three metrics and separate archive
results. The legacy selector passes only with higher final seed mean, no lower
final seed minimum and no lower late-three seed mean than its blind control.
Passing permits proposing fresh paired confirmation on v25; failing does not
justify expanding that branch. No automatic job expansion or parameter tuning.

Limitations: this completion was chosen after seeing the other three arms on
these seeds; it is attribution, not independent confirmation. Interaction is
a descriptive end-to-end algorithm contrast, not an individual operator's
causal effect. A positive interaction cannot repair v27's failed regression
gate, which remains unchanged. Five-seed minima are not robustness guarantees;
final-encoder snapshots are not online recovery curves.

Five focused tests pass locally and on the server: fixed-arm routing, terminal
result reuse, protocol/cost validation, paired contrast signs, and preservation
of the previous decision while requiring all five new results. Submitted
`t99905-t99909` for seeds1001-1005 in order. All five launched once on
scheduler-selected node003-node006 without node pin/preference or automatic
result download; logs show the expected v25 blind method at5k-6k steps.
No training algorithm or paper edits. Results are pending.

### Four-arm completion outcome (2026-09-21)

All five tasks `t99905-t99909` exited0 on their first dispatch, with no retries.
Server-side validation passes for the five new runs and their paired20k warmup,
schedules and evaluation grids; reuse the15 previously validated runs. New cost:
250,000 training interactions,23,940,000 evaluation interactions and10 probes.
Only the374,105-byte `results/iclr_progress_factorial_completion_20260921_summary.json`
was downloaded, plus a compact server-side seed1004 trace readout.

| Condition | Final mean | Final minimum | Late-three mean | Archive mean |
|---|---:|---:|---:|---:|
| v25 adaptive | 4.152535 | 3.317151 | 3.949030 | 4.640734 |
| v25 blind | 3.587026 | 1.661048 | 3.981505 | 4.768026 |
| Aligned adaptive | 3.916043 | 3.229238 | 3.923809 | 4.696545 |
| Aligned blind | 3.768957 | 1.706898 | 3.905047 | 4.719336 |

HV units are millions. V25 adaptive versus blind: final mean+15.77%, observed
minimum+99.70%, late-three mean-0.82%, archive mean-2.67%;3 final wins,2 losses.
Paired final differences for1001-1005 are+0.595404/+0.561466/-0.483082/
+2.765162/-0.611401M. Its frozen gate fails on the late-three metric.
Alignment raises blind final mean5.07% but lowers adaptive final mean5.69%:
the final selector x target interaction is-0.418425M (2 positive,3 negative
seed contrasts), not evidence of synergy. Both prior v27 decisions remain intact.

Seed1004 accounts for most v25 final gain. Its only arm difference is block30,
steps48501-50000: adaptive `recovery:stratified_relabel_envelope` versus blind
`extreme:edge_envelope_policy`. A compact raw-record comparison confirms identical
first29 complete block records and first48 episode records; the last mode
switch is35309, with no switch in the final block. Blind final HV is1.661048M,
down58.92% from its47.5k snapshot, versus adaptive4.426210M at50k. Final training
episode speed returns are189.36 versus502.12. This localizes a useful case.

Limitations: snapshots use each run's final encoder; these contrasts do not
isolate individual operator components or establish population robustness.
The completion is attribution after observing three arms, not fresh confirmation.

Next proposal: diagnose the seed1004 final-block update using existing traces
and operator configurations, with negative-effect seeds1003/1005 as contrasts,
before designing a bounded intervention. Do not blacklist an operator from one
seed, relax the late-snapshot gate, or expand either branch to20 seeds. No new
jobs, algorithm changes or paper edits in this readout.

### Prefix-only terminal-block diagnosis (2026-09-21)

`diagnose_progress_terminal_block.py` reads all five completed v25 pairs, with
zero new interactions or gradient updates. Five focused tests pass locally;
the server readout succeeds on all five pairs. Output:
`results/iclr_progress_terminal_block_20260921_summary.json` (about312KB).
All pairs have matching block/episode records before their first arm divergence.

For seed1004 block30, scoring moves recovery:stratified_relabel_envelope from
4.762% to4.990% probability and extreme:edge_envelope_policy from8.333% to7.842%.
The recovery score is+0.007109 from feedback4501 steps old; the extreme score
is-0.005998 from feedback18001 steps old, before the latest mode switch. The
positive outcome is not evidence of high-confidence avoidance. Feedback arrives
only at block end, so it cannot interrupt this final1500-step update interval.

Seed1005 block26 crosses the mode switch after42806. Its adaptive score-0.489292
combines speed+0.010708 with cost-0.500000; the blind counterpart is-0.485727.
This feedback mixes regime change with operator performance. Its adaptive
block29 is forced into the extreme family by the quota, then samples using
feedback31501 steps old. Seed1003 also ends below blind despite positive final
block progress: current-mode progress and four-mode final HV are not equivalent.

The seed1004 recovery/extreme configurations differ in preference sampling
(stratified/edge), relabel rate(1/0), copies(3/1), critic candidates(4/5), actor
candidates(1/5), and vector coefficient(.05/.06). Both BC coefficients are zero.
Code inspection shows expanded relabel copies also feed the actor update, so
changing `critic_relabel_copies` is not a critic-only intervention.

Limitations: no saved per-update losses/gradients or resumable48.5k state; actor
archives cannot restore that state. Quota-branch `last_probs` is a selected-arm
one-hot marker, so this diagnostic captures actual sampler probabilities instead.
Ground-truth modes only annotate the readout; no oracle enters the selector.

Next bounded proposal: replay seed1004's shared48.5k prefix once, then compare
the factual extreme arm, full recovery positive control, actor-candidates-only
5-to-1 intervention, and relabel-rate-only0-to-1 intervention. Keep other knobs
fixed in each one-factor arm, especially copies=1. First reproduce the factual
and positive controls; interpret component effects only if they reproduce.
Count the prefix plus all four1500-step branches (54.5k training interactions)
and all terminal evaluations; freeze the full execution protocol before launch.
This is a local mechanism test, not an algorithm promotion or20-seed campaign.

### Frozen shared-prefix component forks (2026-09-21)

`terminal_component_forks_v1` replays seed1004's v25 blind prefix through48500,
including all29 operator blocks and the half-finished49th episode. Clone the
full learner/target networks, Adam moments, replay, encoder/optimizer, context
trackers, selector, NumPy/Torch RNG and MuJoCo integration state. Resume the
unchanged training loop for1500 steps in each branch; no updates are discarded
or selected for deployment. One scheduler task runs the four branches serially
from the in-memory prefix, without a required/preferred node.

Frozen order: factual extreme arm; adaptive recovery control; factual with
only actor candidates5->1; factual with only relabel rate0->1. The latter two
keep edge sampling, copies1, critic candidates5, vector coefficient.06 and BC0.
The recovery control uses the original adaptive selector, including its normal
selection RNG draws. Require the historical first48 episode records and first29
blocks to reproduce before branching. Each endpoint control must match all50
episode returns and the saved final4x21 return grid within1e-6, and match block
records/schedules; stop before component branches if either control fails.

Evaluate final policies only, using the same four locked modes,21 preferences,
three episode seeds and1000-step episodes. Drop prefix actor archives from the
branch evaluation, not from the prefix training process. Formal budget:
54,500 training interactions,49,501 learner updates,1,008,000 evaluation
interactions and10 initialization probes. Separate test and failed-attempt costs.
Completed reports are reused; incomplete attempts remain untouched and require
an explicit new output root to retry. No result/model auto-download.

Readout: every branch's final/per-mode HV, all paired final-return differences,
last two training episodes, and actor/critic gradient norms and parameter drift
at the first/last update and every100 steps. Critic gradients are read at its
optimizer step, before actor backward adds to them. These diagnostics add no
updates or interaction. Control reproduction is required before interpretation;
no success threshold is tuned on component outcomes and no automatic expansion.

Limitations: one selected seed, equal update counts rather than equal FLOPs,
and final policies using their own final encoders. This does not identify a
globally safe operator or demonstrate a new adaptive algorithm.

Nine focused tests pass locally and on the server, including exact resumed
versus uninterrupted actor/critic/encoder tensors, returns and block records
for both controls, partial-episode and replay preservation, one-factor scope,
and incomplete-attempt preservation. Verification cost across this turn's
local/remote smoke runs, including the initial metadata-comparison test failure:
449 training interactions,19,404 evaluation interactions and28 probes.
Submitted `t99912`, one dispatch on scheduler-selected node004, with no node
pin/preference or result auto-download. Logs show the v25 blind prefix at7k;
formal control reproduction and component outcomes remain pending.

### Terminal-fork reset-state repair (2026-09-21)

`t99912` exited1 at the factual-control gate. The prefix and episode49 matched
exactly, but episode50's speed return differed by2.893340; the terminal return
grid differed by up to150.769599. Recovery and both component arms never ran.
The incomplete report is retained; only its17,947-byte summary was downloaded.
Scheduler retries `t99913`-`t99915` stopped at the preserved-incomplete-report
check, before new interactions. Actual formal cost:50,000 training interactions,
45,001 learner updates,252,000 evaluation interactions and4 initialization probes.

Root cause: Gym's EzPickle reconstruction discarded the base environment RNG.
The existing clone restored MuJoCo integration state but not the random state
used by the next unseeded episode reset. Restore that RNG and its seed metadata.
The old smoke run ended at its first resumed reset, hiding the bug; it now
performs a training update after that reset. A separate two-generation clone
test checks two unseeded resets and independent RNG objects. The new test fails
before the fix; all10 focused tests pass locally and remotely after it, including
exact actor/critic/encoder, return and selector parity. This turn's diagnostics
and tests cost305 training interactions,12,096 evaluation interactions and24
probes. No algorithm update rule or reproduction tolerance changed.

Retry the same frozen protocol in
`results/iclr_terminal_component_forks_20260921_resetfix`, preserving the old
root. Successful retry cost remains54,500 training /49,501 updates /1,008,000
evaluation /10 probes, in addition to the failed attempt above. Continue to
require both endpoint controls before interpreting either intervention.
Submitted `t99916` with the resetfix root; one dispatch, running on scheduler-
selected node004, no required/preferred node and no result auto-download.
Formal reproduction and component effects remain pending.

### Terminal component results (2026-09-21)

`t99916` completed with exit0, one dispatch and no retries. Both endpoint
controls reproduce all50 episode returns, final4x21 return grids, schedules
and block records exactly (maximum return errors0). The frozen prefix, budget,
evaluation grid, one-factor configurations and gradient records validate;
recomputing all16 mode HVs from the saved terminal returns matches the report.
Only the65,467-byte report was pulled; this readout adds no interactions/updates.

| Branch | Final HV (M) | Change vs factual | Per-mode HV (normal / low_speed / low_cost / locked), M |
|---|---:|---:|---|
| Factual extreme | 1.661048 | baseline | 2.331010 / .875786 / 3.283256 / .154140 |
| Full recovery | 4.426210 | +166.47% | 4.727871 / .910704 / 9.715556 / 2.350707 |
| Actor candidates5->1 only | 4.594243 | +176.59% | 4.914405 / .881035 / 10.105355 / 2.476178 |
| Relabel rate0->1 only | 1.472416 | -11.36% | 1.830516 / .853766 / 3.030301 / .175081 |

Actor-only exceeds full recovery final HV by3.80%, without changing critic
candidates, relabel rate/copies or BC. Final training-episode speed returns are
189.36/502.12/490.20/179.85 in table order. At100 updates the actor's distance
from the fork is.7013 factual versus.2623 actor-only; final distances1.3158
versus1.0298. These descriptive norms do not establish gradient explosion.

Limitations: one outcome-selected seed and one terminal block. Actor-only wins
the same-preference scalar utility at12/0/16/13 of21 weights per mode, despite
higher HV in all four modes. Low_speed's mean paired utility change is-155.742;
set coverage is not preference-conditioned improvement. Candidate reduction also
changes Torch random-number consumption, so the result identifies a useful local
configuration intervention, not a noise-isolated candidate-selection mechanism
or a validated adaptive selector. Existing five-seed/late-three gates stand.

Next bounded proposal: on the same48.5k prefix, retain the five-candidate draws
and their ordering but force the own-preference candidate for the actor update.
Reproduce the factual and current actor-only controls first; compare this
random-consumption-matched intervention with both, using unchanged final and
per-mode/paired-utility readouts. Freeze its full protocol and additional budget
before launch. Then consider the previously identified1003/1005 contrasts;
do not globally disable actor search or expand to20 seeds from this case.
No new jobs, main-algorithm changes or manuscript edits in this readout.

### Frozen random-consumption-matched terminal fork (2026-09-21)

`terminal_rng_matched_v1`: replay the same seed1004 blind prefix once through
48500, then run factual, actor_one and own_matched in that order for1500 updates
each. Reuse full-state cloning including both environment RNGs. Own_matched
retains all five candidate preference/action draws and critic evaluations in
legacy order, but selects candidate0 (the current preference) for the actor
loss. All scalar settings equal factual; relabel rate0, copies1, critic5 and BC0.
The intervention is confined to the diagnostic runner, not the main trainer.

Before the new branch, both controls must reproduce the completed resetfix
run's episode returns and terminal4x21 grids within1e-6, with identical schedules
and block records. Compare NumPy and Torch RNG states directly after every
training step against factual; keep traces in memory, record only mismatch
counts/first step. All1500 comparisons must match for a matched-draw claim.
This aligns random inputs, not evolving policies, replay values or trajectories.

Final-only evaluation remains four locked modes,21 weights,three1000-step
episodes and the same seeds/reference point. Report all three pairwise HV,
per-mode and matched-preference utility contrasts, including every negative
utility difference. Same gradient/drift diagnostics; no additional updates.
Budget:53,000 training interactions,48,001 learner updates,756,000 evaluation
interactions and8 initialization probes. Old attempts/tests remain separate.
Root: `results/iclr_terminal_rng_matched_20260921`; preserve incomplete roots.

Interpretation: a gain over factual supports a local selection effect under
factual random draws; a lost gain means candidate-count results alone do not
isolate selection from random-stream changes. A mixed HV/utility result remains
mixed. No tuned threshold, automatic deployment,20-seed expansion or paper edit.

Fifteen focused tests pass locally and remotely: exact candidate draws/order,
own-action loss/gradients, unchanged first critic update, per-update RNG parity
and mismatch detection, real cross-reset routing, negative-utility retention,
and completed/incomplete result handling. Verification cost across both hosts:
384 training interactions,21,168 evaluation interactions,38 probes and368 learner
updates (including16 synthetic-replay updates). No failed verification attempts.
Submitted `t99917`, running once on scheduler-selected node004 with no node pin
or preference and no result auto-download. Formal results remain pending.

### Random-consumption-matched result (2026-09-21)

`t99917` completed exit0 in one dispatch, no retries. Both controls reproduce
all episode returns and terminal grids exactly, with identical schedules/block
records. All1500 NumPy/Torch state comparisons match (zero mismatches). Budget,
configurations, all12 recomputed mode HVs and every paired return/utility contrast
validate. Pulled only the81,572-byte report; this readout adds no interactions.

| Branch | Final HV (M) | Change vs factual |
|---|---:|---:|
| Factual five-candidate selection | 1.661048 | baseline |
| Actor candidates5->1 | 4.594243 | +176.59% |
| Five draws, own candidate selected | 4.492407 | +170.46% |

Matched own-selection retains96.53% of the actor_one-minus-factual HV increment,
and is2.22% below actor_one in final HV. Its per-mode HVs are
4.340473/.859385/10.286150/2.483620M (normal/low_speed/low_cost/locked).
This supports a local actor-selection effect under factual random draws;
the rescue cannot be explained solely by removing candidate RNG calls.

Limitations: one outcome-selected seed/block, not an adaptive-policy validation.
Low_speed HV falls1.87%; all21 matched-preference utilities fall (mean-138.734).
Utility wins across modes are10/0/16/13 of21. Coverage improvement is not uniform
preference improvement; existing five-seed/late-three gates remain unchanged.
Actual new cost:53,000 training interactions,48,001 learner updates,756,000
evaluation interactions and8 probes; previous failed attempts/tests stay separate.

Next bounded proposal: factual versus RNG-matched own-selection from each blind
48.5k prefix on the previously identified1003/1005 contrasts. Preserve their
native relabel rate1, copies3 and BC.004; their terminal arm is
`extreme:edge_relabel_envelope_policy_bc`, unlike1004. Selection also determines
the action entering BC, so this tests the selected actor update including its
existing BC, not a BC-free replication. Require factual reproduction and RNG
alignment; report all modes/utilities. These are observed contrasts, not fresh
confirmation. No new jobs, algorithm adoption or manuscript edits this turn.

### Frozen native-BC contrast forks (2026-09-21)

`terminal_rng_contrasts_v1` runs seeds1003/1005 as two independent scheduler
tasks, with no node pin/preference. For each, replay its own v25 blind prefix
through48500 once, then factual and RNG-matched own-selection for1500 steps.
The native final arm is `extreme:edge_relabel_envelope_policy_bc`: relabel1,
edge sampling, copies3, critic5, vector coefficient.06, actor5, BC.004. Change
only the candidate selected for the actor update, retaining all five draws and
the selected action's BC term. Relabel copies expand both critic and actor
batches; they remain3. No use of the adaptive run as a same-prefix control.

Require the historical29 blocks/48 episodes/partial49th episode to reproduce,
then factual schedules/block records and all50 episode/final4x21 returns within
1e-6 before the intervention. Require all1500 NumPy/Torch state comparisons to
match. Final-only locked evaluation uses the original four modes,21 weights,
three1000-step episodes, seeds and reference point. Report both complete seed
contrasts, every mode and matched-preference utility, including negative values.

Per-seed budget:51,500 training interactions,46,501 learner updates,504,000
evaluation interactions and6 probes. Total:103,000 /93,002 /1,008,000 /12.
Root `results/iclr_terminal_rng_contrasts_20260921` has separate seed report/run
paths. Preserve incomplete attempts and account for tests/retries separately.
This is a heterogeneity check on observed cases with native BC, not independent
confirmation or a BC-free replication. A mixed result stays mixed; no selector
threshold tuning, automatic algorithm adoption,20-seed expansion or paper edit.

Twenty focused tests pass locally and remotely, including three-copy relabeling,
the selected action's exact BC gradient, first-critic parity, cross-reset native
control reproduction and RNG alignment, plus the unchanged1004 tests. Both
historical references validate. Verification cost:540 training interactions,
30,240 evaluation interactions,54 probes and524 learner updates (including32
synthetic-replay updates), with no failed tests. Submitted `t99936`/`t99937` for
1003/1005; each launched once on scheduler-selected node004/node006 respectively,
without node pin/preference or result auto-download. Formal outcomes are pending.

### Native-BC matched-selection results (2026-09-21)

Both `t99936`/`t99937` completed exit0 on their first dispatch, without retries.
Transient probe errors produced premature failed queue entries while the runs
continued. The matched launcher exit sentinels and complete reports establish
completion; corrected only these two task records, retaining crash forensics.
Both factual controls reproduce exactly, and all1500 NumPy/Torch comparisons
match per seed. Frozen configurations/budgets, all16 mode HVs and all paired
return/utility differences validate. Pulled only two reports totaling90,023 bytes.

| Seed | Factual HV (M) | Own-matched HV (M) | Change | Utility wins, normal/low_speed/low_cost/locked (of21) |
|---|---:|---:|---:|---|
| 1003 | 4.841335 | 5.428544 | +12.13% | 12/1/16/13 |
| 1005 | 3.928553 | 4.396265 | +11.91% | 10/3/17/9 |

Mode HV changes are +6.04/-2.00/+16.55/+13.45% for1003 and
+1.13/+0.54/+12.98/+34.70% for1005. Low_speed matched-utility means fall
49.011 and20.278 respectively. Thus1005 improves all four mode HVs without
uniform preference improvement. In low_cost, its pure-speed preference loses
1418.840 utility despite the mode's12.98% HV gain.

Zero-interaction grid diagnosis: for each saved21-point mean-return grid R,
define B(w)=max_j w^T R(w_j), A(w)=w^T R(w), G(w)=B(w)-A(w).
The exact finite-grid identity is delta A=delta B-delta G. In low_speed,
mean(delta B, delta G) is (-26.364,+22.647) for1003,
(-47.482,+91.252) for the earlier1004 case, and (-17.887,+2.391) for1005.
Both grid-best utility and preference correspondence deteriorate on average;
remapping alone cannot remove the coverage loss. Conversely,1005 low_cost
pure-speed grid-best utility rises393.938 while its own-conditioned output
loses1418.840: that endpoint loss concerns preference correspondence, not the
absence of a faster policy output in the sampled grid.

Interpretation: the local selection effect extends to these two native-BC
contrasts, beyond the selected1004 collapse. Limitations: all three seeds and
the terminal block were previously observed; BC also uses the selected action.
The grid maximum uses evaluation outcomes and is diagnostic only, not a
deployable selector or an estimate of optimal utility. No uniform-benefit or
fresh-confirmation claim; the five-seed/late-three gates remain unchanged.
Formal cost is103,000 training interactions,93,002 learner updates,1,008,000
evaluation interactions and12 probes. This readout adds no interactions or
updates; earlier verification costs remain separate.

Next bounded proposal: test the same RNG-matched intervention in prespecified
nonterminal blocks before designing a new adaptive rule. Keep native arms and
report grid-best plus matched-preference utility alongside HV, to distinguish
stage dependence from preference-mapping damage. Earlier full-training actor-off
results were mixed; do not infer that global removal or a gradient-norm cap is
validated by these terminal forks. Freeze block choices and budget before any
launch. No new jobs, algorithm changes or paper edits in this readout.

### Frozen earlier-block matched contrasts (2026-09-21)

`nonterminal_rng_contrasts_v1` selects the first complete1500-step occurrence
after20k and before48.5k of each seed's native terminal operator, using operator
metadata rather than returns. Frozen windows:1003 steps23001-24500 (native BC),
1004 steps29001-30500 (BC0/relabel0),1005 steps38001-39500 (native BC).
All original scalar settings, candidate draws/order and selected-action BC are
retained; only own_matched forces candidate0 for the actor update.

Run the factual path uninterrupted to its selected endpoint while cloning the
full state at the cut, then run one1500-step own_matched continuation. The
existing state-probe callback may explicitly return False to continue; its old
stop behavior is unchanged. Require exact historical prefix and factual
episode/block/schedule records, and1500 matching NumPy/Torch states per pair.
Use endpoint-only evaluation with its own encoder, four locked modes,21 weights,
three1000-step episodes and the existing evaluation seeds/reference point.
Historical actor snapshots used a later encoder and are not endpoint controls.

Report every endpoint/mode HV and paired preference utility, plus finite-grid
best utility and mapping-gap differences. Grid maxima use evaluation outcomes
only for diagnosis; no oracle is deployed. Budgets for1003/1004/1005 are
26,000/32,000/41,000 training interactions and21,001/27,001/36,001 updates;
each uses504,000 evaluation interactions and4 initialization probes. Total:
99,000 training /84,003 updates /1,512,000 evaluation /12 probes.
Root: `results/iclr_nonterminal_rng_contrasts_20260921`; preserve incomplete
attempts and count failed attempts/tests separately. No result auto-download.

Limitations: previously observed seeds and metadata-selected blocks, not fresh
confirmation. Training stage and encountered context both vary relative to the
terminal forks; this is not an isolated age effect. Mixed or negative results
remain reportable. A positive result does not authorize global actor removal,
a new adaptive selector,20-seed expansion or relaxation of the existing gates.
Freeze these choices before launch; no manuscript edits.

Twenty-five focused tests pass locally and remotely, including unchanged
stop-at-capture behavior, uninterrupted-capture parity, exact resumed control
tensors/returns/RNG for both native arms across resets, metadata-only window
selection and negative-value-preserving grid decomposition. Verification cost
across both hosts:852 training interactions,54,432 evaluation interactions,
86 probes and804 learner updates (including32 synthetic-replay updates).
All three server references validate; no failed tests. Submitted `t99956`-
`t99958` for1003-1005 respectively. Each launched once, on scheduler-selected
node005/node006/node005, without node pin/preference or result auto-download;
logs show6k steps. Formal controls, RNG alignment and effects remain pending.

### Earlier-block matched results (2026-09-21)

All three tasks `t99956`-`t99958` completed exit0 in one dispatch each, with no
retries. Prefix and uninterrupted factual episode/block/schedule records match
exactly; all1500 NumPy/Torch comparisons match per seed. Frozen configurations,
budgets, all24 recomputed mode HVs, paired utilities and finite-grid decompositions
validate. Pulled only the three161,489-byte-total seed reports. Actual cost:
99,000 training interactions,84,003 updates,1,512,000 evaluation interactions
and12 probes; this readout adds no interactions or updates.

| Seed / endpoint | Factual HV (M) | Own-matched HV (M) | Change | Utility wins, normal/low_speed/low_cost/locked (of21) |
|---|---:|---:|---:|---|
| 1003 /24.5k | 2.347980 | 2.458882 | +4.72% | 7/3/5/9 |
| 1004 /30.5k | .222671 | 2.461798 | +1005.58% | 9/3/13/9 |
| 1005 /39.5k | 3.701570 | 4.190933 | +13.22% | 12/1/12/6 |

Together with the terminal forks, all six observed seed/block contrasts improve
endpoint HV, but all six reduce low_speed mean matched-preference utility.
Earlier-block low_speed losses are22.104/81.279/31.011 for1003/1004/1005.
Seed1003 loses mean matched utility in normal,low_speed andlow_cost
(-6.548/-22.104/-11.174); its equally weighted84-point mean changes-4.996.
Its low_cost HV falls6.39%, and grid-best utility falls at all21 preferences
(range-93.261 to-2.421). This is not only preference-mapping damage.
For1004, even low_speed HV+171.87% coexists with mean utility-81.279.

Limitations: six correlated blocks from three previously observed seeds, not
six independent confirmations. All three earlier interventions occur in
training mode2 (low_cost); low_speed was unseen in these factual histories, so
its losses concern unseen-mode generalization, not established skill forgetting.
Native BC/relabeling differ across seeds. These results establish local effects
outside the final block, not a globally safe update or a validated selector.
Existing full-training actor-off negatives and five-seed/late-three gates stand.

Next bounded proposal: a small full-training pilot scoped to the two tested
extreme operators, leaving other actor-search arms and all critic/relabel/BC
settings intact. Preserve the original20k warmup, retain candidate RNG draws,
and first compare against the score-blind v25 control to isolate the update rule.
Freeze runs/budgets before launch and report final,late-snapshot,archive and
per-mode/preference outcomes; gains in endpoint HV do not erase utility losses.
Only a satisfactory full-training result would justify testing integration with
adaptive selection. No new jobs, algorithm adoption or manuscript edits here.

### Frozen scoped full-training pilot (2026-09-21)

`scoped_extreme_pilot_v1`: development seeds1003-1005,50k steps,25k mean dwell,
unchanged v25 score-blind selector and20k warmup. Afterwards select the own
actor candidate only for
`extreme:edge_envelope_policy` and `extreme:edge_relabel_envelope_policy_bc`.
Route by the live operator, retaining draws, critic/relabel/BC settings and all
other actor updates. BC uses the selected action. Expected target updates:
6000/4500/3000. Require all30k post-warmup RNG states and all20 block choices/
configurations to match the controls; no forced historical schedule or oracle.

Replay each control to50k while cloning its20k state, stopping before evaluation.
Require exact historical records, prefix archives, final policy/Q1/Q2/encoder
tensors and all19 actor snapshots against the server checkpoint before reusing
its evaluation. Continue the clone for30k steps. Evaluate19 snapshots with the
branch's final encoder, four locked modes,21 weights,three1000-step episodes
and unchanged seeds/reference point. Checkpoints/raw results stay server-side.

Per-seed new budget:80,000 training interactions,75,001 updates,4,788,000
evaluation interactions and4 probes. Total:240,000 /225,003 /14,364,000 /12.
Control replays add no evaluation. Tests/failed attempts and reused baseline
costs stay separate. Root: `results/iclr_scoped_extreme_pilot_20260921`;
preserve incomplete attempts.

Frozen gates: final-HV mean increases; observed final minimum, mean45k/47.5k/50k
HV and archive mean do not decrease; each mode's mean matched-utility difference
over21 weights and three seeds is nonnegative. Report all seed/mode/weight
differences and grid decomposition. Any failed gate blocks adaptive integration
from this pilot; no outcome-based rule/threshold tuning.

Limitations: observed seeds, not confirmation or adaptive-selection evidence;
final-encoder snapshots are not online recovery. Historical optimizer states
were not saved. A pass permits proposing an adaptive pilot, not adoption or
20-seed expansion. Old negative results/gates stand. Shared trainer and paper
are unchanged.

Five focused tests pass locally and remotely, including native loss/gradient/
RNG scope, uninterrupted checkpoint/archive reproduction, cross-reset scoped
continuation, incomplete-run preservation and negative-utility gate rejection.
Verification cost:276 training /260 updates /54,432 evaluation /12 probes;
no failed tests. Three source results validate; submitted `t99978`-`t99980`
for1003-1005 respectively, without node pin/preference or result auto-download.
Each launched once on scheduler-selected node004/node006/node005; logs show
control replay at7k/7k/8k. Formal reproduction, RNG alignment and effects pending.

### Scoped full-training readout and recovery (2026-09-22)

All three original tasks finished training/evaluation, then exited1 because
`validate_candidate` compared live tuple mode histories with JSON lists.
Fix this to exact array-value comparison; real schedule changes still fail.
Four postprocessing tests pass locally and remotely with zero interactions.
Recover reports into `results/iclr_scoped_extreme_pilot_20260922_recovered`,
preserving original incomplete reports/raw outputs/failed task states. All nine
automatic retries refused incomplete reports before training; no added cost.
Resolve the three APP_BUG_CAP escalations after recovery, without resubmission.

Controls reproduce all final networks and19 actor snapshots exactly; each
30k-step RNG trace has zero mismatches. Actual target updates6000/4500/3000,
all20 post-warmup block choices/configurations, full grids and budgets validate.
Only291,194 bytes of recovered reports/summary were pulled. Formal cost stays
240,000 training /225,003 updates /14,364,000 evaluation /12 probes; tests from
the launch turn remain separate. Recovery adds no training or evaluation.

| Metric, HV in M | Factual | Scoped own-selection | Change |
|---|---:|---:|---:|
| Final mean | 3.476979 | 4.428109 | +27.36% |
| Observed final minimum | 1.661048 | 4.060551 | +144.46% |
| Late-three mean | 4.018931 | 4.266745 | +6.17% |
| Archive mean | 4.841871 | 4.828485 | -0.28% |

Seed1003/1004/1005 final gains are+0.31/+162.92/+3.36%;1004 contributes94.84%
of the summed gain. Seed1003 late-three HV falls1.92%. Mean matched-utility
changes across seeds, normal/low_speed/low_cost/locked, are
+90.963/-54.298/+326.661/+131.867. Low_speed loses at21/20/16 of21 weights
for1003/1004/1005. Its weighted speed/cost contributions average
(+12.430,-69.911)/(+46.001,-136.702)/(+13.210,-27.923): added cost outweighs
speed gains. Grid-best low_speed utility also falls in every seed.

Decision: frozen archive and per-mode utility gates fail; do not advance this
fixed rule to adaptive integration or20-seed confirmation. Next algorithm work
must address the speed/cost tradeoff while retaining the observed collapse
recovery, not tune these failed gates. Limitations: three observed seeds,
low_speed unseen during training, and final-encoder snapshot evaluation; this
does not establish population robustness or old-mode forgetting. No new jobs,
main algorithm adoption or manuscript edits in this readout.

### Scoped adaptive confirmation (2026-09-22)

After reviewing the completed pilot, accept its small archive tradeoff for
further study; retain the original failed gates and reports unchanged. Archive
mean delta is -13,386 HV: normal/low_speed/low_cost/locked contributions are
-63,108/-2,299/+39,098/+12,922. Low_speed accounts for17.17% of the net decline,
not its sole source. Its final HV improves in two seeds despite matched-utility
losses. Do not treat these metrics as interchangeable or claim a causal
decomposition of the cost regression. Defer low-speed-specific loss changes;
report this tradeoff in the next experiment instead of adding another factor.

Freeze `scoped_adaptive_confirmation_v1`: seeds1006-1010, selected without
inspecting outcomes or mode schedules; local/remote result names and current
scheduler records contain no prior matches. Twenty runs cross legacy/scoped
updates with v25 progress adaptive/blind selection. Keep the50k training,
25k dwell,20k warmup, native actor/critic settings and existing feedback signal.
Scoped updates retain all candidate draws and choose candidate0 only on the
same two post-warmup extreme arms. The live selector can change its choices;
do not force historical blocks or claim cross-arm global RNG alignment.

Primary update contrast: scoped adaptive minus legacy adaptive final-policy
HV. Adaptive benefit is a separate contrast against scoped blind; include both
other simple effects and the factorial interaction. Report every paired seed,
mean/median differences, observed minima and the fixed45k/47.5k/50k average.
Archive and per-mode matched utility are secondary tradeoffs, not zero-tolerance
vetoes. Five-seed paired bootstrap intervals are exploratory, not equivalence
or population-robustness claims. No automatic adoption gate or post-result
parameter tuning; judge the full frozen comparison before expanding.

Evaluation remains19 snapshots x4 locked modes x21 weights x3 episodes x1000
steps, with each branch's final encoder and identical paired evaluation seeds.
Planned formal cost:1,000,000 training /900,020 updates /95,760,000 evaluation
/40 initialization probes. Tests and interrupted attempts stay separate.
`run_scoped_adaptive_confirmation.py` preserves interrupted attempts and reuses
saved terminal results after postprocessing failures. Scheduler chooses among
eligible HPC nodes without a required/preferred node; checkpoints/raw grids stay
remote and only compact reports are retrieved. Root:
`results/iclr_scoped_adaptive_confirmation_20260922`. Shared trainer and paper
remain unchanged; this tests integration with the existing progress selector,
not a newly validated independent return evaluator.

Four focused tests pass both locally and on the server, including actual
four-arm short training. Combined test cost:488 training /456 updates /18,144
evaluation /16 probes. Submitted `t100022`-`t100041`, four conditions in the
order above for each seed1006-1010, with no node pin/preference or result
auto-download. Formal outcomes remain pending.

### Scoped adaptive confirmation readout (2026-09-22)

All20 original tasks completed with exit0, one launch each and no retries.
The frozen summary validates complete pairs, warmup histories, schedules,
evaluation metadata and budgets. Formal cost is1,000,000 training /900,020
updates /95,760,000 evaluation /40 probes; prior launch-test costs stay separate.
Only the71,331-byte `results/iclr_scoped_adaptive_confirmation_20260922/readout.json`
was pulled; the full summary, raw grids and models remain on the server.

| Condition, HV in M | Final mean | Final minimum | Late-three mean | Archive mean |
|---|---:|---:|---:|---:|
| Legacy blind | 3.051630 | 1.030181 | 2.721019 | 3.862524 |
| Legacy adaptive | 2.907568 | 1.030181 | 2.815673 | 3.787022 |
| Scoped blind | 2.990904 | 1.114722 | 2.665608 | 3.961750 |
| Scoped adaptive | 3.407581 | 2.747500 | 3.023157 | 4.031887 |

Primary scoped-versus-legacy adaptive: final mean+17.20%, minimum+166.70%,
late mean+7.37%, archive+6.47%. Paired final deltas for1006-1010 are
0/+2.301945/-0.298944/+0.628376/-0.131312M; two wins, two losses, one tie.
Seed1007 contributes92.08% of net gain; median delta is0 and the exploratory
95% paired-bootstrap mean interval is[-0.172102,1.447054]M. Scoped adaptive
versus scoped blind improves final mean13.93%, minimum146.47%, late mean13.41%
and archive1.77%, with one win, one loss, three ties and interval
[-0.080414,1.330443]M. Scoped updates alone lower blind final mean1.99%.

Seed1006 has zero targeted post-warmup updates and all four runs coincide.
For1007, scoped adaptive/blind records match before47,001; only the last two
operator blocks differ. At47.5k both HVs are about3.97M; final values are
3.332127/1.114722M. This is another lower-tail recovery case, not proof that
one isolated operator caused it. Snapshots still use each branch's final encoder.
Primary mean matched-utility deltas by mode are-66.486/-4.068/+53.066/+8.890;
low_speed has two wins, two losses and one tie, not uniform degradation.

Decision: retain the frozen scoped-adaptive candidate and defer low-speed
specialization. The joint configuration has positive aggregate evidence, not
uniform preference improvement or established population robustness. Next work
should extend frozen seed/cross-environment confirmation, not tune this cohort.
No new training, main-trainer adoption or manuscript changes in this readout.

### Frozen seed and three-objective extension (2026-09-22)

`scoped_extension_v1` keeps all four legacy/scoped x blind/adaptive conditions,
the same two targeted arms after20k, native candidate draws, feedback and all
training settings. No new actor/critic loss, selector parameter or mode-specific
rule. HalfCheetah adds seeds1011-1025:60 runs. Hopper and Ant each use
seeds1011-1015:20 runs each. No prior matches were found in local/remote result
names or current BAMOR task records before inspecting any new outcome/schedule.

Primary HalfCheetah confirmation is the new15-seed cohort alone. Report the
observed1006-1010 cohort separately; combined20-seed estimates are descriptive
because expansion followed inspection of the first five. Hopper/Ant are
five-seed transfer checks, not20-seed confirmation. Retain the frozen primary
scoped-adaptive versus legacy-adaptive final-HV contrast, the separate selector
contrast against scoped blind, all paired effects, minima, late-three metrics,
archive and per-mode utility. Do not pool raw HV across environments or adjust
the algorithm after inspecting partial results.

All100 new runs train50k steps with25k dwell, hidden64, batch256, lr3e-4,
gamma.99, learning starts5k and two CPU threads. Evaluation stays19 snapshots,
21 full-simplex preferences and three locked episodes per mode, capped at1000
steps each. Existing presets: HalfCheetah2 objectives/4 modes, Hopper3/4,
Ant3/3. Respect natural episode termination and record actual interaction
counts. New total:5,000,000 training /4,500,100 updates /200 initialization
probes, at most454,860,000 evaluation interactions. Reused20 runs and test or
interrupted-attempt costs remain separate. Final encoders are still used for
snapshot evaluation; reward caps/scaling in the existing presets are unchanged.

Implementation only extends experiment entrypoints and validation/aggregation
to the declared dimensions and mode counts. Native training is unchanged.
Root: `results/iclr_scoped_extension_20260922`. Scheduler selects any eligible
HPC node without pin/preference; no checkpoint/raw-grid download or automatic
result sync. Keep incomplete attempts and reuse saved terminal results after
postprocessing failures. Do not edit the paper on launch success.

Eight local tests pass, including full2-D/3-D grids, routing, actual budgets,
saved-result recovery and separate-cohort aggregation. The two real-training
tests also pass remotely. Recomputing the saved scoped-adaptive seed1007 report
matches it exactly, without new interactions. Combined test cost:1,464 training
/1,368 updates /49,896 evaluation /48 initialization probes. No failed tests.

Submitted100 tasks: HalfCheetah `t100124`-`t100183`, Hopper `t100184`-`t100203`,
Ant `t100204`-`t100223`. Within each environment, seed-major order with
legacy blind/adaptive then scoped blind/adaptive. No required/preferred node,
checkpoint download or result auto-sync. Formal outcomes pending.

### Frozen extension readout (2026-09-22)

All100 tasks completed with exit0 and no requeued training tasks. The frozen
aggregation validates all new pairs and the20 reused runs. New formal cost:
5,000,000 training /4,500,100 updates /400,052,011 evaluation /200 probes.
Reused runs and launch tests retain their separately recorded costs above.
This readout adds no interactions or updates. Only the469,730-byte
`results/iclr_scoped_extension_20260922/readout.json` is retrieved locally;
full summaries, raw evaluations and models stay on the server.

Primary contrast is scoped adaptive versus legacy adaptive. Percentages below
compare cohort summaries within each environment; W/L/T counts paired final HV.

| Cohort | Final mean | Final minimum | Late-three mean | Archive mean | W/L/T |
|---|---:|---:|---:|---:|---:|
| HalfCheetah fresh15 | +0.96% | -0.04% | +4.36% | +1.54% | 9/4/2 |
| Hopper fresh5 | -10.90% | -36.60% | -9.61% | -9.41% | 3/1/1 |
| Ant fresh5 | +3.04% | +10.15% | +1.91% | -1.37% | 3/1/1 |
| HalfCheetah pooled20, descriptive | +4.99% | -0.04% | +5.07% | +2.73% | 11/6/3 |

The fresh HalfCheetah cohort does not reproduce the original five-seed minimum
advantage. Its primary paired mean-delta95% bootstrap interval is
[-0.388655,+0.394999]M. Hopper and Ant intervals are respectively
[-1.813761,+0.505996]G and[-0.008639,+0.109595]G; all include zero.
Separately, scoped adaptive versus scoped blind changes final mean by-0.38%
on fresh HalfCheetah,-1.02% on Ant and+42.49% on Hopper. The Hopper selector
contrast has one win, two losses and two ties, driven by seed1014; it does not
establish superiority over legacy adaptive.

Existing-record diagnosis finds HalfCheetah1011 final HV-61.13% and
Hopper1013-76.75%, both with identical legacy/scoped adaptive operator choices.
HalfCheetah1011 falls from3.592M at47.5k to1.353M at50k; legacy ends3.481M.
Hopper1013 loses final HV in all four modes and its last five completed training
episodes have lengths918/697/290/1000/521 versus1000 throughout for legacy.
These are substantive negative cases, not merely archive-mean noise.

Limitations: snapshots use each branch's final encoder and are not online
learning curves; identical operator sequences do not imply identical data or
isolate one harmful update. Five-seed transfers and observed minima do not
establish population robustness, and intervals spanning zero do not establish
equivalence. The pooled20 estimate remains descriptive, not a replacement for
the frozen fresh15 primary comparison.

Decision: retain this as a mixed-result research branch, not a universal
upgrade or established anti-collapse method. Next isolate the two targeted
update arms on existing negative and positive cases before another expansion;
prioritize that mechanism over low-speed-specific tuning or small archive
changes. No new training, default-algorithm adoption or paper edits this turn.

### Frozen two-arm attribution (2026-09-22)

`scoped_arm_factorial_v1` splits the two operator-specific own-candidate rules:
`envelope_only` changes only `extreme:edge_envelope_policy`; `bc_only` changes
only `extreme:edge_relabel_envelope_policy_bc`. The latter is an operator-level
intervention, not a change to the BC loss alone. Retain all candidate draws,
native critic/relabeling/BC settings,20k warmup and live progress feedback.
The original default router still targets both arms.

Selected cases: HalfCheetah1011/1018 and Hopper1013/1012, respectively the
largest negative/positive final-HV differences in each new cohort. All four
existing endpoint pairs have identical operator sequences. These are deliberately
outcome-selected development cases, not fresh confirmation or a population sample.
Add only the two missing cells per case:8 runs. Reuse the8 legacy/scoped adaptive
endpoints from `scoped_extension_v1`; no baseline retraining or new Ant cohort.

Report each case's E-minus-neither, B-minus-neither, both-minus-B and
both-minus-E effects, plus interaction=both-E-B+neither. Final HV is primary;
retain late-three, archive, per-mode matched utility and operator-sequence
differences. Do not constrain subsequent adaptive choices. This identifies
closed-loop arm effects, not a fixed-data gradient decomposition. Keep positive
and negative cases together; a repaired negative case alone is insufficient.

Training/evaluation protocols are unchanged:50k steps,25k dwell, two CPU threads,
19 snapshots x4 modes x21 preferences x3 episodes, capped at1000 steps each.
New formal budget:400,000 training /360,008 updates /16 initialization probes,
at most38,304,000 evaluation interactions; count actual Hopper terminations.
Reused evidence and launch tests stay separate. Root:
`results/iclr_scoped_arm_factorial_20260922`. Use scheduler-selected eligible HPC
nodes without pin/preference; keep raw evaluations/models remote. No main-trainer
or manuscript changes, environment whitelist, new selector or automatic adoption.

Eight local Python3.10 tests pass, including real2-D/3-D short training, scope,
default-route regression, complete factorial arithmetic and saved-result recovery.
The real-training test also passes on the server. Combined launch-test cost:
488 training /456 updates /18,144 evaluation /16 initialization probes. The first
local command used Python3.8 and stopped at parsing, with zero interactions.
All8 saved reference reports are available with matching paired warmup/evaluation
metadata. Both target arms occur in every selected reference case.

Submitted `t100373`-`t100380`: HalfCheetah1011/1018, then Hopper1013/1012;
within each case, envelope-only then BC-operator-only. No required/preferred node,
checkpoint download or automatic result sync. Formal outcomes pending.

### Two-arm attribution readout (2026-09-22)

All8 conditions are complete. Original tasks saved full training/evaluation data,
then failed when live tuple mode histories were compared with JSON lists. Their8
successful recovery tasks reload those saved results; recovery logs contain no
training/evaluation execution. Preserve the original failures. Fix `validate_pair`
with array-value equality; fresh-live-result and saved-result recovery tests pass
locally and remotely, including rejection of genuinely changed switch times/modes.
These mocked regression tests and this readout add no environment interactions.
Formal cost:400,000 training /360,008 updates /31,732,554 evaluation /16 probes;
the8 reused endpoints and previously recorded launch tests remain separate.

Final-HV changes below are relative to each case's legacy adaptive endpoint:

| Case | Envelope-only | BC-operator-only | Both |
|---|---:|---:|---:|
| HalfCheetah1011, negative | -22.42% | -50.86% | -61.13% |
| HalfCheetah1018, positive | +146.58% | -2.74% | +87.20% |
| Hopper1013, negative | +52.01% | +6.44% | -76.75% |
| Hopper1012, positive | +0.68% | +52.19% | +32.22% |

All four configurations have identical operator sequences in each case.
HC1011 suffers from both individual changes; HC1018 benefits mainly from E.
Hopper1013 exhibits strong negative final-HV interaction:both-E-B+neither=-5.696G;
Hopper1012 benefits mainly from B. Neither single-arm variant preserves every
positive-case gain while repairing both negative cases. Both-on is never the
best final-HV cell here, but B is not uniformly harmful and should not be deleted
based only on the two HalfCheetah cases.

Limitations: these are four outcome-selected cases, not a population estimate.
On Hopper1013, E-only improves late HV4.23% but reduces archive HV12.59% and
mean matched utility in all four modes; its final-HV gain is not universal
preference improvement. Snapshot evaluations still use final encoders, and
matching operator sequences does not establish cross-branch RNG alignment.

Decision: keep the interaction diagnosis, not a universal new default. Next
assess whether training-only feedback can distinguish these update benefits
before designing state-dependent within-operator selection. Avoid environment/
seed lookup rules or another unconditional both-on expansion. Compact evidence:
`results/iclr_scoped_arm_factorial_20260922/readout.json`; full summary/raw data
remain remote. No new formal training or manuscript changes in this readout.

### Frozen native-feedback diagnosis (2026-09-22)

`scoped_arm_feedback_readout_v1` reads the same16 completed runs. No training,
evaluation rollouts, candidate fitting or new selector. Before reading block
scores, fix native terminal per-arm EMA as the primary feedback proxy; mean
and first post-warmup target-block progress are secondary diagnostics. Compare
their signs with final/late/archive effects for the four conditional E/B
contrasts in each case. These16 contrasts are dependent descriptive comparisons,
not16 independent replicates or an online prediction-accuracy estimate.

Reconstruct native choice probabilities from completed prior blocks only;
compare adaptive versus blind probabilities on each factual prefix and between
factorial cells. Retain family-quota effects, feedback ages and mode-switch
annotations. True modes only annotate results; no evaluation outcome enters
feedback/probability reconstruction. Counterfactual score differences from two
separate training runs are not available to a single online selector.

The current selector keys scores by operator name, not native/own update variant.
This diagnosis separates action-space limitations, score direction and probability
sensitivity before proposing within-operator adaptation. No temperature tuning,
environment/seed lookup, automatic adoption or manuscript change.

### Native-feedback result (2026-09-22)

The saved-data audit covers16 runs and320 post-warmup decision prefixes. All9
focused tests pass locally and remotely; training/evaluation interactions and
learner updates added by this step are zero. Native score/EMA recomputation
matches the stored records, and prefix reconstruction excludes current/future
feedback. Full block-level evidence remains on the server.

Primary terminal arm-score differences agree with final-HV effects in6/16
conditional contrasts:HC1011 3/4, HC1018 2/4, Hopper1013 1/4, Hopper1012 0/4.
Agreement is7/16 for late HV and12/16 for archive HV. Mean target progress gives
the same counts; first-target progress gives6/16,5/16,8/16 respectively. These
dependent, selected-case counts are not prediction accuracy or statistical tests.

For Hopper1013, adding B to E raises B's native terminal score by0.005620 while
final HV falls84.71%. E's own/native score difference is identical with and
without B(-0.008865), despite opposite final effects(+52.01% versus-78.16%).
E is last observed at26k and B at36.5k: their terminal credits are24k/13.5k steps
old. None of their target blocks crosses a true mode switch. Merely filtering
switch-crossing blocks would not resolve this example.

Mean native-versus-blind probability TV ranges0.00894-0.02453 across runs.
Family quotas constrain64/320 prefixes; all factorial cells retain the same
realized operator sequences. Probabilities do respond to feedback, but the
selector has no separate original/own update identities. Neither weak probability
movement alone nor a missing variant identity explains the score/outcome mismatch.

Decision: do not plug native progress directly into a variant bandit, lower its
temperature or reverse its sign to fit these cases. Prior128-update/512-step
budgeted and impulse return routers also failed; they are not a ready replacement.
Next study should isolate a single1500-step B intervention from a common state
in the two Hopper cases, then restore the same continuation rule and measure
immediate versus delayed policy retention with contemporaneous encoders. This
would test the missing credit horizon, not replay the existing both-on matrix;
all reproduction, continuation and probing interactions must be budgeted before
launch. No new router, formal training or manuscript changes in this step.
Compact evidence: `results/iclr_scoped_arm_feedback_20260922_readout.json`.

### Frozen single-block credit study (2026-09-22)

`scoped_bc_credit_v1`: Hopper1013 reproduces E-only through32000, then compares
native B versus own-candidate B on32001-33500; Hopper1012 reproduces legacy
through24500 and compares the same change on24501-26000. Each prefix is trained
once and cloned with replay, optimizers, encoder, partial episode and RNG state.
After that one block both branches restore their background update rule and
continue to50000 with live adaptive selection. Later choices may diverge.

Evaluate the common fork once, then each branch at block end, one further block,
40000 and50000: current policy with its contemporaneous encoder,21 simplex
weights x3 paired reset seeds x4 locked modes. Evaluation is RNG-isolated and
never used by training/selection. Read block progress versus immediate/delayed
HV, matched utility, objective means and subsequent operator divergence.
Factual episode/block histories and final returns must reproduce existing raw
results; no historical policy archive is evaluated and no models are downloaded.

Two-job budget:143500 training interactions,133502 learner updates, at most
4536000 evaluation interactions (actual Hopper lengths recorded),8 initialization
probes. Local smoke separately used98 training,90 updates,12096 evaluation and8
initialization interactions;4 tests passed. This is a two-case mechanism
diagnosis, not fresh confirmation or an online routing algorithm. Completed
branches can be re-summarized without training; incomplete attempts are retained.
Remote smoke also passed4 tests (98/90/12096/8 in the same budget order).
Its earlier test-fixture error consumed41/37/1512/2 before correction; no formal
training was launched by either smoke. Scheduler tasks:1013=`t100421`,1012=`t100422`,
2 CPU threads and2GiB each, eligible on node001-node007 with no node preference.

### Single-block credit result (2026-09-22)

Both tasks completed; factual reproduction passed and subsequent operators
matched. Actual cost:143500 training,133502 updates,3496610 evaluation,8 probes.

| Case | Block-end HV delta | One block later | 40k | 50k | Block progress delta |
|---|---:|---:|---:|---:|---:|
| Hopper1013, E-only background | +0.24% | +0.63% | -4.59% | -20.74% | +0.04720 |
| Hopper1012, legacy background | +16.92% | +0.20% | +16.40% | +0.23% | -0.19626 |

1012 shows immediate score/outcome mismatch: native objective2 rises0.0934->0.2023
between halves; pulse is already higher at0.2800->0.2806. Block slope misses early
benefit. For1013, final matched utility rises19.63% but posthoc front-envelope
EUM falls12.63%;81/84 grid utilities decline when selecting the best measured
conditioning preference, rather than executing the requested preference directly.

Next: retain HV as primary; do not adopt the pulse or only lengthen credit windows.
Isolate preference-coverage retention after updates, holding selection fixed.
Limitations: two selected cases; actor/critic/encoder/replay causes remain unsplit.
Three tests passed; readout adds zero interactions. Compact evidence (28KB):
`results/iclr_scoped_bc_credit_20260922/readout.json`.

### Frozen preference-retention test (2026-09-22)

`preference_retention_v1`: reuse the two credit-study native/pulse controls.
Reproduce each prefix once; fork coverage versus current-preference retention
during the same1500-step own-candidate BC pulse, then remove the penalty.
Teacher is the frozen pre-block actor. Add coefficient1 mean-action MSE on the
original256 replay rows/contexts: all7 existing simplex templates versus7
repetitions of the first relabel copy's current preference. Both use1792 action
pairs per update. Critic, BC, encoder and live selector rules stay unchanged;
later realized operators may differ. No evaluation returns enter the update.

Keep the same contemporaneous evaluation times. Primary comparisons: final HV
versus both the unprotected pulse and current-preference control. Inspect all
modes, utility and fixed-context action drift; do not tune the coefficient on
these two selected cases. This tests actor retention, not a return guarantee.
New budget:143500 training,133502 learner updates, at most4536000 evaluation,
12 initialization probes;6000 auxiliary-loss evaluations and10752000 action
pairs, without extra optimizer steps. Source experiment costs remain separate.
Five tests passed locally/remotely. Validation, including a corrected router-field
collision:274 training,268 updates,37800 evaluation,26 probes, recorded separately
in `results/iclr_preference_retention_20260922/validation.json`.
Tasks:1013=`t100454`,1012=`t100455`;2 CPU threads each, submitted RAM2GiB
(scheduler launch estimate512MiB), no required/preferred node.

### Preference-retention result (2026-09-26)

Both tasks completed. Prefix checks and saved summaries passed; each retention
branch applied1500 losses, and later operator identities matched the pulse.
Final mode-mean HV (billions):

| Hopper seed | Native | Unprotected pulse | Current-preference | Seven-template |
|---|---:|---:|---:|---:|
| 1013 | 6.404 | 5.076 | 5.294 | 2.957 |
| 1012 | 1.962 | 1.966 | 1.956 | 1.964 |

For1013, seven-template retention cuts final HV44.13% versus the equally sized
current-preference control and53.82% versus native. Modes0/2 account for most
of the loss. Fixed-context action MSE is0.16765 versus0.17565, so a slightly
smaller action drift did not retain return coverage. At the block end, coverage
and current differ by only-0.02% HV; the damage appears later. For1012, final
coverage versus current is+0.46%, with all four strategies near1.96B. The
pre-block teacher HV is3.52x below the native50k HV for1013 but only1.04x below
for1012, consistent with the risk of anchoring to a weak early actor.

Decision: reject this coefficient-1 frozen-teacher rule; do not tune the same
two selected cases further. Equal forward-pair counts do not establish equal
regularizer gradient magnitudes. Matching operator identities does not freeze
subsequent replay, critic or encoder states. These cases diagnose a failure mode,
not population performance. Actual additional cost:143500 training,133502
updates,3528344 evaluation,12 initialization probes,6000 auxiliary-loss
evaluations and10752000 action pairs. Compact evidence (7.5KB):
`results/iclr_preference_retention_20260922/readout.json`.

### Frozen archive-matched external baseline (2026-09-26)

Stop tuning the failed retention branch. Complete the existing scoped-extension
cohort with `capql_archive` only: HalfCheetah seeds1011-1025, Hopper and Ant
seeds1011-1015. Use the existing morl-baselines CAPQL implementation unchanged,
50k training interactions,25k mean dwell, 19 snapshots at5k:2.5k:50k, and no
intermediate evaluation. Locked modes,21 full-simplex weights, three paired
resets and environment reference points match the completed BAMOR runs.

Primary contrast within each environment: v25 legacy adaptive minus CAPQL
archive, mode-mean archive HV. Report final-actor HV, every seed, paired win/loss,
per-mode differences, training/evaluation time and actual interactions. Legacy
blind and scoped adaptive comparisons are secondary; do not choose a winner
after seeing CAPQL outcomes or pool raw HV across environments. The BAMOR
cohorts were inspected before this baseline was added, so this is retrospective
baseline completion, not a fresh prospective confirmation. The pre-existing
adaptive-versus-blind results remain mixed whatever CAPQL scores.

Planned new cost:25 runs,1,250,000 training interactions, at most113,715,000
evaluation interactions,50 initialization probes. Check all25 results before
interpretation. Keep raw fronts/checkpoints on the server; retrieve only the
compact paired readout. Scheduler chooses any eligible CPU node without a node
preference. No algorithm or manuscript changes follow merely from job launch.

One paired-readout test passed locally and remotely; the remote CAPQL import
preflight passed with the existing morl-baselines checkout on `PYTHONPATH`.
Submitted25 tasks: HalfCheetah `t100551`-`t100566` except `t100556` (an unrelated
scheduler ID), Hopper `t100567`-`t100571`, Ant `t100572`-`t100576`. All25 are
running on scheduler-selected nodes without a node pin/preference. The compact
readout is generated only after all25 terminal outputs validate; no result or
manuscript claim is inferred from launch status.

### Archive-matched external baseline result (2026-09-26)

All25 tasks finished; the server-side readout validates matching snapshots,
full-simplex weights, reset seeds, reference points and training mode schedules.
Actual new cost:1,250,000 training /97,917,947 evaluation interactions /50
initialization probes. Only the53,765-byte
`results/iclr_capql_archive_extension_20260926/readout.json` was retrieved;
raw fronts and models remain remote. No new training was used in the following
snapshot diagnosis.

| Cohort | v25 adaptive - CAPQL archive HV | Archive W/L | Final-actor HV | Final W/L |
|---|---:|---:|---:|---:|
| HalfCheetah fresh15 | +0.554M (+16.21%) | 10/5 | +0.022M (+0.74%) | 8/7 |
| Hopper fresh5 | +3.669B (+140.39%) | 5/0 | +1.213B (+58.67%) | 3/2 |
| Ant fresh5 | +0.025B (+1.17%) | 3/2 | -0.216B (-11.07%) | 0/5 |

Exploratory paired-bootstrap95% mean-delta intervals for archive HV are
[-0.112,+1.169]M on HalfCheetah,[+2.490,+4.801]B on Hopper, and
[-0.152,+0.178]B on Ant. The Ant final-actor interval is
[-0.333,-0.121]B; every one of its15 seed-by-mode final-HV cells loses to
CAPQL. Small cohorts and retrospective baseline addition limit inference.
CAPQL is also faster: BAMOR/CAPQL mean training-time ratios are1.91/1.69/1.65
and evaluation-time ratios1.74/1.79/1.45 for HalfCheetah/Hopper/Ant.

Using the existing final-encoder fronts, Ant's best single BAMOR snapshot is
at10k-15k in all five seeds; every mode's best snapshot is at or before22.5k.
A fixed15k snapshot beats its own50k actor in all five Ant seeds. It loses to
50k in all15 HalfCheetah seeds and four of five Hopper seeds, so fixed early
stopping is not a cross-environment remedy. These snapshot choices are posthoc
diagnostics, not a deployable model selector or independent confirmation.

Decision: v25 has an observed archive-coverage advantage over CAPQL on Hopper, but
neither broad adaptive-selector superiority nor a universal final-policy
advantage is established. Ant's terminal-policy regression and BAMOR's higher
runtime remain central submission gaps. Do not adopt the scoped or frozen-
teacher branches, choose an environment-specific checkpoint from these outcomes,
or change the manuscript to claim a cross-environment win. The next algorithm
question is whether training-only evidence can retain/select a useful actor
without evaluation feedback or uncounted deployment interactions; test that
before another large seed expansion.

### Frozen training-only checkpoint-proxy diagnostic (2026-09-26)

On the already inspected25 legacy-adaptive extension runs, define one causal
score for each of the19 saved actors: hypervolume of completed training-episode
return vectors whose terminal step is in `(snapshot_step-5000,snapshot_step]`,
using the preset fixed environment reference point. Select the highest-scoring
snapshot, breaking ties toward the earliest. The rule does not use true mode
labels, locked-mode fronts, final-actor returns, CAPQL, or future episodes.

Evaluate the selected actor's existing final-encoder locked-mode single-policy
HV against the50k actor, per environment and per seed. Report score ties,
window episode counts and the hindsight-best single actor as a diagnostic
ceiling, not as a selector. This cohort and its best snapshots were already
inspected, so this is a feasibility check only. If it fails to improve Ant or
systematically damages HalfCheetah, reject the proxy without tuning the window
on these runs. No new training/evaluation interactions, checkpoint download,
operator changes or manuscript changes.

### Training-only checkpoint-proxy result (2026-09-26)

All25 existing runs were processed server-side; two causal/snapshot tests pass
locally and remotely. Added training/evaluation interactions:0/0. Only the
34,078-byte `results/iclr_train_only_snapshot_proxy_20260926_readout.json`
was retrieved; models and raw fronts stay remote.

| Cohort | Selected minus 50k single-actor HV | W/L/T | Mean selected / hindsight-best HV |
|---|---:|---:|---:|
| HalfCheetah15 | -0.148M | 2/5/8 | 0.682 |
| Hopper5 | +0.883B | 2/1/2 | 0.708 |
| Ant5 | +0.121B | 4/1/0 | 0.913 |

On Ant, the training-only rule selects30k-42.5k in four of five seeds,
although the hindsight-best single actor is at10k-15k in all five. It lifts
the Ant final-actor mean from1.735B to1.855B, still below CAPQL's1.951B.
HalfCheetah falls from2.940M to2.792M, also below CAPQL's2.918M. These
figures evaluate old actors with each run's final encoder, not an online
deployment procedure. Training episodes are stochastic, preference-mixed and
can straddle mode changes; their window HV is not a deterministic-policy
coverage estimate.

Decision: reject the frozen5k training-episode HV selector. Do not tune its
window or objective scaling on these inspected seeds and do not integrate it
into BAMOR. The next credible actor-retention test would need independent,
budgeted policy-performance evidence and a fresh confirmation cohort; the
previous budgeted operator-return routers are not validated substitutes.

### Frozen independent actor-probe pilot (2026-09-26)

`archive_actor_probe_v1` reuses15 completed v25 legacy-adaptive models (seeds
1011-1015 in each of the three environments), without retraining. Candidate
actors are fixed at10k/25k/40k/50k with the
stored final encoder. Probe preferences are the two2-D vertices plus midpoint,
or the three3-D vertices plus all permutations of(0.2,0.4,0.4); these are
members of the existing21-point evaluation grid. This is an inspected-seed
development pilot, not independent confirmation or an online training selector.

For every seed, preference and actor, instantiate a fresh unlocked piecewise
environment (exponential mean dwell5k; seed4000000+1000*train_seed+weight_index).
Run deterministic episodes at that preference until at least10k interactions;
reset seed per episode is5000000+10000*train_seed+100*weight_index+episode_index.
Select the actor with highest total preference-scalarized reward per actual
probe step, earliest candidate on a tie. The selector sees no mode identity,
locked-mode returns, test seeds, critic or training episode proxy. Different
termination times may cause different reset counts; report actual interactions
and mode exposure. Each candidate has its own identical-seeded environment.

Primary test metric is mean same-preference utility of the selected actors on
the already saved, disjoint-reset locked-mode fronts, contrasted with the50k
actor. Report each seed and each environment, wins/losses and the hindsight-best
candidate as a diagnostic ceiling. Subset-grid mode-wise HV is secondary; do
not compare its numeric value to full21-weight HV. Probe interactions are
additional evaluation cost, not free training evidence. Planned300 probe cells
cost3,000,000 to at most3,299,700 evaluation interactions plus300 initialization
probes; new training and learner updates are zero. If Ant gains only by hurting
HalfCheetah or if probe cost dominates the gain, do not integrate or tune on
these seeds. Models/raw fronts remain remote; pull only compact reports.

### Independent actor-probe pilot result (2026-09-26)

All15 scheduler tasks `t100628`-`t100642` finished without failures; the first
full task was checked before dispatching the other14. Final-actor reproduction
in HalfCheetah, Ant and Hopper matches the saved locked-mode return cell exactly
(maximum error0). Focused tests pass locally and remotely. Formal new cost is
3,015,403 probe evaluation interactions and300 initialization probes, with
zero new training interactions or learner updates. The three successful smoke
replays used6,403 evaluation interactions and3 initialization probes separately;
one earlier interface-error smoke initialized one environment but performed no
wrapped evaluation steps. Only the18,261-byte
`results/iclr_archive_actor_probe_20260926/readout.json` was retrieved. Detailed
probe cells, training models and test fronts remain on the server.

| Cohort | Selected - final mean same-preference test utility | W/L/T | Selected / final / CAPQL-final utility |
|---|---:|---:|---:|
| HalfCheetah5 | +81.61 | 5/0/0 | 295.29 / 213.68 / 54.68 |
| Hopper5 | -88.08 | 2/3/0 | 353.91 / 441.99 / 476.94 |
| Ant5 | +839.63 | 5/0/0 | 1333.22 / 493.58 / 1271.56 |

Secondary subset-grid mode-mean HV improves versus the final actor in all
three cohort means, including Hopper, so it must not substitute for the frozen
same-preference utility primary. Hopper's utility losses span multiple probe
preferences, not one isolated weight. The hindsight-best comparator is a
utility-only diagnostic ceiling; it need not maximize subset HV. Probe actions
use no mode labels, but the validation environment deliberately switches faster
than training and consumes substantial extra interactions. Test fronts were
already observed during development; these are not fresh generalization claims.

Decision: retain the result as evidence that independent deterministic-policy
measurements can sometimes unlock archive value, especially on Ant. Do not
integrate this four-candidate per-preference probe into BAMOR: its additional
interaction cost exceeds the50k training budget by2.4x on HalfCheetah and
about4.8x on Hopper/Ant, and Hopper's primary metric regresses. Do not tune the
probe horizon, score denominator or candidate times on these seeds. A cheaper
selection mechanism needs its own frozen design and fresh budget-matched
comparison against spending the same interactions on training.

### Frozen one-episode probe versus continued training (2026-09-26)

Feasibility test on unused seeds1101-1103 in each of HalfCheetah, Hopper and
Ant. Train the unchanged v25 legacy-adaptive method to50k, capturing the live
trainer/optimizer/replay/environment/RNG state and actors at10k/25k/40k/50k.
For each of the previous pilot's3 or6 preferences and each of four actors,
run exactly one deterministic episode in an independent unlocked environment
with mean dwell5k. Score scalarized return per actual episode step, selecting
the highest-scoring actor per preference (earliest actor on a tie). The
selector sees no test returns or mode labels. Count every probe step and
environment-initialization probe; do not silently use the later test to tune
the score, dwell or candidate roster.

From the identical captured50k state, continue the unchanged trainer for
exactly as many environment steps as the actor probes actually used for that
seed. Reuse the captured RNG and environment state; training/test evaluations
cannot alter the continuation. Primary comparison: selected actor versus
continued actor on mean same-preference scalarized return, averaged over the
frozen preferences and locked modes, with three paired reset seeds per mode.
Also report both against the unchanged50k actor, every seed, per-mode effects,
probe exposure and actual training/evaluation/initialization interactions.
Subset-grid HV is secondary and cannot be substituted for the primary metric
or compared numerically with full21-weight archive HV. This is a fresh-seed
feasibility test of budget allocation, not a CAPQL replication or proof of
online adaptive operator selection. A mixed/negative result leaves v25 and the
matched CAPQL archive finding intact; no manuscript claim follows from launch.

### One-episode probe result (2026-09-26)

All nine scheduler tasks completed (seeds1101-1103 per environment, no failures).
The previously frozen rule and three paired locked-mode reset seeds were used.
Only the compact `results/iclr_cheap_actor_probe_20260926/readout.json` was
retrieved; per-seed probe reports remain server-side. Each seed's continuation
used exactly its probe's wrapped rollout steps. Formal totals:126,587 actor-
probe evaluation steps,126,587 continuation training steps,970,131 disjoint
test evaluation steps, and the common450,000-step training prefixes. The
probes required180 environment-initialization steps versus18 for continuation;
these162 additional physical steps are separately counted, so total physical
interaction budgets are not literally identical. Local smoke runs are separate.

| Environment (3 fresh seeds) | Selected / 50k / continued mean same-preference utility | Selected - continued | Paired W/L | Probe steps per seed |
|---|---:|---:|---:|---:|
| HalfCheetah | 286.30 / 305.50 / 496.28 | -209.99 | 0/3 | 12,000 |
| Hopper | 244.93 / 298.28 / 347.00 | -102.07 | 2/1 | 3,916-8,321 |
| Ant | 1,293.14 / 1,029.75 / 1,189.99 | +103.15 | 2/1 | 22,814-24,000 |

Selected versus unchanged50k wins1/3,1/3,3/3 in HalfCheetah/Hopper/Ant;
continued versus50k wins3/3,2/3,3/3. All180 single-episode probe cells had
zero mode switches, so the cheap rule observed only the initial mode; it cannot
establish cross-mode adaptation. Mean probe/continuation wall times per seed
were6.8/281.9s,4.4/171.0s and20.1/630.3s, respectively. Ant's possible
compute-time tradeoff is real in this pilot but its3-seed, initial-mode-only
gain is not a robust cross-environment result.

Decision: do not integrate or tune the one-episode selector. The original v25
versus CAPQL archive result is unchanged. This pilot instead shows that extra
training interactions are valuable in HalfCheetah and often Hopper; any next
algorithm experiment must confront mode exposure and compare against continued
training, rather than calling archive oracle HV a deployable policy gain.

### Frozen matched 75k budget-sensitivity comparison (2026-09-26)

The one-episode selector is rejected. Use the already paired50k
HalfCheetah/Hopper/Ant seeds1011-1015 to test whether the v25-versus-CAPQL
conclusion changes when both unchanged methods train for75k interactions.
Run v25 legacy adaptive and unmodified morl-baselines CAPQL archive from the
same method-specific seeds and mode schedule as at50k, with no intermediate
evaluation. Both archive the last20 actors at2.5k intervals (27.5k through
75k); both use the existing21-weight locked-mode grid, three paired reset
seeds, reference points and final actor protocol. The old50k archive had19
snapshots over5k-50k, so differences between 50k and75k archive HV include
the changed fixed-capacity time window; final-actor changes do not.

Check each75k run's50k actor against its old saved50k policy parameters and
compare mode-history prefixes. Primary75k external contrast is paired
mode-mean archive HV (v25 minus CAPQL); also report final-actor HV and direct
same-preference utility, all seeds, per-mode differences, timing and actual
training/evaluation/initialization interactions. Compare each method's75k
final actor to its own50k final actor, but do not infer adaptive-operator
selection from an archive win. The cohort has been inspected at50k: this is
paired development diagnosis, not fresh confirmation or a manuscript result.
Run via scheduler without node pinning; keep models and full fronts remote and
retrieve only a compact readout. No result-dependent algorithm edits follow.

Local and remote focused tests pass (3 each). The 30 scheduler tasks were
submitted as `t100711`-`t100712`, `t100714`-`t100733`, and `t100735`-`t100742`
(other scheduler IDs belong to unrelated work). All30 were running on
scheduler-selected node001/004/005/006 when last checked, with no failures.
The paired75k outcome and prefix checks are pending; no manuscript or
algorithm conclusion is inferred from launch.

Follow-up (2026-09-26): the original v25 in-memory mode-history check compared
tuples with JSON lists, so first attempts could fail after writing valid raw
outputs; this comparison is now normalized. The old CAPQL 50k runs did not seed
Gym's independent action-space RNG before their first 5k exploratory actions
or Python's global RNG used by replay-buffer sampling. Its 75k reruns have
genuinely different 50k actors (all fifteen HalfCheetah/Hopper/Ant cases fail
exact-prefix validation), despite matching seeds and mode schedules. Those raw
results are retained and explicitly marked
`complete=false`, `validation_issue=50k_policy_prefix_mismatch`; they cannot
support the frozen continuous-budget comparison. Future CAPQL runs seed both
RNGs explicitly. No result from this invalid matched75 protocol enters
the manuscript.

Completion audit: all 30 computational outputs are present and no matched75
task is active. The scheduler has 18 `done` records and 66 historical `failed`
attempts because 12 CAPQL signatures exhausted automatic retries before the
noncontinuation was recorded; scheduler `done` is not protocol validity. All
15 v25 runs pass the exact 50k policy-prefix check, and all 15 CAPQL runs fail
it. The actual total cost is 2,250,000 training interactions, 116,755,928
evaluation interactions, and 60 initialization-probe steps. At 75k the two
methods have matching archive steps, mode schedules, evaluation weights,
reference points, and episode seeds, but the CAPQL actor trajectories are not
reproducible from the old 50k runs.

Exploratory 75k cross-sectional readout, v25 minus CAPQL; relative differences
are ratios of five-seed means, with wins in parentheses:

| Environment | Archive HV | Final-actor HV | Same-preference final utility |
|---|---:|---:|---:|
| HalfCheetah | -9.1% (3/5) | -31.6% (2/5) | +1.9% (2/5) |
| Hopper | +41.9% (4/5) | +53.5% (4/5) | +7.3% (4/5) |
| Ant | -3.8% (1/5) | -8.7% (0/5) | -42.8% (0/5) |

Ant's direct-utility deficit is concentrated in the energy-save mode: its
energy-objective return is lower by 5,463 on average across the locked grid
and five seeds. This is a diagnostic, not a causal attribution. The 50k and
75k archive windows differ, so their HV difference is not a pure training
effect. Decision: do not promote the 75k result to a matched-continuation or
cross-environment superiority claim. A next bounded experiment should use
fresh seeds, corrected CAPQL RNGs, and a final-policy-first evaluation; only
promising results justify another full archive grid.

### Frozen fresh final-policy 75k check (2026-09-26)

To resolve the deployable-policy question without another 20-snapshot oracle
evaluation, run unchanged v25 and morl-baselines CAPQL on the unused seeds
1201-1205 in HalfCheetah, Hopper, and Ant. Both train from scratch for 75k
interactions with a 25k mean mode dwell; CAPQL now seeds NumPy, PyTorch,
Python replay sampling, and Gym's action space. v25 retains only the 75k actor
for evaluation, which does not enter its training update. CAPQL uses its
final-only path. The primary outcome is mode-mean final-actor HV; direct
same-preference utility and per-mode outcomes are secondary. The 21 full-
simplex weights, three reset seeds per weight and mode, reference points,
mode schedules, and interaction counts must match or the readout is invalid.
The upper budget is 2,250,000 training, 6,930,000 evaluation, and 60
initialization-probe interactions across 30 runs. This is a fresh diagnostic,
not a confirmation of archive-HV superiority; no method edits follow seeing
its results. Keep raw fronts and models on the server, retrieve only a compact
readout. Scheduler chooses among eligible nodes without a node pin.

The 30 task IDs are `t100844`-`t100873`. Local and server-side focused tests
pass (2 each). All 30 reached running state on scheduler-selected
node001/004/005/006; v25 and CAPQL first logs show training started. Outcomes
and final interaction counts are pending.

### Fresh final-policy 75k outcome (2026-09-27)

All 30 tasks completed without retries. The server-side readout passes all
per-run budgets, common-grid/reset/reference-point checks, and paired mode
histories. Only the 147,689-byte readout was copied locally to
`results/iclr_75k_final_fresh_20260926/readout.json`; raw fronts and models
remain server-side. Actual budget: 2,250,000 training, 6,240,415 evaluation,
and 60 initialization-probe interactions.

| Environment | v25 - seeded CAPQL final HV | HV wins | Direct utility | Utility wins |
|---|---:|---:|---:|---:|
| HalfCheetah | +4.5% | 3/5 | +83.7% | 4/5 |
| Hopper | -20.0% | 3/5 | +19.6% | 4/5 |
| Ant | -6.9% | 0/5 | -27.6% | 0/5 |

These are five-seed ratios of means, not significance claims. HalfCheetah and
Hopper HV have large opposing seed deltas; Ant is a consistent failure on both
final-policy metrics. The Ant energy-save mode drives most of its utility
deficit. A server-side preference breakdown shows that v25 beats CAPQL at the
pure-energy vertex on four of five seeds, but loses the energy-containing
mixed-preference groups; thus the failure is not a uniform inability to save
energy. This is a preference-interpolation and/or operator-credit diagnostic,
not yet evidence of a specific causal mechanism. Broad cross-environment
superiority and ICLR-ready empirical claims remain unsupported.

### Frozen Ant selector mechanism ablation (2026-09-27)

Use the already inspected Ant seeds1201-1205 to compare the existing
`bamor_v25_progress_fixed_archive_mix` with the completed v25 adaptive runs at
the same 75k budget. Change only score-driven selector allocation: trainer,
warmup, method family, final-only archive step, full21-weight grid, three
paired reset seeds, and mode schedule stay matched. Primary diagnostic is the
energy-save mode's same-preference direct utility, fixed minus adaptive;
overall direct utility and final-actor HV are secondary, with seeded CAPQL as
an external reference. Budget cap:375,000 training,945,000 evaluation,10
initialization-probe interactions for five new runs. This is posthoc mechanism
diagnosis, not fresh confirmation. No parameter tuning or method modification
follows these five results within this cohort.

The five scheduler tasks are `t100885`-`t100889`. The focused test passes
locally and on the server. All five launched on scheduler-selected
node001/004/005/006 without a node pin; the first log confirms the fixed
selector method is training. Outcomes and actual interaction counts are
recorded below.

### Ant selector ablation outcome (2026-09-27)

All five tasks completed without retries. The server-side readout validates
paired mode schedules and evaluation conditions; only its 48,219-byte summary
was copied to `results/iclr_ant_selector_final75_20260927/readout.json`.
Actual new budget: 375,000 training, 945,000 evaluation, and 10 probe steps.
Fixed minus adaptive energy-save direct utility averages -122.2 (one win,
three losses, one tie); all-mode direct utility is -3.8% (same 1/3/1), while
final-actor HV is +0.75% (2/2/1). Fixed still loses final HV to seeded CAPQL
on all five seeds (-6.2% mean), and direct utility on four of five (-30.3%
mean). Fixed does not rescue the mixed-energy preference groups.

Both arms have identical per-operator trainer configurations and warmup.
Their operator schedules coincide for seed1201 and diverge for the other four;
the exact tie is therefore a real same-trajectory case. Decision: do not
adopt the fixed selector or further tune selector scores on these seeds. The
Ant deficit survives both selector choices and points to shared preference-
conditioned actor/critic updates or operator definitions, not a demonstrated
failure of adaptive score routing. This inspected-seed result is mechanism
diagnosis only, not an ICLR confirmation.

### Frozen Ant mixed-preference critic diagnostic (2026-09-27)

Use the existing v25 final Ant models, seeds1201-1205, without updates. In each
locked mode, probe pure energy `(0,0,1)` and two mixed weights `(0.2,0.2,0.6)`
and `(0.4,0.2,0.4)` on two disjoint reset streams after32 deterministic own-
policy steps. Compare the current-preference action with actions generated at
each distinct objective vertex. Rank every action both under the execution
preference's critic context and under its generating context. From the same
replayed state, take that first action and then execute the unchanged,
deterministic current-preference policy; use raw discounted utility at256 steps
as the primary contrast in the energy-save mode. Natural episode termination
has zero further reward. Report per-seed gains, ranking losses, and the
posthoc best candidate only as a diagnostic ceiling; do not route by oracle.
Maximum new budget:97,920 evaluation interactions plus five initialization
probes, with no training interactions or learner updates. A separate smoke
preflight used44 evaluation interactions and one probe. These are inspected
development seeds; short-horizon deterministic returns need not equal the
soft, stochastic training critic target. No manuscript or algorithm adoption
from this diagnostic alone.

### Ant preference-transfer outcome and next ablation (2026-09-27)

Five scheduler tasks `t100901`, `t100904`-`t100907` completed. The paired
critic diagnostic used97,920 evaluation interactions and five probes, plus
44/one for the separate smoke; no learner updates. In the energy-save mode's
20 mixed-preference states, target-context Q selected a non-own first action
3 times; its mean256-step raw gain was+0.138, versus a posthoc one-action
candidate ceiling of+15.20. Do not route actions with this critic or oracle.
Only the4,951-byte readout was retrieved.

Re-scalarizing the already saved final-policy fronts costs no new interactions.
Across five seeds and all14 mixed-energy grid weights in Ant's energy-save
mode, the same BAMOR model's pure-energy policy beats its own mixed-preference
policy in66/70 cells, by+1,402.7 mean direct utility. The mixed policy trails
CAPQL by1,704.2; even the posthoc best BAMOR vertex trails CAPQL by231.0
on average. The vertex comparison uses identical episode reset grids, but its
selection is posthoc, not a deployable gate. This points to substantial
preference-conditioning/actor interpolation loss within BAMOR's policy family;
it does not uniquely identify the responsible update. The8,482-byte transfer
summary is saved beside the critic readout; raw fronts/models remain remote.

Next isolate the actor update on the already inspected Ant seeds1201-1205:
keep the v25 fixed-progress selector, critic envelope, relabeling, networks,
BC and evaluation unchanged, but set every operator's actor-envelope candidate
count to one. Compare with the completed fixed-progress cohort. Primary metric
is energy-save direct utility averaged over the14 mixed-energy weights;
all-grid utility, final HV and CAPQL are secondary. Maximum new cost is
375,000 training,945,000 evaluation and10 initialization probes. This is a
development mechanism ablation, not independent confirmation or adoption.
The8 focused tests passed locally and on the server. Tasks `t100910`-`t100914`
were submitted with scheduler-selected placement across node001/004/005/006,
without a node pin. All five completed, with matched operator schedules and
only the intended actor-candidate configuration changed. Actual cost was
375,000 training, 945,000 evaluation and 10 probes. Fixed-selector actor-off
minus actor-on improves the primary mixed-energy utility by 1,116.0 (5/5
seeds; 62/70 grid cells improve). All-mode direct utility rises by 249.9
(5/5). Final HV rises 1.69% in mean but wins only 2/5 seeds. Against seeded
CAPQL, mixed-energy utility still loses 4/5 (mean -746.8) and final HV loses
5/5 (-4.66%). Keep this as an active mechanism hypothesis, not an adopted
algorithm or a superiority claim. Only the 4,099-byte summary was retrieved:
`results/iclr_ant_ownpref_actor_final75_20260927/readout.json`.

### Frozen Ant adaptive actor integration (2026-09-27)

On the same inspected seeds1201-1205, retain v25 progress-driven adaptive
operator selection and all existing critic, relabeling, BC, architecture,
training/evaluation settings. Set only `envelope_policy_candidates=1` for
each operator's actor update. Compare the new adaptive actor-off variant with
the finished original adaptive, actor-off fixed, and seeded CAPQL runs. The
primary measure remains energy-save direct utility averaged over the 14
mixed-energy grid weights; all-mode direct utility and final HV are secondary.
Cap new work at 375,000 training, 945,000 evaluation and 10 probe interactions.
This integrates the mechanism into BAMOR's adaptive portfolio, but these
observed seeds are still development evidence; no paper or adoption decision
follows without independent confirmation.
Local and server selector/short-training tests pass; the saved75k result
contract also passes a field-level preflight. Scheduler tasks `t100936`,
`t100938`-`t100941` are running on scheduler-selected node001/004/006,
without a node pin. All five completed with matched evaluation and budget:
375,000 training, 945,000 evaluation and 10 probes. Actor-off adaptive minus
original adaptive gains +223.5 mixed-energy utility (3/5) but loses -733.8
to actor-off fixed (1/3/1); against CAPQL it loses all five seeds on mixed
utility (-1,480.7), all-mode utility (-330.4), and final HV (-5.08%). The
4/5 different operator schedules yield a negative actor-rule x selector
interaction in mixed utility (-892.5 mean); family allocation changes differ
by seed, so do not tune family quotas on these results. Do not adopt this
integrated variant. Only the 5,383-byte readout was retrieved:
`results/iclr_ant_ownpref_adaptive_final75_20260927/readout.json`.

Re-scalarizing the saved Ant fronts without new interactions shows actor-off
fixed reduced the energy-vertex advantage over its own mixed policy from
1,405.7 to 385.3 in the fixed-selector pair. The mixed policy beats its energy
vertex in 27/70 cells, versus 6/70 for actor-on fixed and 62/70 for seeded
CAPQL. This is an incomplete preference-interpolation repair, not proof of a
particular remaining cause.

### Frozen Ant critic-envelope ablation (2026-09-27)

On inspected seeds1201-1205, keep actor-off fixed-progress selection, actor
update, relabeling, vector loss, BC, networks and evaluation unchanged. Set
only `envelope_target_candidates=1` for the critic target, then compare with
the completed actor-off fixed run. Primary: energy-save direct utility over
the same 14 mixed-energy weights; all-mode direct utility and final HV are
secondary, with CAPQL as external reference. Maximum new cost: 375,000
training, 945,000 evaluation and 10 initialization probes. A gain on these
development seeds requires independent confirmation before adoption. No
manuscript change or actor/selector retuning in this cohort. Local and server
tests plus a saved-result contract preflight pass.
Tasks `t100948`-`t100952` completed on scheduler-selected node001/004/006,
with no required node. All five runs passed the frozen protocol checks. Actual
new cost: 375,000 training, 945,000 evaluation and 10 initialization-probe
interactions. Only the 3,972-byte readout was copied locally to
`results/iclr_ant_ownpref_critic_final75_20260927/readout.json`; raw fronts
and models remain server-side.

Removing critic candidates after the actor-candidate ablation improves Ant
mode-0 mixed-energy utility by +339.8 versus actor-off fixed (4/5 seeds), and
all-mode direct utility by +110.2 (5/5). Final HV changes by +0.69% in the
ratio of means, but wins only 2/5 seeds. Against seeded CAPQL, the new variant
still trails by -407.1 on mixed-energy utility (1/5 wins), -53.8 on all-mode
direct utility (2/5), and -4.01% on final HV (1/5). Re-scalarizing its saved
fronts without new interactions shows the pure-energy policy still exceeds
its own mixed-preference policy in 63/70 mode-0 cells, by +691.1 mean utility;
the actor-off fixed control's earlier gap was +385.3. Thus the critic change
raises direct utility but does not repair preference interpolation.

Decision: do not promote the fixed-selector, both-candidates-off variant as
BAMOR or claim external superiority. These are repeatedly inspected development
seeds, and the fixed selector also removes the adaptive-routing contribution.
Stop tuning this Ant cohort. The next algorithm experiment requires a new,
pre-specified adaptive mechanism with independent seeds and a same-budget
score-blind control; more candidate-count ablations here would not close the
paper's adaptive-operator evidence gap. Manuscript claims remain unchanged.

### Frozen pairwise-residual pilot (2026-09-27)

Before new training, two preference-only deployment rules were evaluated from
the already saved Ant fronts with zero interactions. For the both-candidates-
off fixed model, choosing a vertex once per episode with probability equal to
the target weight has expected mode-0 mixed utility 218.9, and routing to the
largest-weight vertex gives 312.4; its own mixed policy gives 645.7. For
original v25 adaptive, the corresponding values are -240.1, -195.3 and
-651.5, all below seeded CAPQL's own-policy 1052.7. Neither generic vertex
rule is a deployable repair. The previously tested independent raw-return
critic also failed its held-out action-improvement gate; do not use critic
ranking to select policies or operators.

Test mixed-preference model capacity instead, without changing v25's feedback
signal or operator portfolio. `PairwiseResidualPolicyNet` adds three zero-
initialized mean-action residual heads in 3-D, gated by `2*w_i*w_j`; their
direct contribution is exactly zero at objective vertices. It retains the
original Gaussian variance, critic, relabeling, BC and own-preference actor/
critic targets. The shared trunk can still change vertex behavior during
training; vertex preservation is not assumed. Compare base/pairwise actor x
adaptive/score-blind progress selection, plus unchanged seeded CAPQL: five
conditions on untouched Ant seeds1301-1305. No mode-specific gate or outcome-
chosen preference is used. Each run trains75k interactions and evaluates only
its final actor on the locked full21-weight grid with three paired resets per
mode. Primary: pairwise-adaptive minus base-adaptive mode-0 utility over the14
mixed-energy weights. Separately report pairwise-adaptive versus pairwise-blind
and CAPQL, the factorial interaction, all-mode direct utility, final HV, all
five seeds and actual interaction counts. Maximum formal cost:1,875,000
training,4,725,000 evaluation and50 initialization-probe interactions. This
is an independent-seed mechanism pilot, not population confirmation; a gain
cannot be attributed uniquely to factorization rather than added parameters.
No manuscript change or automatic adoption follows launch.

Implementation/preflight: 13 focused tests pass locally and in the server
runtime. A separate 16-step Ant 3-D smoke run completes training and all567
shortened-episode evaluation steps; a saved-result field-level preflight passes
for all five conditions without new training. The new cohort's scheduler
signatures and remote result root were absent before launch. Submitted all25
conditions as `t100960`-`t100984`, with scheduler-selected placement among
node001-node006 and no node pin or automatic raw-result/model download. On the
first status check24 were running and one queued; pairwise and CAPQL first logs
confirmed training. All25 tasks subsequently completed with exit0, and all25
reports passed common mode-history, grid/reset/reference, operator-configuration
and budget checks. Actual formal cost:1,875,000 training,4,700,168 evaluation
and50 initialization-probe interactions. Only the12,209-byte summary was
retrieved: `results/iclr_ant_pairwise_final75_20260927/readout.json`.

Primary mode-0 mixed-energy utility: pairwise-adaptive minus base-adaptive is
+39.6 mean, with2 wins and3 losses; pairwise-blind minus base-blind is -412.5
(1/4). Pairwise-adaptive minus pairwise-blind is +177.9 (3/2), while base-
adaptive minus base-blind is -274.2 (0/5). The registered factorial interaction
is positive in all five seeds, mean +452.1, but the pairwise-adaptive arm still
trails seeded CAPQL by247.9 mixed utility (2/3). For pairwise-adaptive versus
CAPQL, all-mode direct utility is essentially tied (+1.2, 2/3), while final
HV is -2.06% in the ratio of means (1/4). Versus base-adaptive, pairwise-
adaptive final HV is +1.62% (3/2), not a consistent win.

The base-blind and pairwise-blind operator sequences match exactly in all five
seeds; their architecture effect is not caused by different selected arms.
Adaptive versus blind schedules diverge in every seed, and base-adaptive versus
pairwise-adaptive schedules diverge in four of five. The interaction is a real
closed-loop contrast, but it does not establish a reliable adaptive benefit or
isolate feedback quality from different policy trajectories. Decision: do not
adopt the pairwise residual method or expand it to other environments on this
pilot. Stop architecture variants on these seeds; current final-policy evidence
still does not support an ICLR claim of broad superiority over CAPQL. Preserve
the positive interaction as a bounded research finding, not a rescued algorithm.

### 2026-09-27: Completed evidence in the manuscript

Replaced the pending-pilot main text with the completed 50k archive, fresh75k
final-policy and Ant factorial results. Four tables are regenerated from compact
per-seed readouts by `paper/make_evidence_tables.py`; source mapping is in
`paper/evidence_sources.md`. Added the v25 target/progress-score specification
and moved the unmatched historical continuous comparison to the appendix.
No new jobs or algorithm changes. Claims now separate archive coverage,
feedback attribution and final-policy utility; broad superiority remains unsupported.

### 2026-09-27: Implementation-facing Lean layer

Added `../proof/BAMOR/Implementation.lean` for target consistency, candidate
reselection sensitivity, bounded progress scores, approximate selected-map
tracking, continuous HV monotonicity and conditional archive-selection error.
The 14 new results compile; axiom reports contain no `sorryAx` or added axioms.
The main method now specifies v25 first, and the PDF uses a compact current
theory appendix instead of duplicate historical exposition (44 to 18 pages).
Historical proof sources and experiment outcomes are preserved. Next research
question: useful stability or decision-quality assumptions for an actual update,
not a contraction claim inherited from the different componentwise operator.
No new training jobs or algorithm changes.

### 2026-09-27: Ant counterexample-to-ablation alignment

Reconciled compact per-seed records with `summarize_ant_mechanism_alignment.py`;
results and interpretation are in `results/iclr_ant_reselection_alignment_20260927/`.
Candidate-search ablations help inspected seeds, but no measurement isolates
vector-target amplification. On independent1301-1305, both-searches-off base
adaptive still loses to base blind on mixed and all-mode utility in all5 seeds
(means -274.2 and -77.8). Hence reselection does not explain the full deficit.
Do not add a stabilization training matrix; prioritize the remaining adaptive
allocation effect without candidate search. This is posthoc evidence alignment,
not a new experiment. Three focused tests pass; no new interactions or downloads.

### 2026-09-27: Effective-configuration credit

`diagnose_ant_allocation.py` reconstructs all ten both-searches-off Ant records
and common logged prefixes.18 arm names represent11 configurations, with separate
EMA scores for aliases. Seed1301 differs only in block20 (33501-35000), BC0 versus
BC0.004, with identical other configurations, subsequent arms and expanded batch
rows. The unseen no-BC label has score0 although its equivalent alias is-0.01698;
BC is-0.01019. BC yields255.1 more final mixed utility and74.2 more all-mode utility,
but24.05M less final HV. This motivates credit sharing, not blanket BC adoption.
Compact diagnosis: `results/iclr_ant_allocation_20260927/analysis.md`.

`ant_shared_credit.py` pools the existing EMA by actual sampler/relabel/update
configuration, retaining labels, quotas, warmup and RNG calls. Original training
code is unchanged. Frozen three-condition pilot on new1401-1405 compares shared
credit, per-label credit and blind, all75k steps with own-preference targets.
Primary is shared-minus-per-label mixed direct utility; blind comparison and
all-mode/HV outcomes stay separate. Protocol and budgets are in
`results/iclr_ant_shared_credit_20260927/protocol.md`. Nine focused tests pass
locally and remotely, including short Ant training with identical warmup records.
No checkpoint download, manuscript change or successful-algorithm claim.

Submitted15 runs as `t101011`-`t101025`; all reached running state on scheduler-
selected node001/004/005/006, with no required/preferred node or automatic result
download. The1401-1405 cohort had no prior BAMOR queue signatures or remote result
paths. Formal outcomes are pending. Local/server short-training tests are separate
from the formal budget (combined upper bound256 training,2268 evaluation and8
initialization-probe interactions on test seed17001).

Completed: all15 tasks exited0 without retries; frozen pair checks passed.
Shared-minus-per-label mixed utility is-202.1 (0 wins/3 losses/2 ties), all-mode
utility-49.1; final HV+0.33% (2/1/2) does not offset the primary regression.
The ties1401/1403 have identical schedules, episodes and fronts with different
correctly reconstructed scores. Shared credit changes2/12/1 blocks in1402/1404/1405.
Against blind, shared mixed utility is+195.2 (2/2/1), versus original per-label
+397.3 (2/2/1); no consistent selector benefit is established. Do not adopt or
expand pooling. Actual formal cost:1,125,000 training,2,835,000 evaluation,30
initialization probes and1,050,015 learner updates. Retrieved only14,062 bytes
of summaries; no new training or manuscript/proof changes in this readout.
See `results/iclr_ant_shared_credit_20260927/analysis.md`.

### 2026-09-27: Seed1405 single-block diagnosis

`diagnose_ant_shared_block.py` confirms the common65k logged trajectory and only
block41 differs, with relabel rate, copy count and BC changing together. Both
rules retain identical scores for the two competing arms (+0.05060/-0.01482);
the selected relabel+BC arm's global probability decreases7.797%->7.593%, while
its within-extreme probability rises23.549%->23.913%. This is not a score-ranking
reversal. Its realized block progress is also worse (-0.18599 versus-0.00173).
Final mode utility differences are[-135.73,+8.14,+4.31]: the current mode2 improves
slightly while old mode0 loses. Component attribution and damage timing remain
unidentified. Next is a common65k-state component fork with block-end and final
evaluation, not another score or frozen-teacher variant. Four local tests and
the server diagnostic pass; retrieved21,983 bytes, no new interactions or jobs.

### 2026-09-27: Common65k component fork

Implemented five seed1405 continuations with paired66.5k/75k evaluation; protocol
is `results/iclr_ant_component_forks_20260927/protocol.md`. Keep shared credit
rejected pending diagnosis. Loss helpers preserve the original objective; the
diagnostic removes first-copy relabel, extra-copy losses or BC while preserving
expanded stochastic forwards and the historical shared schedule.

Short Ant reproduction exposed a mid-episode clone error: `mj_forward` changes
cached body positions used by the next velocity reward. `fork_env` now copies
the complete MuJoCo data without recomputing it. Ant continuation and existing
HalfCheetah fork tests pass locally; this repair changes diagnostic copying,
not native training or previously saved evidence. Formal controls remain pending.

Six new tests pass on both local and server runtimes. Submitted `t101100`, now
running on automatically selected node005 without a required/preferred node.
Formal budget is115000 training,110001 updates,10 probes and at most2079000 eval
interactions; test-seed costs are separate in the protocol. No checkpoint download.

Completed `t101100` with exit0/no retries. Both factual controls reproduce exactly;
all three component continuations match10000 learner RNG states and all7 natural
arm choices. Relative to shared, removing first-copy relabel changes primary
mixed utility by-24.8/-21.2 at66.5k/75k; extra-copy loss removal by+211.3/+28.9;
BC removal by+86.2/+62.0. Shared itself trails original by337.9/74.2, but improves
111.5 during block41 versus original's449.3: the initial deficit is less learning,
not an absolute old-mode decline. No-BC remains12.3 below original at the end.
The no-first-relabel all-weight gain is dominated by the pure-energy endpoint,
while its better progress score also fails to predict primary utility. Next is
a joint extra-copy-loss/BC ablation, not a global BC deletion or new score tuning.
Actual cost115000 training,110001 updates,2075590 evaluation and10 probes. Readout
and short analysis are in `results/iclr_ant_component_forks_20260927/`; only38652
bytes retrieved. Two readout tests pass; no new training or manuscript changes.

### 2026-09-27: Joint copy-loss/BC completion

`run_ant_joint_component.py` adds the missing joint cell with first-copy relabel
retained. Retrain one live shared control, capture65k in memory, then run the
joint continuation with the historical arm schedule and matched learner draws.
The control must reproduce all65k/66.5k/75k grids before the joint branch starts.
Use the saved single-ablation cells to compute the interaction, not a sum of
their benefits. Frozen new cost:85000 training,80001 updates,4 probes and at most
945000 evaluation interactions. Protocol is in
`results/iclr_ant_joint_component_20260927/protocol.md`; no default algorithm,
paper or historical result changes. Twelve focused local tests pass.

Four new tests also pass on the server. Combined test-seed costs253 training,
19656 evaluation,22 probes and48 standalone updates are separate from the formal
budget. No new scientific result is asserted from these tests.

Submitted `t101190`; running on scheduler-selected node004, no required/preferred
node and no automatic result download. Formal reproduction and outcomes pending.

Completed `t101190`: exit0/no retries, all three control grids reproduce exactly,
10000 learner RNG states align and all7 natural arm draws match. Joint removal
changes mixed utility by+180.6/+28.1 versus shared at66.5k/75k, but-157.3/-46.1
versus original. Interactions are-116.9/-62.7; joint loses to copy-loss-only removal
at block end and to BC-only removal at the final point. Reject it as an upgrade;
do not add the single-intervention gains or keep tuning1405. Next is independent-
seed, prespecified-window testing of the conditional copy-loss/BC effect, with
the original algorithm unchanged. Actual new cost85000 training,80001 updates,
944095 evaluation,4 probes. Retrieved14031 bytes; no new training or paper changes.
Results: `results/iclr_ant_joint_component_20260927/analysis.md`.

## Fresh-seed component replication

Freeze seeds1501-1505 and65001-66500 before training. Four copy-loss x BC cells
share a native per-label65k prefix; force the same relabeled arm in the window,
then hold the full-cell continuation schedule fixed through75k. Primary is the
66500 mixed-energy utility interaction, not the best endpoint or seed. Original
algorithm unchanged. Protocol and specs:
`results/iclr_ant_component_replication_20260927/`.
Budget:525000 training,500005 updates,40 probes, evaluation at most8505000.
Three tests pass locally and on the server. Completed test-seed work totals255
training,243 updates,20412 evaluation and24 probes, including the first local
run whose mode-switch assertion required a short-period test fixture. One SSH-
interrupted remote test has unknown completion, bounded by another85 training,
81 updates,6804 evaluation and8 probes; no live test process remained on retry.
Test work is separate from the formal seeds; scientific outcomes pending.

Submitted `t101291`-`t101295` for seeds1501-1505 respectively. All five running
with5000-step progress observed; scheduler selected node005/001/004/006/005,
no required/preferred node. No automatic result download or checkpoint transfer.

Completed t101291-t101295: exit0/no retries, all20 cells and matched-draw/schedule
checks pass. Prespecified negative interaction failed to replicate:66500 mean
+17.59,3 positive/2 negative, one-sided sign p=0.8125. BC removal improves66500
mixed utility by+39.01 (5/5), but75k by-47.13 (3/5); removing extra-copy losses
gives-34.54/-34.95, and joint removal+22.05/+8.97. Joint75k all-mode utility is
-3.19. Keep the original algorithm; stop extending this factorial. Next focus is
the loss of short-term benefit, not a copy-count-based BC switch. Cost525000
training,500005 updates,8498241 evaluation,40 probes; downloaded136285 bytes.
No new training. Readout: `results/iclr_ant_component_replication_20260927/analysis.md`.

Read-only BC continuation diagnosis: all full/no_bc native draws match execution
after66500; BC returns for1502/1505 but only1502 reverses, while1504 reverses
without further BC. The1502/1504 losses are almost entirely objective2 and span
11/14 and12/14 mixed weights;1502's pure-objective2 endpoint improves. No temporal
evaluation exists inside66500-75000, and the last1000 training steps lack means.
Corrected local Ant objective labels: native axes are x/y/control with shared
survival/contact additions, not health/speed/energy. No reward parameters or old
results changed. Six local tests and three server tests pass;195013 bytes retrieved.
New formal interactions0. Prepared (not submitted) a two-condition temporal replay
protocol in `results/iclr_ant_component_replication_20260927/carryover_temporal_protocol.md`.

Implemented `run_ant_bc_temporal.py`: same5 full/no_bc pairs, five intermediate
mode0 grids, native control/survival/contact decomposition in the same episodes,
and descriptive partial-tail means without selector feedback. Both endpoint grids
and training blocks must reproduce. Three local and three server tests pass;
combined test-seed costs536 training,520 updates,24696 evaluation,24 init probes.
Formal budget425000 training,400005 updates,20 probes, evaluation<=6930000.
Protocol/specs: `results/iclr_ant_bc_temporal_20260927/`; no new algorithm or paper
change. Formal replay and scientific diagnosis remain pending.

Submitted t101366-t101370 for seeds1501-1505. All five running with6000-step
logs observed; scheduler selected node004/006/005/001/004, no pinned/preferred
node and no automatic result download. Formal endpoint reproduction is pending.

Completed t101366-t101370: exit0/no retries, all20 historical endpoint grids and
training histories reproduce; all10000 post-cut learner draws match per pair.
All5 BC-off effects become negative at an intermediate measurement;3 recover.
Seed1502 reverses in72500-74000 after the later BC interval;1504 reverses before
its mode switch without later BC. Across35 primary contrasts, survival contributes
0 and control-cost contribution differs from total utility by at most1.40.
Keep the original algorithm; close unconditional BC removal. Next candidate
diagnosis is training-time fixed-state/preference action-cost drift versus held-out
retention, not another BC coefficient. This is retrospective evidence, not a
new CAPQL result. Cost425000 training,400005 updates,6927400 evaluation,20 probes;
downloaded64742 bytes. No new training. Results:
`results/iclr_ant_bc_temporal_20260927/analysis.md`.

Implemented `run_ant_cost_drift.py`: training-only fixed early/recent replay banks,
deterministic negative action-cost signal, no critic or held-out feedback. Same5
full/no_bc pairs; reuse intermediate grid labels and reproduce both final grids.
Primary is within-branch interval tracking by the early bank, grouped by seed;
recent-bank and BC-gap readouts are secondary. Protocol/specs:
`results/iclr_ant_cost_drift_20260927/`. Formal budget425000 training,400005 updates,
20 probes, evaluation<=1890000. Three local and three server tests pass, including
exact short-trajectory reproduction and RNG isolation. Test costs total1018
training,986 updates,40068 evaluation,44 probes, including reference generation
before the initial local output-directory error. No formal result claimed yet.

Submitted t101418-t101422 for seeds1501-1505. All five running with5000-step
training logs observed; scheduler selected node004/001/006/005/004 without any
required/preferred node. No checkpoint or automatic result download. Formal
reproduction and signal-tracking results remain pending.

Completed t101418-t101422: exit0/no retries; all10 final grids and training
histories reproduce, with10000 matched post-cut learner draws per pair. Primary
early-bank Spearman by seed1501-1505:+0.476,-0.294,+0.692,-0.343,+0.406;
direction agrees31/60 intervals (recent bank26/60). Secondary BC-gap direction
agrees17/35 early,19/35 recent. Stop this fixed-input action-cost signal; do not
tune thresholds/banks or extend seeds. Original algorithm unchanged. Cost425000
training,400005 updates,1887400 evaluation,20 probes; downloaded65787 bytes.
No new training. Results:`results/iclr_ant_cost_drift_20260927/analysis.md`.

## Independent v25 versus CAPQL utility confirmation

Return to the positive75k direct-utility observations instead of further Ant
proxy repairs. Freeze HalfCheetah/Hopper seeds1601-1620, unchanged v25 and
seeded CAPQL,75k steps and final policies only:80 runs. Scheduler history and
server result filenames show no prior use of these seeds. Primary is paired
direct utility separately by environment;20000 paired-seed bootstrap samples,
97.5% intervals for the two primary claims. HV, minima, per-mode results and
timing remain secondary. No pooling with the old five seeds, archive selection,
sample-efficiency or adaptive-feedback attribution claim. Protocol/specs:
`results/iclr_utility_confirmation_20260927/`.
Three local and three server tests pass, including exact new/old-entrypoint
short-run equality for both methods/environments. Combined test cost1280
training,16128 evaluation,32 probes. Formal budget6000000 training,
evaluation<=20160000,160 probes. No method or manuscript edits; outcomes pending.

Submitted t101434-t101513, all80 running on scheduler-selected node001/004/005/006
(20 each); no required/preferred node. Both environment/method startup logs
checked, with representative v25 runs at8000 steps and CAPQL training started.
No automatic result/model download. All20 pairs per environment and the frozen
primary readout must finish before interpreting confirmation outcomes.

Completed t101434-t101513: all80 exit0/no retries, all80 raw-to-compact reports
match and frozen paired evaluation/budget checks pass. HalfCheetah utility
296.316 versus191.168 (+55.00%,13/20 wins), paired97.5% mean-difference CI
[-33.796,235.832]; Hopper436.594 versus411.928 (+5.99%,13/20 wins), CI
[-58.645,105.580]. Neither primary utility claim is confirmed; both95% CIs
also cross zero. Secondary final HV is-4.84%/-5.77%; observed training time
is1.67x/1.73x CAPQL. Same algorithm/configuration; this is an independent
confirmation failure, not evidence of an implementation regression. Preserve
all seeds and the prior Ant negatives. Next: existing-result HalfCheetah
preference/mode decomposition before any training-side intervention. Formal
cost6000000 training,16512980 evaluation,160 probes; only382529 bytes retrieved.
No new training or manuscript edits. Results:
`results/iclr_utility_confirmation_20260927/analysis.md`.

## 2026-09-28: Final algorithm decision

The user accepts unchanged `bamor_v25_progress_archive_mix` as the final
BAMOR algorithm. Close v25 failure-seed diagnosis, further algorithm variants
and the proposed recovery-metric experiment. No new jobs or code changes.
This is an algorithm-selection decision; the failed frozen statistical
confirmation criterion remains unchanged.

Use the independent1601-1620 cohort as the main HalfCheetah/Hopper direct-utility
evidence: observed means exceed seeded CAPQL by55.00%/5.99% at75k training
interactions, with13/20 paired wins in each environment. Keep the confidence
intervals, final-HV decreases, higher runtime and prior Ant negatives visible.
Keep the earlier five-seed pilot and archive experiments separate.

Next work is manuscript consolidation, not further algorithm optimization:
present v25 consistently in method/theory/experiments and describe the observed
utility advantage at the tested budget. General superiority, sample efficiency
and population robustness remain unestablished by the current evidence.

### V25 manuscript consolidation

Updated title, abstract, introduction, method framing, experiments, discussion
and conclusion around unchanged v25. The independent1601-1620 cohort now leads
the empirical section; the1201-1205 pilot, retrospective archives and Ant
ablations remain separate. Two new generated tables report direct utility with
the frozen97.5% intervals, and secondary HV/runtime. The table generator
recomputes paired contrasts and both bootstrap intervals from compact records.
Three focused tests pass, including stale-interval and wrong-cohort rejection.
Two local pdflatex passes produce a20-page PDF with no warnings, unresolved
references or over/underfull boxes; page1 and the new table page7 were visually
checked. Existing v25 theorem statements and Lean sources are unchanged; no
new Lean build was needed for these empirical/prose edits. No training, remote
downloads, new citations or submission-template conversion in this step.

### V25 proof audit and repairs

Corrected legacy Lean comments that called residual upper bounds irreducible
error floors or B-dependent uniform bounds asymptotic/end-to-end guarantees,
and attributed a generic bound comparison to CAPQL. Existing theorem statements
and v25 are unchanged. Added explicit strict-winner certification for the
candidate-reselection witness, finite continuous archive volume, and its
real-valued monotonicity. The appendix and external proof-scope record now
include these links. The implementation witness test also covers the actual
portfolio candidate counts.

All five relevant Lean targets build; all17 implementation-facing results
depend only on standard Lean axioms, with no `sorryAx`. Six focused Python
tests pass. Two LaTeX passes produce a20-page PDF without warnings or unresolved
references; revised appendix page14 renders correctly. Current conditional
theorems remain valid, but do not establish neural convergence, causal operator
values, selector regret or superiority to CAPQL. No training or remote downloads.

### Margin-certified target stability

Added three Lean results: soft-score perturbation including entropy, winner
preservation above twice that error, and the resulting vector/scalar target
bound. The main theory summary and appendix now state this matched-candidate
certificate. `lake build BAMORImplementation` passes with20 results; the three
new results use only standard Lean axioms. Six focused Python tests pass,
including all v25 candidate counts, unequal twin critics, entropy-only switches
and an actual winner change at the threshold tie. Two LaTeX passes produce a
warning-free20-page PDF; page13 renders correctly. No algorithm changes,
training jobs, environment interactions or remote downloads. Next: determine
whether a finite-time selector-loss decomposition yields an informative bound
before adding a selector theorem; keep the exploration and quota costs explicit.

### Finite-horizon selector bound: feasibility decision

`analyze_selector_loss.py` reuses the live selector probability helper, including
quota sampling, and decomposes stored-score loss into warmup, family routing,
within-family softmax and exploration costs. For probabilities p and scores s,
let C=max(s)-p.s. If conditional mean block progress mu obeys
max|mu-s|<=eta, one-step loss is at most min{2(1-min p), C+2eta}; sum over blocks
at the realized histories. The distribution-only term is sharp on [-1,1].

The coarse free-selection bound is2.300702, already worse than the range bound2.
Exact score loss is tighter, but no uniform calibration bound has been established.
A constructed zero-score state has C=0 yet permits conditional mean loss40/21;
this is a mathematical example, not a measured training outcome. Therefore do
not add a low-regret or final-utility claim to the manuscript. Six focused tests
pass across all selection branches and the finite-sum inequality. No new Lean
theorem, manuscript change, training or downloads. Close this proof-extension
attempt for unchanged v25 and return to submission consolidation.

### V25 implementation-aligned manuscript consolidation

Aligned Algorithm 1 with per-transition preference sampling, the 5k warmup,
completed-block score updates before gradients, and unscored incomplete tails.
Specified the context-feature filters and made the resource table lead with
the 75k final-policy study, keeping the 50k archive protocol separate. The
operator table now comes directly from v25: 18 family-labelled arms represent
14 distinct sampler/update settings; duplicate settings retain separate scores
and counts. Abstract, theory framing and conclusion include local margin
stability and measured runtime costs without strengthening empirical claims.
Two focused tests pass. Two LaTeX passes produce a warning-free 20-page PDF;
pages 1, 5, 6 and 16 render correctly. No algorithm or Lean changes, new
training, environment interactions or remote downloads.

### 2026-09-30: Full-simplex CAPQL baseline completion

The user reopens algorithm improvement after the v25 freeze. Native CAPQL
uses equal-centered 22.5-degree angle sampling, not uniform simplex sampling.
Preserve v25 and native CAPQL; add only uniform Dirichlet(1) CAPQL controls
on the inspected 1601-1620 HalfCheetah/Hopper cohort, at the same 75k budget.
The shared final-policy validator accepts the explicit new method without
changing either frozen study's method roster. Four tests pass locally and
remotely, including real 2D/3D short runs and unchanged native outcomes after
sampler restoration. All 80 compact reused controls are complete. Tests cost
960 training / 12,096 evaluation interactions plus 24 initialization probes.
Protocol: `results/iclr_capql_full_simplex_20260930/protocol.md`.
Submitted `t106669`-`t106708` with scheduler-selected placement
and no node pinning. Two SSH launch failures were retried successfully;
representative HalfCheetah/Hopper logs confirm 75k training starts.
Completed readout: all 40 done, raw/compact agreement and matched three-method
seed groups verified. HalfCheetah v25 versus full-simplex CAPQL is +5.01%
with interval [-99.423,121.496]; Hopper is +23.53%, [3.030,165.152], but native
CAPQL remains the stronger Hopper baseline and its v25 contrast crosses zero.
Next: common-target learning with adaptive resource allocation, paired against
the same substrate with fixed allocation and both CAPQL samplers. Account for
gradient work; prioritize HalfCheetah normal/locked deficits and preserve Hopper.
Readout: `results/iclr_capql_full_simplex_20260930/analysis.md`.
No algorithm, paper or Lean changes in this readout step; no new training.

### Common-target allocation pilot

Implemented an explicit common-target adaptive/fixed pair, retaining online
transition/latent context with 256-width networks. Every operator uses uniform
replay preferences, one own-preference soft target, full-vector critic MSE,
one actor/critic update and no BC; only collection sampling changes. This is
not the earlier both-searches-off ablation with changing loss/relabel settings.
Actual optimizer steps, expanded rows and parameter counts are now recorded;
legacy learning formulas and RNG calls are unchanged. Eleven local and four
server tests pass, including non-null loss and identical same-replay updates.
Frozen fresh 1801-1805, two environments, five conditions, all 75k. Primary is
adaptive-minus-fixed utility; v25/both CAPQL contrasts remain separate.
Protocol: `results/iclr_common_target_pilot_20260930/protocol.md`.
Verification cost: 1,920 training / 23,940 evaluation interactions plus 50 probes.
Submitted `t106783`-`t106832` with scheduler-selected nodes, without node pinning
or automatic downloads. All 50 are done; raw/compact agreement, matched
evaluations and identical adaptive/fixed measured update work are verified.
Adaptive-minus-fixed utility is +11.43% in HalfCheetah (95% interval
[-0.726,99.414]) and +23.21% in Hopper ([14.078,112.125]); this passes the
positive-mean pilot screen, not confirmation. HalfCheetah adaptive HV beats
both CAPQL variants in all five seeds, but Hopper adaptive utility is still
-38.48% versus v25 and -29.44% versus native CAPQL. Retain v25; do not adopt
or expand this substrate yet. Hopper seed1801's exact tie follows identical
realized block histories; seed1805 shows short grounded training episodes
(adaptive/fixed 54.69/67.55 steps, versus v25 745.65 after step50k).
Next: fixed-allocation single-factor critic-loss diagnostic with unchanged
soft vector targets, networks, sampling and measured work, before reconnecting
adaptation. The bundled substrate contrast does not identify a causal component.
Formal cost: 3,750,000 training / 9,980,245 evaluation interactions plus100 probes.
Readout: `results/iclr_common_target_pilot_20260930/analysis.md`; only compact
JSON downloaded. No new training, algorithm, manuscript or Lean changes.

### Fixed-allocation critic-metric diagnostic

Add a common-target fixed method with loss
(||e||^2/d + (w.e)^2)/(1+||w||^2), retaining the same own-preference soft vector
target and unit-trace critic metric. Original common-target/v25 formulas stay
unchanged. Fresh1901-1905, HalfCheetah/Hopper, vector/projected fixed and v25:
30 matched75k jobs. Reconnect adaptation only for positive Hopper and
nonnegative HalfCheetah utility effects; retain v25 if recovery is incomplete.
Protocol: `results/iclr_common_target_loss_20260930/protocol.md`.
Seven local and seven server tests pass, including real short routes, matched
fixed sampling/work and paired readout. Combined tests cost960 training /
12,096 evaluation interactions plus24 probes. Formal planned cost2,250,000
training / at most7,560,000 evaluation interactions plus60 probes.
Submitted `t107102`-`t107131` with scheduler-selected placement and no pinning
or automatic downloads. All30 now done; raw/compact agreement, matched
evaluation, identical fixed operator histories and update work verified.
Projected-minus-vector utility is +49.36% HalfCheetah and +9.62% Hopper,
with paired95% intervals [-40.295,216.121] and [-99.084,186.282]. This meets
the frozen mean-effect screen for reconnecting adaptation, not confirmation.
Versus same-cohort v25, projected fixed is -29.70% HalfCheetah/+29.97% Hopper.
Hopper HV drops9.34% and its minimum seed utility falls359.027 to294.190
versus vector; HalfCheetah locked utility falls112.625 to68.876.
Next: small projected adaptive/fixed pilot plus v25, retaining the same
target/loss for every operator and tracking these coverage/tail regressions.
Keep v25; no20-seed expansion or tuning on the inspected seeds.
Actual formal cost:2,250,000 training /6,357,017 evaluation interactions
plus60 probes. Readout: `results/iclr_common_target_loss_20260930/analysis.md`.
Only80,384-byte compact JSON pulled. No new training, algorithm, manuscript
or Lean changes during readout.

### Projected common-target allocation pilot

Reconnect existing adaptive collection to the unchanged projected target/loss;
the new adaptive method differs from projected fixed only in score-guided
operator allocation. Fresh2001-2005, HalfCheetah/Hopper, adaptive/fixed/v25,
30 matched75k jobs. Primary: adaptive-minus-fixed utility in each environment.
Track Hopper HV/minimum and HalfCheetah locked utility; hold expansion for
nonpositive utility effects or the declared coverage/locked tradeoffs.
Protocol: `results/iclr_projected_allocation_20260930/protocol.md`.
Four local/four server tests pass, including real short routes, common loss,
matched work and warmup, same-replay update invariance and paired readout.
Combined tests:960 training /12,096 evaluation interactions plus24 probes.
Formal planned budget:2,250,000 training / at most7,560,000 evaluation
interactions plus60 probes. Submitted `t107725`-`t107754`; initially all30 running
with scheduler-selected placement, no pinning and no automatic downloads.
All30 startup records match. Representative adaptive HalfCheetah/Hopper
logs reached8k/9k steps, beyond5k warmup. All30 subsequently completed with
exit0; raw/saved agreement, paired evaluation, the common learning contract
and identical adaptive/fixed measured update work verified.
Adaptive-minus-fixed utility is +0.48% HalfCheetah (paired95% interval
[-18.189,31.116]) and -16.97% Hopper ([-178.620,21.450]). Hopper HV falls15.25%
and minimum seed utility113.166 to48.770; HalfCheetah locked utility improves
164.090 to190.281 but HV falls4.06%. Versus same-cohort v25, adaptive utility
is +41.60% HalfCheetah/-30.97% Hopper, fixed +40.92%/-16.86%.
Close this selector attempt under the frozen nonpositive-mean rule; no20-seed
expansion, tuning on these seeds or replacement of v25. Next: diagnose stored
allocation scores against episode-level returns/duration and mode transitions
before proposing a new signal. HalfCheetah seeds2001/2004 have identical
realized operator/reward histories; their exact metric ties are expected.
Actual formal cost:2,250,000 training /5,944,863 evaluation interactions plus60
probes. Readout: `results/iclr_projected_allocation_20260930/analysis.md`.
Only89,090-byte compact JSON pulled; no new training, algorithm, paper or Lean changes.

### Stored allocation-score diagnostic

No later BAMOR batch in the queue. Reconstruct all920 complete adaptive/fixed
scores from stored half-window rewards; all match. Two focused diagnostic tests pass.
Hopper adaptive has14/230 cross-mode blocks, with2.88x stationary mean absolute
score. Of27 positive-score blocks with complete episodes in both halves,8 have
declining equal-weight episode return and11 shorter episodes;7 of8 return declines
are after forced warmup. Pooled return-delta correlation0.110; fixed also disagrees.
HalfCheetah has no complete two-half episode pairs, so correlation is unavailable.
These censored training observations do not establish causal operator credit.
Keep the selector closed and retain v25. Next design a mode-comparable episode-
return/survival signal before any fresh paired pilot; no tuning on inspected seeds.
Readout: `results/iclr_projected_allocation_20260930/score_diagnostic.md`.
Only44,642-byte JSON pulled; zero additional interactions, no paper/Lean changes.

### Complete-episode allocation pilot

Add episode adaptive/fixed routes on the unchanged projected learning objective.
Hold one sampler per training episode; score cumulative equal-weight raw return
against a pre-episode EMA baseline, without dividing by length or adding bonuses.
Reuse reward-only BAPR surprise/change evidence to reset score/baseline epochs;
discard mixed detected-epoch feedback. No true mode inputs or extra interactions.
Fresh2101-2105, HalfCheetah/Hopper, adaptive/fixed/v25:30 matched75k jobs,
final-only evaluation and identical measured adaptive/fixed update work.
Close for nonpositive utility mean in either environment; hold expansion for
Hopper HV or HalfCheetah locked regression. Retain v25; no inspected-seed tuning.
Protocol: `results/iclr_episode_allocation_20260930/protocol.md`.
Ten local/ten server tests pass, including real routes, causal baseline/reset
handling, paired prefixes, same-replay update invariance and projected regression.
Combined tests:1,920 training /24,192 evaluation plus48 probes and48 standalone
actor/critic optimizer steps. Formal planned:2,250,000 training / evaluation
at most7,560,000 plus60 probes. Submitted `t107810`-`t107839`; initially all30 running
with scheduler-selected placement, no pinning or automatic downloads. All30
startup records match; representative adaptive HalfCheetah/Hopper logs reach
14k/13k, beyond5k warmup. All30 completed with exit0; raw/saved records, paired
warmup, evaluation and identical adaptive/fixed measured update work verified.
Adaptive-fixed utility is -9.94% HalfCheetah (paired95% interval
[-175.227,86.751])/-4.96% Hopper ([-195.242,122.924]); HV is -10.29%/+2.43%.
HalfCheetah locked utility falls234.901 to229.524. Close under the frozen rule;
no20-seed expansion, inspected-seed tuning or v25 replacement.
Fixed-v25 utility means are +33.80%/+6.33%, but both have only2/5 wins and
negative median differences; this is a substrate lead, not confirmed recovery.
Hopper seed2105 has zero true mode changes but9 adaptive/11 fixed detector resets;
reward-only drift does not identify environment changes. Canonical mode_history
also exposes one switch omitted by the event logger at an episode reset; only
compact diagnostic timing fields were corrected, not evaluation or training.
Next separate policy-induced reward drift from environment-change evidence
before another selector pilot; do not retune this detector on formal seeds.
Actual formal cost:2,250,000 training /6,350,089 evaluation plus60 probes.
Readout: `results/iclr_episode_allocation_20260930/analysis.md`.
Only109,349-byte compact JSON pulled; zero added interactions, no algorithm,
paper or Lean changes during readout.

### Episode feedback provenance diagnostic

Reconstruct all20 adaptive/fixed runs on the server; five focused tests pass
locally/remotely, zero added interactions. Hopper adaptive forced allocation
occupies32.67% of completed postwarmup steps;11/52 detected segments have no
free credit, and all47 scale-floor credited episodes have near +/-1 scores.
HalfCheetah forced allocation is only7.71%;90.72% of pure scores are positive,
as are90.77% fixed. Thus reset interruption is not the only unresolved mechanism.
True-mixed credited episodes:10/344 HalfCheetah,7/930 Hopper. Next separate
conditional change evidence from operator-benefit feedback; validate a frozen
(s,a)-only predictor against stationary actor/sampler changes and true changes
before integration. Current encoder takes the targets as inputs; saved artifacts
lack replay transitions. Establish multi-episode variance before scoring.
Readout: `results/iclr_episode_allocation_20260930/feedback_diagnostic.md`.
Retain v25; no new training, algorithm, manuscript or Lean changes.

### Independent conditional change-evidence probe

Implement a frozen(s,a)-only reward/state-delta predictor, separate from the
policy encoder and selector. Reuse all paired source2101-2105 fixed/adaptive
final75k actors, same fixed encoder; new rollout seeds2201-2205. Each probe fits
on2000 normal transitions, calibrates on1000, then runs six1000-step branches
with an identical500-step prefix: stationary, sampler-only, actor-only and
three true mechanism changes. Calibrate error/coverage before test; unavailable
cases count against the screen. Passing only permits considering selector design.
Five tests pass locally/remotely, plus final focused prefix/coverage gate tests.
Prelaunch cost8500 interactions /40 probes /64 predictor updates, including the
initial failed local smoke; zero policy/critic/encoder updates.
Formal planned90,000 interactions /70 probes /5000 predictor updates, zero
policy/critic/encoder updates. Submitted `t108795`-`t108804`, scheduler placement
over available node001-node006 with no pins or automatic downloads.
Protocol: `results/iclr_conditional_change_20260930/protocol.md`.
All10 completed with exit0/no retries;50 branch-prefix pairs verified. Both
environments fail the frozen screen: HalfCheetah prefix alarms3/5 and actor-change
alarms3/5; Hopper actor-change coverage1/5. Sampler-change alarms2/5 in both.
True-mode timely detection counts: HalfCheetah4/5,2/5,3/5; Hopper3/5,5/5,4/5.
Hold integration; retain v25, no tuning or expansion. Next diagnose conditional
residual variation from server-stored trajectories with zero new interactions.
Actual formal costs match the plan. Readout:
`results/iclr_conditional_change_20260930/analysis.md`. No manuscript/Lean changes.
Only47,878-byte compact JSON pulled; full transition data stays on server.

### Stored conditional-residual diagnostic

Reconstruct all10 frozen predictors from server-stored transitions with their
original seeds/config; all saved errors reproduce exactly, including original
calibration/support/alarms. Five focused tests pass locally/remotely. Input-only
10-neighbor matching excludes whole calibration windows and retains missing
comparisons. HalfCheetah prefix has6/25 alarm windows, all matchable; mean matched
loss gap0.04347 =0.02861 local-reference input-mix +0.01486 remaining. Its actor
control has only15/25 matchable windows,2/6 alarm windows matchable, and a remaining
mean gap0.10281 across4 contributing seeds. Hopper actor fit coverage48.72%, only
8/25 eligible windows; matched remaining gap0.07976 across3 seeds. Neither
coverage failure nor a matched residual gap identifies an environment change.
All HalfCheetah calibrations span just1 episode; Hopper spans3/2/4/1/1. Keep v25
and HOLD; next design episode-disjoint calibration/action coverage and error
uncertainty before fresh-seed testing, not threshold tuning or selector resets.
Single CPU2/RAM4GB batch t109124 completed on scheduler-selected node004, exit0/
no retries. Zero added interactions/probes or policy/critic/encoder updates;
5000 formal reconstruction +2000 synthetic-test predictor updates. Only212,474
bytes of compact JSON pulled; no manuscript/Lean changes. Readout:
`results/iclr_conditional_change_20260930/residual_diagnostic/analysis.md`.

### Source-blocked change-evidence development

Use whole-source cyclic splits: three fit seeds, one calibration seed, one test
seed; no trajectory/shared prefix crosses roles. Compare fixed normal data with
expanded actor/sampler normal suffixes, without duplicating prefixes. Each arm
has three frozen64x64 predictors,2000 updates each, and input-only disagreement
scaling; equal update work. Calibration now has4 complete HalfCheetah and4-9
Hopper fixed-normal episodes, rather than one HalfCheetah episode. Previously
inspected data/dependent folds remain development-only. Five tests pass per host.
All10 jobs t109160-t109169 completed exit0/no retries on scheduler-selected
node004/005/006 without pins. Expanded HalfCheetah prefix/stationary alarm sources
fall4/5->2/5 and5/5->2/5; actor alarms2/5->1/5. Hopper actor availability4/5->5/5,
but fragile timely detections5/5->3/5. HalfCheetah mode2 availability remains2/5.
Both arms fail both frozen screens: HOLD, keep v25, no fresh-interaction expansion
or inspected-outcome tuning. Next inspect existing calibration-window tail
provenance and positive/coverage failures before conditional error calibration.
Cost120,000 formal +72 test predictor updates, zero added interactions/probes
or policy/critic/encoder updates. Only119,574-byte compact JSON pulled; no paper/
Lean changes. Readout: `results/iclr_episode_blocked_change_20260930/analysis.md`.

### Stored calibration-tail diagnostic

Reproduce20 thresholds/120 case summaries from server windows and terminal arrays;
six final tests pass on each host. Both Hopper expanded fragile misses have
normal-actor quantile tails; source2105's tails are not initial100-step/reset
windows. A fixed-score/support monotonicity obstruction excludes a shared score
quantile remedy: stationary already alarms in2102; rescuing missed2104 or2105
by300 steps necessarily adds a stationary alarm, so cannot attain>=4/5 timely
and<=1/5 false sources. Raising the score quantile cannot rescue missed positives.
HalfCheetah mode2 seed2101 already alarms at200 steps (max ratio8.62), but only2
windows are eligible;2102/2103 have none. Keep unavailable comparisons missing.
Next independently fit conditional residual scales and separate reward/dynamics
support domains; no quantile tuning or selector integration. Retain v25.
Zero added interactions/probes or optimizer work; two small JSON snapshots,
final retained118,114 bytes. No raw data download, paper/Lean edits or new batch.
Readout: `results/iclr_episode_blocked_change_20260930/calibration_tail_diagnostic/analysis.md`.

### Four-role headwise change-evidence development

Implement separate reward/dynamics predictors with action-only reward footprints
accepted by two-way whole-fit-source validation. Split each cyclic fold into two
predictor-fit sources, one residual-scale source, one final-calibration source,
one test source. Compare shared-predictor constant residual scales with independently
learned conditional scales. Joint window maximum is calibrated once; supported
action evidence does not imply full dynamics coverage. Per-head missing coverage
is retained. Seven tests pass per host; all t109237-t109246 completed exit0 on
scheduler-selected node004/005/006 without pins after SSH launch recovery.
HalfCheetah candidate availability5/5 in all cases; modes timely4/5/5, but prefix
alarms3/5 and stationary/actor alarms2/5 each fail. Hopper candidate passes:
prefix0, stationary/sampler/actor alarms1/0/1, modes timely4/5/5. Constant-scale
control also passes Hopper, so added conditional-scale benefit is not established.
All17 unique HalfCheetah candidate normal alarm windows are speed reward_0;
action reward supplies support where full-state comparisons are missing in
mode2 sources2101/2102. Next decompose saved paired speed-head window scores
into raw residuals/effective conditional scales; zero reconstruction or rollout
work before another model change. Retain v25/HOLD; no alternative-arm promotion.
Actual40,000 predictor+70,000 scale updates; tests48/72 across hosts. Zero new
interactions/probes or policy/critic/encoder updates. Only216,347-byte compact
JSON pulled, raw data stays remote. No paper/Lean edits.
Protocol: `results/iclr_headwise_change_20260930/protocol.md`.

### Stored speed-head scale decomposition

Analyze five HalfCheetah folds using paired JSON windows only; five tests pass
per host. Normal window alarms28->17:11 shared,6 conditional-only,17 global-only.
All six new alarms are source2103; raw loss0.00866-0.01325 is small relative to
its calibration median0.17656, but conditional scales amplify34.9-160.3x.
Threshold factor0.893 is suppressive. Ten shared source2105 alarms have raw loss
above its calibration maximum; removing only the six new windows still leaves
two prefix-alarm sources. Source2103 mode2 goes missed->100-step detection.
Next inspect the scale-fitting input domain separately from predictor-fit support,
then validate reliability treatment with both negatives and true changes. No
test-selected clipping/quantile changes. Zero interactions/init or optimizer work;
only112,733-byte compact JSON pulled. Keep v25/HOLD, no training batch/paper/Lean edits.
Readout: `results/iclr_headwise_change_20260930/scale_diagnostic/analysis.md`.

### Stored scale-input support geometry

Compare predictor-fit and independent scale-fit input domains across all five
HalfCheetah folds. Reproduce saved predictor support, then describe scale support
using fixed calibrationq.99 and scale-source leave100-window-outq.99 references.
Preserve original normal groups/true-mode outcomes; do not change alarms. Five
final tests pass per host. CPU2/RAM4GB task t110007 completed exit0/no retries on
scheduler-selected node004 without pins; all five predictor radii/550 window
counts reproduce. Six new normal alarms have0/6 supported scale windows (mean
44.8% step coverage), but source2103 mode2 also has0/5 (mean0.4%). All11 shared
alarms are supported by the calibration reference (mean99.3%). The scale-source
leave-window reference covers only5/89 normal speed-eligible windows. Thus
neither blanket distance abstention nor lossless global fallback is justified.
Next test a normal-fit affine baseline plus residual neural prediction, preserving
action heads, four roles, optimizer budget and original MLP controls. This target
is one-step reward/state-delta prediction, not prior operator-return ridge fitting.
Zero added interactions/init or optimizer work;320,894 bytes pulled, retained
322,410 after local case-label repair without matrix rerun. Retain v25/HOLD,
no selector/paper/Lean changes.
Protocol: `results/iclr_headwise_change_20260930/scale_support_diagnostic/protocol.md`.

### Affine-residual mean-predictor development

Fit a normal-only standardized affine baseline, then a zero-output-initialized
64x64 residual network with the same2000 updates. Preserve the headwise action
validation, four source roles, independent scale fitting, support and original
joint calibration/screen; saved plain-MLP outcomes remain paired controls.
All11 local and11 remote tests pass. All10 tasks t110041-t110050 completed
exit0/no retries on scheduler-selected node001/004/005/006, without pins.
Original action decisions/support radii/window counts reproduce; availability5/5.
HalfCheetah conditional-scale prefix/stationary false sources3/5->0/5 and
2/5->0/5; modes remain4/5/5 timely, so screen FAIL->PASS. Hopper prefix false
sources0/5->2/5, other controls within1/5 and modes4/5/5, so PASS->FAIL.
The two Hopper triggering heads (2102 reward_1/end100;2103 dynamics/end400)
have lower raw loss than plain MLP and below their calibration raw maxima;
conditional score ratios1.34/1.92 expose remaining scale/calibration mismatch.
Global scales pass only Hopper; no environment-specific arm promotion.
Next normal-only out-of-source residual calibration, not more mean-network
complexity; freeze roles/cost before launch. Retain v25/HOLD, no selector/paper/
Lean changes. Actual40000 predictor+70000 scale updates and20 affine fits,
zero interactions/policy/critic/encoder work;440500-byte compact JSON pulled
plus small alarm excerpt, no raw arrays/checkpoints. Tests51/63 predictor/scale
updates per host. Development evidence only, no new policy/CAPQL claim.
Protocol: `results/iclr_affine_residual_change_20261001/protocol.md`.

### Cross-source normal-scale development

Keep deployed affine/residual prediction, footprints/support and final calibration
unchanged. Produce two whole-source-out residual blocks with auxiliary predictors
on the two mean-fit sources, then pool them with the independent scale source.
Scale learning uses15000 normal steps/source-balanced blocks; no final calibration
or test inputs/targets enter fits. These proxies come from single-source means,
not guaranteed deployed-error bounds; footprint admission/normalization still
use pooled mean-fit data. Original q.99 joint screen/controls remain frozen.
All15 tests pass per host. All10 tasks t110087-t110096 completed exit0/no retries
on scheduler-selected node001/004/005/006, no pins. Mean fitting/action decisions,
support radii/window counts reproduce; availability5/5. HalfCheetah conditional
prefix false sources0->2 and actor1->2; Hopper prefix2->2 and actor1->3, out of5.
Both conditional screens FAIL; timely modes remain4/5/5 in both environments.
Stop directly pooling these proxy residuals. On the same normal fit inputs,
auxiliary/deployed fit-MSE ratios reach86.2 (HalfCheetah reward_0/2102) and1575.5
(Hopper reward_2/2105): different residual producers, not measurements of one
deployed error function. Deployed fit errors are in-sample, so this comparison
does not establish that producer mismatch alone caused the regression.
Next prepare matched deployed-predictor residuals on additional normal trajectories;
check frozen actors/checkpoints and freeze/count collection interactions first.
Actual40000 deployed+80000 auxiliary mean updates,70000 scale updates and60
affine fits; zero interactions/policy/critic/encoder work. Tests147 mean/99 scale
updates per host. Only437476-byte compact JSON pulled, no raw arrays/checkpoints.
Retain v25/HOLD; no selector/paper/Lean changes or environment-specific arm promotion.
Protocol: `results/iclr_cross_source_scale_20261001/protocol.md`.

### Matched deployed-residual scale development

Verify all20 frozen75000-step fixed/adaptive checkpoint files. Collect5000 fresh
normal steps per policy source (stationary3000, sampler1000, actor1000), with
distinct seeds and locked mode0. Shared total50000 rollout steps+30 constructor
probes=50030 added interactions. In each fold, use fresh data only from its two
original mean-fit policy sources plus the old independent scale source;15000
residual targets all come from the same unchanged deployed predictor. Preserve
held-source exclusions, q.99 joint calibration, support and full original screen.
Thirteen tests pass per host; test cost160 rollout+8 initialization steps and
69 mean/90 scale updates per host. Two scheduler stages; fitting submits only
after successful acquisition. All20 tasks done exit0: collection t115235-t115244,
fit t116472-t116481, scheduler-selected node001/004/005/006 with no node pins.
Actual50000 rollout+30 initialization=50030 interactions; mean40000/scale70000
updates and20 affine fits, no auxiliary/policy/critic/encoder updates. Only423829
bytes of compact readout plus small status/window summaries pulled, no raw arrays
or checkpoints. All conditional cases remain available5/5, but HC prefix/stationary
alarms change0/0 ->3/3, sampler/actor1/1 ->2/2; Hopper prefix2 ->1, stationary0,
sampler/actor1/1 ->2/3. Timely mode1/2/3 detections stay4/5/5 in both. Both fail;
global scale passes only Hopper. Retain v25/HOLD; no selector/paper/Lean changes.
HC's first prefix alarms select reward_0: seed2103 end200 score0.1431 ->1.9113,
threshold1.6955 ->1.7832. Next diagnose local scale errors and cross-policy
calibration mismatch offline using existing arrays, before any structural change;
no further collection or threshold tuning. These are dependent mechanism-development
results, not v25 performance regression or CAPQL superiority evidence.
Protocol: `results/iclr_matched_residual_scale_20261001/protocol.md`.

### Matched-scale transfer diagnostic

Freeze all10 existing development folds for read-only paired-window algebra and
input geometry. Recover equal raw losses across old/new scale fits, factor
effective normalization and joint threshold changes, and compare scale-reference
versus calibration-reference input coverage on predictor-supported steps. Count
the common prefix only once. Geometry remains diagnostic, not an alarm gate.
Nineteen tests pass per host, including action-only footprints and source-role
exclusions; zero test/formal interactions or optimizer work. All10 jobs
t116534-t116543 done exit0, scheduler-selected node001/004/005/006. Pull only
448980-byte compact JSON, no raw arrays/checkpoints. Raw errors and predictor
support reproduced. HC35 newly alarmed windows:28 scale-only sufficient,7 joint;
median effective-scale compression3.20x/max16.61x, winning reward_0:26/dynamics:9.
At fixed leave-window radii, new-scale coverage12/35, calibration coverage0/35;
calibration-to-scale radii instead cover35/35. Hopper6 new alarms:1 scale-only/5
joint, new-scale coverage5/6, calibration3/6, both2/6. Both-covered counterexamples
are seed2104 actor end700/800 dynamics with100% coverage of each reference.
Retain4 resolved Hopper windows; old/new alarms HC2/36 and Hopper9/11 per100
unique normal windows. Geometry alone cannot explain every false alarm.
Next freeze an original-versus-expanded calibration comparison using the held
calibration source's existing fresh normal data, unused by this fold's mean/scale
fits. Preserve scale models, original predictor support andq.99, and apply the
entire original screen including mode detection. No collection or geometric
alarm gate. Not yet run. Retain v25/HOLD; no policy/paper/Lean changes. This is
dependent mechanism development, not conditional-variance truth or CAPQL evidence.
Protocol: `results/iclr_matched_residual_scale_20261001/transfer_diagnostic/protocol.md`.

### Fixed-support calibration expansion

Freeze all10 folds for a calibration-only comparison: original5000 plus existing
fresh5000 normal steps from the same held calibration source. Neither enters
mean/scale fitting; test-source fresh trajectories remain unused. Preserve mean
and scale outputs, original support radii,100-step windows and the jointq.99 rule.
No monotone threshold floor, score clipping, geometry gate or per-environment
arm selection. Nine tests pass per host; test cost0 interactions,24 mean/36 scale
updates per host. Formal reconstruction40000 mean/70000 scale updates and20
mean affine fits; added interactions0, prior acquisition50030 charged in parent.
All10 jobs t116548-t116557 done exit0, scheduler-selected node001/004/005/006.
Original calibration and test scores/support reproduce. Conditional HC prefix,
stationary, sampler, actor alarms3/3/2/2 ->2/2/1/2; Hopper1/0/2/3 unchanged.
All cases available5/5; timely mode1/2/3 detections4/5/5 unchanged in both.
Both conditional screens FAIL. Global scale passes only Hopper; HC stationary
alarms2 ->3. Pull only408210-byte compact readout plus small status, no raw
arrays/checkpoints. Actual reconstruction40000 mean/70000 scale updates and20
mean affine fits; added interactions0. Retain v25/HOLD, no policy/critic/encoder/
paper/Lean changes. Stop same-source calibration expansion. Next select
conditional/global scale shrinkage on training-side held normal trajectories,
keeping calibration/test isolated; not yet run. These remain dependent
mechanism-development results, not independent confirmation or CAPQL evidence.
Protocol: `results/iclr_calibration_expansion_20261001/protocol.md`.

### Held-normal residual-scale shrinkage

Freeze all10 folds with matched-scale mean/scale fits unchanged. Use existing
fresh5k normal trajectories from the training-side scale source to select one
global weight shared across heads from1/.75/.5/.25/0 by mean residual NLL. Old
trajectories from that policy remain in scale fitting; fresh validation does not.
Calibration/test sources never enter selection. Keep original5k calibration,
support andq.99 joint windows, and full normal/mode screen. Local10 tests pass
after correcting result metadata; both attempts cost0 interactions,48 mean/72
scale updates. Remote10 tests pass, cost0 interactions,24 mean/36 scale updates.
Formal planned40000 mean/70000 scale updates,20 mean affine fits,
250000 transition-candidate evaluations; added interactions0, prior acquisition
50030 charged in parent. All10 tasks t116577-t116586 done exit0 without retries,
scheduler-selected node001/004/005/006. Both controls reproduce; every selected
global weight0, so the selected arm equals the original conditional arm. HC
prefix/stationary/sampler/actor alarms3/3/2/2; Hopper1/0/2/3, each out of5.
All cases available5/5; timely modes4/5/5 in both. Both screens FAIL, unchanged
against matched-scale5k calibration, not a v25 performance regression. Actual
40000 mean/70000 scale updates,20 mean affine fits,250000 transition-candidate
evaluations and0 added interactions. Pull only511102-byte compact JSON plus small
status; no raw arrays/checkpoints. Retain v25/HOLD, no policy/critic/encoder/
paper/Lean changes. Next whole-policy-source scale validation, keeping NLL first
to isolate the split and using existing normals; calibration/test stay excluded.
Not yet run. Failure of this selection rule does not disprove nonzero shrinkage.
Dependent mechanism development, not confirmation.
Protocol: `results/iclr_residual_scale_shrinkage_20261001/protocol.md`.

### Whole-policy-source scale validation

Freeze all10 folds; keep deployed mean,15k final scale inventory,5k calibration,
support and full original screen. Three inner fits each exclude one whole policy
source from scale NN/global statistics, fitting10k and validating5k. Keep NLL,
common coefficient/grid/tie rule; aggregate candidate NLL, not fold winners.
Refit unchanged final scales after selection. Mean is not source-held; no proxy
predictors. Calibration/test excluded; retain saved trajectory-rule control.
Local/remote7 tests pass per host, cost0 interactions,30 mean/72 scale updates
per host. Formal
planned40000 mean,70000 final+210000 inner=280000 scale updates,20 mean affine
fits,750000 transition-candidate evaluations; interactions0, prior50030 charged
in parent. Interaction-matched, not compute-matched against the cheaper rule.
All10 tasks t116608-t116617 done exit0, no retries, scheduler-selected node001/005.
Both scale controls reproduce. Nonzero weights4/10: HC2102=.75/2105=.5;
Hopper2102=.25/2105=.25. Source-level HC prefix/stationary/sampler/actor alarms
3/3/2/2 unchanged; Hopper1/0/2/3 ->1/0/3/4, each out of5. Cases available5/5,
timely modes4/5/5 unchanged in both; both screens FAIL. New Hopper2102 sampler
alarm is dynamics at post-delay400; Hopper2105 actor reward_0/dynamics at500.
Corresponding thresholds3.1617 ->1.9715 and5.0252 ->2.6244. Held-source NLL gains
did not translate to deployment false-alarm control. Actual40000 mean and70000
final+210000 inner=280000 scale updates,20 mean affine fits,750000 transition-
candidate evaluations,0 added interactions. Pull only731125-byte compact JSON
plus small status, no raw arrays/checkpoints. Retain v25/HOLD, no policy/critic/
encoder/paper/Lean changes. Next paired stored-window diagnosis of normalization,
joint-head and threshold shifts before changing the objective; not yet run.
Protocol: `results/iclr_source_held_scale_shrinkage_20261001/protocol.md`.

### Stored source-held window diagnosis

Freeze all10 existing folds for read-only paired JSON-window decomposition.
Recover equal raw head losses from the unchanged global arm; compare effective
normalization, q.99 thresholds and joint winning heads. Count prefixes once,
retain all resolved alarms and mode cases. Reproduce old/new calibration tail
interpolation and classify algebraic score-only/threshold-only substitutions.
No deployment-rule changes. Local/remote6 tests pass each; test/formal interactions and all
model updates0. Task t116624 done exit0 on scheduler-selected node002 without
pins/retries; all10 folds complete. Pull only179486-byte compact JSON plus status,
no raw arrays/checkpoints. Supported raw losses/support and both quantile tails
reproduce.100 unique normal/75 mode windows per environment. HC alarms36 ->35,
one resolved seed2105 actor end800, no new; Hopper11 ->16, five new, none resolved.
Four new are threshold-sufficient, one jointly required; none score-only or
winning-head switches. Normalization factors0.9533-1.2428 versus threshold factors
1.6037-1.9148. Hopper2105 old reward_0 calibration tail scores shrink to0.1872-
0.2222; new tail operands select dynamics, lowering the common joint threshold.
Mode alarms HC70/Hopper67 unchanged, none gained/lost, eligible75/72. These are
algebraic decompositions, not causal guarantees. Next training-side independent
window calibration and cross-source joint exceedance risk selection, with NLL
control using the same split/models; actual calibration/test stay excluded.
Not yet run. Retain v25/HOLD, no policy/critic/encoder/paper/Lean changes.
Protocol: `results/iclr_source_held_scale_shrinkage_20261001/window_diagnostic/protocol.md`.

### Cross-source calibrated window-risk selection

Freeze all10 folds; three whole-source scale folds retain the deployed mean.
Use the separate fresh5k collection from the training-side scale policy for
inner calibration, excluded from all mean/scale fits. Select a shared coefficient
by equal-source joint-window exceedance rate; ties NLL then larger global weight.
Matched NLL selection shares every fit and validation source. Refit original15k
final scales; read actual calibration/test only after selection. Original full
normal/mode screen unchanged. Local6 tests PASS,0 interactions,24 mean/63 total
scale updates. Initial Python3.8 invocation failed at import before tests/fits;
correct local Python3 runtime passes. Planned formal40000 mean/280000 total scale
updates,20 affine fits,0 added interactions. Scheduler unpinned node001-node006.
Retain v25/HOLD pending results; no policy/critic/encoder/paper/Lean changes.
Protocol: `results/iclr_window_risk_selection_20261001/protocol.md`.

Remote6 tests PASS,0 interactions,24 mean/63 total scale updates. Scheduler
doctor clean, no active BAMOR pending escalation. Submitted t116646-t116655;
initial dispatch1 running on node005,9 queued, no pins/retries. First task has
reconstructed4000 mean updates and completed12000 inner scale updates (two folds).
This is an execution snapshot, not a completed result. After completion compare
the paired window-risk/NLL screens and reproduce prior whole-source NLL scores;
only a both-environment risk-arm PASS merits fresh confirmation.

All10 tasks t116646-t116655 done exit0 without retries/pins, scheduler-selected
nodes001-006. Prior NLL candidate/choice, inner/final fits and every calibration/
test-window result reproduce exactly. Risk versus NLL normal alarms HC35/100
->21/100, Hopper16/94 ->13/94. Source-level prefix/stationary/sampler/actor HC
3/3/2/2 ->2/2/1/1, Hopper1/0/3/4 ->1/0/2/3, each /5. All cases available5/5.
HC timely modes4/5/5 ->4/4/5, Hopper4/5/5 unchanged; HC mode windows70 ->62,
Hopper67 unchanged. Both risk screens FAIL. HC2103 resolves14 normal alarms but
entirely misses mode2; HC2101 inner risk .04 ->0 adds four deployment false
alarms. Hopper global passes with7 normal alarms versus risk13 and conditional11.
Actual40000 mean/280000 total scale updates,20 affine fits; added interactions0.
Readout/reproduction adds no model updates. Pull only135198-byte control JSON,
no raw arrays/checkpoints. Retain v25/HOLD, no policy/critic/encoder/paper/Lean
changes. Next cross-policy calibration transfer plus training-side change-
sensitivity constraints, not more coefficient tuning. Not yet run.

### Training-side change sensitivity constraint

Freeze all10 folds; isolate sensitivity with previous models/calibration/grid
unchanged. Evaluate mode1-3 only from the three training policy sources in each
source-held inner fold. Keep every probe that conditional scale detected within
300 steps, then minimize normal-window risk; ties unchanged. Changed targets
never enter fits; actual calibration/test excluded from selection. Reproduce
normal-only risk/NLL controls. Local6 tests PASS,0 interactions,30 mean/99 total
scale updates. Planned formal40000 mean/280000 scale updates,20 affine fits,
1950000 transition-candidate pairs;0 added interactions.90 fold-probe evaluations
reuse30 trajectories, not90 independent probes. Retain v25/HOLD; this now uses
training intervention labels, not a label-free sensitivity guarantee. Subsequent
cross-policy calibration is a separate step, not changed here.
Protocol: `results/iclr_sensitivity_guarded_selection_20261001/protocol.md`.

Remote6 tests PASS,0 interactions,30 mean/99 total scale updates. Primary and
secondary jumps failed; direct HPC timed out. Uploaded via existing jtl311linux
and remote tests passed after one pre-execution connection retry. No SSH or
scheduler transport configuration changes; doctor clean, ready to submit.

Submitted t116735-t116744, unpinned; initial snapshot1 scheduler-running on
node001,9 queued, no retries. First HC2101 readout complete, both old selectors
and their windows reproduce; guarded weight0 versus risk.25,7 protected training
probes with none lost. Snapshot only: all-fold deployment outcome pending.

All10 tasks t116735-t116744 done exit0, no retries/pins, scheduler-selected
nodes001/002/004/005. Both previous selectors reproduce.80 protected inner
training detections (HC37/Hopper43 of90 fold-probes) retained, none lost.
Against normal-only risk: HC normal alarms21/100 ->17/100, Hopper13/94 ->9/94;
four resolved in each, no new normal alarms. Mode windows HC62/75 and Hopper67/72
unchanged, no gained/lost. Timely modes HC4/4/5 and Hopper4/5/5 unchanged.
Both guarded screens FAIL: HC prefix/stationary2/5; Hopper sampler/actor2/5.
HC2103 inner mode2 all three source probes timely at100, yet deployment mode2
entirely missed. Constraint implementation holds on training probes but fails to
transfer. Actual40000 mean/280000 scale updates,20 affine fits,0 added
interactions. Readout0 model updates; pull only259221-byte control JSON, no raw
data/models. Retain v25/HOLD, no policy/critic/encoder/paper/Lean changes.
Next fixed-model cross-policy calibration comparison, not more sensitivity
constraints. Not yet run.

### Fixed-model cross-policy calibration

Freeze previous guarded final models/weights and original support across all10
folds. Compare original5k, same-policy fresh5k, training-policy fresh5k and
balanced2500+2500 fresh calibration. Whole100-step windows, fixed q.99; only
thresholds change. Pooled arm is prespecified primary; unchanged full screen,
both environments must pass before considering fresh confirmation. Existing
two fresh streams excluded from final model fitting, so add0 interactions.
Local4 tests PASS,0 interactions,36 mean/108 scale updates including parents.
Formal40000 mean/70000 final-scale updates,20 affine fits, no inner reselection.
No policy/critic/encoder/paper/Lean changes; retain v25/HOLD pending results.
Protocol: `results/iclr_cross_policy_calibration_20261001/protocol.md`.

Remote4 tests PASS,0 interactions,36 mean/108 scale updates including parents.
Scheduler doctor clean, no active BAMOR pending escalation; ready to submit.

Submitted t116779-t116788, unpinned node001-node006. Initial3 scheduler-running
on node002/005/001,7 queued, no retries. First HC2101 readout complete: original
guarded alarms reproduced, all test scores/coverage unchanged,4000 mean/6000
final-scale updates,0 interactions. All-fold performance pending; retain v25.

All10 tasks t116779-t116788 done exit0, no retries/pins; doctor clean. Original
guarded alarms reproduce, all four arms preserve test scores/support/coverage.
Normal alarms original/same-policy fresh/training-policy fresh/pooled:HC
17/25/46/30 of100, Hopper9/10/13/6 of94. Primary adds13 HC alarms, removes3
Hopper alarms, no mode-window changes. Timely modes HC4/4/5, Hopper4/5/5;
all cases available5/5. Pooled screen HC FAIL/Hopper PASS, so retain v25/HOLD.
HC2103 mode2 timely scores1.43-1.56 versus normal max0.317, but every calibrated
threshold3.67-3.82; post-hoc scalar separation exists in this fold. It is a
calibration-transfer diagnosis, not a selected deployment threshold. Next test
independent deployment-policy normal startup calibration with explicit budget
and unchanged external controls, not more external-source pooling. Not yet run.
Actual40000 mean/70000 final-scale updates,20 affine fits,0 inner/new interactions;
readout0 updates. Pull only35637-byte control JSON, no raw data/models.
Results: `results/iclr_cross_policy_calibration_20261001/analysis.md`.

### Independent deployment-policy startup calibration

Freeze all10 folds, final guarded models/weight and original support. Primary
uses independent fresh5k unchanged startup transitions from the deployment
policy source; same policy source permitted, evaluation trajectories excluded.
Startup data excluded from model fitting/inner selection; q.99 unchanged, no
changed-case labels in threshold fitting. Keep original/pooled external controls
and every evaluation window. This explicitly changes the source-exclusion
assumption, not a silent correction to the previous protocol.
Required startup5000 transitions+3 initialization probes per fold=50030 across10
folds, already collected/charged upstream; reuse adds0 interactions, not free
deployment. Local4 tests PASS,0 added interactions,42 mean/117 scale updates
including parents. Formal40000 mean/70000 final-scale updates,20 affine fits,
0 inner/policy/critic/encoder updates. Retain v25/HOLD; full primary screen on
both environments required before fresh confirmation. Scheduler unpinned.
Protocol: `results/iclr_deployment_startup_calibration_20261001/protocol.md`.

Remote4 tests PASS,0 added interactions,42 mean/117 scale updates including
parents. Scheduler doctor clean, no active BAMOR pending escalation.

Submitted t116833-t116842; all10 done exit0, no retries/pins, nodes002-006.
Primary independent startup calibration passes both full screens. Normal alarms
original/pooled/startup HC17/30/4 of100,Hopper9/6/2 of94. Paired startup versus
original:HC16 resolved+3 added (actor,2103),Hopper7 resolved+0 added. HC gains5
mode windows, loses0; Hopper unchanged. Timely modes HC4/5/5,Hopper4/5/5;
HC2103 mode2 restored at100 steps, threshold3.6855->0.2508. Both2104 mode1 still
missed; all cases available5/5. Original/pooled controls reproduce and all test
scores/support/coverage remain fixed. This supports deployment-policy calibration
under known unchanged startup, not the previous unseen-policy assumption.
Actual40000 mean/70000 scale updates,20 affine fits,0 new interactions/inner
updates. Startup required50030 already charged upstream; global acquisition
ledger140100 unchanged. Readout0 updates, pull only43870-byte control JSON.
Retain v25/no paper/Lean changes. Next freeze independent new calibration/test
trajectories on the same policy sources; not yet launched, no more threshold
tuning on these development folds.
Results: `results/iclr_deployment_startup_calibration_20261001/analysis.md`.

### Frozen startup fresh-trajectory confirmation

Freeze same10 policy sources/folds, models, weights, support, q.99 and full
screen. New startup5k and evaluation6x1k use disjoint new reset/preference RNG
blocks. Persist threshold before collecting evaluation; preserve all paired
prefixes and active normal interventions. Controls are old external original,
pooled and old deployment-startup thresholds, scored on shared new data.
Formal110090 new interactions (50000 startup+60000 evaluation+90 probes),
cumulative scientific acquisition250190; repeated shared-prefix execution
charged. Reconstruct40000 mean/70000 scale updates,20 affine fits; no inner or
policy/critic/encoder updates. This is fresh trajectories on old policies, not
new policy-seed generalization. Local4 tests PASS:424 real fixture interactions,
48 mean/126 scale updates; retained injected failure costs4. Remote tests pending.
Retain v25 until both-environment primary confirmation; no tuning on new data.
Protocol: `results/iclr_startup_confirmation_20261001/protocol.md`.

Remote4 tests PASS:424 real fixture interactions,48 mean/126 scale updates.
Local+remote verification848 real interactions,96 mean/252 scale updates;
separate from formal110090. Scheduler doctor clean, no active BAMOR escalation.

All10 tasks t116872-t116881 done exit0, no retries/pins, nodes004-006. On shared
new evaluation, external original/pooled/old-startup/new-startup normal alarms
HC18/26/3/1 of100,Hopper11/14/6/7 of97. Primary removes17 HC false alarms with0
added, gains5 mode windows with0 lost; Hopper removes5/adds1 false alarm, mode
windows unchanged. Both timely modes4/5/5; all cases available5/5. Primary HC
PASS/Hopper FAIL:stationary post false alarms2/5, limit1/5. Old-startup Hopper
PASS does not replace the failed primary; retain v25/HOLD. Both2104 mode1 still
missed. Hopper2102 dynamics score4.965 versus new calibration max4.866,38 eligible
calibration windows; Hopper2103 reward_1 score90.64 versus calibration max3.54,
also exceeds old-startup threshold5.41. Next distinguish marginal tail stability
from the large reward-scale spike; no single quantile adjustment justified.
Actual110090 scientific new interactions, cumulative250190; verification848
separate, current-round total110938.40000 mean/70000 scale updates,20 affine fits,
0 inner/policy/critic/encoder updates. Readout0 updates, pull only58237-byte
control JSON, no arrays/models. No integration or paper/Lean changes.
Results: `results/iclr_startup_confirmation_20261001/analysis.md`.
