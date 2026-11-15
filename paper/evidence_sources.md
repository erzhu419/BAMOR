# Main-Text Evidence (2026-10-03)

Regenerate six tables with `python3 paper/make_evidence_tables.py` from the
repository root. The script recomputes contrasts from per-seed compact records
and checks them against the saved summaries. For the twenty-seed cohort it also
recomputes the frozen paired95%/97.5% bootstrap intervals. No new interaction or remote
download is needed. W/L/T includes ties; percentages are ratios of cohort means.

| Manuscript evidence | Local source under `results/` | Scope |
| --- | --- | --- |
| Main75k utility and secondary-outcome tables | `iclr_utility_confirmation_20260927/readout.json` | Seeds1601-1620, HalfCheetah/Hopper; unchanged v25, independent of the earlier pilot; paired97.5% intervals include zero |
| 50k CAPQL archive and score-blind tables | `iclr_capql_archive_extension_20260926/readout.json` | Seeds 1011-1025 HalfCheetah, 1011-1015 Hopper/Ant; retrospective baseline completion |
| Earlier75k final-policy pilot table | `iclr_75k_final_fresh_20260926/readout.json` | Seeds1201-1205; final HV was primary, utility motivated the later independent study; cohorts remain separate |
| Ant structure-by-selector table | `iclr_ant_pairwise_final75_20260927/readout.json` | Seeds 1301-1305; own-preference base/residual x adaptive/blind, plus CAPQL |
| Original matched v23 pilot | `iclr_matched_v23_pilot_20260913_summary.json` | Seeds 20-22; historical development pilot |
| Ant actor-search ablation | `iclr_ant_ownpref_actor_final75_20260927/readout.json` | Inspected seeds 1201-1205; fixed selector |
| Ant adaptive actor-search ablation | `iclr_ant_ownpref_adaptive_final75_20260927/readout.json` | Same inspected seeds; adaptive integration |
| Ant critic-search ablation | `iclr_ant_ownpref_critic_final75_20260927/readout.json` | Same inspected seeds; both searches off, fixed selector |
| Ant within-model preference transfer | `iclr_ant_mixed_critic_20260927/preference_transfer.json` | Posthoc re-scalarization, not a deployable selection rule |

`ICLR_MINIMAL_REVISION.md` records the protocols, interaction accounting,
negative branches, scheduler IDs and CAPQL seeding defect. Raw fronts and
checkpoints remain on the server. The table generator consumes only small local
readouts, not the historical unmatched tables or invalid continuation runs.

## Interpretation

The central argument is that BAMOR-v25 is a progress-driven update portfolio
with higher observed mean direct utility than CAPQL on two custom tasks at 75k;
this outcome is distinct from coverage and adaptive-selector attribution.
The independent 20-seed differences are +55.00%/+5.99%, with 13/20 paired wins
per task and 97.5% intervals crossing zero. Neither primary superiority test
passed. Final-HV means are -4.84%/-5.77%, with training-time ratios 1.67/1.73.
The user selected unchanged v25 as the final algorithm on 2026-09-28; that
decision does not alter the frozen test or promote a negative ablation.

The earlier observed Hopper archive advantage and Ant structure-by-selector
interaction remain separate findings. Ant is a negative transfer result.
General superiority, population robustness and reliable feedback-routing gains
remain unestablished. Historical unmatched results remain in the appendix.
The current formal layer is in `../proof/BAMOR/Implementation.lean`; its statement-to-source map is
`../proof/BAMOR/CURRENT_PROOF_SCOPE.md` (paths from the repository root).
The main method presents v25 directly. Historical derivations remain in
`theory_background.tex` and `historical_proof_catalog.tex`, outside the PDF.
The resource table records the 75k final-policy protocol in
`implementation_details.tex`; the 50k/19-actor archive protocol is separate.
`make_v23_operator_table.py` retains its filename but generates the shared
18-arm specification from the current v25 selector. These are 18 allocation
slots with 14 distinct sampler/update settings; repeated
settings keep separate scores and counts. No arms were merged or changed.

The October 3 manuscript revision centers the argument on transition-derived
context and progress-driven training allocation. `task_protocol.tex` records
the actual custom modes, smooth reward saturation, switching process and
deterministic evaluation from `piecewise_mo_env.py` and
`train_bamor_mujoco.py`. `ant_diagnostics.tex` retains the detailed unsuccessful
repair experiments in the appendix; the main text keeps their negative outcome.
The two-weight utility example in `method_v23.tex` is illustrative mathematics,
not an additional experiment or Lean theorem. Numerical tables, cohorts,
algorithm code and proof statements are unchanged. Later HOLD candidates do
not enter the v25 method or its main evidence.

The conference-length revision keeps the main comparison, archive and
score-blind tables, and Ant's negative utility/HV outcome in the main text.
The earlier five-seed table, bootstrap details, cohort history and interaction
accounting are retained in `experiment_details.tex`; coordinate features,
encoder settings and the resource table are in `implementation_details.tex`.
The main strict-margin inequality restates Proposition `prop:candidate_margin`
from the existing Lean-backed appendix; it is not a new convergence theorem.
Under the unchanged 10pt article geometry, the main text now ends on page 8.
No numerical table, cohort, algorithm or Lean proof source was changed.

## Terminology

| Canonical term | Meaning |
| --- | --- |
| BAMOR (v25) | Unchanged `bamor_v25_progress_archive_mix`; old versions are explicitly labeled |
| Direct utility U | Execute the final policy at the requested preference, then average scalar returns over modes and weights |
| Final HV | Hypervolume of the final preference-conditioned actor's return vectors |
| Archive HV | Hypervolume of retained actor/preference return vectors with the final encoder |
| Twenty-seed study | Independent 1601-1620 cohort; a confirmation study whose primary criterion was not met |

## Remaining Scope

The 50k CAPQL archive cohort omitted action-space and Python replay RNG seeds;
it remains retrospective, not an exactly reproducible continuation baseline.
Both75k studies repair seeding but use different cohorts. This revision
does not supply a new algorithmic advance or establish ICLR main-track readiness.
A focused primary-source contribution comparison is recorded below; submission
format and the remaining bibliography have not been re-audited. The new
formal layer covers target algebra, progress-score bounds, margin-certified
local target stability, conditional tracking and archive semantics. The margin
certificate requires matched candidates and bounded critic/entropy perturbations.
The selector-loss feasibility analysis was not promoted to a performance theorem:
its tighter transfer requires calibration not established by bounded feedback.
Neural convergence and selector regret remain unproved. Historical empirical
outputs were not changed by the proof revision.

## Focused Contribution Comparison (2026-10-03)

Eight primary papers were checked for the closest mechanism overlap, rather
than for an exhaustive priority claim. Five allocation/configuration references
and CAPQL's official record URL were added to the bibliography. OpenReview's
browser challenge required the authors' public CAPQL PDF.

- [CAPQL, ICLR 2023](https://cs.uwaterloo.ca/~y328yu/publication/lu-hy-23/lu-hy-23.pdf), Section 5 and Appendix F: preference-conditioned vector actor--critic and entropy augmentation are prior components, not BAMOR inventions.
- [HPR, verified arXiv v1](https://arxiv.org/html/2601.11604v1), Section 3: return-aligned/neighborhood preference replay preserves CAPQL's architecture and losses. BAMOR changes multiple update settings online; no matched HPR result is available. The source notes acceptance for presentation at AICS 2025.
- [Robust policy bootstrapping](https://arxiv.org/pdf/2308.09734), methodology, pages 5--6: retained steppingstone policies initialize subsequent optimization. BAMOR's evaluated actor archive is separate from its current-policy training.
- [DaCosta et al., GECCO 2008](https://gpbib.pmacs.upenn.edu/gecco2008/docs/p913.pdf), Sections 3.2--4: smoothed operator credit, probability floors and dynamic bandits precede v25. DOI, authors and pages were also verified through Crossref.
- [Graves et al., ICML 2017](https://proceedings.mlr.press/v70/graves17a.html), Sections 2--3: learning-progress feedback drives task allocation. Reward-coordinate block feedback changes the signal and controlled operations, not the generic allocation principle.
- [ESBAS/SSBAS, ICLR 2018](https://www.microsoft.com/en-us/research/wp-content/uploads/2018/04/Reinforcement-Learning-Algorithm-Selection.pdf), Sections 2--3 and 5: off-policy learners share trajectories; a selector chooses their policies. BAMOR instead updates shared network parameters under selected configurations. ESBAS freezes policies within epochs; v25 does not.
- [DAC, ECAI 2020](https://www.tnt.uni-hannover.de/papers/data/1432/20-ECAI-DAC.pdf), problem formulation: configuration policies learn state-dependent control across instances. V25 uses its prescribed online feedback rule without a separately trained configuration policy.
- [MA-DAC, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7f02b39c0424cc4a422994289ca03e46-Abstract-Conference.html), Sections 3--4: joint control of heterogeneous settings in multi-objective optimization is also established. Its target is an evolutionary optimizer, not a preference-conditioned MORL actor--critic.

The supported distinction is a specific context-conditioned MORL update
portfolio, with separate execution, coverage and feedback evidence. Individual
building blocks, learning-progress selection and joint configuration are prior
ideas. The current primary intervals, negative Ant result and score-blind
outcomes remain unchanged. This is a narrower empirical algorithm study,
not an established new selector theory or a demonstrated general improvement
over CAPQL.

For the frozen-v25 submission path, an empirical MORL/adaptive-learning workshop
is the more defensible initial target in my assessment. ICLR main-track
positioning still lacks a clear, isolated advantage over the closest mechanisms.
The next manuscript step is a full reference/publication-status audit and
submission-format preparation; this revision adds no training.
