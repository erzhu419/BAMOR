# BAMOR Artifact Manifest

This file maps the manuscript claims to the retained experiment, plotting, and proof artifacts. Paths are relative to the BAMOR repository unless noted.

The repository snapshot contains the manuscript's rendered figures and tables,
so the PDF builds from a checkout without experiment outputs. Regenerating
empirical figures and tables requires the corresponding local or server results
listed below. The legacy bus simulator also requires local files in `env/data/`.

## Manuscript

- Main source: `paper/paper.tex`
- Current algorithm and losses: `paper/method_v23.tex` (frozen v25; legacy filename)
- Current experimental argument: `paper/experiments_matched.tex`
- Claim-to-source map: `paper/evidence_sources.md`
- Arm specification generated from code: `paper/make_v23_operator_table.py`
- Implementation-facing theory: `paper/theory_current.tex` (appendix)
- Context, selector settings and resources: `paper/implementation_details.tex`
  (appendix)
- Pilot, statistics and cost accounting: `paper/experiment_details.tex`
  (appendix)
- Task specification and Ant diagnostics: `paper/task_protocol.tex`,
  `paper/ant_diagnostics.tex` (appendix)
- Historical synthetic study: `paper/historical_synthetic.tex` (appendix)
- Auxiliary historical theory: `paper/theory_background.tex` (outside the PDF)
- Bibliography: `paper/refs.bib`
- Main PDF build: `cd paper && latexmk -pdf -interaction=nonstopmode paper.tex`

## Main V25 Final-Policy Comparison

- Method: unchanged `bamor_v25_progress_archive_mix`
- Frozen protocol: `results/iclr_utility_confirmation_20260927/protocol.md`
- Compact per-seed evidence and paired intervals:
  `results/iclr_utility_confirmation_20260927/readout.json`
- Runner: `run_utility_confirmation.py`; seeds 1601-1620,
  HalfCheetah/Hopper, 75k interactions, final actor only.
- Main tables: `paper/table_utility_confirmation.tex`,
  `paper/table_utility_secondary.tex`
- Table generator: `paper/make_evidence_tables.py`
- Retrospective archive comparison:
  `results/iclr_capql_archive_extension_20260926/readout.json`
- Earlier final-policy pilot:
  `results/iclr_75k_final_fresh_20260926/readout.json`
- Independent Ant structure-by-selector pilot:
  `results/iclr_ant_pairwise_final75_20260927/readout.json`

The 20-seed study has higher observed mean direct utility, lower mean final HV
and primary intervals including zero. Archive, development and later HOLD
cohorts remain separate; no later candidate replaces v25 in these tables.

## Historical Synthetic 20-Seed Benchmark

- Raw seed outputs: `results/server_curved20/*_seed*.json`
- Combined result file: `results/server_curved20/all_results.json`
- Aggregation entry point: `aggregate_bamor_diagnostics.py`
- Manuscript table: `paper/table_main_results.tex`
- Figure scripts: `paper/make_figures.py`, `paper/make_submfld_figure.py`
- Manuscript figures: `paper/fig_submfld_hero.pdf`, `paper/fig_pareto_permode.pdf`, `paper/fig_training_curves_v2.pdf`

## Historical Continuous-Control Policy-Set Study

These outputs use unmatched archive sizes and a persistent switching timeline
during overall evaluation. They are exploratory, not matched confirmation.

- Aggregation entry point: `aggregate_mujoco_policy_set_union.py`
- Refresh script: `refresh_mujoco_policy_set_artifacts.sh`
- Cross-environment v23 summaries:
  - `results/mujoco_policy_set_union_bamor_v23_crossenv20_summary.json`
  - `results/mujoco_policy_set_union_bamor_v23_crossenv20_summary.md`
  - `results/mujoco_policy_set_union_bamor_v23_crossenv20_seed_diagnostics.csv`
- Earlier policy-set comparison summaries:
  - `results/mujoco_policy_set_union_3d20_summary.json`
  - `results/mujoco_policy_set_union_3d20_summary.md`
  - `results/mujoco_policy_set_union_3d20_seed_diagnostics.csv`
- Manuscript table: `paper/table_mujoco_3d_policy_set.tex`
- Manuscript figure: `paper/fig_mujoco_policy_set_gain.pdf`
- Raw scheduler outputs are retained under `results/mujoco_*`.

## Historical Matched V23 Pilot

- Protocol and adoption boundary: `ICLR_MINIMAL_REVISION.md`
- Training and evaluation: `train_bamor_mujoco.py`
- Tests: `test_matched_evaluation.py`
- Server-side small summary: `python summarize_matched_pilot.py`
- Remote output root: `results/iclr_matched_v23_pilot_20260913`
- Scheduler tasks: `t93490` through `t93516`; seeds 20-22, three methods, three environments.
- `hv_mean_mode` is a mean of separately evaluated mode HV values, not the old
  cross-mode `hv_overall`. The historical union aggregator is not used here.

## Formal Proof Artifact

- Lean project: `../proof`
- BAMOR proof source: `../proof/BAMOR/BAMOR.lean`
- Current implementation-facing proofs: `../proof/BAMOR/Implementation.lean`
- Current proof mapping: `../proof/BAMOR/CURRENT_PROOF_SCOPE.md`
- BAMOR counterproof source: `../proof/BAMOR/BAMOR-Counterproof.lean`
- Imported BAPR source: `../proof/BAPR/BAPR.lean`
- Approximate-contraction source: `../proof/ApproxContraction.lean`
- Proof build from `../proof`: `lake build BAMOR BAMORImplementation 'BAMOR-Counterproof' BAMORRegimeBelief ApproxContraction`
