# BAMOR Artifact Manifest

This manifest maps manuscript figures, tables and mathematical propositions to
their source files, data readouts and Lean 4 formalizations.

This repository contains the anonymized NCA manuscript source, PDF and
rendered figures and tables. Build it without experiment data using the
command below. Regenerating the empirical assets requires the compact
readouts listed in `paper/evidence_sources.md`.

`paper/reproducibility_README.md` describes the separate Online Resource 1
archive. The archive, experiment data, checkpoints and author-facing
submission materials are not included in this Git tree.

## Manuscript & Figure/Table Generators

- Main manuscript: `paper/paper.tex`
- Problem formulation, BAMOR portfolio & target stability theorems: `paper/method_v23.tex`
- Architecture diagram (TikZ, Fig. 1): `paper/Fig1.tex`
- Main empirical evaluation & Ant diagnostics: `paper/experiments_matched.tex`
- Empirical table generator (8 tables): `paper/make_evidence_tables.py`
- Portfolio configuration generator (18 slots, 14 settings, 13 distinct
  behaviors on three-objective tasks): `paper/make_v23_operator_table.py`
- 20-seed utility & per-mode figure generator (Fig. 2): `paper/make_utility_figure.py` (`paper/Fig2.pdf`)
- Appendices:
  - Conditional tracking & archive selection proofs: `paper/theory_current.tex`
  - Context probes, preference samplers & resource settings: `paper/implementation_details.tex`
  - Portfolio configuration table: `paper/table_v23_operators.tex`
  - Benchmark protocol, exploratory pilot & interaction accounting: `paper/task_protocol.tex`, `paper/experiment_details.tex`
  - Pairwise preference-residual Ant diagnostics: `paper/ant_diagnostics.tex`
- Bibliography: `paper/refs.bib`
- Build command: `cd paper && latexmk -pdf -interaction=nonstopmode Fig1.tex paper.tex`

## Empirical Readouts Used in the Manuscript

- 20-seed 75k BAMOR vs. native CAPQL (`seeds 1601-1620`, HalfCheetah & Hopper):
  `results/iclr_utility_confirmation_20260927/readout.json` (`run_utility_confirmation.py`)
- Retrospective 20-seed 75k full-simplex CAPQL (`Dir(1)`):
  `results/iclr_capql_full_simplex_20260930/compact_readout.json` (`run_capql_full_simplex.py`)
- 50k 19-snapshot archive vs. single final policy & Quota-Regulated Portfolio:
  `results/iclr_capql_archive_extension_20260926/readout.json`
- Exploratory 5-seed 75k benchmark (`seeds 1201-1205`, HalfCheetah, Hopper, Ant):
  `results/iclr_75k_final_fresh_20260926/readout.json` (`run_final_policy_fresh.py`)
- Ant candidate-search diagnostics (`seeds 1201-1205`):
  - `results/iclr_ant_ownpref_actor_final75_20260927/readout.json`
  - `results/iclr_ant_ownpref_critic_final75_20260927/readout.json`
  - `results/iclr_ant_ownpref_adaptive_final75_20260927/readout.json`
  - `results/iclr_ant_mixed_critic_20260927/preference_transfer.json`
- Ant $2\times 2$ pairwise residual $\times$ routing study (`seeds 1301-1305`):
  `results/iclr_ant_pairwise_final75_20260927/readout.json` (`run_ant_pairwise_final75.py`)

## Formal Proof Artifact (`../proof`)

- Implementation-facing target stability, progress bounds, tracking & archive proofs:
  `../proof/BAMOR/Implementation.lean`
- Statement-to-theorem map: `../proof/BAMOR/CURRENT_PROOF_SCOPE.md`
- Supporting contraction & belief modules:
  `../proof/BAMOR/BAMOR.lean`, `../proof/BAMOR/BAMOR-Counterproof.lean`,
  `../proof/BAMOR/RegimeBelief.lean`, `../proof/ApproxContraction.lean`
- Verification build:
  `cd ../proof && lake build BAMOR BAMORImplementation 'BAMOR-Counterproof' BAMORRegimeBelief ApproxContraction`
