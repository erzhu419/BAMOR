# Online Resource 1: BAMOR reproducibility archive

**Preference-Aligned Utility and Pareto Coverage in Non-Stationary Multi-Objective Reinforcement Learning**

Neural Computing and Applications

The supplement contains the LaTeX manuscript, rendered figures, seed-level
records used in the paper, analysis and training code, and Lean sources.
All commands below start in the extracted archive root. The benchmarks use
MuJoCo simulation. The supplement provides the compact records for analysis.
Per-episode trajectories and model checkpoints are separate experiment artifacts.

## Regenerate the paper

Use Python 3.10 and the packages listed in `requirements.txt`. The verified
environment used Python 3.10.12 and PyTorch 2.8.0+cu128 with CPU training and
evaluation. A CPU PyTorch build can be used for the analysis and smoke tests.
The zip includes the MIT-licensed CAPQL source and its local imports from the
MORL-Baselines checkout at revision
`17016d3e4e7067e9fa75eccab398f3f1d7b6e2cf` under `third_party/morl-baselines`.
The BAMOR runner supplies a logging stub when Weights & Biases is absent.

With TeX Live (including TikZ and BibTeX) and `latexmk` installed, run:

```sh
python -m pip install -r requirements.txt
python paper/make_evidence_tables.py
python paper/make_nca_review_tables.py
python paper/make_v23_operator_table.py
python paper/make_utility_figure.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error Fig1.tex paper.tex
```

The generation commands consume compact local JSON records and regenerate the
empirical tables, the operator table and Figure 2. The TikZ architecture
diagram is built by LaTeX. `paper/evidence_sources.md` maps sources to claims.
The new reductions in `results/nca_review_20261120/` preserve seed-level
command utility, shortfall, mode history and block allocation. When individual
run JSON is present, `collect_nca_review.py --output <path>` reproduces the
grid reduction; add `--routing` to reduce the 75k score-neutral control.
To check the data without overwriting any reported results, run the evidence
tests from the archive root.

```sh
PYTHONPATH=third_party/morl-baselines OMP_NUM_THREADS=1 python -m pytest -q \
  test_evidence_tables.py test_nca_review.py \
  test_progress_selector.py test_consistent_targets.py test_candidate_margin.py \
  test_capql_full_simplex.py test_utility_confirmation.py \
  test_ant_ownpref_actor_ablation.py test_pairwise_residual.py
```

The second command includes short simulator tests and may emit Gymnasium's
generic scalar-reward warning for the benchmark's vector reward interface.
To reproduce one matched seed from scratch, these commands run three 75k
training jobs plus final evaluation in new output directories:

```sh
export PYTHONPATH=third_party/morl-baselines
python run_utility_confirmation.py --root reruns/native --env mo-halfcheetah-v5 --method v25 --seed 1601
python run_utility_confirmation.py --root reruns/native --env mo-halfcheetah-v5 --method capql --seed 1601
python run_capql_full_simplex.py --root reruns/full_simplex --env mo-halfcheetah-v5 --seed 1601
```

The `--summarize` options of the training runners require complete rerun
cohorts, not the compact records. The table and figure commands above are
sufficient to reproduce the analysis from the compact records.

`run_nca_score_ablation.py` trains the matched score-neutral arm. It takes
`--env`, `--seed`, and `--root`; it accepts the corresponding completed
score-on receipt from the included `iclr_utility_confirmation_20260927`
readout.
The frozen revision protocol is in `markdown/NCA_review_protocol.md`.

## Verify the Lean proofs

From the archive root, with Elan available:

```sh
cd proof
lake exe cache get
lake build BAMORImplementation
```

The toolchain specifies Lean 4.28.0 and the Lake project selects Mathlib
4.28.0. The first build may need to retrieve dependencies and their caches.
`proof/BAMOR/Implementation.lean` contains the implementation-facing
identities, the fixed numerical reselection example, the margin bound, and
conditional tracking and archive results. The parameterized reselection
argument appears in the manuscript. `paper/evidence_sources.md` records
the verification scope.
