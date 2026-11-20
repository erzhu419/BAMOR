# Manuscript Evidence Map

Regenerate all eight empirical tables with `python3 paper/make_evidence_tables.py`,
the portfolio specification table with `python3 paper/make_v23_operator_table.py`,
and the 20-seed utility comparison figure with `python3 paper/make_utility_figure.py`
from the repository root. The main comparisons recompute paired means, counts
and 20,000-resample percentile bootstrap intervals from compact seed records.
The figure checks the same inputs before plotting. The Ant table uses
cross-report alignment from the seed-level candidate interventions, previously
recorded controls and per-seed differences. For controls available only as
differences, the alignment reconstructs their values and checks independent
overlaps where available.

| Manuscript evidence | Local source under `results/` | Scope |
| --- | --- | --- |
| Table 2 (`tab:utility_confirmation`), Table 4 (`tab:utility_secondary`), Fig. 2 | `iclr_utility_confirmation_20260927/readout.json` | 20 paired seeds (`1601-1620`), HalfCheetah & Hopper, 75k interactions, BAMOR vs. native CAPQL ($22.5^\circ$ cone) |
| Table 3 (`tab:full_simplex_capql`), Fig. 2 | `iclr_capql_full_simplex_20260930/compact_readout.json` | Retrospective baseline extension on inspected seeds `1601-1620`. The protocol was frozen before full-simplex CAPQL training. |
| Table 5 (`tab:matched_archive`), Table 6 (`tab:selector_effect`) | `iclr_capql_archive_extension_20260926/readout.json` | 50k BAMOR runs (15 HalfCheetah, 5 Hopper, 5 Ant): 19-snapshot archive vs. single final policy, and progress routing vs. Quota-Regulated Portfolio |
| Table 7 (`tab:ant_search`) | `iclr_ant_ownpref_{actor,critic,adaptive}_final75_20260927/readout.json`, `iclr_ant_selector_final75_20260927/readout.json` and transfer and diagnostic records | Five inspected development seeds. The score-neutral and progress-routed interventions are separate. See `summarize_ant_mechanism_alignment.py`. |
| Appendix Table (`tab:final_policy`) | `iclr_75k_final_fresh_20260926/readout.json` | Exploratory 5-seed 75k benchmark (`1201-1205`) across HalfCheetah, Hopper and Ant |
| Appendix Table (`tab:ant_pairwise`) | `iclr_ant_pairwise_final75_20260927/readout.json` | 5 paired seeds (`1301-1305`), Ant 75k $2\times 2$ pairwise residual actor $\times$ routing interaction |
| Ant within-model vertex vs. mixed transfer | `iclr_ant_mixed_critic_20260927/preference_transfer.json` | Re-scalarization of existing evaluated returns on seeds `1201-1205` (66/70 mode-0 cells) |

The native CAPQL confirmatory contrasts both missed the prespecified criterion.
The two BAMOR pairs of contrasts retain their own 97.5% marginal intervals.
The comparison of CAPQL samplers uses descriptive 95% intervals.

The Hopper routing sensitivity uses the 50k `legacy_adaptive` and
`legacy_blind` seed rows. The full-sample ratio of mean final HVs minus one
is +59.56%. Removing the row with the largest positive paired HV difference
from both arms gives -2.96%. The table includes all seeds. The older CAPQL
archive control in the source JSON had incomplete RNG seeding. Manuscript
archive and routing results use only the BAMOR arms.

## Formal Proof Layer

The Lean 4 formal verification layer (`proof/BAMOR/Implementation.lean` in
the reproducibility archive) machine-checks:
- Scalar-vector soft target projection consistency and continuation-mask identities (`Section 5`).
- Bounded progress, translation at fixed scales, and invariance under joint
  rescaling of rewards and supplied scales (`Section 5`). The scale floor can
  break joint rescaling in code.
- The fixed numerical candidate-reselection example (`Proposition 5.1`) and
  strict score-margin local target stability (`Theorem 5.2`). The
  parameterized extension in the text is an analytical calculation.
- Approximate selected-map tracking (`Proposition A.1`), continuous Lebesgue hypervolume archive monotonicity (`Proposition A.2`), and archive policy-selection regret (`Proposition A.3`).
