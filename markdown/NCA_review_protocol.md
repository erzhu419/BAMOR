# NCA revision evidence protocol

## Paired routing control

Reuse the 20 score-on BAMOR runs (1601--1620) on each of HalfCheetah and
Hopper. Train only the matching score-neutral arm with the pre-existing
`bamor_v25_progress_fixed_archive_mix` implementation. Both arms compute
progress scores; the neutral arm sets them to zero only while making an
allocation decision. Match all other settings of the original BAMOR
configuration, including stochastic collection, warmup and family quotas,
75,000 training steps, mode clock, final-only evaluation, preference grid
and reset seeds. Dispatch independent two-thread tasks without pinned nodes.
Maximum added work: 3,000,000 training interactions, 10,080,000 evaluation
interactions and 80 initialization probe steps. Report actual consumption.

Report score-on minus score-neutral direct utility, separately by task, using
all 20 pairs, a 97.5% marginal paired percentile interval (20,000 whole-seed
resamples, RNG 0), and the paired t interval as a sensitivity check. This is
a reviewer-requested retrospective completion, not a fresh confirmation.
Report the slot allocation for all 46 complete feedback blocks and for the
36 post-warmup complete blocks, plus the paired difference in allocation
counts and chronological slot assignments. Record the final 1,000-step
partial block separately; it does not receive complete-block feedback.

## Zero-interaction reanalysis

Use final-policy return vectors already saved for all three comparison
methods on the two main tasks and the BAMOR/native-CAPQL Ant exploratory
comparison. Recompute the by-mode direct utility and the same-actor grid
maximum, and reconcile both with the archived records. Form request-level
gaps before averaging. Report by train-seen/unseen mode and by simplex
vertex, non-vertex boundary and interior. Pair only within training seed.
Treat the same-reset grid maximum as retrospective, not as a selector
whose expected utility was measured on held-out resets. Report the same
paired t sensitivity on the original utility contrasts. Keep the existing
evaluation protocol and narrow the claims to locked-mode deployment.
