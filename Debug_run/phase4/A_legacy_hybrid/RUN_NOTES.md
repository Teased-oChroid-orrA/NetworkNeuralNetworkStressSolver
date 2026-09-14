# PH4-01 mode A — Hybrid + LegacyMeanIntegral

Status: VERIFIED (issue #65, PH4-01 mode classification matrix)

Real evidence lives at `Debug_run/baseline_legacy_no_hole/` (PH3-01's frozen checkpoint) and
`crates/pinn-solver/src/runner.rs::tests::ph3_09_resuming_the_baseline_checkpoint_tests_whether_more_training_closes_the_traction_gap`
(the real 800-step resume, 2000→2800 steps, that closed the traction gap). Not copied or
relabeled here — see `docs/PHASE_4_IMPLEMENTATION_MANIFEST.md`'s PH4-01 matrix for the full
formulation/coefficient/benchmark table, and `Debug_run/baseline_legacy_no_hole/BASELINE_NOTES.md`
for this mode's own provenance details.

Summary: FAIL on the original 2000-step frozen checkpoint (`sigma_xx_relative_error=0.0192`,
`traction_rms_over_ref=0.0140`), PASS after the real PH3-09 800-step resume to 2800 steps
(`load_transfer_ratio=1.0027`).
