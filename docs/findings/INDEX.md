# Findings index

Single source of truth for this project's engineering-history log. Each entry below is a
chronologically-ordered investigation/feature record for one topic, relocated out of `CLAUDE.md`
(which now carries only durable project instructions and the Intent Layer navigation section) so
no single file has to be read in full to find one finding. Grep this index or the topic titles
below for a keyword before opening a file; each file also greps cleanly by issue number
(`#77`, `#78`, ...) or symbol name on its own.

Naming convention: `findings_<topic>.md`, topic in kebab-case, grouped by subsystem/epic rather
than strictly by crate (most findings here span 2+ crates).

## Core architecture

- [findings_bvp-trait-core.md](findings_bvp-trait-core.md) — Pluggable `BoundaryValueProblem`/`LossTerm` trait, SI/US-customary units, reference-scale normalization.
- [findings_training-loop-optimizer.md](findings_training-loop-optimizer.md) — Multi-domain Converge-tier L-BFGS, SOAP-Muon optimizer, tensor backend alias, `ConvergenceTracker`'s warm-restart cascade.
- [findings_hardware-adaptive-execution.md](findings_hardware-adaptive-execution.md) — `Executor`/`ExecutionPlanner` scaffolding, diagnostics instrumentation, the measured-and-rejected Rayon parallelization, `PerformanceProfile`'s real Eco-mode effect (Phase 1-4).
- [findings_toy-beam.md](findings_toy-beam.md) — `toy_beam`: standalone 1D DEM methodology sanity check.

## GUI / TUI

- [findings_gui-and-tui.md](findings_gui-and-tui.md) — GUI panel wiring (Kirsch/pin-lug/user-defined), training-panel redesign, N-hole heatmap overlay, and the `--tui`/`-T` ratatui dashboard.

## User-defined plate problems (`--problem-spec`)

- [findings_user-problem-ingestion.md](findings_user-problem-ingestion.md) — `UserDefinedProblem`/`step_physics_multi`, v1 scope cuts.
- [findings_issue77-architecture-wiring.md](findings_issue77-architecture-wiring.md) — Issue #77 PH4-45: making the three corrected architectures (`SingleDomain`, `SequentialTwoStage`, log-polar embedding) TOML-reachable.
- [findings_sampling-resampling-regression.md](findings_sampling-resampling-regression.md) — Issue #64/#66/#73: frozen-quadrature-node overfitting bug and its GUI-path twin, fixed via shared resample functions.
- [findings_phase4-closeout.md](findings_phase4-closeout.md) — Issue #63 Phase 4 close-out: no-hole benchmark operational, L5 hole/Kt accuracy still blocked, the real upstream burn-autodiff race fix.

## Multi-hole Kt accuracy (Issue #78 epic, chronological)

- [findings_multi-hole-kt-hardening.md](findings_multi-hole-kt-hardening.md) — Stage 1: N-hole sampling/AMR/Kt-reporting machinery proven correct; accuracy machinery does not yet generalize.
- [findings_multi-hole-kt-ansatz-saturation-scale.md](findings_multi-hole-kt-ansatz-saturation-scale.md) — Stage 2 + three follow-ups: `MultiHoleHardConstraint` ansatz, falsified per-hole-embedding hypothesis, the real root cause (vanishing gradient at the Kt margin) and its derived `TARGET_PHI_AT_MARGIN` fix.
- [findings_multi-hole-kt-trainable-scale-decomposition.md](findings_multi-hole-kt-trainable-scale-decomposition.md) — Items 3-4 and close-out: gradient-trained `hole_scales`, N-hole kinematic decomposition (`MultiAnnularDecompositionProblem`), four previously-open items closed.
- [findings_multi-hole-kt-formulation-audit.md](findings_multi-hole-kt-formulation-audit.md) — Independent closed-form audit proving the saturation_scale fix broke the formulation ceiling, plus a real bug (found and fixed): `hole_bias_fraction`/`hole_bias_include_fixed` were silently inert on the GUI-streaming path whenever persistent-adaptive AMR is active (the default) — now warns instead of silently discarding.

## Process / infrastructure

- [findings_ci-workflow.md](findings_ci-workflow.md) — GitHub Actions concurrency cancel-in-progress group, feature-branch-first push discipline.
- [findings_lbfgs-weight-lookup-bug.md](findings_lbfgs-weight-lookup-bug.md) — `compute_loss_for_lbfgs`'s `debug_assert!`-only guard silently zero-weighting a mismatched loss term in every release build (fixed); the `gui_streaming_step_zero` flake (fixed); the mixed Free/Fixed-hole convergence investigation; **CORRECTED**: the near-Fixed-hole collocation-bias "68% mirror-asymmetry improvement" did not survive a controlled re-test — real effect is ~0, original result was very likely a Wgpu weight-init nondeterminism artifact.

## Adding a new finding

Append to the most specific existing file if the topic matches; otherwise create a new
`findings_<topic>.md` (short header: title + one-line scope, same shape as the existing files)
and add one line to the relevant section above (create a new `##` section here if it's a new
subsystem). Do not add findings directly to `CLAUDE.md` — that file is for durable, current
project instructions only (architecture map, entry points, invariants, pitfalls), not
chronological investigation history.
