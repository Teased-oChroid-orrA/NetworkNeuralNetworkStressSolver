# pinn-solver

## Purpose

Owns: the training loop and everything that drives it — optimizer (SOAP-Muon), all three
problems' physics (Kirsch, pin-lug, user-defined N-hole plate), the `BoundaryValueProblem`/
`LossTerm`/`DomainSamplingStrategy` trait implementations, decision-maker/L-BFGS Converge tier,
checkpointing, AMR sampling density, and the GUI-streaming/headless/TUI training entry points.

Does not own: geometry/material/sampling primitives or `ProblemSpec`/`ArchitectureSpec` parsing
(`pinn-core`, no ML deps), `egui` rendering (`pinn-gui`), CLI/env parsing or GUI/headless/TUI
*dispatch* (`pinn-app` — this crate provides the functions `pinn-app`/`pinn-gui` call into).

Full chronological history for everything below: [`docs/findings/INDEX.md`](../../docs/findings/INDEX.md)
(most entries there are pinn-solver-specific — grep by symbol/issue name).

## Code Map

### Find It Fast

| Looking for... | Go to |
|---|---|
| Kirsch physics (frozen, byte-proven) | `kirsch_problem.rs`, step driver in `training_core.rs::step_physics` |
| Pin-lug / Signorini contact physics | `pinlug_problem.rs`, `signorini.rs` (pure-`f64` CPU oracle, not a production call site), step driver `training_core.rs::step_physics_multi` |
| User-defined N-hole plate problem (`--problem-spec`) | `user_problem.rs` (`UserDefinedProblem`, `AnnularDecompositionProblem`, `MultiAnnularDecompositionProblem` — the three biggest structs in the crate, all in this one 700KB file) |
| New `LossTerm` / boundary condition | `user_problem.rs` (`LossTerm` impls) + `pinn-core/src/problem.rs` (the trait) |
| New hard-constraint ansatz | `kirsch_hole_correction.rs` (`AnnulusAnsatz`, `HoleTractionFreeAnsatz`, `DirichletAnsatz` impls) |
| GUI-streaming plate training loop | `runner.rs::run_user_problem_training_from` (shared by fresh-run and resume) |
| Headless plate training loop | `user_runner.rs::run_headless_user_problem` |
| Kirsch/pin-lug GUI training loops | `runner.rs::run_training` / `run_training_pinlug` (two separate functions, not one branching function — see Contracts) |
| Kirsch/pin-lug headless loops | `headless.rs` |
| SOAP-Muon optimizer | `optim/soap_muon.rs`; runtime AdamW-fallback wrapper is `optim/mod.rs::WeightOptim` |
| Multi-domain Converge-tier L-BFGS | `training_core.rs` (`step_lbfgs_multi`, `compute_gradient_conflict_multi`, `required_loss_weight`) |
| `PinnDecisionMaker` (Explore/Align/Converge) | `decision_maker.rs` |
| Warm-restart cascade on plateau/crash | `controllers.rs::ConvergenceTracker` |
| Per-step wall-clock instrumentation | `diagnostics.rs::StepTimer` (opt-in, zero-cost when disabled) |
| Hardware-adaptive execution stub | `execution.rs` (`Executor`, `ExecutionPlanner` — deliberately still a stub, see Contracts) |
| SAW-BRDR loss-weight adaptation | `saw_brdr.rs` |
| Checkpoint save/load | `checkpoint.rs` |
| Persistent geometry-aware AMR (issue #75) | `user_problem.rs::apply_persistent_adaptive_interior_sample` |
| `ElasticityNet` (the actual MLP) | `network.rs` — gates (`PirateNet`), SIREN, coordinate-skip, `hole_scales` all live here |
| Pin-lug contact-pressure CSV export | `contact_export.rs` |
| 1D toy-problem methodology sanity check | `toy_beam.rs` + `examples/toy_beam.rs` |
| Real end-to-end training-run capture (no GUI) | `examples/audit_trace.rs` — reuses `runner::run_training_user_problem`, writes `updates.jsonl` |
| Instant-inference-over-varying-(E,ν,P) problem | `parametric_problem.rs` (a *different* per-step-caching pattern from the plate problem — see Pitfalls) |

### Key Relationships

- `pinn-core` → `pinn-solver`: one-directional. Nothing in `pinn-core` imports from here.
- `training_core.rs` is the single source of truth for the per-step physics computation; every
  problem (`kirsch_problem.rs`/`pinlug_problem.rs`/`user_problem.rs`) builds a `StepCtx`/
  `MultiStepCtx` and calls into it — no problem re-implements its own step/backward/optimizer
  loop.
- `runner.rs` (GUI-streaming, `TrainingMsg`/`ControlMsg` channel protocol) and `headless.rs`/
  `user_runner.rs` (no channel, plain `println!`) are separate entry-point layers over the same
  `training_core`/problem-trait machinery — see Contracts for why they aren't unified.

## Contracts

- `training_core::step_physics` (Kirsch) and `step_physics_multi` (N-domain) are two separate
  functions BY DESIGN — never collapse one into a thin wrapper around the other. `step_physics`
  is byte-proven against a pre-trait hardcoded reimplementation; `step_physics_multi`'s own test
  proves the two agree at N=1. Collapsing them removes the independent regression oracle each
  one exists to be.
- Converge-tier (L-BFGS) loss-weight lookup (`required_loss_weight`) fails closed — panics on a
  missing key in BOTH debug and release. This replaced a `.unwrap_or(&0.0)`-style silent default
  that zero-weighted a real loss term out of production training for the codebase's entire
  history before anyone noticed (`debug_assert!` compiles to a no-op in `--release`, so every
  release-mode regression run passed clean). See
  [`docs/findings/findings_lbfgs-weight-lookup-bug.md`](../../docs/findings/findings_lbfgs-weight-lookup-bug.md).
- `runner.rs::run_training`/`run_training_pinlug` (Kirsch/pin-lug GUI loops) and
  `headless.rs::run_headless`/`run_headless_pinlug` stay independently-maintained, un-unified
  with the user-defined-plate path's shared `resample_plate_step_data`/`plate_multi_step_ctx` —
  they've never exhibited the staleness-bug class those functions exist to prevent, and forcing
  them through a shared abstraction risks their own frozen/byte-tested paths for no evidenced
  benefit.
- Every `LossTerm::compute()` must keep its returned tensor connected to `inputs`'s live
  autodiff graph end-to-end. A host round-trip (`.into_data()`/`.to_vec()` then
  `Tensor::from_data`) mid-computation creates a fresh, disconnected leaf — the term's scalar
  VALUE still looks correct in logging/SAW-BRDR bookkeeping, but it silently supplies zero
  gradient. This exact bug existed in the Signorini penetration/non-tension penalty terms; fixed
  and now the subject of `compute()`'s own doc-comment warning.
- **RESOLVED, not just documented**: per-step interior/boundary/named resample plus AMR
  (residual-driven sweep and persistent-adaptive, issue #75) used to be TWO independently-
  maintained copies — `runner.rs`'s GUI-streaming loop had real AMR; `user_runner::
  run_headless_user_problem` had none at all, regardless of `amr_enabled`. This is now ONE
  shared implementation, `user_problem.rs::resample_step_data_with_amr`, called every step by
  BOTH `run_headless_user_problem` and `runner::run_user_problem_training_from` — `--headless`/
  `--tui`/GUI/`audit_trace` all run identical sampling+AMR now. Verified end to end (not just
  compiled): a real `--headless` run of `notched_plate_fixed_hole_bias_control.toml` now produces
  Kt bit-for-bit identical to the same config's own `audit_trace` (GUI-streaming) run. The
  `hole_bias_fraction`/`hole_bias_include_fixed`-vs-AMR override (persistent-adaptive AMR always
  wins whenever `amr_enabled=true` and holes are present) is unchanged in effect, but the
  one-time `[warning]` now fires identically on every entry point instead of citing headless as
  exempt. **Consequence**: every headless-obtained Kt number in this file's Issue #77/#78 history
  predates this fix and never benefited from AMR — re-running any hole-bearing config via
  `--headless` will likely produce a different (generally better) Kt than what's documented. See
  [`docs/findings/findings_lbfgs-weight-lookup-bug.md`](../../docs/findings/findings_lbfgs-weight-lookup-bug.md)'s
  CORRECTION section and
  [`docs/findings/findings_multi-hole-kt-formulation-audit.md`](../../docs/findings/findings_multi-hole-kt-formulation-audit.md)
  (both sections, including the follow-up describing this exact fix and its verification).
- **A flat `LossTerm` base weight calibrated for one geometry class can be catastrophically wrong
  for another - SAW-BRDR's adaptive multiplier cannot rescue an orders-of-magnitude raw-magnitude
  gap.** `EquilibriumTerm`'s base weight (`LAM_EQUILIBRIUM_PLATE=50.0`) was calibrated against a
  NO-HOLE plate, where its gradient was too WEAK. For a hole-bearing hard-constraint multi-Free-
  hole geometry, the same term's Hessian-residual magnitude near a hole boundary is enormously
  LARGER instead - real `audit_trace` evidence showed it holding 99.95-100% of SAW-BRDR gradient
  share at EVERY checkpoint of a 3000-step run (including step 0, before any adaptation), and a
  1000x-to-5,000,000x base-weight cut barely moved that share - the gap is many orders of
  magnitude, not a few-x mismatch. Fixed via a hole-presence-conditioned override
  (`LAM_EQUILIBRIUM_PLATE_HOLE_BEARING=0.001`, selected by `self.spec.geometry.holes.is_empty()`
  in `UserDefinedProblem::base_weight`) - the no-hole path is untouched. **A too-aggressive cut
  is its own real failure mode, not just "safe to overshoot"**: `0.00001` looked perfect at a
  300-step snapshot (Kt near FEM) but DIVERGED over the full 3000 steps (removing too much of the
  term's stabilizing PDE-consistency pull) - a short-run snapshot is not sufficient evidence of
  stability for this kind of weight change; always re-check the FULL configured step count.
  Full story: [`docs/findings/findings_multi-hole-kt-formulation-audit.md`](../../docs/findings/findings_multi-hole-kt-formulation-audit.md).

## Entry Points

(See root `CLAUDE.md`'s Entry Points table for CLI-level tasks. Crate-internal ones:)

| Task | Start Here |
|---|---|
| Add a new architecture flag to `[architecture]` | `pinn-core::ArchitectureSpec` (field) + `user_problem.rs::UserDefinedProblem::new_with_hard_constraint_ansatz` (dispatch) |
| Add a new pin-lug Converge-tier cap | `training_core.rs` (search `dynamic_lam_*_cap`) |
| Add a new decision-maker tier transition | `decision_maker.rs::PinnDecisionMaker` |

## Pitfalls

- A disconnected `crossbeam_channel` control receiver (`stop_rx`) is treated as
  `ControlAction::StopImmediately` inside the training step loop (deliberate, correct
  zombie-spin-loop fix). A test that `drop(tx_ctrl)` *before* calling the training function, to
  get prompt post-training cleanup, instead bails out of the step loop before
  `TrainingMsg::Done` is ever sent. See
  `runner::tests::gui_streaming_step_zero_matches_independent_shared_function_computation`'s own
  fix (queue one harmless `ControlMsg::ExportContactPressure` per expected iteration, then drop).
- `hole_bias_quadrature_weights_including_fixed` must be computed PER hole, not pooled across
  all biased holes — `sample_interior` draws an equal point count per hole regardless of that
  hole's own bias-disk area, so holes of different radii have genuinely different local density.
  Invisible in every test geometry that happens to use equal-radius holes (fixed in commit
  `6df74c5`).
- `parametric_problem.rs`'s `FixedPoints`/`build_fixed_points` looks like the same per-step
  caching bug issue #64 fixed elsewhere in this crate — it is not a bug there. Parametric
  problems (instant inference over varying E/ν/P) never claimed per-step resampling; one static
  point draw for the whole run is the intended, unchanged behavior.
- `burn`'s `Module`/`Tensor::clone()` is cheap and identity-sharing (same autodiff `NodeId`), not
  a deep copy that mints a fresh leaf. Two "independent" training runs from "the same starting
  weights" built via clone/move of one shared model instance can silently share autodiff state
  across threads — the real root cause behind a documented upstream `burn`-autodiff race
  (tracel-ai/burn#5573) that looked like float nondeterminism for a long time. Build two
  separate `.init()` calls under the same seed instead.
- `cargo test --release` on this crate is slow to *compile* (5–13 min), not to run — the
  workspace's `[profile.release]` uses `lto="thin"` + `codegen-units=1`, load-bearing for
  `burn`/`wgpu`/cubecl kernel-dispatch performance (a lighter profile was tried and made a
  GPU-kernel-heavy test hang for minutes instead of running in ~1s). Batch multiple test-name
  filters into one invocation rather than re-running the compile per filter.

## Boundaries

### Never

- Don't route `step_physics`/`step_physics_multi` through `execution.rs`'s `Executor` — they're
  single batched `burn` tensor graphs with nothing embarrassingly parallel inside; `Executor`
  wraps ONLY the free-function resample calls, deliberately.
- Don't add Rayon as a dependency of this crate's own code (only `criterion`'s dev-dependency
  pulls it in transitively). `sample_interior` is RNG-order-sensitive (seeded LCG, near-hole
  guarantee ring prepended then truncated) — parallelizing it without deterministic RNG-stream
  partitioning silently changes point streams. Measured and rejected once already (resample cost
  is ~0.05-0.1% of a training step — see
  [`docs/findings/findings_hardware-adaptive-execution.md`](../../docs/findings/findings_hardware-adaptive-execution.md)).
