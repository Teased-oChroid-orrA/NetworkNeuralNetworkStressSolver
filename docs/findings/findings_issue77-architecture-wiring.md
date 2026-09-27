# Issue #77 PH4-45: GUI/headless wiring for the three corrected architectures

> Relocated from `CLAUDE.md` (this project's former single-file findings log) into its own topic file, per the project's "separate findings from CLAUDE.md" convention. Making the single-domain hard-constraint ansatz, sequential two-stage training, and log-polar embedding architectures reachable via `[architecture]` in a TOML spec instead of test-only.

## Issue #77 PH4-45: GUI/headless wiring for the three PH4-41..44-corrected architectures

PH4-41..44 fixed the Kt-measurement bug and proved three architectures (single-domain
hard-constraint ansatz, sequential two-stage, log-polar embedding) converge to FEM within
0.67-1.23% — but none of that was reachable outside test code, and the GUI heatmap for any
`decomposition_applicable` single-centered-Free-hole spec had the SAME field-reconstruction bug
PH4-41 fixed in the diagnostic probe (`evaluate_user_vis_grid` did a bare forward pass, no
ansatz, no affine background). This closes both gaps.

**`evaluate_user_vis_grid` fix — a real, currently-shipping bug, not just a missing feature.**
Gained `ansatz: &dyn DirichletAnsatz`/`affine_strain_pair: Option<(f64,f64)>` parameters and now
routes through `training_core::stencil_forward_with_ansatz` (the same shared helper the
corrected probe uses), adding the affine background strain/displacement exactly like
`probe_hole_boundary_profile_derived`. Every one of its ~11 call sites was fixed the same way
the probe's were — the compiler enumerates them exhaustively, same discipline as PH4-41.
Proven by two closed-form tests mirroring the probe's own proofs
(`evaluate_user_vis_grid_adds_affine_strain_exactly`,
`evaluate_user_vis_grid_reflects_hard_constraint_ansatz_near_hole_boundary`).

**`ProblemSpec.architecture: ArchitectureSpec`** (`pinn-core/src/problem_spec.rs`) — the
first TOML-reachable selector for the three architectures, every field `#[serde(default)]` to
the exact pre-#77 dispatch: `hard_constraint_ansatz: bool`, `hole_bias_fraction: f64`,
`coordinate_embedding: CoordinateEmbeddingSelection` (`Cartesian`/`LogPolar`),
`training_procedure: TrainingProcedure` (`Joint`/`SingleDomain`/`SequentialTwoStage{stage_a_
steps, stage_b_steps}`). See `examples/problems/issue_77_l5_hard_constraint.toml` — PH4-42's
own exact verified config, now runnable via `--headless --problem-spec`, not just a test.

**`TrainingProcedure::SingleDomain` exists because of a real dispatch ambiguity the first
draft of this wiring got wrong.** Every "L5" shape this whole investigation used (single
centered Free hole, enough margin for `annular_partition()`) qualifies for BOTH the plain
single-domain `UserDefinedProblem` path AND `AnnularDecompositionProblem`'s two-domain Joint
path. `hard_constraint_ansatz=true` alone doesn't disambiguate which model it should apply
to — Phase 1's real PH4-42 result used `UserDefinedProblem::new_with_hard_constraint_ansatz`
(single-domain), a DIFFERENT architecture from Phase 3's `AnnularDecompositionProblem::
new_with_log_polar_embedding` (two-domain), even though both can carry the same ansatz flag.
Without `SingleDomain` forcing the single-domain branch, `run_training_user_problem`/
`run_headless_user_problem`'s pre-existing dispatch order (check `AnnularDecompositionProblem::
supports` first) would silently route Phase 1's own example config through the two-domain
architecture instead — reachable, but not the one PH4-42 verified. `Joint` (default) keeps the
exact pre-#77 dispatch; `SingleDomain` is required whenever a spec wants the genuinely
single-domain architecture on a geometry that also happens to qualify for annular
decomposition.

**Live Kt + a correct spliced heatmap for the annular paths** — previously hardcoded
`vis: None`/`kt_estimate: None` ("no correct two-model field evaluator existed"). Fixed via:
- `run_annular_decomposition_training_inner`'s `diagnostics: &mut Vec<AnnularL5Diagnostic>`
  parameter became `on_diagnostic: &mut dyn FnMut(AnnularL5Diagnostic, &annulus_model,
  &outer_model)` — a sink, not an accumulator, so a live caller gets both the diagnostic AND
  the two models at the exact checkpoint. All 10 existing wrapper functions (`run_annular_
  decomposition_training_with_diagnostics_and_*`) and the bare `run_annular_decomposition_
  training` were mechanically updated to pass `&mut |d, _a, _o| diagnostics.push(d)` — same
  values, same order, same cadence, purely a delivery-mechanism change. Same additive-sink
  pattern applied to `run_annular_decomposition_training_sequential` (new trailing
  `on_diagnostic` parameter; the 3 existing test callers pass `&mut |_d, _a, _o| {}` and keep
  reading the returned `Vec` exactly as before).
- `evaluate_annular_vis_grid` (`user_problem.rs`) — the two-domain analogue of `evaluate_user_
  vis_grid`: evaluates both models over the full grid with their own real training-time
  ansatz, then splices per-cell by physical distance from the hole center vs.
  `interface_radius` (annulus model inside, outer model outside). Proven by
  `evaluate_annular_vis_grid_splices_at_the_interface_radius_not_one_model_everywhere` (two
  models with different seeds, asserting each region matches ONLY its own model's field).
- `runner::run_training_annular_decomposition`/the new `run_training_annular_decomposition_
  sequential` compute this at the same "every 10th step/last step" cadence the single-domain
  GUI path already uses, sharing the result with `on_step` via a `Rc<RefCell<Option<...>>>`
  (safe — diagnostic and `on_step` run sequentially within the same training step, never
  concurrently, so the `RefCell` never double-borrows).

**A second real, pre-existing bug this session's own headless smoke test caught** (not
introduced by this work): `run_headless_user_problem`'s "not a trivial collapse" diagnostic
called `model.forward()` directly on a bare 3-column `[xn,yn,0.0]` tensor, bypassing the
embedding transform entirely — panicked on ANY single-hole geometry (the model is built with
`net_input_dim()`=10 for `SingleHoleChart`, not 3). Never caught before because every real
PH4-24..44 run trained through a different function
(`run_user_problem_training_with_diagnostics`/`run_annular_decomposition_training*`), never
this exact plain-headless-against-a-real-hole path. Fixed by routing through `fwd_embedded`
(the same embedding-aware forward every other real call site already uses); regression-proven
by `run_headless_user_problem_completes_on_a_real_single_hole_geometry_without_panicking`.

**GUI surfacing** (`pinn-gui`): `TrainingState` gained `hole_analyses: Vec<HoleAnalysis>`
(populated from `TrainingUpdate.hole_analyses` in `app.rs`'s existing `Update` handler — no new
message variant) since the plain single-domain path's real Kt lives there
(`kt_estimate` stays `None` on that path; the annular paths populate `kt_estimate` directly,
reusing the exact mechanism Kirsch's own Kt readout already used). `params.rs`'s User-Defined
panel shows: a read-only architecture summary (only when non-default — no interactive
widgets, TOML stays the single source of truth for User-Defined mode, matching every other
field there) and both Kt sources (`kt_estimate` and/or per-hole `hole_analyses`), no hardcoded
"theory: 3.000" comparison (no closed-form Kt exists for an arbitrary user geometry).

**Verified end-to-end via real release-binary headless runs** (not just unit tests) for all
three architectures plus the untouched default: `SingleDomain` hard-constraint (dispatches
correctly to the single-domain path despite the L5 geometry also qualifying for annular
decomposition — confirmed by banner text, absent `hole_free` term, present `rotation_gauge`/
`translation_gauge`), `SequentialTwoStage` (`stage_a`/`stage_b` both print, Kt printed at the
end), `Joint` + `LogPolar` + hard-constraint together (annular dispatch, Kt printed), and a
pre-existing `[architecture]`-free example (byte-identical dispatch/output shape to before).
Full workspace regression after every change in this pass: 537+ passed, 0 regressions (same
1 pre-existing, unrelated failure as PH4-41..44 — `compute_loss_for_lbfgs_panics_on_lams_
missing_a_real_term_key`, confirmed failing on unmodified `d9f38fe` via `git stash`).

**Not done in this pass**: a real mouse-driven GUI click-through (load a spec with
`[architecture]` set, click Start, watch the heatmap/Kt update) — same disclosed gap the
original GUI-wiring section above already flagged for its own work, for the same reason
(headless verification of the identical underlying training/diagnostic code already
establishes correctness; the GUI runner is a thin streaming wrapper around it). Worth a real
check before relying on this heavily for a live demo.

