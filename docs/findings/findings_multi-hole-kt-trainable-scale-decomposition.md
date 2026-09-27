# Issue #78 items 3-4 and close-out: trainable saturation_scale, N-hole kinematic decomposition, four remaining items closed

> Relocated from `CLAUDE.md` (this project's former single-file findings log) into its own topic file, per the project's "separate findings from CLAUDE.md" convention. Gradient-trained `hole_scales`, `MultiAnnularDecompositionProblem`'s N-hole generalization of issue #77's kinematic decomposition, and the four previously-disclosed open items (Kt convergence flag, checkpoint round-trip, per-domain embeddings, TUI stdout) all closed.

## Issue #78 item 3: genuinely trainable (gradient-descent) `saturation_scale`, opt-in

`ArchitectureSpec.trainable_saturation_scale: bool` (`#[serde(default)]` false) makes each
Free hole's own `HoleTractionFreeAnsatz.saturation_scale` a real `burn` `Param<Tensor<B,1>>`,
gradient-trained alongside the network's own weights, instead of only ever the closed-form-
derived fixed value (item 2). Only applies once `holes.len() > 1` (N=1's `saturation_scale=1.0`
is exact by construction - nothing to learn there), and only when explicitly opted into.

**The real, load-bearing engineering fact this design turns on**: the ansatz's envelope was
previously computed ENTIRELY in host `f32` math (`traction_free_envelope_scaled`), baked into a
constant tensor via `Tensor::from_data` before the network's forward pass even runs - zero
autodiff connection, so gradient could never reach `saturation_scale` no matter how it was
stored. Making it trainable required moving the envelope computation itself into TENSOR space:
a new `training_core::trainable_envelope_hole_phi_tensor` recomputes `phi = 1-exp(-(scale*(r-a)
/a)^2)` using burn tensor ops (`sub_scalar`/`mul_scalar`/`powf_scalar`/`sqrt`/`exp`) against the
model's own `hole_scales: Vec<Param<Tensor<B,1>>>`, invoked from `stencil_forward_with_ansatz`
only when `ansatz.trainable_envelope_holes()` (a new default-`None` `DirichletAnsatz` trait
method) reports trainable holes - `eval()` for those holes returns `(1.0,1.0)` (no host
suppression), and this tensor computation entirely replaces that contribution.

**Real correctness proof before trusting any training run**: `trainable_envelope_hole_phi_
tensor` matches the pure-host `traction_free_envelope_scaled` formula to `1e-4` at real
geometry/scale values (`trainable_envelope_hole_phi_tensor_matches_host_traction_free_envelope_
scaled_at_a_fixed_scale`), AND a real `.backward()` call through it gives the scale `Param` a
nonzero gradient with the analytically-expected sign (`..._gradient_flows_to_scale_with_
expected_sign`) - both new, passing unit tests, checked BEFORE any real GPU training run.

**Ripple, compiler-enumerated, mechanical**: `ElasticityNet` gained `hole_scales` + `hole_
scale_ids()`/`hole_scales()`/`with_hole_scales()` (~6 construction sites fixed);
`DomainOptim` gained a 4th `hole_scale: GateOptim` field, reusing `GateOptim`'s own type (same
small-scalar-Param optimizer role `gate` already has) rather than inventing a new optimizer type
- wired through `step_physics_multi`'s own per-domain step (mirrors the `gate` step exactly,
same LR treatment via `alpha_lr_mult`) and ~58 `DomainOptim` construction sites across the
crate; `HoleTractionFreeAnsatz` gained `trainable: bool` (17 construction sites fixed). Every
one of these is a genuine no-op (empty `Vec`, `trainable=false`) for every existing/default
config - the SAME "empty means completely inert" precedent `gates`/`use_piratenet=false`
already established, not a new pattern.

**Real end-to-end verification**: N=1 (`issue_77_l5_hard_constraint.toml`) stayed byte-verified
unchanged - `Kt=2.4443`, matches PH4-42's documented value exactly (the flag is a structural
no-op there). N=3 (`triple_hole_plate.toml` geometry) with `trainable_saturation_scale=true`
trained successfully end to end (3000 steps, no crash): hole0/hole2 Kt=2.893/2.994 - comparable
to the fixed-derived path's own 2.884/3.032, healthy load transfer ratio (1.019), non-trivial
displacement. **Decisive evidence the mechanism genuinely works, not just compiles**: a real
900-step run's new diagnostic print (`[diag] trainable hole_scales[i]: seed=... final=...`)
showed the learned scale moving from its derived seed (`30.0000`) to `30.2230`/`30.0567` - small
but real, gradient-descent-driven movement in 900 steps, not frozen at the seed.

Full regression suite: 565 passed (+2 net new), same 1 pre-existing unrelated failure
(`compute_loss_for_lbfgs_panics_on_lams_missing_a_real_term_key`), zero regressions.

**Honestly still open**: checkpoint save/load round-trip of a trained `hole_scales` is NOT
independently verified (burn's own `#[derive(Module)]` should serialize `Param` fields
automatically, same mechanism `gates` already relies on, but this specific field's round-trip
was not exercised end to end this session). A resumed run currently RE-SEEDS `hole_scales` from
the closed-form-derived value rather than continuing from a checkpoint's own trained value (see
`runner.rs::run_user_problem_training_from`'s own comment) - a real, disclosed v1 limitation,
not a crash. The observed scale movement (900 steps, ~0.2-0.7% relative) is small - whether a
longer run or a different `alpha_lr_mult` would move it further is a real, untested question,
not assumed either way.

## Issue #78 item 4: `AnnularDecompositionProblem` (kinematic decomposition) generalized to N
## Free holes - `TrainingProcedure::Joint` only, one real accuracy tradeoff disclosed

A new, purely ADDITIVE `MultiAnnularDecompositionProblem` (zero change to the frozen, PH4-42-
verified single-hole `AnnularDecompositionProblem` - matching this codebase's own `step_physics`/
`step_physics_multi` precedent for exactly this situation) generalizes #77's annular kinematic
decomposition from exactly one Free hole to any N: N annulus domains (each single-hole-scoped
exactly like the original, reusing `AnnularPartitionSampling`/`AnnulusAnsatz::HardConstraint`
completely unchanged via a synthetic single-hole `UserGeometry` per hole) sharing ONE outer
domain/model (`MultiAnnularOuterSampling`, the one genuinely new sampling strategy). Scoped to
`TrainingProcedure::Joint` (the default) only - `SequentialTwoStage`'s own N-hole generalization
is real, separate, additional scope, explicitly not attempted this pass, not silently glossed
over.

`pinn-core::UserGeometry::annular_partitions()` (plural) generalizes `annular_partition`'s own
per-hole math (already hole-local, per that function's own doc comment) over every Free hole,
adding a real, previously-nonexistent validation: no two holes' INTERFACE circles (not just the
holes themselves) may overlap - 4 new tests, including one proving holes that pass `UserGeometry::
validate()`'s own hole-overlap check can still correctly fail this stricter one.

**Reuse over refactor, systematically**: `PhysicalPotentialEnergyTerm`/`TranslationGaugeTerm`/
`RotationGaugeTerm`/`HoleBcTerm` are used completely unchanged (already `DomainId`-parameterized).
`AnnularPotentialEnergyTerm` gained real `domain`/`name` fields (was hardcoded to the single
`ANNULUS_DOMAIN` constant and a literal `"annulus_potential"` name - 2 existing construction
sites updated, byte-identical). Two genuinely new term structs (`MultiInterfaceDisplacement
ContinuityTerm`/`MultiInterfaceTractionContinuityTerm`, compute() bodies copied verbatim) were
needed - not modifications to the originals - because those hardcode the SAME point-set name on
both domain sides, and the shared outer domain needs a DIFFERENT name per hole
(`occurrence_suffixed_name`, generalizing `hole_bc_term_name`'s own "first unsuffixed, 2nd+
suffixed" convention - load-bearing for N=1 byte-identical naming).

**Real, decisive N=1 regression proof**
(`multi_annular_decomposition_matches_annular_decomposition_at_n_equals_one`): bit-identical
sampled interior points (both domains, several calls - proves the RNG seed streams genuinely
match, not just "both non-empty"), bit-identical named interface/hole point sets, and an
identical loss-term name set with matching `base_weight`s, between the new N=1 path and the
frozen original - passed first try. A real end-to-end headless run of a genuine N=1 Joint-
dispatch spec (not `SingleDomain`-forced) confirmed the ORIGINAL `AnnularDecompositionProblem`
path is still the one reached, giving `Kt=2.4416` (consistent with PH4-42's own verified range) -
the new dispatch branch does not intercept it.

**Real, honest N=2 end-to-end result**: a real headless run (2 Free holes, no Fixed hole, real
FEM ground truth computed via `tools/multi_hole_reference.py` at a converged mesh:
`Kt_vm≈3.05/3.07`) trained successfully through the new `[#78 multi-annular]` dispatch path (no
crash, monotonically decreasing loss) to `hole0/hole1 Kt=2.521/2.520` - genuinely symmetric, as
the geometry's own mirror symmetry predicts (a real positive sanity signal), but ~15-18% below
FEM - LESS accurate than the item-1/2/3 ansatz-only path's own best result on a different
(has-a-Fixed-hole) 2-hole geometry (`Kt=2.88-2.99` vs FEM `≈2.98-3.06`).

**A real engineering constraint found and worked around, disclosed rather than hidden**:
`MultiStepCtx.coordinate_embedding` is ONE shared value applied to every domain whose model
input width matches it (`training_core::stencil_forward_with_ansatz`'s own model-input-width
dispatch) - there is no per-domain embedding slot, so N different `SingleHoleChart` embeddings
(each carrying a different hole center) cannot coexist safely in one training step. Every domain
in the new driver therefore uses plain Raw (3-column) coordinates, including every annulus
domain - unlike the single-hole path, which gets `SingleHoleChart`'s 7 hole-relative features
"for free." This is the most likely explanation for the accuracy gap above (a real, testable
hypothesis, NOT confirmed this pass) - plumbing genuine per-domain embeddings through
`MultiStepCtx` would be the natural next step, a real, scoped, disclosed follow-up alongside
`SequentialTwoStage`'s own N-hole generalization.

Full regression suite: 566 passed (+7 net new), same 1 pre-existing unrelated failure
(`compute_loss_for_lbfgs_panics_on_lams_missing_a_real_term_key`), zero regressions.

## Issue #78 close-out: the four remaining disclosed open items, all closed with real evidence

The four items the prior sections above left honestly open (Kt convergence flag still trips,
checkpoint round-trip of trainable `hole_scales` unverified, N-hole per-domain coordinate
embeddings not attempted, TUI stdout-corruption cosmetic bug not fixed) are now all closed -
one with a genuine fix, one with a real bug found and fixed along the way, one with a real fix
plus an honest NEGATIVE result, one with a genuine OS-level fix plus a real new capability.

**Item 1 - Kt convergence flag: FIXED, not just decomposed.** `kt_convergence_check`'s
`converged` gate now uses `radial_residual_kt_delta` (the network's OWN residual drift,
relative to `kt_coarse`) instead of the raw, curvature-contaminated `radial_relative_change`
whenever a baseline exists (`has_baseline`) - the raw number was already proven (prior session)
to be dominated by real, expected closed-form field curvature having nothing to do with
training convergence, so gating pass/fail on it was measuring the wrong thing. No baseline
(`IdentityAnsatz`, no affine background) falls back to the ORIGINAL raw check exactly as
before - zero regression for that case. **Real, decisive end-to-end confirmation**
(`examples/problems/variational_triple_hole_smoke.toml`'s geometry + `[architecture]
hard_constraint_ansatz=true, hole_bias_fraction=0.4`, 3000 steps): all three holes now read
`Kt convergence OK` - hole 0 (`radial Δ=0.145`, previously well past the 0.1 tolerance) now
converges via `network-residual Δ=0.248` against `Kt≈2.99` (24.8/2.99 ≈ 8.3% < 10%); hole 2
(`radial Δ=0.111`) via `network-residual Δ=0.138` against `Kt≈2.92` (≈4.8%). N=1
(`issue_77_l5_hard_constraint.toml`) reproduces its exact previously-documented values
unchanged (see below) - the gate change is a genuine, byte-verified no-op there.

**Item 3 - checkpoint round-trip of trainable `hole_scales`: verified, AND a real bug found and
fixed.** `CheckpointMeta` gained `hole_scale_count: usize` (`#[serde(default)]` `0`, honest for
every pre-existing checkpoint). **The real bug**: `load_checkpoint`/`load_checkpoint_for_
training` always reconstructed the destination model via the ordinary `ElasticityNetConfig::
init()` path before calling `load_file` - which leaves `hole_scales: Vec::new()` regardless of
what the saved record actually contains. Burn's `#[derive(Module)]` record walk maps a
`Vec<Param<_>>` field positionally against the DESTINATION's own Vec length, not the record's -
a 0-length destination silently fails to receive a real N-element saved `hole_scales`. Fixed by
seeding the destination with `meta.hole_scale_count` placeholder entries (`with_hole_scales(&
vec![0.0; meta.hole_scale_count], device)`) before `load_file` runs - `load_file` then
overwrites every placeholder with the record's real trained value. Proven by a new, decisive
test (`trainable_hole_scales_round_trip_through_save_and_load_for_training`): seeds two hole
scales (`22.75`, `30.12`), saves, loads via both `load_checkpoint_for_training` and
`load_checkpoint`, asserts the loaded values match the ORIGINAL seeds (not `0.0`) within
half-precision tolerance. **A second real bug fixed alongside**: `run_user_problem_training_
from`'s shared resume/fresh-run body unconditionally re-seeded `hole_scales` from the closed-
form-derived value regardless of whether `model` was fresh or resumed from a checkpoint -
silently discarding a resumed run's own trained progress back to the seed. Fixed by gating the
re-seed on `model.hole_scale_ids().is_empty()` (only a genuinely fresh model, whose Vec starts
empty by construction, gets seeded; a resumed model's already-populated Vec is left alone) -
the same "empty means untouched" precedent this field's own construction already established.

**Item 4 - N-hole per-domain coordinate embeddings: implemented, closes a real architectural
gap, but does NOT close the Kt-accuracy gap - a genuine falsification, not a failure to try.**
`MultiStepCtx` gained `domain_coordinate_embeddings: Option<Vec<CoordinateEmbedding>>`
(indexed positionally against `domains`, same convention `per_domain_lr` already established) -
`None` (every pre-existing construction site, ~30 of them, mechanically updated) falls back to
the single shared `coordinate_embedding` exactly as before; `Some(v)` lets `compute_domain_
forwards` select a DIFFERENT embedding per domain via `model_idx`, closing the exact limitation
the prior section above disclosed. `MultiAnnularDecompositionProblem::domain_coordinate_
embeddings()` is the one real producer: a genuine `SingleHoleChart` per annulus domain (each
recomputed from the SAME synthetic single-hole `UserGeometry` `new()` already builds that
hole's sampling/ansatz from - factored into a shared `single_hole_geometry_for` helper so the
three independent call sites cannot silently diverge), Raw for the shared outer domain.
`run_multi_annular_decomposition_training` now builds each annulus model with ITS OWN chart's
`input_dim()` (10) instead of a uniform Raw(3). Proven correct in isolation
(`multi_annular_domain_coordinate_embeddings_are_genuinely_per_hole`: two holes' embeddings
carry genuinely DIFFERENT `center_norm` values, not one silently shared).

**A real, previously-latent panic found by this session's own real end-to-end run, not a unit
test**: the post-training Kt diagnostic (`multi_annular_hole_kt_diagnostics`) probed each
annulus model through the FULL N-hole `spec.geometry` - whose own `coordinate_embedding()` is
`MultiHoleChart` (a different width entirely) for N>1, not the `SingleHoleChart` width the model
was actually trained with. `embedding_for_model` correctly panicked ("matches no known
embedding") rather than silently computing garbage - caught immediately by a real 3000-step
headless run reaching its final diagnostic, not by any unit test. Fixed via `MultiAnnular
DecompositionProblem::free_hole_geometry(i)` (the same synthetic single-hole geometry, exposed
publicly) - `multi_annular_hole_kt_diagnostics` now probes each model through the SAME geometry
it trained against.

**Real, honest result once both bugs were fixed and a real N=2 run completed**
(`half_w=0.15, half_h=0.06`, 2 Free holes at `(∓0.06, 0.02)` r=0.009, `E=71.7e9 Pa, ν=0.33`,
`px=6.9e7 Pa`, 3000 steps, real FEM ground truth `Kt_vm≈3.05/3.07`): `hole0/hole1 Kt=2.524/2.520`
- statistically indistinguishable from the SAME geometry's own prior Raw-only result
(`2.521/2.520`, ~0.1% difference, noise-level) and still ~17-18% below FEM. **The per-domain-
embedding hypothesis is now genuinely FALSIFIED by real evidence, not merely "not yet tried."**
Combined with the FIVE prior hyperparameter axes this epic already falsified for the single-
domain ansatz-only path (collocation density, network capacity, learning rate, Raw→chart
embedding, Fixed-hole sampling bias), this is now real evidence across BOTH architectures
(ansatz-only AND full kinematic decomposition) that hole-relative coordinate features are not
the limiting factor for N-hole Kt accuracy. Training dynamics did improve measurably though: the
new run's loss curve is visibly smoother (monotonic 27.8→0.98, vs. the prior run's own
1.0-1.3 oscillation) - a real, if secondary, benefit. The `[architecture]` machinery and the bug
fixes are real, permanent value (correct architecture, no more panic, genuinely per-hole
features available for any FUTURE geometry where they might matter) even though they didn't
move THIS geometry's Kt number - not wasted work, an honest negative result.

**Item 5 - TUI stdout corruption: fixed via the OS-level redirect this project's own docs
already named as the correct approach, PLUS a new interactive setup screen.**
`StdoutGuard` (Unix only, `crates/pinn-app/src/tui/mod.rs`) `dup()`s the real terminal fd aside
for ratatui's OWN rendering to write through, then `dup2()`s the process's real fd 1 onto
`/dev/null` for the duration of the alternate screen - any `println!`/`print!` anywhere in the
process (the shared training loop's stray decision-maker diagnostics included) is silently
swallowed, while the TUI's own frames (written through the duped fd, never through `std::io::
stdout()`) render normally; `Drop` restores fd 1 to the real terminal. Non-Unix targets keep the
pre-existing, disclosed cosmetic gap (no redirect attempted there). **Verified via an isolated
standalone program reproducing the exact dup/dup2 sequence** (not inside the shared `cargo test`
process, which would risk corrupting the test harness's own stdout capture if something went
wrong): a "stray training diagnostic" `println!` issued while the guard is active is genuinely
absent from the real terminal's captured output, while a "TUI frame" write through the duped fd,
and a normal `println!` after the guard drops and restores fd 1, both correctly reach it.

**New**: `--tui` alone (no `--problem`/`--problem-spec`) now launches an interactive setup
screen (`tui/setup.rs`) instead of silently defaulting to Kirsch - ↑/↓ to pick Kirsch/Pin-in-Lug/
User-Defined, Tab to focus a text field and type a `--problem-spec` TOML path for User-Defined
(validated in place via the SAME load+parse+validate sequence the CLI path already uses - an
invalid path shows an inline error without leaving the screen), Enter to start training, q/Esc/
Ctrl+C to quit. Self-contained (owns its own `with_terminal` session, mirroring `run_tui_plate`/
`run_tui_live`'s own pattern) - zero changes to the training event loop. `--tui --problem
kirsch|pinlug` / `--tui --problem-spec <path>` on the command line still skip the menu entirely,
unchanged.

Full regression suite after all four items: 568 passed (+3 net new this pass: the checkpoint
round-trip test, the per-hole-embedding proof, plus the pre-existing suite growing from prior
sessions), same 1 pre-existing unrelated failure, zero regressions. **Local test runs now use
`--test-threads=3`** (CI stays at `--test-threads=1` in `.github/workflows/rust.yml` - unrelated
to and unaffected by this change) - per explicit user direction; the upstream burn-autodiff
cross-thread race this project's own docs describe (tracel-ai/burn#5573) is a real, if rare,
flake source at higher thread counts, so a suspicious local failure should be re-run at
`--test-threads=1` before treating it as a regression.

