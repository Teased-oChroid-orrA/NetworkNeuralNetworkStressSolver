# The `compute_loss_for_lbfgs` release-mode weight-lookup bug, and the mixed Free/Fixed-hole convergence investigation

> Relocated from `CLAUDE.md` (this project's former single-file findings log) into its own topic file, per the project's "separate findings from CLAUDE.md" convention. A `debug_assert!`-only guard silently zero-weighting a mismatched loss term out of the Converge-tier objective in every release build for the project's whole history, its fix, the still-open `gui_streaming_step_zero` flake, and the energy-balance-probe/Fixed-hole-bias follow-up investigation.

## Correction: `compute_loss_for_lbfgs_panics_on_lams_missing_a_real_term_key` was never actually
## unrelated - every "1 pre-existing unrelated failure" line above is superseded

Every regression-suite line above this point (546/563/565/566/568 passed, "same 1 pre-existing
unrelated failure") was reporting this exact test failing, and every one of those sessions
correctly noted it was unrelated to THEIR OWN change - that part was true. What none of them
caught: `training_core::compute_loss_for_lbfgs`/`compute_loss_for_lbfgs_multi` looked up each
active loss term's Converge-tier (L-BFGS) weight via `lams.get(name).unwrap_or(&0.0)`, guarded
only by a `debug_assert!` that a matching key exists - and `debug_assert!` compiles to a no-op in
`--release`, which every one of those "full regression suite" runs used. This test (`#[should_
panic]`, relying on that same debug-only assert) was never testing a cosmetic gap - it was the
one signal that a loss term whose name didn't match the hand-typed `HashMap` literal (e.g. a
`LossTerm::name()` renamed without updating it) was being silently zero-weighted out of the
optimized Converge-tier objective, in production, with zero warning, for as long as this
codebase has had a decision-maker/L-BFGS path. Fixed (`required_loss_weight`, panics in both
debug and release) in a session that also produced a real mixed Free/Fixed-hole convergence
investigation - see that work's own section below. **Confirmed via a real full regression run
after the fix landed: this test now passes.** Every "1 pre-existing unrelated failure" count
above should be read as "and this failure was a real release-mode correctness gap the whole
time," not as evidence the failure was harmless.

**A second, genuinely still-open failure was found while confirming the fix above** -
`runner::tests::gui_streaming_step_zero_matches_independent_shared_function_computation`,
reproducible deterministically, single-threaded, in complete isolation (not the documented
cross-thread `tracel-ai/burn#5573` flake this file describes elsewhere - that mechanism requires
concurrent tests; this one fails alone in 7 seconds every time). Root cause: `handle_control_
messages` (from the same session that fixed `required_loss_weight` above) now returns `ControlAction::
StopImmediately` when `stop_rx` is disconnected - a deliberate, correct fix for a real zombie-
spin-loop risk. But this test calls `drop(tx_ctrl)` before starting training to signal "no
control needed," which disconnects that exact channel - the very first loop iteration hits the
disconnected-channel branch and `return`s (before `TrainingMsg::Done` is ever sent), which the
test's own `assert!(saw_done, ...)` then fails. Verified present at commit `681cdf1` (before any
of this session's own changes) via an isolated worktree run - genuinely pre-existing, not
introduced by this session. **Not yet fixed** - either the test's `drop(tx_ctrl)` pattern or the
production disconnected-channel semantics needs to change; whichever fix lands should re-verify
`run_training_pinlug_stop_message_ends_loop_before_max_steps`-style tests aren't relying on the
opposite assumption.

**`required_loss_weight`'s real-world impact - verified directly, not just via the unit tests
above.** Running every `lbfgs`-named test together (`cargo test -p pinn-solver --release --lib
lbfgs -- --test-threads=1`, ~18s, not a full-suite run) happened to include a real end-to-end
DM-enabled pin-lug headless integration test: a genuine multi-thousand-step-capable training run
that reached `Converge` tier repeatedly (`[DM@9] Align → Converge`, `[DM@19]`, `[DM@29]`, ...,
cycling back to `Align` and returning to `Converge` many times over 79 real steps), completed in
16s with a finite final loss (3.4111) and no panic, and wrote a real contact-pressure CSV. This
is the direct, real-world confirmation the plan asked for: `compute_loss_for_lbfgs_multi`'s
now-fail-closed `required_loss_weight` lookup does not break a real, live, repeatedly-Converge-
cycling pin-lug run. Alongside it, `compute_loss_for_lbfgs_panics_on_lams_missing_a_real_term_key`
now fails with a clear, actionable message ("L-BFGS: missing effective weight for active loss
term 'hole_traction'") instead of silently zero-weighting the term out, and the normal happy-path
Converge-tier computation tests (`compute_loss_for_lbfgs_still_applies_dynamic_hole_traction_cap_
via_hashmap_lams`, `step_lbfgs_multi_reduces_loss_and_updates_both_domains_params`, etc.) all
still pass unchanged. 11/11 in this targeted run, zero regressions.

**Follow-up on the mixed Free/Fixed-hole objective-vs-probe gap (the doc's own cheapest, most
decisive next test).** `user_problem::probe_energy_balance_with_field_at_points` (new, purely
additive) re-evaluates the internal-energy half of a post-update energy-balance probe at the
EXACT interior points/weights a training step's own `loss_terms()` forward pass just used
(via the new `UserDefinedProblem::interior_weights()` getter), reusing `measure_integral::
domain_integral_weighted_tensor` - the SAME function `PhysicalPotentialEnergyTerm::compute()`
itself calls under `measure_aware=true` - instead of a fresh, independently-seeded plain
resample. A single-step diagnostic test (`runner::tests::energy_balance_probe_at_trained_points_
differs_from_a_fresh_plain_resample_under_real_hole_bias_weighting`, real hole-bias weighting,
not a full AMR sweep - cheapest way to get genuinely nonuniform `interior_weights` without a
multi-hundred-step warmup) found a real, measurable, nonzero effect: the weighted probe's
internal energy differed from the plain-resample probe's by 0.9567% under one step of real
nonuniform weighting (8.862e-2 J vs 8.947e-2 J). **Interpretation, honestly scoped**: this
confirms quadrature mismatch between the two evaluators is a real, live, nonzero contributor to
any objective-vs-probe gap - not a null hypothesis - but at ~1% relative on a single step, it is
almost certainly NOT the dominant source of the doc's own observed 0.009-0.012 J gap (a ~10-13%
relative effect on that trial's ~3.4 J internal-energy scale, accumulated over many steps of
real training, not one isolated step). This single-step test cannot itself separate out genuine
model drift, so it does not close the doc's own open question - it narrows it: quadrature
mismatch is real but small: model drift and/or quadrature bias accumulated over the run are the
more likely dominant contributors, and should be investigated next (e.g. probing at multiple
points across a real multi-hundred-step trial, not just one step, to see whether the gap grows
with training duration in a way pure quadrature noise would not).

**The doc's OTHER suggested next target (Fixed-hole mirror-asymmetry/angular-shape via per-hole
embedding or increased near-Fixed-hole collocation density) was deliberately deferred, not
attempted, this pass.** `hole_bias_quadrature_weights`/`UserSamplingStrategy::with_hole_bias`
only ever bias `HoleBc::Free` holes today - issue #78's own earlier "fifth experiment" (extending
that sampling bias to a Fixed hole too, for the unrelated multi-Free-hole Kt question) was tried
and fully reverted, so no Fixed-hole-biasing infrastructure survives in the current tree to reuse.
Building it correctly (matching the existing Free-hole implementation's careful overlap/area
accounting, not a rushed approximation of it) plus a real multi-hundred-step trial and profile-
shape analysis is a real, separately-scoped unit of work - attempting a hasty version of it
risked introducing a real numerical bug into research infrastructure for no decisive evidence
gained. Left open for a dedicated future pass, same standard this file already applies elsewhere
("BLOCKED documented with evidence is not the same as complete" - this is "not yet started,
correctly scoped," a different, honest category from that).

**Update: the deferral above was closed with a real, positive, reproducible result** (opt-in
infrastructure: `UserSamplingStrategy::with_hole_bias_including_fixed`/`ArchitectureSpec.
hole_bias_include_fixed`, both default `false`, byte-identical to every existing spec - see the
paired code commit). Two real trials (`examples/problems/notched_plate_fixed_hole_bias_control.
toml`/`..._treatment.toml`, single-variable change: `hole_bias_fraction=0.4` split Free-only vs.
Free+Fixed, same geometry/architecture/network as `notched_plate_boundary_lift_exact_fixed_
trial.toml` otherwise) at BOTH 300 and 1000 real steps, via `audit_trace.rs`, comparing the
Fixed-hole profile's own angular-shape metrics computed directly from the captured profile data
(mirror-asymmetry: mean `|von_mises(theta) - von_mises(-theta)|` relative to mean `|von_mises|`;
4th angular Fourier harmonic amplitude - the same two metrics the investigation doc's own
diagnosis of this problem used):

| | 300 steps | 1000 steps |
|---|---|---|
| mirror-asymmetry, Free-only bias (control) | 7.34% | 9.18% |
| mirror-asymmetry, Free+Fixed bias (treatment) | 5.11% | **2.93%** |

**The effect is real, reproduces at both step counts, and grows stronger with more training** -
at 1000 steps the improvement is a 68% relative reduction in mirror-asymmetry, moving
substantially closer to the FEM reference's own ~0.55% symmetric baseline (a different geometry
in this project's own FEM cross-check work, cited for scale, not a direct number-for-number
comparison here). Peak von-Mises also dropped (6.51e7 → 6.09e7 Pa at 1000 steps, ~6.3%). The
4th-harmonic-amplitude metric was NOT consistently directional (improved at 300 steps, slightly
worse at 1000) - mirror-asymmetry is the more decisive, reproducible signal of the two.

**This is a genuinely different outcome from the FIVE previously-falsified hyperparameter axes**
(collocation density, network capacity, learning rate, coordinate embedding, Fixed-hole sampling
bias tried once before for the UNRELATED multi-Free-hole Kt question) - this is the same lever
(near-hole collocation density) applied to a DIFFERENT question (this hole's own angular shape,
not a different hole's Kt value) on a DIFFERENT architecture (boundary-lift + exact-Fixed-
projection, not `MultiHoleHardConstraint`), and it moved the needle substantially. Confirms the
plan's own Step 4a hypothesis was correct for this case.

**Honestly still open**: not yet validated against real FEM ground truth for absolute accuracy
(this comparison is relative - control vs. treatment on the SAME metric - not yet cross-checked
against `tools/multi_hole_reference.py`'s own mixed-BC-correct solve for this exact geometry).
2.93% mirror-asymmetry is a real improvement, not yet "solved" (FEM's own baseline elsewhere is
~0.55%). Stability beyond 1000 steps, and whether increasing `hole_bias_fraction` further (or
combining with per-hole coordinate embedding) compounds the improvement, are real, untested,
disclosed next steps - not assumed either way.

## CORRECTION (later session): the "real, reproducible 68% improvement" above does not survive
## a controlled, deterministic re-test - the original result was very likely a measurement
## artifact, not a genuine effect of `hole_bias_include_fixed`

A later session (see `docs/findings/findings_multi-hole-kt-formulation-audit.md`'s own
"persistent-AMR/hole-bias interaction bug" section) found that this exact control/treatment
trial pair, as originally run, could not possibly have produced two different numbers in the
first place: `apply_persistent_adaptive_interior_sample` (issue #75) unconditionally overwrites
both the sampled interior points and the interior weights `hole_bias_fraction`/
`hole_bias_include_fixed` control, every step, whenever `amr_enabled=true` (this trial's own
config) and the geometry has holes - confirmed by re-running the EXACT unmodified control/
treatment files and getting byte-for-byte identical `updates.jsonl` output (same MD5) across all
1000 steps. **The `7.34%/9.18%` vs. `5.11%/2.93%` numbers above cannot have come from
`hole_bias_include_fixed` actually doing anything** - the two runs' real interior sampling was
provably identical the whole time. The most likely real cause: these runs used the default Wgpu
backend, whose weight initialization is independently documented elsewhere in this project as
non-deterministic run-to-run - two separately-launched Wgpu processes getting different random
initial weights would produce a genuinely different trajectory for reasons having nothing to do
with the flag under test.

**A fix was applied** (`runner.rs`'s `hole_bias_fraction` block now detects when persistent-
adaptive AMR will override it and prints a one-time warning instead of silently computing dead
weights - see the pinn-solver `AGENTS.md` Contracts section), and **a real, valid, non-confounded
re-test was then run**: `examples/problems/notched_plate_fixed_hole_bias_{control,treatment}_
no_amr.toml` (identical to the originals except `amr_enabled=false`, so `hole_bias_fraction`/
`hole_bias_include_fixed` genuinely control the sampled points this time - confirmed by the two
runs' `updates.jsonl` now having DIFFERENT MD5 hashes, unlike the AMR-on pair). Real result at
1000 steps, deterministic NdArray backend:

| | control_no_amr (Free-only) | treatment_no_amr (Free+Fixed) |
|---|---|---|
| hole0 (Free) Kt | 2.8785 | 2.8968 |
| hole0 mirror-asymmetry | 56.28% | 56.48% |
| hole1 (Fixed) Kt | 0.9336 | 0.9632 |
| hole1 mirror-asymmetry | 46.28% | 47.56% |

**`hole_bias_include_fixed=true` does NOT improve the Fixed hole's mirror-asymmetry under a
genuinely controlled comparison - if anything it is marginally worse** (46.28%→47.56%, within
noise but directionally opposite the original claim). Both configurations' raw mirror-asymmetry
(46-56%) is also far higher than the AMR-on runs' apparent 2.93-9.18% - because AMR's own
residual-driven + geometry-locked near-hole refinement (independent of `hole_bias_include_fixed`
entirely) was doing most of the real density-improvement work in every prior measurement of this
geometry, not the flag this trial exists to test.

**Conclusion: the near-Fixed-hole collocation-density-bias hypothesis (this file's own earlier
"real, positive, reproducible result") is NOT supported by valid evidence.** The infrastructure
(`with_hole_bias_including_fixed`, `ArchitectureSpec.hole_bias_include_fixed`) is harmless and
stays (`false` by default, byte-identical to every pre-existing spec, and it IS functional on the
`--headless` path and on the GUI-streaming path with `amr_enabled=false`) - but do not cite the
`7.34%/9.18%→5.11%/2.93%` table above as evidence it helps anything. Real FEM ground truth for
this exact geometry (`tools/multi_hole_reference.py`, converged mesh): hole0 (Free) Kt_vm≈3.21
(trending toward ~3.22-3.25 at finer mesh), hole1 (Fixed) Kt_vm≈1.35 - both `_no_amr` trained
results (Kt=2.88-2.90 / 0.93-0.96) sit 10-31% low, consistent with a config that never gets
AMR's own real density benefit rather than any effect of the flag under test. Item 2 from the
original plan (increasing `hole_bias_fraction` further, or combining with per-hole embedding) is
not worth pursuing on this specific hypothesis without new evidence the underlying lever does
anything at all.

