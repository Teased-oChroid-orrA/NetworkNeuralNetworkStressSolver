# Phase 4 Implementation Manifest

Issue #63 is authoritative. Updated as work proceeds; `VERIFIED` requires the issue's
mathematical, source, test, runtime-artifact, and benchmark evidence.

## PH4-01 — Freeze and classify solver modes

Status: VERIFIED (issue #64 sub-issue #65)

Hypothesis: legacy Hybrid success and corrected Variational support are distinct modes.

### Mode classification matrix (no-hole, square plate, `half_w=half_h=0.10`, Al 7075-T6, `px=6.9e7` Pa)

Built entirely from existing evidence — no new training run required for this item (issue #65's
own "avoid a full run when a cheaper method proves the same thing" discipline; see the "Not
independently run" notes below for exactly where new evidence would actually be needed).

| | **A: Hybrid + LegacyMeanIntegral** | **B: Hybrid + MeasureAware** | **C: Variational + MeasureAware (pre-#64)** | **D: Variational + MeasureAware (post-#64, corrected)** |
|---|---|---|---|---|
| Formulation | `Hybrid(interior_energy, equilibrium, outer_traction, external_work)` | same term set, measure-aware integral path | `Variational` (`physical_potential`, essential constraints only) | same as C |
| Integration mode | plain `.mean()` | `domain_integral_tensor`/`boundary_integral_tensor` (unweighted, since no AMR sweep on this config) | measure-aware, same as B | measure-aware, same as B |
| Active terms | interior_energy, equilibrium, outer_traction, external_work (independently weighted) | same | physical_potential (atomic U-W), translation_gauge | same |
| Physical coefficients | `LAM_INTERIOR_ENERGY=1.0`, `LAM_EXTERNAL_WORK=20.0` — **independently weighted, the PH4-02 defect this epic exists to fix** | same as A | atomic, single `LAM_PHYSICAL_POTENTIAL=1.0` scale (PH4-03) | same as C |
| Sampling | `UserSamplingStrategy`, pre-#64 (static point cloud for no-hole geometries) | same as A | same as A (the actual root cause) | **post-#64: genuine per-call jittered stratified resampling** |
| AMR status | off | off | off | off |
| Derivative backend | FD | FD | FD | FD |
| Convergence evidence | PH3-10 real multi-signal trend evidence, `plausibly_converged` | not independently run (see below) | `#[ignore]`d full-length/16000-step runs both diverged/failed L4 (`ph3_14_variational_bridge_...`) | `assess_convergence` real evidence, ignored regression test passing |
| Benchmark result (P2-14) | FAIL on the original 2000-step frozen checkpoint (`sigma_xx_relative_error=0.0192`, `traction_rms_over_ref=0.0140`); **PASS after PH3-09's real 800-step resume to 2800 steps** (`Debug_run/baseline_legacy_no_hole/`, `ph3_09_resuming_the_baseline_checkpoint_...`) | not independently run (see below) | FAIL, five-of-five hard metrics, every controlled-ladder stage (`Debug_run/phase4/{D,E}_*`) | **PASS, all five hard metrics** (`Debug_run/phase4/issue64_resample_fix/shipped_example_final.log`; `sigma_xx=0.0010`, `sigma_yy/ref=0.0007`, `sigma_xy/ref=0.0003`, `traction_rms/ref=0.0011`, `load_transfer=0.9991`) |
| Energy balance | PH3-01/09 real values (see baseline notes) | not independently run | `1.6e-3`–`4.4e-3` (health PASS) but hard metrics FAIL — the exact "health passes, field doesn't" gap #64 root-caused | `4.7111e-4`, health PASS, hard metrics also PASS |
| Load transfer | `1.0027` (post-resume) | not independently run | `0.60`–`0.88` across every controlled stage | `0.9991` |
| Field errors | PH3-01/09 evidence | not independently run | PH4-15's `validate_no_hole_fields` did not exist yet | PASS — independent field validation wired in and passing (issue #64) |

**Mode B — why "not independently run" is the honest, correct entry, not a gap silently
approximated (issue #63 rule #16):** for THIS SPECIFIC configuration — a square plate
(`half_w==half_h`) with uniform (non-AMR) sampling — `ExternalWorkTerm`'s own doc comment and
`InteriorEnergyTerm`'s `ref_energy_absolute` derivation (this file, `user_problem.rs`) establish
that the measure-aware weighted-tensor path is algebraically forced to equal the legacy
`.mean()` path: `domain_integral_tensor`'s `mean(f)*area*thickness`, divided by
`ref_energy_absolute = ref_energy*(domain_area*thickness)`, reduces to exactly
`mean(f)/ref_energy` — the legacy path's own formula — with the `area*thickness` factor
cancelling out on both sides. Both paths' unweighted equivalence is already independently
covered by the existing, passing
`interior_energy_term_measure_aware_with_no_weights_matches_domain_integral_tensor_directly`/
`external_work_term_measure_aware_matches_boundary_integral_tensor_directly` tests. So for THIS
square/uniform/no-AMR cell, Mode A's real result IS Mode B's real result by construction — an
independent training run would be uninformative, not missing evidence. **This equivalence does
NOT extend to a non-square plate** (different `ds` per edge — see sub-issue #68) or an
AMR-nonuniform sampling regime (see sub-issue #67) — those are exactly where Mode B would need
its own independent run, and are correctly scoped to those sub-issues, not this one.

This matrix makes it structurally explicit (issue #63's own wording) that Mode A's real PASS
does not imply Mode C/D's status — Mode C is real, documented, historical FAILURE evidence, and
Mode D's real PASS is issue #64's own fix, independently earned, not inherited from Mode A/B.

Evidence before change: Phase 3 recorded Hybrid + legacy mean as operational for its narrow
no-hole configuration, and Variational + measure-aware as not operational. Source showed
Variational also had independently weighted `interior_energy` and `external_work` terms.

## PH4-02 — Live mathematical objective audit

Status: IMPLEMENTED

Hypothesis: live SAW-BRDR/base weighting changed the physical `U:W_ext` ratio.

Evidence before change: `UserDefinedProblem::loss_terms()` returned separate terms and
`step_physics_multi()` independently multiplied every term by its SAW/base weight. The base
weights were `LAM_INTERIOR_ENERGY=1.0` and `LAM_EXTERNAL_WORK=20.0`.

Mathematical derivation and source mapping: `PHASE_4_MATHEMATICAL_OBJECTIVE_AUDIT.md`.

Change made: none in this audit item; PH4-03 owns remediation.

Tests/runtime artifact/benchmark/regression: pending PH4-03 through PH4-06.

Known limitation: historical Phase 3 Variational evidence used the invalid independent ratio;
it cannot establish corrected-Variational behavior.

Mode artifact directories are present. A/B/C notes explicitly retain blocked historical or
unexecuted states; no missing runtime evidence is silently synthesized.

## PH4-03 — Create atomic physical U-W functional

Status: VERIFIED (issue #64 sub-issue #66)

Hypothesis: one `LossTerm` containing `U-W_ext` prevents adaptive/base weighting from changing
the physical coefficient ratio.

Change made: `PhysicalPotentialEnergyTerm` requests both live point sets and returns one tensor;
`FormulationSelection::Variational` activates it, while legacy Hybrid remains explicit and
unchanged for compatibility.

Tests: `cargo check -p pinn-solver --features ndarray-backend` passed. Focused production-path
tests passed: `physical_potential_is_one_live_atomic_u_minus_w_term` and all three
`variational_formulation_*` tests; the latter executes `step_physics_multi` with the live
Variational term set.

PH4-03 affine acceptance: `physical_potential_live_term_has_the_correct_affine_minimizer`
constructs actual interior/boundary `DomainForwardOutputs` in production point-set order and
proves the atomic term has its minimum and zero numerical derivative at `a=sigma0/E`.

Runtime artifact/benchmark (#66): the shipped `examples/problems/variational_no_hole_plate.toml`
run (post-#64 fix) recorded `U=6.646538 J`, `W_ext=13.28682 J`, `Pi=-6.640279 J` — matching
`Pi=U-W_ext` exactly (`shipped_example_final.log`) — and `runner::tests::
variational_no_hole_plate_trains_and_produces_real_benchmark_evidence` (`#[ignore]`d, real
~23min production-scale run) now asserts `benchmark.passed` and the persisted checkpoint's
`Pi=U-W_ext` consistency directly, replacing the pre-#64 `is_finite()`-only assertions. A fast
(2.46s) companion, `runner::tests::corrected_variational_checkpoint_persists_a_
reconstructable_objective_snapshot`, proves the same persistence mechanism independent of
convergence, per issue #65/#66's own "avoid a full run when a cheaper method suffices" policy.

## PH4-04 — Live integral unbiasedness

Status: VERIFIED (issue #64 sub-issue #65)

Evidence before change: PH3 proved helper and separate `InteriorEnergyTerm`/`ExternalWorkTerm`
branches, but did not exercise one atomic production Pi term. PH4-03's focused live-term test
now proves the measure-aware `PhysicalPotentialEnergyTerm` uses both real domain and boundary
integral primitives. `physical_potential_live_measure_aware_weights_remove_nonuniform_interior_bias`
executes the actual atomic term with a deliberately biased AMR compensation shape and verifies
its weighted result against the differentiable physical integral.

New evidence (#65): `ph4_04_interior_energy_integral_agrees_across_uniform_nonuniform_and_amr_like_sampling`
(`user_problem.rs`) exercises the real production `InteriorEnergyTerm` for a KNOWN, spatially
nonconstant strain field (`exx=a*x, eyy=b*y, exy=0` — the shear term deliberately zeroed to
remove any engineering-vs-tensor shear-convention ambiguity from the proof), with a hand-derived
closed-form analytical integral (`(2/3)*E/(1-nu^2)*(a^2+b^2)`, from integrating the known
`strain_energy_density` formula over `[-1,1]x[-1,1]`). Three independently-shaped point
distributions — a plain uniform grid, a 1D left/right density split (80/20), and a 2D
corner-refinement split mimicking real AMR behavior (60% of points in one quadrant, 25% of the
area) — each with correctly-derived compensation weights (`mean(weight)=1`) — all agree with the
analytical value within 2%, and with each other. A fast, deterministic unit test (no training
loop, `<0.1s`), not a 30+ minute run (issue #65's own verification-cost policy).

`ExternalWorkTerm`/`W_ext` is deliberately excluded from this proof: its boundary integral
already uses the EXACT known per-point arc-length `ds` (`boundary_integral_tensor`), not an
MC-style density estimator — `AdaptiveGrid` only ever refines the interior quadtree, so the
uniform/nonuniform/AMR distinction this item asks about is squarely an interior-integral
question. Real AMR-sweep-in-the-loop evidence (as opposed to this hand-constructed proof of the
underlying estimator's correctness) is sub-issue #67's job.

## PH4-05 — Physical versus optimization values

Status: VERIFIED (issue #64 sub-issue #66)

Evidence before change: `EnergyBalance` persisted internal energy and half prescribed work for
the linear energy-balance diagnostic, but no report exposed full `W_ext`, `Pi`, normalization,
active terms, or live SAW-BRDR weights. Its pre-PH4 probe also used model-derived traction,
which is not the prescribed-load work in the variational functional.

Mathematical derivation: linear energy balance uses `U = W_ext/2`; potential minimization uses
`Pi = U-W_ext`. The report therefore reconstructs full work as twice the diagnostic half-work,
not by changing the energy-balance convention.

Change made: `probe_energy_balance` now integrates prescribed `tbar dot u`. New serialized
`MathematicalObjectiveSnapshot` distinguishes physical/normalized `U`, full `W_ext`, and `Pi`;
records reference energy, geometry measures, estimator, active terms, base weights, constraints,
last live adaptive weights, and optimizer policy. Plate checkpoint and GUI export use the same
`build_plate_authoritative_report` producer.

Files changed: `user_problem.rs`, `provenance.rs`, `runner.rs`, downstream
`app-egui/src/stress_solver.rs`.

Tests: focused prescribed-work test and snapshot reconstruction/JSON round-trip pass. Solver
`cargo check` passes. Downstream `cargo check` and `cargo test` pass.

Integration evidence (2026-09-12): solver commit
`cacdcd86b8dc01a1f582bde87e63370d4521e0ce`, dirty; powershell_tool commit
`0500a6ad62c0923378769fbeb9fea4e8c1583919`, dirty only at required
`app-egui/src/stress_solver.rs` compatibility call-site. `app-egui/Cargo.lock` SHA-256:
`3aba2a2dd087c976d5817c4cfd50cb04651555bc6180317fda927a6a0e3f3cdf`.
Commands passed: `cargo check`, `cargo test`, and `cargo build --release` in
`../powerShell/powershell_tool/app-egui`.

Runtime artifact/benchmark/regression (#66): see PH4-03's own updated entry above — the same
shipped, checkpoint-producing corrected-Variational run and its `Pi=U-W_ext`-consistent
persisted snapshot serve as this item's evidence too.

## PH4-06 — Variational divergence root cause

Status: VERIFIED (issue #64) — root cause found and fixed; see the dedicated "PH4-06 root cause
found and fixed" sub-section below for the full writeup. The narrative immediately below this
line is the historical investigation record leading up to that fix and is kept as-is.

Status (historical, pre-fix): INVESTIGATING

Runtime evidence: `Debug_run/phase4/D_variational_atomic_pi/` records the controlled full-NN,
uniform, AMR-off, 15-step run after atomic Pi and rotational gauge changes. It is finite and
loss decreases, but L4 fails decisively (`load_transfer_ratio=0.1996`; all five hard metrics
fail). This rules out treating finite loss or a short decreasing trajectory as convergence.
Its recorded energy-balance value predates PH4-05's prescribed-traction correction and must not
be used as physical-Pi evidence. Next ladder comparison must change one variable only.

Post-PH4-05 rerun: same 15-step D command completed. Correct prescribed-work energy-balance
error is `1.7482`; L4 still fails all five hard metrics. Runtime evidence is diagnostic only.

New source finding: pure Variational was still receiving synthetic, fixed-weight
`constitutive_consistency` although it declared no direct-stress consumer. This hidden
constraint is now suppressed unless active terms explicitly consume direct stress. Focused
live regression asserts no `constitutive_consistency` gradient entry in pure Variational.

Additional source trace: `compute_domain_forwards` applies `IdentityAnsatz` (unit factors),
scales displacement by `u_ref`, scales direct stress by `px`, and converts normalized FD
derivatives with `FdConfig.sx/sy`; `PhysicalPotentialEnergyTerm` consumes the same first `n`
rows and shared point-set measures. No scaling or sample-order mismatch was found in this
static trace. Runtime optimization failure therefore remains unresolved rather than being
silently attributed to quadrature or sign error.

Controlled runtime controls: 1,000 steps with `fd_h=1e-2` and fixed unit
`physical_potential` weight reached normalized `Pi=-1.02345`, L4 health pass
(`energy_balance_error=1.61e-3`), but load transfer `0.8336` and stress/traction hard metrics
failed. Increasing only outer-boundary sampling to 1024 points reached normalized
`Pi=-1.02045`, L4 health pass, load transfer `0.8781`, and still failed hard metrics. A
3,000-step 512/256 run reached `Pi=-1.04625` and load transfer `0.6053`. These controls rule
out simple optimizer duration, SAW multiplier drift, and boundary sample count as sufficient
fixes; L4 remains blocked.

Capacity control (`hidden_dim=32`, `n_hidden=3`, otherwise identical to the 1024-boundary
control) reached normalized `Pi=-1.02487`, L4 health pass (`1.93e-3`), but load transfer
`0.8044` and stress/traction hard metrics failed. Larger capacity alone does not recover the
boundary field.

Follow-up tracking: GitHub sub-issue [#64](https://github.com/Teased-oChroid-orrA/NetworkNeuralNetworkStressSolver/issues/64)
tracks field-recovery failure and candidate general neural residual/linear representation work.

Issue #64 architecture update: `ElasticityNet` now has a zero-initialized, trainable
`[x_norm, y_norm, z] -> [u, v]` coordinate-residual projection. Production `fwd` and
`fwd_masked` retain raw normalized coordinates while the MLP continues to receive its optional
Fourier embedding. The residual is added only to displacement columns; direct stress columns
remain MLP-produced. This is geometry-independent and leaves the embedded-input `forward` API
unchanged. The residual weight and bias participate in ordinary optimizer gradient collection;
width/depth growth and pruning retain them unchanged. New checkpoints persist this branch; a
legacy-record fallback loads old checkpoints with an exactly zero residual. Focused checkpoint
round-trip and legacy-load tests pass, and `cargo check -p pinn-app --features ndarray-backend`
passes. The WGPU-only network unit suite cannot execute on this host because no Metal adapter is
available. No L4 result is claimed: controlled runtime ladder and field acceptance remain open.

Issue #64 controlled H result: coordinate residual was tested for 1,000 steps against F's fixed
seed, corrected objective, `fd_h=1e-2`, 512 interior, 256 boundary, AMR-off configuration.
It improved health (`energy_balance_error=2.7854e-4`) but failed every hard L4 metric:
`sigma_xx_relative_error=0.1525`, `sigma_yy_over_ref=0.0410`,
`sigma_xy_over_ref=0.0239`, `traction_rms_over_ref=0.1231`, and load transfer `0.8049`.
F MLP-only measured load transfer `0.8838` and sigma-x error `0.1196`; H is not a winning
configuration and is not extended. Evidence: `Debug_run/phase4/H_coordinate_residual/`.

Production change: `step_physics_multi` now fixes the atomic Variational term multiplier at
unit scale; constraint terms remain adaptive. Focused objective tests and the full user-problem
focused suite pass. Runtime controls show this removes multiplier drift but does not satisfy
hard L4, so item remains INVESTIGATING.

Controlled rerun changing only that condition is persisted in D notes: total loss drops from
`14.08062` to `0.361697` at step zero, proving the hidden constraint dominated optimizer loss;
however corrected physical metrics remain effectively unchanged after 15 steps
(`load_transfer=0.1990`, energy balance `1.7463`, L4 five-of-five fail). Root cause remains
unresolved; no optimizer tuning has been applied.

E ladder evidence (same corrected path, only budget 15 to 200): Pi falls to `-1.898899e-1 J`,
but load transfer falls to `0.0699` and all hard L4 metrics fail. Final physical-Pi gradient
norm is `2.100251`; translation/rotation gauge norms are `1.46e-8`/`7.49e-9`, ruling out gauge
dominance in this run. Full stdout, config, and notes are persisted under
`Debug_run/phase4/E_variational_stabilized/`. Do not interpret its loose L4 health pass as a
hard benchmark pass.

The same E configuration was rerun after the dimensional correction to the translation gauge
only. At step 199, loss is `1.771334e-1`, physical-Pi gradient `2.181287`, translation-gauge
gradient `6.169782e-2`, and rotation-gauge gradient `2.548643e-9`; neither gauge dominates.
Physical results remain invalid: `U=2.799427e-1 J`, `W_ext=-7.325458e-1 J`,
`Pi=1.012488e0 J`, load transfer `0.0780`, health fails, and all five L4 hard metrics fail.
This is persisted in `E_variational_stabilized/solver.log` and `RUN_NOTES.md`. Next required
controlled ladder stage is restricted affine representation, not optimizer tuning.

Restricted-neural ladder stage is now executing from
`D_variational_atomic_pi/restricted_affine_neural.toml`: existing MLP only, two hidden units
and one hidden layer, with every other E setting identical. It is intentionally not an
analytical substitution or a benchmark-specific ansatz. It completed with
`Pi=-3.575255e-1 J`, physical gradient `6.821348e-1`, translation gradient `3.410605e-1`,
load transfer `0.0960`, L4 health failure, and five-of-five hard L4 failure. Persisted notes
and full output are `D_variational_atomic_pi/RESTRICTED_AFFINE_NEURAL_NOTES.md` and
`restricted_affine_neural.log`. Capacity alone is not root cause; optimizer contract is now
the next controlled candidate, without changing physical objective or benchmarks.

### PH4-06 root cause found and fixed (issue #64): a "resample every step" that never resampled

Status: VERIFIED

Root cause: `UserSamplingStrategy::sample_interior`/`sample_boundary` (`user_problem.rs`) are
called fresh every training step (`user_runner.rs`'s `for step in 0..max_steps` loop), with
clear intent to resample collocation points every step. For any no-hole geometry (every no-hole
example, including `variational_no_hole_plate.toml`) this was a no-op: `sample_interior`'s
stratified grid was a pure function of `(geometry, n)` with a fixed `+0.5` cell-center offset,
`contains_for_collocation` always returns true with zero holes so the RNG-based rejection
fallback (itself reseeded from the same constant `SEED_INTERIOR` every call) was never reached,
and `sample_boundary` had no randomness at all. So the network trained against one literal,
unchanging finite point cloud for the entire run. `InteriorEnergyTerm`/`PhysicalPotentialEnergyTerm`'s
`U` and `ExternalWorkTerm`'s `W_ext` are both plain `mean(f(x_i))` Monte-Carlo estimators
(`measure_integral::domain_integral_tensor`/`boundary_integral_tensor`) — unbiased only if the
`x_i` vary across the optimization trajectory. With a static node set, the optimizer could (and
did) sculpt energy density artificially low AT those frozen nodes while the field diverged
between them — the textbook Deep Ritz Method "quadrature-node overfitting" failure, and exactly
what this manifest's own diagnosis already named ("sampled-functional underintegration / neural
between-sample exploitation"): sampled `Pi` reaching `-1.02` to `-1.05`, below the true
continuum affine minimum of exactly `-1`, which is impossible for the real convex functional but
trivial for a biased fixed-sample estimate of it.

Fix: `sample_interior`/`sample_boundary` now draw genuinely different points on every call via
jittered stratified sampling, seeded from a per-instance atomic call counter mixed into the RNG
seed (`interior_calls`/`boundary_calls`, `AtomicU64` — `DomainSamplingStrategy: Send + Sync`
rules out `Cell`). Same stratum/coverage structure, same `ds` quadrature-weight validity, fully
reproducible run-to-run (same base seed constant -> same full sequence of per-call point sets).
Does not touch `Pi`, SAW-BRDR weighting, any threshold, or `pinn_core::sampling`/
`PinLugSamplingStrategy` (Kirsch's frozen path and pin-lug's separately-tuned path, out of
scope). New tests `sample_interior_resamples_different_points_across_consecutive_calls`/
`sample_boundary_resamples_different_points_across_consecutive_calls` prove real resampling;
every pre-existing containment/margin/edge-placement property test still passes unmodified.

Runtime evidence: at the ORIGINAL 2048/512/2000-step config, the fix alone took normalized `Pi`
to `-0.9998945` and `load_transfer_ratio` from historical `0.60-0.88` to `0.9699`, and cut
`sigma_xx_relative_error` from `0.12-0.15` to `0.0183` — real, large improvement, but 3 of 5
hard L4 metrics still narrowly missed threshold. Raising quadrature resolution (more points
shrink the jittered-stratified estimator's residual variance) and step budget closed the
remaining gap: `n_interior=4096`, `n_boundary=2048`, `max_steps=3000` (now
`variational_no_hole_plate.toml`'s shipped config) reached `normalized_Pi=-1.000017` and
**PASSED all five P2-14 hard thresholds**
(`sigma_xx_relative_error=0.0010`, `sigma_yy_over_ref=0.0007`, `sigma_xy_over_ref=0.0003`,
`traction_rms_over_ref=0.0011`, `load_transfer_ratio=0.9991`), confirmed by two independent
real `cargo run -p pinn-app --release --features ndarray-backend -- --headless --problem-spec
examples/problems/variational_no_hole_plate.toml` runs
(`Debug_run/phase4/issue64_resample_fix/higher_res.log`,
`Debug_run/phase4/issue64_resample_fix/shipped_example_final.log`).

Independent field validation (issue #64/PH4-15's own blocking condition — required a verified
corrected no-hole L4 companion, which now exists): `validate_no_hole_fields` is now wired into
`run_headless_user_problem`'s no-hole output, evaluated on a separate 96x96 grid (not the
training collocation points) using the already-computed vis-diagnostic fields. Both real runs
above printed `P2-14 independent field validation PASSED`
(`sigma_xx_err=0.0010 sigma_yy/ref=0.0007 sigma_xy/ref=0.0003`, `rigid_translation=1.69e-8`,
`rigid_rotation=4.90e-6` — no residual rigid-body mode hiding in the published fields either).

Durable regression proof: `user_problem::tests::
issue_64_resampling_fix_passes_l4_and_independent_field_validation` (`#[ignore]`d, real
training loop at the shipped config, matching `toy_beam`'s own expensive-training-loop-test
precedent) trains a real model and asserts both `run_no_hole_benchmark(...).passed` and the
independent field check pass — `cargo test -p pinn-solver --release --features ndarray-backend
-- --ignored user_problem::tests::issue_64_resampling_fix_passes_l4_and_independent_field_validation`,
confirmed passing (1386.85s).

Full workspace suite after this fix: 457 passed, 0 failed (2 pre-existing, unrelated stale test
literals in `network.rs` — `awake_mask_matches_awake_weight_ids_classification`,
`coordinate_skip_represents_affine_displacement_and_leaves_stress_mlp_only` — corrected in the
same pass; both predate this session's sampling work, confirmed via `git stash`, and were simply
never updated when the coordinate-skip feature was added).

Remaining PH4 items this unblocks: PH4-15 (independent field validation) is now real evidence,
not blocked. PH4-09/13/14/16/17 (AMR, hole-boundary Kt, L5, non-square, multi-hole) remain
correctly BLOCKED — each needs its own dedicated evidence beyond a passing no-hole companion,
per their own stated blocking conditions; this fix does not itself unblock them.

### Follow-up (sub-issue #66): a second, independent instance of the same bug class, plus a real AMR finding

While producing #66's checkpoint-persistence evidence, the shipped config's GUI-streaming path
(`runner::run_training_user_problem`, distinct from the headless CLI path `user_runner::
run_headless_user_problem` used for all evidence above) was found to have its OWN separate
regression: `run_user_problem_training_from` cached `sample_boundary`/`named_point_sets`'s
output once before the training loop instead of resampling every step — a real, correct-at-the-
time perf optimization from before #64 (the caller's own doc comment said so, and was true
then), never updated when #64 made those functions genuinely vary per call. Fixed by moving the
resampling call inside the per-step loop (interior resampling's interaction with AMR is left to
sub-issue #67, not conflated with this fix).

Even after that fix, three real runs still landed borderline (exactly one of `sigma_xx`/
`traction_rms_over_ref`/`load_transfer_ratio` narrowly missing threshold each time, never more
than one, never by much). Re-evaluating the SAME trained checkpoints at up to 16x more boundary
evaluation points (2048 → 32768) changed the reported metrics by nothing measurable — this
definitively rules out evaluation-time quadrature/sampling-count noise as the cause. Disabling
AMR entirely (`spec.training.amr_enabled = false`) on an otherwise-identical run reproduced
headless's comfortable pass exactly: `sigma_xx_relative_error=0.00108`, `sigma_yy_over_
ref=0.00077`, `sigma_xy_over_ref=0.00044`, `traction_rms_over_ref=0.00121`, `load_transfer_
ratio=1.00009`, all PASS.

Root cause: AMR is a GUI-streaming-only feature (headless never implements it at all) and
defaults to `amr_enabled=true`; issue #63's own PH4-09 policy already states "AMR = OFF, uniform
sampling = ON until the baseline is mathematically correct" — AMR itself has no dedicated
evidence yet (sub-issue #67 exists for exactly this). The shipped `variational_no_hole_plate.
toml` was silently violating that already-written policy by never overriding the default.
`amr_enabled = false` added to the shipped file makes it actually honor the policy — not a new
workaround, a correction to match what the epic already said.

## PH4-07 — Gauge/nullspace compatibility

Status: IMPLEMENTED

Evidence before change: pure-Neumann user plates register `TranslationGaugeTerm`, which removes
two translations only. Source inspection found no rotational gauge, nullspace projection, or
point-constraint alternative.

Change made: `RotationGaugeTerm` uses outer-edge circulation to constrain mean infinitesimal
rotation. It is only added to corrected Variational pure-Neumann runs, preserving frozen legacy
Hybrid/Strong trajectories. `rotation_gauge_removes_only_rigid_rotation_not_affine_symmetric_
strain` proves rigid rotation is penalized and affine extension is not.

Runtime/benchmark proof and constraint-gradient dominance evidence remain required before
VERIFIED.

Hole-side runtime evidence (issue #63 sub-issue #69): the three real hole smoke runs above show
the gauge terms behaving exactly per their documented design intent — `variational_single_hole_
smoke.toml` (single `Free` hole, pure-Neumann): `rotation_gauge` registered and well-behaved
(`raw=1.76e-9`, `lambda=24.7`, `grad_norm=1.04e-8` at step 799 — small, present, not dominating).
`variational_notched_smoke.toml`/`variational_triple_hole_smoke.toml` (both include a `Fixed`
hole): `translation_gauge` correctly NOT registered (prints as the diagnostic's own "absent"
fallback) since the `Fixed` hole already anchors the geometry — `hole_fixed`'s own term is
active instead (`raw≈1.6e-10`-`2.7e-9`, essentially satisfied). No constraint term dominates
`physical_potential`'s own gradient norm in any of the three runs. Not yet VERIFIED (that still
needs the full runtime/benchmark proof at real convergence, not a short smoke budget), but real,
positive, hole-topology-diverse evidence toward it.

## PH4-08 — Variational optimizer contract

Status: VERIFIED (issue #63 sub-issue #71)

Superseding finding: this item's own INVESTIGATING trail (peak-LR probe, uniform-resolution
probe, incomplete `F_no_hole_final` run) chased optimizer instability as root cause for the
non-stationary Pi / failing load-transfer pattern. Root cause was NOT the optimizer — it was
`UserSamplingStrategy::sample_interior`/`sample_boundary` returning a frozen, non-varying point
cloud every step (issue #64), which biases the Monte-Carlo Pi estimator regardless of LR or
optimizer choice. No optimizer-side change was made.

Post-fix real evidence the same optimizer contract (plain constant/scheduled AdamW via
`step_physics_multi`, unchanged this whole investigation) is stable once fed a genuinely
resampled estimator: PH4-16's non-square run passes all five P2-14 hard metrics
(`load_transfer_ratio=0.9971`); PH4-09's AMR-off A/B runs pass reliably
(`sigma_xx≈0.0009`, `load_transfer≈1.000`); issue #64/#66's own real-run verification. No run
this session (across #64/#66/#67/#68/#69/#70) showed divergence, NaN/Inf, or oscillatory
non-convergence attributable to the optimizer itself.

### Original INVESTIGATING trail (kept for record)

Controlled probe changes only peak learning rate from `1e-3` to `3e-3` at fixed full NN,
atomic Pi, gauges, uniform sampling, AMR-off, seed, and 200-step budget. It decreases sampled
normalized Pi from `+3.616969e-1` to `-1.767713e-1`; translation-gauge raw value falls from
`1.623270e-1` to `4.686160e-6`, and its final gradient (`6.311712e-3`) is far below physical
Pi (`2.010310`). Yet load transfer is `0.0868` and all L4 metrics fail.

Probe normalized Pi is `-1.804754e-1`; exact affine continuum minimum is `-1` under common
reference scaling, so this is non-stationary rather than a discrete lower-than-continuum
minimum. Next test changes uniform collocation resolution only. Full evidence:
`E_variational_stabilized/OPTIMIZER_PEAK_LR_3E-3_NOTES.md`.

Fourfold uniform-resolution control (`512/256` points, all other peak-LR settings fixed) ends
at normalized Pi `-1.538412e-1`, physical gradient `2.127576`, translation gradient
`3.486059e-3`, and load transfer `0.0836`; L4 health and all hard metrics still fail. Thus low
quadrature count alone is not root cause. AMR remains off. Evidence:
`E_variational_stabilized/UNIFORM_RESOLUTION_512_NOTES.md`.

`F_no_hole_final` now runs justified convergence gate: 1,000 steps at same corrected 512/256
uniform configuration and `3e-3` peak LR. One sequential solver process. Final log decides
whether Pi, physical gradients, gauges, energy balance, field errors, and load transfer
stabilize; no acceptance status assigned before evidence.

Run ended before final ledger at step 100; `F_no_hole_final/solver.log` is retained and marked
incomplete in `F_no_hole_final/RUN_NOTES.md`. Partial trend data cannot establish convergence.

## PH4-09 — AMR evidence

Status: VERIFIED-DISABLED for the canonical no-hole benchmark specifically (real quality
evidence below still supports this, independent of the crash fix); the CRASH itself is now
FIXED (issue #74, verified — see the new section below). These are two separate findings this
item made and they resolve on two separate timelines: the quality regression is a real,
standing reason to keep AMR off for the no-hole case; the crash was a separate implementation
bug that no longer blocks using AMR at all, including for hole geometries where it was always
the intended fix (see PH4-14/issue #70's re-attempt with AMR now enabled).

### A real interior-weight staleness bug, found and fixed

Auditing AMR's own collocation sampling for the same defect class #64 found (this item's own
charter): `AdaptiveGrid::sample_points`/`sample_points_with_density` are themselves fully
deterministic (leaf-cell centers, no jitter) — fine, since they're only used on the ONE step a
sweep fires. The real bug was in `UserDefinedProblem::set_interior_weights`'s lifecycle:
`runner::run_user_problem_training_from` set AMR's density-compensation weights on a sweep step
and never cleared them — before issue #73, `data.int_norm` also stayed frozen at that sweep's
own points between sweeps, so the (stale) weights and (stale) points at least stayed matched;
after #73 made `data.int_norm` genuinely resample every step (correct, fixing the #64-class
staleness bug for interior points), the OLD sweep's compensation weights kept being applied to
brand-new, unrelated freshly-resampled points for up to `amr_interval` (1000) steps — silently
reintroducing bias/noise into the interior energy estimate for most of a typical run, exactly
the kind of thing that would show up as "worse than AMR-off" without an obvious cause. Fixed:
`problem.set_interior_weights(None)` now runs at the top of every step by default; the AMR
sweep block re-sets it to `Some(...)` only for the exact step a sweep fires — restoring
`set_interior_weights`'s own already-documented contract ("None before the first sweep - plain
uniform-random sampling is already unbiased, no compensation needed") for every non-sweep step,
not just before the first sweep. Regression-tested by `user_problem::tests::
set_interior_weights_none_clears_a_previously_set_weighting_back_to_unweighted`.

### The real A/B evidence

Three independent real runs of the shipped no-hole Variational config, AMR on (post-boundary-fix,
pre- and post- the interior-weight fix above): consistently borderline, exactly one of
`sigma_xx`/`traction_rms_over_ref`/`load_transfer_ratio` narrowly missing threshold each time.
Two independent real runs with AMR off (otherwise identical): comfortable, reliable passes
matching headless's own quality exactly (`sigma_xx≈0.0009`, `load_transfer≈1.000`). Full detail
in this file's own PH4-06 follow-up section (sub-issue #66).

### A second, more serious finding: AMR + Variational can crash outright

`runner::tests::ph4_09_controlled_comparison_fixed_sampling_vs_amr_corrected_variational_same_
budget` (PH3-12's own controlled comparison pattern, applied to Variational) reproducibly
crashes — NOT a quality regression, a hard panic — inside `step_physics_multi`'s `.backward()`
call, somewhere between step 1200 and step 2200 of a 2200-step AMR-enabled run, with
burn-autodiff's own internal panic: `"Node should have a step registered, did you forget to
call Tensor::register_grad on the tensor where you need gradients?"`. Confirmed via `git stash`/
manual disable that this predates and is fully independent of every #64/#66/#67/#73 change this
session (reproduces identically with the new interior-weights fix both present and reverted).
Confirmed via PH3-12's own "Status: VERIFIED" history that the IDENTICAL 2200-step AMR budget
runs cleanly for the legacy Hybrid formulation — this crash is specific to combining AMR with
the Variational formulation's term structure, not a general AMR-after-N-steps framework issue.

Root-cause hypothesis (not yet fully verified — the actual fix is out of this item's scope, see
below): `training_core::probe_interior_energy_residuals` (AMR's before/after-sweep residual
probe) runs its forward pass through the SAME live `Autodiff<BInner>`-backed model the main
training step uses, purely to read `.into_data()` residual magnitudes — it never calls
`.backward()` on its own probe graph, nor does anything explicitly detach it. Over repeated
sweep checks (2 per 2200-step run: steps 200 and 1200, each doing 1-2 probe forward passes),
these orphaned autodiff graph nodes may accumulate in burn-autodiff's internal node registry
without ever being released, eventually causing an internal bookkeeping inconsistency that
surfaces later as this panic — a plausible mechanism given this codebase's own already-
documented, analogous burn-autodiff limitation (`probe_term_gradients`'s doc comment: "burn's
autodiff does not support an independent `.backward()` call on a tensor that shares upstream
graph nodes with another tensor already ... backpropagated separately"). Variational-specific
because `PhysicalPotentialEnergyTerm` spans two point sets (`interior`+`outer_boundary`) in one
term, unlike Hybrid's separate single-point-set terms — plausibly interacting differently with
`compute_domain_forwards`'s graph construction when alternating with AMR's interior-only probe
term, though this specific mechanism is NOT independently confirmed (would require tracing
`compute_domain_forwards`'s internal forward-pass/graph-node caching in detail).

### Issue #74 fix — VERIFIED, root cause confirmed (not just hypothesized)

The hypothesis above is confirmed correct, via direct comparison rather than inference alone:
Kirsch's own frozen AMR sweep (`headless.rs`) already converts to `BInner` before its residual
forward pass (`let model_val: ElasticityNet<BInner> = model.valid();`) — the exact mechanism
hypothesized as the fix — and has run cleanly for the identical 2200-step budget (PH3-12,
Hybrid formulation) the whole time. `training_core::probe_interior_energy_residuals` (the
shared function pin-lug and the Variational/user-plate path use) did NOT do this — it ran the
probe forward pass through the live `Autodiff<BInner>`-backed model directly, building and then
abandoning an autodiff graph every sweep check with no `.backward()` ever called on it.

**Fix**: `training_core::compute_domain_forwards` (private) is now generic over
`Bk: Backend<Device = BDevice>` (was hardcoded to `B = Autodiff<BInner>`) — the private
`Computed` struct, `ShiftedStress<Bk>`/`HessianData<Bk>` (already generic type aliases, unused
before), and every internal helper call (`norm_pts_to_tensor`, `assemble_stencil`,
`fwd_masked`, `compute_strains`, `compute_hessian`) now take the turbofish `Bk` instead of a
hardcoded `B`. This is a zero-behavior-change genericization for the real training step: every
existing call site (`step_physics_multi`, `compute_gradient_conflict_multi`,
`compute_loss_for_lbfgs_multi`) passes live `&[&ElasticityNet<B>]` models and Rust infers
`Bk = B` automatically — none of those call sites needed to change at all, confirmed by a clean
`cargo build --workspace` with zero other edits. `probe_interior_energy_residuals` is the ONLY
call site that changed behavior: it now converts its `B`-typed models to `BInner` via
`.valid()` (mirroring Kirsch's own established pattern exactly) and calls
`compute_domain_forwards::<BInner>` explicitly, so the probe never enters the autodiff graph at
all — there is nothing left to orphan. `LossTerm`/`DomainForwardOutputs` (both used for the real
gradient-requiring step) were NOT touched — they stay hardcoded to `B`, which is correct since
they genuinely need gradients; only the probe-only, always-`unreachable!()`-bodied
`InteriorProbeTerm` was ever driving this through the wrong backend.

**Verification** (not merely "compiles"): `runner::tests::ph4_09_controlled_comparison_fixed_
sampling_vs_amr_corrected_variational_same_budget` — the exact test that used to reproducibly
crash — now runs the full 2200 steps to completion with no panic (483.9s wall-clock, release,
NdArray). Real honest numbers reported (test does not force a pass, per issue #63's own rule):
fixed-sampling `sigma_xx_relative_error=0.01093`, AMR-enabled `sigma_xx_relative_error=0.01125`
— both narrowly miss the 0.01 threshold at this shorter 2200-step budget, consistent with (not
contradicting) the quality-regression evidence above; this test's own job was never to prove
AMR wins, only that the comparison itself completes safely, which it now does. Full workspace
`cargo build` (default Wgpu + `ndarray-backend`) clean. Full fast (non-`#[ignore]`d) suite:
461/462 passed — the one failure (`runner::tests::gui_streaming_step_zero_matches_independent_
shared_function_computation`) is the ALREADY-documented contention-flaky test from earlier this
epic (see CLAUDE.md's Phase 4 close-out section), confirmed non-regressive by an isolated
single-threaded re-run (`... --test-threads=1`), which passed cleanly. No pin-lug or Kirsch
regression — both share `compute_domain_forwards`/`probe_interior_energy_residuals` and neither
showed any new failure in the full-suite run.

**Conclusion**: the crash is fixed and verified, not merely patched-and-hoped. AMR is now safe
to use with Variational for real, including on hole geometries — see PH4-14/issue #70 for the
re-attempt this unblocks. AMR still stays OFF for the canonical no-hole benchmark specifically,
because that decision was never about the crash — it was about the independently-measured
quality regression (three real AMR-on runs, consistently borderline-worse than AMR-off), which
this fix does not address and was never meant to.

## PH4-10 — Formulation-aware convergence

Status: VERIFIED (issue #63 sub-issue #71)

Change made: headless Variational runs now emit persisted cadence records for normalized Pi,
whole physical-block SAW weight, and translation-gauge raw value, in addition to existing final
physical/constraint gradient ledger, U/W/Pi, energy balance, field metrics, and load transfer.
The optimizer and 512-point controls exercise this live path in their solver logs.

Superseded limitation: this item previously read "no run meets all convergence conditions."
That was true only before the #64 sampling-estimator fix. Real runs now DO meet all convergence
conditions on this same cadence-record machinery: issue #64's own no-hole verification, #66's
GUI-streaming parity run, and #68's non-square run (`normalized_Pi=-1.000012`, all five P2-14
hard thresholds PASS) all emit and satisfy these records end-to-end. No changes to the cadence-
record code itself were needed to reach this status — the machinery was already correct; it was
starved of a correct estimator to converge against.

## PH4-11 — DifferentialOperator production integration

Status: IMPLEMENTED

Evidence before change: declared FD-only production policy existed, but generic live forward
path called `fd_stencil::compute_strains` directly. Policy was descriptive, not a choke point.

Change made: `differential_operator::PRODUCTION_POLICY` declares FD as sole
weight-autodiff-safe backend. `production_strain` now drives
`training_core::compute_domain_forwards`, plus user diagnostics, legacy runner/headless, and
parametric production paths. AD stays an explicit
first-derivative diagnostic because Burn 0.21 returns detached input-gradient tensors and does
not support nested differentiation; Hessians remain FD-only.

Tests: six focused differential-operator tests pass under NdArray. Direct FD calls left in
manufactured fields and unit-test oracles are intentionally non-production.

Code-audit confirmation (issue #63 sub-issue #71): grepped every non-test call site of
`compute_strains`/strain computation in `user_problem.rs`/`training_core.rs` — all eight+
production sites (`evaluate_user_vis_grid`, `probe_reaction_force`, both AMR residual probes,
the main `compute_domain_forwards` path, etc.) import `production_strain as compute_strains`;
none bypass it with a direct `fd_stencil::compute_strains` call. `PRODUCTION_POLICY` is a real
choke point, not merely descriptive, confirmed by code reading, not assumption.

Status: VERIFIED (issue #63 sub-issue #71, closed out) — real runtime proof now exists for every
formulation this codebase claims to support.

`user_problem::tests::issue_71_real_strong_and_hybrid_formulation_no_hole_runtime_evidence`
(`#[ignore]`d, real training, release/NdArray, identical no-hole config to Variational's own
verified baseline — `half_w=half_h=0.10`, Al7075-T6, `px=6.9e7`, `hidden_dim=64`/`n_hidden=8`,
3000 steps, `n_interior=n_boundary=4096`, AMR off):

- **`Strong` (`equilibrium`+`outer_traction`): PASSES cleanly** —
  `sigma_xx_relative_error=0.00223`, `sigma_yy_over_ref=0.00283`, `sigma_xy_over_ref=0.00064`,
  `traction_rms_over_ref=0.00264`, `load_transfer_ratio=1.00227`, `passed=true`. Genuine new
  evidence — the strong-form PDE-residual path (`EquilibriumTerm`, needs a real Hessian forward
  pass, materially more expensive per step than Variational's `PhysicalPotentialEnergyTerm`)
  converges correctly on this baseline.
- **`Hybrid` (`interior_energy`+`equilibrium`+`outer_traction`+`external_work`, the canonical
  `default_formulation()` 4-term set): FAILS** — `sigma_xx_relative_error=0.619`,
  `load_transfer_ratio=1.499`, all five hard thresholds fail. Reported exactly as measured, no
  correction applied. Not a new mystery — consistent with this file's own pre-existing note that
  "the legacy Hybrid L4 artifact... fails current hard thresholds"; this is now a *fresh*,
  *current-codebase* confirmation of that same known limitation, not a stale claim.

`Weak` still has no user-defined-problem implementation at all (unchanged, matches
`FORMULATION_SUPPORT_MATRIX.md`'s own note). PH4-11 is now VERIFIED — every wired formulation
has real runtime evidence, whether it passes (`Variational`, `Strong`) or honestly doesn't
(`Hybrid`).

## PH4-12 — Executable FieldKind

Status: IMPLEMENTED

Evidence before change: `FieldKind` supplied dependency-chain reporting only. Consumers could
slice raw output ad hoc, so constitutive field was not structurally distinct from auxiliary
direct stress.

Change made: `field_graph::resolve_field` resolves typed network output, displacement, strain,
constitutive stress, or direct mDEM stress and rejects missing prerequisites. Live atomic
potential requests displacement through resolver; free-hole traction requests direct stress.
Negative tests prove two-channel network cannot resolve direct stress and constitutive request
does not silently return direct stress.

Source correction: `evaluate_user_vis_grid` was classified and documented as constitutive but
published direct mDEM stress in engineering fields. It now publishes constitutive stress;
direct-minus-constitutive remains only explicit residual diagnostic. This removes a real
source-policy contradiction, not a benchmark correction.

Tests: 11 focused FieldKind tests pass under NdArray.

Status: VERIFIED (issue #63 sub-issue #71). Both conditions this item was waiting on are now
met. Migration audit (code-read, this session): grepped every non-test `resolve_field`/
`FieldKind::` production call site in `user_problem.rs` — exactly two
(`evaluate_user_vis_grid`'s `Displacement` resolve, hole-BC's `DirectStress` resolve), both
already migrated; no stray consumer bypasses the resolver. Real hole runtime evidence: issue
#69's three real hole smoke runs (single/2-hole-mixed-BC/3-hole-asymmetric, all headless CLI,
`Debug_run/phase4/issue69_topology/`) exercise `FieldKind`-resolved stress source live
end-to-end with no resolution failures (see PH4-17).

Downstream compatibility after public resolver addition (2026-09-13): in
`../powerShell/powershell_tool/app-egui`, `cargo check` and `cargo test` passed; release binary
`target/release/app-egui` was rebuilt at `2026-09-13 05:24:14`. Final SHA/dirty/Cargo.lock
evidence remains consolidated under PH4-21.

## PH4-13 — Hole-boundary stress source

Status: VERIFIED (issue #63 sub-issue #69)

Source policy is implemented: boundary-limit Kt uses derived constitutive stress at an explicit
radial offset (`probe_hole_boundary_profile_derived`); direct mDEM stress remains a separate
hole-BC diagnostic. The no-hole L4 companion this was blocked on now exists (#64/#66/#67).

Audit (this item's own charter — "no silent mixing of direct and constitutive stresses"):
`training_core::stress_source_report`/`stress_source_report_from_terms` already generalize the
manual per-term audit into a structural, always-available answer (iterates `problem.loss_terms()`,
filters to terms declaring a `StressSource`) — this machinery already existed and is already
tested (`stress_source_report_matches_the_kt_investigation_docs_written_conclusion`, part of the
already-green 460+ suite). No direct-vs-derived mixing found for any active term across the
three real hole runs below. Kt itself is NOT promoted to an accepted result here — that remains
L5's job (sub-issue #70); this item is specifically about the stress-SOURCE policy being correct
and audited, which it is.

## PH4-14 — Real L5

Status: REAL EVIDENCE COLLECTED, DOES NOT PASS (issue #63 sub-issue #70)

Blocking condition resolved: a verified corrected-Variational no-hole L4 companion now exists
(#64/#66/#67). `user_problem::tests::issue_70_real_l5_single_hole_kt_against_verified_no_hole_
companion` (`#[ignore]`d, real dual in-process training run, release/NdArray, 3000 steps each,
n_interior=n_boundary=4096, hidden_dim=64/n_hidden=8, Al7075-T6, px=6.9e7, single Free hole
radius=0.005 in half_w=half_h=0.10 plate — ratio=0.05, `InfiniteApprox`-eligible) ran to
completion (3550s wall-clock):

- No-hole companion: PASSES cleanly (`sigma_xx_relative_error=0.00084`, `sigma_yy_over_ref=
  0.00075`, `sigma_xy_over_ref=0.00014`, `traction_rms_over_ref=0.00100`,
  `load_transfer_ratio=0.99943`, `passed=true`) — required "verified companion" gate satisfied.
- Hole benchmark: `kt=1.0076`, `reference_kind=InfiniteApprox`,
  `relative_error_vs_infinite_theory=0.664` (66.4%), `passed=false`,
  `failures=["kt_vs_infinite_theory"]`. The test's own assertions (finite/positive Kt, correct
  `InfiniteApprox` classification, `relative_error_vs_infinite_theory.is_some()`) all pass — the
  test deliberately does NOT assert `kt≈3.0`, per issue #63's own "no benchmark-specific hacks to
  force a pass" rule. Reported here exactly as measured: **L5 acceptance does not pass.**

### Root-cause diagnosis (cheap, no new training run)

Before committing to another ~1hr run, checked the standing hypothesis — near-hole collocation
starvation — with a zero-cost sampling-only diagnostic (no network, no training,
`UserSamplingStrategy::sample_interior` called directly against the same L5 hole geometry, 50
calls × 4096 points, `cargo test` debug profile, 0.02s): only **0.55%** of interior points
(≈22.5 of 4096 per call) fall within 2 hole-radii of the hole boundary; only 4.7% fall within 5
hole-radii. This confirms the mechanism: uniform Monte-Carlo sampling starves the sharp
near-boundary stress-concentration region of training signal for a small hole
(ratio=0.05), independent of training duration — a structural sampling-resolution limit, not a
code bug (the sampler itself is correct: containment, FD-safe margin, and per-step resampling
are all already verified — see #64/#69's own tests). More steps alone would help only slowly;
the standard remedy is adaptive refinement biasing collocation density toward the hole boundary
— exactly what `pinn_core::amr::AdaptiveGrid` already does.

**Real dependency identified at the time**: robust small-hole Kt convergence was believed to be
blocked on **issue #74** (the AMR+Variational autodiff crash) — AMR being the mechanism that
would fix this sampling gap, but disabled for Variational because it crashed past ~1200-2200
steps.

### Re-attempt after #74's fix — AMR does NOT meaningfully improve Kt (honest negative result)

Issue #74 landed and was independently verified (see PH4-09). This unblocked a real re-attempt:
`user_problem::tests::issue_70_real_l5_with_amr_enabled_after_issue_74_fix` — identical
configuration to the test above, `amr_enabled: true` for the hole training, inline AMR sweep
logic mirroring `runner::run_user_problem_training_from`'s own block exactly (3 sweeps fire at
steps 200/1200/2200, each genuinely re-densifying the point set: `4096→706→1381→2281` points as
training progresses and the residual signal sharpens).

**Result: `kt=1.0088` with AMR vs `kt=1.0076` without — a 0.13% relative change, i.e. no
meaningful improvement** (`relative_error_vs_infinite_theory=0.6637` with AMR vs `0.6641`
without — both ≈66% error against the theoretical `3.0`). AMR is confirmed functionally correct
and doing real work — a companion zero-cost fixture (`adaptive_grid_density_near_l5_hole_rises_
above_uniform_baseline_after_synthetic_residual_sweep`, no training, proves `AdaptiveGrid::adapt`
genuinely raises near-hole point density by >3x once its residual signal indicates a concentration
there) confirms the refinement mechanism itself works — it just isn't converging Kt in this
configuration.

**Why AMR-as-implemented likely isn't enough, honestly hypothesized (not yet verified)**: AMR's
refinement signal is `|dem_energy_per_point|` (+ constitutive-consistency residual), which
reflects where the CURRENT network's residual is large — not literally "distance to the hole."
Early in training (step 200, the first sweep) the network has barely learned the true field yet,
so its residual signal may not yet correlate tightly with the true stress-concentration location;
by the time later sweeps (step 1200, 2200) have more physically meaningful residual signal to
work from, a large fraction of the 3000-step budget is already spent. This is a plausible
chicken-and-egg mechanism, not a confirmed root cause — distinguishing it from "just needs a
finer initial resolution" or "just needs more/earlier sweeps" would need its own controlled
experiment, not assumed here.

**Conclusion, honestly stated**: PH4-14/L5 is NOT VERIFIED and is not being forced to pass, with
or without AMR. Issue #74 (the crash) is genuinely fixed — that part of the original hypothesis
was correct. But fixing the crash did not, by itself, solve L5's underlying Kt accuracy problem,
which is a real, more nuanced finding than "L5 is blocked on #74." Real non-hacked evidence
exists for both attempts. Candidate next steps for a future session (none attempted here,
per the verification-cost policy without stronger justification first): earlier/more frequent
AMR sweeps within the same budget, a much larger uniform `n_interior` as an AMR-free alternative,
or a hybrid initial-density bias (structural, not residual-driven) seeded from hole geometry
directly rather than waiting for the network's own residual to discover it.

## PH4-15 — Independent displacement and strain validation

Status: VERIFIED for no-hole (issue #64), VERIFIED-VIA-EQUILIBRIUM for hole geometries (issue
#63 sub-issue #69)

Implementation: added `validate_no_hole_fields`, an independent grid validator over published
displacement, strain, and constitutive stress fields. It reports L2/L∞ field errors plus rigid
translation and antisymmetric-gradient rotation residuals. Test
`no_hole_field_validation_accepts_affine_and_detects_translation` passes with an exact affine
field and rejects a unit rigid translation. This test is measure/source independent and does
not consume training loss. No-hole side wired into headless output and VERIFIED by #64/#66/#68's
real runs (see those items' own entries).

Hole-side extension (#69): no closed-form affine reference exists for a holed geometry, so
`validate_no_hole_fields`'s own exact-field-comparison approach doesn't generalize directly.
Instead wired `probe_reaction_force` (already implemented and tested, never previously called
from any printed output — a real, previously-missing wiring gap, not a missing implementation)
into `user_runner.rs`'s headless output: for ANY valid elastic solution, hole or no-hole, the
net resultant traction integrated around the WHOLE closed outer boundary must be zero (far-field
loading self-cancels around a closed rectangle regardless of interior holes) — a real, physical,
closed-form-independent invariant. Three real smoke runs (single-hole, 2-hole mixed-BC
non-square, 3-hole asymmetric mixed-BC — `Debug_run/phase4/issue69_topology/`), all at a short
800-step budget (machinery smoke test, not a convergence claim — see PH4-17 below): equilibrium
error `0.3%-1.5%` of reference force in all three, a real, independently-computed signal the
trained field is at least roughly physically consistent even before full convergence.

## PH4-16 — Non-square geometry

Status: VERIFIED (issue #63 sub-issue #68)

The corrected no-hole field baseline this item was blocked on now exists (#64/#66/#67). New
example `examples/problems/variational_no_hole_plate_nonsquare.toml`: identical formulation/
material/load/network/training configuration to the validated square baseline, `half_w=0.15`/
`half_h=0.08` (aspect ratio 1.875:1, deliberately far from square so a latent square-domain
assumption would show up clearly rather than being masked by a near-1 ratio). No code changes
were needed — `run_no_hole_benchmark`/`validate_no_hole_fields`/`ExternalWorkTerm`'s measure-
aware boundary integral already read `geometry.half_w`/`half_h` independently (the codebase's
own prior audit already confirmed the non-square `ds` gap was closed for the measure-aware path
specifically), so this item was purely a missing runtime artifact, not a missing fix.

Real headless CLI run (`Debug_run/phase4/issue68_nonsquare/solver.log`): `normalized_Pi=
-1.000012` (matches the true continuum affine minimum), all five P2-14 hard thresholds PASS
(`sigma_xx_relative_error=0.0016`, `sigma_yy_over_ref=0.0009`, `sigma_xy_over_ref=0.0006`,
`traction_rms_over_ref=0.0026`, `load_transfer_ratio=0.9971`), independent field validation
(separate grid, not training points) also PASSES. Quality is slightly lower than the square
case's own numbers (e.g. `sigma_xx=0.0016` vs `0.0010`) but comfortably within threshold with
real margin — consistent with a genuinely harder (more elongated) geometry, not a latent bug.

## PH4-17 — Arbitrary topology/multiple holes

Status: MACHINERY-VERIFIED (issue #63 sub-issue #69); hard Kt-accuracy acceptance remains
sub-issue #70's job

The no-hole L4 companion this was blocked on now exists (#64/#66/#67). Three new example
configs (`variational_single_hole_smoke.toml`, `variational_notched_smoke.toml` — 2-hole,
non-square, mixed Free/Fixed BC — `variational_triple_hole_smoke.toml` — 3-hole, asymmetric,
mixed BC), all Corrected-Variational, all run cleanly end-to-end via the real headless CLI path
(`Debug_run/phase4/issue69_topology/`): no crashes, finite/sane diagnostics throughout
(sampling near holes, `FieldKind`-resolved stress source, hole-BC terms — `hole_free`/
`hole_fixed` — gauge terms — `translation_gauge` correctly NOT registered when a `Fixed` hole
already anchors the geometry, `rotation_gauge` active and well-behaved for the pure-Free
single-hole case), Kt computed and finite for every hole in every run (1, 2, and 3 holes
respectively), closed-boundary equilibrium error 0.3%-1.5% of reference force in all three.

Deliberately short (800-step) smoke budget — this item's own acceptance criteria is "does the
topology machinery work" (sampling, stencil validity, boundary measure, `FieldKind`, gauge/
nullspace, QoI source), not "does Kt converge to an accurate value," which needs much longer
training and is explicitly sub-issue #70 (real L5)'s separate job per issue #63's own text
("L4 requires a specific verified no-hole L4 companion... a finite Kt from an unconverged model
is diagnostic only"). The Kt values these smoke runs report (0.5-0.6, all well below the
physically-expected >1 stress-concentration range) are exactly the "unconverged, diagnostic
only" case that text describes — not a regression, an honest reflection of the short budget.

## PH4-18 — Formulation support matrix

Status: VERIFIED (issue #63 sub-issue #72)

`docs/FORMULATION_SUPPORT_MATRIX.md` updated with real evidence from the full #64-#71 chain.
Variational's L4 no-hole (square and non-square), measure-aware integration, translation gauge,
FieldKind enforcement, and QoI stress source rows are now `VERIFIED`, each citing a specific
real run or fast fixture (no speculative promotion). L5 hole/Kt moved from
`EXPLICITLY_UNSUPPORTED` to `SUPPORTED_WITH_LIMITATION` for Variational — the mechanism works
and was exercised for real (issue #70), it just doesn't yet reach the theoretical reference
value; `EXPLICITLY_UNSUPPORTED` would have been dishonest now that a real, non-crashing,
correctly-classified result exists. AMR row for Variational now explicitly states
`VERIFIED_DISABLED` with the crash reference (#67/#74), not a bare capability label. Strong/
Hybrid/Weak rows remain unchanged from their pre-existing conservative state — no evidence was
collected for those formulations this session, so none was claimed.

## PH4-19 — Mathematical objective snapshot

Status: VERIFIED (issue #64 sub-issue #66)

Shared plate checkpoint and GUI-report builder now saves `MathematicalObjectiveSnapshot`; see
PH4-05. Backward JSON compatibility uses `#[serde(default)]`.

Regression: `serve_loaded_plate_checkpoint_saves_a_full_authoritative_report_for_a_no_hole_geometry`
now performs actual save/load JSON round-trip and asserts physical `U`, full `W_ext`, `Pi`,
active terms, and positive reference energy are present in persisted report.

Real checkpoint artifact from corrected Variational training (#66): `runner::tests::
variational_no_hole_plate_trains_and_produces_real_benchmark_evidence` (`#[ignore]`d, real
production-scale run of the shipped, now-passing config) saves a checkpoint via the existing
`run_training_user_problem`/`ControlMsg::SaveCheckpoint` machinery and asserts the persisted
`MathematicalObjectiveSnapshot`'s `Pi=U-W_ext` consistency, non-empty `active_terms`/
`base_weights`, and a finite `normalized_pi` — a reviewer can reconstruct the physical objective
from the saved artifact alone, per this item's own acceptance wording. A fast (2.46s) companion
test, `runner::tests::corrected_variational_checkpoint_persists_a_reconstructable_objective_snapshot`,
proves the identical persistence mechanism on a tiny (16/16-point) spec, independent of whether
training has converged — added specifically so this mechanism doesn't need re-verifying via
another 20+ minute run for every future change (issue #65/#66's own verification-cost policy).

## PH4-20 — General physical regressions

Status: VERIFIED (issue #63 sub-issue #72)

Rather than write a new duplicate consolidated suite, curated the fixture list issue #63/#72
specify against the ALREADY-EXISTING default (non-`#[ignore]`d) test suite — nearly every
fixture category already has a fast, real-physical-assertion test that already runs in every
`cargo test --workspace`, confirmed by name in this session's CI logs:

| Fixture (issue #63/#72) | Covered by (fast, default-run) |
| --- | --- |
| Affine L0 | `verification_ladder::tests::affine_amplitude_test_recovers_a_exact_for_{aluminum,steel}...` |
| Uniform/nonuniform integration | `ph4_04_interior_energy_integral_agrees_across_uniform_nonuniform_and_amr_like_sampling` |
| Non-square plate | **gap — added this pass**: `no_hole_field_validation_accepts_affine_on_a_non_square_plate` (zero-cost affine-field variant of the real #68 run, `half_w=0.15`/`half_h=0.08`) |
| Strong/Hybrid/Variational no-hole (term activation) | `{strong,hybrid,variational}_formulation_on_a_no_hole_geometry_activates_...` |
| Single/multiple holes (sampling/stencil) | `named_point_sets_returns_one_ring_per_hole_...`, `hole_ring_points_lie_on_their_hole_circle_...`, `sample_interior_points_never_fall_inside_any_hole_...` |
| Boundary stress projection | `hoop_stress_projection_matches_hand_computed_values_at_cardinal_angles` |
| Kt radial/angular convergence | **explicit gap, not closed**: genuine convergence-as-training-progresses cannot be fast-tested without a real training run; covered only by the `#[ignore]`d issue #70 L5 test and issue #69's real smoke runs, not by the default suite |
| AMR invariance | `ph4_04_interior_energy_integral_agrees_across_uniform_nonuniform_and_amr_like_sampling`; `set_interior_weights_none_clears_a_previously_set_weighting_back_to_unweighted` |
| FieldKind enforcement | the 11 focused `FieldKind`/`resolve_field` tests (PH4-12) |
| Objective/provenance round-trip | `corrected_variational_checkpoint_persists_a_reconstructable_objective_snapshot`, `serve_loaded_plate_checkpoint_saves_a_full_authoritative_report_for_a_no_hole_geometry` |
| Operational gate | `verification_ladder::tests::operational_gate_{passes_only_when_both_l0_and_l4_pass,fails_at_l0...,fails_at_l4...}` |

One real gap found and closed this pass: no fast non-square fixture existed (only the real,
expensive #68 headless run). Added `no_hole_field_validation_accepts_affine_on_a_non_square_
plate` (`user_problem.rs`), same zero-cost exact-affine-field technique as the existing square
fixture, `half_w!=half_h`, passes with the same tight tolerances. Runs by default.

One gap deliberately NOT closed: Kt radial/angular *convergence* (as opposed to structural
correctness of the angular-profile mechanism, which `probe_hole_boundary_profile_samples_
points_on_the_circle_and_computes_consistent_von_mises` already covers) inherently requires a
real trained model and cannot be made fast without losing what it's testing — forcing a fake
"convergence" fixture would violate the no-benchmark-hacking rule as surely as forcing a Kt
threshold would. This is the honest, correctly-scoped limit of what PH4-20 can guard for free.

## PH4-21 — Final operational status

Status: COMPUTED (issue #63 sub-issue #72) — see `docs/GENERAL_SOLVER_OPERATIONAL_STATUS.md`
for the full scoped breakdown. Summary, per capability, not a blanket claim:

- Corrected-Variational no-hole (square AND non-square): **OPERATIONAL**. This was the item
  blocking every downstream PH4-XX item; it is now real, evidenced, and closed (#64/#66/#67/#68).
- Corrected-Variational multi-hole topology machinery: **PARTIALLY_OPERATIONAL** (machinery
  works; Kt accuracy does not).
- Corrected-Variational hole/Kt accuracy (L5): **NOT_OPERATIONAL** — real attempt made (#70),
  root cause diagnosed (near-hole sampling starvation), blocked on issue #74 (AMR+Variational
  autodiff crash), not on further tuning.
- AMR + Variational (any geometry): **NOT_OPERATIONAL**, deliberately disabled pending #74.
- Strong/Hybrid/Weak `FormulationSelection`: **NOT_OPERATIONAL** (untested in a real run this
  epic; Weak has no implementation).

This supersedes the prior BLOCKED status, which predates the #64 root-cause fix and is now
factually wrong (it cited `load_transfer_ratio=0.1996`, a number from the broken pre-fix state).
Not inflated to a blanket OPERATIONAL claim, per this item's own acceptance wording — the parts
that don't work (L5 accuracy, AMR+Variational, Strong/Hybrid/Weak) are stated as plainly as the
parts that do.

Validation: full workspace test suite passed 460/461 non-ignored tests as of the last complete
local run this session (one flaky f32-precision assertion, unrelated to any Phase 4 physics
change, fixed separately — see the network.rs commit fixing `coordinate_skip_represents_
affine_displacement_and_leaves_stress_mlp_only`). CI is the authoritative full-suite gate per
this project's own "CI over local runs" convention; a full local run was avoided where CI
coverage sufficed. Clippy `-D warnings` pre-existing `pinn-core` warnings are unrelated to this
epic's scope and not treated as Phase 4 proof either way.

## PH4-22 — Issue #75: persistent geometry-aware AMR for small-hole Kt convergence

Status: CORE MECHANISM VERIFIED AND WORKING; HYPOTHESIS DISPROVEN BY REAL EVIDENCE (L5 still
NOT_OPERATIONAL). Issue #75's own "Definition of done" condition 7 explicitly allows this
outcome: "L5 passes against a valid reference, OR remains explicitly open with measured evidence
and a narrowed next hypothesis" — this section delivers the latter, honestly, not the former.

### What was built (workstreams A-D, all implemented and tested)

- **Workstream A** (`pinn-core/src/amr.rs`): `AdaptiveGrid::sample_points_jittered_with_density`
  — same leaf topology/DFS order as the existing deterministic-center samplers, but each point
  is a fresh, uniformly-jittered draw within its own leaf's bounds every call (issue #64's
  invariant extended to AMR). 6 new fast tests (topology-preserving freshness, seed
  reproducibility, containment, density-metadata pairing, hole-zone density, no-hole geometry).
- **Workstream B** (`pinn-core/src/amr.rs`): `sample_hole_annulus` — geometry-seeded, uniform-
  BY-AREA (not by-radius; `r² ~ Uniform(inner_r², outer_r²)`) quadrature in an annulus around
  each hole, active from training step 0 without waiting on the network's own residual signal.
  6 new fast tests (radial range, statistical uniform-by-area distribution, area-share sum,
  freshness, reproducibility, degenerate-input safety).
- **Workstream C** (`pinn-solver/src/user_problem.rs`): `apply_persistent_adaptive_interior_
  sample` — the single source of truth for "uniform vs persistent geometry-aware adaptive"
  interior sampling, combining A+B, always returning points and matching compensation weights
  from the same call (never stale). No-op (pass `amr: None`, or a no-hole geometry) preserves
  every existing caller's behavior exactly. 5 new fast tests, including a real numerical
  acceptance check: near-hole fraction after this function jumps from uniform sampling's
  measured ~0.55% to consistently >5% (real observed: 9-14%, see below).
- **Workstream D** (`pinn-solver/src/runner.rs`, `run_user_problem_training_from`): wired
  persistent sampling into the real production training loop. The pre-existing sweep-only AMR
  block (probe residuals, `update_residuals`, `adapt()`) still runs at the normal interval,
  unconditionally — topology tracking is unchanged. What changed: for a HOLE-bearing geometry
  with `amr_enabled=true`, the legacy block's own `data.int_norm` overwrite is skipped (a new
  block below it calls `apply_persistent_adaptive_interior_sample` every step instead, not just
  sweep steps); for a no-hole geometry, the legacy sweep-only path is completely unchanged
  (byte-for-byte — verified below).

### Real verification (not just "compiles")

- `runner::tests::issue_75_persistent_adaptive_amr_survives_a_real_hole_bearing_training_run`
  (`#[ignore]`d, 300 real steps, release): completes without panic, `total_loss` finite
  throughout, at least one AMR sweep fires. Real telemetry confirms the mechanism works exactly
  as designed: near-hole density (within 2 hole-radii) stays at **9-14% throughout the entire
  run**, active from step 0 — compare uniform sampling's own measured ~0.55% for the identical
  geometry. Point count grows 632→932 after the step-200 sweep, density staying strong.
- `runner::tests::ph4_09_controlled_comparison_fixed_sampling_vs_amr_corrected_variational_same_
  budget` (the no-hole, 2200-step crash-reproduction/regression test) re-run after this change:
  completes cleanly, numbers consistent with pre-#75 runs (`sigma_xx≈0.011` both branches) —
  confirms the legacy no-hole AMR path is genuinely unaffected by this change.
- Full fast suite: 467 passed (up from 462, confirming the new fast tests run and pass), 1
  known pre-existing contention-flaky test (`gui_streaming_step_zero_matches_independent_
  shared_function_computation` — third distinct failure value across three separate full-suite
  runs this session, confirmed non-regressive by isolated re-run every time).

### The decisive real experiment — and the honest negative result

`user_problem::tests::issue_75_real_l5_with_persistent_geometry_aware_amr` (`#[ignore]`d, real
dual training run, identical L5 configuration to every prior L5 attempt this epic: 3000 steps,
`n_interior=n_boundary=4096`, ratio=0.05 hole, against a verified-passing no-hole companion):

- No-hole companion: passes cleanly (`sigma_xx_relative_error=0.00084`, `load_transfer_ratio=
  0.99943`) — identical to every prior run of this exact config, confirming persistent AMR
  (hole-specific) has zero effect on the no-hole path, as designed.
- Hole benchmark: **`kt=1.0023`**, `relative_error_vs_infinite_theory=0.6659` (66.6%).

**Compare all three real L5 attempts this epic, same configuration throughout:**

| Sampling strategy | Kt | Relative error |
| --- | --- | --- |
| Uniform (no AMR) | 1.0076 | 66.4% |
| Old sweep-only AMR (post-#74 fix) | 1.0088 | 66.4% |
| **Persistent geometry-aware AMR (#75)** | **1.0023** | **66.6%** |

**Despite a real, verified, sustained ~20x increase in near-hole collocation density (0.55% →
9-14%), geometry-seeded from step 0, Kt did not move in any meaningful direction.** This is a
genuine, important, honestly-reported negative result for the epic's central hypothesis — not a
bug, not an implementation defect (the mechanism itself is proven correct and working by every
test above), and not something to hide, force, or explain away with a benchmark-specific
correction. Sampling density alone is now disproven as the (or at least the dominant) bottleneck.

### Narrowed next hypothesis (per issue #75's own prioritized "Risks and follow-up hypotheses")

Issue #75's own text anticipated exactly this outcome and gave a priority-ordered investigation
list. With density now ruled out, the next candidates per that list are:

1. Compare against an independent finite-element/high-resolution numerical reference for this
   exact finite square geometry (not yet done — `Kt=3.0` is the infinite-plate value; this
   finite, `ratio=0.05` square plate's TRUE reference value has never been independently
   established in this codebase).
2. Measure constitutive-consistency error specifically in the FD-safe annulus outside the hole.
3. Measure hole traction residual using DERIVED stress, not only direct mDEM stress.
4. Evaluate whether direct stress satisfies hole traction while derived stress remains
   physically incorrect near the boundary (this codebase's free-hole traction term acts on
   DIRECT mDEM stress at the hole ring — see `HoleBcTerm` — while Kt itself is measured from
   DERIVED constitutive stress at an offset; if these two representations disagree near the
   hole specifically, satisfying one loss term would not guarantee the other is physically
   correct, independent of how many collocation points are nearby).

Given more density didn't help, (3)/(4) — a real mismatch between what the loss trains against
(direct stress at the ring) and what Kt measures (derived stress at an offset) — is the most
promising next lead, not yet investigated with real evidence as of this note.

### Follow-up real experiment — Strong formulation (explicit hole-boundary loss) + persistent AMR

Reading `UserDefinedProblem::loss_terms()` after the null density result above found a real,
code-grounded reason it might not be surprising: `hole_free_active = !matches!(formulation,
Variational)` — every L5 attempt so far used `Variational`, for which a `Free` hole registers
**no `HoleBcTerm` at all**. Traction-free at the hole is enforced only implicitly through the
energy functional's natural boundary condition, not as an explicit local loss. `Strong`
formulation is structurally different: it registers an explicit `HoleBcTerm::Free` (direct-
stress traction-free penalty, evaluated locally at the hole ring) alongside `EquilibriumTerm`/
`OuterTractionTerm`, and had already proven it passes the no-hole L4 benchmark cleanly (#71).

`user_problem::tests::issue_75_real_l5_strong_formulation_with_persistent_amr` (`#[ignore]`d,
real dual training run, Strong formulation combined with persistent AMR, same L5 hole
configuration): no-hole companion passes cleanly again (`sigma_xx_relative_error=0.00223`,
`load_transfer_ratio=1.00227` — identical to #71's own measured Strong result, a real
consistency check). Hole result: **`kt=0.1057`, `relative_error_vs_infinite_theory=0.9648`
(96.5% error) — WORSE than every prior attempt, not better.**

**Updated real comparison table, all four attempts on the identical L5 hole configuration:**

| Configuration | Kt | Relative error |
| --- | --- | --- |
| Variational, uniform sampling (no AMR) | 1.0076 | 66.4% |
| Variational, old sweep-only AMR | 1.0088 | 66.4% |
| Variational, persistent geometry-aware AMR | 1.0023 | 66.6% |
| **Strong (explicit hole loss) + persistent AMR** | **0.1057** | **96.5%** |

Two real, code-grounded hypotheses have now been tested and disproven with real evidence in
this epic: (1) sampling density alone (workstreams A-D, the epic's own central mechanism) — no
meaningful effect; (2) adding an explicit local hole-boundary loss term (`Strong` formulation)
— actively worse, not better. `HoleBcTerm::Free` operates on the fixed `"hole_i"` named ring
point set (unaffected by the interior-sampling changes in this epic), so this regression is not
an AMR/HoleBcTerm interaction artifact — it reflects Strong's own optimization dynamics for
this specific hole problem being worse than Variational's, independent of AMR.

**Honest status, per this epic's own "Definition of done" condition 7**: L5 remains explicitly
open with real measured evidence and a narrowed set of ruled-out causes, not a passing result.
Issue #75's own prioritized "Risks and follow-up hypotheses" list items 1-4 (independent FEM/
high-resolution reference for this finite geometry; constitutive-consistency error in the near-
hole annulus; hole traction residual via DERIVED not direct stress; whether direct and derived
stress representations agree near the hole at all) are DIAGNOSTIC measurements on an already-
trained model, not more formulation/sampling experiments — a fundamentally different, more
targeted next step than the two real training-based hypotheses just tested, and a natural point
to pause the active real-experiment phase and report comprehensively rather than continue
guessing at formulation/mechanism changes without a new, specific, evidenced hypothesis to test.

## PH4-23 — Correction to PH4-09/issue #74: the real root cause and fix

Status: VERIFIED. This corrects (does not merely supplement) PH4-09's own "Issue #74 fix"
section above — that diagnosis was real but incomplete, and its "FIXED and verified" claim on
issue #74 was based on insufficient evidence (two clean runs of an intermittent crash).

### What was actually wrong with the original diagnosis

The original fix (routing `probe_interior_energy_residuals` through `BInner` instead of the
live `Autodiff` graph) is a real, valid improvement — it removes a genuine forward-without-
backward graph leak — but it was **not** the cause of the crash `ph4_09_controlled_comparison_
fixed_sampling_vs_amr_corrected_variational_same_budget` reproduces. Direct evidence: the crash
fires on a fresh model's very first `.backward()` call, immediately after the prior arm
finishes — before AMR's own `AMR_WARMUP_STEPS` (200) even elapses, so no AMR residual probe has
ever run at the point of failure.

### Real root cause (confirmed via upstream source and issue tracker, not inferred)

This is a **real, confirmed, currently-unreleased upstream burn-autodiff bug**:
[`tracel-ai/burn` issue #5573](https://github.com/tracel-ai/burn/issues/5573), "A concurrent
`backward()` frees another thread's autodiff steps, silently losing a gradient" — fixed by
[PR #5647](https://github.com/tracel-ai/burn/pull/5647) ("retain input nodes until child
registration"), merged 2026-09-11. Not in any published release (0.21.0 predates it; 0.22.0 is
still pre-release).

Mechanism, read directly from burn-autodiff 0.21.0's own source
(`burn-autodiff/src/runtime/graph.rs`): `AutodiffServer` state lives behind a **process-global**
`static STATE: Mutex<Option<GraphLocator>>`. Every `.backward()` call anywhere in the process
triggers `GraphCleaner::cleanup_orphaned_entries()`, sweeping **every** graph currently tracked
process-wide via an unsound liveness check (`Arc::strong_count > 1`). A node can be live-but-
momentarily-unreferenced during normal op internals (e.g. mid-`float_cat`, per #5573's own
detailed trace); a concurrent sweep from a *different* thread's backward() can free it in that
window.

Two independent, compounding triggers were found in **our own test harnesses** (not production
training code):

1. `ph4_09_controlled_comparison_...`/`ph3_12_controlled_comparison_...` spawned a **separate OS
   thread per comparison arm**. Even though arm 1 fully joins before arm 2 starts, `burn-
   ndarray`'s persistent process-global rayon thread pool can leave trailing work in flight -
   genuine cross-thread overlap despite the sequential-looking test structure.
2. The same two tests built one `initial_model` and handed `.clone()` to arm 1 while moving the
   *original* into arm 2. **Cloning a burn `Module`/`Tensor` is a cheap clone that shares the
   same underlying autodiff `NodeId`** - it does not mint a fresh leaf. Arm 1 drives that shared
   identity through ~2200 real `.backward()` calls; arm 2 touching the same identity afterward
   is exactly #5573's own "leaf reused after an earlier `backward()`" trigger, independent of
   any thread boundary at all.

`gui_streaming_step_zero_matches_independent_shared_function_computation`'s long-documented
"contention-flaky under full-suite load" behavior (attributed all epic to vague floating-point
nondeterminism) has the same shape as trigger (1): a real `.backward()` call in the test's main
thread (the "independent reference" computation) immediately followed by a separately-spawned
training thread. This is very likely the actual, previously-undiagnosed cause.

### Fix

`crates/pinn-solver/src/runner.rs`: for all three tests, (a) call `run_user_problem_training_
from`/`run_training_user_problem` directly, synchronously, in the test's own single thread
instead of spawning a background thread per call, and (b) where a shared model previously fed
both arms via `.clone()`/move, build two **independently-initialized** models from the same
seed (`B::seed` + `.init()`) instead - matching the pattern `issue_70_real_l5_*`/`issue_75_
real_l5_*`/`issue_71_real_strong_and_hybrid_*` already used (which is why those real Kt results
were never at risk from either mechanism - they never share a model identity or a second
thread).

**Also attempted, reverted**: pinned `burn`/`burn-ndarray` to the exact post-#5647 commit via a
git dependency. Burn's `main` has since undergone a much larger `Tensor<B, D>` API shape change
beyond that one fix, producing 118+ compile errors across this codebase - a full migration is
wildly disproportionate to fixing one concurrency bug in our own test harnesses. Reverted to the
published `version = "0.21"`; see the dependency's own comment in the workspace `Cargo.toml` for
when to revisit.

### Verification (real, not just "compiles" or "one clean run")

- Both `ph4_09`/`ph3_12` re-run under **forced concurrency** (2-3 concurrent release processes
  competing for CPU, the same condition that reliably reproduced the crash before the fix): all
  passed cleanly. The two independent `ph4_09` processes produced **byte-identical** final
  results (`sigma_xx_relative_error=0.011453909629973476` in both) - real evidence of genuine
  determinism restored, not just absence of a crash this one time.
- `gui_streaming_step_zero_...` passes 10/10 in isolation with stable timing (no more left/right
  divergence across runs).
- Full `cargo test -p pinn-solver --features ndarray-backend -- --test-threads=1` (the exact CI
  invocation): **471 passed, 0 failed, 34 ignored** - clean, no flaky-test failure at all.
- A `cargo test` run WITHOUT `--test-threads=1` (cargo's own default, multiple test *functions*
  running concurrently) can still occasionally show `gui_streaming_step_zero_...` fail - this is
  the SAME upstream bug at the cross-test-function level (a completely different, unrelated test
  function's own training thread racing with this one), not a new or unfixed issue. **CI is
  unaffected**: `.github/workflows/rust.yml` already runs every job with `--test-threads=1` (for
  an unrelated, pre-existing GPU/lavapipe-contention reason, documented in its own comment) -
  different test functions never run concurrently there. For a reliable full-suite run locally,
  use `--test-threads=1`.

### Durable lesson

A burn `Module`/`Tensor` `.clone()` is a **cheap, identity-sharing clone**, not a deep copy that
mints a fresh autodiff leaf. Any future test (or production code) that wants two genuinely
independent training runs/graphs from "the same starting weights" must construct them via two
separate `.init()` calls under the same seed, never via `.clone()`/move of one shared instance.

## PH4-24 — Issue #77's real root cause: no hole-BC signal + a Monte-Carlo SNR floor, not a
## representation limit

Every #77 attempt through the annular/outer decomposition (rational features Kt=1.046,
chart-envelope Kt=1.135, plain two-domain decomposition Kt=1.065 - all near the "no hole at all"
Kt=1.000) changed *representation* while holding two things constant that turned out to be the
actual gate: (a) neither `Variational`'s `loss_terms()` nor `AnnularDecompositionProblem`'s ever
registered a term referencing the hole boundary at all (`hole_free_active =
!matches!(formulation, Variational)`; the annular sampler emitted an orphaned `"hole_0"` point
set consumed by zero terms - `validate_loss_terms` only checked declared domains exist, never
that a declared point set is read by anything), and (b) the hole's own contribution to the
domain-integrated strain energy Pi, restricted to the region collocation points can actually
reach, is only **0.38% of Pi** at L5's hole size (`a=0.005`) - below the **~0.134%** Monte-Carlo
standard error of the SAME n=4096 energy estimator (SNR~2.8, computed analytically from the
closed-form Kirsch field). A pure uniform affine field (`sigma_xx=px, sigma_yy=sigma_xy=0` -
genuinely no hole effect) scores **exactly** Kt=1.000 under the reported metric, so the observed
1.02-1.13 band across three representations is "affine plus noise," not "concentration
under-resolved" - exactly why changing representation alone never moved the number.

**Fix (three-step plan, sequential, each gated on a real run before the next)**:

1. **Kinematic decomposition + corrected hole term** (this entry): `u_total = u_affine + u_hole`
   for the single-centered-Free-hole case, `u_affine` the exact closed-form uniaxial-tension
   field (`run_no_hole_benchmark`'s own reference solution). The network represents only the
   residual correction instead of competing for gradient budget against the dominant,
   trivially-learned affine part. Every energy functional adds the constant affine strain
   before calling into the EXISTING `dem_energy_per_point`/`compute_stress` machinery, so the
   cross term `C:eps_affine:eps_hole` (mathematically required for argmin-equivalence to the
   true Pi - dropping it would reproduce this file's own PH4-03 "not a uniform rescaling,
   changes the stationary point" defect one level deeper) is present by construction, verified
   by a dedicated correctness-gate test (`affine_strain_cross_term_is_present_not_dropped`)
   against an independently hand-derived quadratic expansion. `HoleBcTerm`'s traction-free
   residual is retargeted to `-sigma_affine.n` (not zero - the network no longer represents the
   whole field) and switched from the direct mDEM stress head (documented elsewhere in this
   file as never developing real spatial structure) to a derived/constitutive read on a new
   FD-safe ring. A rejected alternative considered first - scaling the strain-energy integrand
   inside `r/a<=2` to artificially inflate the hole's share of the objective - was NOT used: it
   is the weak form of elasticity with a spatially varying modulus, manufacturing a spurious
   stress discontinuity at the weight boundary and changing which BVP is actually being solved,
   the same defect class as this file's own historical `lambda_U`/`lambda_W` bug in spatial
   form. `problem::validate_point_sets_consumed` added as general hardening - would have caught
   the orphaned `"hole_0"` point set structurally.

   **Real result** (`issue_77_l5_annular_decomposition_converges_to_fem_reference`, release,
   3000 steps, `n_interior=4096`, FEM reference `2.460638516`): **Kt=1.213417**, error 50.687%
   - fails the test's own 5% acceptance gate, but moved in the predicted direction from the
   pre-fix annular decomposition's Kt=1.065 (+0.148, real, not noise-floor-sized). Expected and
   predeclared as a *partial* result before this fix landed: collocation points still cannot
   reach the boundary layer at all (`ring_anchor_margin_m` scales with plate size, not hole
   size - at L5 the exclusion radius is `1.08*radius`), which is what Step 2 (hole-relative
   collocation margin) and Step 3 (energy-estimator variance reduction, reusing #76's cut-cell
   quadrature) address next. Reported honestly per this file's own no-benchmark-hacking
   discipline - not a pass, a real, predicted, directionally-correct partial improvement.

Full workspace regression: `cargo test -p pinn-core -p pinn-solver --lib --features
ndarray-backend -- --test-threads=1` -> 484 passed, 0 failed, 36 ignored (pinn-core: 137 passed).
Exposed and fixed two pre-existing test-fixture bugs unrelated to this fix while getting there
(both in the chart-embedding rewrite that landed alongside this piece, not introduced by it):
`tiny_model(n_fourier)` used a stale Fourier-width formula instead of the current
`coordinate_embedding().input_dim()`, and `legacy_record_loads_with_zero_coordinate_skip` built
its "legacy" fixture at the new chart-embedded width instead of the literal pre-embedding width
its own `input_dim: None` fallback assumes - both produced a real
`burn-ndarray::ops::matmul::Dimensions are incompatible` panic, not a regression from this fix.

Does not close issue #77 (or #74/#76) - Steps 2 and 3 remain, reported with evidence as each
completes.

## PH4-25 — Step 2 (hole-relative ring margin): implemented, real result flat vs Step 1

`ring_anchor_margin_m` (PH4-24) is plate-scaled (`fd_h * max(half_w,half_h)`), not hole-scaled -
for L5's `radius=0.005` this makes the decomposed hole-traction term's own FD-safe ring sit at
`r=1.08*radius`, a margin/radius ratio that gets proportionally worse as the hole shrinks (the
same ratio for `single_hole_plate.toml`'s larger `radius=0.02` is already `1.02*radius` "for
free" under the same plate-scaled formula).

**Fix**: `hole_ring_margin_m(radius) = 0.02*radius` (hole-relative, uniform ratio regardless of
plate size) plus `hole_ring_fd_config` - a genuinely SMALLER `FdConfig` sized specifically for
this tighter margin, since the margin is bounded below by whatever FD step the ring's own
stencil uses. Required threading a new `MultiStepCtx::hole_fd`/`FrozenMultiStepCtx::hole_fd`
field through `compute_domain_forwards` (selected only for point sets whose name ends `"_fd"` -
every other point set's own `fd`/FD accuracy is completely unaffected) rather than just editing
a margin formula, once it became clear `ctx.fd` is one single global step shared by every point
set today - confirmed by reading every `assemble_stencil`/`compute_strains` call site in
`compute_domain_forwards` before implementing, not assumed. Explicitly scoped to NOT touch
`ring_anchor_margin_m` itself, the interior-collocation exclusion, or the Kt-measurement probe's
own margin/radius convention (`probe_hole_boundary_profile_derived`) - changing the measurement
convention would invalidate `FEM_KT=2.460638516`'s comparability without a fresh FEM re-run,
explicitly out of scope for this step.

**Real result** (same test, same release/3000-step/`n_interior=4096` configuration as PH4-24):
**Kt=1.206957**, error 50.949% - essentially FLAT versus PH4-24's `1.213417` (a ~0.5% move,
inside run-to-run noise, not a real improvement). Reported honestly: Step 2 alone did not move
Kt further. Working interpretation (not yet independently verified): the hole-traction term's
own probe-ring RADIUS was not the binding constraint - the interior energy estimator's Monte-
Carlo variance (the OTHER half of this session's SNR finding, still governed by the untouched,
plate-scaled interior-collocation margin) is the more likely remaining bottleneck, which is
exactly what Step 3 (stratified/variance-reduced sampling, reusing #76's cut-cell quadrature)
targets next. This interpretation is a hypothesis for Step 3 to test, not a re-litigated claim.

Full workspace regression: 487 passed, 0 failed, 36 ignored (3 new focused tests added for the
hole-relative margin/FD-config machinery, all passing; zero regressions).

Does not close issue #77 (or #74/#76).

## PH4-26 — Step 2 flat result explained: Kt peaks around step 3000, then DECLINES with
## more training — a shared, loss-driven LR schedule starves the annulus domain, not undertraining

PH4-25's flat Step 2 result prompted a zero-cost check (this session): the annulus domain's own
energy estimator, restricted to just the annulus region (`r` in `[1.02*radius, 3*radius]`,
`n_annulus=2048`), has SNR~12.2 against the same closed-form Kirsch field - well above detection
threshold. Sampling variance was never the annulus estimator's bottleneck post-decomposition;
Step 3 (stratified sampling) as originally planned would not have helped and was not implemented
as originally scoped.

The per-checkpoint term-gradient ledger (`issue_77_l5_annular_diagnostic_trace`, real run, Steps
1+2 active) showed every term has live, nonzero gradient (no term is dead/starved to zero), but
`physical_potential` (outer) sat flat in a `0.98-1.10` band from step 0 - it converges almost
immediately post-decomposition, since it now represents only the small residual correction to
the exact affine background field. Total loss - dominated by `physical_potential`'s magnitude -
was correspondingly still decreasing only slowly by step 2700, while Kt was still rising at the
final (step 2999) checkpoint. This looked like undertraining.

**It was not.** A real 12000-step extended run (`issue_77_l5_extended_convergence_trend_trace`,
release, 9705.57s, checkpoints `[0, 1500, 3000, 6000, 9000, 11999]`) gives the real trajectory:

| step | Kt |
|---|---|
| 0 | 0.255174 |
| 1500 | 1.166180 |
| **3000** | **1.287459 (peak)** |
| 6000 | 1.169102 |
| 9000 | 1.010714 |
| 11999 | 0.976746 |

Kt rises to a real peak around step 3000, then **declines steadily for the remaining 9000
steps**, ending below its step-1500 value. More training actively hurt, not helped - the
opposite of this entry's own working hypothesis after PH4-25.

**Root cause, confirmed by reading the code (not the gradient-starvation mechanism a plausible-
sounding first guess suggested)**: `run_annular_decomposition_training_inner`
(`user_runner.rs:137`) uses **one shared `LrSchedule`** (`ReduceLROnPlateau`, `patience=500`,
`factor=0.7`, then cosine annealing) for BOTH domains, driven by TOTAL loss. Since total loss is
dominated by `physical_potential` (flat from step 0), the schedule detects a "plateau" almost
immediately and begins decaying LR - by design, cascading toward `MIN_LR`/cosine-annealed-floor
within a few thousand steps of the plateau starting. This collapses the LEARNING RATE available
to BOTH domains, including the annulus domain, right around when Kt peaks - not because its own
gradient is diluted (`.backward()` splits gradients back to each domain's own `ParamId`s, a
separately-tested invariant unaffected by another term's magnitude), but because the OPTIMIZER
STEP SIZE available to make further use of that gradient collapses, governed by a scheduler
reading the same blunt, `physical_potential`-dominated signal this investigation already flagged
as misleading for MONITORING purposes (PH4-24/25's own ledger) - here it turns out to also
corrupt the OPTIMIZATION dynamics themselves, a materially different and more serious mechanism
than "just a bad dashboard metric."

**Prep work landed, not yet wired into production training**: `pinn_solver::controllers::
DualMetricStopAdvisor` (`controllers.rs`) - a small, purpose-built plateau detector for STOP
decisions only (loss AND Kt tracked independently via a new sibling `PlateauMonitor`, NOT a
reuse of `ConvergenceTracker::check_plateau`, whose one-shot restart-budget semantics don't fit
a repeatable stop query). Explicitly does NOT touch the training objective/gradient - 4 new
focused tests, including one that directly reproduces this investigation's own real finding
(loss-plateaued-but-Kt-still-improving must not signal stop). A user-proposed companion idea -
per-term "initial-value loss normalization" (`term / term(0)`) to rebalance
`annulus_potential`/`physical_potential` in the shared loss sum - was evaluated and explicitly
REJECTED: it assigns each term a different constant (here, a ~62x relative reweighting at step
0), which is the same "not a uniform rescaling of Pi, changes the stationary point" defect this
file's own PH4-03 fixed once already and this investigation's own sub-domain-energy-mask
rejection (see `#77`'s planning history) fixed a second time in spatial form - `annulus_potential`
and `physical_potential` are two additive pieces of ONE energy functional (`Pi = U_annulus +
U_outer - W_ext`), not independent multi-task losses free to be reweighted per-term.

**Next concrete lever (not yet implemented)**: give the annulus domain its own `LrSchedule`,
decoupled from the outer domain's early-plateauing total loss - a genuinely new mechanism this
step_physics_multi/two-domain path doesn't have today (one shared schedule for N domains).
`DualMetricStopAdvisor` remains useful independently as an early-stop trigger near the real Kt
peak (would have signaled a stop around step 3000-4500 in this run), but is a mitigation for the
symptom (train past the peak), not the root cause (why the peak is followed by decline at all).

Full workspace regression after this session's additions: 491 passed, 0 failed, 38 ignored -
zero regressions.

Does not close issue #77 (or #74/#76).

## PH4-27 — Step 4 implemented: per-domain LrSchedule fixes the decline, does NOT close the
## Kt gap - real, honest, mixed result

Implemented PH4-26's own next-lever recommendation: `MultiStepCtx` gained `per_domain_lr:
Option<Vec<f64>>` (one entry per domain, `None` = every existing caller's exact pre-Step-4
behavior — the single shared `lr_sched.step(total_scalar)` result applied to every domain,
byte-for-byte unchanged). `step_physics_multi`'s per-domain optimizer loop now reads
`ctx.per_domain_lr.get(i)` before falling back to the shared `lr`. `run_annular_decomposition_
training_inner` now runs two independent `LrSchedule` instances (one per domain), each fed its
own domain's weighted-loss aggregate via a new `training_core::domain_weighted_loss` helper
(sums `raw*lambda` over every active term whose `domains()` includes that domain — a
cross-domain term counts toward both, which is correct for a monitoring/scheduling signal, not
part of the optimization objective itself) computed from the PREVIOUS step's `StepOutput` (a
one-step lag, the natural convention for any LR schedule reacting to "the last observed
reading"). The single shared schedule (`lr_sched_shared`) is kept only so `step_physics_multi`'s
own unconditional `lr_sched.step(...)` bookkeeping still runs — its result no longer drives
either domain's optimizer once the override is set.

7 new focused tests (2 in `controllers.rs` already landed with PH4-26's prep work, 3 new in
`training_core.rs`: a real two-domain fixture with identical gradients and different LR
overrides proving domain-specific step sizes actually differ, a byte-identical-fallback
regression proof for `per_domain_lr: None`, and a `domain_weighted_loss` correctness proof
including the "term touches only one domain" zero-aggregate case). Full suite: 494 passed, 0
failed, 38 ignored.

**Real result** (`issue_77_l5_extended_convergence_trend_trace`, same 12000-step config,
identical seed/checkpoints as PH4-26's run, only the per-domain-LR fix changed):

| step | pre-fix Kt (PH4-26) | post-fix Kt | delta |
|---|---|---|---|
| 0 | 0.255174 | 0.255174 | 0 |
| 1500 | 1.166180 | 1.168147 | +0.002 |
| 3000 | **1.287459 (peak)** | 1.064374 | **-0.223** |
| 6000 | 1.169102 | 1.009547 | -0.160 |
| 9000 | 1.010714 | 1.012618 | +0.002 |
| 11999 | 0.976746 (declined below step-1500 value) | **1.020403** | **+0.044** |

**The specific pathology PH4-26 identified is fixed**: the post-fix trajectory stabilizes
around 1.00-1.02 from step 6000 onward (1.010 -> 1.013 -> 1.020, essentially flat/slightly
rising) instead of continuing to decline past its peak (1.169 -> 1.011 -> 0.977 pre-fix). This
confirms the LR-starvation mechanism was real: decoupling the annulus domain's learning rate
from the outer domain's early-plateauing loss removes the degradation, exactly as PH4-26
predicted.

**It does NOT close the Kt gap.** The real, transient peak the pre-fix run reached at step 3000
(1.287) is also gone post-fix (1.064 at the same step) — the new schedule changes the early
dynamics too, trading the old run's higher-but-decaying peak for a lower, stable plateau. Kt
still settles near 1.0-1.02, far from the FEM reference (2.460638516). **Training dynamics (the
LR-schedule bug) were a real, now-fixed defect, but not the (or not the only) root cause of why
Kt doesn't approach the FEM reference at all.** Reported honestly, per this project's own
no-benchmark-hacking discipline — a real bug fix with a real, mixed, non-breakthrough result.

**Open question for the next step**: with representation (#77's first three attempts),
collocation margin (PH4-25/Step 2), sampling variance (PH4-26/Step 3's zero-cost SNR check),
and this training-dynamics bug (PH4-27/Step 4) all now addressed or ruled out, what's left to
explain Kt settling near the affine floor regardless of which of these axes is changed? Two
untested candidates going into the next step: (a) the ANNULUS domain's own new `LrSchedule`
might itself still be decaying prematurely on its own small-magnitude loss signal (self-inflicted
rather than outer-domain-inflicted this time) — not yet checked, since the diagnostic ledger's
`learning_rate` field currently still reports the now-decorative SHARED schedule's value, not
either real per-domain one; (b) the interface-continuity terms (weight 100, enforcing
displacement/traction continuity at `r=3a`) may be over-constraining the annulus field toward
smoothness/compatibility with the (affine-dominated) outer field, independent of any LR
dynamics. (a) is cheap to check (a diagnostic-visibility fix, no new training run needed to
implement, though confirming it still needs one); (b) would need a real weight-sensitivity run.

Does not close issue #77 (or #74/#76).

## PH4-28 — Candidate (a) ruled out with real evidence: the annulus domain's own LR never
## decays; LR is no longer the bottleneck at all

Added `annulus_lr`/`outer_lr` fields to `AnnularL5Diagnostic` (the pre-existing `learning_rate`
field now only reports the decorative shared schedule's result, kept for backward JSON
compatibility) and threaded the real per-domain values from `run_annular_decomposition_
training_inner` through to the diagnostic writer - a visibility-only change, zero training
logic touched.

**Real result** (`issue_77_l5_annular_diagnostic_trace`, 3000 steps, release, Step 4's
per-domain-LR fix active):

| step | Kt | annulus_lr | outer_lr |
|---|---|---|---|
| 0 | 0.255 | 1.00e-5 | 1.00e-5 |
| 300 | 0.348 | 9.90e-4 | 9.90e-4 |
| 1500 | 1.220 | **9.90e-4** | 4.85e-4 |
| 2999 | 1.231 | **9.90e-4** | 1.66e-4 |

The annulus domain's own `LrSchedule` reaches its post-warmup peak (9.90e-4) by step 300 and
**never decays for the rest of the run** - its own loss aggregate (`annulus_potential` +
`hole_free` + the interface terms) never plateaus long enough to trigger `ReduceLROnPlateau`.
The outer domain's LR correctly decays (9.90e-4 -> 4.85e-4 -> 1.66e-4) as its own,
genuinely-plateaued loss triggers its own schedule - exactly the intended, per-domain-decoupled
behavior Step 4 was built to produce.

**Candidate (a) is ruled out.** The annulus domain has full, undecayed learning rate for the
entire run and Kt still plateaus around 1.2-1.3, not approaching the FEM reference (2.4606).
Training dynamics/LR are conclusively NOT the remaining bottleneck - four different axes
(representation, collocation margin, sampling variance, and now training dynamics/LR) have
each been raised, tested with real evidence, and ruled out or fixed without closing the Kt gap.

**Next: candidate (b)**, the interface-continuity terms (`interface_displacement_continuity`/
`interface_traction_continuity`, base weight 100.0 each) may be over-constraining the annulus
field toward compatibility with the (affine-dominated) outer field's smoothness at `r=3a`,
independent of any LR dynamics - the one remaining candidate from PH4-27's own list, and the
only one not yet given a real, evidenced test.

Does not close issue #77 (or #74/#76).

## PH4-29 — Candidate (b) also ruled out: interface-continuity weight is not the bottleneck
## either. Every tested axis has now been addressed or ruled out with real evidence

Added `AnnularDecompositionProblem::interface_weight` (default `100.0` via `new()`, byte-
identical to every existing caller; `new_with_interface_weight(spec, weight)` for a real
controlled comparison) and a matching experimental entry point,
`run_annular_decomposition_training_with_diagnostics_and_interface_weight` - zero blast radius
on the 8 existing production/test callers of the two pre-existing public runners. 1 new cheap
correctness test (override changes only the two interface terms' `base_weight`, nothing else);
full suite 495 passed, 0 failed, 39 ignored.

**Real result** (`issue_77_interface_weight_reduced_l5_trace`, identical config/seed/
checkpoints to the PH4-28 baseline run, `interface_weight=10.0` vs the default `100.0`):

| step | baseline Kt (weight=100) | reduced Kt (weight=10) |
|---|---|---|
| 0 | 0.255 | 0.255 |
| 300 | 0.348 | 0.353 |
| 1500 | 1.220 | 1.168 |
| 2999 | 1.231 | 1.251 |

A 10x reduction in the interface-continuity weight produces changes well within run-to-run
noise (step 1500 is actually slightly LOWER with the reduced weight; the final Kt is only
+0.02 higher) - no consistent, meaningful movement in either direction. **Candidate (b) is
ruled out.**

**Every axis raised in this investigation has now been tested with real evidence and either
fixed (kinematic decomposition + corrected hole term, Step 1; per-domain LR, Step 4) or ruled
out (representation - #77's original three attempts; collocation margin - Step 2; sampling
variance - Step 3's zero-cost SNR check; annulus-own LR decay - PH4-28; interface-continuity
weight - this entry).** Kt still settles near 1.0-1.3 versus the FEM reference of 2.4606,
under a formulation, sampling scheme, and training schedule that are all now individually
confirmed to be behaving as intended.

**Working hypothesis for the next step** (not yet tested): this may be a more fundamental
limitation of PURE VARIATIONAL/DEM training for a sharp, localized stress concentration at
this hole-to-plate size ratio, rather than a bug in any one mechanism. This codebase's own
existing `EquilibriumTerm` doc comment (for the single-domain `UserDefinedProblem` path)
already documents a version of this: pure energy-integral minimization can under-resolve
sharp local features, and that problem's `Strong`/`Hybrid` formulations exist specifically to
add an explicit PDE-residual (equilibrium) term as an independent, LOCAL source of gradient
pressure near the feature - unlike the domain-integrated energy term, whose gradient at any
one point is diluted by the whole domain's integral. `AnnularDecompositionProblem` has NO
such option today - it offers only the variational energy term on each domain, no strong-form/
equilibrium residual anywhere. Adding an equivalent capability (a Hessian-derived-stress
equilibrium residual on the annulus domain specifically, mirroring `EquilibriumTerm`'s own
existing, working implementation) is a real, scoped feature addition - not a config tweak -
and the next candidate to test.

Does not close issue #77 (or #74/#76).

## PH4-30 — Strong-form residual on the annulus domain: real result is WORSE, not better.
## Every readily-identifiable axis has now been tested; the investigation needs a strategy
## decision, not another mechanism guess

Implemented PH4-29's working hypothesis: `AnnularDecompositionProblem::include_annulus_
equilibrium` (default `false`, byte-identical to every existing caller) reuses the SAME
`EquilibriumTerm` the single-domain `UserDefinedProblem`'s Strong/Hybrid formulations already
use (no new struct - it was already generic over `domain`/`point_set`), scoped to
`ANNULUS_DOMAIN`'s own `"interior"` points, with `ref_div2` reusing the same
`stress_per_length2` normalization. `new_with_annulus_equilibrium`/
`run_annular_decomposition_training_with_diagnostics_and_annulus_equilibrium` mirror PH4-29's
own opt-in pattern exactly. 2 new tests (registration/gating correctness, and confirming
`needs_hessian()==true` so the Hessian forward pass genuinely runs). Full suite: 496 passed, 0
failed, 40 ignored.

**Real result** (`issue_77_annulus_equilibrium_l5_trace`, identical config/seed/checkpoints to
every prior controlled comparison in this investigation):

| step | baseline Kt (no equilibrium) | with annulus equilibrium |
|---|---|---|
| 0 | 0.255 | 0.255 |
| 300 | 0.348 | **0.273** |
| 1500 | 1.220 | **0.982** |
| 2999 | 1.231 | **0.986** |

Adding the strong-form residual made Kt WORSE at every checkpoint after step 0, not better -
final Kt is lower (0.986 vs 1.231) and never even reaches the pure-variational baseline's
step-1500 value. The working hypothesis (a domain-integrated energy term's gradient is diluted
by the whole domain, while a local residual supplies pressure directly at the concentration) is
**not supported** by this evidence - if anything the added term competed with, rather than
reinforced, the existing terms' progress within the same step budget (step-0 total loss jumped
from 28.7 to 421.9 with the new term's initial residual, and the run may simply need more steps
to digest that - not yet distinguished from a genuine interference effect, but reported as a
real negative result either way, not assumed favorable).

**Status after six tested axes**: representation (#77's original three attempts - ruled out),
collocation margin (Step 2 - ruled out, flat), sampling variance (Step 3's zero-cost SNR check
- ruled out), training-dynamics/LR (Step 4 - a real bug FIXED, PH4-28 - the annulus's own
schedule also ruled out as a factor), interface-continuity weight (PH4-29 - ruled out, 10x
change had no effect), and now a strong-form residual (PH4-30 - tested, real result is worse).
Every readily-identifiable mechanism within the current annular-decomposition architecture has
now been given a real, evidenced test. Kt remains in the 1.0-1.3 range versus the FEM reference
of 2.4606 (a mesh-converged, independently-validated target - `#76`'s own FEM tool - not in
question). This investigation has reached the point where continuing to guess individual
mechanisms has a low prior of success without a genuinely different architectural idea or a
decision to accept the current state and redirect effort - a strategy decision for the project
owner, not something to keep probing autonomously.

Does not close issue #77 (or #74/#76).

## PH4-31 — Spectral-bias hypothesis tested via multi-scale Fourier features: real result is
## flat, not a breakthrough. A user-proposed conformal-mapping rewrite was evaluated and
## rejected before this was implemented (its diagnosis was contradicted by PH4-29's own evidence)

A conformal-mapping / single-domain boundary-fitted-coordinate rewrite was proposed, diagnosing
"annular decomposition's artificial interfaces dilute the gradient near the hole." Evaluated
and rejected before any implementation: (1) its physical premise ("stress spikes exponentially
near the hole") is factually wrong - the Kirsch hoop stress decays smoothly and algebraically
(3.0 -> 2.53 -> 1.52 -> 1.07 from r/a=1 to r/a=3, verified this session); (2) its root cause is
directly contradicted by PH4-29's own real A/B test (cutting the interface weight 10x, the
literal test of "are interfaces damping the signal," produced no effect); (3) its one reusable
idea (richer hole-local coordinate features) is a more elaborate version of something this
codebase already tried (`SingleHoleChart` rational features, #77's original attempts) with a
documented negative result. The pattern actually in the data - Kt resisting every push on
sampling, weighting, schedule, and even an added local residual term, while sitting just above
the pure-affine floor - matches **spectral bias**, a well-established property of MLPs
(Rahaman et al. 2019; Tancik et al. 2020 for coordinate networks; Wang/Teng/Perdikaris for
PINNs): fast at learning smooth structure, slow/unable to represent sharp local features even
with correct, healthy gradient signal. This was tested directly via a **multi-scale Fourier
feature embedding** - additive, low-risk, reusing this codebase's own dormant `fourier_embed`
machinery - rather than the proposed rewrite, which would have discarded five already-validated
pieces (kinematic decomposition, the corrected hole term, annular decomposition, per-domain LR,
the FEM reference itself) to chase an evidence-contradicted diagnosis.

**Implementation**: `CoordinateEmbedding::SingleHoleChart` gained `n_fourier: usize` (default
`0`, byte-identical to every existing caller - verified by a dedicated test before anything
else). `network::chart_embed` appends `4*n_fourier` columns - `sin`/`cos` of the HOLE-RELATIVE
`(qx,qy)` (not raw x,y - encoding frequency content at the hole's own scale, standard guidance
in the Fourier-features literature) at dyadic frequencies `2^l*pi`, same convention
`fourier_embed` already used elsewhere in this file. Wired opt-in to the annulus domain only
(`AnnularDecompositionProblem`/`run_annular_decomposition_training_with_diagnostics_and_
annulus_fourier`), mirroring the exact `interface_weight`/`include_annulus_equilibrium`
precedent - zero blast radius on every existing caller. 8 new tests (dimension formula,
known-value, FD-continuity, byte-identical-at-zero, end-to-end wiring smoke test).

**A real bug surfaced and was fixed during this step, not just the hypothesis test**: the
first real run panicked with `IncompatibleShapes { left: [720, 10], right: [26, 64] }` -
`probe_hole_boundary_profile_derived`/`probe_hole_stress_profile_direct_at_radius` (used only
by `annular_l5_diagnostic`'s checkpoint-triggered diagnostic path, never the main training
step) hardcoded `geometry.coordinate_embedding()` (always the plain, `n_fourier=0` embedding)
instead of deriving the embedding actually matching the model's own saved width. Fixed via a
new `embedding_for_model` helper that inverts `model.input_dim()` back to the correct
`CoordinateEmbedding` - mirroring `training_core::compute_domain_forwards`'s own "the model's
saved architecture is authoritative" dispatch pattern exactly, not a new invention. The smoke
test that should have caught this originally passed anyway because it ran zero diagnostic
checkpoints (`&[]`) - fixed to use `&[0]` specifically so this bug class can't hide behind an
undertested code path again.

**Real result** (`issue_77_annulus_fourier_l5_trace`, `n_fourier=4`, identical config/seed/
checkpoints to every prior PH4-28/29/30 comparison):

| step | baseline Kt (no Fourier) | with n_fourier=4 |
|---|---|---|
| 0 | 0.255 | 1.276* |
| 300 | 0.348 | 1.063 |
| 1500 | 1.220 | 1.311 |
| 2999 | 1.231 | **1.238** |

*Step 0 is not a fair comparison point - Fourier features add real high-frequency content to
an UNTRAINED network's random-weight output, so the elevated step-0 Kt reflects init noise, not
a physically meaningful reading (the same reason the step-0 loss was also higher, 79.8 vs
28.7). The meaningful comparison is the trained endpoint.

**Final Kt (1.238) is statistically indistinguishable from baseline (1.231)** - a difference
of 0.007, well within the noise band every other controlled comparison in this investigation
has shown (PH4-29's interface-weight test moved by a similar 0.02 with no real effect). The
spectral-bias hypothesis, at least as implemented here (`n_fourier=4`, hole-relative dyadic
Fourier features), is **not supported** by this result either.

**Not yet conclusive on spectral bias generally** - `n_fourier=4` (max frequency `8*pi`, per
the dyadic `2^l*pi` schedule) is a modest choice; a genuinely under-powered frequency range
would look exactly like this flat result. Before fully ruling out spectral bias as the
mechanism, a real sweep (e.g. `n_fourier=8` or higher) is the next honest step, not yet run.

Full workspace regression: 502 passed, 0 failed, 41 ignored - zero regressions.

Does not close issue #77 (or #74/#76).

## PH4-32 — Fourier sweep at n_fourier=8: real result is WORSE, not flat. Spectral-bias-via-
## input-features hypothesis is now rejected, not just unsupported.

PH4-31 left one honest gap open: `n_fourier=4` was flat, but a modest frequency count could
mean under-powered features rather than a wrong hypothesis. Ran the same controlled comparison
(`issue_77_annulus_fourier_n8_l5_trace`, `n_fourier=8`, max dyadic frequency `128*pi` vs `8*pi`
at n=4, identical geometry/material/load/network/seed/checkpoints to every prior PH4-28..31
comparison) to settle it.

**Real result**:

| step | baseline Kt (no Fourier) | n_fourier=4 | n_fourier=8 |
|---|---|---|---|
| 0 | 0.255 | 1.276* | 4.781* |
| 300 | 0.348 | 1.063 | 0.720 |
| 1500 | 1.220 | 1.311 | 0.679 |
| 2999 | 1.231 | 1.238 | **0.587** |

*Step 0 is init noise, not physically meaningful (PH4-31's own caveat, more pronounced here -
more frequencies means more high-frequency content in an untrained network's random output).

Step-0 loss confirms the same pattern from the other side: `2207` at n=8 vs `79.8` at n=4 vs
`28.7` baseline - adding frequency content made the initial residual landscape dramatically
worse, not just noisier. Training loss at n=8 is also visibly non-monotonic step-to-step
(1.51 -> 1.60 -> 1.09 -> 1.07 -> 1.15 -> 1.03, `l5_fourier_n8_run.log`) where n=4 and baseline
both settle smoothly - consistent with a harder, not easier, optimization landscape.

**This is a real trend, not noise**: Kt at the trained endpoint moves monotonically WORSE as
`n_fourier` increases (1.231 -> 1.238 -> 0.587 for 0/4/8). If spectral bias via under-powered
input frequencies were the real bottleneck, more frequency content should move Kt toward the
FEM target (2.4606), not away from the pure-affine floor (1.0) and below it. Instead, doubling
the frequency range roughly halved the final Kt. The most likely mechanism: the added
high-frequency basis functions expand the annulus network's effective input dimensionality
(10 -> 26 -> 42) and raster a much rougher loss surface, which a fixed-budget 3000-step AdamW
run does not have time to optimize through - an optimization-difficulty cost, not a
representation win.

**Conclusion**: the multi-scale Fourier feature approach (as implemented - hole-relative dyadic
frequencies, opt-in to the annulus domain only) is now REJECTED as a fix for issue #77's Kt
gap, based on two real controlled data points showing a monotonic worsening trend, not just an
absence of improvement. This closes out the spectral-bias-via-input-encoding hypothesis
specifically; it does not rule out spectral bias as a phenomenon in this network (a
SIREN-style sinusoidal-activation rewrite is a structurally different, network-wide change from
adding input features, and remains an untested, higher-risk Tier-2 idea per PH4-31's own
scope note - not run here).

Full workspace regression: unaffected (no production code changed in this step - same
production code as PH4-31, only a new `#[ignore]`d comparison test with a different constant).

Does not close issue #77 (or #74/#76).

## PH4-33 — User-requested isolation test: quasi-infinite-plate geometry rules out finite-
## domain/finite-width effects as the dominant remaining cause of the Kt gap

User-proposed test: does L5's Kt gap (network ~1.0-1.3 vs FEM 2.4606) partly reflect a
finite-plate/finite-width sampling or boundary-proximity artifact, rather than a purely
training/representation problem? Requested geometry, converted to SI: hole radius 0.125in
(0.003175m), plate 10in x 10in (half_w=half_h=0.127m), thickness 0.125in (0.003175m) - a
hole/half-width ratio of 0.025, half of L5's own 0.05, pushing toward the idealized-infinite-
plate regime the Kirsch Kt=3.0 solution assumes. Zero production code changes needed - a new
`quasi_infinite_geometry()`/`ProblemSpec` reusing every existing mechanism (kinematic
decomposition, corrected hole term, annular decomposition, per-domain LR - `3*radius=0.009525
< half_w=0.127`, comfortably satisfying `annular_partition()`), verified by a cheap gate test
(`quasi_infinite_geometry_still_supports_annular_decomposition`) before any real run.

**A correct comparison needed a FRESH FEM reference for this geometry, not L5's 2.4606** - a
smaller hole/plate ratio has its own, different, mesh-converged finite-plate Kt; comparing
against the wrong number would misattribute a real geometry difference as training error.
Computed via the same independent `tools/finite_plate_reference.py` CST tool issue #76 already
validated (`--half-width 0.127 --half-height 0.127 --radius 0.003175 --young 71.7e9 --poisson
0.33 --traction 6.9e7 --fd-h 1e-3`), converged to two consecutive under-2%-change mesh
refinements at `512x128 -> 1024x256 -> 2048x512` (max relative change 0.0019 on the final
refinement): **`probe_fd_kt_vm = 2.0738642734198516`**.

**Confirmed before trusting this comparison**: the FEM tool's probe-ring placement
(`probe_radius = radius + 4*fd_h*max(half_w,half_h)`, `tools/finite_plate_reference.py`) and the
PINN diagnostic's own ring placement (`ring_anchor_margin_m`, `user_problem.rs`, used by
`annular_l5_diagnostic` for every `kt_derived_fd_vm` reading in this whole investigation) are
the SAME plate-scaled formula with the same `RING_ANCHOR_SAFETY_FACTOR=4.0` - re-deriving L5's
own `FEM_KT=2.460638516` from the tool at L5's default geometry reproduces it exactly
(`probe_fd_kt_vm=2.46063851616289`), confirming this is the correct field to read for a
like-for-like comparison, not a different convention that would need reconciling.

**Real result** (`issue_77_quasi_infinite_plate_l5_trace`, identical network/training
hyperparameters and checkpoints to every PH4-28..32 comparison - only geometry changed):

| step | Kt |
|---|---|
| 0 | 0.462* |
| 300 | 0.357 |
| 1500 | 1.160 |
| 2999 | **1.116** |

*step 0 is untrained-network init noise, not physically meaningful (same caveat as every prior
comparison in this investigation).

Relative error against the CORRECT geometry-specific FEM target: `(2.0739-1.116)/2.0739 =
46.18%`, versus L5's own baseline relative error `(2.4606-1.231)/2.4606 = 49.97%`. A ~3.8
percentage point reduction - real, but small, and well within the noise band this investigation
has already characterized on every other axis (PH4-29's interface-weight A/B moved by a similar
margin with an explicitly "no effect" verdict).

**Interesting secondary finding, worth noting but not the main conclusion**: the quasi-infinite
geometry's own correct FEM target (2.074) is LOWER than L5's own FEM target (2.461), despite
having a smaller (more idealized) hole/plate ratio that would naively suggest a Kt closer to
3.0. This is explained by the plate-scaled probe-ring formula above: a bigger plate at the same
`fd_h` pushes the probe ring proportionally further (in units of hole radii) from the hole
boundary for a smaller hole, and Kirsch's hoop stress decays with distance from the hole - so
part of what "quasi-infinite" bought in finite-width relief, this fixed relative-margin
convention partially gave back by reading the stress field further from the true concentration.
This is a real, disclosed property of the comparison methodology, not a bug - both the FEM tool
and the PINN diagnostic apply it identically (confirmed above), so it doesn't invalidate the
comparison, but it does mean "closer to the ideal infinite-plate regime" and "closer to the
naive Kt=3.0 number" are not the same statement once this probe convention is held fixed.

**Conclusion: finite-domain/finite-width effects are NOT the dominant remaining cause of the Kt
gap.** Halving the hole/plate ratio (a real, physically meaningful step toward the idealized
regime the proposed diagnosis targeted) produced only a ~4-point relative-error improvement,
not the large closure a genuine finite-width-artifact explanation would predict. Combined with
every other ruled-out axis in this investigation (collocation margin, sampling variance,
LR/training dynamics, interface-continuity weight, strong-form residual, spectral-bias/Fourier
features at two frequency counts, and now finite-domain scale), the remaining ~46-50% gap keeps
resisting every mechanism this session has tested with real evidence - narrowing the search
further rather than being explained away. Per the standing instruction, this does not close
issue #77 (or #74/#76).

Full workspace regression: 0 changed production files in this step (new geometry + two new
tests only) - full suite run alongside this test, 0 failures.

Does not close issue #77 (or #74/#76).

## PH4-34a — SIREN sinusoidal activation tested: real result WORSE than baseline, same pattern
## as the rejected Fourier-feature candidate

User-directed test, following PH4-31/32's rejection of input-encoding Fourier features as a
spectral-bias fix: does the activation FUNCTION itself (not just the input encoding) explain
the network's apparent difficulty representing sharp near-hole structure? Implemented SIREN
(Sitzmann et al. 2020) - `sin(omega_0*z)` replacing `tanh` in every `ElasticityNet` layer
(including `layers[0]`; `out` stays linear either way), with SIREN's own specific weight-init
scheme (`U(-1/fan_in, 1/fan_in)` for the first layer, `U(-sqrt(6/fan_in)/omega_0,
sqrt(6/fan_in)/omega_0)` for hidden layers) - a genuinely different, network-wide
representational change from an input-feature addition, opt-in via `ElasticityNetConfig::
use_siren`/`siren_omega_0`, byte-identical default (`false`/tanh), wired to the annulus domain
only via `run_annular_decomposition_training_with_diagnostics_and_siren` (same scope as every
prior candidate). 5 new gate tests (byte-identical-at-false, finite-output-and-differs-from-
tanh, init-bound verification against Sitzmann's own formula) all pass before the real run.

**Real result** (`issue_77_annulus_siren_l5_trace`, identical network/training hyperparameters
and checkpoints to every PH4-28..33 comparison - only the activation function changed):

| step | baseline Kt (tanh) | SIREN Kt |
|---|---|---|
| 0 | 0.255 | 243.14* |
| 300 | 0.348 | 2.315* |
| 1500 | 1.220 | 1.092 |
| 2999 | 1.231 | **1.058** |

*Step 0/300 are extreme init noise, more pronounced than every prior candidate's init-noise
caveat - SIREN's `sin(omega_0*z)` with `omega_0=30` on an untrained network produces a MUCH
higher-variance random output than tanh's saturating range, visible directly in the real
step-0 loss (`2.94e5`, several orders of magnitude above baseline's `28.7`, n_fourier=4's
`79.8`, or even n_fourier=8's `2207`). Training loss is also visibly non-monotonic throughout
the whole run (step 900: 1.406 -> step 1200: 1.597 -> step 1500: 1.672 -> step 1800: 1.476 ->
... -> step 2700: 1.227, `l5_siren_run.log`) where every tanh-based comparison in this
investigation settles smoothly - a harder optimization landscape, not a cleaner one.

**Conclusion: SIREN is REJECTED as a fix for issue #77's Kt gap** - the trained-endpoint Kt
(1.058) is worse than baseline (1.231), the same direction and rough magnitude as the rejected
Fourier-feature candidates (PH4-31/32: 1.238 then 0.587). Combined with those two results, the
evidence now points at BOTH tested spectral-bias remedies (input-encoding richness and
activation-function richness) making optimization harder without buying back any of that cost
in representational accuracy, on this specific loss landscape and training budget. This does
not prove spectral bias is not a real property of this network (a genuinely well-tuned SIREN
configuration - different `omega_0`, a proper per-layer init audit, more training steps to let
the harder landscape settle - might behave differently), but two independently-implemented
candidates both moving the wrong direction is real evidence against "add more expressive
high-frequency capacity" as a low-effort fix, not merely an absence of evidence for it.

Full workspace regression: run alongside this test; see PH4-34b for the combined suite result.

Does not close issue #77 (or #74/#76).

## PH4-34b — Gradient-share hypothesis (derived from this investigation's own diagnostic data,
## not a new architecture): does `hole_free`'s late-training gradient dominance suppress Kt?

While SIREN trained, re-examined the `AnnularTermDiagnostic.gradient_share` field already
collected by EVERY prior real comparison run in this investigation (baseline, both Fourier
sweeps, interface-weight, equilibrium, quasi-infinite) - no new run needed to form this
hypothesis, only inspection of JSON already on disk. A consistent, previously-unremarked
pattern emerged across five of six runs (the annulus-equilibrium run, PH4-30, is the outlier -
see below):

| run | step 1500 hole_free raw / grad_share | step 2999 hole_free raw / grad_share | step 2999 physical_potential grad_share |
|---|---|---|---|
| baseline | 2.18e-4 / 0.117 | 4.76e-4 / **0.784** | 0.148 |
| interface-weight=10 | 1.35e-4 / 0.433 | 1.67e-4 / **0.780** | (not separately re-checked) |
| quasi-infinite | 5.07e-4 / 0.360 | 2.56e-4 / **0.769** | (not separately re-checked) |
| n_fourier=4 | 2.14e-4 / 0.361 | 1.13e-4 / 0.359 | (mixed - see note below) |
| n_fourier=8 | 2.45e-3 / 0.896 | 6.21e-4 / **0.900** | (dominated throughout) |
| annulus-equilibrium (PH4-30, the outlier) | 6.25e-4 / 0.007 | 1.62e-4 / 0.005 | (equilibrium term dominates instead) |

`hole_free`'s RAW loss is already tiny (~1e-4, the traction-free boundary condition is
effectively satisfied) by step 1500 in every run - but in 4 of 6 runs its GRADIENT SHARE
climbs to 70-90% of the entire optimization's gradient budget by the final checkpoint, while
`physical_potential` (raw ~1.0, nowhere near converged - this IS the actual energy functional
that shapes the interior stress field) gets as little as ~15% at the same checkpoint in the
baseline run. The optimizer spends most of its late-training gradient budget re-polishing an
already-satisfied boundary condition instead of the term that would resolve the interior
field's true stress concentration. This is evidence-first, not a new architecture guess - a
different class of candidate from every prior mechanism tested (representation, sampling, LR,
interface weight, strong-form residual, spectral bias, finite-domain scale), and the first one
grounded directly in gradient-accounting data already collected rather than a new training run.

**Implementation**: `AnnularDecompositionProblem` gained a `hole_free_weight: f32` field
(default `LAM_HOLE_FREE`=100.0 via `new()`, byte-identical to every pre-PH4-34 caller) and
`new_with_hole_free_weight`, mirroring the exact `interface_weight` precedent (PH4-28/29) -
`base_weight`'s `"hole_free"` arm now reads `self.hole_free_weight` instead of the hardcoded
constant. Threaded through `run_annular_decomposition_training_inner` (a 10th parameter,
defaulted to `100.0` at every existing call site) and a new
`run_annular_decomposition_training_with_diagnostics_and_hole_free_weight` entry point. 1 new
gate test proves the override changes ONLY `hole_free`'s own base weight.

**Real result pending** - `issue_77_hole_free_weight_reduced_l5_trace` (10x reduction, `100.0
-> 10.0`) launched as a real, controlled A/B against the same baseline config/checkpoints;
result to be appended here once the run completes.

Does not close issue #77 (or #74/#76).
