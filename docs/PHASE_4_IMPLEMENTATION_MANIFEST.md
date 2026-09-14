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

Status: VERIFIED-DISABLED (issue #63 sub-issue #67) — AMR stays off for the canonical no-hole
Variational benchmark, backed by real evidence this time, not just unmet-prerequisite BLOCKED.
Issue #63's own text explicitly sanctions this outcome: "AMR may be disabled for uniform
problems if evidence shows it worsens the solve" — that evidence now exists, twice over.

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

**Deliberately not fixed in this pass**: the likely correct fix (route AMR's residual probe
through `model.valid()`/the non-autodiff `BInner` backend, since it never needs gradients at
all) touches `compute_domain_forwards`/`probe_interior_energy_residuals` — shared infrastructure
also used by Kirsch/pin-lug's own AMR paths — and deserves its own careful, independently-
verified change with its own regression proof, not a rushed fix bolted onto this item. Tracked
as its own follow-up issue (see the parent epic #63 for the link) with this exact reproduction
recorded so it doesn't need re-discovering.

**Conclusion**: AMR remains disabled (`amr_enabled = false`) for the shipped canonical
no-hole Variational benchmark. This is not a workaround pending future work — it's the correct,
policy-compliant, now doubly-evidenced state (quality regression AND a crash risk) per issue
#63's own explicit allowance. Re-enabling AMR for this formulation requires the autodiff crash
fix above landing and its own fresh A/B evidence, not just re-flipping the flag.

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

Runtime proof status by formulation: `Variational` has real production runtime proof this
session (issue #64/#66/#68/#69's real headless/GUI runs all exercise this exact code path
successfully). `Strong`/`Hybrid` `FormulationSelection` variants are wired
(`user_problem.rs` term-selection match) but exercised only in unit tests this session, not a
real training run — `Weak` has no user-defined-problem implementation at all (matches
`FORMULATION_SUPPORT_MATRIX.md`'s own note). Remains IMPLEMENTED, not VERIFIED, pending real
runtime proof for `Strong`/`Hybrid` specifically — do not promote without a real run of those
variants; the Variational-only evidence above does not generalize.

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

**Real dependency now identified**: robust small-hole Kt convergence is blocked on **issue #74**
(the AMR+Variational autodiff crash, PH4-09) — AMR is the mechanism that would fix this sampling
gap, but it's currently disabled for Variational precisely because it crashes past ~1200-2200
steps. L5 cannot be pushed to a real pass without either (a) #74 landing so AMR can safely bias
near-hole density, or (b) a much larger uniform `n_interior` (untested, expensive, and
`4096→~10-100x` would likely be needed given the 0.55% near-boundary fraction — not attempted
this pass per the verification-cost policy without stronger justification first).

**Conclusion, honestly stated**: PH4-14/L5 is NOT VERIFIED and is not being forced to pass.
Real, non-hacked evidence now exists (this item's actual job) showing the mechanism blocking it.
Follow-up: either extend issue #74's scope or file a new tracked issue for "AMR-free small-hole
Kt convergence strategy" before attempting a further L5 run.

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

Status: IMPLEMENTED

`docs/FORMULATION_SUPPORT_MATRIX.md` records only `VERIFIED`, `SUPPORTED_WITH_LIMITATION`, and
`EXPLICITLY_UNSUPPORTED`; it does not promote legacy Hybrid evidence to generalized support.
Runtime evidence remains insufficient for VERIFIED capability rows.

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

Status: INVESTIGATING

Focused atomic-objective, gauge, FieldKind, differential, provenance, and topology sampling
regressions pass. Full physical matrix remains pending corrected L4 and independent field data;
no benchmark-specific correction is permitted.

## PH4-21 — Final operational status

Status: BLOCKED

Blocking conditions remain: corrected Variational L4 fails; controlled ladder has not isolated
the remaining divergence; PH4-04/07-20 lack required runtime and independent benchmark proof.
Issue #63 forbids declaring OPERATIONAL or tuning around these failures.

Validation limitation: focused NdArray solver tests and compilation pass. A hard-coded WGPU
unit-test alias prevented CPU verification on this machine; it now uses the selected backend and
the formerly blocked energy tests pass. Full-suite completion has not yet been captured. Clippy
with `-D warnings` still fails pre-existing `pinn-core` warnings. Neither is treated as Phase 4
proof.
